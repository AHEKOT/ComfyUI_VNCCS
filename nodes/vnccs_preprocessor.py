"""VNCCS Preprocessor node

This node detects a face in an image using a bbox detector, then pastes a piece of image2 into the detected face region.

- INPUTS: image1 (base image), image2 (piece to paste), bbox_detector, threshold, dilation, drop_size
- OUTPUTS: modified image1 with image2 pasted into the face bbox

This file relies on runtime objects provided by ComfyUI.
Includes local implementations of tensor_crop and tensor_paste.
"""

import types
import sys
import math
try:
    import node_helpers
except Exception:
    # minimal safe fallback for environments without ComfyUI
    class _NodeHelpersFallback:
        @staticmethod
        def conditioning_set_values(conditioning, values, append=False):
            # best-effort: if conditioning looks like a list of (tensor, dict) pairs, attach values
            try:
                new_conditioning = []
                for cond in conditioning:
                    if isinstance(cond, (list, tuple)) and len(cond) >= 2:
                        cond_tensor = cond[0]
                        cond_dict = dict(cond[1]) if isinstance(cond[1], dict) else {}
                        if append:
                            for k, v in values.items():
                                if k in cond_dict and isinstance(cond_dict[k], list):
                                    cond_dict[k].extend(v if isinstance(v, list) else [v])
                                else:
                                    cond_dict[k] = list(v) if isinstance(v, list) else [v]
                        else:
                            cond_dict.update(values)
                        new_conditioning.append((cond_tensor, cond_dict))
                    else:
                        new_conditioning.append(cond)
                return new_conditioning
            except Exception:
                return conditioning
    node_helpers = _NodeHelpersFallback()

try:
    import comfy.utils
except Exception:
    # minimal comfy.utils fallback with common_upscale passthrough
    class _ComfyUtilsFallback:
        @staticmethod
        def common_upscale(samples, width, height, upscale_method, crop):
            # passthrough for fallback
            return samples
    comfy = _ComfyUtilsFallback()

import torch
from nodes import MAX_RESOLUTION


def _tensor_crop(image, crop_region):
    """Crop tensor image using crop_region coordinates [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = crop_region
    return image[:, y1:y2, x1:x2, :]


def _tensor_paste(image1, image2, crop_region, feather=0):
    """Paste image2 onto image1 at crop_region position with optional feather blending"""
    x1, y1, x2, y2 = crop_region
    h, w = y2 - y1, x2 - x1
    
    # Ensure image2 has batch dimension
    if len(image2.shape) == 3:  # [H, W, C] format
        image2 = image2.unsqueeze(0)
    
    # Resize image2 to match crop region size if needed
    if image2.shape[1] != h or image2.shape[2] != w:
        image2 = torch.nn.functional.interpolate(image2.permute(0, 3, 1, 2), 
                                               size=(h, w), 
                                               mode='bicubic', 
                                               align_corners=False).permute(0, 2, 3, 1)
    
    # Remove batch dimension if present for single image pasting
    if image2.shape[0] == 1:
        image2 = image2.squeeze(0)
    
    if feather > 0:
        # Create feather mask
        import torch.nn.functional as F
        
        # Create a mask with the same size as the crop region
        mask = torch.ones((h, w), dtype=image2.dtype, device=image2.device)
        
        # Apply gaussian blur to create feather effect
        # Convert to 4D for conv2d
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create gaussian kernel
        kernel_size = min(feather * 2 + 1, min(h, w) // 2 * 2 + 1)  # Ensure odd kernel size
        if kernel_size > 1:
            # Simple box blur approximation for gaussian
            kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)
            kernel = kernel / kernel.numel()
            
            # Apply blur
            mask_4d = F.conv2d(mask_4d, kernel, padding=kernel_size//2)
            mask_4d = F.conv2d(mask_4d, kernel, padding=kernel_size//2)
        
        mask = mask_4d.squeeze(0).squeeze(0)  # Back to [H, W]
        
        # Ensure mask values are between 0 and 1
        mask = torch.clamp(mask, 0, 1)
        
        # Apply mask to image2
        # Expand mask to match image channels
        mask_expanded = mask.unsqueeze(-1).expand_as(image2)  # [H, W, C]
        
        # Alpha blending: image1 * (1 - mask) + image2 * mask
        image1[:, y1:y2, x1:x2, :] = image1[:, y1:y2, x1:x2, :] * (1 - mask_expanded) + image2 * mask_expanded
    else:
        # Paste with full opacity (original behavior)
        image1[:, y1:y2, x1:x2, :] = image2
    
    return image1


class VNCCS_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "bbox_detector": ("BBOX_DETECTOR", ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 300, "step": 1}),
                "invert_image2": ("BOOLEAN", {"default": False}),
                "detect_on": (["image1", "image2"], {"default": "image1"}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preprocess"

    CATEGORY = "VNCCS/preprocessing"

    def preprocess(self, image1, image2, bbox_detector, threshold=0.5, dilation=10, drop_size=10, feather=0, invert_image2=False, detect_on="image2", mask=None):
        if len(image1) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_Preprocessor does not allow image1 batches.')

        if len(image2) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_Preprocessor does not allow image2 batches.')

        # Invert image2 if requested
        if invert_image2:
            image2 = (1 - image2).clone()
        else:
            image2 = image2.clone()

        # Fixed crop_factor for bbox detection
        crop_factor = 1.0

        # Select which image to detect bboxes on
        detect_image = image1 if detect_on == "image1" else image2

        print(f'[VNCCS] DEBUG: Detecting bboxes on: {detect_on}')
        print(f'[VNCCS] DEBUG: Using bbox_detector: {type(bbox_detector).__name__}')
        print(f'[VNCCS] DEBUG: Calling bbox_detector.detect with params:')
        print(f'[VNCCS] DEBUG:   threshold={threshold}, dilation={dilation}, crop_factor={crop_factor}, drop_size={drop_size}')
        
        try:
            segs_result = bbox_detector.detect(detect_image, threshold, dilation, crop_factor, drop_size)
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to detect segments with bbox_detector: {str(e)}')
        
        # Handle different return formats from bbox detectors
        if isinstance(segs_result, tuple) and len(segs_result) == 2:
            segs = segs_result
        else:
            segs = segs_result

        print(f'[VNCCS] DEBUG: segs[0] (shape info): {segs[0]}')
        print(f'[VNCCS] DEBUG: Total segments found: {len(segs[1])}')

        # Validate segs format
        if not isinstance(segs, tuple) or len(segs) != 2:
            raise Exception(f'[VNCCS] ERROR: Invalid segs format from bbox_detector. Expected tuple of length 2, got: {type(segs)}')
        
        if not isinstance(segs[1], (list, tuple)):
            raise Exception(f'[VNCCS] ERROR: Invalid segments list. Expected list or tuple, got: {type(segs[1])}')

        if len(segs[1]) == 0:
            print(f'[VNCCS] DEBUG: No segments found on image2, using central crop')
            # Fallback: use central crop
            img_h, img_w = image2.shape[1], image2.shape[2]
            h_crop = min(img_h, 256)  # arbitrary size
            w_crop = min(img_w, 256)
            start_h = max(0, (img_h - h_crop) // 2)
            start_w = max(0, (img_w - w_crop) // 2)
            x1, y1, x2, y2 = start_w, start_h, start_w + w_crop, start_h + h_crop
            segments = [(x1, y1, x2, y2)]
        else:
            # Use all segments
            segments = []
            for seg in segs[1]:
                x1, y1, x2, y2 = seg.crop_region
                segments.append((x1, y1, x2, y2))

        print(f'[VNCCS] DEBUG: Processing {len(segments)} segments')

        # Process each segment and combine them on image1
        modified_image = image1.clone()
        
        for seg_idx, (x1, y1, x2, y2) in enumerate(segments):
            print(f'[VNCCS] DEBUG: Segment {seg_idx}: region ({x1},{y1},{x2},{y2})')

            # Scale coordinates back to image2 dimensions if segs[0] != image2.shape
            img_h, img_w = image2.shape[1], image2.shape[2]
            scale_h = img_h / segs[0][0]
            scale_w = img_w / segs[0][1]
            x1_scaled = int(x1 * scale_w)
            y1_scaled = int(y1 * scale_h)
            x2_scaled = int(x2 * scale_w)
            y2_scaled = int(y2 * scale_h)

            print(f'[VNCCS] DEBUG: Scaled region: ({x1_scaled},{y1_scaled},{x2_scaled},{y2_scaled})')

            # Apply dilation to crop_region
            x1_scaled = max(0, x1_scaled - dilation)
            y1_scaled = max(0, y1_scaled - dilation)
            x2_scaled = min(img_w, x2_scaled + dilation)
            y2_scaled = min(img_h, y2_scaled + dilation)

            # Ensure minimum size and x2 > x1, y2 > y1
            min_size = 10
            if x2_scaled <= x1_scaled:
                x2_scaled = min(img_w, x1_scaled + min_size)
            if y2_scaled <= y1_scaled:
                y2_scaled = min(img_h, y1_scaled + min_size)

            crop_region = (x1_scaled, y1_scaled, x2_scaled, y2_scaled)

            print(f'[VNCCS] DEBUG: Adjusted region after dilation: ({x1_scaled},{y1_scaled},{x2_scaled},{y2_scaled})')

            # Crop the segment from image2
            image2_seg = _tensor_crop(image2, crop_region)
            print(f'[VNCCS] DEBUG: Cropped segment {seg_idx} from image2')

            # Use the same crop_region for pasting on image1
            crop_region_paste = crop_region

            # Resize segment if needed (in case dimensions differ)
            h_paste, w_paste = y2_scaled - y1_scaled, x2_scaled - x1_scaled
            if image2_seg.shape[1] != h_paste or image2_seg.shape[2] != w_paste:
                image2_seg = torch.nn.functional.interpolate(image2_seg.permute(0, 3, 1, 2), 
                                                           size=(h_paste, w_paste), 
                                                           mode='bicubic', 
                                                           align_corners=False).permute(0, 2, 3, 1)

            # Paste segment onto modified image
            modified_image = _tensor_paste(modified_image, image2_seg, crop_region_paste, feather)
            print(f'[VNCCS] DEBUG: Pasted segment {seg_idx}')

        # Apply mask to the final image if provided
        if mask is not None:
            print(f'[VNCCS] DEBUG: Applying mask to final image')
            # Ensure mask has the same spatial dimensions as modified_image
            if mask.shape[0] != modified_image.shape[1] or mask.shape[1] != modified_image.shape[2]:
                # Resize mask to match image dimensions
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                    size=(modified_image.shape[1], modified_image.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # Back to [H, W]
            else:
                mask_resized = mask
            
            # Expand mask to match image channels [H, W] -> [H, W, C]
            mask_expanded = mask_resized.unsqueeze(-1).expand_as(modified_image)  # [H, W, C]
            
            # Apply mask: where mask=0, use white background; where mask=1, use modified_image
            white_bg = torch.ones_like(modified_image)
            modified_image = white_bg * (1 - mask_expanded) + modified_image * mask_expanded

        return (modified_image,)


NODE_CLASS_MAPPINGS = {
    "VNCCS_Preprocessor": VNCCS_Preprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Preprocessor": "VNCCS Preprocessor",
}