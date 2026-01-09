"""VNCCS QWEN Detailer node

This node detects objects in an image using a bbox detector, then enhances each detected region
using QWEN image generation with the cropped region as reference.
- INPUTS: image, controlnet_image, bbox_detector, model, clip, vae, prompt, inpaint_prompt, instruction, etc.
- OUTPUTS: enhanced image

The node uses QWEN's vision-language capabilities to enhance detected regions.
ControlNet image (optional) provides additional detail guidance for each detected region.
ControlNet image is automatically resized to match main image dimensions if needed.
- instruction: System prompt describing how to analyze and modify images
- inpaint_prompt: Specific instructions for what to do with each detected region

This file relies on runtime objects provided by ComfyUI.
Includes local implementations of tensor_crop and tensor_paste to avoid impact.utils dependency.
"""

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
import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from nodes import MAX_RESOLUTION

def _tensor_resize(image, width, height, method="lanczos"):
    """
    Resize image tensor [B, H, W, C] to new dimensions.
    If method is 'lanczos', uses PIL for high-quality downsampling (prevents compression artifacts).
    """
    if image.shape[1] == height and image.shape[2] == width:
        return image
        
    if method == "lanczos":
        import torch
        import numpy as np
        from PIL import Image
        
        # Ensure image is in [B, H, W, C]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        res_images = []
        for img in image:
            # Convert to PIL
            img_np = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize
            scaled_pil = pil_img.resize((width, height), resample=Image.Resampling.LANCZOS)
            
            # Convert back to tensor
            scaled_tensor = torch.from_numpy(np.array(scaled_pil).astype(np.float32) / 255.0)
            res_images.append(scaled_tensor)
            
        return torch.stack(res_images, dim=0).to(image.device)
    else:
        # Use PyTorch interpolate for other methods (bicubic, area, etc)
        import torch.nn.functional as F
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Convert to [B, C, H, W] for interpolate
        image_bchw = image.permute(0, 3, 1, 2)
        
        kwargs = {"size": (height, width), "mode": method}
        if method in ["linear", "bilinear", "bicubic", "trilinear"]:
            kwargs["align_corners"] = False
            
        scaled = F.interpolate(image_bchw, **kwargs)
        return scaled.permute(0, 2, 3, 1)  # Back to [B, H, W, C]

try:
    import comfy.samplers
    SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
    SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS
except Exception:
    # Fallback for environments without ComfyUI
    SAMPLER_NAMES = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "uni_pc", "uni_pc_bh2"]
    SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]


def _make_square_crop(image_tensor, crop_region, full_image_shape, target_qwen_size=None):
    """
    Convert rectangular crop into square by extending to sides with available space.
    If not enough space, pad with black pixels and track padding info.
    
    Args:
        image_tensor: [B, H, W, C] full image tensor
        crop_region: (x1, y1, x2, y2) in image coordinates
        full_image_shape: (H, W) of full image
        target_qwen_size: Target square size (e.g. 1024). If None, use adaptive defaults.
    
    Returns:
        square_crop: [1, S, S, C] square crop at target size
        square_info: dict with metadata for later unpadding
            - orig_size: (h, w) original crop size
            - square_size: S (square side)
            - pad_top, pad_bottom, pad_left, pad_right: pixels added
            - was_padded: bool, if True need to crop black pixels after generation
    """
    import torch
    
    x1, y1, x2, y2 = crop_region
    crop_h = y2 - y1
    crop_w = x2 - x1
    full_h, full_w = full_image_shape
    
    # Make square by extending to the longer side
    square_size = max(crop_h, crop_w)
    
    # Calculate how much space we need on each side
    pad_h = square_size - crop_h
    pad_w = square_size - crop_w
    
    # Try to balance padding: add to sides with available space
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Adjust if we're at image boundaries
    if y1 - pad_top < 0:
        pad_bottom += pad_top - y1
        pad_top = y1
    if y2 + pad_bottom > full_h:
        pad_top += (y2 + pad_bottom) - full_h
        pad_bottom = full_h - y2
    
    if x1 - pad_left < 0:
        pad_right += pad_left - x1
        pad_left = x1
    if x2 + pad_right > full_w:
        pad_left += (x2 + pad_right) - full_w
        pad_right = full_w - x2
    
    # Extract square region from image
    square_y1 = max(0, y1 - pad_top)
    square_y2 = min(full_h, y2 + pad_bottom)
    square_x1 = max(0, x1 - pad_left)
    square_x2 = min(full_w, x2 + pad_right)
    
    # Crop from image
    square_crop = _tensor_crop(image_tensor, (square_x1, square_y1, square_x2, square_y2))
    
    # Calculate actual padding that was applied
    actual_pad_top = y1 - square_y1
    actual_pad_left = x1 - square_x1
    actual_pad_bottom = square_y2 - y2
    actual_pad_right = square_x2 - x2
    actual_square_size = square_y2 - square_y1
    
    # If still not square, pad with black
    if actual_square_size != square_x2 - square_x1:
        # This shouldn't happen with current logic, but be safe
        target_size_side = max(square_y2 - square_y1, square_x2 - square_x1)
        h_current, w_current = actual_square_size, square_x2 - square_x1
        pad_h_final = target_size_side - h_current
        pad_w_final = target_size_side - w_current
        
        if pad_h_final > 0 or pad_w_final > 0:
            padded = torch.nn.functional.pad(
                square_crop.permute(0, 3, 1, 2),
                (0, pad_w_final, 0, pad_h_final),
                mode='constant',
                value=0
            ).permute(0, 2, 3, 1)
            square_crop = padded
    
    # Resize to target size for QWEN
    import torch.nn.functional as F
    
    # If target_qwen_size is not provided, use adaptive defaults
    if target_qwen_size is None:
        if actual_square_size < 256:
            target_qwen_size = 512
        elif actual_square_size < 512:
            target_qwen_size = 768
        else:
            target_qwen_size = 1536
    
    square_crop_resized = F.interpolate(
        square_crop.permute(0, 3, 1, 2),
        size=(target_qwen_size, target_qwen_size),
        mode='bicubic',
        align_corners=False
    ).permute(0, 2, 3, 1)
    
    square_info = {
        'orig_size': (crop_h, crop_w),
        'orig_crop_region': crop_region,
        'square_size': actual_square_size,
        'square_crop_region': (square_x1, square_y1, square_x2, square_y2),
        'pad_top': actual_pad_top,
        'pad_bottom': actual_pad_bottom,
        'pad_left': actual_pad_left,
        'pad_right': actual_pad_right,
        'was_padded': actual_pad_top > 0 or actual_pad_bottom > 0 or actual_pad_left > 0 or actual_pad_right > 0,
        'target_qwen_size': target_qwen_size  # Store for reverse transformation
    }
    
    return square_crop_resized, square_info



def _unsquare_crop(square_image_resized, square_info):
    """
    Reverse the square crop transformation:
    1. Resize from adaptive size (512/768/1536) back to original square size
    2. Remove black padding if added
    3. Return to original crop size
    """
    import torch
    import torch.nn.functional as F
    
    # Get the QWEN target size from square_info (defaults to 1536 for backward compatibility)
    target_qwen_size = square_info.get('target_qwen_size', 1536)
    square_size = square_info['square_size']
    orig_h, orig_w = square_info['orig_size']
    
    # Step 1: Resize from QWEN size back to square size
    # For downscaling: use lanczos (via PIL) for high quality
    if target_qwen_size > square_size:
        resize_mode_1 = 'lanczos'
    else:
        resize_mode_1 = 'bicubic'
    
    square_resized = _tensor_resize(square_image_resized, square_size, square_size, method=resize_mode_1)
    
    # Remove padding if it was added
    if square_info['was_padded']:
        pad_top = square_info['pad_top']
        pad_bottom = square_info['pad_bottom']
        pad_left = square_info['pad_left']
        pad_right = square_info['pad_right']
        
        h = square_size
        w = square_size
        
        crop_h_start = pad_top
        crop_h_end = h - pad_bottom
        crop_w_start = pad_left
        crop_w_end = w - pad_right
        
        square_unpadded = square_resized[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end, :]
    else:
        square_unpadded = square_resized
    
    # Step 2: Resize to original crop size
    # This is ALWAYS downscaling, use lanczos for high quality
    resize_mode_2 = 'lanczos' 
    
    original_resized = _tensor_resize(square_unpadded, orig_w, orig_h, method=resize_mode_2)
    
    return original_resized


def _tensor_crop(image, crop_region):

    """Crop tensor image using crop_region coordinates [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = crop_region
    return image[:, y1:y2, x1:x2, :]


def _tensor_paste(image1, image2, crop_region, feather=0, mask=None):
    """Paste image2 onto image1 at crop_region position with optional feather blending and/or custom mask"""
    
    x1, y1, x2, y2 = crop_region
    h, w = y2 - y1, x2 - x1
    
    # Ensure image2 has batch dimension
    if len(image2.shape) == 3:  # [H, W, C] format
        image2 = image2.unsqueeze(0)
    
    # Ensure image2 matches the crop region size
    if image2.shape[1] != h or image2.shape[2] != w:
        # Resize image2 to match crop region
        # Use lanczos for downscaling, bicubic for upscaling
        is_downscaling = (h < image2.shape[1]) or (w < image2.shape[2])
        resize_mode = 'lanczos' if is_downscaling else 'bicubic'
        
        image2 = _tensor_resize(image2, w, h, method=resize_mode)
    
    # Remove batch dimension if present for single image pasting
    if image2.shape[0] == 1:
        image2 = image2.squeeze(0)
    
    # Final mask calculation
    final_mask = None
    
    # 1. Use custom mask (e.g. from SAM) if provided
    if mask is not None:
        # Convert numpy array to tensor if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().to(image2.device)
        
        # Ensure mask is [H, W] and matches current crop size
        if len(mask.shape) == 4: # [B, C, H, W]
            mask = mask.squeeze(0).squeeze(0)
        elif len(mask.shape) == 3: # [C, H, W] or [H, W, C]
            if mask.shape[0] == 1: mask = mask.squeeze(0)
            elif mask.shape[2] == 1: mask = mask.squeeze(2)
        
        # Normalize mask to [0, 1] range (may come as 0-255 or other ranges)
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_max > 1.0:
            mask = (mask - mask_min) / (mask_max - mask_min) if mask_max != mask_min else mask
            
        if mask.shape[0] != h or mask.shape[1] != w:
            # Resize mask to current dimensions
            import torch.nn.functional as F
            mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            final_mask = mask_resized.squeeze(0).squeeze(0)
        else:
            final_mask = mask
    
    # 2. Apply feathering if needed
    if feather > 0:
        # Create a basic box mask if no mask exists
        if final_mask is None:
            final_mask = torch.ones((h, w), dtype=image2.dtype, device=image2.device)
            
        # Apply simple box blur to create feather effect
        import torch.nn.functional as F
        mask_4d = final_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel_size = min(feather + 1, min(h, w) // 2 * 2 + 1)
        if kernel_size > 1:
            kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=final_mask.dtype, device=final_mask.device)
            kernel = kernel / kernel.numel()
            mask_4d = F.conv2d(mask_4d, kernel, padding=kernel_size//2)
            if mask_4d.shape[2:] != (h, w):
                mask_4d = F.interpolate(mask_4d, size=(h, w), mode='bilinear', align_corners=False)
        final_mask = torch.clamp(mask_4d.squeeze(0).squeeze(0), 0, 1)
        
    if final_mask is not None:
        # Expand mask to match image channels
        mask_expanded = final_mask.unsqueeze(-1).expand_as(image2)  # [H, W, C]
        
        # When we have a SAM mask, use it to blend only at the EDGES
        # The mask=1 area should be 100% image2, only blend where mask transitions
        # Apply a strong edge mask: use only where mask < 0.95 (the transition zone)
        if feather > 0:
            # With feathering: use the feathered mask (it's already transition-aware)
            image1[:, y1:y2, x1:x2, :] = image1[:, y1:y2, x1:x2, :] * (1 - mask_expanded) + image2 * mask_expanded
        else:
            # Without feathering: use mask only to create soft edges (full opacity in center)
            # Create a threshold to ensure center is fully replaced
            threshold = 0.5
            edge_mask = torch.clamp(final_mask * 2, 0, 1)  # Make transition sharper
            edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(image2)
            image1[:, y1:y2, x1:x2, :] = image1[:, y1:y2, x1:x2, :] * (1 - edge_mask_expanded) + image2 * edge_mask_expanded
    else:
        # Paste with full opacity
        image1[:, y1:y2, x1:x2, :] = image2
    
    return image1


class VNCCS_QWEN_Detailer:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "bbox_detector": ("BBOX_DETECTOR", ),
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 300, "min": -512, "max": 512, "step": 1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 300, "step": 1}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (SAMPLER_NAMES, ),
                "scheduler": (SCHEDULER_NAMES, ),
                "denoise": ("FLOAT", {"default": 1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tiled_vae_decode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
            "optional": {
                "controlnet_image": ("IMAGE", ),
                "sam_model_opt": ("SAM_MODEL", ),
                "segm_detector_opt": ("SEGM_DETECTOR", ),
                "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"], {"default": "center-1"}),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"], {"default": "False"}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                "inpaint_mode": ("BOOLEAN", {"default": False}),
                "inpaint_prompt": ("STRING", {"multiline": True, "default": "[!!!IMPORTANT!!!] Inpaint mode: draw only inside black box."}),
                "color_match_method": (["disabled", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"], {"default": "disabled"}),
                "qwen_2511": ("BOOLEAN", {"default": True}),
                "distortion_fix": ("BOOLEAN", {"default": True, "label_on": "Drift Fix", "label_off": "Standard VL"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"

    CATEGORY = "VNCCS/detailing"

    def detail(self, image, bbox_detector, model, clip, vae, prompt,
                threshold=0.5, dilation=10, drop_size=10,
                feather=0, steps=20, cfg=8.0, seed=0, sampler_name="euler", scheduler="normal",
                denoise=0.5, tiled_vae_decode=False, tile_size=512, controlnet_image=None, target_size=1024,
                upscale_method="lanczos", crop_method="center",
                instruction="", inpaint_mode=False, inpaint_prompt="", qwen_2511=True,
                color_match_method="disabled",
                sam_model_opt=None, segm_detector_opt=None,
                sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
                sam_bbox_expansion=0, sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative="False",
                distortion_fix=True):

        # Fixed parameters
        crop_factor = 1.0
        target_vl_size = 384
        color_match_strength = 1.0
        color_match_multithread = True

        if len(image) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_QWEN_Detailer does not allow image batches.')

        if controlnet_image is not None:
            if len(controlnet_image) > 1:
                raise Exception('[VNCCS] ERROR: VNCCS_QWEN_Detailer does not allow controlnet image batches.')

            # Auto-resize controlnet_image to match main image dimensions if needed
            if controlnet_image.shape[1:3] != image.shape[1:3]:  # Check height and width
                # Resize ControlNet image to match main image dimensions
                import torch.nn.functional as F
                controlnet_image = F.interpolate(
                    controlnet_image.permute(0, 3, 1, 2),  # [B, H, W, C] -> [B, C, H, W]
                    size=(image.shape[1], image.shape[2]),   # Target height, width
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Detect segments using bbox detector
        try:
            segs_result = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size)
            
            # Refine detection with SAM or Segm if provided (Impact Pack logic)
            if sam_model_opt is not None:
                import impact.core as core
                sam_mask = core.make_sam_mask(sam_model_opt, segs_result, image, sam_detection_hint, sam_dilation,
                                              sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                              sam_mask_hint_use_negative)
                segs_result = core.segs_bitwise_and_mask(segs_result, sam_mask)
            elif segm_detector_opt is not None:
                import impact.core as core
                segm_segs = segm_detector_opt.detect(image, threshold, dilation, crop_factor, drop_size)
                # Apply bitwise AND with segm mask
                segm_mask = core.segs_to_combined_mask(segm_segs)
                segs_result = core.segs_bitwise_and_mask(segs_result, segm_mask)
                
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to detect/refine segments: {str(e)}')
        
        # Handle different return formats from bbox detectors
        if isinstance(segs_result, tuple) and len(segs_result) == 2:
            # Standard Impact Pack format: (shape, [SEG, ...])
            segs = segs_result
        else:
            # Fallback: assume segs_result is already the (shape, [SEG, ...]) tuple
            segs = segs_result

        # Validate segs format
        if not isinstance(segs, tuple) or len(segs) != 2:
            raise Exception(f'[VNCCS] ERROR: Invalid segs format from bbox_detector. Expected tuple of length 2, got: {type(segs)}')
        
        if not isinstance(segs[1], (list, tuple)):
            raise Exception(f'[VNCCS] ERROR: Invalid segments list. Expected list or tuple, got: {type(segs[1])}')

        # Apply dilation to crop_region manually to control cropped image size
        image_height, image_width = segs[0]
        valid_segs = []
        min_region_size = 10  # Minimum width and height to avoid too small regions
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            x1 = max(0, x1 - dilation)
            y1 = max(0, y1 - dilation)
            x2 = min(image_width, x2 + dilation)
            y2 = min(image_height, y2 + dilation)
            
            # Extract cropped_mask if available (for SAM/Segm refinement)
            cropped_mask = None
            if hasattr(seg, 'cropped_mask') and seg.cropped_mask is not None:
                cropped_mask = seg.cropped_mask.copy() if isinstance(seg.cropped_mask, np.ndarray) else seg.cropped_mask
            
            if x2 - x1 >= min_region_size and y2 - y1 >= min_region_size:
                # Create new seg-like object with adjusted crop_region
                adjusted_seg = type('AdjustedSEG', (), {
                    'crop_region': (x1, y1, x2, y2),
                    'bbox': seg.bbox if hasattr(seg, 'bbox') else (x1, y1, x2, y2),
                    'cropped_mask': cropped_mask
                })()
                # Copy other attributes if needed
                for attr in dir(seg):
                    if not attr.startswith('_') and not hasattr(adjusted_seg, attr):
                        try:
                            setattr(adjusted_seg, attr, getattr(seg, attr))
                        except:
                            pass
                valid_segs.append(adjusted_seg)
        segs = (segs[0], valid_segs)

        if len(segs[1]) == 0:
            # No segments detected, return original image
            return (image,)

        enhanced_image = image.clone()

        for seg_idx, seg in enumerate(segs[1]):
            # Crop the segment from the image
            cropped_image = _tensor_crop(enhanced_image, seg.crop_region)
            
            # Apply inpaint mode: fill bbox area with black, keeping dilation context (crop_factor is fixed at 1.0)
            if inpaint_mode and hasattr(seg, 'bbox'):
                # Calculate bbox position relative to crop_region
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = seg.bbox
                crop_x1, crop_y1, crop_x2, crop_y2 = seg.crop_region
                rel_x1 = max(0, int(bbox_x1 - crop_x1))
                rel_y1 = max(0, int(bbox_y1 - crop_y1))
                rel_x2 = min(int(crop_x2 - crop_x1), int(bbox_x2 - crop_x1))
                rel_y2 = min(int(crop_y2 - crop_y1), int(bbox_y2 - crop_y1))
                if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                    cropped_image[0, rel_y1:rel_y2, rel_x1:rel_x2, :] = 0  # Fill with black
                
                # Calculate extended bbox for controlnet: bbox + half dilation
                dilation_x = int(bbox_x1 - crop_x1)
                dilation_y = int(bbox_y1 - crop_y1)
                half_dilation = max(dilation_x, dilation_y) // 2
                ext_x1 = max(crop_x1, int(bbox_x1 - half_dilation))
                ext_y1 = max(crop_y1, int(bbox_y1 - half_dilation))
                ext_x2 = min(crop_x2, int(bbox_x2 + half_dilation))
                ext_y2 = min(crop_y2, int(bbox_y2 + half_dilation))
                extended_bbox = (ext_x1, ext_y1, ext_x2, ext_y2)
                # Relative positions for extended bbox
                rel_ext_x1 = max(0, int(ext_x1 - crop_x1))
                rel_ext_y1 = max(0, int(ext_y1 - crop_y1))
                rel_ext_x2 = min(int(crop_x2 - crop_x1), int(ext_x2 - crop_x1))
                rel_ext_y2 = min(int(crop_y2 - crop_y1), int(ext_y2 - crop_y1))
            
            # Crop the corresponding region from controlnet image if provided
            cropped_controlnet = None
            if controlnet_image is not None:
                # In inpaint mode, overlay controlnet extended bbox data on a copy of cropped_image
                if inpaint_mode and hasattr(seg, 'bbox'):
                    cropped_controlnet = cropped_image.clone()  # Use cropped_image as base
                    # Crop the extended bbox region from controlnet_image
                    bbox_crop = _tensor_crop(controlnet_image, tuple(int(x) for x in extended_bbox))
                    # Overlay controlnet data onto the extended bbox area
                    cropped_controlnet[0, rel_ext_y1:rel_ext_y2, rel_ext_x1:rel_ext_x2, :] = bbox_crop[0, :, :, :]
                else:
                    cropped_controlnet = _tensor_crop(controlnet_image, seg.crop_region)
            
            # Pre-scale cropped image to target_size by long side before encoding
            # This ensures consistent high-quality latent encoding without re-scaling in encode_qwen
            import torch.nn.functional as F
            crop_h_actual = cropped_image.shape[1]
            crop_w_actual = cropped_image.shape[2]
            
            # **NEW: Convert crop to square for better QWEN processing**
            square_image, square_info = _make_square_crop(enhanced_image, seg.crop_region, (enhanced_image.shape[1], enhanced_image.shape[2]), target_qwen_size=target_size)
            if cropped_controlnet is not None:
                # Also square the controlnet if it exists
                square_controlnet, _ = _make_square_crop(controlnet_image, seg.crop_region, (controlnet_image.shape[1], controlnet_image.shape[2]), target_qwen_size=target_size)
            else:
                square_controlnet = None
            
            cropped_image = square_image
            if cropped_controlnet is not None:
                cropped_controlnet = square_controlnet
            
            # Save original crop for drift correction reference (it's already at target_size now)
            original_crop = cropped_image.clone()
            
            # Use the cropped image as reference for QWEN generation
            # encode_qwen will handle consistent encoding without additional re-scaling
            positive_cond, latent, _, _, _, negative_cond, _, _ = self.encode_qwen(

                clip=clip, prompt=prompt, vae=vae,
                image1=cropped_image, image2=cropped_controlnet,
                target_size=target_size, target_vl_size=target_vl_size,
                upscale_method=upscale_method, crop_method=crop_method,
                instruction=instruction, inpaint_mode=inpaint_mode, inpaint_prompt=inpaint_prompt,
                distortion_fix=distortion_fix
            )

            # Generate enhanced version using the model
            latent_tensor = latent["samples"] if isinstance(latent, dict) and "samples" in latent else latent
            latent_dict = {"samples": latent_tensor} if not isinstance(latent_tensor, dict) else latent_tensor
            # Apply QWEN 2511 conditioning modification if enabled
            try:
                if qwen_2511:
                    method = "index_timestep_zero"
                    positive_cond = node_helpers.conditioning_set_values(positive_cond, {"reference_latents_method": method})
                    negative_cond = node_helpers.conditioning_set_values(negative_cond, {"reference_latents_method": method})
            except Exception:
                pass

            samples = self.sample_latent(model, latent_dict, positive_cond, negative_cond, steps, cfg, seed, sampler_name, scheduler, denoise)
            
            # common_ksampler returns a list, get the first (and only) latent
            if isinstance(samples, (list, tuple)) and len(samples) > 0:
                samples = samples[0]
            if isinstance(samples, dict) and "samples" in samples:
                samples = samples["samples"]

            # Decode to image
            if tiled_vae_decode:
                enhanced_cropped = vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size)
            else:
                enhanced_cropped = vae.decode(samples)
            
            # Ensure proper image format [B, H, W, C] and remove batch dimension if present
            if len(enhanced_cropped.shape) == 4 and enhanced_cropped.shape[1] in [1, 3, 4]:  # [B, C, H, W] format
                enhanced_cropped = enhanced_cropped.permute(0, 2, 3, 1)  # Convert to [B, H, W, C]
            elif len(enhanced_cropped.shape) == 3:  # [H, W, C] format
                enhanced_cropped = enhanced_cropped.unsqueeze(0)  # Add batch dimension
            
            # Remove batch dimension for single image
            if enhanced_cropped.shape[0] == 1:
                enhanced_cropped = enhanced_cropped.squeeze(0)
            
            # **NEW: Reverse the square transformation (unsquare and remove padding)**
            if len(enhanced_cropped.shape) == 3:  # [H, W, C]
                enhanced_cropped = enhanced_cropped.unsqueeze(0)  # Add batch: [1, H, W, C]
            enhanced_cropped = _unsquare_crop(enhanced_cropped, square_info)
            if enhanced_cropped.shape[0] == 1:
                enhanced_cropped = enhanced_cropped.squeeze(0)  # Remove batch
            
            # Debug: Check if enhanced_cropped matches original crop size
            orig_h, orig_w = square_info['orig_size']
            actual_h, actual_w = enhanced_cropped.shape[0], enhanced_cropped.shape[1]
            
            # If there's a mismatch, resize to exact original crop size to avoid _tensor_paste doing extra resize
            if actual_h != orig_h or actual_w != orig_w:
                enhanced_cropped = _tensor_resize(enhanced_cropped, orig_w, orig_h, method='lanczos')
                if enhanced_cropped.shape[0] == 1:
                    enhanced_cropped = enhanced_cropped.squeeze(0)

            # Apply color matching BEFORE pasting (on clean generated image, not mixed)
            if color_match_method != "disabled":
                # Crop the region from ORIGINAL image for color matching reference
                cropped_original = _tensor_crop(image, seg.crop_region)
                # Apply color match to the clean generated enhanced_cropped
                enhanced_cropped = self._apply_color_match(cropped_original, enhanced_cropped, color_match_method, color_match_strength, color_match_multithread)

            # Paste back to the original image (now with color-corrected enhanced_cropped)
            seg_mask = seg.cropped_mask if hasattr(seg, 'cropped_mask') else None
            enhanced_image = _tensor_paste(enhanced_image, enhanced_cropped, seg.crop_region, feather, mask=seg_mask)
            
        return (enhanced_image,)

    def _apply_color_match(self, image_ref, image_target, method, strength=1.0, multithread=True):
        """
        Apply color matching from reference image to target image.
        """
        if strength == 0:
            return image_target

        try:
            from color_matcher import ColorMatcher
        except ImportError:
            return image_target
        
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0) if len(image_target.shape) == 4 else 1
        
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        def process(i):
            try:
                cm = ColorMatcher()
                image_target_np_i = images_target_np if batch_size == 1 else (images_target_np[i] if images_target_np.ndim == 4 else images_target_np)
                image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else (image_ref_np[i] if image_ref_np.ndim == 4 else image_ref_np)
                
                # Main color transfer process
                image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method)
                
                if strength != 1:
                    # Apply strength blending
                    image_result = image_target_np_i + strength * (image_result - image_target_np_i)
                
                return torch.from_numpy(image_result).float()
                
            except Exception as e:
                image_target_np_i = images_target_np if batch_size == 1 else (images_target_np[i] if images_target_np.ndim == 4 else images_target_np)
                return torch.from_numpy(image_target_np_i).float()

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out = out.clamp(0, 1)
        return out

    def encode_qwen(self, clip, prompt, vae=None, image1=None, image2=None,
                    target_size=1024, target_vl_size=384,
                    upscale_method="lanczos", crop_method="center",
                    instruction="", inpaint_mode=False, inpaint_prompt="",
                    distortion_fix=True):

        pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 0}
        ref_latents = []
        images = [{"image": image1, "vl_resize": True}]
        if image2 is not None:
            images.append({"image": image2, "vl_resize": True})

        vl_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = instruction or "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""

        if inpaint_mode:
            base_prompt = inpaint_prompt or "(Fill black area with image) "
            base_prompt += prompt
        else:
            base_prompt = prompt

        # If distortion_fix is enabled, we skip standard VL image logic for the clip
        # but ALWAYS process images to get ref_latents
        for i, image_obj in enumerate(images):
            image = image_obj["image"]
            vl_resize = image_obj["vl_resize"]
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = (samples.shape[3] * samples.shape[2])
                
                # Standard VL processing logic (Used if NOT distortion_fix)
                if not distortion_fix:
                    total = int(target_vl_size * target_vl_size)
                    scale_by = math.sqrt(total / current_total)
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop_method)
                    image_vl = s.movedim(1, -1)
                    vl_images.append(image_vl)
                    image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                
                # VAE encode for ref_latents (Always done)
                if vae is not None:
                    width = (samples.shape[3] + 7) // 8 * 8
                    height = (samples.shape[2] + 7) // 8 * 8
                    crop = crop_method
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)

                    image_vae = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image_vae[:, :, :, :3]))

        # Condition logic based on distortion_fix
        if distortion_fix:
            # Fix mode: Only text prompt, rely on ref_latents for guidance
            tokens = clip.tokenize(base_prompt, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            # Standard mode: Include image_prompt chunks and VL images
            tokens = clip.tokenize(image_prompt + base_prompt, images=vl_images, llama_template=llama_template)
            condition_inputs = {
                 "vl_images": vl_images if vl_images else None
            }
            # Note: clip.encode_from_tokens_scheduled doesn't take images kwarg directly usually,
            # but QWEN patched clips might. Assuming standard usage pattern for QWEN encoders:
            # Standard Encoder uses: `tokens = clip.tokenize(image_prompt + prompt, images=vl_images, ...)`
            # and then `conditioning = clip.encode_from_tokens_scheduled(tokens)`
            conditioning = clip.encode_from_tokens_scheduled(tokens)

        # ReferenceLatent technique
        conditioning_with_ref = conditioning
        if len(ref_latents) > 0:
            conditioning_with_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        conditioning_negative = conditioning_with_ref if len(ref_latents) > 0 else conditioning

        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}

        return (conditioning_with_ref, latent_out, image1, None, None, conditioning_negative, conditioning, conditioning)

    def sample_latent(self, model, latent, positive, negative, steps, cfg, seed, sampler_name, scheduler, denoise):
        # Use ComfyUI's common_ksampler
        from nodes import common_ksampler
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)


class VNCCS_BBox_Extractor:
    """Extract regions detected by BBox detector from image"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 300, "min": -512, "max": 512, "step": 1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "extract"
    CATEGORY = "VNCCS/detailing"

    def extract(self, image, bbox_detector, threshold=0.5, dilation=300, drop_size=10):
        """Extract regions detected by BBox detector"""
        
        if len(image) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_BBox_Extractor does not allow image batches.')

        # Fixed crop_factor for bbox detection
        crop_factor = 1.0
        
        try:
            segs_result = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size)
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to detect segments with bbox_detector: {str(e)}')
        
        # Handle different return formats from bbox detectors
        if isinstance(segs_result, tuple) and len(segs_result) == 2:
            segs = segs_result
        else:
            segs = segs_result

        # Validate segs format
        if not isinstance(segs, tuple) or len(segs) != 2:
            raise Exception(f'[VNCCS] ERROR: Invalid segs format from bbox_detector. Expected tuple of length 2, got: {type(segs)}')
        
        if not isinstance(segs[1], (list, tuple)):
            raise Exception(f'[VNCCS] ERROR: Invalid segments list. Expected list or tuple, got: {type(segs[1])}')

        # Apply dilation to crop_region manually
        image_height, image_width = segs[0]
        valid_segs = []
        min_region_size = 10  # Minimum width and height to avoid too small regions
        
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            x1 = max(0, x1 - dilation)
            y1 = max(0, y1 - dilation)
            x2 = min(image_width, x2 + dilation)
            y2 = min(image_height, y2 + dilation)
            
            # Extract cropped_mask if available (for SAM/Segm refinement)
            cropped_mask = None
            if hasattr(seg, 'cropped_mask') and seg.cropped_mask is not None:
                cropped_mask = seg.cropped_mask.copy() if isinstance(seg.cropped_mask, np.ndarray) else seg.cropped_mask
            
            if x2 - x1 >= min_region_size and y2 - y1 >= min_region_size:
                # Create new seg-like object with adjusted crop_region
                adjusted_seg = type('AdjustedSEG', (), {
                    'crop_region': (x1, y1, x2, y2),
                    'bbox': seg.bbox if hasattr(seg, 'bbox') else (x1, y1, x2, y2),
                    'cropped_mask': cropped_mask
                })()
                # Copy other attributes if needed
                for attr in dir(seg):
                    if not attr.startswith('_') and not hasattr(adjusted_seg, attr):
                        try:
                            setattr(adjusted_seg, attr, getattr(seg, attr))
                        except:
                            pass
                valid_segs.append(adjusted_seg)

        if len(valid_segs) == 0:
            # No segments detected, return empty image
            return (torch.zeros((1, 1, 1, 3), dtype=image.dtype, device=image.device),)

        # Extract cropped images
        extracted_images = []
        max_h = 0
        max_w = 0
        
        # First pass: find max dimensions
        for seg in valid_segs:
            x1, y1, x2, y2 = seg.crop_region
            h = y2 - y1
            w = x2 - x1
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        
        # Second pass: crop and pad to max dimensions
        for i, seg in enumerate(valid_segs):
            cropped = _tensor_crop(image, seg.crop_region)
            x1, y1, x2, y2 = seg.crop_region
            h = y2 - y1
            w = x2 - x1
            
            # Pad if needed to match max dimensions
            if h < max_h or w < max_w:
                pad_h_bottom = max_h - h
                pad_w_right = max_w - w
                cropped = torch.nn.functional.pad(cropped, (0, 0, 0, pad_w_right, 0, pad_h_bottom), mode='constant', value=0)
            
            extracted_images.append(cropped)

        # Concatenate all extracted images along batch dimension
        extracted_batch = torch.cat(extracted_images, dim=0)
        
        return (extracted_batch,)


# Registration mapping so Comfy finds the node
NODE_CLASS_MAPPINGS = {
    "VNCCS_QWEN_Detailer": VNCCS_QWEN_Detailer,
    "VNCCS_BBox_Extractor": VNCCS_BBox_Extractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_QWEN_Detailer": "VNCCS QWEN Detailer",
    "VNCCS_BBox_Extractor": "VNCCS BBox Extractor",
}
