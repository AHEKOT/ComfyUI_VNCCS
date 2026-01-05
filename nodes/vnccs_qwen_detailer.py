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

try:
    import comfy.samplers
    SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
    SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS
except Exception:
    # Fallback for environments without ComfyUI
    SAMPLER_NAMES = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "uni_pc", "uni_pc_bh2"]
    SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]


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
    
    # Ensure image2 matches the crop region size
    if image2.shape[1] != h or image2.shape[2] != w:
        # Resize image2 to match crop region
        # Handle different input formats
        if len(image2.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
            if image2.shape[-1] in [1, 3, 4]:  # [B, H, W, C] format
                image2 = torch.nn.functional.interpolate(image2.permute(0, 3, 1, 2), 
                                                       size=(h, w), 
                                                       mode='bicubic', 
                                                       align_corners=False).permute(0, 2, 3, 1)
            else:  # [B, C, H, W] format  
                image2 = torch.nn.functional.interpolate(image2, 
                                                       size=(h, w), 
                                                       mode='bicubic', 
                                                       align_corners=False).permute(0, 2, 3, 1)
        elif len(image2.shape) == 3:  # [H, W, C] format
            image2 = image2.unsqueeze(0)  # Add batch dimension
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
class VNCCS_QWEN_Detailer:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

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
                "feather": ("INT", {"default": 5, "min": 0, "max": 300, "step": 1}),
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
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                "inpaint_mode": ("BOOLEAN", {"default": False}),
                "inpaint_prompt": ("STRING", {"multiline": True, "default": "[!!!IMPORTANT!!!] Inpaint mode: draw only inside black box."}),
                "qwen_2511": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"

    CATEGORY = "VNCCS/detailing"

    def detail(self, image, bbox_detector, model, clip, vae, prompt,
                threshold=0.5, dilation=10, drop_size=10,
                feather=5, steps=20, cfg=8.0, seed=0, sampler_name="euler", scheduler="normal",
                denoise=0.5, tiled_vae_decode=False, tile_size=512, controlnet_image=None, target_size=1024,
                upscale_method="lanczos", crop_method="center",
                instruction="", inpaint_mode=False, inpaint_prompt="", qwen_2511=True):

        # Fixed crop_factor for bbox detection
        crop_factor = 1.0
        # Fixed target_vl_size for QWEN vision processing
        target_vl_size = 384

        if len(image) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_QWEN_Detailer does not allow image batches.')

        if controlnet_image is not None:
            if len(controlnet_image) > 1:
                raise Exception('[VNCCS] ERROR: VNCCS_QWEN_Detailer does not allow controlnet image batches.')

            # Auto-resize controlnet_image to match main image dimensions if needed
            if controlnet_image.shape[1:3] != image.shape[1:3]:  # Check height and width
                print(f'[VNCCS] INFO: Resizing ControlNet image from {controlnet_image.shape[1:3]} to {image.shape[1:3]}')
                # Resize ControlNet image to match main image dimensions
                import torch.nn.functional as F
                controlnet_image = F.interpolate(
                    controlnet_image.permute(0, 3, 1, 2),  # [B, H, W, C] -> [B, C, H, W]
                    size=(image.shape[1], image.shape[2]),   # Target height, width
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

                # Detect segments using bbox detector
        print(f'[VNCCS] DEBUG: Using bbox_detector: {type(bbox_detector).__name__}')
        print(f'[VNCCS] DEBUG: Calling bbox_detector.detect with params:')
        print(f'[VNCCS] DEBUG:   threshold={threshold}, dilation={dilation}, crop_factor={crop_factor} (fixed), drop_size={drop_size}')
        
        try:
            segs_result = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size)
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to detect segments with bbox_detector: {str(e)}')
        
        # Handle different return formats from bbox detectors
        if isinstance(segs_result, tuple) and len(segs_result) == 2:
            # Standard Impact Pack format: (shape, [SEG, ...])
            segs = segs_result
        else:
            # Fallback: assume segs_result is already the (shape, [SEG, ...]) tuple
            segs = segs_result

        print(f'[VNCCS] DEBUG: segs[0] (shape info): {segs[0]}')
        print(f'[VNCCS] DEBUG: Total segments found: {len(segs[1])}')

        # Validate segs format
        if not isinstance(segs, tuple) or len(segs) != 2:
            raise Exception(f'[VNCCS] ERROR: Invalid segs format from bbox_detector. Expected tuple of length 2, got: {type(segs)}')
        
        if not isinstance(segs[1], (list, tuple)):
            raise Exception(f'[VNCCS] ERROR: Invalid segments list. Expected list or tuple, got: {type(segs[1])}')

        # Debug: Log dilation effect
        if len(segs[1]) > 0:
            print(f'[VNCCS] DEBUG: dilation={dilation}, found {len(segs[1])} segments')
            for i, seg in enumerate(segs[1][:3]):  # Log first 3 segments
                x1, y1, x2, y2 = seg.crop_region
                area = (x2 - x1) * (y2 - y1)
                print(f'[VNCCS] DEBUG: segment {i}: region=({x1},{y1},{x2},{y2}), area={area}px')
        else:
            print(f'[VNCCS] DEBUG: No segments found with dilation={dilation}')

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
            if x2 - x1 >= min_region_size and y2 - y1 >= min_region_size:
                # Create new seg-like object with adjusted crop_region
                adjusted_seg = type('AdjustedSEG', (), {
                    'crop_region': (x1, y1, x2, y2),
                    'bbox': seg.bbox if hasattr(seg, 'bbox') else (x1, y1, x2, y2)
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

        # Debug: Log after dilation application
        if len(segs[1]) > 0:
            print(f'[VNCCS] DEBUG: After applying dilation, {len(segs[1])} valid segments')
            for i, seg in enumerate(segs[1][:3]):  # Log first 3 segments
                x1, y1, x2, y2 = seg.crop_region
                area = (x2 - x1) * (y2 - y1)
                print(f'[VNCCS] DEBUG: segment {i}: adjusted region=({x1},{y1},{x2},{y2}), area={area}px')
        else:
            print(f'[VNCCS] DEBUG: No valid segments after dilation')

        if len(segs[1]) == 0:
            # No segments detected, return original image
            return (image,)

        enhanced_image = image.clone()

        for seg in segs[1]:
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
            
            # Calculate target size based on crop region to maintain quality
            crop_h, crop_w = seg.crop_region[3] - seg.crop_region[1], seg.crop_region[2] - seg.crop_region[0]
            # Use crop region size as target, but ensure minimum size for quality and divisible by 8 for VAE
            min_target_size = 256  # Minimum size to maintain quality
            dynamic_target_size = max(crop_h, crop_w, min_target_size)
            dynamic_target_size = ((dynamic_target_size + 7) // 8) * 8  # Round up to nearest multiple of 8
            
            # Use the cropped image as reference for QWEN generation
            # Create conditioning using QWEN encoder logic
            conditioning, latent, _, _, _, _, _, _ = self.encode_qwen(
                clip=clip, prompt=prompt, vae=vae,
                image1=cropped_image, image2=cropped_controlnet,
                target_size=dynamic_target_size, target_vl_size=target_vl_size,
                upscale_method=upscale_method, crop_method=crop_method,
                instruction=instruction, inpaint_mode=inpaint_mode, inpaint_prompt=inpaint_prompt
            )

            # Generate enhanced version using the model
            latent_tensor = latent["samples"] if isinstance(latent, dict) and "samples" in latent else latent
            latent_dict = {"samples": latent_tensor} if not isinstance(latent_tensor, dict) else latent_tensor
            # Apply QWEN 2511 conditioning modification if enabled
            try:
                if qwen_2511:
                    method = "index_timestep_zero"
                    conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": method})
            except Exception:
                pass

            samples = self.sample_latent(model, latent_dict, conditioning, [], steps, cfg, seed, sampler_name, scheduler, denoise)
            
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

            # Paste back to the original image
            enhanced_image = _tensor_paste(enhanced_image, enhanced_cropped, seg.crop_region, feather)

        return (enhanced_image,)

    def encode_qwen(self, clip, prompt, vae=None, image1=None, image2=None,
                    target_size=1024, target_vl_size=384,
                    upscale_method="lanczos", crop_method="center",
                    instruction="", inpaint_mode=False, inpaint_prompt=""):

        pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 0}
        ref_latents = []
        images = [{"image": image1, "vl_resize": True}]
        if image2 is not None:
            images.append({"image": image2, "vl_resize": True})

        vae_images = []
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

        for i, image_obj in enumerate(images):
            image = image_obj["image"]
            vl_resize = image_obj["vl_resize"]
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                current_total = (samples.shape[3] * samples.shape[2])
                
                # VAE encode (always resize to target_size)
                total = int(target_size * target_size)
                scale_by = math.sqrt(total / current_total)
                if crop_method == "pad":
                    crop = "center"
                    scaled_width = round(samples.shape[3] * scale_by)
                    scaled_height = round(samples.shape[2] * scale_by)
                    canvas_width = math.ceil(samples.shape[3] * scale_by / 8.0) * 8
                    canvas_height = math.ceil(samples.shape[2] * scale_by / 8.0) * 8

                    canvas = torch.zeros((samples.shape[0], samples.shape[1], canvas_height, canvas_width), dtype=samples.dtype, device=samples.device)
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]

                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    s = canvas
                else:
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    crop = crop_method
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)

                image_vae = s.movedim(1, -1)
                ref_latents.append(vae.encode(image_vae[:, :, :, :3]))
                vae_images.append(image_vae)

                # VL processing (always resize to target_vl_size for consistency)
                total = int(target_vl_size * target_vl_size)
                scale_by = math.sqrt(total / current_total)
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop_method)
                image_vl = s.movedim(1, -1)
                vl_images.append(image_vl)
                image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"

        tokens = clip.tokenize(image_prompt + base_prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        conditioning_full_ref = conditioning
        if len(ref_latents) > 0:
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}

        return (conditioning_full_ref, latent_out, image1, None, None, conditioning, conditioning, conditioning)

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

        print(f'[VNCCS] BBox Extractor: Using bbox_detector: {type(bbox_detector).__name__}')
        print(f'[VNCCS] BBox Extractor: threshold={threshold}, dilation={dilation}, drop_size={drop_size}')
        
        try:
            segs_result = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size)
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to detect segments with bbox_detector: {str(e)}')
        
        # Handle different return formats from bbox detectors
        if isinstance(segs_result, tuple) and len(segs_result) == 2:
            segs = segs_result
        else:
            segs = segs_result

        print(f'[VNCCS] BBox Extractor: Found {len(segs[1])} segments')

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
            if x2 - x1 >= min_region_size and y2 - y1 >= min_region_size:
                # Create new seg-like object with adjusted crop_region
                adjusted_seg = type('AdjustedSEG', (), {
                    'crop_region': (x1, y1, x2, y2),
                    'bbox': seg.bbox if hasattr(seg, 'bbox') else (x1, y1, x2, y2)
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
            print('[VNCCS] BBox Extractor: No valid segments found')
            return (torch.zeros((1, 1, 1, 3), dtype=image.dtype, device=image.device),)

        # Extract cropped images
        extracted_images = []
        for i, seg in enumerate(valid_segs):
            cropped = _tensor_crop(image, seg.crop_region)
            extracted_images.append(cropped)
            x1, y1, x2, y2 = seg.crop_region
            print(f'[VNCCS] BBox Extractor: Segment {i}: region=({x1},{y1},{x2},{y2}), size={x2-x1}x{y2-y1}')

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
