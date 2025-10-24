"""VNCCS QWEN Encoder node

This node is a simplified variant of the QWEN image-to-conditioning encoder
that accepts exactly three images and three per-image weights (0.0-1.0, step 0.01, quadratic mapping for fine control)
to control each image's influence via weighted reference latents. Also includes use_ref flags to exclude latents from reference_latents while keeping VL influence.

Class: VNCCS_QWEN_Encoder
- INPUTS: clip, prompt, vae (optional), image1/2/3, weight1..weight3, prompt_weight, use_ref1..3, resize/control flags
- OUTPUTS: conditioning_full_ref (weighted all), latent, images, conditioning_with_first_ref (weighted first only), conditioning_first_weighted (weighted first + weighted others)

This file relies on runtime objects provided by ComfyUI (clip, vae, comfy.utils, node_helpers).
"""

import types
import sys
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
        def common_upscale(samples, width, height, method, crop):
            # best-effort: return input unchanged
            return samples
    comfy = types.SimpleNamespace(utils=_ComfyUtilsFallback())

import math
try:
    import torch
except Exception:
    torch = None
try:
    import numpy as np
except Exception:
    np = None
try:
    from PIL import Image
except Exception:
    Image = None
import numbers


class VNCCS_QWEN_Encoder:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 8}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                "control_instruction": ("STRING", {"multiline": True, "default": "This control image defines the composition, structure, and style elements to be applied to the generated image. Determine the type of ControlNet that this image represents. Analyze how this control image influences the overall layout, proportions, and visual style, and incorporate those changes accordingly."}),
                # per-image weights (0.0 to 1.0, step 0.01, quadratic mapping: 0.1 -> 1% influence, 0.01 -> 0.01%)
                "weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # prompt weight (0.0 to 1.0, step 0.01, controls prompt influence on conditioning)
                "prompt_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                # use as reference latents (if false, latent not added to reference_latents, but image still used for VL)
                "use_ref1": ("BOOLEAN", {"default": True}),
                "use_ref2": ("BOOLEAN", {"default": True}),
                "use_ref3": ("BOOLEAN", {"default": True}),
                # start/end percent for image2 (like ControlNet, controls when latent2 influence applies)
                "start_percent2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("conditioning_full_ref", "latent", "target_image1", "target_image2", "target_image3", "conditioning_with_first_ref", "conditioning_first_weighted", "conditioning_ref2_timing")
    FUNCTION = "encode"

    CATEGORY = "VNCCS/encoding"

    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               control_instruction="",
               weight1=1.0, weight2=1.0, weight3=1.0,
               prompt_weight=1.0,
               use_ref1=True, use_ref2=True, use_ref3=True,
               start_percent2=0.0, end_percent2=1.0,
               ):
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        images = [
            {
                "image": image1,
                "vl_resize": True 
            },
            {
                "image": image2,
                "vl_resize": True 
            },
            {
                "image": image3,
                "vl_resize": True 
            }
        ]
        
        vae_images = []
        vl_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            # for handling mis use of instruction
            if template_prefix in instruction:
                # remove prefix from instruction
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                # remove suffix from instruction
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                # remove {} from instruction
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                vl_resize = image_obj["vl_resize"]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        # pad image to upper size
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / 8.0) * 8
                        canvas_height = math.ceil(samples.shape[2] * scale_by / 8.0) * 8
                        
                        # pad image to canvas size
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": 1 / scale_by
                        }
                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                    
                    if vl_resize:
                        # print("vl_resize")
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            # pad image to upper size
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by)
                            canvas_height = math.ceil(samples.shape[2] * scale_by)
                            
                            # pad image to canvas size
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by)
                            height = round(samples.shape[2] * scale_by)
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_images.append(image)
                    # handle non resize vl images
                    if i == 0:
                        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    else:
                        if control_instruction == "":
                            control_instruction = "This control image defines the composition, structure, and style elements to be applied to the generated image. Determine the type of ControlNet that this image represents. Analyze how this control image influences the overall layout, proportions, and visual style, and incorporate those changes accordingly."
                        image_prompt += "Picture {}: {} <|vision_start|><|image_pad|><|vision_end|>".format(i + 1, control_instruction)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Mix with prompt-only conditioning if prompt_weight < 1.0
        if prompt_weight < 1.0:
            tokens_base = clip.tokenize(prompt, images=[], llama_template=llama_template)
            conditioning_base = clip.encode_from_tokens_scheduled(tokens_base)
            conditioning = [(prompt_weight * img_t + (1 - prompt_weight) * base_t, img_d) for (img_t, img_d), (base_t, _) in zip(conditioning, conditioning_base)]
        
        conditioning_full_ref = conditioning
        conditioning_with_first_ref = conditioning
        if len(ref_latents) > 0:
            # Apply weights to ref_latents
            weights_list = [weight1, weight2, weight3]
            use_refs = [use_ref1, use_ref2, use_ref3]
            ref_latents_weighted = [ (w ** 2) * latent for w, latent in zip(weights_list[:len(ref_latents)], ref_latents) ]
            
            # Filter out zero-weighted or non-reference latents for full_ref
            ref_latents_full = [latent for latent, w, use in zip(ref_latents_weighted, weights_list[:len(ref_latents)], use_refs[:len(ref_latents)]) if w > 0 and use]
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents_full}, append=True)
            
            # Create conditioning_ref2_timing: only reference_latents from image2 with timing
            conditioning_ref2_timing = conditioning_full_ref  # default if not applicable
            if len(ref_latents) > 1 and weights_list[1] > 0 and use_refs[1]:
                conditioning_ref2_timing = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents_weighted[1]], "start_percent": start_percent2, "end_percent": end_percent2})
            
            # First ref: only if weight1 > 0 and use_ref1
            ref_latents_first = [ref_latents_weighted[0]] if weights_list[0] > 0 and use_refs[0] else []
            conditioning_with_first_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents_first}, append=True)
            
            # conditioning_first_weighted: first (if >0 and use) + others (if >0 and use)
            ref_latents_first_weighted = []
            if weights_list[0] > 0 and use_refs[0]:
                ref_latents_first_weighted.append(ref_latents_weighted[0])
            for i in range(1, len(ref_latents_weighted)):
                if weights_list[i] > 0 and use_refs[i]:
                    ref_latents_first_weighted.append(ref_latents_weighted[i])
            conditioning_first_weighted = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents_first_weighted}, append=True)
        else:
            conditioning_first_weighted = conditioning
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 3:
            vae_images.extend([None] * (3 - len(vae_images)))
        o_image1, o_image2, o_image3 = vae_images
        
        conditioning_first_weighted = conditioning_with_first_ref
        
        return (conditioning_full_ref, latent_out, o_image1, o_image2, o_image3, conditioning_with_first_ref, conditioning_first_weighted, conditioning_ref2_timing)


# Registration mapping so Comfy finds the node
NODE_CLASS_MAPPINGS = {
    "VNCCS_QWEN_Encoder": VNCCS_QWEN_Encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_QWEN_Encoder": "VNCCS QWEN Encoder",
}
