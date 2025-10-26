"""VNCCS QWEN Encoder node

This node is a simplified variant of the QWEN image-to-conditioning encoder
that accepts exactly three images and three per-image weights (0.0-1.0, step 0.01, quadratic mapping for fine control)
to control each image's influence via weighted reference latents. Also includes use_ref flags to exclude latents from reference_latents while keeping VL influence.

Class: VNCCS_QWEN_Encoder
- INPUTS: clip, prompt, vae, image1/2/3, weight1..weight3, vl_size, latent_image_index, resize/control flags
- OUTPUTS: positive, negative, latent

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
                "vae": ("VAE", ),
            },
            "optional": 
            {
                "latent_image_index": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "image1_name": ("STRING", {"default": "Picture 1"}),
                "image2_name": ("STRING", {"default": "Picture 2"}),
                "image3_name": ("STRING", {"default": "Picture 3"}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "weight3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "vl_size": ("INT", {"default": 384, "min": 256, "max": 1024, "step": 8}),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "VNCCS/encoding"

    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None,
               target_size=1024, 
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               image1_name="Picture 1", image2_name="Picture 2", image3_name="Picture 3",
               weight1=1.0, weight2=1.0, weight3=1.0,
               vl_size=384,
               latent_image_index=1,
               ):
        
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
        names = [image1_name, image2_name, image3_name]

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
                    
                    if vl_resize:
                        # print("vl_resize")
                        total = int(vl_size * vl_size)
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
                    image_prompt += "{}: <|vision_start|><|image_pad|><|vision_end|>".format(names[i])
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        conditioning_full_ref = conditioning
        if len(ref_latents) > 0:
            # Apply weights to ref_latents
            weights_list = [weight1, weight2, weight3]
            ref_latents_weighted = [ (w ** 2) * latent for w, latent in zip(weights_list[:len(ref_latents)], ref_latents) ]
            
            # Filter out zero-weighted latents for full_ref
            ref_latents_full = [latent for latent, w in zip(ref_latents_weighted, weights_list[:len(ref_latents)]) if w > 0]
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents_full}, append=True)
        
        # Create negative conditioning by zeroing out the positive conditioning tensors
        conditioning_negative = [(torch.zeros_like(cond[0]), cond[1]) for cond in conditioning_full_ref]
        
        # Return latent of selected image if available, otherwise return empty latent
        if len(ref_latents) >= latent_image_index:
            samples = ref_latents[latent_image_index - 1]
        else:
            samples = torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        
        return (conditioning_full_ref, conditioning_negative, latent_out)


# Registration mapping so Comfy finds the node
NODE_CLASS_MAPPINGS = {
    "VNCCS_QWEN_Encoder": VNCCS_QWEN_Encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_QWEN_Encoder": "VNCCS QWEN Encoder",
}
