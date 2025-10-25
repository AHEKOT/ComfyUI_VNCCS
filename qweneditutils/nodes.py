import node_helpers
import comfy.utils
import math
import torch
import torch
import numpy as np
from PIL import Image   


class CropWithPadInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pad_info": ("ANY", ),  # pad_info dictionary containing x, y, width, height and scale
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("cropped_image", "scale_by", )
    FUNCTION = "crop_image"

    CATEGORY = "image"

    def crop_image(self, image, pad_info):
        # Extract pad information from the original padding process:
        # In the original function:
        # - resized_samples are placed at canvas[:, :, :resized_height, :resized_width]
        # - pad_info = {"x": 0, "y": 0, "width": canvas_width - resized_width, "height": canvas_height - resized_height}
        x = pad_info.get("x", 0)  # This is always 0 in the original function
        y = pad_info.get("y", 0)  # This is always 0 in the original function
        width_padding = pad_info.get("width", 0)  # Right/bottom padding added
        height_padding = pad_info.get("height", 0)  # Right/bottom padding added
        scale_by = pad_info.get("scale_by", 1.0)
        
        img = image.movedim(-1, 1)  # Convert from (H, W, C) to (C, H, W)
        
        # Calculate the original content dimensions before padding was added
        original_content_width = img.shape[3] - width_padding
        original_content_height = img.shape[2] - height_padding
        
        # Crop to get just the original content area (which was placed at position (0,0))
        cropped_img = img[:, :, x:original_content_height, y:original_content_width]
        
        # Convert back to (H, W, C) format
        cropped_image = cropped_img.movedim(1, -1)
        
        return (cropped_image, scale_by)


def get_nearest_resolution(image, resolution=1024):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    # Calculate target dimensions that:
    # 1. Maintain the aspect ratio
    # 2. Have an area of approximately resolution^2 (1024*1024 = 1048576)
    # 3. Are divisible by 8
    target_area = resolution * resolution
    
    # width = height * image_ratio
    # width * height = target_area
    # height * image_ratio * height = target_area
    # height^2 = target_area / image_ratio
    height_optimal = math.sqrt(target_area / image_ratio)
    width_optimal = height_optimal * image_ratio
    
    # Round to nearest multiples of 8
    height_8 = round(height_optimal / 8) * 8
    width_8 = round(width_optimal / 8) * 8
    
    # Ensure minimum size of 64x64
    height_8 = max(64, height_8)
    width_8 = max(64, width_8)
    
    closest_resolution = (width_8, height_8)
    closest_ratio = width_8 / height_8

    return closest_ratio, closest_resolution


def crop_image(image,resolution):
    height, width, _ = image.shape
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    image_ratio = width / height
    
    # Determine which dimension to scale by to minimize cropping
    scale_with_height = True
    if image_ratio < closest_ratio: 
        scale_with_height = False
    
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    return image

def convert_float_unit8(image):
    image = image.astype(np.float32) * 255
    return image.astype(np.uint8)

def convert_unit8_float(image):
    image = image.astype(np.float32)
    image = image / 255.
    return image
def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",height,width)
    if scale_with_height: 
        # Scale based on height, then crop width if needed
        up_scale = height / closest_resolution[1]
    else:
        # Scale based on width, then crop height if needed
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = expanded_closest_size[0] - width
    diff_y = expanded_closest_size[1] - height

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x > 0:
        # Need to crop width (image is wider than needed)
        crop_x = diff_x // 2
        cropped_image = image[:, crop_x:width - diff_x + crop_x, :]
    elif diff_y > 0:
        # Need to crop height (image is taller than needed)
        crop_y = diff_y // 2
        cropped_image = image[crop_y:height - diff_y + crop_y, :, :]
    else:
        # No cropping needed
        cropped_image = image

    height, width, _ = cropped_image.shape  
    f_width, f_height = closest_resolution
    cropped_image = convert_float_unit8(cropped_image)
    # print("cropped_image:",cropped_image)
    img_pil = Image.fromarray(cropped_image)
    resized_img = img_pil.resize((f_width, f_height), Image.LANCZOS)
    resized_img = np.array(resized_img)
    resized_img = convert_unit8_float(resized_img)
    return resized_img, crop_x, crop_y


class TextEncodeQwenImageEdit_lrzjason:
    @classmethod
    def INPUT_TYPES(s):
        resolution_choices = [
            2048, 1536, 1328, 1024, 768, 512
        ]
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image": ("IMAGE", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "resolution": (resolution_choices, {
                    "default": 1024,
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT", )
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None, enable_resize=True, resolution=1024):
        ref_latent = None
        if image is None:
            images = []
        else:
            # bs, h, w, c
            # ([1, 1248, 832, 3])
            if enable_resize:
                samples = image.squeeze(0).numpy()
                cropped_image = crop_image(samples,resolution)
                cropped_image = torch.from_numpy(cropped_image).unsqueeze(0)
                image = cropped_image
                
            images = [image]
            if vae is not None:
                ref_latent = vae.encode(image)
                # print("ref_latent:",ref_latent.shape)
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]})
            
        return (conditioning, image, {"samples":ref_latent}, )


class TextEncodeQwenImageEditPlus_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["disabled", "center"]
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
                "image4": ("IMAGE", ),
                "image5": ("IMAGE", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "enable_vl_resize": ("BOOLEAN", {"default": True}),
                "skip_first_image_resize": ("BOOLEAN", {"default": False}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "image1", "image2", "image3", "image4", "image5", "latent")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None, image4=None, image5=None, 
               enable_resize=True, enable_vl_resize=True, skip_first_image_resize=False,
               upscale_method="bicubic",
               crop="center",
               instruction=""
               ):
        ref_latents = []
        images = [image1, image2, image3, image4, image5]
        images_vl = []
        vae_images = []
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

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = (samples.shape[3] * samples.shape[2])
                total = int(1024 * 1024)
                scale_by = 1  # Default scale
                if enable_resize:
                    scale_by = math.sqrt(total / current_total)
                width = round(samples.shape[3] * scale_by / 8.0) * 8
                height = round(samples.shape[2] * scale_by / 8.0) * 8
                if vae is not None:
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                # print("before enable_vl_resize scale_by", scale_by)
                # print("before enable_vl_resize width,height", width,height)
                if enable_vl_resize and not skip_first_image_resize and i == 0:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / current_total)
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                # print("after enable_vl_resize width,height", width,height)
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                image = s.movedim(1, -1)
                images_vl.append(image)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 5:
            vae_images.extend([None] * (5 - len(vae_images)))
        o_image1, o_image2, o_image3, o_image4, o_image5 = vae_images
        return (conditioning, o_image1, o_image2, o_image3, o_image4, o_image5, latent_out)



class TextEncodeQwenImageEditPlusAdvance_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
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
                "vl_resize_image1": ("IMAGE", ),
                "vl_resize_image2": ("IMAGE", ),
                "vl_resize_image3": ("IMAGE", ),
                "not_resize_image1": ("IMAGE", ),
                "not_resize_image2": ("IMAGE", ),
                "not_resize_image3": ("IMAGE", ),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "target_image1", "target_image2", "target_image3", "vl_resized_image1", "vl_resized_image2", "vl_resized_image3", "conditioning_with_first_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, 
               vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None,
               not_resize_image1=None, not_resize_image2=None, not_resize_image3=None, 
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               ):
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        images = [not_resize_image1, not_resize_image2, not_resize_image3, 
                  vl_resize_image1, vl_resize_image2, vl_resize_image3]
        vl_resized_images = []
        
        images = [
            {
                "image": not_resize_image1,
                "vl_resize": False 
            },
            {
                "image": not_resize_image2,
                "vl_resize": False 
            },
            {
                "image": not_resize_image3,
                "vl_resize": False 
            },
            {
                "image": vl_resize_image1,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image2,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image3,
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
                        vl_resized_images.append(image)
                    # handle non resize vl images
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_first_ref = conditioning
        if len(ref_latents) > 0:
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            
            conditioning_with_first_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[0]]}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 3:
            vae_images.extend([None] * (3 - len(vae_images)))
        o_image1, o_image2, o_image3 = vae_images
        
        if len(vl_resized_images) < 3:
            vl_resized_images.extend([None] * (3 - len(vl_resized_images)))
        vl_image1, vl_image2, vl_image3 = vl_resized_images
        
        return (conditioning_full_ref, latent_out, o_image1, o_image2, o_image3, vl_image1, vl_image2, vl_image3, conditioning_with_first_ref, pad_info)

def validate_vl_resize_indexs(vl_resize_indexs_str, valid_length):
    try:
        indexes = [int(i)-1 for i in vl_resize_indexs_str.split(",")]
        # remove duplicates
        indexes = list(set(indexes))
    except ValueError as e:
        raise ValueError(f"Invalid format for vl_resize_indexs: {e}")

    if not indexes:
        raise ValueError("vl_resize_indexs must not be empty")

    indexes = [idx for idx in indexes if 0 <= idx < valid_length]

    return indexes

class TextEncodeQwenImageEditPlusPro_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
    vl_resize_indexs = [1,2,3]
    main_image_index = 1
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
                "image4": ("IMAGE", ),
                "image5": ("IMAGE", ),
                "vl_resize_indexs": ("STRING", {"default": "1,2,3"}),
                "main_image_index": ("INT", {"default": 1, "max": 5, "min": 1}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "image1", "image2", "image3", "image4", "image5", "conditioning_with_main_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None,
               image4=None, image5=None, 
               vl_resize_indexs="1,2,3",
               main_image_index=1,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               ):
        # check vl_resize_indexs is valid indexes and not out of range
        resize_indexs = validate_vl_resize_indexs(vl_resize_indexs,5)
        
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        temp = [image1, image2, image3, image4, image5]
        images = []
        for i, image in enumerate(temp):
            image_dict = {
                "image": image,
                "vl_resize": False
            }
            if i in resize_indexs:
                image_dict['vl_resize'] = True
            images.append(image_dict)
        vl_resized_images = []
        
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
                        vl_resized_images.append(image)
                    # handle non resize vl images
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_main_ref = conditioning
        samples = torch.zeros(1, 4, 128, 128)
        
        if len(ref_latents) > 0:
            # remap main_image_index from start from 1 to 0
            main_image_index = main_image_index - 1
            if main_image_index >= len(ref_latents):
                print("\n Auto fixing main_image_index to the first image index")
                main_image_index = 0
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            conditioning_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            # Return latent of first image if available, otherwise return empty latent
            # samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
            samples = ref_latents[main_image_index]
        latent_out = {"samples": samples}
        if len(vae_images) < len(images):
            vae_images.extend([None] * (len(images) - len(vae_images)))
        image1, image2, image3, image4, image5 = vae_images
        
        return (conditioning_full_ref, latent_out, image1, image2, image3, image4, image5, conditioning_with_main_ref, pad_info)

NODE_CLASS_MAPPINGS = {
    "CropWithPadInfo": CropWithPadInfo,
    "TextEncodeQwenImageEdit_lrzjason": TextEncodeQwenImageEdit_lrzjason,
    "TextEncodeQwenImageEditPlus_lrzjason": TextEncodeQwenImageEditPlus_lrzjason,
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": TextEncodeQwenImageEditPlusAdvance_lrzjason,
    "TextEncodeQwenImageEditPlusPro_lrzjason": TextEncodeQwenImageEditPlusPro_lrzjason,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropWithPadInfo": "Crop With Pad Info",
    "TextEncodeQwenImageEdit_lrzjason": "TextEncodeQwenImageEdit 小志Jason(xiaozhijason)",
    "TextEncodeQwenImageEditPlus_lrzjason": "TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)",
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": "TextEncodeQwenImageEditPlusAdvance 小志Jason(xiaozhijason)",
    "TextEncodeQwenImageEditPlusPro_lrzjason": "TextEncodeQwenImageEditPlusPro 小志Jason(xiaozhijason)",
}