import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch
import cv2
import copy
import io
import os
import json
import torch
import base64
import random
import requests
from typing import List, Dict, Tuple

from PIL import Image, ImageOps, ImageFilter
import numpy as np

import folder_paths


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0
def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)

def prepare_image_for_preview(image: Image.Image, output_dir: str, prefix=None):
    if prefix is None:
        prefix = "preview_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    # save image to temp folder
    (
        outdir,
        filename,
        counter,
        subfolder,
        _,
    ) = folder_paths.get_save_image_path(prefix, output_dir, image.width, image.height)
    file = f"{filename}_{counter:05}_.png"
    image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)

def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)

def load_images_from_url(urls: List[str], keep_alpha_channel=False):
    images = []
    masks = []

    for url in urls:
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        elif url.startswith("file://"):
            url = url[7:]
            if not os.path.isfile(url):
                raise Exception(f"File {url} does not exist")

            i = Image.open(url)
        elif url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))
        elif url.startswith(("/view?", "/api/view?")):
            from urllib.parse import parse_qs

            qs_idx = url.find("?")
            qs = parse_qs(url[qs_idx + 1 :])
            filename = qs.get("name", qs.get("filename", None))
            if filename is None:
                raise Exception(f"Invalid url: {url}")

            filename = filename[0]
            subfolder = qs.get("subfolder", None)
            if subfolder is not None:
                filename = os.path.join(subfolder[0], filename)

            dirtype = qs.get("type", ["input"])
            if dirtype[0] == "input":
                url = os.path.join(folder_paths.get_input_directory(), filename)
            elif dirtype[0] == "output":
                url = os.path.join(folder_paths.get_output_directory(), filename)
            elif dirtype[0] == "temp":
                url = os.path.join(folder_paths.get_temp_directory(), filename)
            else:
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)
        elif url == "":
            continue
        else:
            url = folder_paths.get_annotated_filepath(url)
            if not os.path.isfile(url):
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)

        i = ImageOps.exif_transpose(i)
        has_alpha = "A" in i.getbands()
        mask = None

        if "RGB" not in i.mode:
            i = i.convert("RGBA") if has_alpha else i.convert("RGB")

        if has_alpha:
            mask = i.getchannel("A")

            # recreate image to fix weird RGB image
            alpha = i.split()[-1]
            image = Image.new("RGB", i.size, (0, 0, 0))
            image.paste(i, mask=alpha)
            image.putalpha(alpha)

            if not keep_alpha_channel:
                image = image.convert("RGB")
        else:
            image = i

        images.append(image)
        masks.append(mask)

    return (images, masks)

# http://127.0.0.1:8188/api/view?filename=1.png&subfolder=&type=output&rand=0.04266252963576478
class LoadImageFromUrl_lrzjason:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", ),
            },
            "optional": {
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "has_image")
    CATEGORY = "Art Venture/Image"
    FUNCTION = "load_image"

    def load_image(self, filename, seed, keep_alpha_channel=False, output_mode=False):
        image = f"/view?filename={filename}.png&subfolder=&type=output&rand={seed}"
        
        urls = image.strip().split("\n")
        images, masks = load_images_from_url(urls, keep_alpha_channel)
        if len(images) == 0:
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images = [tensor2pil(image)]
            masks = [tensor2pil(mask, mode="L")]

        previews = []
        np_images = []
        np_masks = []

        for image, mask in zip(images, masks):
            # save image to temp folder
            preview = prepare_image_for_preview(image, self.output_dir, self.filename_prefix)
            image = pil2tensor(image)

            if mask:
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            previews.append(preview)
            np_images.append(image)
            np_masks.append(mask.unsqueeze(0))

        if output_mode:
            result = (np_images, np_masks, True)
        else:
            has_size_mismatch = False
            if len(np_images) > 1:
                for image in np_images[1:]:
                    if image.shape[1] != np_images[0].shape[1] or image.shape[2] != np_images[0].shape[2]:
                        has_size_mismatch = True
                        break

            if has_size_mismatch:
                raise Exception("To output as batch, images must have the same size. Use list output mode instead.")

            result = ([torch.cat(np_images)], [torch.cat(np_masks)], True)

        return {"ui": {"images": previews}, "result": result}

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
                width = round(samples.shape[3] * scale_by / 64.0) * 64
                height = round(samples.shape[2] * scale_by / 64.0) * 64
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
    crop_methods = ["center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [1024,512,392,384]
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
                "target_vl_size": (s.target_vl_sizes, {"default": 392}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("conditioning", "latent", "target_image1", "target_image2", "target_image3", "vl_resized_image1", "vl_resized_image2", "vl_resized_image3")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, 
               vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None,
               not_resize_image1=None, not_resize_image2=None, not_resize_image3=None, 
               target_size=1024, 
               target_vl_size=392,
               upscale_method="lanczos",
               crop="center",
               instruction="",
               ):
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
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                    
                    if vl_resize:
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        width = round(samples.shape[3] * scale_by / 28) *28
                        height = round(samples.shape[2] * scale_by / 28) *28
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)
                    # handle non resize vl images
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 3:
            vae_images.extend([None] * (3 - len(vae_images)))
        o_image1, o_image2, o_image3 = vae_images
        
        if len(vl_resized_images) < 3:
            vl_resized_images.extend([None] * (3 - len(vl_resized_images)))
        vl_image1, vl_image2, vl_image3 = vl_resized_images
        
        return (conditioning, latent_out, o_image1, o_image2, o_image3, vl_image1, vl_image2, vl_image3)


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": TextEncodeQwenImageEditPlus_lrzjason,
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": TextEncodeQwenImageEditPlusAdvance_lrzjason,
    "LoadImageFromUrl_lrzjason": LoadImageFromUrl_lrzjason
    
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": "TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)",
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": "TextEncodeQwenImageEditPlusAdvance 小志Jason(xiaozhijason)",
    "LoadImageFromUrl_lrzjason": "LoadImageFromUrl 小志Jason(xiaozhijason)"
}