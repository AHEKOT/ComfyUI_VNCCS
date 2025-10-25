import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
from comfy.utils import common_upscale
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import folder_paths

class VNCCS_QWEN_Encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "vl_size": ("INT", {"default": 448, "min": 224, "max": 1024, "step": 32}),
                "prompt_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "image3": ("IMAGE",),
                "weight3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "image4": ("IMAGE",),
                "weight4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "VNCCS"

    def encode(self, images, weights, vl_size, prompt_weight, image2=None, weight2=1.0, image3=None, weight3=1.0, image4=None, weight4=1.0):
        # Load QWEN model
        model_path = folder_paths.get_folder_paths("diffusers")[0] + "/Qwen/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)

        device = mm.get_torch_device()
        model.to(device)

        # Prepare images list
        image_list = [images]
        weight_list = [weights]
        if image2 is not None:
            image_list.append(image2)
            weight_list.append(weight2)
        if image3 is not None:
            image_list.append(image3)
            weight_list.append(weight3)
        if image4 is not None:
            image_list.append(image4)
            weight_list.append(weight4)

        # Process each image
        conditionings = []
        for img, w in zip(image_list, weight_list):
            # Resize image
            img = common_upscale(img, vl_size, vl_size, "bilinear", "center")
            img = img.squeeze(0).numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)

            # Encode with QWEN
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[pil_img], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                # Get image features (assuming last tokens are image)
                image_features = hidden_states[:, -inputs["image_grid_thw"].shape[1]:, :]

            # Create conditioning
            cond = image_features * w
            conditionings.append(cond)

        # Mix conditionings based on prompt_weight
        if prompt_weight < 1.0:
            # Mix with original prompt (assuming prompt is separate, but for now, just average)
            mixed = torch.mean(torch.stack(conditionings), dim=0)
        elif prompt_weight > 1.0:
            # Boost the conditionings
            mixed = torch.mean(torch.stack(conditionings), dim=0) * prompt_weight
        else:
            mixed = torch.mean(torch.stack(conditionings), dim=0)

        return (mixed,)

NODE_CLASS_MAPPINGS = {
    "VNCCS_QWEN_Encoder": VNCCS_QWEN_Encoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_QWEN_Encoder": "VNCCS QWEN Encoder"
}
