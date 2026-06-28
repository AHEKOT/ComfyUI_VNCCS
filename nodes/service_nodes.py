import json
import os
import re

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI runtime import
    folder_paths = None

from .character_generator import VNCCS_EmotionsGenerator
from .emotion_generator_v2 import build_anima_emotion_prompt, build_emotion_pipe, get_custom_node_path, load_emotions_data


SERVICE_ANIMA_SETTINGS = {
    "generation_mode": "anima",
    "ckpt_name": "Illustrious\\ILFlatMix.safetensors",
    "diffusion_model_name": "anima-base-v1.0.safetensors",
    "clip_name": "qwen_3_06b_base.safetensors",
    "vae_name": "qwen_image_vae.safetensors",
    "clip_type": "stable_diffusion",
    "sampler": "er_sde",
    "scheduler": "simple",
    "steps": 12,
    "cfg": 1,
    "seed": 0,
    "seed_mode": "fixed",
    "turbo_enabled": True,
    "turbo_previous_settings": {"steps": 30, "cfg": 4},
    "dmd_lora_name": "Anima/anima-turbo-lora-v0.1.safetensors",
    "dmd_lora_strength": 1,
    "lora_stack": [],
}


def _service_anima_generation_settings():
    settings = dict(SERVICE_ANIMA_SETTINGS)
    settings["mode_settings"] = {
        "anima": dict(SERVICE_ANIMA_SETTINGS),
    }
    return settings


def _safe_emotion_filename(safe_name):
    name = str(safe_name or "").strip()
    if not name:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid emotion safe_name: {name!r}")
    return f"{name}.png"


def _flatten_emotions_config():
    data = load_emotions_data()
    items = []
    seen = set()
    for _category, emotions in data.items():
        for emotion in emotions or []:
            if not isinstance(emotion, dict):
                continue
            safe_name = str(emotion.get("safe_name") or "").strip()
            if not safe_name:
                continue
            if safe_name in seen:
                raise ValueError(f"Duplicate emotion safe_name in emotions.json: {safe_name}")
            seen.add(safe_name)
            prompt = str(emotion.get("natural_prompt") or "").strip()
            if not prompt:
                prompt = f"The character expresses {safe_name.replace('-', ' ')}."
            items.append({
                "safe_name": safe_name,
                "prompt": prompt,
                "description": str(emotion.get("description") or "").strip(),
            })
    if not items:
        raise RuntimeError("No emotions with safe_name found in emotions-config/emotions.json")
    return items


def _tensor_to_rgb_png(image, path):
    tensor = image.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Unsupported generated image shape: {tuple(tensor.shape)}")
    arr = (tensor.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr[..., :3], mode="RGB").save(path, format="PNG")


def _service_emotions_output_dir():
    if folder_paths is not None:
        try:
            return os.path.join(folder_paths.get_output_directory(), "VNCCS", "ServiceEmotions")
        except Exception:
            pass
    return os.path.join(get_custom_node_path(), "output", "VNCCS", "ServiceEmotions")


def _output_image_ui_item(path):
    if folder_paths is None:
        return {
            "filename": os.path.basename(path),
            "subfolder": "",
            "type": "output",
        }
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    abs_path = os.path.abspath(path)
    rel = os.path.relpath(abs_path, output_dir)
    if rel.startswith(".."):
        subfolder = ""
    else:
        subfolder = os.path.dirname(rel).replace(os.sep, "/")
    return {
        "filename": os.path.basename(path),
        "subfolder": subfolder,
        "type": "output",
    }


class VNCCS_Service_Emotions_Generator(VNCCS_EmotionsGenerator):
    OUTPUT_NODE = True
    INPUT_IS_LIST = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("IMAGE",),
                "character_prompt": ("STRING", {"default": "", "multiline": True}),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_dir", "count")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/service"
    DESCRIPTION = "Generate one reference image for every emotion in emotions-config/emotions.json."

    def _default_pipe(self):
        pipe, _seed = build_emotion_pipe("Anima", json.dumps(_service_anima_generation_settings()))
        return pipe

    def _prompt_for_emotion(self, item, character_prompt=""):
        prompt = build_anima_emotion_prompt(item.get("prompt", ""), item.get("description", ""), item.get("safe_name", ""))
        character_prompt = str(character_prompt or "").strip()
        if character_prompt:
            return f"{prompt} Character details: {character_prompt}"
        return prompt

    def generate(self, character, character_prompt="", denoise=0.55):
        source_items = self._image_list(character)
        if not source_items:
            raise ValueError("VNCCS Service Emotions Generator requires a character IMAGE input.")
        source = self._safe_image_batch(source_items[0], stage="service emotions source")
        if not torch.is_tensor(source) or source.ndim != 4:
            raise ValueError("VNCCS Service Emotions Generator received an invalid character IMAGE.")

        pipe = self._default_pipe()

        output_dir = _service_emotions_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        emotion_items = _flatten_emotions_config()
        face_denoise = max(0.0, min(1.0, float(denoise)))
        positive_prompt = (
            "masterpiece, best quality, anime style, close-up portrait, same character, "
            "preserve character identity, preserve hairstyle, preserve outfit details"
        )
        character_prompt = str(character_prompt or "").strip()
        if character_prompt:
            positive_prompt = f"{positive_prompt}, Character details: {character_prompt}"
        negative_prompt = (
            "bad quality, worst quality, low quality, blurry, jpeg artifacts, deformed face, "
            "distorted eyes, extra eyes, extra mouth, changed character identity"
        )

        saved = []
        ui_images = []
        base_seed = int(self._extract_pipe(pipe).get("seed", 0) or 0)
        for index, item in enumerate(emotion_items):
            filename = _safe_emotion_filename(item["safe_name"])
            out_path = os.path.join(output_dir, filename)
            _full, face_crop = self._run_emotion_generation_one(
                source,
                None,
                pipe,
                self._prompt_for_emotion(item, character_prompt),
                character_prompt,
                negative_prompt,
                base_seed + index,
                face_denoise,
            )
            face_crop = self._safe_image_batch(face_crop, stage="service emotion cropped result")
            _tensor_to_rgb_png(face_crop, out_path)
            saved.append(out_path)
            ui_images.append(_output_image_ui_item(out_path))
            print(f"[VNCCS Service Emotions Generator] Saved {out_path}")

        if not saved:
            raise RuntimeError("No service emotion images were generated.")
        return {
            "ui": {
                "images": ui_images,
                "output_dir": [output_dir],
                "count": [len(saved)],
            },
            "result": (output_dir, len(saved)),
        }


NODE_CLASS_MAPPINGS = {
    "VNCCS_Service_Emotions_Generator": VNCCS_Service_Emotions_Generator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Service_Emotions_Generator": "VNCCS Service Emotions Generator",
}
