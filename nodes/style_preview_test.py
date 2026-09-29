"""Temporary standalone batch preview for the Character Creator V2 style catalog."""

import math
import os
import re

import folder_paths

from .character_creator_v2 import (
    ANIMA_DEFAULTS,
    CHARACTER_STYLE_CATALOG,
    CharacterCreatorV2,
    QI2_DEFAULTS,
    _character_style_prompt,
    create_generation_latent,
    decode_generation_samples,
    encode_generation_conditioning,
    load_generation_assets,
    load_generation_clip,
    normalize_gen_settings,
    prepare_qi2_model,
    sample_generation_latent,
    tensor2pil,
    validate_anima_conditioning,
)


_PROMPT_DEFAULTS = {
    "sex": "female",
    "age": 18,
    "framing": "cowboy_shot",
    "race": "human",
    "skin_color": "",
    "hair": "black hair, long hair",
    "eyes": "",
    "face": "",
    "body": "",
    "additional_details": "",
    "nsfw": False,
    "lora_prompt": "",
}
_MODE_PROMPT_DEFAULTS = {
    "anima": {
        "aesthetics": "masterpiece, best quality, score_7",
        "negative_prompt": "bad quality, worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    },
    "qi2": {
        "aesthetics": "",
        "negative_prompt": "bad quality, worst quality, low quality, blurry, jpeg artifacts",
    },
}
_MODEL_PREFERENCES = {
    "anima": "anima-base-v1.0.safetensors",
    "qi2": QI2_DEFAULTS["diffusion_model_name"],
}
_SUBFOLDER = "VNCCS/style_previews"


def _select_diffusion_model(mode):
    available = folder_paths.get_filename_list("diffusion_models")
    preferred = _MODEL_PREFERENCES[mode]
    for name in available:
        if name.replace("\\", "/").casefold().endswith(preferred.casefold()):
            return name
    markers = ("anima",) if mode == "anima" else ("qwen_image_2.1", "qwen-image-2.1", "qi2")
    for name in available:
        if any(marker in name.casefold() for marker in markers) and "turbo" not in name.casefold():
            return name
    raise ValueError(f"No {mode.upper()} diffusion model found in ComfyUI diffusion_models")


def _square_side(scale):
    scale = max(1.0, min(4.0, float(scale)))
    stepped = round(scale, 1)
    target_size = {1.3: 1344, 1.5: 1536}.get(stepped, round(stepped * 1024))
    return max(8, round(math.sqrt(target_size * 1024) / 8) * 8)


def _next_output_path(output_dir, mode, style_id):
    if not re.fullmatch(r"[a-z0-9_]+", style_id):
        raise ValueError(f"Invalid style ID for output filename: {style_id!r}")
    prefix = f"{mode}_{style_id}"
    counter = 1
    while True:
        filename = f"{prefix}_{counter:04d}.png"
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return path, filename
        counter += 1


class VNCCSStylePreviewTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["Anima", "QI2"],),
            "scale_mp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            "background_color": (["Green", "Blue", "Transparent"],),
        }}

    RETURN_TYPES = ()
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "VNCCS/Testing"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def generate(self, model, scale_mp, background_color):
        mode = model.lower()
        if mode not in _MODE_PROMPT_DEFAULTS:
            raise ValueError(f"Unsupported generation model: {model}")
        if background_color not in {"Green", "Blue", "Transparent"}:
            raise ValueError(f"Unsupported background color: {background_color}")
        if background_color == "Transparent" and mode != "qi2":
            raise ValueError("Transparent background is available only with QI2")

        side = _square_side(scale_mp)
        settings = normalize_gen_settings({
            **(ANIMA_DEFAULTS if mode == "anima" else QI2_DEFAULTS),
            "generation_mode": mode,
            "diffusion_model_name": _select_diffusion_model(mode),
            "turbo_enabled": False,
            "lora_stack": [],
        })
        _, diffusion_model, clip, vae = load_generation_assets(settings)
        if mode == "qi2":
            diffusion_model, turbo = prepare_qi2_model(diffusion_model, settings)
            if turbo:
                raise RuntimeError("QI2 turbo was enabled unexpectedly")

        output_dir = os.path.join(folder_paths.get_output_directory(), _SUBFOLDER)
        os.makedirs(output_dir, exist_ok=True)
        images = []
        for group in CHARACTER_STYLE_CATALOG["groups"]:
            for style in group["styles"]:
                info = {
                    **_PROMPT_DEFAULTS,
                    **_MODE_PROMPT_DEFAULTS[mode],
                    "style": style["id"],
                    "background_color": background_color,
                }
                positive_text, negative_text = CharacterCreatorV2.construct_prompt(
                    info, mode, include_style=mode != "qi2",
                )
                if mode == "qi2" and images:
                    clip = load_generation_clip(settings)
                positive, negative, _ = encode_generation_conditioning(
                    clip, vae, positive_text, negative_text, settings,
                    style_reference=_character_style_prompt(info),
                    character_info=info,
                )
                if mode == "anima":
                    validate_anima_conditioning(positive, negative, settings["clip_name"])
                latent = create_generation_latent(diffusion_model, side, side, settings)
                sampled = sample_generation_latent(
                    model=diffusion_model,
                    positive=positive,
                    negative=negative,
                    latent=latent,
                    seed=0,
                    steps=settings["steps"],
                    cfg=settings["cfg"],
                    sampler_name=settings["sampler"],
                    scheduler=settings["scheduler"],
                    gen_settings=settings,
                    qi2_turbo=False,
                )
                image = decode_generation_samples(vae, sampled, settings)
                if background_color == "Transparent" and image.shape[-1] != 4:
                    raise RuntimeError("QI2 returned no alpha channel for transparent background")
                path, filename = _next_output_path(output_dir, mode, style["id"])
                tensor2pil(image).save(path)
                images.append({"filename": filename, "subfolder": _SUBFOLDER, "type": "output"})
                print(f"[VNCCS Style Preview Test] Saved {path}")

        return {"ui": {"images": images}, "result": ()}


NODE_CLASS_MAPPINGS = {"VNCCSStylePreviewTest": VNCCSStylePreviewTest}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCSStylePreviewTest": "VNCCS Style Preview Test"}
