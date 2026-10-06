"""Standalone randomized single-image preview for Character Creator V2 styles."""
from ..operation_logger import log_event, logged_operation

import json
import math
import os
import random
import re

import comfy.sd
import comfy.utils
import folder_paths
import torch

from .character_creator_v2 import (
    ANIMA_DEFAULTS,
    CHARACTER_STYLE_CATALOG,
    CHARACTER_PRESETS,
    CharacterCreatorV2,
    QI2_DEFAULTS,
    QI2_TEXT_GENERATION_DEFAULTS,
    QI2_STYLE_REFERENCE_HEADING,
    QI2_STYLE_REWRITE_RULES,
    _character_style_prompt,
    _ensure_prompt_server_progress_context,
    _qi2_character_fields,
    _qi2_expanded_field_prompt,
    _qi2_json_result,
    _qi2_prompt_rewriter_system_prompt,
    _strip_unit_prompt_weights,
    create_generation_latent,
    decode_generation_samples,
    encode_generation_conditioning,
    get_lora_full_path,
    load_generation_assets,
    load_generation_clip,
    normalize_gen_settings,
    prepare_qi2_model,
    sample_generation_latent,
    tensor2pil,
    validate_anima_conditioning,
)
from .runtime_cleanup import inference_stage


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
_FIELD_TAG_GROUPS = {
    "race": ("races",),
    "skin_color": ("skin_color",),
    "hair": ("hair_color", "hair_pattern", "hair_length", "hair_texture", "hairstyles", "hair_framing"),
    "eyes": ("eye_color", "eye_features"),
    "face": ("face_shape", "face_details"),
    "body": ("body_type", "breast_size"),
    "additional_details": ("details",),
}
_RANDOM_FIELDS = ("sex", "age", "framing", *_FIELD_TAG_GROUPS, "style")
_BREAST_SIZE_ANALYSIS_PROMPT = """Classify the visible breast size of this adult fictional character.
Base the assessment only on breast volume and outward projection relative to
the character's ribcage and upper torso. These cup labels are an approximate
visual scale for character art, not physical bra measurements.

Use this scale consistently:
Flat Chest: absent or almost absent breast projection.
A-Cup: very slight volume and projection.
B-Cup: small volume, clearly present but modest relative to the ribcage.
C-Cup: moderately small volume and projection.
D-Cup: medium volume, balanced with the ribcage and upper torso.
E-Cup: moderately full volume, clearly above medium.
F-Cup: large volume and substantial projection.
G-Cup: very large volume, prominent relative to the upper torso.
H-Cup: exceptionally large volume that dominates the upper torso.
I-Cup: strongly exaggerated volume, beyond an exceptionally large silhouette.
J-Cup: the extreme oversized end of the scale, with unmistakably disproportionate
volume dominating the entire upper torso. Ordinary large breasts do not meet this criterion.

Do not infer greater volume from cleavage, a low neckline, exposed skin,
bra padding, highlights, wide hips, a narrow waist or camera foreshortening.
Do not invent hidden volume. If clothing or framing prevents a reliable
assessment, return Cannot determine instead of guessing a cup label.

Return exactly one category: Flat Chest, A-Cup, B-Cup, C-Cup, D-Cup, E-Cup,
F-Cup, G-Cup, H-Cup, I-Cup or J-Cup. If the image cannot be assessed, return
only Cannot determine. Do not include explanations, descriptions, comparisons,
other categories, prefixes, JSON or additional text.
Do not force an extreme category when the image supports a middle category."""
_ANIMA_ARTISTS = tuple((
    "~am314|amakawa tamawo|afukuro|akizero1510|agoto|aka shiba|alzi xiaomi|anmi|asteroid ill|asagina gi|"
    "ask (askzy)|aspara|atdan|bai qi-qsr|bilibili xiaolu|BLACK MATERIAL|bukurote|bysau|c home|cancer (zjcconan)|"
    "charlie herve|chen bin|chyan|chyoel|ciloranko|cogecha|dm (dai miao)|mabo9317|demizu posuka|"
    "dino (dinoartforame)|dishwasher1910|dsmile|duan henglong|elina (e2n04n)|esuthio|fjsmu|fkey|"
    "free style (yohan1754)|fruitsrabbit|fuzichoco|grandia lee|guweiz|hito komoru|hxxg|hyouuma|icecake|ichigoame|"
    "irodori warabi|iuui|jazz jack|jima|john kafka|kaede (sayappa)|kanamochi kanato|kaneni|karei|kase daiki|"
    "kentllaall|kikimi|kim eb|koujuan|krenz|k-suwabe|kuroduki (pieat)|lack|lam (ramdayo)|lanzi (415460661)|"
    "leo-dont-want-to-be-a-painter|lifeline (a384079959)|linnnoj|lirseven|lm7 (op-center)|luicent|"
    "lunch (lunchicken)|maccha (mochancc)|mamuru|masukudo (hamamoto hikaru)|takayama toshiaki|yushe quetzalli|"
    "zukky000|yaziri|starshadowmagician|tabisumika|tahra|tianliang duohe fangdongye|timbougami|tira 27|tizibade|"
    "tom-neko (zamudo akiyuki)|tsubonari|u tnmn|uenomigi|urakata (uracata)|vlfdus 0|voxel style|wanke|wenjun lin|"
    "xilmo|shanyao jiang tororo|yoneyama mai|youamo|Antonio J. Manzanedo|armored core|baotong yu|greg rutkowski|"
    "maung thuta|cuso4 suiwabutu|da mao banlangen|gamecg|sumieokazu|ying yi|zakkizaki|uzuki (uzukinokaze)|"
    "toto (caaaaarrot)|ttk (kirinottk)|syagamu|suaynnai wanzi|so-tora|soono (rlagpfl)|sleepy pang|shokuen|"
    "sho (sho lwlw)|shisantian|shiraho (color-56)|shion (mirudakemann)|shiba (s hi ba)|sheya tin|Shan Yanan|"
    "sencha (senchat)|ryota-h|rolua|hmear|ribao|reoen|rei (sanbonzakura)|quuni|quadruped mechas|qtian|qizhu|pvc|"
    "pottsness|potg (piotegu)|pixel art|jon davies|zaebucca|mui (muilog3)|Rachels Ham|zhenhaoowo|orihira|"
    "omone hokoma agm|omao|ohisashiburi|nyatabe|nonokuro|noco (pixiv14976070)|ningen mame|nep (nep 76)|nekojira|"
    "neco|nakamura_eight|nababa|mon (monmon2133)|momoko (momopoco)|mochizuki kei|mocha|min (120716)|mika pikazo|"
    "mignon|majin deng|liduke|xu youdian|gishiki (gshk)|kupuru (hirumamiyuu)|fukuro daizi"
).split("|"))


def _random_character_info(mode, background, age_min, age_max, anima_style_override, randomize,
                           nsfw_mode="SFW", framing_mode=None):
    if not 1 <= age_min <= age_max <= 100:
        raise ValueError("Age range must satisfy 1 <= age_min <= age_max <= 100")
    if nsfw_mode not in {"SFW", "NSFW", "Random"}:
        raise ValueError(f"Unsupported NSFW mode: {nsfw_mode}")
    if framing_mode not in {None, "Full Body", "Cowboy Shot", "Random"}:
        raise ValueError(f"Unsupported framing mode: {framing_mode}")
    info = {**_PROMPT_DEFAULTS, **_MODE_PROMPT_DEFAULTS[mode], "background_color": background}
    if randomize.get("sex", True):
        info["sex"] = random.choice(("female", "male"))
    if randomize.get("age", True):
        info["age"] = random.randint(age_min, age_max)
    info["nsfw"] = info["age"] >= 1 and (
        nsfw_mode == "NSFW" or nsfw_mode == "Random" and random.choice((False, True))
    )
    if framing_mode in {"Full Body", "Cowboy Shot"}:
        info["framing"] = "full_body" if framing_mode == "Full Body" else "cowboy_shot"
    elif framing_mode == "Random" or framing_mode is None and randomize.get("framing", True):
        info["framing"] = random.choice(("cowboy_shot", "full_body"))
    tags = CHARACTER_PRESETS["tags"]
    for field, groups in _FIELD_TAG_GROUPS.items():
        if not randomize.get(field, True):
            continue
        selected = []
        for group in groups:
            if group == "breast_size" and (info["sex"] != "female" or info["age"] < 1):
                continue
            options = [item["tag"] for item in tags[group]]
            # Identity groups have one choice; decorative details can form a set.
            count = random.randint(1, min(3, len(options))) if group in {"details", "face_details", "eye_features"} else 1
            selected.extend(random.sample(options, count))
        info[field] = ", ".join(selected)
    styles = [style for group in CHARACTER_STYLE_CATALOG["groups"] for style in group["styles"]]
    if mode == "anima" and anima_style_override:
        artist = random.choice(_ANIMA_ARTISTS) if randomize.get("style", True) else _ANIMA_ARTISTS[0]
        info.update(style="custom", custom_style=f"@{artist}")
        style_id = "artist_" + re.sub(r"[^a-z0-9]+", "_", artist.lower()).strip("_")
    else:
        style = random.choice(styles) if randomize.get("style", True) else next(
            style for style in styles if style["id"] == CHARACTER_STYLE_CATALOG["default_style"]
        )
        info["style"] = style_id = style["id"]
    return info, style_id


def _set_observed_breast_size(info, size):
    """Replace provisional size tags inside body, retaining other body traits."""
    size_tags = {
        item["tag"] for item in CHARACTER_PRESETS["tags"]["breast_size"]
        if item["tag"] != "mole_on_breast"
    }
    body = [tag.strip() for tag in info["body"].split(",") if tag.strip()]
    kept = [tag for tag in body if tag not in size_tags]
    body_description = f"The character has {', '.join(kept)}. " if kept else ""
    info["body"] = f"{body_description}Breast size: {size}"


def _rewrite_preview_output(clip, info, style_reference):
    """Use the shared field rewriter while keeping observed adult body traits exact."""
    from .character_generator import _call_comfy_node

    _ensure_prompt_server_progress_context()
    fields = _qi2_character_fields(info)
    pe_fields = {key: value for key, value in fields.items() if key not in {"aesthetics", "lora_prompt"}}
    system_prompt = _qi2_prompt_rewriter_system_prompt()
    generated = _call_comfy_node(
        "TextGenerate", clip=clip,
        prompt=f"{system_prompt}\n\n{QI2_STYLE_REWRITE_RULES}\n\ncharacter_fields:\n"
               + json.dumps(pe_fields, ensure_ascii=False),
        **QI2_TEXT_GENERATION_DEFAULTS,
    )[0]
    parsed = _qi2_json_result(generated)
    expanded = parsed.get("fields", {}) if parsed else {}
    expanded = dict(expanded) if isinstance(expanded, dict) else {}
    body_marker = "__VNCCS_PREVIEW_OBSERVED_BODY__"
    compile_fields = dict(fields)
    if info["age"] >= 1:
        # Insert the complete analysis response after the shared compiler's cleanup.
        compile_fields["body"] = body_marker
        expanded["body"] = body_marker
    prompt = _strip_unit_prompt_weights(_qi2_expanded_field_prompt(
        json.dumps({"fields": expanded}, ensure_ascii=False), compile_fields,
    ))
    if info["age"] >= 1:
        prompt = prompt.replace(body_marker + ".", info["body"], 1)
    if style_reference:
        prompt += f"\n\n{QI2_STYLE_REFERENCE_HEADING}\n{style_reference}"
    return prompt


def _analyze_breast_size(image):
    from .character_generator import _call_comfy_node

    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError("Breast size analysis requires exactly one RGB or RGBA image")
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        rgb = rgb * alpha + (1 - alpha)
    height, width = rgb.shape[1:3]
    scale = min(1.0, 1024 / max(height, width))
    if scale < 1.0:
        rgb = torch.nn.functional.interpolate(
            rgb.movedim(-1, 1), size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="bilinear", align_corners=False,
        ).movedim(1, -1)
    # Diffusion/VAE execution can displace the original encoder's mutable state.
    analysis_settings = normalize_gen_settings(QI2_DEFAULTS)
    analysis_settings["clip_type"] = "stable_diffusion"
    clip = load_generation_clip(analysis_settings)
    _ensure_prompt_server_progress_context()
    generated = _call_comfy_node(
        "TextGenerate", clip=clip, prompt=_BREAST_SIZE_ANALYSIS_PROMPT, image=rgb,
        **{
            **QI2_TEXT_GENERATION_DEFAULTS,
            "max_length": 512,
            "sampling_mode": {"sampling_mode": "off"},
        },
    )[0]
    log_event("analysis_output", component="StylePreview", level="debug", output=generated)
    return generated


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
        }, "optional": {
            "turbo_enabled": ("BOOLEAN", {"default": False, "label_on": "Turbo", "label_off": "Normal"}),
            "anima_style_override": ("BOOLEAN", {"default": False, "tooltip": "Use one exact @artist from the Anima override list. Ignored for QI2."}),
            "age_min": ("INT", {"default": 18, "min": 1, "max": 100}),
            "age_max": ("INT", {"default": 40, "min": 1, "max": 100}),
            **{f"randomize_{field}": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "Legacy control for workflows without framing_mode. The framing selector takes precedence."
                    if field == "framing" else "Choose new values each run. When disabled, use the preview default."
                ),
            }) for field in _RANDOM_FIELDS},
            "nsfw_mode": (["SFW", "NSFW", "Random"], {
                "default": "SFW", "tooltip": "Clothing mode. NSFW is restricted to adults; Random uses SFW for characters under 18.",
            }),
            "framing_mode": (["Full Body", "Cowboy Shot", "Random"], {
                "default": "Random", "tooltip": "Fixed crop or a new random crop each run. Overrides the legacy randomize_framing control.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "VNCCS/Testing"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    @logged_operation("StylePreview", "generate")
    @inference_stage()
    def generate(self, model, scale_mp, background_color, turbo_enabled=False,
                 anima_style_override=False, age_min=18, age_max=40, nsfw_mode="SFW",
                 framing_mode=None, **kwargs):
        mode = model.lower()
        if mode not in _MODE_PROMPT_DEFAULTS:
            raise ValueError(f"Unsupported generation model: {model}")
        if background_color not in {"Green", "Blue", "Transparent"}:
            raise ValueError(f"Unsupported background color: {background_color}")
        if background_color == "Transparent" and mode != "qi2":
            raise ValueError("Transparent background is available only with QI2")

        info, style_id = _random_character_info(
            mode, background_color, int(age_min), int(age_max), anima_style_override,
            {field: kwargs.get(f"randomize_{field}", True) for field in _RANDOM_FIELDS},
            nsfw_mode=nsfw_mode,
            framing_mode=framing_mode,
        )
        style_reference = _character_style_prompt(info)
        if mode == "anima":
            # Anima consumes the original tags, including the original size tag.
            # Keep curated species prose exclusive to the rewritten output.
            positive_text, negative_text = CharacterCreatorV2.construct_prompt(
                {**info, "race": ""}, mode, include_style=True,
            )
            if info["race"]:
                positive_text += f", ({info['race']})"
        else:
            positive_text, negative_text = CharacterCreatorV2.construct_prompt(info, mode, include_style=False)

        side = _square_side(scale_mp)
        settings = normalize_gen_settings({
            **(ANIMA_DEFAULTS if mode == "anima" else QI2_DEFAULTS),
            "generation_mode": mode,
            "diffusion_model_name": _select_diffusion_model(mode),
            "turbo_enabled": bool(turbo_enabled),
            "lora_stack": [],
        })
        if mode == "anima" and turbo_enabled:
            settings.update(steps=12, cfg=1.0)
        _, diffusion_model, clip, vae = load_generation_assets(settings)
        turbo = False
        if mode == "qi2":
            diffusion_model, turbo = prepare_qi2_model(diffusion_model, settings)
        elif turbo_enabled:
            lora_path = get_lora_full_path(settings["dmd_lora_name"])
            if not lora_path:
                raise ValueError(f"Anima Turbo LoRA not found: {settings['dmd_lora_name']}")
            diffusion_model, clip = comfy.sd.load_lora_for_models(
                diffusion_model, clip, comfy.utils.load_torch_file(lora_path, safe_load=True),
                settings["dmd_lora_strength"], 0.0,
            )

        positive, negative, prompt = encode_generation_conditioning(
            clip, vae, positive_text, negative_text, settings,
            style_reference=style_reference if mode == "qi2" else "",
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
            seed=random.getrandbits(63),
            steps=settings["steps"],
            cfg=settings["cfg"],
            sampler_name=settings["sampler"],
            scheduler=settings["scheduler"],
            gen_settings=settings,
            qi2_turbo=turbo,
        )
        image = decode_generation_samples(vae, sampled, settings)
        if background_color == "Transparent" and image.shape[-1] != 4:
            raise RuntimeError("QI2 returned no alpha channel for transparent background")

        output_dir = os.path.join(folder_paths.get_output_directory(), _SUBFOLDER)
        os.makedirs(output_dir, exist_ok=True)
        path, filename = _next_output_path(output_dir, mode, style_id)
        tensor2pil(image).save(path)
        log_event("preview_saved", component="StylePreview", file=os.path.basename(path))
        if info["age"] >= 1:
            breast_size = _analyze_breast_size(image)
            _set_observed_breast_size(info, breast_size)
            log_event("body_analysis", component="StylePreview", breast_size=breast_size)
        if mode == "anima" or info["age"] >= 1:
            rewrite_clip = load_generation_clip(normalize_gen_settings(QI2_DEFAULTS))
            prompt = _rewrite_preview_output(rewrite_clip, info, style_reference)
            del rewrite_clip
        return {
            "ui": {"images": [{"filename": filename, "subfolder": _SUBFOLDER, "type": "output"}]},
            "result": (image, prompt),
        }


NODE_CLASS_MAPPINGS = {"VNCCSStylePreviewTest": VNCCSStylePreviewTest}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCSStylePreviewTest": "VNCCS Style Preview Test"}
