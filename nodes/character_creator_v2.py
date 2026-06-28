
import os
import json
import torch
import folder_paths
import comfy.sd
import comfy.samplers
import comfy.utils
import nodes
import server
from aiohttp import web
from PIL import Image
import io
import base64
import numpy as np
import traceback
import inspect
import random
import re

from ..utils import (
    load_character_info, ensure_character_structure, EMOTIONS, MAIN_DIRS,
    save_config, build_face_details, generate_seed, dedupe_tokens,
    apply_sex, append_age, load_config, age_strength,
    list_characters, character_dir, base_output_dir,
    sheets_dir, faces_dir, normalize_hair_tags, ensure_safe_name,
    get_full_path_agnostic,
)
from .vnccs_utils import _ensure_qwen_vl_assets

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def list_pose_preview_files(character_name, costume=None):
    try:
        base_char_path = character_dir(character_name)
        sprite_roots = []
        if costume:
            sprite_roots.extend([
                os.path.join(base_char_path, "Sprites", costume, "Neutral"),
                os.path.join(base_char_path, "Sprites", costume),
            ])
        sprite_roots.extend([
            os.path.join(base_char_path, "Sprites", "Naked", "Neutral"),
            os.path.join(base_char_path, "Sprites", "Original", "Neutral"),
            os.path.join(base_char_path, "Sprites", "Naked"),
            os.path.join(base_char_path, "Sprites", "Original"),
        ])
        print(f"[VNCCS Debug] Checking Pose Preview Paths: {sprite_roots}")

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files = []
        for poses_dir in sprite_roots:
            if not os.path.isdir(poses_dir):
                continue
            root_files = [
                os.path.join(poses_dir, filename)
                for filename in os.listdir(poses_dir)
                if os.path.isfile(os.path.join(poses_dir, filename))
                and os.path.splitext(filename)[1].lower() in image_exts
            ]
            if root_files:
                files = sorted(root_files)
                break
        print(f"[VNCCS Debug] Pose preview files found: {len(files)}")
        return files
    except Exception as e:
        print(f"[VNCCS] Pose Preview List Failed: {e}")
        return []


def get_pose_preview(character_name, index=None, costume=None):
    """
    Pick a current pose image from the selected character's Sprites folder.
    Returns: (torch.Tensor (1,H,W,3), index, count) or (None, None, 0)
    """
    try:
        files = list_pose_preview_files(character_name, costume=costume)
        if not files:
            return None, None, 0

        count = len(files)
        if index is None:
            selected_index = random.randrange(count)
        else:
            selected_index = int(index) % count
        selected_file = files[selected_index]
        print(f"[VNCCS Debug] Using pose preview file: {selected_file}")
        img = Image.open(selected_file).convert("RGB")
        return pil2tensor(img), selected_index, count
    except Exception as e:
        print(f"[VNCCS] Pose Preview Fallback Failed: {e}")
        return None, None, 0


def get_random_pose_preview(character_name):
    image, _index, _count = get_pose_preview(character_name)
    return image

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

# --- PREVIEW CACHE ---
PREVIEW_CACHE = {
    "asset_key": None,
    "asset_obj": None, # (model, clip, vae)
    "loras": {}       # name -> tensor_dict
}

ILLUSTRIOUS_DEFAULTS = {
    "generation_mode": "illustrious",
    "ckpt_name": "",
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20,
    "cfg": 8.0,
}

ANIMA_DEFAULTS = {
    "generation_mode": "anima",
    "diffusion_model_name": "",
    "clip_name": "qwen_3_06b_base.safetensors",
    "vae_name": "qwen_image_vae.safetensors",
    "clip_type": "stable_diffusion",
    "sampler": "er_sde",
    "scheduler": "simple",
    "steps": 30,
    "cfg": 4.0,
    "turbo_enabled": False,
    "dmd_lora_name": "anima\\anima-turbo-lora-v0.1.safetensors",
    "dmd_lora_strength": 1.0,
    "lora_stack": [],
}

DEFAULT_PREVIEW_WIDTH = 640
DEFAULT_PREVIEW_HEIGHT = 1536


def safe_filename_list(category):
    try:
        return folder_paths.get_filename_list(category)
    except Exception:
        return []


def get_lora_full_path(lora_name):
    return get_full_path_agnostic(folder_paths, "loras", lora_name, require_exists=True)


def _validate_character_wizard_gguf(path, file_label="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_label} was not written: {path}")

    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise ValueError(f"{file_label} is too small to be a valid GGUF file ({size} bytes)")

    with open(path, "rb") as file:
        magic = file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{file_label} is not a valid GGUF file (magic={magic!r})")


def _find_character_wizard_model():
    base_path = folder_paths.models_dir
    possible_names = [
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        "qwen2-vl-7b-instruct-q4_k_m.gguf",
    ]
    search_dirs = [os.path.join(base_path, "LLM"), os.path.join(base_path, "llm"), base_path]

    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in possible_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def _ensure_character_wizard_model():
    model_path, _mmproj_path = _ensure_qwen_vl_assets()
    return model_path


def _extract_character_tag_options(tags_data):
    tags = tags_data.get("tags", {}) if isinstance(tags_data, dict) else {}

    def collect(value):
        items = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("tag"):
                    items.append(str(item["tag"]))
        elif isinstance(value, dict):
            for sub_value in value.values():
                items.extend(collect(sub_value))
        return items

    return {
        "race": collect(tags.get("races", [])),
        "hair": collect(tags.get("hair_color", [])) + collect(tags.get("hairstyles", [])),
        "eyes": collect(tags.get("eyes", {})),
        "body": collect(tags.get("breast_size", [])),
        "additional_details": collect(tags.get("details", [])),
    }


SKIN_COLOR_OPTIONS = [
    "light skin",
    "fair skin",
    "pale skin",
    "tan skin",
    "dark skin",
    "brown skin",
    "olive skin",
    "blue skin",
    "green skin",
    "grey skin",
]

SKIN_COLOR_HINT_RE = re.compile(
    r"\b(skin|complexion|pale|fair|light[- ]skinned|tan|tanned|dark[- ]skinned|"
    r"brown[- ]skinned|black|black[- ]skinned|afro|african|african[- ]american|"
    r"olive|blue[- ]skinned|green[- ]skinned|grey[- ]skinned|gray[- ]skinned)\b|"
    r"(афро|африкан|темн[а-яё]+(?:\\s+кож[а-яё]+)?|смугл[а-яё]+)",
    re.IGNORECASE,
)

RACE_OPTION_TAGS = {
    "animal_ears", "cat_girl", "fox_girl", "rabbit_girl", "horse_girl", "kemonomimi",
    "furry", "furry_female", "demon_girl", "demon_horns", "dragon_girl", "dragon_horns",
    "wings", "feathered_wings", "pointy_ears", "vampire", "mermaid", "naga",
    "rabbit_ears", "horse_tail", "tail",
}
HUMAN_HINT_RE = re.compile(
    r"\b(human|person|student|schoolboy|schoolgirl|man|woman|boy|girl|guy|"
    r"afro|african|african[- ]american|black)\b|"
    r"(человек|студент|студентка|парень|девушка|мужчина|женщина|афро|африкан)",
    re.IGNORECASE,
)
YOUNG_BODY_HINT_RE = re.compile(
    r"\b(young|student|teen|teenage|schoolboy|schoolgirl|college)\b|"
    r"(молод|юноша|юная|студент|студентка|подрост)",
    re.IGNORECASE,
)
SLIM_BODY_HINT_RE = re.compile(
    r"\b(slim|slender|thin|skinny|lean|petite)\b|"
    r"(стройн|худ|тонк)",
    re.IGNORECASE,
)
ATHLETIC_BODY_HINT_RE = re.compile(
    r"\b(athletic|fit|sporty|muscular)\b|"
    r"(атлет|спорт|мускул|подтянут)",
    re.IGNORECASE,
)


def _parse_character_wizard_json(content):
    data = None
    try:
        import json_repair
        data = json_repair.loads(content)
    except Exception:
        data = None

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        try:
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]
            else:
                match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
            data = json.loads(json_str.strip())
        except Exception:
            data = None

    if not isinstance(data, dict):
        return None

    result = {}
    for key in ["race", "skin_color", "hair", "eyes", "face", "body", "additional_details"]:
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()

    sex = str(data.get("sex", "female")).strip().lower()
    result["sex"] = "male" if sex.startswith("m") else "female"
    try:
        age = int(float(data.get("age", 18)))
    except Exception:
        age = 18
    result["age"] = max(1, min(100, age))
    return result


def _split_prompt_tokens(value):
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _join_prompt_tokens(tokens):
    seen = set()
    out = []
    for token in tokens:
        normalized = token.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            out.append(normalized)
    return ", ".join(out)


def _infer_skin_color_from_description(description):
    text = str(description or "").lower()
    if re.search(r"\b(afro|african|african[- ]american|black|black[- ]skinned)\b|афро|африкан|темн|смугл", text):
        return "dark skin"
    if re.search(r"\b(brown[- ]skinned|brown skin)\b|коричнев", text):
        return "brown skin"
    if re.search(r"\b(tan|tanned)\b|загорел", text):
        return "tan skin"
    if re.search(r"\b(olive)\b|оливков", text):
        return "olive skin"
    if re.search(r"\b(pale)\b|бледн", text):
        return "pale skin"
    if re.search(r"\b(fair|light[- ]skinned|light skin)\b|светл", text):
        return "fair skin"
    return ""


def _normalize_wizard_race(race, description):
    tokens = _split_prompt_tokens(race)
    kept = []
    for token in tokens:
        key = token.strip().lower()
        if key in {"human", "person", "man", "woman", "boy", "girl"}:
            continue
        if key in {"afro", "afro_student", "african", "african_student", "black", "black_student", "student"}:
            continue
        if key in RACE_OPTION_TAGS:
            kept.append(token)
    if kept:
        return _join_prompt_tokens(kept)
    if HUMAN_HINT_RE.search(description or ""):
        return "human"
    return ""


def _normalize_wizard_body(body, description, sex):
    tokens = _split_prompt_tokens(body)
    kept = []
    for token in tokens:
        key = token.lower()
        if key in {"body", "unknown", "none", "n/a"}:
            continue
        kept.append(token)

    if any(re.search(r"\b(slim|slender|thin|skinny|lean|petite|athletic|fit|muscular|average build|normal build)\b", t, re.I) for t in kept):
        return _join_prompt_tokens(kept)

    text = str(description or "")
    if SLIM_BODY_HINT_RE.search(text):
        kept.append("slim build")
    elif ATHLETIC_BODY_HINT_RE.search(text):
        kept.append("athletic build")
    elif YOUNG_BODY_HINT_RE.search(text):
        kept.append("average build")

    if sex == "female" and not any("breast" in t.lower() or "chest" in t.lower() for t in kept):
        kept.append("small_breasts")

    return _join_prompt_tokens(kept)


def postprocess_character_wizard_result(parsed, user_description):
    result = dict(parsed or {})
    description = str(user_description or "")
    result["race"] = _normalize_wizard_race(result.get("race", ""), description)

    inferred_skin = _infer_skin_color_from_description(description)
    if inferred_skin:
        result["skin_color"] = inferred_skin
    elif not SKIN_COLOR_HINT_RE.search(description):
        result["skin_color"] = ""

    result["body"] = _normalize_wizard_body(
        result.get("body", ""),
        description,
        result.get("sex", "female"),
    )
    return result


def normalize_gen_settings(gen_settings):
    normalized = dict(gen_settings or {})
    generation_mode = str(normalized.get("generation_mode", "illustrious")).lower()
    mode_settings = normalized.get("mode_settings", {})
    mode_profile = mode_settings.get(generation_mode, {}) if isinstance(mode_settings, dict) else {}
    defaults = ANIMA_DEFAULTS if generation_mode == "anima" else ILLUSTRIOUS_DEFAULTS
    merged = dict(defaults)
    merged.update(normalized)
    if isinstance(mode_profile, dict):
        merged.update(mode_profile)
    merged["generation_mode"] = generation_mode
    return merged


def _call_loader_node(class_names, method_names, **kwargs):
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for class_name in class_names:
        loader_cls = mappings.get(class_name)
        if loader_cls is None:
            continue
        loader = loader_cls()
        for method_name in method_names:
            method = getattr(loader, method_name, None)
            if method is None:
                continue
            signature = inspect.signature(method)
            accepted_kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature.parameters
            }
            result = method(**accepted_kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result
    return None


def _call_node_method(class_names, method_names, **kwargs):
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for class_name in class_names:
        node_cls = mappings.get(class_name)
        if node_cls is None:
            continue
        node_instance = node_cls()
        for method_name in method_names:
            method = getattr(node_instance, method_name, None)
            if method is None:
                continue
            signature = inspect.signature(method)
            accepted_kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature.parameters
            }
            return method(**accepted_kwargs)
    return None


def load_anima_assets(gen_settings):
    diffusion_model_name = gen_settings.get("diffusion_model_name")
    clip_name = gen_settings.get("clip_name")
    vae_name = gen_settings.get("vae_name")
    clip_type_name = str(gen_settings.get("clip_type", "stable_diffusion") or "stable_diffusion").lower()

    if not diffusion_model_name:
        raise ValueError("No Diffusion Model selected in Character Creator V2 ANIMA mode")
    if not clip_name:
        raise ValueError("No CLIP selected in Character Creator V2 ANIMA mode")
    if not vae_name:
        raise ValueError("No VAE selected in Character Creator V2 ANIMA mode")

    model = None
    if model is None:
        model = _call_loader_node(
            ["UNETLoader", "Load Diffusion Model"],
            ["load_unet", "load_model", "load_diffusion_model"],
            unet_name=diffusion_model_name,
            model_name=diffusion_model_name,
            diffusion_model_name=diffusion_model_name,
            weight_dtype="default",
        )
    if model is None and hasattr(comfy.sd, "load_diffusion_model"):
        diffusion_model_path = get_full_path_agnostic(folder_paths, "diffusion_models", diffusion_model_name)
        if diffusion_model_path:
            model = comfy.sd.load_diffusion_model(diffusion_model_path)

    clip = None
    if clip is None:
        clip = _call_loader_node(
            ["CLIPLoader", "Load CLIP"],
            ["load_clip", "load_model"],
            clip_name=clip_name,
            model_name=clip_name,
            type=clip_type_name,
            device="default",
        )
    if clip is None and hasattr(comfy.sd, "load_clip"):
        clip_path = get_full_path_agnostic(folder_paths, "text_encoders", clip_name)
        if clip_path:
            clip_type = getattr(comfy.sd.CLIPType, clip_type_name.upper(), None)
            if clip_type is None:
                raise ValueError(
                    f"ComfyUI CLIPType.{clip_type_name.upper()} is not available. "
                    "ANIMA uses the official workflow CLIPLoader type 'stable_diffusion'."
                )
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
            )

    vae = None
    if vae is None:
        vae = _call_loader_node(
            ["VAELoader", "Load VAE"],
            ["load_vae", "load_model"],
            vae_name=vae_name,
            model_name=vae_name,
        )
    if vae is None and hasattr(comfy.sd, "load_vae"):
        vae_path = get_full_path_agnostic(folder_paths, "vae", vae_name)
        if vae_path:
            vae = comfy.sd.load_vae(vae_path)

    if model is None:
        raise ValueError(f"Failed to load Diffusion Model '{diffusion_model_name}'")
    if clip is None:
        raise ValueError(f"Failed to load CLIP '{clip_name}'")
    if vae is None:
        raise ValueError(f"Failed to load VAE '{vae_name}'")

    return model, clip, vae


def load_generation_assets(gen_settings):
    generation_mode = str(gen_settings.get("generation_mode", "illustrious")).lower()

    if generation_mode == "anima":
        asset_key = (
            generation_mode,
            gen_settings.get("diffusion_model_name", ""),
            gen_settings.get("clip_name", ""),
            gen_settings.get("vae_name", ""),
        )
        model, clip, vae = load_anima_assets(gen_settings)
        return asset_key, model, clip, vae

    ckpt_name = gen_settings.get("ckpt_name")
    if not ckpt_name:
        raise ValueError("No Checkpoint selected in Character Creator V2")

    ckpt_path = get_full_path_agnostic(folder_paths, "checkpoints", ckpt_name, require_exists=True)
    if not ckpt_path:
        raise ValueError(f"Checkpoint path not found for '{ckpt_name}'")

    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    model, clip, vae = out[:3]

    if model is None:
        raise ValueError(f"Failed to load Model from checkpoint '{ckpt_name}'")
    if clip is None:
        raise ValueError(f"Failed to load CLIP from checkpoint '{ckpt_name}'")
    if vae is None:
        raise ValueError(f"Failed to load VAE from checkpoint '{ckpt_name}'")

    return (generation_mode, ckpt_name), model, clip, vae


def get_generation_resolution(gen_settings):
    return DEFAULT_PREVIEW_WIDTH, DEFAULT_PREVIEW_HEIGHT


def create_generation_latent(model, width, height, gen_settings, batch_size=1):
    if str(gen_settings.get("generation_mode", "illustrious")).lower() == "anima":
        generated = _call_node_method(
            ["EmptyLatentImage"],
            ["generate"],
            width=width,
            height=height,
            batch_size=batch_size,
        )
        if isinstance(generated, tuple) and generated:
            return generated[0]
        if generated is not None:
            return generated
    return {"samples": torch.zeros([batch_size, 4, height // 8, width // 8], device=model.load_device)}


def sample_generation_latent(model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, gen_settings):
    if str(gen_settings.get("generation_mode", "illustrious")).lower() == "anima":
        sampled = _call_node_method(
            ["KSampler"],
            ["sample"],
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            latent=latent,
            denoise=1.0,
        )
        if isinstance(sampled, tuple) and sampled:
            return sampled[0]
        if sampled is not None:
            return sampled

    return nodes.common_ksampler(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent,
        denoise=1.0,
    )[0]


def encode_generation_prompt(clip, text, gen_settings):
    if str(gen_settings.get("generation_mode", "illustrious")).lower() == "anima":
        encoded = _call_node_method(
            ["CLIPTextEncode"],
            ["encode"],
            clip=clip,
            text=text,
        )
        if isinstance(encoded, tuple) and encoded:
            return encoded[0]
        if encoded is not None:
            return encoded

    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def validate_anima_conditioning(positive, negative, clip_name):
    def context_width(conditioning):
        try:
            if not conditioning:
                return None
            return conditioning[0][0].shape[-1]
        except Exception:
            return None

    widths = [width for width in (context_width(positive), context_width(negative)) if width is not None]
    bad_widths = [width for width in widths if width != 1024]
    if bad_widths:
        raise ValueError(
            "ANIMA conditioning has the wrong text-encoder width "
            f"{bad_widths[0]} instead of 1024. Select the official ANIMA text encoder "
            f"'qwen_3_06b_base.safetensors' in the CLIP field; current CLIP is '{clip_name}'."
        )


def decode_generation_samples(vae, samples, gen_settings):
    def unwrap_latent_samples(value):
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        seen_ids = set()
        while isinstance(value, dict) and "samples" in value:
            value_id = id(value)
            if value_id in seen_ids:
                break
            seen_ids.add(value_id)
            value = value["samples"]
            while isinstance(value, (list, tuple)) and value:
                value = value[0]
        return value

    if str(gen_settings.get("generation_mode", "illustrious")).lower() == "anima":
        latent_payload = samples if isinstance(samples, dict) else {"samples": samples}
        latent_tensor = unwrap_latent_samples(latent_payload)
        decode_payload = {"samples": latent_tensor}
        decoded = _call_node_method(
            ["VAEDecode"],
            ["decode"],
            samples=decode_payload,
            vae=vae,
        )
        if isinstance(decoded, tuple) and decoded:
            return decoded[0]
        if decoded is not None:
            return decoded
        return vae.decode(latent_tensor)
    latent_samples = unwrap_latent_samples(samples)
    return vae.decode_tiled(latent_samples, tile_x=512, tile_y=512)

if server:
    @server.PromptServer.instance.routes.get("/vnccs/context_lists")
    async def get_context_lists(request):
        try:
            # Checkpoints
            checkpoints = safe_filename_list("checkpoints")
            diffusion_models = safe_filename_list("diffusion_models")
            text_encoders = safe_filename_list("text_encoders")
            vae_models = safe_filename_list("vae")
            
            # Samplers & Schedulers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
            
            # Character List
            characters = list_characters()
            
            # LoRA List
            loras = safe_filename_list("loras")

            return web.json_response({
                "checkpoints": checkpoints,
                "diffusion_models": diffusion_models,
                "text_encoders": text_encoders,
                "vae_models": vae_models,
                "samplers": samplers,
                "schedulers": schedulers,
                "characters": characters,
                "loras": loras
            })
        except Exception as e:
            return web.Response(status=500, text=str(e))
            
    @server.PromptServer.instance.routes.get("/vnccs/character_info")
    async def get_character_info(request):
        try:
            name = request.rel_url.query.get("character", "")
            if not name:
                return web.json_response({})
            name = ensure_safe_name(name, "character")
                
            config = load_config(name)
            if config and "character_info" in config:
                return web.json_response(config["character_info"])
            return web.json_response({})
            
        except Exception as e:
            traceback.print_exc() # Print to console
            return web.Response(status=500, text="Failed to load character info")

    @server.PromptServer.instance.routes.get("/vnccs/get_cached_preview")
    async def get_cached_preview(request):
        try:
            character = request.rel_url.query.get("character", "")
            if not character or ".." in character or "/" in character or "\\" in character:
                return web.Response(status=400)
            
            c_path = os.path.join(character_dir(character), "cache", "preview.png")
            if os.path.exists(c_path):
                return web.FileResponse(c_path)
            print(f"[VNCCS] Cached preview not found at: {c_path}")
            return web.Response(status=404)
        except Exception as e:
            print(f"[VNCCS] Error serving cached preview: {e}")
            return web.Response(status=500, text=str(e))

    @server.PromptServer.instance.routes.get("/vnccs/get_character_pose_preview")
    async def get_character_pose_preview(request):
        try:
            character = request.rel_url.query.get("character", "")
            if not character or ".." in character or "/" in character or "\\" in character:
                return web.Response(status=400)
            costume = request.rel_url.query.get("costume") or None
            if costume:
                try:
                    costume = ensure_safe_name(costume, "costume")
                except ValueError:
                    return web.Response(status=400)

            index_raw = request.rel_url.query.get("index")
            try:
                index = int(index_raw) if index_raw not in (None, "") else None
            except (TypeError, ValueError):
                index = None
            files = list_pose_preview_files(character, costume=costume)
            if not files:
                return web.Response(status=404)

            count = len(files)
            selected_index = random.randrange(count) if index is None else index % count
            return web.FileResponse(
                files[selected_index],
                headers={
                    "X-VNCCS-Preview-Index": str(selected_index),
                    "X-VNCCS-Preview-Count": str(count),
                },
            )
        except Exception as e:
            print(f"[VNCCS] Error serving character pose preview: {e}")
            return web.Response(status=500, text=str(e))

    @server.PromptServer.instance.routes.get("/vnccs/get_character_pose_preview_meta")
    async def get_character_pose_preview_meta(request):
        try:
            character = request.rel_url.query.get("character", "")
            if not character or ".." in character or "/" in character or "\\" in character:
                return web.json_response({"count": 0}, status=400)
            costume = request.rel_url.query.get("costume") or None
            if costume:
                try:
                    costume = ensure_safe_name(costume, "costume")
                except ValueError:
                    return web.json_response({"count": 0}, status=400)
            files = list_pose_preview_files(character, costume=costume)
            return web.json_response({"count": len(files)})
        except Exception as e:
            print(f"[VNCCS] Error serving character pose preview metadata: {e}")
            return web.json_response({"count": 0, "error": str(e)}, status=500)

    @server.PromptServer.instance.routes.get("/vnccs/get_tags")
    async def get_tags(request):
        try:
            # Locate the file relative to the node
            # Assuming nodes/character_creator_v2.py -> ../character_template/character_tags.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)
            tags_path = os.path.join(root_dir, "character_template", "character_tags.json")
            
            if not os.path.exists(tags_path):
                return web.Response(status=404, text="Tags file not found")
                
            with open(tags_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return web.json_response(data)
        except Exception as e:
            return web.Response(status=500, text=str(e))

    @server.PromptServer.instance.routes.post("/vnccs/character_wizard")
    async def vnccs_character_wizard(request):
        try:
            try:
                import llama_cpp
            except Exception as e:
                return web.json_response({
                    "error": "DEPENDENCY_MISSING",
                    "message": f"llama-cpp-python is required for Character Wizard: {e}",
                    "model_name": "llama-cpp-python",
                }, status=500)

            post = await request.json()
            user_description = str(post.get("description", "")).strip()
            if not user_description:
                return web.Response(status=400, text="No character description provided")

            try:
                model_path = _ensure_character_wizard_model()
            except Exception as e:
                return web.json_response({
                    "error": "MODEL_DOWNLOAD_FAILED",
                    "message": str(e),
                    "model_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                }, status=500)

            try:
                _validate_character_wizard_gguf(model_path, os.path.basename(model_path))
            except Exception as e:
                return web.json_response({
                    "error": "MODEL_INVALID",
                    "message": f"Qwen GGUF model file is invalid or incomplete: {e}",
                    "model_name": os.path.basename(model_path),
                }, status=422)

            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)
            tags_path = os.path.join(root_dir, "character_template", "character_tags.json")
            tags_data = {}
            if os.path.exists(tags_path):
                with open(tags_path, "r", encoding="utf-8") as f:
                    tags_data = json.load(f)
            tag_options = _extract_character_tag_options(tags_data)

            system_prompt = (
                "You are a professional anime/game character designer. "
                "Convert broad character ideas into concise structured character fields. "
                "Output valid JSON only."
            )
            user_prompt = f"""
Create a character from this abstract idea:
{user_description}

Prefer these exact existing tags when they fit. Only invent a different tag or phrase if no listed tag matches the character:
{json.dumps(tag_options, ensure_ascii=False)}

Use one of these skin_color values only when the user's idea explicitly mentions skin tone or complexion:
{json.dumps(SKIN_COLOR_OPTIONS, ensure_ascii=False)}

Return a raw JSON object with exactly these keys:
- sex: "male" or "female"
- age: integer from 1 to 100
- race
- skin_color
- body
- face
- hair
- eyes
- additional_details

Rules:
- Use comma-separated prompt fragments for text fields.
- The race field is for species/fantasy traits only. For normal humans set race to "human".
- Never put ethnicity, nationality, profession, role, clothing, or archetype in race. Examples of invalid race values: "afro_student", "black student", "asian girl", "teacher".
- Put skin tone in skin_color, not race. "afro", "African", "African-American", "black", or similar means skin_color should be "dark skin" unless another skin tone is explicit.
- For body, always provide a visible body/build descriptor. Use listed breast/chest tags when relevant, and add concise build phrases like "slim build", "average build", "athletic build" when useful.
- For race, hair, eyes, body and additional_details, prefer exact tags from the provided tag list when they fit.
- For skin_color, do not guess a default. Use an empty string unless the user's idea explicitly mentions skin tone, complexion, or non-human skin color.
- Do not use "pale skin" as a fallback.
- Do not describe clothing or outfit items.
- Do not add background, camera, pose, quality tags, style tags, nsfw, nudity, sex acts, or negative prompts.
- Keep fields practical for the existing character form.
- If a field is not needed, use an empty string.
- Set sex and age explicitly based on the user's description. If unspecified, infer a reasonable adult character.

Example:
{{
  "sex": "female",
  "age": 24,
  "race": "demon_girl, demon_horns",
  "skin_color": "",
  "body": "medium_breasts, slim waist",
  "face": "mole_under_eye, sharp features",
  "hair": "white_hair, long_hair, blunt_bangs",
  "eyes": "red_eyes, glowing",
  "additional_details": "tattoo, black_nails"
}}
"""

            print(f"[CharacterCreatorV2] Character Wizard loading model: {model_path}")
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=6144,
                n_gpu_layers=-1,
                verbose=False,
            )

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=900,
                temperature=0.3,
            )

            content = response["choices"][0]["message"]["content"]
            print(f"[CharacterCreatorV2] Character Wizard raw output: {content}")
            parsed = _parse_character_wizard_json(content or "")
            if parsed is None:
                return web.json_response({
                    "error": "PARSE_ERROR",
                    "message": "Failed to parse Character Wizard JSON output.",
                    "raw": content or "",
                }, status=500)
            parsed = postprocess_character_wizard_result(parsed, user_description)

            return web.json_response(parsed)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "error": "INFERENCE_ERROR",
                "message": f"Engine Error: {e}",
                "model_name": "Qwen2VL (Check Console)",
            }, status=500)

    @server.PromptServer.instance.routes.post("/vnccs/preview_generate")
    async def preview_generate(request):
        try:
            data = await request.json()
            gen_settings = normalize_gen_settings(data.get("gen_settings", {}))
            char_info = data.get("character_info", {})
            character_name = data.get("character", "Unknown")

            # Generate Prompt
            positive_text, negative_text = CharacterCreatorV2.construct_prompt(char_info)
            
            steps = int(gen_settings.get("steps", 20))
            cfg = float(gen_settings.get("cfg", 8.0))
            sampler_name = gen_settings.get("sampler", "euler")
            scheduler = gen_settings.get("scheduler", "normal")
            seed = generate_seed(int(gen_settings.get("seed", 0)))
            generation_mode = gen_settings.get("generation_mode", "illustrious")

            # Resolution
            width, height = get_generation_resolution(gen_settings)

            # Load Models (With Cache)
            global PREVIEW_CACHE
            
            with torch.inference_mode():
                asset_key = None
                if generation_mode == "anima":
                    asset_key = (
                        generation_mode,
                        gen_settings.get("diffusion_model_name", ""),
                        gen_settings.get("clip_name", ""),
                        gen_settings.get("vae_name", ""),
                    )
                else:
                    asset_key = (generation_mode, gen_settings.get("ckpt_name", ""))

                if PREVIEW_CACHE["asset_key"] == asset_key and PREVIEW_CACHE["asset_obj"]:
                    print(f"[VNCCS] Preview: Using Cached Assets {asset_key}")
                    model, clip, vae = PREVIEW_CACHE["asset_obj"]
                else:
                    print(f"[VNCCS] Preview: Loading Assets {asset_key}")
                    try:
                        _, model, clip, vae = load_generation_assets(gen_settings)
                    except ValueError as exc:
                        return web.Response(status=400, text=str(exc))
                    PREVIEW_CACHE["asset_key"] = asset_key
                    PREVIEW_CACHE["asset_obj"] = (model, clip, vae)

                # Clone to avoid tainting cached model with LoRAs
                model = model.clone()
                clip = clip.clone()

                # Helper for Cached LoRA
                def apply_lora_cached(m, c, l_name, l_strength, clip_strength=None):
                    if not l_name or l_name == "None": return m, c
                    
                    lora_dict = PREVIEW_CACHE["loras"].get(l_name)
                    if not lora_dict:
                        l_path = get_lora_full_path(l_name)
                        if l_path:
                            lora_dict = comfy.utils.load_torch_file(l_path, safe_load=True)
                            PREVIEW_CACHE["loras"][l_name] = lora_dict
                            print(f"[VNCCS] Preview: Cached LoRA '{l_name}'")
                    
                    if lora_dict:
                        return comfy.sd.load_lora_for_models(m, c, lora_dict, l_strength, l_strength if clip_strength is None else clip_strength)
                    return m, c

                if generation_mode == "anima":
                    if gen_settings.get("turbo_enabled"):
                        dmd_lora_name = gen_settings.get("dmd_lora_name")
                        dmd_lora_strength = float(gen_settings.get("dmd_lora_strength", 1.0))
                        model, clip = apply_lora_cached(model, clip, dmd_lora_name, dmd_lora_strength, 0.0)

                    lora_stack = gen_settings.get("lora_stack", [])
                    for l_item in lora_stack:
                        model, clip = apply_lora_cached(model, clip, l_item.get("name"), float(l_item.get("strength", 1.0)))
                else:
                    dmd_lora_name = gen_settings.get("dmd_lora_name")
                    dmd_lora_strength = float(gen_settings.get("dmd_lora_strength", 1.0))
                    model, clip = apply_lora_cached(model, clip, dmd_lora_name, dmd_lora_strength)

                    age_lora_name = gen_settings.get("age_lora_name")
                    if age_lora_name:
                        age = int(char_info.get("age", 18))
                        age_str = age_strength(age)
                        model, clip = apply_lora_cached(model, clip, age_lora_name, age_str)

                    lora_stack = gen_settings.get("lora_stack", [])
                    for l_item in lora_stack:
                        model, clip = apply_lora_cached(model, clip, l_item.get("name"), float(l_item.get("strength", 1.0)))

                # 2. Encode Prompts
                positive_cond = encode_generation_prompt(clip, positive_text, gen_settings)
                negative_cond = encode_generation_prompt(clip, negative_text, gen_settings)
                if generation_mode == "anima":
                    validate_anima_conditioning(positive_cond, negative_cond, gen_settings.get("clip_name", ""))

                # Patch PromptServer
                if not hasattr(server.PromptServer.instance, 'last_prompt_id'):
                    server.PromptServer.instance.last_prompt_id = 'preview_gen'

                # 3. Sample
                latent = create_generation_latent(model, width, height, gen_settings)
                sampled = sample_generation_latent(
                    model=model,
                    positive=positive_cond,
                    negative=negative_cond,
                    latent=latent,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    gen_settings=gen_settings,
                )

                # 4. Decode
                vae_decoded = decode_generation_samples(vae, sampled, gen_settings)
                i = 255. * vae_decoded.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])

            # Save Smart Cache
            try:
                c_dir = os.path.join(character_dir(character_name), "cache")
                os.makedirs(c_dir, exist_ok=True)
                c_path = os.path.join(c_dir, "preview.png")
                img.save(c_path)
            except Exception as e:
                print(f"[VNCCS] Failed to save preview cache: {e}")

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return web.json_response({"image": img_b64})

        except Exception as e:
            traceback.print_exc()
            return web.Response(status=500, text=str(e))


class CharacterCreatorV2:
    """
    Standalone GUI for Character Creation with On-Demand Preview.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}), 
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "*")
    RETURN_NAMES = ("character", "sheets_path", "background")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @staticmethod
    def construct_prompt(info):
        """
        Centralized logic for constructing positive/negative prompts from character info.
        """
        aesthetics = info.get("aesthetics", "masterpiece")
        sex = info.get("sex", "female")
        age = int(info.get("age", 18))
        
        # Base Prompt
        positive_prompt = f"{aesthetics}, simple background, expressionless, solo, cowboy_shot"
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
        
        # NSFW / Clothing
        nsfw_val = info.get("nsfw", False)
        is_nsfw = nsfw_val if isinstance(nsfw_val, bool) else str(nsfw_val).lower() in ("true", "1", "yes")
        
        if is_nsfw:
             nude_phrase = "(naked, nude, penis)" if sex == "male" else "(naked, nude, vagina, nipples)"
        else:
             nude_phrase = "(bare chest, wear white boxers)" if sex == "male" else "(wear white bra and panties)"
        positive_prompt += f", {nude_phrase}"
        
        # Age
        positive_prompt = append_age(positive_prompt, age, sex)

        background_color = info.get("background_color", "")
        if background_color:
            positive_prompt += f", {background_color} background"
        
        # Physical Attributes
        for attr in ["race", "hair", "eyes", "face", "body", "skin_color", "additional_details"]:
            val = info.get(attr, "")
            if attr == "hair":
                val = normalize_hair_tags(val)
            if val:
                positive_prompt += f", ({val}:{1.0 if 'skin' in attr else 1.0})"

        # LoRA Trigger
        lora_prompt = info.get("lora_prompt", "")
        if lora_prompt:
            positive_prompt += f", {lora_prompt}"

        # Negative Prompt
        neg = info.get("negative_prompt", "")
        negative_prompt = dedupe_tokens(f"{neg},{gender_negative}")

        return positive_prompt, negative_prompt

    def process(self, widget_data="{}", unique_id=None):
        # Clear Preview Cache to free memory for workflow run
        global PREVIEW_CACHE
        if PREVIEW_CACHE["asset_obj"]:
            # Explicitly delete references to help GC
            del PREVIEW_CACHE["asset_obj"]
            PREVIEW_CACHE["asset_obj"] = None
        PREVIEW_CACHE["asset_key"] = None
        PREVIEW_CACHE["loras"].clear()

        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # Extract values
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        gen_settings = normalize_gen_settings(data.get("gen_settings", {}))
        info_owner = str(info.get("name", "") or "").strip()
        if info_owner and info_owner != str(character_name):
            raise ValueError(
                f"Refusing to save character_info for '{info_owner}' into '{character_name}'. "
                "Reload the character in Character Creator V2 and try again."
            )
        
        # 1. Generate Prompts
        positive_prompt, negative_prompt = self.construct_prompt(info)
        
        # 2. Re-extract local vars for saving / outputs
        face_details = build_face_details(info)
        face_details += ", (expressionless:1.0)"

        # Save Config logic
        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name) # Uses default "Naked", "neutral", "sheet_neutral"
        faces_path = faces_dir(character_name)   # Uses default "Naked", "neutral", "face_neutral"

        ensure_character_structure(character_name)

        config = load_config(character_name) or {
            "character_info": {},
            "folder_structure": {
                "main_directories": MAIN_DIRS,
                "emotions": EMOTIONS
            },
            "character_path": character_path,
            "config_version": "2.0"
        }

        info["name"] = character_name
        info["seed"] = gen_settings.get("seed", 0)
        config["character_info"] = info
        config["character_path"] = character_path
        if "costumes" not in config:
            config["costumes"] = {}
        save_config(character_name, config)


        # 3. Load Models & Construct Pipe
        # ----------------------------------------------------------------
        _, model, clip, vae = load_generation_assets(gen_settings)

        # Helper to apply LoRA
        def apply_lora_safe(m, c, l_name, l_strength, clip_strength=None):
            if not l_name or l_name == "None": return m, c
            l_path = get_lora_full_path(l_name)
            if l_path:
                lora = comfy.utils.load_torch_file(l_path, safe_load=True)
                return comfy.sd.load_lora_for_models(m, c, lora, l_strength, l_strength if clip_strength is None else clip_strength)
            return m, c

        # Apply DMD2
        if gen_settings.get("generation_mode") == "anima":
            if gen_settings.get("turbo_enabled"):
                dmd_name = gen_settings.get("dmd_lora_name")
                dmd_str = float(gen_settings.get("dmd_lora_strength", 1.0))
                model, clip = apply_lora_safe(model, clip, dmd_name, dmd_str, 0.0)

            stack = gen_settings.get("lora_stack", [])
            for item in stack:
                model, clip = apply_lora_safe(model, clip, item.get("name"), float(item.get("strength", 1.0)))
        else:
            dmd_name = gen_settings.get("dmd_lora_name")
            dmd_str = float(gen_settings.get("dmd_lora_strength", 1.0))
            model, clip = apply_lora_safe(model, clip, dmd_name, dmd_str)

            # Apply Age LoRA
            age_name = gen_settings.get("age_lora_name")
            if age_name:
                age = int(info.get("age", 18))
                age_str = age_strength(age)
                model, clip = apply_lora_safe(model, clip, age_name, age_str)

            # Apply Stack
            stack = gen_settings.get("lora_stack", [])
            for item in stack:
                model, clip = apply_lora_safe(model, clip, item.get("name"), float(item.get("strength", 1.0)))

        # Encode Conditioning
        conditioning_pos = encode_generation_prompt(clip, positive_prompt, gen_settings)
        conditioning_neg = encode_generation_prompt(clip, negative_prompt, gen_settings)
        if gen_settings.get("generation_mode") == "anima":
            validate_anima_conditioning(conditioning_pos, conditioning_neg, gen_settings.get("clip_name", ""))

        # Construct Pipe Object
        class PipeContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        pipe = PipeContext(
            model=model,
            clip=clip,
            vae=vae,
            pos=conditioning_pos,
            neg=conditioning_neg,
            seed_int=generate_seed(int(gen_settings.get("seed", 0))),
            sample_steps=int(gen_settings.get("steps", ILLUSTRIOUS_DEFAULTS["steps"])),
            cfg=float(gen_settings.get("cfg", ILLUSTRIOUS_DEFAULTS["cfg"])),
            denoise=1.0,
            sampler_name=gen_settings.get("sampler", ILLUSTRIOUS_DEFAULTS["sampler"]),
            scheduler=gen_settings.get("scheduler", ILLUSTRIOUS_DEFAULTS["scheduler"])
        )

        # 4. Generate Image (Smart Cache Logic)
        
        # Determine Cache Path
        cache_dir = os.path.join(character_dir(character_name), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "preview.png")

        # Check Validity
        preview_valid = data.get("preview_valid", False)
        
        image = None

        # Determine source
        preview_source = "gen"
        if widget_data:
             if isinstance(data, dict):
                # widget_data is a string (JSON), but here it seems passed as 'widget_data' argument which might be raw string
                # logic above parsed it into 'data' dict. use that.
                preview_source = data.get("preview_source", "gen")
             else:
                print(f"[VNCCS] Preview source ignored because widget data is not a dict: {type(data).__name__}")
        
        print(f"[VNCCS] Processing - Source: {preview_source}, Valid: {preview_valid}")

        # LOGIC:
        # 1. If source == 'pose' (User just loaded character), we MUST use a saved pose sprite.
        #    We ignore the existing cache file because it might be stale.
        #    We overwrite the cache with the selected pose preview.
        # 2. If source == 'gen' (User generated previously), we use the cache.
        
        if preview_valid:
             if preview_source == "pose":
                  selected_index = data.get("sprite_preview_index")
                  try:
                      selected_index = int(selected_index)
                  except (TypeError, ValueError):
                      selected_index = None
                  print(f"[VNCCS] Source is Pose. Force-loading selected pose preview index={selected_index}.")
                  image, _selected_index, _count = get_pose_preview(character_name, index=selected_index)
                  if image is not None:
                       print("[VNCCS] Pose preview loaded successfully. Overwriting Cache.")
                       try:
                           c_img = tensor2pil(image)
                           os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                           c_img.save(cache_path)
                       except Exception as exc:
                           print(f"[VNCCS] Failed to save pose preview cache '{cache_path}': {exc}")
                  else:
                       print("[VNCCS] Pose preview load failed. Will try cache/regen.")

             # If image not set yet (source=gen OR pose load failed), try cache
             if image is None and os.path.exists(cache_path):
                 print(f"[VNCCS] Smart Cache Hit: Loading existing preview for '{character_name}'")
                 try:
                     i = Image.open(cache_path)
                     image = pil2tensor(i)
                 except Exception as e:
                     print(f"[VNCCS] Failed to load cache: {e}. Regenerating.")

        if image is None:
             # Fallback: If cache is missing (even if source=gen), try saved pose before regen
             if preview_valid:
                 print(f"[VNCCS] Cache Miss. Attempting Pose Preview Fallback...")
                 image = get_random_pose_preview(character_name)
                 if image is not None:
                     print(f"[VNCCS] Pose Preview Fallback Successful. Updating Cache.")
                     try:
                        c_img = tensor2pil(image)
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        c_img.save(cache_path)
                        # Notify Frontend
                        server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": unique_id, "character": character_name})
                     except Exception as e:
                        print(f"[VNCCS] Failed to save/notify on pose preview fallback: {e}")
                 else:
                     print(f"[VNCCS] Pose Preview Fallback Failed. Regenerating...")

        if image is None:
            print(f"[VNCCS] Regenerating Preview...")
            
            width, height = get_generation_resolution(gen_settings)
            latent = create_generation_latent(model, width, height, gen_settings)
            
            try:
                sampled = sample_generation_latent(
                    model=model,
                    seed=generate_seed(int(gen_settings.get("seed", 0))),
                    steps=int(gen_settings.get("steps", ILLUSTRIOUS_DEFAULTS["steps"])),
                    cfg=float(gen_settings.get("cfg", ILLUSTRIOUS_DEFAULTS["cfg"])),
                    sampler_name=gen_settings.get("sampler", ILLUSTRIOUS_DEFAULTS["sampler"]),
                    scheduler=gen_settings.get("scheduler", ILLUSTRIOUS_DEFAULTS["scheduler"]),
                    positive=conditioning_pos,
                    negative=conditioning_neg,
                    latent=latent,
                    gen_settings=gen_settings,
                )
                
                image = decode_generation_samples(vae, sampled, gen_settings)
                
                # Update Cache
                try:
                    # Tensor [1,H,W,3] -> PIL
                    c_img = tensor2pil(image)
                    c_img.save(cache_path)
                    print(f"[VNCCS] Saved new preview cache to {cache_path}")
                    # Notify Frontend
                    server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": unique_id, "character": character_name})
                except Exception as e:
                    print(f"[VNCCS] Failed to save cache: {e}")

            except Exception as e:
                print(f"[VNCCS] Generation failed in process: {e}")
                image = torch.zeros((1, 512, 512, 3))

        # Get background color
        background_color = info.get("background_color", "Green")

        # Resize full sheet (1024x3072) to single character size (512x1536)
        if torch.is_tensor(image) and image.shape[1] == 3072 and image.shape[2] == 1024:
            pil_img = tensor2pil(image)
            pil_img = pil_img.resize((512, 1536), Image.LANCZOS)
            image = pil2tensor(pil_img)

        return (
            image,
            sheets_path,
            background_color
        )


NODE_CLASS_MAPPINGS = {
    "CharacterCreatorV2": CharacterCreatorV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterCreatorV2": "VNCCS Character Creator V2",
}
