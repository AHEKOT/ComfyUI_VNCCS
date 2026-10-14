from ..operation_logger import log_event, logged_operation

import os
import asyncio
import json
import re
import io
import base64
import torch
import numpy as np
import comfy.sd
import comfy.utils
from PIL import Image, ImageOps

from ..utils import (
    character_dir, list_characters,
    load_character_info,
    apply_sex, append_age, generate_seed, build_face_details,
    list_costumes, load_costume_info, ensure_safe_name, safe_join_under,
    privileged_route, file_fingerprint, config_path, atomic_output_path, character_storage_lock
)
from .emotion_library import load_emotion_library, save_emotion_library
from .runtime_cleanup import inference_stage
from .character_creator_v2 import (
    ANIMA_DEFAULTS,
    ILLUSTRIOUS_DEFAULTS,
    load_anima_assets,
    load_generation_assets,
    normalize_gen_settings,
    get_lora_full_path,
    _generation_file_signatures,
    apply_generation_lora,
)
from .vnccs_control_center import is_qi2_turbo_model
from .vnccs_pipe import VNCCS_Pipe


QI2_TURBO_LORA_NAME = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
QI2_DEFAULTS = {
    "generation_mode": "qi2",
    "diffusion_model_name": "qwen_image_2.1_int8_convrot.safetensors",
    "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
    "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
    "clip_type": "qwen_image",
    "sampler": "euler",
    "scheduler": "simple",
    "steps": 25,
    "cfg": 3.0,
    "turbo_enabled": False,
    "dmd_lora_name": QI2_TURBO_LORA_NAME,
    "dmd_lora_strength": 1.0,
    "lora_stack": [],
    "qi2_cache": {"device": "gpu", "dtype": "int8"},
}
QI2_TURBO_ENTRY = {
    "name": "Qwen Image 2.1 Viggle Turbo",
    "type": "TurboLora",
    "kind": "QI2",
    "local_path": f"models/loras/{QI2_TURBO_LORA_NAME}",
}

# --- ComfyUI Server Imports ---
try:
    import server
    from aiohttp import web
except ImportError:
    log_event("warning", component="EmotionStudio", level="warning", message='VNCCS Warning: Running outside ComfyUI environment. API routes will not be registered.')
    server = None
    web = None

# --------------------------------------------------------------------
# Helper to get emotion config path
def get_custom_node_path():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Helper function to load the emotions JSON
def load_emotions_data():
    return load_emotion_library(emotions_config_path(), user_emotions_config_path())


def emotions_config_path():
    return os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")


def emotion_images_dir():
    return os.path.join(get_custom_node_path(), "emotions-config", "images")


def user_emotions_config_path():
    return safe_join_under(os.path.dirname(emotions_config_path()), "user", "emotions.json")


def user_emotion_images_dir():
    return os.path.join(os.path.dirname(user_emotions_config_path()), "images")


def make_unique_emotion_safe_name(title, existing_names):
    base = re.sub(r"[^a-z0-9]+", "-", str(title or "").lower()).strip("-")[:120].rstrip("-")
    if not base:
        base = "custom-emotion"

    safe_name = base
    counter = 2
    # Neutral directories contain source poses, never generated emotions.
    existing = set(existing_names or []) | {"neutral"}
    while safe_name in existing:
        suffix = f"-{counter}"
        safe_name = base[:120 - len(suffix)].rstrip("-") + suffix
        counter += 1
    return ensure_safe_name(safe_name, "emotion")


MAX_CUSTOM_EMOTION_PIXELS = 16 * 1024 * 1024


def save_custom_emotion_image(image_data, safe_name):
    if not image_data:
        return

    if not isinstance(image_data, str) or not image_data.startswith("data:image/"):
        raise ValueError("Image must be a data:image URL.")

    try:
        _header, encoded = image_data.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid image data URL.") from exc

    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 12 * 1024 * 1024:
        raise ValueError("Image is too large.")

    with Image.open(io.BytesIO(raw)) as img:
        if img.width * img.height > MAX_CUSTOM_EMOTION_PIXELS:
            raise ValueError(f"Emotion image exceeds {MAX_CUSTOM_EMOTION_PIXELS:,} pixels. Use a smaller image.")
        img = ImageOps.exif_transpose(img).convert("RGBA")
        path = safe_join_under(user_emotion_images_dir(), f"{ensure_safe_name(safe_name, 'emotion')}.webp")
        with atomic_output_path(path) as temporary:
            img.save(temporary, format="WEBP", quality=92, method=6)


def _save_custom_emotion(payload):
    title = str(payload.get("name", "") or "").strip()
    if not title:
        raise ValueError("Emotion name is required.")
    description = str(payload.get("description", "") or "").strip()
    natural_prompt = str(payload.get("natural_prompt", "") or "").strip() or f"The character expresses {title}."

    config_path = user_emotions_config_path()
    with character_storage_lock(config_path):
        data = load_emotions_data()
        existing_names = {emotion["safe_name"] for entries in data.values() for emotion in entries}
        safe_name = make_unique_emotion_safe_name(title, existing_names)
        save_custom_emotion_image(payload.get("image_data"), safe_name)
        emotion = {"key": title, "description": description, "safe_name": safe_name,
                   "natural_prompt": natural_prompt}
        data.setdefault("Custom", []).append(emotion)
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                users = json.load(handle)
        except FileNotFoundError:
            users = {}
        users["Custom"] = data["Custom"]
        save_emotion_library(config_path, users)
    return {**emotion, "category": "Custom"}


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def default_generation_settings():
    settings = dict(ANIMA_DEFAULTS)
    settings["generation_mode"] = "anima"
    settings.setdefault("seed", 0)
    settings.setdefault("seed_mode", "fixed")
    settings.setdefault("mode_settings", {
        "illustrious": dict(ILLUSTRIOUS_DEFAULTS),
        "anima": dict(ANIMA_DEFAULTS),
        "qi2": dict(QI2_DEFAULTS),
    })
    return settings


def resolve_generation_seed(gen_settings, seed_resolved=False):
    seed = int(gen_settings.get("seed", 0) or 0)
    if str(gen_settings.get("seed_mode", "fixed") or "fixed").lower() == "randomize" and not seed_resolved:
        return generate_seed(0)
    return seed


@inference_stage()
def build_emotion_pipe(generation_model="Anima", generation_settings="{}"):
    parsed = json.loads(generation_settings) if generation_settings else {}
    if not isinstance(parsed, dict):
        raise ValueError("Emotion generation settings must be an object")
    profiles = parsed.get("mode_settings", {})
    if not isinstance(profiles, dict):
        raise ValueError("Emotion model profiles must be an object")
    for settings in [parsed, *profiles.values()]:
        if not isinstance(settings, dict):
            raise ValueError("Emotion model profiles must contain objects")
        if "qi2_cache" in settings and not isinstance(settings["qi2_cache"], dict):
            raise ValueError("Emotion QI2 cache settings must be an object")
        previous = settings.get("turbo_previous_settings")
        if previous is not None and not isinstance(previous, dict):
            raise ValueError("Emotion previous turbo settings must be an object or null")
        stack = settings.get("lora_stack", [])
        if not isinstance(stack, list) or any(not isinstance(item, dict) for item in stack):
            raise ValueError("Emotion LoRA settings must be objects in a list")
    seed_resolved = parsed.pop("_vnccs_seed_resolved", False) is True

    mode = str(generation_model or parsed.get("generation_mode") or "Anima").lower()
    if mode not in ("illustrious", "anima", "qi2"):
        mode = "anima"

    if mode == "qi2":
        gen_settings = dict(QI2_DEFAULTS)
        mode_settings = parsed.get("mode_settings", {}) if isinstance(parsed, dict) else {}
        mode_profile = mode_settings.get("qi2", {}) if isinstance(mode_settings, dict) else {}
        selected_model = parsed.get("diffusion_model_name") if isinstance(parsed, dict) else ""
        if isinstance(mode_profile, dict):
            selected_model = mode_profile.get("diffusion_model_name", selected_model)
        if is_qi2_turbo_model(selected_model):
            gen_settings.update(steps=8, cfg=1.0)
        if isinstance(parsed, dict):
            gen_settings.update(parsed)
        if isinstance(mode_profile, dict):
            gen_settings.update(mode_profile)
        gen_settings["generation_mode"] = "qi2"
        gen_settings["clip_type"] = "qwen_image"
        if is_qi2_turbo_model(gen_settings.get("diffusion_model_name")):
            gen_settings["turbo_enabled"] = False
        if gen_settings.get("turbo_enabled"):
            gen_settings["steps"] = 6
            gen_settings["cfg"] = 1.0
        model, clip, vae = load_anima_assets(gen_settings)
    else:
        gen_settings = normalize_gen_settings({**parsed, "generation_mode": mode})
        _, model, clip, vae = load_generation_assets(gen_settings)

    apply_lora_safe = apply_generation_lora

    if mode == "anima":
        if gen_settings.get("turbo_enabled"):
            model, clip = apply_lora_safe(
                model,
                clip,
                gen_settings.get("dmd_lora_name"),
                gen_settings.get("dmd_lora_strength", 1.0),
                0.0,
            )
        for item in gen_settings.get("lora_stack", []) or []:
            if isinstance(item, dict):
                model, clip = apply_lora_safe(model, clip, item.get("name"), item.get("strength", 1.0))
    elif mode == "illustrious":
        dmd_name = gen_settings.get("dmd_lora_name")
        if dmd_name:
            model, clip = apply_lora_safe(model, clip, dmd_name, gen_settings.get("dmd_lora_strength", 1.0))
        for item in gen_settings.get("lora_stack", []) or []:
            if isinstance(item, dict):
                model, clip = apply_lora_safe(model, clip, item.get("name"), item.get("strength", 1.0))
    else:
        # Viggle uses VNCCS' unmerged adapter and latent-aware sigma schedule in
        # VNCCS Emotions Generator.  Only ordinary user LoRAs are merged here.
        for item in gen_settings.get("lora_stack", []) or []:
            if isinstance(item, dict):
                model, clip = apply_lora_safe(model, clip, item.get("name"), item.get("strength", 1.0))

    # Seed controls are shared between model profiles; the selected UI seed wins.
    for key in ("seed", "seed_mode"):
        if key in parsed:
            gen_settings[key] = parsed[key]
    seed = resolve_generation_seed(gen_settings, seed_resolved=seed_resolved)
    pipe = VNCCS_Pipe().process_pipe(
        model=model, clip=clip, vae=vae, seed_int=seed,
        sample_steps=int(gen_settings.get("steps", 0) or 0),
        cfg=float(gen_settings.get("cfg", 0.0) or 0.0), denoise=1.0,
        sampler_name=gen_settings.get("sampler"),
        scheduler=gen_settings.get("scheduler"),
    )[9]
    if mode == "anima":
        pipe.model_entry = {
            "name": "Anima",
            "kind": "Anima",
            "local_path": gen_settings.get("diffusion_model_name", ""),
        }
    elif mode == "qi2":
        cache = gen_settings.get("qi2_cache", {})
        cache = cache if isinstance(cache, dict) else {}
        pipe.model_entry = {
            "name": "Qwen Image 2.1",
            "type": "unet",
            "kind": "QI2",
            "local_path": f"models/diffusion_models/{gen_settings.get('diffusion_model_name', '')}",
        }
        pipe.model_kind = "qi2"
        pipe.qi2_cache = {
            "device": cache.get("device") if cache.get("device") in {"auto", "gpu", "cpu", "off"} else "gpu",
            "dtype": cache.get("dtype") if cache.get("dtype") in {"default", "int8", "int4"} else "int8",
        }
        pipe.lora_entries = [dict(QI2_TURBO_ENTRY)]
        pipe.lora_states = [{
            "name": QI2_TURBO_ENTRY["name"],
            "auto_apply": bool(gen_settings.get("turbo_enabled")),
            "strength": float(gen_settings.get("dmd_lora_strength", 1.0) or 1.0),
        }]
    return pipe, seed


def build_anima_emotion_prompt(natural_prompt, emotion_description, fallback_emotion):
    emotion_text = str(natural_prompt or "").strip()
    if not emotion_text:
        emotion_text = f"The character expresses {str(fallback_emotion or '').replace('-', ' ')}."

    emotion_tags = str(emotion_description or "").strip()
    if emotion_tags:
        emotion_text = f"{emotion_text}\n\nEmotion Tags: {emotion_tags}"
    return emotion_text


def _load_sprite_tensor(path):
    with Image.open(path) as opened:
        img = ImageOps.exif_transpose(opened)
        has_alpha = img.mode == "RGBA" or img.mode == "LA" or (img.mode == "P" and "transparency" in img.info)
        rgb_arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        alpha_arr = np.asarray(img.convert("RGBA").getchannel("A"), dtype=np.uint8).copy() if has_alpha else None
    image = torch.from_numpy(rgb_arr).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
    mask = (
        1.0 - torch.from_numpy(alpha_arr).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
        if alpha_arr is not None
        else None
    )
    return image, mask


def list_costume_sprite_paths(character, costume):
    """Return sorted current neutral/source sprite paths without decoding them."""
    costume = ensure_safe_name(costume, "costume")
    character_path = character_dir(character)
    try:
        root = safe_join_under(character_path, "Sprites", costume)
    except ValueError:
        return []
    paths = []
    if os.path.isdir(root):
        neutral_paths = []
        direct_paths = []
        seen_roots = set()
        for subfolder in ("Neutral", "neutral", ""):
            try:
                source_root = safe_join_under(root, subfolder) if subfolder else root
                if not os.path.isdir(source_root):
                    continue
                directory_stat = os.stat(source_root)
                names = os.listdir(source_root)
            except (ValueError, OSError):
                continue
            directory_key = (directory_stat.st_dev, directory_stat.st_ino)
            if directory_key in seen_roots:
                continue
            seen_roots.add(directory_key)
            for name in names:
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                    continue
                try:
                    path = safe_join_under(root, subfolder, name) if subfolder else safe_join_under(root, name)
                except ValueError:
                    continue
                if os.path.isfile(path):
                    (neutral_paths if subfolder else direct_paths).append(path)
        seen_paths = set()
        paths = []
        for path in sorted(neutral_paths or direct_paths):
            path_key = os.path.normcase(os.path.abspath(path))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            paths.append(path)
    return paths


def load_costume_sprite_images(character, costume, selected_pose_indices=None, *, first_only=False):
    """Decode only the requested current neutral/source sprites for a costume."""
    paths = list_costume_sprite_paths(character, costume)
    selected = set(selected_pose_indices) if selected_pose_indices is not None else None

    loaded = []
    for pose_index, path in enumerate(paths, start=1):
        if selected is not None and pose_index not in selected:
            continue
        try:
            image, mask = _load_sprite_tensor(path)
            loaded.append((image, mask, path))
            if first_only:
                return loaded
        except Exception as exc:
            log_event('warning', component='EmotionStudio', level='warning', message=f'Failed to load sprite {path}: {exc}', error=str(exc))

    if loaded:
        return loaded

    log_event("warning", component="EmotionStudio", level="warning", message=f'No sprites found for {character}/{costume}. Generate sprites first.')
    return []


def costume_has_source_sprites(character, costume):
    return bool(list_costume_sprite_paths(character, costume))


def costume_face_details(character, costume):
    costume_info = load_costume_info(character, costume) or {}
    parts = []
    face = str(costume_info.get("face", "") or "").strip()
    head = str(costume_info.get("head", "") or "").strip()
    if face:
        parts.append(f"wear {face} on face")
    if head:
        parts.append(f"wear {head} on head")
    return ", ".join(parts), face, head

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

if server:
    @server.PromptServer.instance.routes.get("/vnccs/get_emotions")
    async def get_emotions_config(request):
        try:
            data = await asyncio.to_thread(load_emotions_data)
            return web.json_response(data)
        except Exception as e:
            return web.Response(status=500, text=f"Error loading emotions.json: {e}")

    @server.PromptServer.instance.routes.post("/vnccs/add_custom_emotion")
    @privileged_route
    async def add_custom_emotion(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Emotion request must be an object")
            emotion = await asyncio.to_thread(_save_custom_emotion, payload)

            try:
                EmotionGeneratorV2.SAFE_NAME_MAP = None
                EmotionGeneratorV2.EMOTIONS_DATA = None
            except NameError:
                pass

            return web.json_response({"emotion": emotion})
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            log_event('warning', component='EmotionStudio', level='warning', message=f'Failed to add custom emotion: {e}', error=str(e))
            return web.json_response({"error": f"Failed to add custom emotion: {e}"}, status=500)

    @server.PromptServer.instance.routes.get("/vnccs/get_character_costumes")
    async def get_character_costumes(request):
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response([])
        
        try:
            costumes = [
                costume for costume in list_costumes(character)
                if costume_has_source_sprites(character, costume)
            ]
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(costumes)

    @server.PromptServer.instance.routes.get("/vnccs/get_character_sheet_preview")
    async def get_character_sheet_preview(request):
        character = request.rel_url.query.get("character", "")
        if not character:
             return web.Response(status=404)

        try:
            costume = request.rel_url.query.get("costume", "Naked")
            sprites = (load_costume_sprite_images(character, costume, first_only=True)
                       or load_costume_sprite_images(character, "Original", first_only=True))
            if not sprites:
                return web.Response(status=404, text="No sprites found. Generate sprites first.")

            image, _mask, _path = sprites[0]
            tensor = image[0].detach().cpu().clamp(0, 1)
            arr = (tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return web.Response(body=img_byte_arr.getvalue(), content_type='image/png')
        except ValueError as e:
            return web.Response(status=400, text=str(e))
        except Exception as e:
            log_event('warning', component='EmotionStudio', level='warning', message=f'Failed to serve sprite preview: {e}', error=str(e))
            return web.Response(status=500)

    @server.PromptServer.instance.routes.get("/vnccs/get_emotion_image")
    async def get_emotion_image(request):
        try:
            name = ensure_safe_name(request.rel_url.query.get("name", ""), "emotion")
            image_path = safe_join_under(emotion_images_dir(), f"{name}.webp")
            user_image_path = safe_join_under(user_emotion_images_dir(), f"{name}.webp")
            if os.path.exists(user_image_path):
                image_path = user_image_path
        except ValueError as error:
            return web.Response(status=400, text=str(error))

        if not os.path.exists(image_path):
            return web.Response(status=404)

        return web.FileResponse(image_path)


class EmotionGeneratorV2:
    
    EMOTIONS_DATA = None
    SAFE_NAME_MAP = None
    EMOTIONS_FINGERPRINT = None

    def __init__(self):
        self._setup_emotions_data()

    @classmethod
    def _setup_emotions_data(cls):
        path = emotions_config_path()
        fingerprint = (file_fingerprint(path), file_fingerprint(user_emotions_config_path()))
        cached = cls.SAFE_NAME_MAP
        if cached is not None and cls.EMOTIONS_FINGERPRINT == fingerprint:
            return cached

        try:
            data = load_emotions_data()

            safe_name_map = {}
            for category, emotion_list in data.items():
                for emotion in emotion_list:
                    if 'safe_name' in emotion and emotion['safe_name']:
                        safe_name_map[emotion['safe_name']] = {
                                "key": emotion['key'],
                                "description": emotion['description'],
                                "natural_prompt": emotion.get('natural_prompt', ''),
                                "category": category
                        }
        except Exception as e:
            log_event('warning', component='EmotionStudio', level='warning', message=f'ERROR: Failed to load emotions data: {e}', error=str(e))
            raise ValueError(f"Unable to load emotion catalog: {e}. Repair it before generating emotions.") from e
        cls.SAFE_NAME_MAP = safe_name_map
        # A catalog write during loading must invalidate this older snapshot.
        cls.EMOTIONS_FINGERPRINT = fingerprint
        return safe_name_map

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        if not characters:
            characters = ["Character Name"]

        return {
            "required": {
                "generation_model": (["Illustrious", "Anima", "QI2"], {"default": "Anima"}),
                "generation_settings": ("STRING", {"default": json.dumps(default_generation_settings()), "multiline": False}),
                "prompt_style": (["SDXL Style", "Anima"], {"default": "Anima"}),
                "character": (characters, {"default": characters[0] if characters else "Character Name"}),
                # JSON lists passed as strings from frontend
                "costumes_data": ("STRING", {"default": "[]", "multiline": False}),
                "emotions_data": ("STRING", {"default": "[]", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VNCCS_PIPE", "STRING")
    RETURN_NAMES = ("images", "pipe", "emotion_data")
    OUTPUT_IS_LIST = (True, False, True)
    FUNCTION = "generate_emotions_v2"
    CATEGORY = "VNCCS"

    @classmethod
    def IS_CHANGED(cls, character="Character Name", costumes_data="[]", **kwargs):
        settings = json.loads(kwargs.get("generation_settings") or "{}")
        mode = str(kwargs.get("generation_model") or settings.get("generation_mode") or "Anima").lower()
        profiles = settings.get("mode_settings", {})
        profile = profiles.get(mode, {}) if isinstance(profiles, dict) else {}
        seed_mode = settings.get("seed_mode", profile.get("seed_mode", "fixed") if isinstance(profile, dict) else "fixed")
        if str(seed_mode or "fixed").lower() == "randomize" and settings.get("_vnccs_seed_resolved") is not True:
            return float("nan")
        paths = [config_path(character), emotions_config_path(), user_emotions_config_path()]
        for costume in json.loads(costumes_data):
            paths.extend(list_costume_sprite_paths(character, costume))
        assets = _generation_file_signatures(normalize_gen_settings({**settings, "generation_mode": mode}))
        return json.dumps([[file_fingerprint(path) for path in paths], assets])

    @logged_operation("EmotionStudio", "prepare_tasks")
    def generate_emotions_v2(self, generation_model="Anima", generation_settings="{}", prompt_style="Anima", character="Character Name", costumes_data="[]", emotions_data="[]"):
        info = load_character_info(character)
        if not isinstance(info, dict) or not info:
            raise ValueError(f"Character profile missing for '{character}'. Save the character in Creator or Cloner before generating emotions.")
        selected_costumes = json.loads(costumes_data)
        if not isinstance(selected_costumes, list):
            raise ValueError("Selected costumes must be a list")
        for costume in selected_costumes:
            ensure_safe_name(costume, "costume")
            load_costume_info(character, costume)
        selected_emotions = json.loads(emotions_data)
        if not isinstance(selected_emotions, list):
            raise ValueError("Selected emotions must be a list")
        for emotion in selected_emotions:
            if ensure_safe_name(emotion, "emotion").casefold() == "neutral":
                raise ValueError("Emotion name 'neutral' is reserved for source poses. Add this emotion again with a different name.")
        emotion_map = self._setup_emotions_data()
        for emotion in selected_emotions:
            if emotion not in emotion_map:
                raise ValueError(f"Selected emotion '{emotion}' is not available. Restore its catalog entry or deselect it before generating emotions.")
        generation_settings_data = json.loads(generation_settings) if generation_settings else {}
        if not isinstance(generation_settings_data, dict):
            raise ValueError("Emotion generation settings must be an object")
        raw_pose_indices = generation_settings_data.get("selected_pose_indices")
        selected_pose_indices = None
        if raw_pose_indices is not None:
            if not isinstance(raw_pose_indices, list):
                raise ValueError("Selected pose indices must be positive integers in a list.")
            selected_pose_indices = set()
            for value in raw_pose_indices:
                try:
                    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                        raise ValueError
                    if isinstance(value, str) and not re.fullmatch(r"[+-]?[0-9](?:_?[0-9])*", value.strip()):
                        raise ValueError
                    index = int(value)
                    if index <= 0 or isinstance(value, float) and value != index:
                        raise ValueError
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError("Selected pose indices must be positive integers in a list.") from error
                selected_pose_indices.add(index)
        pipe, pipe_seed = build_emotion_pipe(generation_model, generation_settings)
        mode = str(generation_model or "Anima").lower()
        effective_prompt_style = "Anima" if mode == "anima" else "SDXL Style"
        
        selected_costumes = [
            costume for costume in selected_costumes
            if costume_has_source_sprites(character, costume)
        ]

        # --- SETUP ---
        character_path = character_dir(character)
        images = []
        emotion_data = []
        
        # Helper to get info
        if info:
            aesthetics = info.get("aesthetics", "")
            background_color = info.get("background_color", "blue")
            sex = info.get("sex", "")
            age = info.get("age", 18)
            race = info.get("race", "")
            eyes = info.get("eyes", "")
            hair = info.get("hair", "")
            face_features = info.get("face", "")
            body = info.get("body", "")
            skin_color = info.get("skin_color", "")
            additional_details = info.get("additional_details", "")
            lora_prompt = info.get("lora_prompt", "")
            
            negative_prompt = info.get("negative_prompt", "") 
            negative_prompt = negative_prompt + ", (facial droplet), (water drop), (water), (water droplets), (water drops)"
            
            seed = pipe_seed
            base_negative_prompt = negative_prompt
            positive_prompt = aesthetics
        else:
             seed = pipe_seed
             base_negative_prompt = ""
             positive_prompt = ""
             log_event("warning", component="EmotionStudio", level="warning", message=f'Character info not found for {character}')

        # --- GENERATION LOOP ---
        for costume in selected_costumes:
            # Check if valid costume dir
            # Note: selected_costumes comes from frontend which lists them via API
            costume_details, costume_face, costume_head = costume_face_details(character, costume)
            
            sprite_paths = list_costume_sprite_paths(character, costume)
            if not sprite_paths:
                log_event("warning", component="EmotionStudio", level="warning", message=f'Failed to load sprites for costume {costume}')
                continue
            selected_sprite_paths = [
                (sprite_index, source_path)
                for sprite_index, source_path in enumerate(sprite_paths, start=1)
                if selected_pose_indices is None or sprite_index in selected_pose_indices
            ]
            if not selected_sprite_paths:
                log_event("warning", component="EmotionStudio", level="warning", message=f'No selected source poses remain for costume {costume}')
                continue

            for emotion_key in selected_emotions:
                
                emotion_details_data = emotion_map[emotion_key]
                emotion_description = emotion_details_data['description']
                natural_prompt = emotion_details_data.get('natural_prompt', '')

                # Path construction
                sprite_dir = os.path.join(character_path, "Sprites", costume, emotion_key)
                
                sheet_output_path = os.path.join(sprite_dir, f"sprite_{emotion_key}_")
                
                # Prompt Building (Same as V1)
                positive_prompt = f"{aesthetics}"
                if background_color: positive_prompt += f", {background_color} background"
                if race: positive_prompt += f", {race} race"
                if hair: positive_prompt += f", {hair} hair"
                if eyes: positive_prompt += f", {eyes} eyes"
                if body: positive_prompt += f", {body} body"
                if skin_color: positive_prompt += f", {skin_color} skin"
                if additional_details: positive_prompt += f", ({additional_details})"
                
                positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
                negative_prompt = f"{base_negative_prompt}, {gender_negative}"
                positive_prompt = append_age(positive_prompt, age, sex)
                
                if lora_prompt: positive_prompt += f", {lora_prompt}"

                face_details = build_face_details(info)
                if costume_details:
                    face_details = f"{face_details}, {costume_details}" if face_details else costume_details
                
                if mode == "qi2":
                    emotion_text = build_anima_emotion_prompt(
                        natural_prompt,
                        emotion_description,
                        emotion_key,
                    )
                elif effective_prompt_style == "Anima":
                    if face_details:
                        positive_prompt += f", Character face details: {face_details}"
                    emotion_text = build_anima_emotion_prompt(natural_prompt, emotion_description, emotion_key)
                else:
                    # SDXL Style (Original logic)
                    emotion_text = f"({emotion_key}, {emotion_description}), {face_details}"
                
                for sprite_index, _source_path in selected_sprite_paths:
                    emotion_data.append(json.dumps({
                        "emotion_prompt": emotion_text,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed,
                        "sprite_output_path": sheet_output_path,
                        "source_path": _source_path,
                        "character": character,
                        "background_color": background_color,
                        "costume": costume,
                        "costume_face": costume_face,
                        "costume_head": costume_head,
                        "face_details": face_details,
                        "emotion": emotion_key,
                        "sprite_index": sprite_index,
                    }, ensure_ascii=False))

        # Source sprites are loaded lazily by VNCCS_EmotionsGenerator. Returning
        # full-resolution tensors here used to decode every pose before the pose
        # selection was applied and duplicated them once per selected emotion.
        log_event("tasks_prepared", character=character, costumes=len(selected_costumes),
                  emotions=len(selected_emotions), tasks=len(emotion_data))
        return [], pipe, emotion_data

NODE_CLASS_MAPPINGS = {
    "EmotionGeneratorV2": EmotionGeneratorV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionGeneratorV2": "VNCCS Emotion Studio"
}
