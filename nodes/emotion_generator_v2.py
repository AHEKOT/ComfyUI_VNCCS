
import os
import json
import re
import io
import torch
import numpy as np
import comfy.sd
import comfy.utils
from PIL import Image, ImageOps

from ..utils import (
    character_dir, list_characters,
    load_character_info,
    apply_sex, append_age, generate_seed, build_face_details,
    list_costumes, load_costume_info
)
from .character_creator_v2 import (
    ANIMA_DEFAULTS,
    ILLUSTRIOUS_DEFAULTS,
    load_generation_assets,
    normalize_gen_settings,
    get_lora_full_path,
)
from .vnccs_pipe import VNCCS_Pipe

# --- ComfyUI Server Imports ---
try:
    import server
    from aiohttp import web
except ImportError:
    print("VNCCS Warning: Running outside ComfyUI environment. API routes will not be registered.")
    server = None
    web = None

# --------------------------------------------------------------------
# Helper to get emotion config path
def get_custom_node_path():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Helper function to load the emotions JSON
def load_emotions_data():
    """Load emotions.json from the emotions-config folder."""
    config_path = os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"emotions.json not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def default_generation_settings():
    settings = dict(ANIMA_DEFAULTS)
    settings["generation_mode"] = "anima"
    settings.setdefault("seed", 0)
    settings.setdefault("seed_mode", "fixed")
    settings.setdefault("mode_settings", {
        "illustrious": dict(ILLUSTRIOUS_DEFAULTS),
        "anima": dict(ANIMA_DEFAULTS),
    })
    return settings


def build_emotion_pipe(generation_model="Anima", generation_settings="{}"):
    try:
        parsed = json.loads(generation_settings) if generation_settings else {}
    except Exception:
        parsed = {}

    mode = str(generation_model or parsed.get("generation_mode") or "Anima").lower()
    if mode not in ("illustrious", "anima"):
        mode = "anima"

    merged = default_generation_settings()
    if isinstance(parsed, dict):
        merged.update(parsed)
    merged["generation_mode"] = mode
    gen_settings = normalize_gen_settings(merged)

    _, model, clip, vae = load_generation_assets(gen_settings)

    def apply_lora_safe(m, c, lora_name, strength, clip_strength=None):
        if not lora_name or lora_name == "None" or float(strength or 0) == 0:
            return m, c
        lora_path = get_lora_full_path(lora_name)
        if not lora_path:
            print(f"[VNCCS Emotion Studio] LoRA not found: {lora_name}")
            return m, c
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return comfy.sd.load_lora_for_models(
            m,
            c,
            lora,
            float(strength),
            float(strength) if clip_strength is None else float(clip_strength),
        )

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
    else:
        dmd_name = gen_settings.get("dmd_lora_name")
        if dmd_name:
            model, clip = apply_lora_safe(model, clip, dmd_name, gen_settings.get("dmd_lora_strength", 1.0))
        for item in gen_settings.get("lora_stack", []) or []:
            if isinstance(item, dict):
                model, clip = apply_lora_safe(model, clip, item.get("name"), item.get("strength", 1.0))

    seed = generate_seed(int(gen_settings.get("seed", 0) or 0))
    pipe_node = VNCCS_Pipe()
    pipe_result = pipe_node.process_pipe(
        model=model,
        clip=clip,
        vae=vae,
        seed_int=seed,
        sample_steps=int(gen_settings.get("steps", 0) or 0),
        cfg=float(gen_settings.get("cfg", 0.0) or 0.0),
        denoise=1.0,
        sampler_name=gen_settings.get("sampler"),
        scheduler=gen_settings.get("scheduler"),
        lora_name="none",
        lora_strength=1.0,
    )
    return pipe_result[9], seed


def _load_sprite_tensor(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode == "RGBA" or img.mode == "LA" or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    image = torch.from_numpy(arr[..., :3]).unsqueeze(0)
    mask = torch.from_numpy(1.0 - arr[..., 3]).unsqueeze(0) if has_alpha else None
    return image, mask


def load_costume_sprite_images(character, costume):
    """Return all current neutral/source sprites for a costume."""
    root = os.path.join(character_dir(character), "Sprites", costume)
    paths = []
    if os.path.isdir(root):
        neutral_paths = []
        seen_neutral_roots = set()
        for neutral_name in ("Neutral", "neutral"):
            neutral_root = os.path.join(root, neutral_name)
            if not os.path.isdir(neutral_root):
                continue
            neutral_key = os.path.normcase(os.path.abspath(neutral_root))
            if neutral_key in seen_neutral_roots:
                continue
            seen_neutral_roots.add(neutral_key)
            neutral_paths.extend(
                os.path.join(neutral_root, name)
                for name in os.listdir(neutral_root)
                if os.path.isfile(os.path.join(neutral_root, name))
                and os.path.splitext(name)[1].lower() in IMAGE_EXTS
            )
        direct = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isfile(os.path.join(root, name)) and os.path.splitext(name)[1].lower() in IMAGE_EXTS
        ]
        seen_paths = set()
        paths = []
        for path in sorted(neutral_paths or direct):
            path_key = os.path.normcase(os.path.abspath(path))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            paths.append(path)

    loaded = []
    for path in paths:
        try:
            image, mask = _load_sprite_tensor(path)
            loaded.append((image, mask, path))
        except Exception as exc:
            print(f"[VNCCS Emotion Studio] Failed to load sprite {path}: {exc}")

    if loaded:
        return loaded

    print(f"[VNCCS Emotion Studio] No sprites found for {character}/{costume}. Run migration or generate sprites first.")
    return []


def costume_face_details(character, costume):
    costume_info = load_costume_info(character, costume) or {}
    parts = []
    face = str(costume_info.get("face", "") or "").strip()
    head = str(costume_info.get("head", "") or "").strip()
    if face:
        parts.append(f"(wear {face} on face:1.0)")
    if head:
        parts.append(f"(wear {head} on head:1.0)")
    return ", ".join(parts), face, head

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

if server:
    @server.PromptServer.instance.routes.get("/vnccs/get_emotions")
    async def get_emotions_config(request):
        try:
            data = load_emotions_data()
            return web.json_response(data)
        except Exception as e:
            return web.Response(status=500, text=f"Error loading emotions.json: {e}")

    @server.PromptServer.instance.routes.get("/vnccs/get_character_costumes")
    async def get_character_costumes(request):
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response([])
        
        costumes = list_costumes(character)
        return web.json_response(costumes)

    @server.PromptServer.instance.routes.get("/vnccs/get_character_sheet_preview")
    async def get_character_sheet_preview(request):
        character = request.rel_url.query.get("character", "")
        if not character:
             return web.Response(status=404)

        try:
            costume = request.rel_url.query.get("costume", "Naked")
            sprites = load_costume_sprite_images(character, costume) or load_costume_sprite_images(character, "Original")
            if not sprites:
                return web.Response(status=404, text="No sprites found. Run migration or generate sprites first.")

            image, _mask, _path = sprites[0]
            tensor = image[0].detach().cpu().clamp(0, 1)
            arr = (tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return web.Response(body=img_byte_arr.getvalue(), content_type='image/png')
        except Exception as e:
            print(f"[VNCCS Emotion Studio] Failed to serve sprite preview: {e}")
            return web.Response(status=500)

    @server.PromptServer.instance.routes.get("/vnccs/get_emotion_image")
    async def get_emotion_image(request):
        name = request.rel_url.query.get("name", "")
        if not name or ".." in name or "/" in name or "\\" in name:
            return web.Response(status=400)
            
        from urllib.parse import unquote
        name = unquote(name).strip() 
        
        # Absolute path resolution logic
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        image_path = os.path.join(root_dir, "emotions-config", "images", f"{name}.png")
        
        if not os.path.exists(image_path):
            return web.Response(status=404)
            
        return web.FileResponse(image_path)


class EmotionGeneratorV2:
    
    EMOTIONS_DATA = None
    SAFE_NAME_MAP = None

    def __init__(self):
        self._setup_emotions_data()

    @classmethod
    def _setup_emotions_data(cls):
        if cls.SAFE_NAME_MAP is not None:
            return

        try:
            config_path = os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

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
            cls.SAFE_NAME_MAP = safe_name_map
        except Exception as e:
            print(f"[VNCCS] ERROR: Failed to load emotions data: {e}")
            cls.SAFE_NAME_MAP = {}

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        if not characters:
            characters = ["Character Name"]

        return {
            "required": {
                "generation_model": (["Illustrious", "Anima"], {"default": "Anima"}),
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

    def generate_emotions_v2(self, generation_model="Anima", generation_settings="{}", prompt_style="Anima", character="Character Name", costumes_data="[]", emotions_data="[]"):
        pipe, pipe_seed = build_emotion_pipe(generation_model, generation_settings)
        
        try:
            selected_costumes = json.loads(costumes_data)
        except:
            selected_costumes = []

        try:
            selected_emotions = json.loads(emotions_data)
        except:
            selected_emotions = []

        # --- SETUP ---
        if self.SAFE_NAME_MAP is None:
            self._setup_emotions_data()
            
        character_path = character_dir(character)
        info = load_character_info(character)
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
            
            config_seed = info.get("seed", 0)
            seed = pipe_seed or generate_seed(config_seed)
            base_negative_prompt = negative_prompt
            positive_prompt = aesthetics
        else:
             seed = pipe_seed or 0
             base_negative_prompt = ""
             positive_prompt = ""
             print(f"Character info not found for {character}")

        # --- GENERATION LOOP ---
        for costume in selected_costumes:
            # Check if valid costume dir
            # Note: selected_costumes comes from frontend which lists them via API
            costume_details, costume_face, costume_head = costume_face_details(character, costume)
            
            sprite_items = load_costume_sprite_images(character, costume)
            if not sprite_items:
                print(f"Failed to load sprites for costume {costume}")
                continue

            for emotion_key in selected_emotions:
                
                emotion_details_data = self.SAFE_NAME_MAP.get(emotion_key)
                if not emotion_details_data:
                    print(f"Warning: Unknown emotion key {emotion_key}")
                    emotion_description = "unknown emotion"
                    natural_prompt = ""
                else:
                    emotion_description = emotion_details_data['description']
                    natural_prompt = emotion_details_data.get('natural_prompt', '')

                # Path construction
                sprite_dir = os.path.join(character_path, "Sprites", costume, emotion_key)
                
                sheet_output_path = os.path.join(sprite_dir, f"sprite_{emotion_key}_")
                
                # Prompt Building (Same as V1)
                positive_prompt = f"{aesthetics}"
                if background_color: positive_prompt += f", {background_color} background"
                if race: positive_prompt += f", ({race} race:1.0)"
                if hair: positive_prompt += f", ({hair} hair:1.0)"
                if eyes: positive_prompt += f", ({eyes} eyes:1.0)"
                if body: positive_prompt += f", ({body} body:1.0)"
                if skin_color: positive_prompt += f", ({skin_color} skin:1.0)"
                if additional_details: positive_prompt += f", ({additional_details})"
                
                positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
                negative_prompt = f"{base_negative_prompt}, {gender_negative}"
                positive_prompt = append_age(positive_prompt, age, sex)
                
                if lora_prompt: positive_prompt += f", {lora_prompt}"

                face_details = build_face_details(info)
                if costume_details:
                    face_details = f"{face_details}, {costume_details}" if face_details else costume_details
                
                if prompt_style in ("Anima", "QWEN Style"):
                    if face_details:
                        positive_prompt += f", Character face details: {face_details}"
                    emotion_text = natural_prompt.strip() or f"The character expresses {emotion_key.replace('-', ' ')}."
                else:
                    # SDXL Style (Original logic)
                    emotion_text = f"({emotion_key}, {emotion_description}), {face_details}"
                
                for sprite_index, (img_tensor, mask_tensor, _source_path) in enumerate(sprite_items, start=1):
                    images.append(img_tensor)
                    emotion_data.append(json.dumps({
                        "emotion_prompt": emotion_text,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed,
                        "sprite_output_path": sheet_output_path,
                        "source_path": _source_path,
                        "character": character,
                        "costume": costume,
                        "costume_face": costume_face,
                        "costume_head": costume_head,
                        "face_details": face_details,
                        "emotion": emotion_key,
                        "sprite_index": sprite_index,
                    }, ensure_ascii=False))

        # Return results even if no images (user may not have connected image input)
        # But still return valid emotion data
        if not images:
            # Return empty lists for images/masks but keep emotion/prompt data valid
            return [], pipe, emotion_data

        return images, pipe, emotion_data

NODE_CLASS_MAPPINGS = {
    "EmotionGeneratorV2": EmotionGeneratorV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionGeneratorV2": "VNCCS Emotion Studio"
}
