
import os
import json
import torch
import folder_paths
import server
from aiohttp import web
from PIL import Image
import numpy as np
import traceback
import re

from ..utils import (
    character_dir, save_costume_info,
    load_costume_info, list_costumes, ensure_costume_structure,
    sheets_dir,
    ensure_safe_name, safe_join_under, safe_relative_path
)
from .character_generator import _call_comfy_node

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WORKFLOW_ENCODER_CLASS = "VNCCS_QWEN_Encoder"
WORKFLOW_ENCODER_INSTRUCTION = (
    "Describe the character and their key features (body shape, physical characteristics, "
    "clothing, items, accessories). Then explain how the user's text instruction should alter "
    "or modify the character. Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original character where appropriate."
)
WORKFLOW_ENCODER_DEFAULTS = {
    "target_size": 1024,
    "upscale_method": "lanczos",
    "crop_method": "disabled",
    "latent_image_index": 1,
    "weight1": 1,
    "weight2": 1,
    "weight3": 1,
    "vl_size": 384,
    "instruction": WORKFLOW_ENCODER_INSTRUCTION,
    "qwen_2511": True,
    "background_color": "White",
}
WORKFLOW_SAMPLER_DEFAULTS = {
    "seed": 200413815563996,
    "steps": 4,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "karras",
    "denoise": 1,
}
WORKFLOW_DECODE_DEFAULTS = {
    "tile_size": 512,
    "overlap": 64,
    "temporal_size": 64,
    "temporal_overlap": 8,
}
WORKFLOW_CLOTHES_CORE_LORA = "qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.6.safetensors"
BACKGROUND_RGB = {
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
}


def _latest_image_file(files):
    files = [path for path in files if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS]
    if not files:
        return None
    return max(files, key=lambda path: (os.path.getmtime(path), path))


def get_latest_sprite_path(character, costume="Naked"):
    try:
        base = os.path.join(character_dir(character), "Sprites", costume)
        if not os.path.isdir(base):
            return None

        neutral_dir = os.path.join(base, "Neutral")
        if os.path.isdir(neutral_dir):
            neutral_files = []
            for root, _dirs, filenames in os.walk(neutral_dir):
                for filename in filenames:
                    neutral_files.append(os.path.join(root, filename))
            best = _latest_image_file(neutral_files)
            if best:
                return best

        direct_files = [
            os.path.join(base, filename)
            for filename in os.listdir(base)
            if os.path.isfile(os.path.join(base, filename))
        ]
        best = _latest_image_file(direct_files)
        if best:
            return best

        nested_files = []
        for root, _dirs, filenames in os.walk(base):
            for filename in filenames:
                nested_files.append(os.path.join(root, filename))
        return _latest_image_file(nested_files)
    except Exception:
        return None


def list_preview_sprite_files(character, costume=None):
    try:
        base_char_path = character_dir(character)
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
        for root in sprite_roots:
            if not os.path.isdir(root):
                continue
            files = [
                os.path.join(root, filename)
                for filename in os.listdir(root)
                if os.path.isfile(os.path.join(root, filename))
                and os.path.splitext(filename)[1].lower() in IMAGE_EXTS
            ]
            if files:
                return sorted(files)
        return []
    except Exception as exc:
        print(f"[ClothesDesigner] Failed to list preview sprites for {character}/{costume}: {exc}")
        return []


def resolve_comfy_image_path(image_info):
    if isinstance(image_info, dict):
        image_name = image_info.get("name")
        subfolder = image_info.get("subfolder", "")
    else:
        image_name = image_info
        subfolder = ""

    if not image_name:
        raise FileNotFoundError("Clone image entry has no file name.")

    safe_name = safe_relative_path(image_name, "image_name")
    safe_subfolder = safe_relative_path(subfolder, "subfolder") if subfolder else ""
    parts = [safe_subfolder, safe_name] if safe_subfolder else [safe_name]
    candidate = safe_join_under(folder_paths.get_input_directory(), *parts)

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Clone image '{image_name}' was not found in ComfyUI input folder. "
        "Upload the clone reference image again in VNCCS Clothes Designer."
    )


def _validate_clothes_wizard_gguf(path, file_label="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_label} was not written: {path}")

    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise ValueError(f"{file_label} is too small to be a valid GGUF file ({size} bytes)")

    with open(path, "rb") as file:
        magic = file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{file_label} is not a valid GGUF file (magic={magic!r})")


def _find_clothes_wizard_model():
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


def _parse_clothes_wizard_json(content):
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
    for key in ["top", "bottom", "shoes", "head", "face"]:
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()
    return result


def _is_clothes_core_lora_name(value):
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return "vnccs" in normalized and "clothes" in normalized and "core" in normalized


def _normalize_lora_rel_path(value):
    raw = str(value or "").strip().replace("\\", "/")
    if not raw or raw.lower() == "none":
        return ""
    for prefix in ("models/loras/", "loras/"):
        if raw.lower().startswith(prefix):
            return raw[len(prefix):]
    return raw


def _resolve_pipe_clothes_core_lora(pipe):
    for entry in getattr(pipe, "lora_entries", []) or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        rel_path = _normalize_lora_rel_path(entry.get("local_path") or entry.get("path"))
        if _is_clothes_core_lora_name(name) or _is_clothes_core_lora_name(rel_path):
            return rel_path
    raise ValueError("VNCCS Clothes Designer requires VNCCS Clothes Core LoRA from Control Center pipe.")


class PipeContext:
    def __init__(self, source=None, **updates):
        s = source
        self.model = getattr(s, "model", None) if s is not None else None
        self.clip = getattr(s, "clip", None) if s is not None else None
        self.vae = getattr(s, "vae", None) if s is not None else None
        self.pos = getattr(s, "pos", None) if s is not None else None
        self.neg = getattr(s, "neg", None) if s is not None else None
        self.seed_int = getattr(s, "seed_int", getattr(s, "seed", 0)) if s is not None else 0
        self.sample_steps = getattr(s, "sample_steps", getattr(s, "steps", 0)) if s is not None else 0
        self.cfg = getattr(s, "cfg", 0.0) if s is not None else 0.0
        self.denoise = getattr(s, "denoise", 1.0) if s is not None else 1.0
        self.sampler_name = getattr(s, "sampler_name", None) if s is not None else None
        self.scheduler = getattr(s, "scheduler", None) if s is not None else None
        raw_loader_type = getattr(s, "loader_type", None) if s is not None else None
        self.loader_type = "standard" if raw_loader_type == "nunchaku" else raw_loader_type
        # TECH DEBT: deprecated Nunchaku fields kept as None for compatibility.
        # Delete after old workflow JSON is migrated.
        self.nunchaku_kind = None
        self.nunchaku_settings = None
        self.model_entry = getattr(s, "model_entry", None) if s is not None else None
        for key, value in updates.items():
            setattr(self, key, value)

class ClothesDesigner:
    """
    VNCCS Clothes Designer Node
    Wraps standard ComfyUI nodes for GGUF loading, Sampling, and VAE decoding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "pipe": ("VNCCS_PIPE",),
            },
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
    def _normalize_background_color(value):
        bg_col = str(value or "Green").strip().capitalize()
        return bg_col if bg_col in BACKGROUND_RGB else "Green"

    @staticmethod
    def _is_editable_costume(value):
        costume = str(value or "").strip()
        return bool(costume) and costume not in {"Naked", "Original"}

    @staticmethod
    def _emit_validation_error(unique_id, message):
        if unique_id is None:
            return
        try:
            server.PromptServer.instance.send_sync(
                "vnccs.clothes_designer.validation_error",
                {"node_id": str(unique_id), "message": message},
            )
        except Exception:
            pass

    @staticmethod
    def _find_breasts_desc(char_info):
        """Search all string fields in character info for a breast/chest description."""
        # Prioritise 'body' field, then check the rest
        fields = ["body"] + [k for k in char_info if k != "body"]
        for key in fields:
            val = char_info.get(key)
            if not isinstance(val, str):
                continue
            m = re.search(r'[a-zA-Z\s]*(?:breasts?|flat\s+chest)[a-zA-Z\s]*', val, re.IGNORECASE)
            if m:
                return m.group(0).strip().strip(",").strip()
        return None

    @staticmethod
    def construct_prompt(data):
        active_tab = data.get("activeTab", "generate")

        if active_tab == "clone" and data.get("clone_image"):
            return "Dress character: clothes, footwear and accessories from Picture 2", ""

        info = data.get("costume_info", {})
        parts = []
        for k in ["top", "bottom", "head", "shoes", "face"]:
            v = info.get(k, "").strip()
            if v: parts.append(v)

        clothes_desc = "\n".join(parts)
        bg_col = ClothesDesigner._normalize_background_color(data.get("gen_settings", {}).get("background_color"))
        hex_col = "00FF00" if bg_col == "Green" else "0000FF"

        positive_prompt = (
            f"Dress the character:\n{clothes_desc}\n"
            f"solid {bg_col.lower()} ({hex_col}) background"
        )
        negative_prompt = "bad quality, worst quality, (naked, nude, nipple, penis, vagina:2.0)"
        return positive_prompt, negative_prompt

    @staticmethod
    def get_cache_paths(character, costume):
        def clean_name(n):
            return "".join([c for c in n if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        safe_costume = clean_name(costume)
        cache_dir = os.path.join(character_dir(character), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        img_path = os.path.join(cache_dir, f"preview_{safe_costume}.png")
        info_path = os.path.join(cache_dir, f"preview_info_{safe_costume}.json")
        return img_path, info_path

    def get_reference_sprite(self, character_name, data=None):
        try:
            selected = data.get("selected_preview_sprite") if isinstance(data, dict) else None
            if isinstance(selected, dict) and selected.get("character") == character_name:
                costume = selected.get("costume") or None
                try:
                    if costume:
                        costume = ensure_safe_name(costume, "costume")
                    index = int(selected.get("index", 0))
                    files = list_preview_sprite_files(character_name, costume)
                    if files:
                        sprite_path = files[index % len(files)]
                        img = Image.open(sprite_path).convert("RGB")
                        image_np = np.array(img).astype(np.float32) / 255.0
                        sprite_tensor = torch.from_numpy(image_np).unsqueeze(0)
                        print(f"[ClothesDesigner] Using selected preview sprite for Picture 1: {sprite_path} (index={index})")
                        return sprite_tensor
                    print(f"[ClothesDesigner] Selected preview sprite list is empty for {character_name}/{costume}; falling back to latest base sprite.")
                except Exception as exc:
                    print(f"[ClothesDesigner] Failed to load selected preview sprite {selected}: {exc}. Falling back to latest base sprite.")

            sprite_path = get_latest_sprite_path(character_name, "Naked") or get_latest_sprite_path(character_name, "Original")
            if sprite_path:
                img = Image.open(sprite_path).convert("RGB")
                image_np = np.array(img).astype(np.float32) / 255.0
                sprite_tensor = torch.from_numpy(image_np).unsqueeze(0)
                return sprite_tensor
            print(f"[ClothesDesigner] No Naked/Original sprites found for {character_name}. Run migration or generate sprites first.")
            return None
        except:
            return None

    def process(self, pipe=None, widget_data="{}", unique_id=None):
        # CRITICAL FIX: Ensure PromptServer has last_prompt_id for preview system
        if not hasattr(server.PromptServer.instance, "last_prompt_id"):
             server.PromptServer.instance.last_prompt_id = "vnccs_api_preview"

        try:
            if isinstance(widget_data, str):
                data = json.loads(widget_data)
            else: data = widget_data
        except:
            data = {}

        character_name = data.get("character", "Unknown")
        costume_name = str(data.get("costume") or "").strip()
        gen_settings = data.get("gen_settings", {})
        active_tab = data.get("activeTab", "generate")

        if not self._is_editable_costume(costume_name):
            message = "Create a new costume first, then select it before generating a preview."
            self._emit_validation_error(unique_id, message)
            raise ValueError(message)

        if pipe is None:
            raise ValueError("Clothes Designer requires an incoming VNCCS pipe from Control Center.")

        model = getattr(pipe, "model", None)
        clip = getattr(pipe, "clip", None)
        vae = getattr(pipe, "vae", None)
        if model is None or clip is None or vae is None:
            raise ValueError("Incoming VNCCS pipe is missing model, clip, or vae.")
        
        # 0. Cache Check
        import hashlib
        self.get_cache_paths(character_name, costume_name)

        # Stable Hash Calculation
        try:
            if isinstance(widget_data, str):
                msg_obj = json.loads(widget_data)
            else:
                msg_obj = widget_data
            
            # Canonical JSON representation
            canonical_str = json.dumps(msg_obj, sort_keys=True, separators=(',', ':'))
            input_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
        except:
             input_hash = "INVALID"

        # 0. Validation
        def has_base_body(c):
             return bool(
                 get_latest_sprite_path(c, "Naked")
                 or get_latest_sprite_path(c, "Original")
             )

        if not has_base_body(character_name):
             raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sprites.")

        # 1. Prompt
        positive_prompt, negative_prompt = self.construct_prompt(data)
        
        # 2. Paths
        sheet_path = sheets_dir(character_name, costume_name, "neutral") 

        ref_image = self.get_reference_sprite(character_name, data)
        if ref_image is None:
            raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sprites.")
        
        # Load Clone Image. In clone mode it becomes Picture 2 from the reference workflow.
        clone_image_tensor = None
        if active_tab == "clone" and data.get("clone_image"):
             try:
                 c_info = data["clone_image"]
                 image_path = resolve_comfy_image_path(c_info)
                 print(f"[ClothesDesigner] Clone reference image resolved: {image_path}")
                 
                 i = Image.open(image_path).convert("RGB")
                 
                 # Convert to Tensor (H,W,C)
                 i = torch.from_numpy(np.array(i).astype(np.float32) / 255.0).unsqueeze(0)
                 
                 clone_image_tensor = i
             except Exception as e:
                 print(f"[ClothesDesigner] Failed to load clone image for encoder: {e}")
                 raise

        image2 = clone_image_tensor if active_tab == "clone" and clone_image_tensor is not None else None
        print(
            "[ClothesDesigner] Encoder inputs: "
            f"mode={active_tab}, image1=reference sprite, "
            f"image2={'clone reference' if image2 is not None else 'none'}, prompt={positive_prompt!r}"
        )
        pos_cond, neg_cond, empty_latent = _call_comfy_node(
            WORKFLOW_ENCODER_CLASS,
            clip=clip,
            prompt=positive_prompt,
            vae=vae,
            image1=ref_image,
            image2=image2,
            image3=None,
            image1_name="Picture 1",
            image2_name="Picture 2",
            image3_name="Picture 3",
            **WORKFLOW_ENCODER_DEFAULTS,
        )
        
        seed_int = int(getattr(pipe, "seed_int", getattr(pipe, "seed", 0)) or WORKFLOW_SAMPLER_DEFAULTS["seed"])
        sample_steps = int(getattr(pipe, "sample_steps", getattr(pipe, "steps", 0)) or WORKFLOW_SAMPLER_DEFAULTS["steps"])
        cfg = float(getattr(pipe, "cfg", 0.0) or WORKFLOW_SAMPLER_DEFAULTS["cfg"])
        denoise = float(getattr(pipe, "denoise", 0.0) or WORKFLOW_SAMPLER_DEFAULTS["denoise"])
        sampler_name = getattr(pipe, "sampler_name", None) or WORKFLOW_SAMPLER_DEFAULTS["sampler_name"]
        scheduler = getattr(pipe, "scheduler", None) or WORKFLOW_SAMPLER_DEFAULTS["scheduler"]

        out_pipe = PipeContext(
            source=pipe,
            model=model, clip=clip, vae=vae,
            pos=pos_cond, neg=neg_cond,
            seed_int=seed_int,
            sample_steps=sample_steps,
            cfg=cfg,
            denoise=denoise,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )

        clothes_core_lora = _resolve_pipe_clothes_core_lora(pipe)
        print(f"[ClothesDesigner] Applying VNCCS Clothes Core LoRA from pipe: {clothes_core_lora} (strength=1)")
        sampler_model = _call_comfy_node(
            "LoraLoaderModelOnly",
            model=model,
            lora_name=clothes_core_lora,
            strength_model=1,
        )[0]

        # 4. Sampling using the incoming Control Center pipe configuration
        print("[ClothesDesigner] Sampling...")
        latent_result = _call_comfy_node(
            "KSampler",
            model=sampler_model, seed=out_pipe.seed_int, steps=out_pipe.sample_steps,
            cfg=out_pipe.cfg, sampler_name=out_pipe.sampler_name, scheduler=out_pipe.scheduler,
            positive=pos_cond, negative=neg_cond, latent_image=empty_latent, denoise=out_pipe.denoise
        )[0]

        def normalize_decode_input(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            if isinstance(value, dict):
                return {k: normalize_decode_input(v) for k, v in value.items()}
            if isinstance(value, list):
                return [normalize_decode_input(v) for v in value]
            if isinstance(value, tuple):
                return tuple(normalize_decode_input(v) for v in value)
            return value

        latent_for_decode = normalize_decode_input(latent_result)

        # 5. Decode
        print("[ClothesDesigner] VAE Decoding...")
        try:
            with torch.inference_mode():
                image, = _call_comfy_node(
                    "VAEDecodeTiled",
                    vae=vae,
                    samples=latent_for_decode,
                    **WORKFLOW_DECODE_DEFAULTS,
                )
        except Exception as e:
            print(f"[ClothesDesigner] VAEDecodeTiled failed ({e}), falling back to VAEDecode...")
            with torch.inference_mode():
                image, = _call_comfy_node("VAEDecode", vae=vae, samples=latent_for_decode)

        # Cache for UI preview
        try:
             c_img_path, c_info_path = self.get_cache_paths(character_name, costume_name)
             i_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
             i_pil.save(c_img_path)
             
             # Save Cache Info
             with open(c_info_path, "w") as f:
                 json.dump({
                     "hash": input_hash,
                     "widget_data": data # Save parsed data or original string
                 }, f)
                 
             print(f"[ClothesDesigner] Sending Preview Update Event: ID={unique_id}, Char={character_name}")
             server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": str(unique_id), "character": character_name})
        except Exception as e:
             print(f"[ClothesDesigner] Failed to send preview update: {e}")
             traceback.print_exc()

        bg_color = gen_settings.get("background_color", "Green")
        return (image, sheet_path, bg_color)

# --- API ---

@server.PromptServer.instance.routes.get("/vnccs/list_costumes")
async def vnccs_list_costumes(request):
    character = request.rel_url.query.get("character", "")
    if not character: return web.json_response([])
    try:
        character = ensure_safe_name(character, "character")
        data = list_costumes(character)
        return web.json_response(data)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/vnccs/get_costume")
async def vnccs_get_costume(request):
    character = request.rel_url.query.get("character", "")
    costume = request.rel_url.query.get("costume", "")
    if not character or not costume: return web.json_response({})
    try:
        character = ensure_safe_name(character, "character")
        costume = ensure_safe_name(costume, "costume")
        data = load_costume_info(character, costume)
        return web.json_response(data)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.post("/vnccs/save_costume")
async def vnccs_save_costume(request):
    try:
        data = await request.json()
        character = data.get("character")
        costume = data.get("costume")
        info = data.get("info", {})
        if not character or not costume: return web.Response(status=400)
        character = ensure_safe_name(character, "character")
        costume = ensure_safe_name(costume, "costume")
        ensure_costume_structure(character, costume)
        save_costume_info(character, costume, info)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.Response(status=500, text=str(e))


@server.PromptServer.instance.routes.post("/vnccs/clothes_wizard")
async def vnccs_clothes_wizard(request):
    try:
        try:
            import llama_cpp
        except Exception as e:
            return web.json_response({
                "error": "DEPENDENCY_MISSING",
                "message": f"llama-cpp-python is required for Clothes Wizard: {e}",
                "model_name": "llama-cpp-python",
            }, status=500)

        post = await request.json()
        user_description = str(post.get("description", "")).strip()
        if not user_description:
            return web.Response(status=400, text="No clothes description provided")

        model_path = _find_clothes_wizard_model()
        if not model_path:
            return web.json_response({
                "error": "MODEL_MISSING",
                "message": "No Qwen GGUF model found.",
                "model_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            }, status=404)

        try:
            _validate_clothes_wizard_gguf(model_path, os.path.basename(model_path))
        except Exception as e:
            return web.json_response({
                "error": "MODEL_INVALID",
                "message": f"Qwen GGUF model file is invalid or incomplete: {e}",
                "model_name": os.path.basename(model_path),
            }, status=422)

        system_prompt = (
            "You are a professional anime/game character costume designer. "
            "Convert broad outfit ideas into concrete, visual clothing prompts. "
            "Output valid JSON only."
        )
        user_prompt = f"""
Expand this abstract clothing idea into detailed outfit parts:
{user_description}

Return a raw JSON object with exactly these string keys:
- top
- bottom
- shoes
- head
- face (ONLY wearable face items/accessories)

Rules:
- Each value must be a detailed visual description suitable for image generation.
- Do not repeat the same abstract phrase from the user.
- Describe materials, colors, shape, trims, accessories, fit, and distinctive details.
- If a category is not needed, use an empty string.
- Keep descriptions clothing-focused.
- The "face" field is NOT for facial expression, makeup, blush, eyeshadow, lipstick, skin, cheeks, or facial features.
- Use "face" only for wearable/accessory items placed on the face, such as glasses, sunglasses, goggles, mask, veil, eyepatch, respirator, scarf over mouth, piercings, stickers, or temporary tattoos.
- If there is no wearable face item, set "face" to an empty string.
- Do not describe the body, pose, background, camera, quality tags, nudity, sex acts, facial expression, makeup, blush, eyeshadow, lipstick, skin, or cheeks.

Example for "Santa Claus costume":
{{
  "top": "red velvet Santa coat with thick white fur trim on cuffs, hem and front opening, black leather belt with square gold buckle, long sleeves, festive winter fabric texture",
  "bottom": "matching red velvet trousers with white fur cuffs, fitted but comfortable costume pants",
  "shoes": "black polished leather boots with rounded toes and folded cuffs",
  "head": "red Santa hat with white fur brim and white pom-pom, slightly tilted",
  "face": ""
}}
"""

        print(f"[ClothesDesigner] Clothes Wizard loading model: {model_path}")
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.35,
        )

        content = response["choices"][0]["message"]["content"]
        print(f"[ClothesDesigner] Clothes Wizard raw output: {content}")
        parsed = _parse_clothes_wizard_json(content or "")
        if parsed is None:
            return web.json_response({
                "error": "PARSE_ERROR",
                "message": "Failed to parse Clothes Wizard JSON output.",
                "raw": content or "",
            }, status=500)

        return web.json_response(parsed)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({
            "error": "INFERENCE_ERROR",
            "message": f"Engine Error: {e}",
            "model_name": "Qwen2VL (Check Console)",
        }, status=500)


@server.PromptServer.instance.routes.get("/vnccs/get_preview")
async def vnccs_get_preview(request):
    character = request.rel_url.query.get("character", "")
    costume = request.rel_url.query.get("costume", "Naked")
    if not character: return web.Response(status=404)
    try:
        character = ensure_safe_name(character, "character")
        costume = ensure_safe_name(costume, "costume")
    except ValueError as e:
        return web.Response(status=400, text=str(e))

    naked_sprite = get_latest_sprite_path(character, "Naked")
    original_sprite = get_latest_sprite_path(character, "Original")
    if not naked_sprite and not original_sprite:
         return web.Response(status=400, text="Character incomplete. Run migration or generate sprites first.")

    force_cache = request.rel_url.query.get("force_cache", "") == "true"
    
    target_file = None
    
    # helper logic duplicated/inlined since we can't easily call instance static method from here without instance
    # actually we can use ClothesDesigner.get_cache_paths
    cache_file, _ = ClothesDesigner.get_cache_paths(character, costume)

    # If force_cache is true, try cache first
    if force_cache and os.path.exists(cache_file):
        target_file = cache_file

    # Otherwise, try costume sprites first.
    if not target_file:
         target_file = get_latest_sprite_path(character, costume)

    # For the base character preview, Original is the direct SFW fallback for Naked.
    if not target_file and costume == "Naked":
         target_file = original_sprite

    # Fallback to cache if sprites are not available for this costume.
    if not target_file:
         if os.path.exists(cache_file): target_file = cache_file
         
    # Final fallback to base sprites.
    if not target_file:
         target_file = naked_sprite or original_sprite

    if target_file and os.path.exists(target_file):
         with open(target_file, "rb") as f:
             return web.Response(body=f.read(), content_type="image/png")
             
    return web.Response(status=404)


NODE_CLASS_MAPPINGS = {
    "ClothesDesigner": ClothesDesigner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothesDesigner": "VNCCS Clothes Designer",
}
