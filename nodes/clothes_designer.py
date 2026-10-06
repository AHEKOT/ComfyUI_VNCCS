from ..operation_logger import log_event, log_stage, logged_operation

from .preview_runtime import run_wizard_job

import os
import json
import hashlib
import torch
import folder_paths
import server
from aiohttp import web
from PIL import Image
import numpy as np
import re

from ..utils import (
    character_dir, save_costume_info, delete_costume,
    load_costume_info, list_costumes, ensure_costume_structure,
    sheets_dir,
    ensure_safe_name, safe_join_under, safe_relative_path, atomic_output_path,
    validate_costume_info, privileged_route
)
from .character_generator import (
    _call_comfy_node,
    _encode_flux_klein,
    _resolution_scale_megapixels,
    _resolution_scale_value,
    VNCCS_CharacterGenerator,
    H3_FRAME_COUNT,
    NATIVE_BACKGROUND_PROMPT,
)
from .vnccs_control_center import _entry_kind, _find_model_on_disk, _apply_lora_standard
from .qwen_vl import _ensure_qwen_vl_assets, _find_qwen_vl_model, QWEN_VL_MODEL_FILENAME
from .qwen_vl import configure_qwen_text_chat, parse_wizard_json, _validate_gguf_file

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
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
BACKGROUND_RGB = {
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
}
TRANSPARENT_BACKGROUND = "Transparent"


def _save_preview_cache(image, image_path, info_path, info):
    # Invalidate the old image/metadata pair before publishing either new file.
    # Failure preserves the previous image but cannot authorize its reuse with
    # a different prompt or reference. A later normal run can rebuild the cache.
    try:
        os.unlink(info_path)
    except FileNotFoundError:
        pass
    with atomic_output_path(image_path) as temporary:
        image.save(temporary, format="PNG")
    with atomic_output_path(info_path) as temporary:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(info, handle)


def _qi2_edit_system_prompt():
    # Keep the official Qwen edit template separate from Creator's T2I template.
    roots = dict.fromkeys(
        os.path.dirname(os.path.dirname(path))
        for path in (os.path.abspath(__file__), os.path.realpath(__file__))
    )
    for root in roots:
        path = os.path.join(root, "character_template", "qi2_edit_prompt_rewriter.txt")
        try:
            with open(path, "r", encoding="utf-8") as prompt_file:
                prompt = prompt_file.read().strip()
            if prompt:
                return prompt
        except OSError:
            continue
    raise RuntimeError("QI2 edit prompt template is missing. Restore character_template/qi2_edit_prompt_rewriter.txt.")


def _qi2_edit_image_batch(references):
    """Fit prepared references into TextGenerate's RGB batch in encoder order."""
    target = references[0]
    height, width = target.shape[1:3]
    scale = min(1.0, 1024 / max(height, width))
    height, width = max(1, round(height * scale)), max(1, round(width * scale))
    batch = []
    for reference in references:
        if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[-1] not in (3, 4):
            raise ValueError("QI2 edit references must contain exactly one RGB or RGBA image each.")
        # Prepared RGBA references already have their RGB composited over white.
        rgb = reference[..., :3].to(device=target.device, dtype=target.dtype)
        ref_height, ref_width = rgb.shape[1:3]
        fit = min(height / ref_height, width / ref_width)
        fit_height = max(1, min(height, round(ref_height * fit)))
        fit_width = max(1, min(width, round(ref_width * fit)))
        if (ref_height, ref_width) != (fit_height, fit_width):
            rgb = torch.nn.functional.interpolate(
                rgb.movedim(-1, 1), size=(fit_height, fit_width),
                mode="bilinear", align_corners=False,
            ).movedim(1, -1)
        canvas = rgb.new_ones((1, height, width, 3))
        top, left = (height - fit_height) // 2, (width - fit_width) // 2
        canvas[:, top:top + fit_height, left:left + fit_width] = rgb
        batch.append(canvas)
    return torch.cat(batch, dim=0)


def _clothes_background_prompt(background_color):
    if background_color == TRANSPARENT_BACKGROUND:
        return NATIVE_BACKGROUND_PROMPT
    if background_color == "Blue":
        return (
            "flat uniform pure blue background, exact RGB (0, 0, 255), "
            "hex #0000FF, no purple, no violet, no gradient"
        )
    return (
        "flat uniform pure green background, exact RGB (0, 255, 0), "
        "hex #00FF00, no gradient"
    )


def _rewrite_qi2_clothes_prompt(clip, prompt, references, background_color, system_prompt):
    from .character_creator_v2 import QI2_TEXT_GENERATION_DEFAULTS, _qi2_rewritten_prompt

    constraints = (
        "Edit the outfit of the character in the image using the requested clothing. "
        "Preserve the identity, face, hair, body proportions, pose, framing and rendering medium of the image. "
        f"{_clothes_background_prompt(background_color)} No background scenery, patterns, or shapes."
    )
    request = (
        f"{prompt}\n{constraints}\n"
        "This is an outfit edit of the target canvas, preserving its framing and aspect ratio."
    )
    log_stage("rewriting_prompt", component="ClothesDesigner")
    generated_text = _call_comfy_node(
        "TextGenerate", clip=clip,
        prompt=f"{system_prompt}\n\n{request}",
        image=_qi2_edit_image_batch(references),
        **QI2_TEXT_GENERATION_DEFAULTS,
    )[0]
    rewritten = _qi2_rewritten_prompt(generated_text, prompt)
    # Keep application-owned identity and background after PE too.
    return f"{rewritten}\n{constraints}"


def _clothes_target_size(settings, model_kind):
    default = 1536 if model_kind == "minimaxh3" else 1024
    value = settings.get("target_size")
    if value is None or value == "":
        return default
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Resolution scale must be an integer between 512 and 4096.") from None
    if isinstance(value, bool) or size != float(value) or not 512 <= size <= 4096:
        raise ValueError("Resolution scale must be an integer between 512 and 4096.")
    return _resolution_scale_value(size, default=default)


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
        log_event('warning', component='ClothesDesigner', level='warning', message=f'Failed to list preview sprites for {character}/{costume}: {exc}', error=str(exc))
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


def _clone_reference_digest(image_path):
    digest = hashlib.sha256()
    with open(image_path, "rb") as reference_file:
        for chunk in iter(lambda: reference_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_clothes_wizard_model():
    return _find_qwen_vl_model()


def _ensure_clothes_wizard_model():
    model_path, _mmproj_path = _ensure_qwen_vl_assets(allow_download=False, require_mmproj=False)
    return model_path


def _parse_clothes_wizard_json(content):
    data = parse_wizard_json(content)
    if data is None:
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
    model_kind = _entry_kind(getattr(pipe, "model_entry", None))
    for entry in getattr(pipe, "lora_entries", []) or []:
        if not isinstance(entry, dict):
            continue
        entry_kind = _entry_kind(entry)
        if entry_kind and model_kind and entry_kind != model_kind:
            continue
        name = entry.get("name", "")
        rel_path = _normalize_lora_rel_path(entry.get("local_path") or entry.get("path"))
        if _is_clothes_core_lora_name(name) or _is_clothes_core_lora_name(rel_path):
            return rel_path
    if model_kind == "qi2":
        return ""
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
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @classmethod
    def IS_CHANGED(cls, widget_data="{}", **kwargs):
        data = json.loads(widget_data) if isinstance(widget_data, str) else widget_data
        if not isinstance(data, dict):
            return ""
        source_path = cls.reference_sprite_path(data.get("character", ""), data)
        identity = {"source": _clone_reference_digest(source_path) if source_path else None}
        if data.get("activeTab") == "clone" and data.get("clone_image"):
            identity["donor"] = _clone_reference_digest(resolve_comfy_image_path(data["clone_image"]))
        return json.dumps(identity, sort_keys=True)

    @staticmethod
    def _normalize_background_color(value):
        bg_col = str(value or "Green").strip().capitalize()
        return bg_col if bg_col in {*BACKGROUND_RGB, TRANSPARENT_BACKGROUND} else "Green"

    @staticmethod
    def _effective_background_color(value, model_kind=""):
        background = ClothesDesigner._normalize_background_color(value)
        if background == TRANSPARENT_BACKGROUND and str(model_kind or "").lower() != "qi2":
            return "Green"
        return background

    @staticmethod
    def _background_output_value(background_color):
        return "Alpha" if background_color == TRANSPARENT_BACKGROUND else background_color

    @staticmethod
    def _pil_image_tensor(image):
        has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
        converted = image.convert("RGBA" if has_alpha else "RGB")
        array = np.asarray(converted, dtype=np.float32) / 255.0
        return torch.from_numpy(array.copy()).unsqueeze(0)

    @staticmethod
    def _prepare_reference_background(image, background_color, preserve_transparency=False):
        if not torch.is_tensor(image) or image.ndim not in {3, 4} or image.shape[-1] < 4:
            return image
        batch = image.unsqueeze(0) if image.ndim == 3 else image
        rgb = batch[..., :3]
        alpha = batch[..., 3:4].clamp(0.0, 1.0)
        if background_color == TRANSPARENT_BACKGROUND and preserve_transparency:
            # Qwen's vision tower composites alpha over white while its VAE keeps
            # all four channels. Clean hidden RGB so both paths see the same plate.
            clean_rgb = rgb * alpha + (1.0 - alpha)
            prepared = torch.cat([clean_rgb, alpha], dim=-1)
        else:
            color = BACKGROUND_RGB.get(background_color, BACKGROUND_RGB["Green"])
            plate = torch.tensor(color, dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, 3)
            prepared = rgb * alpha + plate * (1.0 - alpha)
        return prepared[0] if image.ndim == 3 else prepared

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
    def construct_prompt(data, model_kind=""):
        active_tab = data.get("activeTab", "generate")
        is_clone = active_tab == "clone"
        if is_clone and not data.get("clone_image"):
            raise ValueError("Upload a clothing reference image before using Clone Clothes.")
        bg_col = ClothesDesigner._effective_background_color(
            data.get("gen_settings", {}).get("background_color"), model_kind,
        )
        background_prompt = _clothes_background_prompt(bg_col)

        if is_clone:
            positive_prompt = (
                "Dress character to clothes from image 2\n"
                f"{background_prompt}"
            )
        else:
            info = data.get("costume_info", {})
            parts = []
            for k in ["top", "bottom", "head", "shoes", "face"]:
                v = info.get(k, "").strip()
                if v: parts.append(v)

            clothes_desc = "\n".join(parts)
            positive_prompt = (
                f"Dress the character:\n{clothes_desc}\n"
                f"{background_prompt}"
            )
        negative_prompt = "bad quality, worst quality, (naked, nude, nipple, penis, vagina:2.0)"
        if is_clone:
            negative_prompt += (
                ", background scenery, patterned background, shapes in background, "
                "multicolored background, textured background, gradient background"
            )
        if bg_col == "Blue":
            negative_prompt += ", purple background, violet background, gradient background"
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

    @staticmethod
    def reference_sprite_path(character_name, data=None):
        if not character_name:
            return None
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
                        return sprite_path
                    log_event("fallback", component="ClothesDesigner", level="warning", message=f'Selected preview sprite list is empty for {character_name}/{costume}; falling back to latest base sprite.')
                except Exception as exc:
                    log_event('fallback', component='ClothesDesigner', level='warning', message=f'Failed to load selected preview sprite {selected}: {exc}. Falling back to latest base sprite.', error=str(exc))

            sprite_path = get_latest_sprite_path(character_name, "Naked") or get_latest_sprite_path(character_name, "Original")
            return sprite_path
        except (ValueError, OSError):
            return None

    def get_reference_sprite(self, character_name, data=None):
        sprite_path = self.reference_sprite_path(character_name, data)
        if not sprite_path:
            return None
        with Image.open(sprite_path) as img:
            return self._pil_image_tensor(img)

    @logged_operation("ClothesDesigner", "preview")
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
        log_event("preview_settings", character=character_name, costume=costume_name, mode=active_tab)

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
        
        # 0. Validation
        def has_base_body(c):
             return bool(
                 get_latest_sprite_path(c, "Naked")
                 or get_latest_sprite_path(c, "Original")
             )

        if not has_base_body(character_name):
             raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sprites.")

        model_kind = _entry_kind(getattr(pipe, "model_entry", None))
        is_h3 = model_kind == "minimaxh3"
        is_qi2 = model_kind == "qi2"
        background_color = self._effective_background_color(
            gen_settings.get("background_color"), model_kind,
        )

        # 1. Prompt
        positive_prompt, negative_prompt = self.construct_prompt(data, model_kind=model_kind)

        # 2. Paths
        sheet_path = sheets_dir(character_name, costume_name, "neutral")

        default_steps = 25 if model_kind == "qi2" else WORKFLOW_SAMPLER_DEFAULTS["steps"]
        default_cfg = 3.0 if model_kind == "qi2" else WORKFLOW_SAMPLER_DEFAULTS["cfg"]
        seed_value = gen_settings.get("seed")
        if seed_value is None or seed_value == "":
            seed_value = getattr(pipe, "seed_int", getattr(pipe, "seed", None))
        seed_int = int(WORKFLOW_SAMPLER_DEFAULTS["seed"] if seed_value is None else seed_value)
        sample_steps = int(getattr(pipe, "sample_steps", getattr(pipe, "steps", 0)) or default_steps)
        cfg = float(getattr(pipe, "cfg", 0.0) or default_cfg)
        denoise = float(getattr(pipe, "denoise", 0.0) or WORKFLOW_SAMPLER_DEFAULTS["denoise"])
        sampler_name = getattr(pipe, "sampler_name", None) or WORKFLOW_SAMPLER_DEFAULTS["sampler_name"]
        scheduler = getattr(pipe, "scheduler", None) or WORKFLOW_SAMPLER_DEFAULTS["scheduler"]
        clothes_core_lora = _resolve_pipe_clothes_core_lora(pipe)
        target_size = _clothes_target_size(gen_settings, model_kind)
        clone_image_path = resolve_comfy_image_path(data["clone_image"]) if active_tab == "clone" else None
        clone_reference_hash = _clone_reference_digest(clone_image_path) if clone_image_path else None
        # Resolve the actual selected image before consulting the persistent cache.
        ref_image = self.get_reference_sprite(character_name, data)
        if ref_image is None:
            raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sprites.")
        source_reference_hash = hashlib.sha256(
            ref_image.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        use_qi2_rewriter = is_qi2 and active_tab != "clone"
        edit_system_prompt = _qi2_edit_system_prompt() if use_qi2_rewriter else None

        # 3. Cache check
        c_img_path, c_info_path = self.get_cache_paths(character_name, costume_name)
        try:
            cache_payload = {
                "model_cache_key": getattr(pipe, "model_cache_key", None),
                "widget_data": data,
                "prompts": {"positive": positive_prompt, "negative": negative_prompt},
                "clone_reference_sha256": clone_reference_hash,
                "source_reference": {"sha256": source_reference_hash, "shape": list(ref_image.shape)},
                "sampler": {
                    "seed": seed_int,
                    "steps": sample_steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                },
                "clothes_core_lora": clothes_core_lora,
                "resolution": {"model_kind": model_kind, "target_size": target_size},
            }
            if use_qi2_rewriter:
                from .character_creator_v2 import QI2_TEXT_GENERATION_DEFAULTS

                cache_payload["qi2_edit_rewriter"] = {
                    "version": 1,
                    "system_prompt_sha256": hashlib.sha256(edit_system_prompt.encode("utf-8")).hexdigest(),
                    "text_generation": QI2_TEXT_GENERATION_DEFAULTS,
                }
            if is_qi2:
                cache_payload["qi2_cache"] = getattr(pipe, "qi2_cache", {})
                cache_payload["qi2_turbo"] = sorted(
                    item.get("name", "") for item in (getattr(pipe, "lora_states", []) or [])
                    if item.get("auto_apply") and "viggle" in str(item.get("name", "")).lower()
                )
            canonical_str = json.dumps(cache_payload, sort_keys=True, separators=(',', ':'))
            input_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
            # Custom or externally modified pipes cannot provide a stable asset identity.
            if cache_payload["model_cache_key"] is None:
                input_hash = "INVALID"
        except Exception:
             input_hash = "INVALID"

        if input_hash != "INVALID" and os.path.exists(c_img_path) and os.path.exists(c_info_path):
            try:
                with open(c_info_path, "r", encoding="utf-8") as f:
                    cache_info = json.load(f)
                if cache_info.get("hash") == input_hash:
                    log_event("preview_cache", component="ClothesDesigner", character=character_name, costume=costume_name, cache="hit")
                    with Image.open(c_img_path) as img:
                        image = self._pil_image_tensor(img)
                    server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": str(unique_id), "character": character_name})
                    return (image, sheet_path, self._background_output_value(background_color))
            except Exception as exc:
                log_event('fallback', component='ClothesDesigner', level='warning', message=f'Cache read failed, regenerating preview: {exc}', error=str(exc))

        ref_image = self._prepare_reference_background(
            ref_image, background_color, preserve_transparency=is_qi2,
        )
        
        # Load Clone Image. In clone mode it becomes Picture 2 from the reference workflow.
        clone_image_tensor = None
        if active_tab == "clone" and data.get("clone_image"):
             try:
                 log_event("diagnostic", component="ClothesDesigner", level="debug", message=f'Clone reference image resolved: {clone_image_path}')
                 
                 with Image.open(clone_image_path) as source_image:
                     i = self._pil_image_tensor(source_image)
                 i = self._prepare_reference_background(
                     i, background_color, preserve_transparency=is_qi2,
                 )
                 
                 clone_image_tensor = i
             except Exception as e:
                 log_event('warning', component='ClothesDesigner', level='warning', message=f'Failed to load clone image for encoder: {e}', error=str(e))
                 raise

        image2 = clone_image_tensor if active_tab == "clone" and clone_image_tensor is not None else None
        if use_qi2_rewriter:
            positive_prompt = _rewrite_qi2_clothes_prompt(
                clip, positive_prompt, (ref_image,),
                background_color, edit_system_prompt,
            )
        log_event("encoder_inputs", component="ClothesDesigner", mode=active_tab, clone_reference=image2 is not None, prompt_chars=len(positive_prompt))
        is_klein = model_kind == "klein9b"
        if not (is_h3 or is_qi2 or is_klein):
            raise ValueError(f"Unsupported clothes model family: {model_kind or 'unknown'}")
        encoder_kwargs = {
            "clip": clip,
            "prompt": positive_prompt,
            "vae": vae,
            "image1": ref_image,
            "image2": image2,
            "image3": None,
        }
        if is_klein:
            encoder_kwargs.update(
                upscale_method="lanczos",
                megapixels=_resolution_scale_megapixels(target_size),
                resolution_steps=1,
            )
        if is_h3:
            audio_vae = getattr(pipe, "audio_vae", None)
            if audio_vae is None:
                raise ValueError("MiniMax H3 requires the audio VAE from VNCCS Control Center.")
            width, height = VNCCS_CharacterGenerator()._resolution_scale_dimensions(ref_image, target_size)
            references = {"ref_image_1": ref_image}
            if image2 is not None:
                references["ref_image_2"] = image2
            pos_cond, empty_latent = _call_comfy_node(
                "MiniMaxH3ReferenceToVideo", clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=positive_prompt, width=width, height=height, length=H3_FRAME_COUNT,
                ref_image_size="match", ref_images=references,
            )
            neg_cond = None
        elif is_qi2:
            pos_cond, neg_cond, empty_latent = VNCCS_CharacterGenerator()._qi2_encode(
                {"clip": clip, "vae": vae}, positive_prompt,
                (ref_image, image2), target_size=target_size,
                negative_prompt=negative_prompt,
            )
        else:
            pos_cond, neg_cond, empty_latent = _call_comfy_node(_encode_flux_klein, **encoder_kwargs)
        
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

        sampler_model = model
        if clothes_core_lora:
            log_event("lora_applied", component="ClothesDesigner", model=clothes_core_lora, strength=1)
            lora_path, exists = _find_model_on_disk(f"models/loras/{clothes_core_lora}")
            if not exists:
                raise ValueError(f"VNCCS Clothes Core LoRA is not installed: {clothes_core_lora}")
            sampler_model = _apply_lora_standard(model, None, lora_path, 1)[0]
        if is_qi2:
            qi2_generator = VNCCS_CharacterGenerator()
            sampler_model, qi2_turbo = qi2_generator._qi2_prepare_model(
                sampler_model, pipe, {"qi2_cache": getattr(pipe, "qi2_cache", {})},
            )

        # 4. Sampling using the incoming Control Center pipe configuration
        log_stage("sampling", component="ClothesDesigner")
        if is_h3:
            sampler = _call_comfy_node("KSamplerSelect", sampler_name=sampler_name)[0]
            sigmas = _call_comfy_node("BasicScheduler", model=sampler_model, scheduler=scheduler, steps=sample_steps, denoise=denoise)[0]
            guider = _call_comfy_node("BasicGuider", model=sampler_model, conditioning=pos_cond)[0]
            noise = _call_comfy_node("RandomNoise", noise_seed=seed_int)[0]
            latent_result = _call_comfy_node(
                "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                sigmas=sigmas, latent_image=empty_latent,
            )[0]
        elif is_qi2:
            latent_result = qi2_generator._qi2_sample(
                sampler_model, pos_cond, neg_cond, empty_latent,
                {"seed": out_pipe.seed_int, "steps": out_pipe.sample_steps,
                 "cfg": out_pipe.cfg, "sampler_name": out_pipe.sampler_name,
                 "scheduler": out_pipe.scheduler, "denoise": out_pipe.denoise},
                turbo=qi2_turbo,
            )
        else:
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
        log_stage("decoding", component="ClothesDesigner")
        if is_qi2 or is_h3:
            with torch.inference_mode():
                image, = _call_comfy_node("VAEDecode", vae=vae, samples=latent_for_decode)
        else:
            try:
                with torch.inference_mode():
                    image, = _call_comfy_node(
                        "VAEDecodeTiled",
                        vae=vae,
                        samples=latent_for_decode,
                        **WORKFLOW_DECODE_DEFAULTS,
                    )
            except Exception as e:
                log_event('fallback', component='ClothesDesigner', level='warning', message=f'VAEDecodeTiled failed ({e}), falling back to VAEDecode...', error=str(e))
                with torch.inference_mode():
                    image, = _call_comfy_node("VAEDecode", vae=vae, samples=latent_for_decode)

        if is_h3:
            image = VNCCS_CharacterGenerator()._h3_first_frame_to_cpu(image)

        # Publish the preview before reporting success to the widget.
        i_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        _save_preview_cache(i_pil, c_img_path, c_info_path, {"hash": input_hash, "widget_data": data})
        try:
             log_event("diagnostic", component="ClothesDesigner", level="debug", message=f'Sending Preview Update Event: ID={unique_id}, Char={character_name}')
             server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": str(unique_id), "character": character_name})
        except Exception as e:
             log_event('warning', component='ClothesDesigner', level='warning', message=f'Failed to send preview update: {e}', error=str(e))
             log_event('exception', component='ClothesDesigner', level='debug', exc_info=True, message='Operation failed', error=str(e))

        return (image, sheet_path, self._background_output_value(background_color))

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
@privileged_route
async def vnccs_save_costume(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Costume request must be an object")
        character = data.get("character")
        costume = data.get("costume")
        info = validate_costume_info(data.get("info", {}))
        if not character or not costume: return web.Response(status=400)
        character = ensure_safe_name(character, "character")
        costume = ensure_safe_name(costume, "costume")
        ensure_costume_structure(character, costume)
        if not save_costume_info(character, costume, info):
            return web.json_response({"error": f"Could not save costume '{costume}'. Check storage permissions and free space."}, status=500)
        return web.json_response({"status": "ok"})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        return web.Response(status=500, text=str(e))


@server.PromptServer.instance.routes.post("/vnccs/delete_costume")
@privileged_route
async def vnccs_delete_costume(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Costume request must be an object")
        if not isinstance(data.get("character"), str) or not isinstance(data.get("costume"), str):
            raise ValueError("Character and costume must be names.")
        warning = delete_costume(data["character"], data["costume"])
        result = {"status": "ok"}
        if warning:
            result["warning"] = warning
        return web.json_response(result)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except FileNotFoundError as error:
        return web.json_response({"error": str(error)}, status=404)
    except Exception as error:
        return web.json_response({"error": str(error)}, status=500)


@logged_operation("ClothesDesigner", "wizard")
def _clothes_wizard_response(post):
    try:
        try:
            import llama_cpp
        except Exception as e:
            return web.json_response({
                "error": "DEPENDENCY_MISSING",
                "message": f"llama-cpp-python is required for Clothes Wizard: {e}",
                "model_name": "llama-cpp-python",
            }, status=500)

        user_description = str(post.get("description", "")).strip()
        if not user_description:
            return web.Response(status=400, text="No clothes description provided")

        try:
            model_path = _ensure_clothes_wizard_model()
        except Exception as e:
            return web.json_response({
                "error": "MODEL_MISSING" if isinstance(e, FileNotFoundError) else "MODEL_INVALID",
                "message": str(e),
                "model_name": QWEN_VL_MODEL_FILENAME,
            }, status=500)

        try:
            _validate_gguf_file(model_path, os.path.basename(model_path))
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

        log_stage("loading_wizard_model", component="ClothesDesigner", model=os.path.basename(model_path))
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

        configure_qwen_text_chat(llm)
        log_stage("generating_description")
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.35,
        )

        log_stage("parsing_description")
        content = response["choices"][0]["message"]["content"]
        log_event("wizard_output", component="ClothesDesigner", level="debug", output=content)
        parsed = _parse_clothes_wizard_json(content or "")
        if parsed is None:
            return web.json_response({
                "error": "PARSE_ERROR",
                "message": "Failed to parse Clothes Wizard JSON output.",
                "raw": content or "",
            }, status=500)

        return web.json_response(parsed)
    except Exception as e:
        log_event('exception', component='ClothesDesigner', level='debug', exc_info=True, message='Operation failed', error=str(e))
        return web.json_response({
            "error": "INFERENCE_ERROR",
            "message": f"Engine Error: {e}",
            "model_name": QWEN_VL_MODEL_FILENAME,
        }, status=500)


@server.PromptServer.instance.routes.post("/vnccs/clothes_wizard")
@privileged_route
async def vnccs_clothes_wizard(request):
    try:
        post = await request.json()
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid JSON request"}, status=400)
    if not isinstance(post, dict):
        return web.json_response({"error": "Request must be an object"}, status=400)
    return await run_wizard_job(_clothes_wizard_response, post, "clothes")


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
         return web.Response(status=400, text="Character incomplete. Generate sprites first.")

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
