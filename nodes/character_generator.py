"""VNCCS Character Generator.

Replacement node for the Step 1 Pose Generation -> Upscaler -> BG Remove
subgraph chain. It executes the same processing stages internally and exposes a
DOM widget for stage previews/settings.
"""

import base64
import inspect
import io
import json
import os
import shutil
import traceback
from urllib.parse import urlencode

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI runtime import
    folder_paths = None

try:
    import nodes as comfy_nodes
except Exception:  # pragma: no cover
    comfy_nodes = None

try:
    import server
except Exception:  # pragma: no cover
    server = None

from .vnccs_pipe import VNCCS_Pipe
from .vnccs_control_center import (
    _apply_lora_nunchaku,
    _apply_lora_standard,
    _find_model_on_disk,
    _rel_within_folder,
)
from .vnccs_qwen_encoder import VNCCS_QWEN_Encoder
from .vnccs_utils import VNCCSChromaKey, VNCCS_MaskExtractor, VNCCS_RMBG2
from ..utils import character_dir


def _folder_list(kind, fallback):
    if folder_paths is None:
        return list(fallback)
    try:
        values = folder_paths.get_filename_list(kind)
        return values or list(fallback)
    except Exception:
        return list(fallback)


def _first_tensor(value):
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        if not value:
            return None
        if all(torch.is_tensor(v) for v in value):
            return torch.cat(value, dim=0)
        return value[0]
    return value


def _call_comfy_node(class_name, **kwargs):
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if comfy_nodes else {}
    cls = mappings.get(class_name)
    if cls is None:
        local_mappings = {
            "VNCCS_QWEN_Encoder": VNCCS_QWEN_Encoder,
            "VNCCS_RMBG2": VNCCS_RMBG2,
            "VNCCSChromaKey": VNCCSChromaKey,
        }
        cls = local_mappings.get(class_name)
    if cls is None:
        raise RuntimeError(f"Required node '{class_name}' is not available")

    instance = cls()
    method_name = getattr(cls, "FUNCTION", None)
    method = getattr(instance, method_name, None) if method_name else None
    if method is None:
        for candidate in ("process", "process_image", "load_model", "loadmodel", "load", "sample", "decode"):
            method = getattr(instance, candidate, None)
            if method is not None:
                break
    if method is None:
        raise RuntimeError(f"Node '{class_name}' has no callable FUNCTION")

    signature = inspect.signature(method)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    accepted = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in signature.parameters}
    return method(**accepted)


def _tensor_to_png_data_url(image, max_items=12):
    image = _first_tensor(image)
    if image is None or not torch.is_tensor(image):
        return None
    try:
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        previews = []
        for item in tensor[:max_items]:
            array = (item.numpy() * 255).astype(np.uint8)
            mode = "RGBA" if array.shape[-1] == 4 else "RGB"
            pil = Image.fromarray(array, mode=mode)
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            previews.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
        return previews
    except Exception:
        return None


def _safe_cache_part(value, fallback="node"):
    value = str(value or "").strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe or fallback


def _character_cache_dir_from_sheets_path(sheets_path, character_name="", unique_id=None):
    character_root = _character_root_from_sheets_path(sheets_path, character_name)
    if not character_root:
        return None
    if unique_id:
        return os.path.join(character_root, "cache", "poses", _safe_cache_part(unique_id))
    return os.path.join(character_root, "cache", "poses", "shared")


def _character_root_from_sheets_path(sheets_path, character_name=""):
    if isinstance(sheets_path, list):
        sheets_path = sheets_path[0] if sheets_path else ""
    sheets_path = str(sheets_path or "").strip()
    if sheets_path:
        parts = os.path.abspath(sheets_path).split(os.sep)
        if "Sheets" in parts:
            character_root = os.sep.join(parts[:parts.index("Sheets")])
            if character_root:
                return character_root
        if os.path.isdir(sheets_path):
            return os.path.abspath(sheets_path)

    character_name = str(character_name or "").strip()
    if character_name and character_name != "Unknown":
        return character_dir(character_name)

    return None


def _costume_name_from_sheets_path(sheets_path, fallback="Naked"):
    if isinstance(sheets_path, list):
        sheets_path = sheets_path[0] if sheets_path else ""
    parts = os.path.abspath(str(sheets_path or "")).split(os.sep)
    if "Sheets" in parts:
        index = parts.index("Sheets")
        if len(parts) > index + 1 and parts[index + 1]:
            return parts[index + 1]
    return str(fallback or "Naked").strip() or "Naked"


def _view_url_for_output_path(path):
    if folder_paths is None:
        return None
    try:
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        abs_path = os.path.abspath(path)
        rel = os.path.relpath(abs_path, output_dir)
        if rel.startswith(".."):
            return None
        subfolder = os.path.dirname(rel).replace(os.sep, "/")
        query = urlencode({
            "filename": os.path.basename(rel),
            "subfolder": subfolder,
            "type": "output",
            "t": int(os.path.getmtime(abs_path)),
        })
        return f"/view?{query}"
    except Exception:
        return None


def _tensor_to_preview_urls(image, unique_id, stage, cache_dir=None, max_items=12):
    image = _first_tensor(image)
    if image is None or not torch.is_tensor(image):
        return None
    if not cache_dir or folder_paths is None:
        return _tensor_to_png_data_url(image, max_items=max_items)

    try:
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        out_dir = cache_dir
        os.makedirs(out_dir, exist_ok=True)

        previews = []
        for index, item in enumerate(tensor[:max_items], start=1):
            array = (item.numpy() * 255).astype(np.uint8)
            mode = "RGBA" if array.shape[-1] == 4 else "RGB"
            pil = Image.fromarray(array, mode=mode)
            filename = f"{stage}_{index:02d}.png"
            path = os.path.join(out_dir, filename)
            pil.save(path, format="PNG")
            previews.append(_view_url_for_output_path(path) or path)
        return previews
    except Exception:
        return _tensor_to_png_data_url(image, max_items=max_items)


def _rotate_preview_cache(cache_dir):
    if not cache_dir:
        return
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    try:
        os.makedirs(cache_dir, exist_ok=True)
        version_dir = os.path.join(cache_dir, "V1")
        if os.path.isdir(version_dir):
            shutil.rmtree(version_dir)

        files = [
            filename for filename in os.listdir(cache_dir)
            if os.path.isfile(os.path.join(cache_dir, filename))
            and os.path.splitext(filename)[1].lower() in image_exts
        ]
        if not files:
            return

        os.makedirs(version_dir, exist_ok=True)
        for filename in files:
            os.replace(os.path.join(cache_dir, filename), os.path.join(version_dir, filename))
    except Exception:
        pass


DEFAULT_WIDGET_DATA = {
    "common": {
        "target_size": "1024",
    },
    "upscaler": {
        "mode": "seedvr",
        "model": "seedvr2_ema_3b-Q4_K_M.gguf",
        "vae": "ema_vae_fp16.safetensors",
        "gan_model": "2x_APISR_RRDB_GAN_generator.pth",
        "device": "cuda:0",
        "offload_device": "cpu",
        "seed": 42,
        "resolution": 2048,
        "max_resolution": 3840,
        "batch_size": 1,
        "uniform_batch_size": False,
        "color_correction": "lab",
        "temporal_overlap": 0,
        "prepend_frames": 0,
        "input_noise_scale": 0,
        "latent_noise_scale": 0,
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "cache_dit": False,
        "attention_mode": "sdpa",
        "encode_tiled": True,
        "encode_tile_size": 1024,
        "encode_tile_overlap": 128,
        "decode_tiled": True,
        "decode_tile_size": 1024,
        "decode_tile_overlap": 128,
        "tile_debug": "false",
        "cache_vae": False,
        "enable_debug": False,
    },
    "bg_remove": {
        "tolerance": 0.5,
        "despill_strength": 1.0,
        "despill_kernel_size": 3,
        "despill_color": "black",
    },
    "pose_generation": {
        "target_size": "1024",
    },
    "emotion_generation": {
        "face_denoise": 0.55,
    },
    "remove_clothes": {
        "prompt": "Dress character: White underwear",
    },
}

POSE_GENERATION_LORA_NAME = "VNCCS Pose Studio QIE2511"
CLOTHES_CORE_LORA_NAME = "VNCCS Clothes Core"


class VNCCS_CharacterGenerator:
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("IMAGE",),
                "character": ("IMAGE",),
                "pipe": ("VNCCS_PIPE",),
                "prompt": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": True}),
                "background": (["Green", "Blue", "White", "Alpha"], {"default": "Green"}),
                "widget_data": ("STRING", {"default": json.dumps(DEFAULT_WIDGET_DATA), "multiline": True}),
            },
            "optional": {
                "sheets_path": ("STRING", {"default": "", "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("sheet", "faces", "pose_generation", "upscaled")
    FUNCTION = "process"
    CATEGORY = "VNCCS"
    DESCRIPTION = "Replacement for VNCCS Pose Generation, Upscaler, and BG Remove subgraphs."

    def _settings(self, widget_data):
        widget_data = self._unwrap_scalar(widget_data)
        data = json.loads(widget_data) if isinstance(widget_data, str) and widget_data.strip() else {}
        merged = json.loads(json.dumps(DEFAULT_WIDGET_DATA))
        for section, values in (data or {}).items():
            if isinstance(values, dict) and section in merged:
                merged[section].update(values)
        return merged

    def _widget_data(self, widget_data):
        widget_data = self._unwrap_scalar(widget_data)
        try:
            return json.loads(widget_data) if isinstance(widget_data, str) and widget_data.strip() else {}
        except Exception:
            return {}

    def _unwrap_scalar(self, value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _emit(self, unique_id, stage, status, images=None, message="", current=None, total=None, cache_dir=None, lora_info=None):
        if server is None or not unique_id:
            return
        payload = {
            "node_id": str(unique_id),
            "stage": stage,
            "status": status,
            "message": message,
        }
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if lora_info is not None:
            payload["lora_info"] = lora_info
        if images is not None:
            payload["images"] = _tensor_to_preview_urls(images, unique_id, stage, cache_dir=cache_dir)
        try:
            server.PromptServer.instance.send_sync("vnccs.character_generator.stage", payload)
        except Exception:
            pass

    def _extract_pipe(self, pipe):
        out = VNCCS_Pipe().process_pipe(pipe=pipe)
        return {
            "model": out[0],
            "clip": out[1],
            "vae": out[2],
            "seed": int(out[5] or 0),
            "steps": int(out[6] or 1),
            "cfg": float(out[7] or 1.0),
            "sampler": out[10] or "euler",
            "scheduler": out[11] or "simple",
        }

    def _find_lora(self, pipe, lora_name):
        entries = getattr(pipe, "lora_entries", []) or []
        states = getattr(pipe, "lora_states", []) or []
        entry = None
        target = str(lora_name or "").strip().lower()
        for candidate in entries:
            candidate_name = str(candidate.get("name", "")).strip().lower()
            if candidate_name == target or target in candidate_name:
                entry = candidate
                break

        if not entry:
            return {
                "name": lora_name,
                "file": "",
                "status": "missing",
                "message": f"{lora_name}: not found in Control Center",
            }

        state = next(
            (
                item for item in states
                if str(item.get("name", "")).strip().lower() == target
                or target in str(item.get("name", "")).strip().lower()
            ),
            {},
        )
        strength = float(state.get("strength", 1.0) if state else 1.0)
        full_path, exists = _find_model_on_disk(entry.get("local_path", ""))
        filename = os.path.basename(full_path or entry.get("local_path", ""))
        rel_path = _rel_within_folder(entry.get("local_path", ""))
        return {
            "name": entry.get("name", lora_name),
            "file": filename,
            "path": full_path,
            "rel_path": rel_path,
            "strength": strength,
            "exists": bool(exists),
            "status": "ready" if exists else "missing",
            "message": f"{entry.get('name', lora_name)}: {filename or 'not found'}",
        }

    def _find_pose_lora(self, pipe):
        return self._find_lora(pipe, POSE_GENERATION_LORA_NAME)

    def _find_clothes_lora(self, pipe):
        return self._find_lora(pipe, CLOTHES_CORE_LORA_NAME)

    def _apply_lora_to_model(self, model, clip, pipe, lora_info, stage_label):
        if not lora_info or not lora_info.get("exists") or not lora_info.get("path"):
            message = lora_info.get("message") if lora_info else stage_label
            raise RuntimeError(f"{stage_label} requires LoRA from VNCCS Control Center: {message}")
        strength = float(lora_info.get("strength", 1.0))
        loader_type = getattr(pipe, "loader_type", "standard") or "standard"
        print(f"[VNCCS Character Generator] Applying {stage_label} LoRA before sampler: {lora_info.get('name')} ({lora_info.get('file')}, strength={strength})")
        if loader_type == "nunchaku":
            return _apply_lora_nunchaku(
                model,
                lora_info["path"],
                strength,
                settings=getattr(pipe, "nunchaku_settings", None) or {},
                model_entry=getattr(pipe, "model_entry", None),
            )
        rel_path = lora_info.get("rel_path") or lora_info.get("file")
        try:
            return _call_comfy_node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=rel_path,
                strength_model=strength,
            )[0]
        except Exception as exc:
            print(f"[VNCCS Character Generator] LoraLoaderModelOnly failed for '{rel_path}', using direct loader: {exc}")
        model_lora, _ = _apply_lora_standard(model, clip, lora_info["path"], strength)
        return model_lora

    def _apply_pose_lora_to_model(self, model, clip, pipe, lora_info):
        return self._apply_lora_to_model(model, clip, pipe, lora_info, "Pose Generation")

    def _list_to_batch(self, values):
        if torch.is_tensor(values):
            return values if values.ndim == 4 else values.unsqueeze(0)
        if isinstance(values, list):
            tensors = []
            for value in values:
                value = self._list_to_batch(value)
                if value is not None:
                    tensors.append(value)
            if tensors:
                return torch.cat(tensors, dim=0)
        return values

    def _split_batch(self, images):
        images = self._list_to_batch(images)
        if images is None or not torch.is_tensor(images):
            return []
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return [images[i:i + 1] for i in range(images.shape[0])]

    def _image_list(self, images):
        if isinstance(images, tuple):
            images = images[0]
        if isinstance(images, list):
            result = []
            for image in images:
                result.extend(self._image_list(image))
            return result
        if torch.is_tensor(images):
            if images.ndim == 3:
                return [images.unsqueeze(0)]
            if images.ndim == 4:
                return [images[i:i + 1] for i in range(images.shape[0])]
        return []

    def _run_list_mapped(self, class_name, list_kwargs, **kwargs):
        count = max((len(v) for v in list_kwargs.values()), default=0)
        outputs = None
        for index in range(count):
            call_kwargs = dict(kwargs)
            for key, values in list_kwargs.items():
                call_kwargs[key] = values[index]
            result = _call_comfy_node(class_name, **call_kwargs)
            if outputs is None:
                outputs = [[] for _ in result]
            for out_index, value in enumerate(result):
                outputs[out_index].append(value)
        return tuple(outputs or [])

    def _run_pose_generation(self, poses, character, pipe, prompt, settings, lora_info=None):
        pipe_values = self._extract_pipe(pipe)
        pose_parts = self._image_list(poses)
        character_rgb = VNCCS_MaskExtractor().fill_alpha_with_color(character)[0]

        positive_list, negative_list, latent_list = self._run_list_mapped(
            "VNCCS_QWEN_Encoder",
            {"image1": pose_parts},
            clip=pipe_values["clip"],
            vae=pipe_values["vae"],
            prompt=prompt,
            image2=character_rgb,
            target_size=int(settings["target_size"]),
            upscale_method="lanczos",
            crop_method="disabled",
            image1_name="image 1",
            image2_name="image 2",
            image3_name="image 3",
            weight1=1,
            weight2=1,
            weight3=1,
            vl_size=384,
            latent_image_index=1,
            instruction=(
                "Describe the character and their key features (body shape, physical characteristics, clothing, "
                "items, accessories). Then explain how the user's text instruction should alter or modify the "
                "character. Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original character where appropriate."
            ),
            qwen_2511=True,
        )

        sampler_model = self._apply_pose_lora_to_model(pipe_values["model"], pipe_values["clip"], pipe, lora_info)
        sampled_list = self._run_list_mapped(
            "KSampler",
            {"positive": positive_list, "negative": negative_list, "latent_image": latent_list},
            model=sampler_model,
            seed=pipe_values["seed"],
            steps=pipe_values["steps"],
            cfg=pipe_values["cfg"],
            sampler_name=pipe_values["sampler"],
            scheduler=pipe_values["scheduler"],
            denoise=1,
        )[0]

        decoded_list = self._run_list_mapped(
            "VAEDecodeTiled",
            {"samples": sampled_list},
            vae=pipe_values["vae"],
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )[0]

        return torch.cat([self._list_to_batch(image) for image in decoded_list], dim=0)

    def _run_remove_clothes(self, character, pipe, settings, lora_info=None):
        pipe_values = self._extract_pipe(pipe)
        character_rgb = VNCCS_MaskExtractor().fill_alpha_with_color(character)[0]

        positive, negative, latent = _call_comfy_node(
            "VNCCS_QWEN_Encoder",
            clip=pipe_values["clip"],
            vae=pipe_values["vae"],
            prompt=settings.get("prompt", DEFAULT_WIDGET_DATA["remove_clothes"]["prompt"]),
            image1=character_rgb,
            target_size=int(settings.get("target_size") or DEFAULT_WIDGET_DATA["common"]["target_size"]),
            upscale_method="lanczos",
            crop_method="disabled",
            image1_name="image 1",
            image2_name="image 2",
            image3_name="image 3",
            weight1=1,
            weight2=1,
            weight3=1,
            vl_size=384,
            latent_image_index=1,
            instruction=(
                "Describe the character and their key features (body shape, physical characteristics, clothing, "
                "items, accessories). Then explain how the user's text instruction should alter or modify the "
                "character. Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original character where appropriate."
            ),
            qwen_2511=True,
        )

        sampler_model = self._apply_lora_to_model(
            pipe_values["model"],
            pipe_values["clip"],
            pipe,
            lora_info,
            "Remove Clothes",
        )
        sampled = _call_comfy_node(
            "KSampler",
            model=sampler_model,
            positive=positive,
            negative=negative,
            latent_image=latent,
            seed=pipe_values["seed"],
            steps=pipe_values["steps"],
            cfg=pipe_values["cfg"],
            sampler_name=pipe_values["sampler"],
            scheduler=pipe_values["scheduler"],
            denoise=1,
        )[0]

        return _call_comfy_node(
            "VAEDecodeTiled",
            samples=sampled,
            vae=pipe_values["vae"],
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )[0]

    def _run_upscaler_models(self, settings):
        dit = _call_comfy_node(
            "SeedVR2LoadDiTModel",
            model=settings["model"],
            device=settings["device"],
            blocks_to_swap=int(settings["blocks_to_swap"]),
            swap_io_components=bool(settings["swap_io_components"]),
            offload_device=settings["offload_device"],
            cache_model=bool(settings["cache_dit"]),
            attention_mode=settings["attention_mode"],
        )[0]
        vae = _call_comfy_node(
            "SeedVR2LoadVAEModel",
            model=settings["vae"],
            device=settings["device"],
            encode_tiled=bool(settings["encode_tiled"]),
            encode_tile_size=int(settings["encode_tile_size"]),
            encode_tile_overlap=int(settings["encode_tile_overlap"]),
            decode_tiled=bool(settings["decode_tiled"]),
            decode_tile_size=int(settings["decode_tile_size"]),
            decode_tile_overlap=int(settings["decode_tile_overlap"]),
            tile_debug=settings["tile_debug"],
            offload_device=settings["offload_device"],
            cache_model=bool(settings["cache_vae"]),
        )[0]
        return dit, vae

    def _run_gan_upscaler_model(self, settings):
        return _call_comfy_node(
            "UpscaleModelLoader",
            model_name=settings["gan_model"],
        )[0]

    def _run_upscale_one(self, image, dit, vae, background, settings):
        upscaled = self._run_seedvr_upscale_one(image, dit, vae, settings)
        return VNCCS_RMBG2().process_image(
            upscaled,
            "RMBG-2.0",
            sensitivity=0.85,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background=str(background or "Green"),
        )[0]

    def _run_seedvr_upscale_one(self, image, dit, vae, settings):
        return _call_comfy_node(
            "SeedVR2VideoUpscaler",
            image=image,
            dit=dit,
            vae=vae,
            seed=int(settings["seed"]),
            resolution=int(settings["resolution"]),
            max_resolution=int(settings["max_resolution"]),
            batch_size=int(settings["batch_size"]),
            uniform_batch_size=bool(settings["uniform_batch_size"]),
            color_correction=settings["color_correction"],
            temporal_overlap=int(settings["temporal_overlap"]),
            prepend_frames=int(settings["prepend_frames"]),
            input_noise_scale=float(settings["input_noise_scale"]),
            latent_noise_scale=float(settings["latent_noise_scale"]),
            offload_device=settings["offload_device"],
            enable_debug=bool(settings["enable_debug"]),
        )[0]

    def _run_gan_upscale_one(self, image, upscale_model):
        return _call_comfy_node(
            "ImageUpscaleWithModel",
            upscale_model=upscale_model,
            image=image,
        )[0]

    def _run_upscaler(self, image, background, settings, unique_id=None, cache_dir=None, stage="upscaler"):
        images = self._split_batch(image)
        total = len(images)
        mode = str(settings.get("mode", "seedvr") or "seedvr").lower()
        if mode == "gan":
            model = self._run_gan_upscaler_model(settings)
            results = []
            for index, item in enumerate(images, start=1):
                result = self._run_gan_upscale_one(item, model)
                results.append(self._list_to_batch(result))
                partial = torch.cat(results, dim=0)
                self._emit(
                    unique_id,
                    stage,
                    "running" if index < total else "done",
                    partial,
                    f"GAN upscaled image {index} of {total}",
                    index,
                    total,
                    cache_dir=cache_dir,
                )
            return torch.cat(results, dim=0) if results else image

        dit, vae = self._run_upscaler_models(settings)
        results = []
        for index, item in enumerate(images, start=1):
            result = self._run_upscale_one(item, dit, vae, background, settings)
            results.append(self._list_to_batch(result))
            partial = torch.cat(results, dim=0)
            self._emit(
                unique_id,
                stage,
                "running" if index < total else "done",
                partial,
                f"Upscaled image {index} of {total}",
                index,
                total,
                cache_dir=cache_dir,
            )
        return torch.cat(results, dim=0) if results else image

    def _run_source_upscaler(self, image, settings, unique_id=None, cache_dir=None, stage="source_upscaler"):
        images = self._split_batch(image)
        total = len(images)
        mode = str(settings.get("mode", "seedvr") or "seedvr").lower()
        if mode == "gan":
            model = self._run_gan_upscaler_model(settings)
            results = []
            for index, item in enumerate(images, start=1):
                result = self._run_gan_upscale_one(item, model)
                results.append(self._list_to_batch(result))
                partial = torch.cat(results, dim=0)
                self._emit(
                    unique_id,
                    stage,
                    "running" if index < total else "done",
                    partial,
                    f"GAN upscaled source image {index} of {total}",
                    index,
                    total,
                    cache_dir=cache_dir,
                )
            return torch.cat(results, dim=0) if results else image

        dit, vae = self._run_upscaler_models(settings)
        results = []
        for index, item in enumerate(images, start=1):
            result = self._run_seedvr_upscale_one(item, dit, vae, settings)
            results.append(self._list_to_batch(result))
            partial = torch.cat(results, dim=0)
            self._emit(
                unique_id,
                stage,
                "running" if index < total else "done",
                partial,
                f"Upscaled source image {index} of {total}",
                index,
                total,
                cache_dir=cache_dir,
            )
        return torch.cat(results, dim=0) if results else image

    def _run_bg_remove(self, images, settings):
        return VNCCSChromaKey().chroma_key(
            images,
            float(settings["tolerance"]),
            float(settings["despill_strength"]),
            int(settings["despill_kernel_size"]),
            settings["despill_color"],
        )[0]

    def _tensor_item_to_pil(self, image):
        array = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        mode = "RGBA" if array.shape[-1] == 4 else "RGB"
        return Image.fromarray(array, mode=mode)

    def _version_dir(self, target_dir):
        version = 1
        while True:
            candidate = os.path.join(target_dir, f"V{version}")
            if not os.path.exists(candidate):
                return candidate
            version += 1

    def _save_final_sprites(self, images, sheets_path, character_name="", sprite_set="Naked"):
        character_root = _character_root_from_sheets_path(sheets_path, character_name)
        if not character_root:
            return []

        images = self._list_to_batch(images)
        if images is None or not torch.is_tensor(images):
            return []
        if images.ndim == 3:
            images = images.unsqueeze(0)

        sprite_set = str(sprite_set or "Naked").strip() or "Naked"
        target_dir = os.path.join(character_root, "Sprites", sprite_set, "Neutral")
        os.makedirs(target_dir, exist_ok=True)

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        existing_images = [
            filename for filename in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, filename))
            and os.path.splitext(filename)[1].lower() in image_exts
        ]
        if existing_images:
            version_dir = self._version_dir(target_dir)
            os.makedirs(version_dir, exist_ok=True)
            for filename in existing_images:
                src = os.path.join(target_dir, filename)
                os.replace(src, os.path.join(version_dir, filename))

        saved = []
        for index, image in enumerate(images, start=1):
            filename = f"sprite_pose_{index:04d}.png"
            path = os.path.join(target_dir, filename)
            self._tensor_item_to_pil(image).save(path, format="PNG")
            saved.append(path)
        return saved

    def process(self, poses, character, pipe, prompt, background="Green", widget_data="{}", sheets_path="", unique_id=None):
        settings = self._settings(widget_data)
        widget_payload = self._widget_data(widget_data)
        character_name = widget_payload.get("character_name", "")
        character = self._unwrap_scalar(character)
        pipe = self._unwrap_scalar(pipe)
        prompt = self._unwrap_scalar(prompt)
        background = self._unwrap_scalar(background)
        sheets_path = self._unwrap_scalar(sheets_path)
        unique_id = self._unwrap_scalar(unique_id)
        cache_dir = _character_cache_dir_from_sheets_path(sheets_path, widget_payload.get("character_name", ""), unique_id)
        try:
            _rotate_preview_cache(cache_dir)
            pose_lora_info = self._find_pose_lora(pipe)
            input_total = len(self._image_list(poses))
            self._emit(
                unique_id,
                "pose_generation",
                "running",
                message="Encoding pose list",
                current=0,
                total=input_total,
                cache_dir=cache_dir,
                lora_info=pose_lora_info,
            )
            pose_images = self._run_pose_generation(poses, character, pipe, prompt, settings["pose_generation"], lora_info=pose_lora_info)
            pose_total = pose_images.shape[0] if torch.is_tensor(pose_images) and pose_images.ndim == 4 else 0
            self._emit(unique_id, "pose_generation", "done", pose_images, f"Generated {pose_total} pose images", pose_total, pose_total, cache_dir=cache_dir, lora_info=pose_lora_info)

            up_total = pose_total
            self._emit(unique_id, "upscaler", "running", pose_images, f"Preparing upscaler for {up_total} images", 0, up_total, cache_dir=cache_dir)
            upscaled = self._run_upscaler(pose_images, background, settings["upscaler"], unique_id=unique_id, cache_dir=cache_dir)

            bg_total = upscaled.shape[0] if torch.is_tensor(upscaled) and upscaled.ndim == 4 else 0
            self._emit(unique_id, "bg_remove", "running", upscaled, f"Removing background for {bg_total} images", 0, bg_total, cache_dir=cache_dir)
            final_images = self._run_bg_remove(upscaled, settings["bg_remove"])
            saved_paths = self._save_final_sprites(final_images, sheets_path, character_name)
            saved_suffix = f"; saved {len(saved_paths)} sprites" if saved_paths else ""
            self._emit(unique_id, "bg_remove", "done", final_images, f"Background removed from {bg_total} images{saved_suffix}", bg_total, bg_total, cache_dir=cache_dir)
            return final_images, final_images, pose_images, upscaled
        except Exception as exc:
            print("[VNCCS Character Generator] Failed:", exc)
            traceback.print_exc()
            self._emit(unique_id, "error", "error", message=str(exc))
            raise


class VNCCS_CharacterCloneGenerator(VNCCS_CharacterGenerator):
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("IMAGE",),
                "character": ("IMAGE",),
                "pipe": ("VNCCS_PIPE",),
                "prompt": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": True}),
                "background": (["Green", "Blue", "White", "Alpha"], {"default": "Green"}),
                "widget_data": ("STRING", {"default": json.dumps(DEFAULT_WIDGET_DATA), "multiline": True}),
            },
            "optional": {
                "sheets_path": ("STRING", {"default": "", "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "original_sprites",
        "faces",
        "naked_sprites",
        "original_pose_generation",
        "original_upscaled",
        "remove_clothes",
        "naked_pose_generation",
    )
    FUNCTION = "process"
    CATEGORY = "VNCCS"
    DESCRIPTION = "Clone generator that creates original-clothes and cleaned-clothes sprite sets."

    def _clone_settings(self, widget_data):
        settings = self._settings(widget_data)
        common_size = settings.get("common", {}).get("target_size") or settings["pose_generation"].get("target_size", "1024")
        settings["pose_generation"]["target_size"] = common_size
        settings["remove_clothes"]["target_size"] = common_size
        return settings

    def _run_sprite_branch(self, poses, character, pipe, prompt, background, settings, unique_id, cache_dir, stage_prefix, pose_lora_info):
        pose_stage = f"{stage_prefix}_pose_generation"
        up_stage = f"{stage_prefix}_upscaler"
        bg_stage = f"{stage_prefix}_bg_remove"

        input_total = len(self._image_list(poses))
        self._emit(
            unique_id,
            pose_stage,
            "running",
            message="Encoding pose list",
            current=0,
            total=input_total,
            cache_dir=cache_dir,
            lora_info=pose_lora_info,
        )
        pose_images = self._run_pose_generation(poses, character, pipe, prompt, settings["pose_generation"], lora_info=pose_lora_info)
        pose_total = pose_images.shape[0] if torch.is_tensor(pose_images) and pose_images.ndim == 4 else 0
        self._emit(unique_id, pose_stage, "done", pose_images, f"Generated {pose_total} pose images", pose_total, pose_total, cache_dir=cache_dir, lora_info=pose_lora_info)

        self._emit(unique_id, up_stage, "running", pose_images, f"Preparing upscaler for {pose_total} images", 0, pose_total, cache_dir=cache_dir)
        upscaled = self._run_upscaler(
            pose_images,
            background,
            settings["upscaler"],
            unique_id=unique_id,
            cache_dir=cache_dir,
            stage=up_stage,
        )

        bg_total = upscaled.shape[0] if torch.is_tensor(upscaled) and upscaled.ndim == 4 else 0
        self._emit(unique_id, bg_stage, "running", upscaled, f"Removing background for {bg_total} images", 0, bg_total, cache_dir=cache_dir)
        final_images = self._run_bg_remove(upscaled, settings["bg_remove"])
        self._emit(unique_id, bg_stage, "done", final_images, f"Background removed from {bg_total} images", bg_total, bg_total, cache_dir=cache_dir)
        return final_images, pose_images, upscaled

    def process(self, poses, character, pipe, prompt, background="Green", widget_data="{}", sheets_path="", unique_id=None):
        settings = self._clone_settings(widget_data)
        widget_payload = self._widget_data(widget_data)
        character_name = widget_payload.get("character_name", "")
        nsfw_value = widget_payload.get("nsfw_enabled", True)
        if isinstance(nsfw_value, str):
            nsfw_enabled = nsfw_value.strip().lower() in ("true", "1", "yes", "on")
        else:
            nsfw_enabled = bool(nsfw_value)
        character = self._unwrap_scalar(character)
        pipe = self._unwrap_scalar(pipe)
        prompt = self._unwrap_scalar(prompt)
        background = self._unwrap_scalar(background)
        sheets_path = self._unwrap_scalar(sheets_path)
        unique_id = self._unwrap_scalar(unique_id)
        cache_dir = _character_cache_dir_from_sheets_path(sheets_path, widget_payload.get("character_name", ""), unique_id)
        try:
            _rotate_preview_cache(cache_dir)
            pose_lora_info = self._find_pose_lora(pipe)

            original_final, original_pose, original_upscaled = self._run_sprite_branch(
                poses,
                character,
                pipe,
                prompt,
                background,
                settings,
                unique_id,
                cache_dir,
                "original",
                pose_lora_info,
            )
            original_saved = self._save_final_sprites(original_final, sheets_path, character_name, "Original")
            if original_saved:
                self._emit(unique_id, "original_bg_remove", "done", original_final, f"Saved {len(original_saved)} original sprites", cache_dir=cache_dir)

            if not nsfw_enabled:
                return original_final, original_final, original_final, original_pose, original_upscaled, character, original_pose

            clothes_lora_info = self._find_clothes_lora(pipe)
            remove_total = 1
            self._emit(
                unique_id,
                "remove_clothes",
                "running",
                character,
                "Removing clothes from source character",
                0,
                remove_total,
                cache_dir=cache_dir,
                lora_info=clothes_lora_info,
            )
            naked_character = self._run_remove_clothes(character, pipe, settings["remove_clothes"], lora_info=clothes_lora_info)
            self._emit(
                unique_id,
                "remove_clothes",
                "done",
                naked_character,
                "Source character cleaned",
                remove_total,
                remove_total,
                cache_dir=cache_dir,
                lora_info=clothes_lora_info,
            )

            naked_final, naked_pose, _naked_upscaled = self._run_sprite_branch(
                poses,
                naked_character,
                pipe,
                prompt,
                background,
                settings,
                unique_id,
                cache_dir,
                "naked",
                pose_lora_info,
            )
            naked_saved = self._save_final_sprites(naked_final, sheets_path, character_name, "Naked")
            if naked_saved:
                self._emit(unique_id, "naked_bg_remove", "done", naked_final, f"Saved {len(naked_saved)} naked sprites", cache_dir=cache_dir)

            return original_final, original_final, naked_final, original_pose, original_upscaled, naked_character, naked_pose
        except Exception as exc:
            print("[VNCCS Character Clone Generator] Failed:", exc)
            traceback.print_exc()
            self._emit(unique_id, "error", "error", message=str(exc))
            raise


class VNCCS_ClothesGenerator(VNCCS_CharacterGenerator):
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("IMAGE",),
                "character": ("IMAGE",),
                "pipe": ("VNCCS_PIPE",),
                "prompt": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": True}),
                "background": (["Green", "Blue", "White", "Alpha"], {"default": "Green"}),
                "widget_data": ("STRING", {"default": json.dumps(DEFAULT_WIDGET_DATA), "multiline": True}),
            },
            "optional": {
                "sheets_path": ("STRING", {"default": "", "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("sprites", "faces", "source_upscaled", "pose_generation", "upscaled")
    FUNCTION = "process"
    CATEGORY = "VNCCS"
    DESCRIPTION = "Clothes sprite generator with source upscale followed by pose generation, upscale, and BG remove."

    def _run_clothes_pose_generation(self, poses, character, pipe, prompt, background, settings, lora_info=None):
        pose_images = self._run_pose_generation(poses, character, pipe, prompt, settings, lora_info=lora_info)
        return VNCCS_RMBG2().process_image(
            pose_images,
            "RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=True,
            background=str(background or "Green"),
        )[0]

    def process(self, poses, character, pipe, prompt, background="Green", widget_data="{}", sheets_path="", unique_id=None):
        settings = self._settings(widget_data)
        widget_payload = self._widget_data(widget_data)
        character_name = widget_payload.get("character_name", "")
        character = self._unwrap_scalar(character)
        pipe = self._unwrap_scalar(pipe)
        prompt = self._unwrap_scalar(prompt)
        background = self._unwrap_scalar(background)
        sheets_path = self._unwrap_scalar(sheets_path)
        unique_id = self._unwrap_scalar(unique_id)
        costume_name = _costume_name_from_sheets_path(
            sheets_path,
            widget_payload.get("costume") or widget_payload.get("costume_name") or "Naked",
        )
        cache_dir = _character_cache_dir_from_sheets_path(sheets_path, widget_payload.get("character_name", ""), unique_id)
        try:
            _rotate_preview_cache(cache_dir)
            pose_lora_info = self._find_pose_lora(pipe)

            source_total = len(self._image_list(character)) or 1
            self._emit(
                unique_id,
                "source_upscaler",
                "running",
                character,
                f"Preparing source upscaler for {source_total} image",
                0,
                source_total,
                cache_dir=cache_dir,
            )
            source_upscaled = self._run_source_upscaler(
                character,
                settings["upscaler"],
                unique_id=unique_id,
                cache_dir=cache_dir,
                stage="source_upscaler",
            )

            input_total = len(self._image_list(poses))
            self._emit(
                unique_id,
                "pose_generation",
                "running",
                message="Encoding pose list",
                current=0,
                total=input_total,
                cache_dir=cache_dir,
                lora_info=pose_lora_info,
            )
            pose_images = self._run_clothes_pose_generation(
                poses,
                source_upscaled,
                pipe,
                prompt,
                background,
                settings["pose_generation"],
                lora_info=pose_lora_info,
            )
            pose_total = pose_images.shape[0] if torch.is_tensor(pose_images) and pose_images.ndim == 4 else 0
            self._emit(unique_id, "pose_generation", "done", pose_images, f"Generated {pose_total} clothed pose images", pose_total, pose_total, cache_dir=cache_dir, lora_info=pose_lora_info)

            self._emit(unique_id, "upscaler", "running", pose_images, f"Preparing upscaler for {pose_total} images", 0, pose_total, cache_dir=cache_dir)
            upscaled = self._run_upscaler(pose_images, background, settings["upscaler"], unique_id=unique_id, cache_dir=cache_dir)

            bg_total = upscaled.shape[0] if torch.is_tensor(upscaled) and upscaled.ndim == 4 else 0
            self._emit(unique_id, "bg_remove", "running", upscaled, f"Removing background for {bg_total} images", 0, bg_total, cache_dir=cache_dir)
            final_images = self._run_bg_remove(upscaled, settings["bg_remove"])
            saved_paths = self._save_final_sprites(final_images, sheets_path, character_name, costume_name)
            saved_suffix = f"; saved {len(saved_paths)} sprites to {costume_name}" if saved_paths else ""
            self._emit(unique_id, "bg_remove", "done", final_images, f"Background removed from {bg_total} images{saved_suffix}", bg_total, bg_total, cache_dir=cache_dir)
            return final_images, final_images, source_upscaled, pose_images, upscaled
        except Exception as exc:
            print("[VNCCS Clothes Generator] Failed:", exc)
            traceback.print_exc()
            self._emit(unique_id, "error", "error", message=str(exc))
            raise


class VNCCS_EmotionsGenerator(VNCCS_CharacterGenerator):
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pipe": ("VNCCS_PIPE",),
                "emotion_data": ("STRING", {"default": "", "forceInput": True}),
                "widget_data": ("STRING", {"default": json.dumps(DEFAULT_WIDGET_DATA), "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("sprites", "faces")
    FUNCTION = "process"
    CATEGORY = "VNCCS"
    DESCRIPTION = "Emotion sprite generator that replaces the Step 3 emotion workflow."

    def _as_list(self, value):
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, list):
            return value
        return [value]

    def _mask_from_source_path(self, path):
        path = str(path or "").strip()
        if not path or not os.path.exists(path):
            return None
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            has_alpha = img.mode == "RGBA" or img.mode == "LA" or (img.mode == "P" and "transparency" in img.info)
            if not has_alpha:
                return None
            alpha = img.convert("RGBA").getchannel("A")
            arr = np.array(alpha).astype(np.float32) / 255.0
            return torch.from_numpy(1.0 - arr).unsqueeze(0)
        except Exception as exc:
            print(f"[VNCCS Emotions Generator] Failed to load source mask '{path}': {exc}")
            return None

    def _load_source_sprite_from_path(self, path):
        path = str(path or "").strip()
        if not path or not os.path.exists(path):
            return None, None
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            has_alpha = img.mode == "RGBA" or img.mode == "LA" or (img.mode == "P" and "transparency" in img.info)
            img = img.convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(arr[..., :3]).unsqueeze(0)
            mask = torch.from_numpy(1.0 - arr[..., 3]).unsqueeze(0) if has_alpha else None
            return image, mask
        except Exception as exc:
            print(f"[VNCCS Emotions Generator] Failed to load source sprite '{path}': {exc}")
            return None, None

    def _parse_emotion_data(self, emotion_data):
        items = []
        for raw in self._as_list(emotion_data):
            if isinstance(raw, dict):
                items.append(raw)
                continue
            text = str(raw or "").strip()
            if not text:
                items.append({})
                continue
            try:
                parsed = json.loads(text)
                items.append(parsed if isinstance(parsed, dict) else {"emotion_prompt": text})
            except Exception:
                items.append({"emotion_prompt": text})
        return items

    def _face_prefix_from_sprite_prefix(self, sprite_prefix):
        prefix = str(sprite_prefix or "").strip()
        if not prefix:
            return ""
        normalized = prefix.replace("\\", os.sep).replace("/", os.sep)
        parts = normalized.split(os.sep)
        try:
            idx = parts.index("Sprites")
            parts[idx] = "Faces"
        except ValueError:
            return ""
        basename = parts[-1]
        if basename.startswith("sprite_"):
            parts[-1] = "face_" + basename[len("sprite_"):]
        return os.sep.join(parts)

    def _emotion_pairs(self, widget_payload, emotions, total):
        pairs = widget_payload.get("emotion_pairs")
        if isinstance(pairs, list) and pairs:
            labels = []
            for index, pair in enumerate(pairs[:total], start=1):
                if isinstance(pair, dict):
                    costume = pair.get("costume") or "Costume"
                    emotion = pair.get("emotion") or f"Emotion {index}"
                    labels.append((f"emotion_{index:04d}", f"{costume} / {emotion}"))
            if labels:
                while len(labels) < total:
                    index = len(labels) + 1
                    labels.append((f"emotion_{index:04d}", f"Emotion {index}"))
                return labels
        labels = []
        for index, prompt in enumerate(emotions[:total], start=1):
            label = str(prompt or "").splitlines()[0].replace("Change emotion:", "").strip() or f"Emotion {index}"
            labels.append((f"emotion_{index:04d}", label[:64]))
        return labels

    def _save_rgba_image(self, image, mask, prefix, index):
        prefix = str(prefix or "").strip()
        if not prefix:
            return ""
        directory = os.path.dirname(prefix)
        os.makedirs(directory, exist_ok=True)
        filename = f"{os.path.basename(prefix)}{index:04d}.png"
        path = os.path.join(directory, filename)

        rgb = image.detach().cpu().clamp(0, 1)
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb_arr = (rgb.numpy() * 255).astype(np.uint8)

        if mask is None:
            alpha = np.full(rgb_arr.shape[:2], 255, dtype=np.uint8)
        else:
            m = mask.detach().cpu().clamp(0, 1)
            if m.ndim == 3:
                m = m[0]
            alpha = ((1.0 - m.numpy()) * 255).astype(np.uint8)
            if alpha.shape != rgb_arr.shape[:2]:
                alpha = np.array(Image.fromarray(alpha).resize((rgb_arr.shape[1], rgb_arr.shape[0]), Image.Resampling.LANCZOS))

        Image.fromarray(np.dstack([rgb_arr[..., :3], alpha]), mode="RGBA").save(path, format="PNG")
        return path

    def _run_emotion_generation_one(self, image, mask, pipe, emotion_prompt, positive_prompt, negative_prompt, seed, face_denoise=0.55):
        pipe_values = self._extract_pipe(pipe)

        positive = _call_comfy_node(
            "CLIPTextEncode",
            clip=pipe_values["clip"],
            text=str(positive_prompt or ""),
        )[0]
        negative = _call_comfy_node(
            "CLIPTextEncode",
            clip=pipe_values["clip"],
            text=str(negative_prompt or ""),
        )[0]

        bbox_detector = _call_comfy_node(
            "UltralyticsDetectorProvider",
            model_name="bbox/face_yolov8m.pt",
        )[0]
        sam_model = _call_comfy_node(
            "SAMLoader",
            model_name="sam_vit_b_01ec64.pth",
            device_mode="AUTO",
        )[0]
        segm_detector = _call_comfy_node(
            "UltralyticsDetectorProvider",
            model_name="bbox/face_yolov8m.pt",
        )[1]

        detailed = _call_comfy_node(
            "FaceDetailer",
            image=image,
            model=pipe_values["model"],
            clip=pipe_values["clip"],
            vae=pipe_values["vae"],
            positive=positive,
            negative=negative,
            bbox_detector=bbox_detector,
            sam_model_opt=sam_model,
            segm_detector_opt=segm_detector,
            guide_size=1536,
            guide_size_for=True,
            max_size=1536,
            seed=int(seed or pipe_values["seed"]),
            steps=pipe_values["steps"],
            cfg=pipe_values["cfg"],
            sampler_name=pipe_values["sampler"],
            scheduler=pipe_values["scheduler"],
            denoise=float(face_denoise),
            feather=50,
            noise_mask=True,
            force_inpaint=True,
            bbox_threshold=0.1,
            bbox_dilation=10,
            bbox_crop_factor=1.7,
            sam_detection_hint="center-1",
            sam_dilation=25,
            sam_threshold=0.93,
            sam_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            sam_mask_hint_use_negative="False",
            drop_size=10,
            wildcard=str(emotion_prompt or ""),
            cycle=1,
            inpaint_model=False,
            noise_mask_feather=20,
            tiled_encode=True,
            tiled_decode=True,
        )
        full_image = self._list_to_batch(detailed[0])
        face_crop = self._list_to_batch(detailed[1]) if len(detailed) > 1 and detailed[1] is not None else full_image
        return full_image, face_crop

    def process(self, images, pipe, emotion_data, widget_data="{}", unique_id=None):
        widget_payload = self._widget_data(widget_data)
        emotion_settings = widget_payload.get("emotion_generation", {}) if isinstance(widget_payload, dict) else {}
        try:
            face_denoise = float(emotion_settings.get("face_denoise", DEFAULT_WIDGET_DATA["emotion_generation"]["face_denoise"]))
        except Exception:
            face_denoise = DEFAULT_WIDGET_DATA["emotion_generation"]["face_denoise"]
        face_denoise = max(0.0, min(1.0, face_denoise))
        pipe = self._unwrap_scalar(pipe)
        unique_id = self._unwrap_scalar(unique_id)
        image_items = self._image_list(images)
        data_items = self._parse_emotion_data(emotion_data)
        source_items = []
        for index, meta in enumerate(data_items):
            source_image, source_mask = self._load_source_sprite_from_path(meta.get("source_path") if isinstance(meta, dict) else "")
            if source_image is None:
                source_image = image_items[index] if index < len(image_items) else None
                source_mask = None
            if source_image is not None:
                source_items.append((source_image, source_mask))
        emotion_items = [str(item.get("emotion_prompt", "")) for item in data_items]
        sprite_paths = [str(item.get("sprite_output_path", "")) for item in data_items]
        total = min(len(source_items), len(data_items))
        groups = []
        for index in range(total):
            key = sprite_paths[index] if index < len(sprite_paths) and sprite_paths[index] else emotion_items[index]
            if not groups or groups[-1]["key"] != key:
                groups.append({"key": key, "indices": []})
            groups[-1]["indices"].append(index)
        stage_labels = self._emotion_pairs(widget_payload, emotion_items, len(groups))
        cache_dir = None

        results = []
        faces = []
        try:
            for group_index, group in enumerate(groups):
                stage_key, stage_label = stage_labels[group_index]
                group_source = torch.cat([source_items[i][0] for i in group["indices"]], dim=0)
                self._emit(unique_id, stage_key, "running", group_source, f"Generating {stage_label}", 0, len(group["indices"]), cache_dir=cache_dir)
                group_results = []
                for local_index, index in enumerate(group["indices"], start=1):
                    image, mask = source_items[index]
                    meta = data_items[index] if index < len(data_items) else {}
                    if mask is not None and (mask.shape[-2] != image.shape[1] or mask.shape[-1] != image.shape[2]):
                        mask_img = Image.fromarray((mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
                        mask_img = mask_img.resize((image.shape[2], image.shape[1]), Image.Resampling.LANCZOS)
                        mask = torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0).unsqueeze(0)
                    seed = int(meta.get("seed", 0) or 0)
                    result, face_crop = self._run_emotion_generation_one(
                        image,
                        mask,
                        pipe,
                        meta.get("emotion_prompt", emotion_items[index]),
                        meta.get("positive_prompt", ""),
                        meta.get("negative_prompt", ""),
                        seed + index,
                        face_denoise,
                    )
                    results.append(result)
                    # FaceDetailer crops are intentionally variable-size. Save them
                    # as files, but keep the IMAGE output batch shape-stable.
                    faces.append(result)
                    group_results.append(result)
                    if index < len(sprite_paths):
                        self._save_rgba_image(result, mask, sprite_paths[index], local_index)
                        face_prefix = self._face_prefix_from_sprite_prefix(sprite_paths[index])
                        if face_prefix:
                            self._save_rgba_image(face_crop, None, face_prefix, local_index)
                    partial = torch.cat(group_results, dim=0)
                    self._emit(
                        unique_id,
                        stage_key,
                        "running" if local_index < len(group["indices"]) else "done",
                        partial,
                        f"Done {stage_label}",
                        local_index,
                        len(group["indices"]),
                        cache_dir=cache_dir,
                    )

            if not results:
                raise RuntimeError("No emotion images to generate. Select at least one costume and one emotion.")
            return torch.cat(results, dim=0), torch.cat(faces, dim=0)
        except Exception as exc:
            print("[VNCCS Emotions Generator] Failed:", exc)
            traceback.print_exc()
            self._emit(unique_id, "error", "error", message=str(exc))
            raise


NODE_CLASS_MAPPINGS = {
    "VNCCS_CharacterGenerator": VNCCS_CharacterGenerator,
    "VNCCS_CharacterCloneGenerator": VNCCS_CharacterCloneGenerator,
    "VNCCS_ClothesGenerator": VNCCS_ClothesGenerator,
    "VNCCS_EmotionsGenerator": VNCCS_EmotionsGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_CharacterGenerator": "VNCCS Character Generator",
    "VNCCS_CharacterCloneGenerator": "VNCCS Character Clone Generator",
    "VNCCS_ClothesGenerator": "VNCCS Clothes Generator",
    "VNCCS_EmotionsGenerator": "VNCCS Emotions Generator",
}
