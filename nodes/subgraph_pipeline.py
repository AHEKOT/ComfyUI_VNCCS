"""VNCCS Character Sheet Pipeline.

Replacement node for the Step 1 Pose Generation -> Upscaler -> BG Remove
subgraph chain. It executes the same processing stages internally and exposes a
DOM widget for stage previews/settings.
"""

import base64
import inspect
import io
import json
import os
import traceback
from urllib.parse import urlencode

import numpy as np
import torch
from PIL import Image

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

from .sheet_manager import VNCCSSheetManager
from .vnccs_pipe import VNCCS_Pipe
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
            "VNCCSSheetManager": VNCCSSheetManager,
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


def _character_cache_dir_from_sheets_path(sheets_path, character_name=""):
    if isinstance(sheets_path, list):
        sheets_path = sheets_path[0] if sheets_path else ""
    sheets_path = str(sheets_path or "").strip()
    if sheets_path:
        parts = os.path.abspath(sheets_path).split(os.sep)
        if "Sheets" in parts:
            character_root = os.sep.join(parts[:parts.index("Sheets")])
            if character_root:
                return os.path.join(character_root, "cache", "poses")
        if os.path.isdir(sheets_path):
            return os.path.join(os.path.abspath(sheets_path), "cache", "poses")

    character_name = str(character_name or "").strip()
    if character_name and character_name != "Unknown":
        return os.path.join(character_dir(character_name), "cache", "poses")

    return None


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


DEFAULT_WIDGET_DATA = {
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
}


class VNCCS_CharacterSheetPipeline:
    OUTPUT_NODE = True

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
        data = json.loads(widget_data) if isinstance(widget_data, str) and widget_data.strip() else {}
        merged = json.loads(json.dumps(DEFAULT_WIDGET_DATA))
        for section, values in (data or {}).items():
            if isinstance(values, dict) and section in merged:
                merged[section].update(values)
        return merged

    def _widget_data(self, widget_data):
        try:
            return json.loads(widget_data) if isinstance(widget_data, str) and widget_data.strip() else {}
        except Exception:
            return {}

    def _emit(self, unique_id, stage, status, images=None, message="", current=None, total=None, cache_dir=None):
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
        if images is not None:
            payload["images"] = _tensor_to_preview_urls(images, unique_id, stage, cache_dir=cache_dir)
        try:
            server.PromptServer.instance.send_sync("vnccs.pipeline.stage", payload)
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

    def _run_pose_generation(self, poses, character, pipe, prompt, settings):
        pipe_values = self._extract_pipe(pipe)
        pose_parts = VNCCSSheetManager().process_sheet("split", poses, 1024, 3072, False)[0]
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

        sampled_list = self._run_list_mapped(
            "KSampler",
            {"positive": positive_list, "negative": negative_list, "latent_image": latent_list},
            model=pipe_values["model"],
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
        upscaled = _call_comfy_node(
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

    def _run_gan_upscale_one(self, image, upscale_model):
        return _call_comfy_node(
            "ImageUpscaleWithModel",
            upscale_model=upscale_model,
            image=image,
        )[0]

    def _run_upscaler(self, image, background, settings, unique_id=None, cache_dir=None):
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
                    "upscaler",
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
                "upscaler",
                "running" if index < total else "done",
                partial,
                f"Upscaled image {index} of {total}",
                index,
                total,
                cache_dir=cache_dir,
            )
        return torch.cat(results, dim=0) if results else image

    def _run_bg_remove(self, images, settings):
        sheet = VNCCSSheetManager().process_sheet("compose", images, 1024, 6144, False)[0][0]
        sheet = VNCCSChromaKey().chroma_key(
            sheet,
            float(settings["tolerance"]),
            float(settings["despill_strength"]),
            int(settings["despill_kernel_size"]),
            settings["despill_color"],
        )[0]
        faces = self._extract_faces(sheet)
        return sheet, faces

    def _preview_bg_remove(self, images, settings):
        return VNCCSChromaKey().chroma_key(
            images,
            float(settings["tolerance"]),
            float(settings["despill_strength"]),
            int(settings["despill_kernel_size"]),
            settings["despill_color"],
        )[0]

    def _extract_faces(self, sheet):
        try:
            detector = _call_comfy_node("UltralyticsDetectorProvider", model_name="bbox/face_yolov8m.pt")[0]
            return _call_comfy_node(
                "VNCCS_BBox_Extractor",
                image=sheet,
                bbox_detector=detector,
                confidence=0.5,
                crop_size=300,
                padding=10,
            )[0]
        except Exception:
            return self._fallback_face_grid(sheet)

    def _fallback_face_grid(self, sheet):
        img = sheet
        if img.ndim == 3:
            img = img.unsqueeze(0)
        h = img.shape[1]
        w = img.shape[2]
        cell_w = max(1, w // 6)
        cell_h = max(1, h // 2)
        crops = []
        for row in range(2):
            for col in range(6):
                cell = img[0, row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w, :]
                top = int(cell_h * 0.04)
                bottom = int(cell_h * 0.34)
                left = int(cell_w * 0.22)
                right = int(cell_w * 0.78)
                face = cell[top:bottom, left:right, :]
                face = torch.nn.functional.interpolate(
                    face.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                crops.append(face)
        return torch.cat(crops, dim=0)

    def process(self, poses, character, pipe, prompt, background="Green", widget_data="{}", sheets_path="", unique_id=None):
        settings = self._settings(widget_data)
        widget_payload = self._widget_data(widget_data)
        cache_dir = _character_cache_dir_from_sheets_path(sheets_path, widget_payload.get("character_name", ""))
        try:
            self._emit(unique_id, "pose_generation", "running", message="Encoding pose list", current=0, total=12, cache_dir=cache_dir)
            pose_images = self._run_pose_generation(poses, character, pipe, prompt, settings["pose_generation"])
            pose_total = pose_images.shape[0] if torch.is_tensor(pose_images) and pose_images.ndim == 4 else 0
            self._emit(unique_id, "pose_generation", "done", pose_images, f"Generated {pose_total} pose images", pose_total, pose_total, cache_dir=cache_dir)

            up_total = pose_total
            self._emit(unique_id, "upscaler", "running", pose_images, f"Preparing upscaler for {up_total} images", 0, up_total, cache_dir=cache_dir)
            upscaled = self._run_upscaler(pose_images, background, settings["upscaler"], unique_id=unique_id, cache_dir=cache_dir)

            bg_total = upscaled.shape[0] if torch.is_tensor(upscaled) and upscaled.ndim == 4 else 0
            self._emit(unique_id, "bg_remove", "running", upscaled, f"Composing and removing background for {bg_total} images", 0, bg_total, cache_dir=cache_dir)
            sheet, faces = self._run_bg_remove(upscaled, settings["bg_remove"])
            bg_preview = self._preview_bg_remove(upscaled, settings["bg_remove"])
            self._emit(unique_id, "bg_remove", "done", bg_preview, f"Background removed from {bg_total} images", bg_total, bg_total, cache_dir=cache_dir)
            return sheet, faces, pose_images, upscaled
        except Exception as exc:
            print("[VNCCS Pipeline] Failed:", exc)
            traceback.print_exc()
            self._emit(unique_id, "error", "error", message=str(exc))
            raise


NODE_CLASS_MAPPINGS = {
    "VNCCS_CharacterSheetPipeline": VNCCS_CharacterSheetPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_CharacterSheetPipeline": "VNCCS Character Sheet Pipeline",
}
