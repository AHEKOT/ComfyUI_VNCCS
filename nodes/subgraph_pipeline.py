"""VNCCS Character Sheet Pipeline.

Replacement node for the Step 1 Pose Generation -> Upscaler -> BG Remove
subgraph chain. It executes the same processing stages internally and exposes a
DOM widget for stage previews/settings.
"""

import base64
import inspect
import io
import json
import traceback

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


DEFAULT_WIDGET_DATA = {
    "upscaler": {
        "model": "seedvr2_ema_3b-Q4_K_M.gguf",
        "vae": "ema_vae_fp16.safetensors",
        "device": "cuda:0",
        "offload_device": "cpu",
        "seed": 42,
        "resolution": 2048,
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

    def _emit(self, unique_id, stage, status, images=None, message=""):
        if server is None or not unique_id:
            return
        payload = {
            "node_id": str(unique_id),
            "stage": stage,
            "status": status,
            "message": message,
        }
        if images is not None:
            payload["images"] = _tensor_to_png_data_url(images)
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

    def _run_pose_generation(self, poses, character, pipe, prompt, settings):
        pipe_values = self._extract_pipe(pipe)
        pose_parts = VNCCSSheetManager().process_sheet("split", poses, 1024, 3072, False)[0]
        character_rgb = VNCCS_MaskExtractor().fill_alpha_with_color(character)[0]

        encoded_items = []
        for pose_image in pose_parts:
            positive, negative, latent = VNCCS_QWEN_Encoder().encode(
                clip=pipe_values["clip"],
                vae=pipe_values["vae"],
                prompt=prompt,
                image1=pose_image,
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
            encoded_items.append((positive, negative, latent))

        sampled_items = []
        for positive, negative, latent in encoded_items:
            samples = _call_comfy_node(
                "KSampler",
                model=pipe_values["model"],
                seed=pipe_values["seed"],
                steps=pipe_values["steps"],
                cfg=pipe_values["cfg"],
                sampler_name=pipe_values["sampler"],
                scheduler=pipe_values["scheduler"],
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1,
            )[0]
            sampled_items.append(samples)

        decoded_items = []
        for samples in sampled_items:
            image = _call_comfy_node(
                "VAEDecodeTiled",
                samples=samples,
                vae=pipe_values["vae"],
                tile_size=512,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
            )[0]
            decoded_items.append(image)

        return torch.cat(decoded_items, dim=0)

    def _run_upscaler(self, image, background, settings):
        dit = _call_comfy_node(
            "SeedVR2LoadDiTModel",
            model=settings["model"],
            device=settings["device"],
            blocks_to_swap=0,
            swap_io_components=False,
            offload_device=settings["offload_device"],
            cache_model=False,
            attention_mode="sdpa",
        )[0]
        vae = _call_comfy_node(
            "SeedVR2LoadVAEModel",
            model=settings["vae"],
            device=settings["device"],
            encode_tiled=True,
            encode_tile_size=1024,
            encode_tile_overlap=128,
            decode_tiled=True,
            decode_tile_size=1024,
            decode_tile_overlap=128,
            tile_debug="false",
            offload_device=settings["offload_device"],
            cache_model=False,
        )[0]
        upscaled = _call_comfy_node(
            "SeedVR2VideoUpscaler",
            image=image,
            dit=dit,
            vae=vae,
            seed=int(settings["seed"]),
            resolution=int(settings["resolution"]),
            max_resolution=3840,
            batch_size=1,
            uniform_batch_size=False,
            color_correction="lab",
            temporal_overlap=0,
            prepend_frames=0,
            input_noise_scale=0,
            latent_noise_scale=0,
            offload_device=settings["offload_device"],
            enable_debug=False,
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

    def process(self, poses, character, pipe, prompt, background="Green", widget_data="{}", unique_id=None):
        settings = self._settings(widget_data)
        try:
            self._emit(unique_id, "pose_generation", "running")
            pose_images = self._run_pose_generation(poses, character, pipe, prompt, settings["pose_generation"])
            self._emit(unique_id, "pose_generation", "done", pose_images)

            self._emit(unique_id, "upscaler", "running", pose_images)
            upscaled = self._run_upscaler(pose_images, background, settings["upscaler"])
            self._emit(unique_id, "upscaler", "done", upscaled)

            self._emit(unique_id, "bg_remove", "running", upscaled)
            sheet, faces = self._run_bg_remove(upscaled, settings["bg_remove"])
            self._emit(unique_id, "bg_remove", "done", self._preview_bg_remove(upscaled, settings["bg_remove"]))
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
