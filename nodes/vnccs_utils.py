"""Shared image processing and the standalone VNCCS Chroma Key node."""
from ..operation_logger import log_event, log_stage

import os
import inspect
import math
import platform
import threading
import torch
import numpy as np
import cv2
import torch.nn.functional as F

try:
    from ..utils import get_full_path_agnostic
except Exception:
    from utils import get_full_path_agnostic

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Device selection for optional SAM3 recovery
def _select_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"

# Register the optional SAM3 model directory
if folder_paths:
    folder_paths.add_model_folder_path("sam3", os.path.join(folder_paths.models_dir, "sam3"))

SAM3_MODEL_REPO_ID = "yolain/sam3-safetensors"
SAM3_MODEL_FILENAME = "sam3-fp16.safetensors"
SAM3_MODEL_REVISION = "eb174af94625028887dfe92d2d8483ca5a5d3336"
_SAM3_DOWNLOAD_LOCK = threading.Lock()

def _sam3_recovery_runtime_supported():
    """Easy SAM3 currently requires Triton/decord and cannot run on macOS/MPS."""
    return platform.system().lower() != "darwin"

# --- Shared helpers ---

def _flatten_image_tensors(value):
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        if value.ndim == 4:
            return [value[i:i + 1] for i in range(value.shape[0])]
        if value.ndim == 3:
            return [value.unsqueeze(0)]
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_image_tensors(item))
        return result
    return []

def _normalize_image_batch(value, target_hw=None, stage="utils batch"):
    items = []
    for item in _flatten_image_tensors(value):
        if not torch.is_tensor(item):
            continue
        if not torch.is_floating_point(item):
            item = item.float()
        if item.numel() and item.max() > 1.5:
            item = item / 255.0
        items.append(item.clamp(0.0, 1.0))
    if not items:
        return value
    if target_hw is None:
        target_hw = (int(items[0].shape[1]), int(items[0].shape[2]))
    target_channels = max(int(item.shape[-1]) for item in items)
    shapes = [(int(item.shape[1]), int(item.shape[2]), int(item.shape[3])) for item in items]
    target_shape = (int(target_hw[0]), int(target_hw[1]), target_channels)
    if any(shape != target_shape for shape in shapes):
        log_event("diagnostic", component="ImageProcessing", level="debug", message=f'Normalizing {stage}: {shapes} -> {target_shape}')
    normalized = []
    for item in items:
        if item.shape[-1] < target_channels:
            pad_value = 1.0 if target_channels == 4 and item.shape[-1] == 3 else 0.0
            pad = torch.full((*item.shape[:-1], target_channels - item.shape[-1]), pad_value, dtype=item.dtype, device=item.device)
            item = torch.cat([item, pad], dim=-1)
        elif item.shape[-1] > target_channels:
            item = item[..., :target_channels]
        if (item.shape[1], item.shape[2]) != target_hw:
            item = F.interpolate(
                item.movedim(-1, 1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1).clamp(0.0, 1.0)
        normalized.append(item)
    return torch.cat(normalized, dim=0)

def _ensure_float01(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to float in [0, 1] range."""
    t = tensor
    if not torch.is_floating_point(t):
        t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)

def _box_blur_2d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    src = mask.unsqueeze(0).unsqueeze(0)
    blurred = F.avg_pool2d(src, kernel_size=kernel_size, stride=1, padding=radius)
    return blurred.squeeze(0).squeeze(0)

def _morph(mask: torch.Tensor, radius: int, mode: str) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    src = mask.unsqueeze(0).unsqueeze(0)
    if mode == "dilate":
        out = F.max_pool2d(src, kernel_size=kernel_size, stride=1, padding=radius)
    elif mode == "erode":
        out = -F.max_pool2d(-src, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        raise ValueError(f"Unsupported morph mode: {mode}")
    return out.squeeze(0).squeeze(0)

def _remove_small_islands(mask: torch.Tensor, min_neighbors: int) -> torch.Tensor:
    if min_neighbors <= 0:
        return mask
    hard = (mask > 0.5).float()
    src = hard.unsqueeze(0).unsqueeze(0)
    neighbors = F.conv2d(src, torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype), padding=1)
    keep = (neighbors.squeeze(0).squeeze(0) >= float(min_neighbors)).float()
    return torch.where(keep > 0.0, mask, torch.zeros_like(mask))

def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _get_comfy_node_class(class_names):
    try:
        import nodes as comfy_nodes
    except Exception:
        comfy_nodes = None
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if comfy_nodes else {}
    for class_name in class_names:
        cls = mappings.get(class_name)
        if cls is not None:
            return cls
    return None

def _is_comfy_node_output(value):
    return hasattr(value, "result") and hasattr(value, "args") and type(value).__name__ == "NodeOutput"

def _unwrap_node_result(value):
    if _is_comfy_node_output(value):
        block_execution = getattr(value, "block_execution", None)
        if block_execution:
            raise RuntimeError(str(block_execution))
        value = getattr(value, "result", None)
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value

def _call_registered_node(class_names, method_names=None, **kwargs):
    cls = _get_comfy_node_class(class_names)
    if cls is None:
        raise RuntimeError(f"Required node '{'/'.join(class_names)}' is not available")

    instance = cls()
    candidates = []
    function_name = getattr(cls, "FUNCTION", None)
    if function_name:
        candidates.append(function_name)
    if method_names:
        candidates.extend(method_names)
    candidates.extend(("execute", "process", "process_image", "load_model", "loadmodel", "load", "segment", "segment_image"))

    for method_name in candidates:
        method = getattr(instance, method_name, None)
        if method is None:
            continue
        signature = inspect.signature(method)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        accepted = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in signature.parameters}
        while True:
            try:
                result = method(**accepted)
                break
            except TypeError as exc:
                message = str(exc)
                marker = "unexpected keyword argument "
                if marker not in message:
                    raise
                unexpected = message.split(marker, 1)[1].strip().strip("'\"")
                if unexpected not in accepted:
                    raise
                accepted = dict(accepted)
                accepted.pop(unexpected, None)
        return _unwrap_node_result(result)

    raise RuntimeError(f"Node '{'/'.join(class_names)}' has no callable FUNCTION")

def _sam3_model_dir():
    if folder_paths is None or not getattr(folder_paths, "models_dir", None):
        return os.path.join("models", "sam3")
    try:
        folders = folder_paths.get_folder_paths("sam3") or []
        for folder in folders:
            if folder:
                return folder
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, "sam3")

def _find_sam3_model_file(filename):
    if folder_paths is not None:
        try:
            path = get_full_path_agnostic(folder_paths, "sam3", filename, require_exists=True)
            if path and os.path.exists(path):
                return path
        except Exception:
            pass
    target = os.path.join(_sam3_model_dir(), filename)
    return target if os.path.exists(target) else None

def _ensure_sam3_model_available():
    filename = SAM3_MODEL_FILENAME
    filename = os.path.basename(str(filename or SAM3_MODEL_FILENAME))
    existing = _find_sam3_model_file(filename)
    if existing:
        return filename
    if hf_hub_download is None:
        raise RuntimeError(
            "SAM3 model is missing and huggingface_hub is not installed. "
            f"Install huggingface_hub or place '{filename}' in '{_sam3_model_dir()}'."
        )

    with _SAM3_DOWNLOAD_LOCK:
        existing = _find_sam3_model_file(filename)
        if existing:
            return filename

        repo_id = SAM3_MODEL_REPO_ID
        revision = SAM3_MODEL_REVISION
        target_dir = _sam3_model_dir()
        os.makedirs(target_dir, exist_ok=True)
        log_stage("download", component="SAM3", file=filename, repository=repo_id)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=target_dir,
                token=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download SAM3 model. "
                f"Place '{filename}' in '{target_dir}' manually or check access to Hugging Face repo '{repo_id}'. "
                f"Original error: {exc}"
            ) from exc

        if not os.path.exists(path):
            raise RuntimeError(f"SAM3 download completed but '{path}' was not created")
        log_event("asset_ready", component="SAM3", file=os.path.basename(path))
        return filename

def _normalize_mask_batch(value, target_hw, batch_size, stage="mask"):
    tensors = []

    def collect(item):
        if isinstance(item, tuple):
            if item:
                collect(item[0])
            return
        if isinstance(item, list):
            for child in item:
                collect(child)
            return
        if torch.is_tensor(item):
            tensors.append(item)

    collect(value)
    if not tensors:
        raise RuntimeError(f"VNCCS Chroma Key: {stage} did not return a tensor mask")

    masks = []
    for tensor in tensors:
        t = _ensure_float01(tensor.detach() if tensor.requires_grad else tensor)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] == batch_size and tuple(t.shape[-2:]) == tuple(target_hw):
                pass
            elif t.shape[-1] <= 4 and tuple(t.shape[:2]) == tuple(target_hw):
                t = t[..., 0].unsqueeze(0)
            elif t.shape[0] <= 4 and tuple(t.shape[-2:]) == tuple(target_hw):
                t = t[0:1]
        elif t.ndim == 4:
            if t.shape[-1] <= 4:
                t = t[..., 0]
            elif t.shape[1] <= 4:
                t = t[:, 0, :, :]
        if t.ndim != 3:
            continue
        if tuple(t.shape[-2:]) != tuple(target_hw):
            t = F.interpolate(
                t.unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        masks.append(t.clamp(0.0, 1.0))

    if not masks:
        raise RuntimeError(f"VNCCS Chroma Key: {stage} mask shape is unsupported")

    mask = torch.cat(masks, dim=0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    if mask.shape[0] < batch_size:
        raise RuntimeError(f"VNCCS Chroma Key: {stage} returned {mask.shape[0]} masks for {batch_size} images")
    return mask[:batch_size].clamp(0.0, 1.0)

# --- Guided Filter Helper ---
class VNCCS_MaskExtractor:
    """Fill alpha channel with bright green color."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "fill_alpha_with_color"
    CATEGORY = "VNCCS"

    def fill_alpha_with_color(self, image):
        if image is None:
            raise ValueError("No image provided")
        img = _ensure_float01(image)
        added_batch = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            added_batch = True
        if img.shape[-1] < 4:
            out = img[..., :3]
            return (out.squeeze(0) if added_batch else out,)
        rgb = img[..., :3]
        alpha = img[..., 3]
        if alpha.ndim == 4 and alpha.shape[1] == 1:
            alpha = alpha.squeeze(1)
        alpha = alpha.clamp(0.0, 1.0)
        r, g, b = 0.0, 1.0, 0.0
        device = rgb.device
        dtype = rgb.dtype
        bg = torch.tensor([r, g, b], dtype=dtype, device=device).view(1, 1, 1, 3)
        alpha3 = alpha.unsqueeze(-1)
        out = rgb * alpha3 + bg * (1.0 - alpha3)
        if added_batch:
            out = out.squeeze(0)
        return (out,)

class VNCCSChromaKey:
    """VNCCS Chroma Key - soft chroma key with edge decontamination."""

    SAM3_RECOVERY_ERODE_RADIUS = 4
    SAM3_RECOVERY_MIN_FOREGROUND_OVERLAP = 0.55

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "softness": ("FLOAT", {"default": 0.12, "min": 0.001, "max": 1.0, "step": 0.01}),
                "despill_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "matte_cleanup": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_recover": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_decontaminate": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_choke": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "matte_method": (["chroma_soft", "guided_edge", "pymatting_if_available", "screen_matte"], {"default": "guided_edge"}),
                "screen_mode": (["auto", "green", "blue", "red"], {"default": "auto"}),
                "output_mode": (["straight_rgba", "premultiplied_rgba"], {"default": "straight_rgba"}),
                "use_sam3_recovery_mask": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled", "display_name": "Use SAM3 Recovery Mask"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "matte", "edge_debug")
    CATEGORY = "VNCCS"
    FUNCTION = "chroma_key"
    DESCRIPTION = """
    VNCCS Chroma Key - automatically detects background color from image borders.
    Uses soft chroma keying, edge-guided matte cleanup, foreground recovery, and
    edge-only decontamination for cleaner hair and outlines.
    The opt-in screen_matte method runs on the selected GPU, estimates the actual
    plate color automatically, and removes isolated screen artifacts. Its color
    unmixing is controlled jointly by despill, foreground recovery and edge
    decontamination; screen_mode is used only by the legacy methods.
    """

    def chroma_key(
        self,
        image,
        tolerance,
        softness,
        despill_strength,
        edge_width,
        matte_cleanup,
        foreground_recover,
        edge_decontaminate,
        edge_choke,
        matte_method,
        screen_mode,
        output_mode,
        use_sam3_recovery_mask=False,
        sam3_settings=None,
    ):
        image = _normalize_image_batch(image, stage="chroma key input")
        if _as_bool(use_sam3_recovery_mask, False):
            if not _sam3_recovery_runtime_supported():
                log_event("fallback", component="ImageProcessing", level="warning", message='SAM3 recovery is unsupported on macOS; using chroma key without recovery')
            else:
                try:
                    return self._chroma_key_with_sam3_recovery(
                        image=image,
                        tolerance=tolerance,
                        softness=softness,
                        despill_strength=despill_strength,
                        edge_width=edge_width,
                        matte_cleanup=matte_cleanup,
                        foreground_recover=foreground_recover,
                        edge_decontaminate=edge_decontaminate,
                        edge_choke=edge_choke,
                        matte_method=matte_method,
                        screen_mode=screen_mode,
                        output_mode=output_mode,
                        sam3_settings=sam3_settings,
                    )
                except Exception as exc:
                    log_event('fallback', component='ImageProcessing', level='warning', message=f'SAM3 recovery failed; using chroma key without recovery: {type(exc).__name__}: {exc}', error=str(exc))

        if len(image.shape) == 4:
            rgba_list = []
            matte_list = []
            debug_list = []
            for frame in image:
                rgba, alpha, debug = self._process_single(
                    frame,
                    tolerance,
                    softness,
                    despill_strength,
                    edge_width,
                    matte_cleanup,
                    foreground_recover,
                    edge_decontaminate,
                    edge_choke,
                    matte_method,
                    screen_mode,
                    output_mode,
                )
                rgba_list.append(rgba)
                matte_list.append(alpha)
                debug_list.append(debug)
            return (torch.stack(rgba_list), torch.stack(matte_list), torch.stack(debug_list))

        rgba, alpha, debug = self._process_single(
            image,
            tolerance,
            softness,
            despill_strength,
            edge_width,
            matte_cleanup,
            foreground_recover,
            edge_decontaminate,
            edge_choke,
            matte_method,
            screen_mode,
            output_mode,
        )
        return (rgba.unsqueeze(0), alpha.unsqueeze(0), debug.unsqueeze(0))

    def _chroma_key_with_sam3_recovery(
        self,
        image,
        tolerance,
        softness,
        despill_strength,
        edge_width,
        matte_cleanup,
        foreground_recover,
        edge_decontaminate,
        edge_choke,
        matte_method,
        screen_mode,
        output_mode,
        sam3_settings=None,
    ):
        sam3_settings = sam3_settings if isinstance(sam3_settings, dict) else {}
        batch = _normalize_image_batch(image, stage="sam3 recovery chroma key input")
        target_hw = (int(batch.shape[1]), int(batch.shape[2]))
        recovery_candidates = self._run_sam3_recovery_masks(batch, target_hw, sam3_settings)

        rgba_list = []
        matte_list = []
        debug_list = []
        for index, frame in enumerate(batch):
            rgba, alpha, debug = self._process_single(
                frame,
                tolerance,
                softness,
                despill_strength,
                edge_width,
                matte_cleanup,
                foreground_recover,
                edge_decontaminate,
                edge_choke,
                matte_method,
                screen_mode,
                output_mode,
            )
            recovery_mask = self._select_sam3_recovery_mask(
                recovery_candidates[index],
                alpha,
                sam3_settings,
            )
            rgba, alpha, debug = self._restore_recovery_details(
                original=frame,
                rgba=rgba,
                alpha=alpha,
                debug=debug,
                recovery_mask=recovery_mask,
                output_mode=output_mode,
                erode_radius=int(sam3_settings.get("sam3_erode_radius", self.SAM3_RECOVERY_ERODE_RADIUS)),
            )
            rgba_list.append(rgba)
            matte_list.append(alpha)
            debug_list.append(debug)

        return (torch.stack(rgba_list), torch.stack(matte_list), torch.stack(debug_list))

    def _run_sam3_recovery_masks(self, image: torch.Tensor, target_hw, settings=None):
        settings = settings if isinstance(settings, dict) else {}
        sam3_model_name = str(settings.get("sam3_model", "") or "").strip() or _ensure_sam3_model_available()
        requested_device = str(settings.get("sam3_device", "auto") or "auto").strip()
        device = _select_torch_device() if requested_device.lower() == "auto" else requested_device
        sam3_model = _call_registered_node(
            ["LoadSam3Model", "easy sam3ModelLoader"],
            method_names=("load_model", "loadmodel", "load"),
            model=sam3_model_name,
            segmentor=str(settings.get("sam3_segmentor", "image") or "image"),
            device=device,
            precision=str(settings.get("sam3_precision", "bf16") or "bf16"),
        )
        batch_size = int(image.shape[0])
        recovery_candidates = []
        # Older Easy-SAM3 releases stack variable-length detection boxes across
        # a batch. Segment frames individually while keeping the model loaded.
        for index in range(batch_size):
            result = _call_registered_node(
                ["Sam3ImageSegmentation", "easy sam3ImageSegmentation"],
                method_names=("segment", "segment_image", "process", "execute"),
                sam3_model=sam3_model,
                images=image[index:index + 1, ..., :3],
                prompt=str(settings.get("sam3_prompt", "face, clothes, accessories, hat, boots, eyes")),
                threshold=float(settings.get("sam3_threshold", 0.40)),
                keep_model_loaded=index < batch_size - 1,
                add_background=str(settings.get("sam3_add_background", "none") or "none"),
                detection_limit=int(settings.get("sam3_detection_limit", -1)),
                coordinates_positive=None,
                coordinates_negative=None,
                bboxes=None,
                mask=None,
            )
            recovery_candidates.append(
                self._sam3_recovery_candidates_from_result(
                    result,
                    target_hw=target_hw,
                    stage=f"SAM3 recovery image {index + 1}/{batch_size}",
                )
            )
        return recovery_candidates

    def _sam3_recovery_candidates_from_result(self, result, target_hw, stage="SAM3 recovery"):
        raw_masks = None
        if isinstance(result, (tuple, list)) and len(result) > 2 and torch.is_tensor(result[2]):
            raw_masks = result[2]
        if raw_masks is not None:
            candidates = self._canonicalize_sam3_mask_candidates(raw_masks, target_hw)
            if candidates is not None:
                return candidates

            log_event("fallback", component="ImageProcessing", level="warning",
                      message=f"{stage} could not interpret individual mask shape {tuple(raw_masks.shape)}; using the combined SAM3 mask")

        combined_source = result[0] if isinstance(result, (tuple, list)) and result else result
        combined = _normalize_mask_batch(
            combined_source,
            target_hw=target_hw,
            batch_size=1,
            stage=stage,
        )
        return combined[:1]

    def _canonicalize_sam3_mask_candidates(self, raw_masks, target_hw):
        """Convert tensor masks of arbitrary rank to canonical [objects, H, W]."""
        masks = raw_masks.detach() if raw_masks.requires_grad else raw_masks
        if masks.ndim < 2 or masks.numel() == 0:
            return None

        target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
        if target_h <= 0 or target_w <= 0:
            return None
        shape = tuple(int(size) for size in masks.shape)

        # Locate the spatial plane by meaning rather than by a fixed tensor
        # layout. Every other axis may represent a batch, object, channel, or
        # a singleton wrapper added by a third-party node version; all of them
        # can safely become the candidate-mask axis because SAM3 is invoked for
        # one source image at a time.
        spatial_planes = []
        for axis in range(masks.ndim - 1):
            first = shape[axis]
            second = shape[axis + 1]
            if first <= 0 or second <= 0:
                continue
            direct_cost = abs(math.log(first / target_h)) + abs(math.log(second / target_w))
            transposed_cost = abs(math.log(second / target_h)) + abs(math.log(first / target_w))
            if direct_cost <= transposed_cost:
                cost = direct_cost
                transpose = False
            else:
                cost = transposed_cost
                transpose = True
            # Resolution similarity identifies the spatial plane even when a
            # model emits masks at its native size. Area and later placement
            # only break ties; they do not encode a particular layout.
            spatial_planes.append((cost, -(first * second), -axis, axis, axis + 1, transpose))

        if not spatial_planes:
            return None
        _, _, _, spatial_y, spatial_x, transpose_spatial = min(spatial_planes)

        source_h = shape[spatial_y]
        source_w = shape[spatial_x]
        if source_h <= 0 or source_w <= 0:
            return None

        non_spatial_axes = [
            axis for axis in range(masks.ndim)
            if axis not in (spatial_y, spatial_x)
        ]
        candidate_count = 1
        for axis in non_spatial_axes:
            candidate_count *= shape[axis]
        if candidate_count <= 0:
            return None

        axis_order = non_spatial_axes + [spatial_y, spatial_x]
        if axis_order != list(range(masks.ndim)):
            masks = masks.permute(axis_order)
        masks = masks.reshape(candidate_count, source_h, source_w)
        if transpose_spatial:
            masks = masks.transpose(-2, -1)

        masks = _ensure_float01(masks)
        if tuple(masks.shape[-2:]) != (target_h, target_w):
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return masks.clamp(0.0, 1.0)

    def _select_sam3_recovery_mask(
        self,
        candidates: torch.Tensor,
        alpha: torch.Tensor,
        settings=None,
    ) -> torch.Tensor:
        settings = settings if isinstance(settings, dict) else {}
        min_overlap = max(
            0.0,
            min(
                1.0,
                float(settings.get("sam3_min_foreground_overlap", self.SAM3_RECOVERY_MIN_FOREGROUND_OVERLAP)),
            ),
        )
        candidates = candidates.to(device=alpha.device, dtype=alpha.dtype).clamp(0.0, 1.0)
        confident_foreground = (alpha >= 0.5).to(dtype=alpha.dtype)
        kept = []
        for candidate in candidates:
            # Accept or reject each SAM3 object as a whole. Do not clip it to
            # the chroma matte: that would manufacture a visible contour.
            hard_candidate = (candidate >= 0.5).to(dtype=alpha.dtype)
            candidate_area = hard_candidate.sum()
            if float(candidate_area.item()) <= 0:
                continue
            foreground_overlap = (hard_candidate * confident_foreground).sum() / candidate_area
            if float(foreground_overlap.item()) >= min_overlap:
                kept.append(candidate)
        log_event("sam3_recovery", component="ImageProcessing", level="debug", kept=len(kept), candidates=int(candidates.shape[0]))
        if not kept:
            return torch.zeros_like(alpha)
        return torch.stack(kept, dim=0).amax(dim=0).clamp(0.0, 1.0)

    def _restore_recovery_details(
        self,
        original: torch.Tensor,
        rgba: torch.Tensor,
        alpha: torch.Tensor,
        debug: torch.Tensor,
        recovery_mask: torch.Tensor,
        output_mode: str,
        erode_radius=None,
    ):
        erode_radius = self.SAM3_RECOVERY_ERODE_RADIUS if erode_radius is None else max(0, int(erode_radius))
        shrunk = _morph(
            recovery_mask.clamp(0.0, 1.0),
            erode_radius,
            "erode",
        ).clamp(0.0, 1.0)
        if shrunk.max() <= 0:
            return rgba, alpha, debug

        original_rgb = _ensure_float01(original)[..., :3]
        restored_alpha = torch.maximum(alpha, shrunk).clamp(0.0, 1.0)
        restored_rgb = torch.lerp(
            rgba[..., :3],
            original_rgb,
            shrunk.unsqueeze(-1),
        ).clamp(0.0, 1.0)
        if output_mode == "premultiplied_rgba":
            restored_rgb = restored_rgb * restored_alpha.unsqueeze(-1)
        restored_rgba = torch.cat([restored_rgb, restored_alpha.unsqueeze(-1)], dim=-1)
        restored_debug = torch.stack([debug[..., 0], restored_alpha, 1.0 - restored_alpha], dim=-1).clamp(0.0, 1.0)
        return restored_rgba, restored_alpha, restored_debug

    def _process_single(
        self,
        image,
        tolerance,
        softness,
        despill_strength,
        edge_width,
        matte_cleanup,
        foreground_recover,
        edge_decontaminate,
        edge_choke,
        matte_method,
        screen_mode,
        output_mode,
    ):
        if matte_method == "screen_matte":
            from .chroma_screen_matte import screen_matte

            return screen_matte(
                _ensure_float01(image), tolerance=tolerance, softness=softness,
                despill_strength=despill_strength, edge_width=edge_width,
                matte_cleanup=matte_cleanup, foreground_recover=foreground_recover,
                edge_decontaminate=edge_decontaminate, edge_choke=edge_choke,
                output_mode=output_mode,
            )
        image = _ensure_float01(image)[..., :3]
        height, width, _ = image.shape
        key_color = self._detect_key_color(image)
        dominant_idx = self._dominant_channel(key_color, screen_mode)
        other_indices = [idx for idx in range(3) if idx != dominant_idx]

        alpha = self._build_soft_alpha(
            image=image,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            tolerance=float(tolerance),
            softness=float(softness),
        )
        alpha = self._cleanup_alpha(alpha, int(edge_width), float(matte_cleanup))

        if matte_method == "guided_edge":
            alpha = self._guided_edge_refine(image, alpha, int(edge_width), float(matte_cleanup))
        elif matte_method == "pymatting_if_available":
            alpha = self._pymatting_refine_if_available(image, alpha, int(edge_width))

        alpha = alpha.clamp(0.0, 1.0)
        edge = self._edge_band(alpha, int(edge_width))
        alpha = self._choke_spill_edge(
            image=image,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            amount=float(edge_choke),
        )

        # Upscalers and image codecs can shift broad areas of an otherwise
        # continuous screen far enough from the sampled key color that the
        # per-pixel matte leaves visible background patches. Run component
        # cleanup after edge choke so enclosed background is classified from
        # the final matte confidence rather than from the softer initial matte.
        alpha = self._suppress_connected_key_fringe(
            image=image,
            alpha=alpha,
            key_color=key_color,
            tolerance=float(tolerance),
            softness=float(softness),
            amount=1.0,
        )
        edge = self._edge_band(alpha, int(edge_width))

        recovered = self._recover_foreground(
            image=image,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            amount=float(foreground_recover),
        )
        despill_strength = max(0.0, min(1.0, float(despill_strength)))
        despilled = self._edge_despill(
            image=recovered,
            alpha=alpha,
            edge=edge,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            strength=despill_strength,
        )
        # Despill is the master control for edge color correction. Previously,
        # decontamination and color bleeding stayed active even at despill=0,
        # which made the despill slider appear to have almost no effect.
        decontaminate_amount = float(edge_decontaminate) * despill_strength
        despilled = self._edge_decontaminate(
            image=despilled,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            amount=decontaminate_amount,
        )
        despilled = self._bleed_clean_edge_colors(
            image=despilled,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            radius=max(2, int(edge_width) + 2),
            amount=despill_strength,
        )
        if output_mode == "premultiplied_rgba":
            rgb_out = despilled * alpha.unsqueeze(-1)
        else:
            rgb_out = despilled

        rgba = torch.cat([rgb_out.clamp(0.0, 1.0), alpha.unsqueeze(-1)], dim=-1)
        debug = torch.stack([edge, alpha, 1.0 - alpha], dim=-1).clamp(0.0, 1.0)

        if rgba.shape[:2] != (height, width):
            raise RuntimeError("VNCCS Chroma Key changed image dimensions unexpectedly.")

        return rgba, alpha, debug

    def _detect_key_color(self, image: torch.Tensor) -> torch.Tensor:
        height, width, _ = image.shape
        ch = max(1, height // 20)
        cw = max(1, width // 20)
        patches = [
            image[0:ch, 0:cw, :3],
            image[0:ch, width - cw : width, :3],
            image[height - ch : height, 0:cw, :3],
            image[height - ch : height, width - cw : width, :3],
        ]

        stable_colors = []
        for patch in patches:
            pixels = patch.reshape(-1, 3)
            if pixels.std(dim=0, unbiased=False).mean() < 0.02:
                stable_colors.append(pixels.median(dim=0)[0])

        if stable_colors:
            return torch.stack(stable_colors).median(dim=0)[0]

        y_margin = max(1, height // 10)
        x_margin = max(1, width // 10)
        border_pixels = torch.cat(
            [
                image[0:y_margin, :, :3].reshape(-1, 3),
                image[height - y_margin : height, :, :3].reshape(-1, 3),
                image[:, 0:x_margin, :3].reshape(-1, 3),
                image[:, width - x_margin : width, :3].reshape(-1, 3),
            ],
            dim=0,
        )
        return border_pixels.median(dim=0)[0]

    def _dominant_channel(self, key_color: torch.Tensor, screen_mode: str) -> int:
        if screen_mode == "red":
            return 0
        if screen_mode == "green":
            return 1
        if screen_mode == "blue":
            return 2
        return int(torch.argmax(key_color).item())

    def _build_soft_alpha(
        self,
        image: torch.Tensor,
        key_color: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        tolerance: float,
        softness: float,
    ) -> torch.Tensor:
        eps = 1e-6
        chroma = image / (image.sum(dim=-1, keepdim=True) + eps)
        key_chroma = key_color / (key_color.sum() + eps)
        chroma_dist = torch.sqrt(((chroma - key_chroma) ** 2).sum(dim=-1))
        rgb_dist = torch.sqrt(((image - key_color) ** 2).sum(dim=-1))

        dom = image[..., dominant_idx]
        other_max = torch.maximum(image[..., other_indices[0]], image[..., other_indices[1]])
        other_avg = (image[..., other_indices[0]] + image[..., other_indices[1]]) * 0.5
        screen_excess = dom - (other_max * 0.65 + other_avg * 0.35)

        key_dom = key_color[dominant_idx]
        key_other_max = torch.maximum(key_color[other_indices[0]], key_color[other_indices[1]])
        key_other_avg = (key_color[other_indices[0]] + key_color[other_indices[1]]) * 0.5
        key_excess = torch.clamp(key_dom - (key_other_max * 0.65 + key_other_avg * 0.35), min=0.05)

        hue_similarity = 1.0 - self._smoothstep(tolerance, tolerance + softness, chroma_dist)
        rgb_similarity = 1.0 - self._smoothstep(tolerance * 1.5, tolerance * 1.5 + softness * 2.0, rgb_dist)
        screen_affinity = self._smoothstep(key_excess * 0.2, key_excess * 0.85 + 1e-6, screen_excess)

        background = (hue_similarity * 0.55 + rgb_similarity * 0.45) * screen_affinity

        strong_screen = self._smoothstep(key_excess * 0.75, key_excess * 1.25 + 1e-6, screen_excess)
        background = torch.maximum(background, strong_screen * hue_similarity * 0.85).clamp(0.0, 1.0)
        return 1.0 - background

    def _smoothstep(self, edge0: float | torch.Tensor, edge1: float | torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = ((value - edge0) / (edge1 - edge0 + 1e-6)).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _cleanup_alpha(self, alpha: torch.Tensor, edge_width: int, amount: float) -> torch.Tensor:
        if amount <= 0.0:
            return alpha
        cleaned = _remove_small_islands(alpha, min_neighbors=max(1, int(2 + amount * 5)))
        if edge_width > 0:
            opened = _morph(_morph(cleaned, 1, "erode"), 1, "dilate")
            edge = self._edge_band(cleaned, max(1, edge_width))
            cleaned = torch.lerp(cleaned, opened, edge * amount * 0.35)
            blurred = _box_blur_2d(cleaned, max(1, edge_width // 2))
            cleaned = torch.lerp(cleaned, blurred, edge * amount * 0.25)
        return cleaned.clamp(0.0, 1.0)

    def _guided_edge_refine(self, image: torch.Tensor, alpha: torch.Tensor, edge_width: int, amount: float) -> torch.Tensor:
        if edge_width <= 0 or amount <= 0.0:
            return alpha
        luminance = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        radius = max(1, edge_width)
        mean_i = _box_blur_2d(luminance, radius)
        mean_a = _box_blur_2d(alpha, radius)
        corr_i = _box_blur_2d(luminance * luminance, radius)
        corr_ia = _box_blur_2d(luminance * alpha, radius)
        var_i = corr_i - mean_i * mean_i
        cov_ia = corr_ia - mean_i * mean_a
        linear_a = cov_ia / (var_i + 0.01)
        linear_b = mean_a - linear_a * mean_i
        refined = _box_blur_2d(linear_a, radius) * luminance + _box_blur_2d(linear_b, radius)
        edge = self._edge_band(alpha, edge_width)
        return torch.lerp(alpha, refined.clamp(0.0, 1.0), edge * amount).clamp(0.0, 1.0)

    def _pymatting_refine_if_available(self, image: torch.Tensor, alpha: torch.Tensor, edge_width: int) -> torch.Tensor:
        try:
            from pymatting import estimate_alpha_cf
        except Exception:
            return self._guided_edge_refine(image, alpha, edge_width, 0.5)

        trimap = torch.full_like(alpha, 0.5)
        trimap = torch.where(alpha > 0.98, torch.ones_like(trimap), trimap)
        trimap = torch.where(alpha < 0.02, torch.zeros_like(trimap), trimap)

        image_np = image.detach().cpu().numpy().astype("float64")
        trimap_np = trimap.detach().cpu().numpy().astype("float64")
        try:
            matte_np = estimate_alpha_cf(image_np, trimap_np)
        except Exception:
            return self._guided_edge_refine(image, alpha, edge_width, 0.5)

        matte = torch.from_numpy(np.asarray(matte_np)).to(device=image.device, dtype=image.dtype)
        return matte.clamp(0.0, 1.0)

    def _suppress_connected_key_fringe(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        key_color: torch.Tensor,
        tolerance: float,
        softness: float,
        amount: float,
    ) -> torch.Tensor:
        if amount <= 0.0:
            return alpha

        eps = 1e-6
        chroma = image / (image.sum(dim=-1, keepdim=True) + eps)
        key_chroma = key_color / (key_color.sum() + eps)
        chroma_dist = torch.sqrt(((chroma - key_chroma) ** 2).sum(dim=-1))
        rgb_dist = torch.sqrt(((image - key_color) ** 2).sum(dim=-1))

        luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        key_luma = key_color[0] * 0.299 + key_color[1] * 0.587 + key_color[2] * 0.114
        luma_gate = luma >= (key_luma * 0.65).clamp(0.18, 0.72)

        # Both distances must agree. Using either distance independently makes
        # pale skin and other low-saturation foreground colors look similar to
        # a bright screen and can connect them to the border component.
        connected_chroma = tolerance + softness * 0.25
        connected_rgb = tolerance * 1.5 + softness * 0.25
        strict_candidate = (chroma_dist <= connected_chroma) & (rgb_dist <= connected_rgb) & luma_gate

        # A one-pixel frame artifact or a lighting gradient can preserve the
        # screen hue while changing brightness enough to fail RGB distance.
        # Keep this hue-only extension deliberately narrow and use it only as
        # part of component analysis below.
        same_hue_limit = max(0.035, min(0.12, tolerance * 0.5 + softness * 0.1))
        # Hue alone is useful for following a shifted screen through a border
        # artifact, but it must never override a confident foreground matte.
        # Dark blue/cyan clothing can share the screen hue while being far from
        # the sampled key in RGB space; the old unconditional hue extension
        # connected those details to the border and erased entire line regions.
        hue_extension = chroma_dist <= same_hue_limit
        # Component cleanup is a residual-background pass, not a second keyer.
        # Trust confident foreground from the soft matte even when its color is
        # close to the screen; otherwise a one-pixel connection can erase a
        # complete dark garment or a long anti-aliased outline.
        candidate = (strict_candidate | hue_extension) & (alpha <= 0.55)

        candidate_np = candidate.detach().cpu().numpy().astype(np.uint8)
        if candidate_np.max() <= 0:
            return alpha

        try:
            _, labels = cv2.connectedComponents(candidate_np, connectivity=4)
        except Exception:
            return alpha

        border_labels = np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ],
            axis=0,
        )
        border_labels = np.unique(border_labels[border_labels > 0])

        # Background may also be fully enclosed by an arm, hair, or clothing.
        # Remove such components only when the preliminary matte itself says
        # that nearly all of the component is background. This keeps similarly
        # colored opaque foreground details intact.
        alpha_np = alpha.detach().cpu().numpy()
        flat_labels = labels.reshape(-1)
        component_count = int(labels.max()) + 1
        pixel_counts = np.bincount(flat_labels, minlength=component_count)
        alpha_sums = np.bincount(flat_labels, weights=alpha_np.reshape(-1), minlength=component_count)
        foreground_counts = np.bincount(
            flat_labels,
            weights=(alpha_np.reshape(-1) >= 0.5).astype(np.float32),
            minlength=component_count,
        )
        safe_counts = np.maximum(pixel_counts, 1)
        mean_alpha = alpha_sums / safe_counts
        foreground_fraction = foreground_counts / safe_counts
        enclosed_background = np.flatnonzero((mean_alpha <= 0.25) & (foreground_fraction <= 0.10))

        removable_labels = np.unique(np.concatenate([border_labels, enclosed_background]))
        removable_labels = removable_labels[removable_labels > 0]
        if removable_labels.size == 0:
            return alpha

        connected_np = np.isin(labels, removable_labels)
        connected = torch.from_numpy(connected_np).to(device=alpha.device, dtype=alpha.dtype)
        suppression = connected * max(0.0, min(1.0, float(amount)))
        return (alpha * (1.0 - suppression)).clamp(0.0, 1.0)

    def _edge_band(self, alpha: torch.Tensor, edge_width: int) -> torch.Tensor:
        if edge_width <= 0:
            return ((alpha > 0.01) & (alpha < 0.99)).float()
        hard = (alpha > 0.5).float()
        dilated = _morph(hard, edge_width, "dilate")
        eroded = _morph(hard, edge_width, "erode")
        return (dilated - eroded).clamp(0.0, 1.0)

    def _recover_foreground(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        key_color: torch.Tensor,
        amount: float,
    ) -> torch.Tensor:
        if amount <= 0.0:
            return image
        safe_alpha = alpha.unsqueeze(-1).clamp(0.08, 1.0)
        reconstructed = (image - (1.0 - safe_alpha) * key_color) / safe_alpha
        reconstructed = reconstructed.clamp(0.0, 1.0)
        edge_weight = (1.0 - (alpha - 0.5).abs() * 2.0).clamp(0.0, 1.0).unsqueeze(-1)
        edge_weight = edge_weight * edge.unsqueeze(-1)
        return torch.lerp(image, reconstructed, edge_weight * amount).clamp(0.0, 1.0)

    def _edge_despill(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        strength: float,
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image

        result = image.clone()
        dom = image[..., dominant_idx]
        other1 = image[..., other_indices[0]]
        other2 = image[..., other_indices[1]]
        limit = torch.maximum(other1, other2) * 0.75 + ((other1 + other2) * 0.5) * 0.25
        corrected_dom = torch.minimum(dom, limit)

        spill = (dom - limit).clamp(0.0, 1.0)
        neutral = image.clone()
        neutral[..., dominant_idx] = corrected_dom
        neutral[..., other_indices[0]] = (neutral[..., other_indices[0]] + spill * 0.25).clamp(0.0, 1.0)
        neutral[..., other_indices[1]] = (neutral[..., other_indices[1]] + spill * 0.25).clamp(0.0, 1.0)

        edge_weight = torch.maximum(edge, ((alpha > 0.0) & (alpha < 0.98)).float() * 0.5).unsqueeze(-1)
        return torch.lerp(result, neutral, edge_weight * strength).clamp(0.0, 1.0)

    def _choke_spill_edge(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        key_color: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        amount: float,
    ) -> torch.Tensor:
        if amount <= 0.0:
            return alpha

        dom = image[..., dominant_idx]
        other_max = torch.maximum(image[..., other_indices[0]], image[..., other_indices[1]])
        other_avg = (image[..., other_indices[0]] + image[..., other_indices[1]]) * 0.5
        screen_excess = dom - (other_max * 0.65 + other_avg * 0.35)

        key_dom = key_color[dominant_idx]
        key_other_max = torch.maximum(key_color[other_indices[0]], key_color[other_indices[1]])
        key_other_avg = (key_color[other_indices[0]] + key_color[other_indices[1]]) * 0.5
        key_excess = torch.clamp(key_dom - (key_other_max * 0.65 + key_other_avg * 0.35), min=0.05)

        spill_weight = self._smoothstep(key_excess * 0.08, key_excess * 0.55 + 1e-6, screen_excess)
        choke = edge * spill_weight * amount
        return (alpha * (1.0 - choke)).clamp(0.0, 1.0)

    def _edge_decontaminate(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        key_color: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        amount: float,
    ) -> torch.Tensor:
        if amount <= 0.0:
            return image

        dom = image[..., dominant_idx]
        other_max = torch.maximum(image[..., other_indices[0]], image[..., other_indices[1]])
        other_avg = (image[..., other_indices[0]] + image[..., other_indices[1]]) * 0.5
        screen_excess = (dom - (other_max * 0.7 + other_avg * 0.3)).clamp(0.0, 1.0)

        key_strength = torch.clamp(key_color[dominant_idx], min=0.1)
        subtract_amount = (screen_excess / key_strength).clamp(0.0, 1.0)

        decontaminated = image - key_color * (subtract_amount * edge).unsqueeze(-1)
        decontaminated = decontaminated.clamp(0.0, 1.0)

        src_luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        dst_luma = decontaminated[..., 0] * 0.299 + decontaminated[..., 1] * 0.587 + decontaminated[..., 2] * 0.114
        luma_gain = (src_luma / (dst_luma + 1e-4)).clamp(0.5, 1.5).unsqueeze(-1)
        decontaminated = (decontaminated * luma_gain).clamp(0.0, 1.0)

        return torch.lerp(image, decontaminated, edge.unsqueeze(-1) * amount).clamp(0.0, 1.0)

    def _bleed_clean_edge_colors(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        key_color: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        radius: int,
        amount: float,
    ) -> torch.Tensor:
        """Replace key-contaminated edge RGB with nearby opaque foreground RGB."""
        if amount <= 0.0 or radius <= 0:
            return image

        # A 0.98 matte pixel is still visibly blended with the screen. Treating
        # it as clean foreground makes the nearest-color lookup point back to
        # the contaminated pixel itself, leaving a dotted halo untouched.
        # Prefer genuinely opaque color anchors and retain the old threshold
        # only as a fallback for mattes that never reach full opacity.
        opaque_np = (alpha >= 0.995).detach().cpu().numpy()
        if not opaque_np.any():
            opaque_np = (alpha >= 0.98).detach().cpu().numpy()
        if not opaque_np.any():
            return image

        # Pull reference colors from just inside the silhouette. Boundary
        # pixels can reach alpha=1 while their RGB still contains screen color,
        # especially after image scaling. Using them as distance-transform
        # seeds merely copies the halo along the contour.
        erosion_iterations = max(1, min(2, int(radius) // 2))
        trusted_opaque_np = cv2.erode(
            opaque_np.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=erosion_iterations,
        ).astype(bool)
        if not trusted_opaque_np.any():
            trusted_opaque_np = opaque_np

        distance_np, labels = cv2.distanceTransformWithLabels(
            (~trusted_opaque_np).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        image_np = image.detach().cpu().numpy()
        nearest_lookup = np.zeros((int(labels.max()) + 1, 3), dtype=image_np.dtype)
        opaque_y, opaque_x = np.nonzero(trusted_opaque_np)
        nearest_lookup[labels[opaque_y, opaque_x]] = image_np[opaque_y, opaque_x]
        nearest = torch.from_numpy(nearest_lookup[labels]).to(device=image.device, dtype=image.dtype)

        dom = image[..., dominant_idx]
        other1 = image[..., other_indices[0]]
        other2 = image[..., other_indices[1]]
        other_max = torch.maximum(other1, other2)
        other_avg = (other1 + other2) * 0.5
        screen_excess = (dom - (other_max * 0.7 + other_avg * 0.3)).clamp(0.0, 1.0)

        key_dom = key_color[dominant_idx]
        key_other1 = key_color[other_indices[0]]
        key_other2 = key_color[other_indices[1]]
        key_other_max = torch.maximum(key_other1, key_other2)
        key_other_avg = (key_other1 + key_other2) * 0.5
        key_excess = torch.clamp(key_dom - (key_other_max * 0.7 + key_other_avg * 0.3), min=0.05)
        dominant_affinity = self._smoothstep(key_excess * 0.08, key_excess * 0.55 + 1e-6, screen_excess)

        # Dominant-channel despill misses a teal screen mixed into blue
        # foreground because the contaminated blue channel can remain higher
        # than green. Detect that case from the full RGB trajectory between the
        # nearest opaque foreground color and the sampled screen color.
        key_direction = key_color.reshape(1, 1, 3) - nearest
        direction_norm_sq = (key_direction * key_direction).sum(dim=-1).clamp(min=1e-5)
        projection = (((image - nearest) * key_direction).sum(dim=-1) / direction_norm_sq).clamp(0.0, 1.0)
        projected_color = nearest + projection.unsqueeze(-1) * key_direction
        orthogonal_error = torch.sqrt(((image - projected_color) ** 2).sum(dim=-1))
        relative_error = orthogonal_error / torch.sqrt(direction_norm_sq)
        # Resampling and compression bend a real spill trajectory away from an
        # ideal RGB line. A narrow 0.30 cutoff left alternating cyan/green
        # pixels behind on otherwise clean blue outlines.
        trajectory_affinity = 1.0 - self._smoothstep(0.06, 0.60, relative_error)
        # Even a 5-10% screen contribution is visible as a saturated one-pixel
        # halo after compositing. Reach full correction early; trajectory
        # affinity, rather than contribution size, guards unrelated edge color.
        projected_affinity = self._smoothstep(0.005, 0.08, projection) * trajectory_affinity
        spill_affinity = torch.maximum(dominant_affinity, projected_affinity)

        # Guided refinement can leave screen-contaminated pixels at 0.98-0.99
        # alpha slightly inside the hard 0.5 matte contour. Include those
        # uncertain colors in despill without changing their alpha or widening
        # the geometric edge band used by matte cleanup.
        uncertain_color = ((alpha > 0.001) & (alpha < 0.995)).to(dtype=alpha.dtype)
        color_edge = torch.maximum(edge, uncertain_color)
        partial_weight = color_edge * spill_affinity * max(0.0, min(1.0, float(amount)))
        distance = torch.from_numpy(distance_np).to(device=alpha.device, dtype=alpha.dtype)
        transparent_near_edge = ((alpha <= 0.001) & (distance <= float(radius))).to(dtype=alpha.dtype)
        weight = torch.maximum(partial_weight, transparent_near_edge).unsqueeze(-1)
        return torch.lerp(image, nearest, weight).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {
    "VNCCSChromaKey": VNCCSChromaKey,
    "VNCCS_MaskExtractor": VNCCS_MaskExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSChromaKey": "VNCCS Chroma Key",
    "VNCCS_MaskExtractor": "VNCCS Mask Extractor",
}
