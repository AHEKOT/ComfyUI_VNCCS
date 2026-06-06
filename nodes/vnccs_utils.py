"""VNCCS Utility Nodes
Contains utility image processing nodes for future extraction to separate repository:
- VNCCSChromaKey - Green screen removal
- VNCCS_ColorFix - Contrast and saturation adjustment
- VNCCS_Resize - Image resizing
- VNCCS_MaskExtractor - Alpha channel extraction
- VNCCS_RMBG2 - Background removal using various models
"""

import os
import json
import random
import base64
import io
import torch
import numpy as np
from typing import Tuple
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import torch.nn.functional as F

try:
    from .model_path_utils import get_full_path_agnostic
except Exception:
    from model_path_utils import get_full_path_agnostic


class _CompatReturnTypes(tuple):
    def __getitem__(self, index):
        if isinstance(index, int) and index >= len(self):
            index = len(self) - 1
        return super().__getitem__(index)


try:
    import folder_paths
    from huggingface_hub import hf_hub_download
except ImportError:
    folder_paths = None

# Device selection used by RMBG/BiRefNet models
def _select_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


device = _select_torch_device()
OUTFITS_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "character_template",
    "outfits.json",
)
QWEN_VL_MODEL_NAMES = [
    "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
    "qwen2-vl-7b-instruct-q4_k_m.gguf",
]
QWEN_VL_MMPROJ_NAMES = [
    "mmproj-F16.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F32.gguf",
    "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
    "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
]

# Ensure RMBG model folder paths are registered
if folder_paths:
    folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))
    folder_paths.add_model_folder_path("birefnet", os.path.join(folder_paths.models_dir, "RMBG", "BiRefNet"))

# --- Shared helpers ---
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


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
        print(f"[VNCCS Batch Safety] Normalizing {stage}: {shapes} -> {target_shape}")
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


def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)


def _validate_gguf_file(path, file_label="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_label} was not written: {path}")

    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise ValueError(f"{file_label} is too small to be a valid GGUF file ({size} bytes)")

    with open(path, "rb") as file:
        magic = file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{file_label} is not a valid GGUF file (magic={magic!r})")


def _llm_search_dirs():
    if not folder_paths or not hasattr(folder_paths, "models_dir"):
        return []
    base_path = folder_paths.models_dir
    return [os.path.join(base_path, "LLM"), os.path.join(base_path, "llm"), base_path]


def _find_qwen_vl_model():
    for directory in _llm_search_dirs():
        if not os.path.isdir(directory):
            continue
        for name in QWEN_VL_MODEL_NAMES:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def _find_qwen_vl_mmproj(model_path):
    if not model_path:
        return None

    model_dir = os.path.dirname(model_path)
    for name in QWEN_VL_MMPROJ_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path

    try:
        for filename in os.listdir(model_dir):
            if "mmproj" in filename.lower() and filename.lower().endswith(".gguf"):
                return os.path.join(model_dir, filename)
    except OSError:
        return None
    return None


def _get_qwen_vl_chat_handler(llama_cpp):
    chat_format = getattr(llama_cpp, "llama_chat_format", None)
    if chat_format is None:
        try:
            import llama_cpp.llama_chat_format as chat_format
        except Exception:
            chat_format = None

    if chat_format is not None:
        for name in ("Qwen25VLChatHandler", "Qwen2VLChatHandler"):
            handler = getattr(chat_format, name, None)
            if handler is not None:
                return handler
        for name in dir(chat_format):
            if "Qwen" in name and "VL" in name and "Handler" in name:
                return getattr(chat_format, name)
        handler = getattr(chat_format, "Llava15ChatHandler", None)
        if handler is not None:
            return handler

    raise RuntimeError("No QwenVL/Llava chat handler found in llama-cpp-python.")


def _tensor_to_vl_data_uri(image, max_size=768):
    if isinstance(image, list):
        image = image[0]
    if len(image.shape) == 4:
        image = image[0]

    pil_image = tensor2pil(_ensure_float01(image)).convert("RGB")
    width, height = pil_image.size
    if max(width, height) > max_size:
        scale = max_size / float(max(width, height))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_clothing_tags(clothing_tags):
    if isinstance(clothing_tags, list):
        clothing_tags = clothing_tags[0]
    return str(clothing_tags or "").strip().strip(",").strip()


def _build_vl_analyzer_prompt(clothing_tags):
    tags = _normalize_clothing_tags(clothing_tags)
    tags_text = tags if tags else "(no clothing tags provided)"
    return f"""
Analyze the clothes worn by the character in the image.

Clothing tags that must be used as mandatory hints:
{tags_text}

Rules:
- Output plain natural language only.
- Describe only clothing and wearable accessories.
- Use the image as visual evidence, but you must account for the provided clothing tags when describing the outfit.
- If a tag is not clearly visible but is plausible, include it in natural wording.
- Do not output raw comma-separated tags.
- Do not mention pose, body shape, face, hair, background, camera, image quality, nudity, or sex acts.
- Do not use elegant, poetic, luxury, fashion-magazine, or marketing phrases.
- Keep it practical and direct, 1 to 3 short sentences.
- If color or material is unclear, use simple cautious wording.
""".strip()


class VNCCS_ClothesTemplates:
    """Return a random clothes tag template from character_template/outfits.json."""

    ALL_AESTHETICS = "ВСЕ"
    OUTFITS_PATH = OUTFITS_JSON_PATH

    @classmethod
    def _load_outfits(cls) -> list[dict]:
        try:
            with open(cls.OUTFITS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise RuntimeError(f"VNCCS Clothes Templates: outfits.json not found: {cls.OUTFITS_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VNCCS Clothes Templates: invalid outfits.json: {exc}") from exc

        if not isinstance(data, list):
            raise RuntimeError("VNCCS Clothes Templates: outfits.json must contain a list of outfit records.")
        return [entry for entry in data if isinstance(entry, dict) and str(entry.get("content", "")).strip()]

    @classmethod
    def _aesthetic_choices(cls) -> list[str]:
        try:
            outfits = cls._load_outfits()
            aesthetics = sorted({str(entry.get("aesthetic", "")).strip() for entry in outfits if entry.get("aesthetic")})
        except RuntimeError:
            aesthetics = []
        return [cls.ALL_AESTHETICS] + aesthetics

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aesthetic": (cls._aesthetic_choices(), {"default": cls.ALL_AESTHETICS}),
                "is_explicit": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    CATEGORY = "VNCCS"
    FUNCTION = "random_template"

    @classmethod
    def IS_CHANGED(cls, aesthetic, is_explicit):
        return random.random()

    def random_template(self, aesthetic=ALL_AESTHETICS, is_explicit=True):
        if isinstance(aesthetic, list):
            aesthetic = aesthetic[0]
        if isinstance(is_explicit, list):
            is_explicit = is_explicit[0]

        aesthetic = str(aesthetic)
        want_explicit = bool(is_explicit)
        outfits = self._load_outfits()

        matches = [
            entry for entry in outfits
            if (aesthetic in (self.ALL_AESTHETICS, "ALL") or str(entry.get("aesthetic", "")) == aesthetic)
            and bool(entry.get("is_explicit", False)) == want_explicit
        ]

        if not matches:
            explicit_label = "true" if want_explicit else "false"
            raise RuntimeError(
                f"VNCCS Clothes Templates: no outfits found for aesthetic='{aesthetic}', is_explicit={explicit_label}."
            )

        content = str(random.choice(matches).get("content", "")).strip().strip(",").strip()
        return (content,)


class VNCCS_VLAnalyzer:
    """Analyze character clothing from an image using the same Qwen VL GGUF stack as the wizards."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clothing_tags": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    CATEGORY = "VNCCS"
    FUNCTION = "analyze_clothes"

    def analyze_clothes(self, image, clothing_tags=""):
        try:
            import llama_cpp
            import llama_cpp.llama_chat_format
        except Exception as exc:
            raise RuntimeError(f"llama-cpp-python is required for VNCCS VL analyzer: {exc}") from exc

        model_path = _find_qwen_vl_model()
        if not model_path:
            raise RuntimeError("VNCCS VL analyzer: no QwenVL GGUF model found.")

        try:
            _validate_gguf_file(model_path, os.path.basename(model_path))
        except Exception as exc:
            raise RuntimeError(f"VNCCS VL analyzer: QwenVL model file is invalid or incomplete: {exc}") from exc

        mmproj_path = _find_qwen_vl_mmproj(model_path)
        if not mmproj_path:
            raise RuntimeError("VNCCS VL analyzer: no QwenVL vision projector (mmproj) GGUF found.")

        try:
            _validate_gguf_file(mmproj_path, os.path.basename(mmproj_path))
        except Exception as exc:
            raise RuntimeError(f"VNCCS VL analyzer: QwenVL vision projector file is invalid or incomplete: {exc}") from exc

        prompt = _build_vl_analyzer_prompt(clothing_tags)
        image_uri = _tensor_to_vl_data_uri(image)
        HandlerCls = _get_qwen_vl_chat_handler(llama_cpp)

        print(f"[VNCCS VL Analyzer] Loading model: {model_path}")
        print(f"[VNCCS VL Analyzer] Loading mmproj: {mmproj_path}")
        chat_handler = HandlerCls(clip_model_path=mmproj_path, verbose=False)
        llm = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

        system_prompt = (
            "You are a practical anime/game character clothing analyst. "
            "Describe visible clothes in simple natural language. "
            "Follow the user's clothing tag hints."
        )
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]},
            ],
            max_tokens=350,
            temperature=0.2,
        )

        content = response["choices"][0]["message"]["content"]
        description = str(content or "").strip()
        if not description:
            raise RuntimeError("VNCCS VL analyzer: empty response from QwenVL.")

        print(f"[VNCCS VL Analyzer] Output: {description}")
        return (description,)


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





# --- Guided Filter Helper ---
class FastGuidedFilter:
    """
    Fast Guided Filter implementation in PyTorch.
    Used for mask refinement (matting) by using the original image as guidance.
    """
    def __init__(self, r: int, eps: float):
        self.r = r
        self.eps = eps

    def box_filter(self, x: torch.Tensor):
        return F.avg_pool2d(x, self.r * 2 + 1, stride=1, padding=self.r)

    def filter(self, guidance: torch.Tensor, input_mask: torch.Tensor):
        """
        guidance: [1, 3, H, W]
        input_mask: [1, 1, H, W]
        """
        # guidance should be luminance or handled per-channel, 
        # but for matting we usually use the RGB channels to build the model
        
        # Mean
        mean_I = self.box_filter(guidance)
        mean_p = self.box_filter(input_mask)
        mean_Ip = self.box_filter(guidance * input_mask)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = self.box_filter(guidance * guidance)
        var_I = mean_II - mean_I * mean_I
        
        # Linear coefficients
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        
        # Average coefficients
        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)
        
        # Resulting sharpened mask
        output = (mean_a * guidance + mean_b).mean(dim=1, keepdim=True)
        return torch.clamp(output, 0.0, 1.0)


# --- Color Fix Node ---
class VNCCS_ColorFix:
    """Adjust contrast and saturation of an image. Supports alpha channel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "VNCCS"
    FUNCTION = "color_fix"

    def _apply_to_rgb(self, rgb: torch.Tensor, contrast: float, saturation: float) -> torch.Tensor:
        lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        lum = lum.unsqueeze(-1)
        rgb = lum * (1.0 - saturation) + rgb * saturation
        rgb = (rgb - 0.5) * contrast + 0.5
        return rgb.clamp(0.0, 1.0)

    def color_fix(self, image, contrast=1.0, saturation=1.0):
        image = _normalize_image_batch(image, stage="color fix")

        if len(image.shape) == 4:
            results = []
            for i in range(image.shape[0]):
                out = self._process_single(image[i], contrast, saturation)
                results.append(out)
            return (torch.stack(results),)

        out = self._process_single(image, contrast, saturation)
        return (out.unsqueeze(0),)

    def _process_single(self, img: torch.Tensor, contrast: float, saturation: float) -> torch.Tensor:
        img = _ensure_float01(img)
        h, w, c = img.shape

        has_alpha = (c == 4)
        if has_alpha:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
        else:
            rgb = img
            alpha = None

        rgb = self._apply_to_rgb(rgb, float(contrast), float(saturation))

        if has_alpha:
            out = torch.cat([rgb, alpha.clamp(0.0, 1.0)], dim=-1)
        else:
            out = rgb

        return out


# --- Resize Node ---
class VNCCS_Resize:
    """Resize an image to specified width and height using chosen resample method."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "method": (["nearest", "bilinear", "bicubic", "lanczos"], {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "VNCCS"
    FUNCTION = "resize"

    def _ensure_uint8_pil(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor
        if not torch.is_floating_point(t):
            t = t.float()
        if t.max() <= 1.5:
            arr = (t.cpu().numpy() * 255.0).astype('uint8')
        else:
            arr = t.cpu().numpy().astype('uint8')
        return Image.fromarray(arr)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        a = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(a)

    def resize(self, image, width, height, method="bilinear"):
        image = _normalize_image_batch(image, target_hw=(int(height), int(width)), stage="resize")

        if len(image.shape) == 4:
            results = []
            for i in range(image.shape[0]):
                out = self._resize_single(image[i], int(width), int(height), method)
                results.append(out)
            return (torch.stack(results),)

        out = self._resize_single(image, int(width), int(height), method)
        return (out.unsqueeze(0),)

    def _resize_single(self, img: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
        img = _ensure_float01(img)
        h, w, c = img.shape
        has_alpha = (c == 4)

        pil_img = self._ensure_uint8_pil(img[..., :3] if has_alpha else img)

        resample_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }

        resample = resample_map.get(method, Image.BILINEAR)
        pil_resized = pil_img.resize((width, height), resample=resample)

        if has_alpha:
            pil_alpha = self._ensure_uint8_pil(img[..., 3])
            pil_alpha_resized = pil_alpha.resize((width, height), resample=Image.NEAREST)
            pil_rgba = Image.merge('RGBA', (*pil_resized.split(), pil_alpha_resized))
            out = self._pil_to_tensor(pil_rgba)
        else:
            out = self._pil_to_tensor(pil_resized)

        return out


# --- Mask Extractor Node ---
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

    def _hex_to_rgb_norm(self, hex_color: str = "#00FF00"):
        c = (hex_color or "#00FF00").strip().lstrip('#')
        if len(c) == 3:
            c = ''.join(ch*2 for ch in c)
        c = c[:6].ljust(6, '0')
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
        return r, g, b

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
        r, g, b = self._hex_to_rgb_norm()
        device = rgb.device
        dtype = rgb.dtype
        bg = torch.tensor([r, g, b], dtype=dtype, device=device).view(1, 1, 1, 3)
        alpha3 = alpha.unsqueeze(-1)
        out = rgb * alpha3 + bg * (1.0 - alpha3)
        if added_batch:
            out = out.squeeze(0)
        return (out,)


# --- RMBG Model Loaders ---
# Security model: RMBG/BEN architecture code is vendored in this repository.
# The HuggingFace downloads below are limited to model weights/config data.
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "revision": "1cd4787601caeb4c8e826dba7ea8e2163b5208df",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors"
        },
        "cache_dir": "RMBG-2.0"
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "revision": "0f5946638dcbc7dbc1e56306b823959958ac0ae0",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors"
        },
        "cache_dir": "INSPYRENET"
    },
    "BEN": {
        "type": "ben",
        "repo_id": "1038lab/BEN",
        "revision": "12124b2fa3f6a519e7b8771a81fa9f5cb5ed5078",
        "files": {
            "BEN_Base.pth": "BEN_Base.pth"
        },
        "cache_dir": "BEN"
    },
    "BEN2": {
        "type": "ben2",
        "repo_id": "1038lab/BEN2",
        "revision": "7e1bfdf0b53c9d93d82ed2bccf240ba33b3aed38",
        "files": {
            "BEN2_Base.pth": "BEN2_Base.pth"
        },
        "cache_dir": "BEN2"
    }
}


class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG") if folder_paths else "models/RMBG"
    
    def get_cache_dir(self, model_name):
        cache_path = os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        env_key = "VNCCS_" + "".join(ch if ch.isalnum() else "_" for ch in model_name.upper()) + "_REVISION"
        revision = os.environ.get(env_key) or model_info.get("revision")
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    revision=revision,
                    local_dir=cache_dir
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        self.model = None
        self.current_model_version = None


class RMBGModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            cache_dir = self.get_cache_dir(model_name)
            try:
                try:
                    from ._vendored_birefnet import BiRefNet
                    from ._vendored_birefnet_config import BiRefNetConfig
                except Exception:
                    from _vendored_birefnet import BiRefNet
                    from _vendored_birefnet_config import BiRefNetConfig

                self.model = BiRefNet(config=BiRefNetConfig())
                weights_path = os.path.join(cache_dir, "model.safetensors")
                try:
                    try:
                        import safetensors.torch
                        self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                    except ImportError:
                        from transformers.modeling_utils import load_state_dict
                        state_dict = load_state_dict(weights_path)
                        self.model.load_state_dict(state_dict)
                except Exception as load_error:
                    pytorch_weights = os.path.join(cache_dir, "pytorch_model.bin")
                    if os.path.exists(pytorch_weights):
                        self.model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
                    else:
                        raise RuntimeError(f"Failed to load weights: {str(load_error)}")

            except Exception as e:
                handle_model_error(f"Error loading model: {str(e)}")

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
            
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            original_sizes = [tensor2pil(img).size for img in images]

            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                outputs = self.model(input_batch)
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    results = outputs[-1].sigmoid().cpu()
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    results = outputs['logits'].sigmoid().cpu()
                elif isinstance(outputs, torch.Tensor):
                    results = outputs.sigmoid().cpu()
                else:
                    try:
                        if hasattr(outputs, 'last_hidden_state'):
                            results = outputs.last_hidden_state.sigmoid().cpu()
                        else:
                            for k, v in outputs.items():
                                if isinstance(v, torch.Tensor):
                                    results = v.sigmoid().cpu()
                                    break
                    except:
                        handle_model_error("Unable to recognize model output format")
                
                masks = []
                
                for i, (result, (orig_w, orig_h)) in enumerate(zip(results, original_sizes)):
                    result = result.squeeze()
                    result = result * (1 + (1 - params["sensitivity"]))
                    result = torch.clamp(result, 0, 1)

                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0),
                                         size=(orig_h, orig_w),
                                         mode='bilinear').squeeze()
                    
                    masks.append(tensor2pil(result))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")


class CustomBiRefNetModel(RMBGModel):
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            model_path = get_full_path_agnostic(folder_paths, "birefnet", model_name, require_exists=True) if folder_paths else None
            if not model_path or not os.path.exists(model_path):
                raise RuntimeError(f"Custom model file {model_name} not found")

            if os.path.isdir(model_path):
                handle_model_error(
                    "Loading local HuggingFace BiRefNet directories is disabled because it requires "
                    "trust_remote_code. Use a safetensors/pth BiRefNet weights file with the pinned "
                    "RMBG-2.0 architecture instead."
                )
            else:
                try:
                    try:
                        from ._vendored_birefnet import BiRefNet
                        from ._vendored_birefnet_config import BiRefNetConfig
                    except Exception:
                        from _vendored_birefnet import BiRefNet
                        from _vendored_birefnet_config import BiRefNetConfig

                    self.model = BiRefNet(config=BiRefNetConfig())
                    weights_path = model_path
                    try:
                        if weights_path.endswith(".safetensors"):
                            try:
                                import safetensors.torch
                                state_dict = safetensors.torch.load_file(weights_path)
                            except ImportError:
                                from transformers.modeling_utils import load_state_dict
                                state_dict = load_state_dict(weights_path)
                        else:
                            state_dict = torch.load(weights_path, map_location="cpu")
                            
                        if "state_dict" in state_dict:
                            state_dict = state_dict["state_dict"]
                        elif "net" in state_dict:
                            state_dict = state_dict["net"]
                        
                        unwrapped_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('module.'):
                                unwrapped_state_dict[k[7:]] = v
                            else:
                                unwrapped_state_dict[k] = v

                        self.model.load_state_dict(unwrapped_state_dict)
                    except Exception as load_error:
                        raise RuntimeError(f"Failed to load weights from {weights_path}: {str(load_error)}")

                except Exception as e:
                    handle_model_error(f"Error loading custom BiRefNet file {model_name}: {str(e)}")

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name


class InspyrenetModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                import transparent_background
                self.model = transparent_background.Remover()
                self.current_model_version = model_name
            except Exception as e:
                handle_model_error(f"Failed to initialize transparent_background: {str(e)}")
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            foreground = self.model.process(resized_image, type='rgba')
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in Inspyrenet processing: {str(e)}")


class BENModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            try:
                from ._vendored_ben import BEN_Base
            except Exception:
                from _vendored_ben import BEN_Base
            
            model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
            self.model = BEN_Base()
            self.model.loadcheckpoints(model_weights_path)
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            processed_input = resized_image.convert("RGBA")
            
            with torch.no_grad():
                _, foreground = self.model.inference(processed_input)
            
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in BEN processing: {str(e)}")


class BEN2Model(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                cache_dir = self.get_cache_dir(model_name)
                try:
                    from ._vendored_ben2 import BEN_Base
                except Exception:
                    from _vendored_ben2 import BEN_Base
                
                model_weights_path = os.path.join(cache_dir, "BEN2_Base.pth")
                self.model = BEN_Base()
                self.model.loadcheckpoints(model_weights_path)
                
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.current_model_version = model_name
                
            except Exception as e:
                handle_model_error(f"Error loading BEN2 model: {str(e)}")
    
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)
            
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]
            
            batch_size = 3
            all_masks = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_pil_images = []
                original_sizes = []
                
                for img in batch_images:
                    orig_image = tensor2pil(img)
                    w, h = orig_image.size
                    original_sizes.append((w, h))
                    
                    aspect_ratio = h / w
                    new_w = params["process_res"]
                    new_h = int(params["process_res"] * aspect_ratio)
                    resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
                    processed_input = resized_image.convert("RGBA")
                    batch_pil_images.append(processed_input)
                
                with torch.no_grad():
                    try:
                        foregrounds = self.model.inference(batch_pil_images)
                        if not isinstance(foregrounds, list):
                            foregrounds = [foregrounds]
                    except Exception as e:
                        handle_model_error(f"Error in BEN2 inference: {str(e)}")
                
                for foreground, (orig_w, orig_h) in zip(foregrounds, original_sizes):
                    foreground = foreground.resize((orig_w, orig_h), Image.LANCZOS)
                    mask = foreground.split()[-1]
                    all_masks.append(mask)
            
            if len(all_masks) == 1:
                return all_masks[0]
            return all_masks

        except Exception as e:
            handle_model_error(f"Error in BEN2 processing: {str(e)}")


def refine_foreground(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")
    
    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()
    
    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]      
        thresh = 0.45
        mask_binary = (mask > thresh).astype(np.float32)
        
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)
        
        alpha = 0.85
        mask_refined = np.where(transition_mask,
                              alpha * mask + (1-alpha) * edge_blur,
                              mask_binary)
        
        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region,
                              mask_refined * 0.98,
                              mask_refined)
        
        result = []
        for c_idx in range(image_np.shape[1]):
            channel = image_np[i, c_idx]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))


# --- RMBG2 Node ---
class VNCCS_RMBG2:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel(),
            "INSPYRENET": InspyrenetModel(),
            "BEN": BENModel(),
            "BEN2": BEN2Model()
        }
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to be processed for background removal.",
            "model": "Select the background removal model to use.",
            "sensitivity": "Adjust the strength of mask detection.",
            "process_res": "Set the processing resolution.",
            "mask_blur": "Amount of blur to apply to mask edges.",
            "mask_offset": "Adjust the mask boundary.",
            "background": "Choose output type: Alpha (transparent) or Color.",
            "invert_output": "Enable to invert both the image and mask output.",
            "refine_foreground": "Use Fast Foreground Colour Estimation."
        }
        
        birefnet_models = [f for f in folder_paths.get_filename_list("birefnet") if f.endswith((".safetensors", ".pth"))] if folder_paths else []
        all_models = list(AVAILABLE_MODELS.keys()) + birefnet_models
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (all_models, {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "refine_foreground": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Green", "Blue", "White"], {"default": "Alpha"}),
            }
        }

    RETURN_TYPES = _CompatReturnTypes(("IMAGE", "MASK", "IMAGE"))
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "VNCCS"

    def process_image(self, image, model, **params):
        try:
            image = _normalize_image_batch(image, stage="rmbg input")
            processed_images = []
            processed_masks = []
            
            if model in AVAILABLE_MODELS:
                model_instance = self.models[model]
                
                cache_status, message = model_instance.check_model_cache(model)
                if not cache_status:
                    print(f"Cache check: {message}")
                    print("Downloading required model files...")
                    download_status, download_message = model_instance.download_model(model)
                    if not download_status:
                        handle_model_error(download_message)
                    print("Model files downloaded successfully")
                
                model_type = AVAILABLE_MODELS[model]["type"]
            else:
                if not hasattr(self, "custom_birefnet_model"):
                    self.custom_birefnet_model = CustomBiRefNetModel()
                model_instance = self.custom_birefnet_model
                model_type = "rmbg" # BiRefNet behaves identical to rmbg output-wise
            
            def _process_pair(img, mask):
                if isinstance(mask, list):
                    masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
                    mask_local = masks[0] if masks else None
                elif isinstance(mask, Image.Image):
                    mask_local = mask.convert("L")
                else:
                    mask_local = mask
                
                mask_tensor_local = pil2tensor(mask_local)
                mask_tensor_local = mask_tensor_local * (1 + (1 - params["sensitivity"]))
                mask_tensor_local = torch.clamp(mask_tensor_local, 0, 1)
                mask_img_local = tensor2pil(mask_tensor_local)
                
                if params["mask_blur"] > 0:
                    mask_img_local = mask_img_local.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                
                if params["mask_offset"] != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MinFilter(3))
                
                if params["invert_output"]:
                    mask_img_local = Image.fromarray(255 - np.array(mask_img_local))
                
                img_tensor_local = torch.from_numpy(np.array(tensor2pil(img))).permute(2, 0, 1).unsqueeze(0) / 255.0
                mask_tensor_b1hw = torch.from_numpy(np.array(mask_img_local)).unsqueeze(0).unsqueeze(0) / 255.0
                
                orig_image_local = tensor2pil(img)
                
                if params.get("refine_foreground", False):
                    refined_fg_local = refine_foreground(img_tensor_local, mask_tensor_b1hw)
                    refined_fg_local = tensor2pil(refined_fg_local[0].permute(1, 2, 0))
                    r, g, b = refined_fg_local.split()
                    foreground_local = Image.merge('RGBA', (r, g, b, mask_img_local))
                else:
                    orig_rgba_local = orig_image_local.convert("RGBA")
                    r, g, b, _ = orig_rgba_local.split()
                    foreground_local = Image.merge('RGBA', (r, g, b, mask_img_local))
                
                def hex_to_rgba(hex_color):
                    hex_color = hex_color.lstrip('#')
                    if len(hex_color) == 6:
                        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                        a = 255
                    elif len(hex_color) == 8:
                        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
                    else:
                        raise ValueError("Invalid color format")
                    return (r, g, b, a)
                
                if params["background"] == "Green":
                    rgba = hex_to_rgba("#00FF00")
                    bg_image = Image.new('RGBA', orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                elif params["background"] == "Blue":
                    rgba = hex_to_rgba("#0000FF")
                    bg_image = Image.new('RGBA', orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                elif params["background"] == "White":
                    rgba = hex_to_rgba("#FFFFFF")
                    bg_image = Image.new('RGBA', orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                else:
                    processed_images.append(pil2tensor(foreground_local))
                
                processed_masks.append(pil2tensor(mask_img_local))
            
            if model_type in ("rmbg", "ben2"):
                images_list = [img for img in image]
                chunk_size = 4
                for start in range(0, len(images_list), chunk_size):
                    batch_imgs = images_list[start:start + chunk_size]
                    masks = model_instance.process_image(batch_imgs, model, params)
                    if isinstance(masks, Image.Image):
                        masks = [masks]
                    for img_item, mask_item in zip(batch_imgs, masks):
                        _process_pair(img_item, mask_item)
            else:
                for img in image:
                    mask = model_instance.process_image(img, model, params)
                    _process_pair(img, mask)
            
            mask_images = []
            for mask_tensor in processed_masks:
                mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images.append(mask_image)
            
            target_hw = None
            if processed_images:
                first = processed_images[0]
                target_hw = (int(first.shape[1]), int(first.shape[2]))
            mask_image_output = _normalize_image_batch(mask_images, target_hw=target_hw, stage="rmbg mask image")
            processed_image_output = _normalize_image_batch(processed_images, target_hw=target_hw, stage="rmbg output")
            processed_mask_output = torch.cat(processed_masks, dim=0)
            
            return (processed_image_output, processed_mask_output, mask_image_output)
            
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")
            empty_mask = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
            empty_mask_image = empty_mask.reshape((-1, 1, empty_mask.shape[-2], empty_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            return (image, empty_mask, empty_mask_image)


class VNCCSChromaKey:
    """VNCCS Chroma Key - soft chroma key with edge decontamination."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "softness": ("FLOAT", {"default": 0.16, "min": 0.001, "max": 1.0, "step": 0.01}),
                "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "matte_cleanup": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_recover": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_decontaminate": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_choke": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "matte_method": (["chroma_soft", "guided_edge", "pymatting_if_available"], {"default": "guided_edge"}),
                "screen_mode": (["auto", "green", "blue", "red"], {"default": "auto"}),
                "output_mode": (["straight_rgba", "premultiplied_rgba"], {"default": "straight_rgba"}),
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
    ):
        image = _normalize_image_batch(image, stage="chroma key input")
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
        edge = self._edge_band(alpha, int(edge_width))

        recovered = self._recover_foreground(
            image=image,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            amount=float(foreground_recover),
        )
        despilled = self._edge_despill(
            image=recovered,
            alpha=alpha,
            edge=edge,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            strength=float(despill_strength),
        )
        despilled = self._edge_decontaminate(
            image=despilled,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            amount=float(edge_decontaminate),
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
            if pixels.std(dim=0).mean() < 0.02:
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
        subtract_amount = subtract_amount * edge * amount

        decontaminated = image - key_color * subtract_amount.unsqueeze(-1)
        decontaminated = decontaminated.clamp(0.0, 1.0)

        src_luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        dst_luma = decontaminated[..., 0] * 0.299 + decontaminated[..., 1] * 0.587 + decontaminated[..., 2] * 0.114
        luma_gain = (src_luma / (dst_luma + 1e-4)).clamp(0.5, 1.5).unsqueeze(-1)
        decontaminated = (decontaminated * luma_gain).clamp(0.0, 1.0)

        return torch.lerp(image, decontaminated, edge.unsqueeze(-1) * amount).clamp(0.0, 1.0)


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "VNCCS_ClothesTemplates": VNCCS_ClothesTemplates,
    "VNCCS_VLAnalyzer": VNCCS_VLAnalyzer,
    "VNCCSChromaKey": VNCCSChromaKey,
    "VNCCS_ColorFix": VNCCS_ColorFix,
    "VNCCS_Resize": VNCCS_Resize,
    "VNCCS_MaskExtractor": VNCCS_MaskExtractor,
    "VNCCS_RMBG2": VNCCS_RMBG2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_ClothesTemplates": "VNCCS Clothes Templates",
    "VNCCS_VLAnalyzer": "VNCCS VL analyzer",
    "VNCCSChromaKey": "VNCCS Chroma Key",
    "VNCCS_ColorFix": "VNCCS Color Fix",
    "VNCCS_Resize": "VNCCS Resize",
    "VNCCS_MaskExtractor": "VNCCS Mask Extractor",
    "VNCCS_RMBG2": "VNCCS RMBG2",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_ClothesTemplates": "VNCCS",
    "VNCCS_VLAnalyzer": "VNCCS",
    "VNCCSChromaKey": "VNCCS",
    "VNCCS_ColorFix": "VNCCS",
    "VNCCS_Resize": "VNCCS",
    "VNCCS_MaskExtractor": "VNCCS",
    "VNCCS_RMBG2": "VNCCS",
}
