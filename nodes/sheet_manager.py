import os
import torch
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageFilter
from torchvision import transforms
import folder_paths
import shutil
import sys
import importlib.util
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
import cv2
import types
import torch.nn.functional as F

# Device selection used by RMBG models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure RMBG model folder path is registered
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

class VNCCSSheetManager:
    """VNCCS Sheet Manager - split sheets into parts or compose images into square sheets."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["split", "compose"], {"default": "split"}),
                "images": ("IMAGE",),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 6144, "step": 64}),
                "target_height": ("INT", {"default": 3072, "min": 64, "max": 6144, "step": 64}),
            },
            "optional": {
                "safe_margin": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("images",)
    CATEGORY = "VNCCS"
    FUNCTION = "process_sheet"
    INPUT_IS_LIST = True
    DESCRIPTION = """
    VNCCS Sheet Manager - split sheets into parts or compose images into square sheets.
    
    Split mode: Divides sheet into 12 parts (2x6 grid) and resizes each to target dimensions.
    Compose mode: Arranges up to 12 images in 2x6 grid to create target_height×target_height square.
    """

    def split_sheet(self, image_tensor: torch.Tensor, target_width: int, target_height: int) -> List[torch.Tensor]:
        """Split a sheet into 12 parts arranged in 2 rows of 6."""
        # Input: [height, width, channels] for a single image
        height, width, channels = image_tensor.shape
        
        # Calculate actual dimensions per part
        part_width = width // 6  # 6 columns
        part_height = height // 2  # 2 rows
        
        parts = []
        
        # Extract 12 parts: 6 from top row, 6 from bottom row
        for row in range(2):  # 2 rows
            for col in range(6):  # 6 columns
                y_start = row * part_height
                y_end = (row + 1) * part_height
                x_start = col * part_width
                x_end = (col + 1) * part_width
                
                part = image_tensor[y_start:y_end, x_start:x_end, :]
                
                # Resize part to target dimensions if needed
                if part.shape[0] != target_height or part.shape[1] != target_width:
                    part = torch.nn.functional.interpolate(
                        part.unsqueeze(0).permute(0, 3, 1, 2), 
                        size=(target_height, target_width), 
                        mode="bilinear"
                    ).squeeze().permute(1, 2, 0)
                
                parts.append(part)
        
        return parts

    def compose_sheet(self, image_tensors: torch.Tensor, target_height: int, safe_margin: bool = False) -> torch.Tensor:
        """Compose images into a fixed 2x6 grid to create square sheets."""
        # Fixed layout: 2 rows × 6 columns = 12 images
        num_rows = 2
        num_columns = 6
        expected_batch_size = num_rows * num_columns  # 12
        
        # Assuming images is a batch of images (B, H, W, C)
        batch_size, img_height, img_width, channels = image_tensors.shape
        
        print(f"Input: {batch_size} images of {img_height}x{img_width}")
        print(f"Target layout: {num_rows} rows × {num_columns} columns")
        print(f"Expected total images: {expected_batch_size}")
        
        # Handle batch size mismatch
        if batch_size < expected_batch_size:
            # Pad with zeros if less than 12 images
            padding_needed = expected_batch_size - batch_size
            padding = torch.zeros((padding_needed, img_height, img_width, channels), 
                                dtype=image_tensors.dtype, device=image_tensors.device)
            image_tensors = torch.cat([image_tensors, padding], dim=0)
            batch_size = expected_batch_size
        elif batch_size > expected_batch_size:
            # Take only first 12 images if more than 12
            image_tensors = image_tensors[:expected_batch_size]
            batch_size = expected_batch_size
        
        margin = 4 if safe_margin else 0

        # Calculate target cell size for square sheet
        # For 2x6 grid to fit in target_height x target_height square:
        # cell_height = target_height // 2
        # cell_width = target_height // 6
        cell_height = target_height // 2
        cell_width = target_height // 6

        # Ensure dimensions are divisible by 8
        cell_height = max(1, (cell_height // 8) * 8)
        cell_width = max(1, (cell_width // 8) * 8)

        sheet_height = num_rows * cell_height
        sheet_width = num_columns * cell_width

        print(f"Cell size: {cell_height}x{cell_width}")
        print(f"Final sheet dimensions: {sheet_height}x{sheet_width}")

        # Create the final sheet
        sheet = torch.zeros((sheet_height, sheet_width, channels), dtype=image_tensors.dtype, device=image_tensors.device)

        if safe_margin:
            # Fill sheet with pure green background (and opaque alpha if present)
            if channels >= 3:
                sheet[:, :, 0] = 0.0
                sheet[:, :, 1] = 1.0
                sheet[:, :, 2] = 0.0
            if channels == 4:
                sheet[:, :, 3] = 1.0
        start_y = 0
        start_x = 0
        
        for idx, image in enumerate(image_tensors):
            target_inner_height = max(1, cell_height - 2 * margin)
            target_inner_width = max(1, cell_width - 2 * margin)

            # Resize image to fit within margins (or full cell if margin disabled)
            resized_image = torch.nn.functional.interpolate(
                image.unsqueeze(0).permute(0, 3, 1, 2), 
                size=(target_inner_height, target_inner_width), 
                mode="bilinear"
            ).squeeze().permute(1, 2, 0)
            
            # Calculate position in grid
            row = idx // num_columns
            col = idx % num_columns
            
            # Place image in sheet
            cell_origin_y = start_y + row * cell_height
            cell_origin_x = start_x + col * cell_width

            y_start = cell_origin_y + margin
            y_end = y_start + resized_image.shape[0]
            x_start = cell_origin_x + margin
            x_end = x_start + resized_image.shape[1]
            
            sheet[y_start:y_end, x_start:x_end, :] = resized_image
        
        return sheet.unsqueeze(0)

    def process_sheet(self, mode, images, target_width, target_height, safe_margin=False):
        """Main processing function."""
        # Handle list inputs since INPUT_IS_LIST = True
        mode = mode[0] if isinstance(mode, list) else mode
        target_width = target_width[0] if isinstance(target_width, list) else target_width
        target_height = target_height[0] if isinstance(target_height, list) else target_height
        safe_margin = safe_margin[0] if isinstance(safe_margin, list) else safe_margin
        
        if mode == "split":
            # Split expects a single image [height, width, channels]
            if isinstance(images, list):
                images = images[0]  # Take first image from list
            if len(images.shape) == 4:
                # If we got a batched image [1, H, W, C], squeeze it
                images = images[0]
            parts = self.split_sheet(images, target_width, target_height)
            # Convert list of parts to list of individual images
            result_list = [part.unsqueeze(0) for part in parts]
        elif mode == "compose":
            # Compose expects a batch [batch, height, width, channels]
            if isinstance(images, list):
                # If images is a list of tensors, concatenate them into a batch
                images = torch.cat(images, dim=0)
            elif len(images.shape) == 3:
                # If we got a single image, add batch dimension
                images = images.unsqueeze(0)
            result = self.compose_sheet(images, target_height, safe_margin)
            # Convert single image to list
            result_list = [result]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return (result_list,)


class VNCCSSheetExtractor:
    """VNCCS Sheet Extractor - returns one of the 12 sheet parts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "part_index": ("INT", {"default": 0, "min": 0, "max": 11}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 6144, "step": 64}),
                "target_height": ("INT", {"default": 3072, "min": 64, "max": 6144, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "VNCCS"
    FUNCTION = "extract"
    DESCRIPTION = """
    Returns one part of the sheet (2×6 grid) at the given index. Indices 0-5 are the top row left-to-right, 6-11 are the bottom row.
    """

    def extract(self, image, part_index, target_width, target_height):
    # Extract actual values (support list inputs)
        part_index = part_index[0] if isinstance(part_index, list) else part_index
        target_width = target_width[0] if isinstance(target_width, list) else target_width
        target_height = target_height[0] if isinstance(target_height, list) else target_height

    # Support for list/batch input
        if isinstance(image, list):
            image = image[0]
        if len(image.shape) == 4:
            image = image[0]

        manager = VNCCSSheetManager()
        parts = manager.split_sheet(image, target_width, target_height)

        if not parts:
            raise ValueError("Failed to split sheet: parts is empty")

    # Clamp index to valid range
        part_index = max(0, min(len(parts) - 1, part_index))
        selected = parts[part_index]

        return (selected.unsqueeze(0),)


class VNCCSChromaKey:
    """VNCCS Chroma Key - simple RGB-based green screen removal."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "despill_kernel_size": ("INT", {"default": 3, "min": 1, "max": 9, "step": 2}),
                "despill_color": (["interior_average", "black"], {"default": "interior_average"}),
            }
        }

    # Now returns a single IMAGE tensor with an alpha channel (RGBA)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "VNCCS"
    FUNCTION = "chroma_key"
    DESCRIPTION = """
    VNCCS Chroma Key - automatically detects background color from image borders.
    Uses RGB distance with tolerance to mask out the background.
    Despill strength controls blending of edge pixels.
    Despill kernel size determines edge detection area.
    Despill color chooses between interior average or black.
    """

    def chroma_key(self, image, tolerance, despill_strength, despill_kernel_size, despill_color):
        """Main chroma key function.

        Returns a single IMAGE tensor containing RGBA data. Alpha is computed as (1 - mask)
        where mask==1 indicates background and mask==0 indicates foreground.
        """
        # Handle batch dimension
        if len(image.shape) == 4:  # [B, H, W, C]
            batch_size = image.shape[0]
            rgba_list = []
            for i in range(batch_size):
                rgb_img, mask = self.chroma_key_single(image[i], tolerance, despill_strength, despill_kernel_size, despill_color)
                # mask: H,W (0..1), rgb_img: H,W,3
                alpha = (1.0 - mask).unsqueeze(-1).clamp(0.0, 1.0)
                rgba = torch.cat([rgb_img, alpha], dim=-1)
                rgba_list.append(rgba)
            return (torch.stack(rgba_list),)
        else:  # Single image [H, W, C]
            rgb_img, mask = self.chroma_key_single(image, tolerance, despill_strength, despill_kernel_size, despill_color)
            alpha = (1.0 - mask).unsqueeze(-1).clamp(0.0, 1.0)
            rgba = torch.cat([rgb_img, alpha], dim=-1)
            return (rgba.unsqueeze(0),)

    def chroma_key_single(self, image, tolerance, despill_strength, despill_kernel_size, despill_color):
        """Process single image - auto-detect background color and mask."""
        # Auto-detect background color from image borders
        height, width, _ = image.shape
        border_width = max(1, min(height, width) // 10)  # 10% of smaller dimension
        
        # Collect border pixels
        top_border = image[:border_width, :, :]
        bottom_border = image[-border_width:, :, :]
        left_border = image[:, :border_width, :]
        right_border = image[:, -border_width:, :]
        
        border_pixels = torch.cat([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ], dim=0)
        
        # Average border color as key color (using median for robustness)
        key_color = border_pixels.median(dim=0)[0]
        key_r, key_g, key_b = key_color[0], key_color[1], key_color[2]
        
        # Compute Euclidean distance in RGB space
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        distance = torch.sqrt((r - key_r)**2 + (g - key_g)**2 + (b - key_b)**2)
        
        # Mask pixels within tolerance
        mask = (distance <= tolerance).float()
        
        # Apply dispill to edges
        corrected_image = self.apply_dispill(image, mask, despill_strength, despill_kernel_size, despill_color)
        
        # Apply mask - make background transparent/black
        final_image = corrected_image * (1 - mask.unsqueeze(-1))
        
        return final_image, mask

    def apply_dispill(self, image, mask, despill_strength, despill_kernel_size, despill_color):
        """Apply dispill correction to edge pixels."""
        # Foreground mask (where mask == 0)
        foreground_mask = (mask == 0).float()
        
        # Erode foreground to get pure interior
        kernel_size = despill_kernel_size
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        padding = kernel_size // 2
        
        # Erode: pixel is foreground only if all neighbors are foreground
        eroded_conv = torch.nn.functional.conv2d(foreground_mask.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
        eroded = (eroded_conv == kernel_size * kernel_size).float().squeeze()
        
        # Edge pixels: foreground pixels that are not in eroded (on the boundary)
        edges = foreground_mask * (1 - eroded)
        edges_bool = edges > 0
        
        # Determine despill color
        if despill_color == "black":
            despill_color_tensor = torch.zeros(3, device=image.device, dtype=image.dtype)
        else:  # interior_average
            # Average color from pure interior (eroded foreground)
            if eroded.sum() > 0:
                interior_pixels = image[eroded > 0]
                despill_color_tensor = interior_pixels.mean(dim=0)
            else:
                # Fallback to all foreground if no eroded interior
                interior_pixels = image[foreground_mask > 0]
                if interior_pixels.numel() > 0:
                    despill_color_tensor = interior_pixels.mean(dim=0)
                else:
                    return image  # No foreground, skip
        
        # Blend edges with despill color
        blended = image.clone()
        edges_expanded = edges_bool.unsqueeze(-1)
        blended = torch.where(edges_expanded, 
                            (1 - despill_strength) * image + despill_strength * despill_color_tensor, 
                            image)
        
        return blended


NODE_CLASS_MAPPINGS = {
    "VNCCSSheetManager": VNCCSSheetManager,
    "VNCCSSheetExtractor": VNCCSSheetExtractor,
    "VNCCSChromaKey": VNCCSChromaKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS Sheet Manager",
    "VNCCSSheetExtractor": "VNCCS Sheet Extractor",
    "VNCCSChromaKey": "VNCCS Chroma Key"
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS",
    "VNCCSSheetExtractor": "VNCCS",
    "VNCCSChromaKey": "VNCCS"
}


class VNCCS_ColorFix:
    """Adjust contrast and saturation of an image. Supports alpha channel.

    Contrast and saturation are multipliers where 1.0 = unchanged, 0.0 = neutral (gray/midpoint for contrast),
    2.0 = doubled effect.
    """

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

    def _ensure_float01(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor
        if not torch.is_floating_point(t):
            t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0.0, 1.0)

    def _apply_to_rgb(self, rgb: torch.Tensor, contrast: float, saturation: float) -> torch.Tensor:
        # rgb: (..., 3) float in [0,1]
        # compute luminance (grayscale) using Rec. 709 / BT.601 approximated weights
        lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        lum = lum.unsqueeze(-1)

        # saturation: blend between grayscale and color; >1.0 will amplify color differences
        rgb = lum * (1.0 - saturation) + rgb * saturation

        # contrast: scale around 0.5 midpoint
        rgb = (rgb - 0.5) * contrast + 0.5

        return rgb.clamp(0.0, 1.0)

    def color_fix(self, image, contrast=1.0, saturation=1.0):
        # support list inputs
        if isinstance(image, list):
            image = image[0]

        added_batch = False
        # support batched input [B,H,W,C]
        if len(image.shape) == 4:
            results = []
            for i in range(image.shape[0]):
                out, _ = self._process_single(image[i], contrast, saturation)
                results.append(out)
            return (torch.stack(results),)

        out, single = self._process_single(image, contrast, saturation)
        # return single image as batch (consistent with other nodes)
        return (out.unsqueeze(0),)

    def _process_single(self, img: torch.Tensor, contrast: float, saturation: float) -> Tuple[torch.Tensor, None]:
        # img: [H,W,C]
        img = self._ensure_float01(img)
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

        return out, None


class VNCCS_Resize:
    """Resize an image to specified width and height using chosen resample method. Supports alpha channel.
    Methods: nearest, bilinear, bicubic, lanczos
    """

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
        # tensor: H,W,C in float [0,1] or int 0-255
        t = tensor
        if not torch.is_floating_point(t):
            t = t.float()
        if t.max() <= 1.5:
            arr = (t.cpu().numpy() * 255.0).astype('uint8')
        else:
            arr = t.cpu().numpy().astype('uint8')
        # arr shape H,W,C
        return Image.fromarray(arr)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        a = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(a)

    def resize(self, image, width, height, method="bilinear"):
        # normalize inputs
        if isinstance(image, list):
            image = image[0]

        # If batched
        if len(image.shape) == 4:
            results = []
            for i in range(image.shape[0]):
                out = self._resize_single(image[i], int(width), int(height), method)
                results.append(out)
            return (torch.stack(results),)

        out = self._resize_single(image, int(width), int(height), method)
        return (out.unsqueeze(0),)

    def _resize_single(self, img: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
        img = self._ensure_float01(img) if hasattr(self, '_ensure_float01') else img
        # ensure float [0,1]
        if not torch.is_floating_point(img):
            img = img.float()
        if img.max() > 1.5:
            img = img / 255.0

        h, w, c = img.shape
        has_alpha = (c == 4)

        # Convert to PIL image for robust resampling
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
            # handle alpha separately to preserve exact alpha channel
            pil_alpha = self._ensure_uint8_pil(img[..., 3])
            pil_alpha_resized = pil_alpha.resize((width, height), resample=Image.NEAREST)
            # merge
            pil_rgba = Image.merge('RGBA', (*pil_resized.split(), pil_alpha_resized))
            out = self._pil_to_tensor(pil_rgba)
        else:
            out = self._pil_to_tensor(pil_resized)

        return out


# VNCCS Mask Extractor - fill alpha with color
class VNCCS_MaskExtractor:
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

    def _ensure_float01(self, tensor):
        t = tensor
        if not torch.is_floating_point(t):
            t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0.0, 1.0)

    def fill_alpha_with_color(self, image):
        if image is None:
            raise ValueError("No image provided")
        img = image
        img = self._ensure_float01(img)
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
        # Use fixed bright green by default
        r, g, b = self._hex_to_rgb_norm()
        device = rgb.device
        dtype = rgb.dtype
        bg = torch.tensor([r, g, b], dtype=dtype, device=device).view(1, 1, 1, 3)
        alpha3 = alpha.unsqueeze(-1)
        out = rgb * alpha3 + bg * (1.0 - alpha3)
        if added_batch:
            out = out.squeeze(0)
        return (out,)

# --- Begin copy of AILab RMBG implementation (kept unchanged) ---
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        "cache_dir": "RMBG-2.0"
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors"
        },
        "cache_dir": "INSPYRENET"
    },
    "BEN": {
        "type": "ben",
        "repo_id": "1038lab/BEN",
        "files": {
            "model.py": "model.py",
            "BEN_Base.pth": "BEN_Base.pth"
        },
        "cache_dir": "BEN"
    },
    "BEN2": {
        "type": "ben2",
        "repo_id": "1038lab/BEN2",
        "files": {
            "BEN2_Base.pth": "BEN2_Base.pth",
            "BEN2.py": "BEN2.py"
        },
        "cache_dir": "BEN2"
    }
}


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)


class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
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
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
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
                # Primary path: Modern transformers compatibility mode (optimized for newer versions)
                try:
                    from transformers import PreTrainedModel
                    import json

                    config_path = os.path.join(cache_dir, "config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    birefnet_path = os.path.join(cache_dir, "birefnet.py")
                    BiRefNetConfig_path = os.path.join(cache_dir, "BiRefNet_config.py")

                    # Load the BiRefNetConfig
                    config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", BiRefNetConfig_path)
                    config_module = importlib.util.module_from_spec(config_spec)
                    sys.modules["BiRefNetConfig"] = config_module
                    config_spec.loader.exec_module(config_module)

                    # Fix and load birefnet module
                    with open(birefnet_path, 'r') as f:
                        birefnet_content = f.read()

                    birefnet_content = birefnet_content.replace(
                        "from .BiRefNet_config import BiRefNetConfig",
                        "from BiRefNetConfig import BiRefNetConfig"
                    )

                    module_name = f"custom_birefnet_model_{hash(birefnet_path)}"
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                    exec(birefnet_content, module.__dict__)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                            BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                            model_config = BiRefNetConfig()
                            self.model = attr(model_config)

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
                            break

                    if self.model is None:
                        raise RuntimeError("Could not find suitable model class")

                except Exception as modern_e:
                    print(f"[RMBG INFO] Using standard transformers loading (fallback mode)...")
                    try:
                        self.model = AutoModelForImageSegmentation.from_pretrained(
                            cache_dir,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    except Exception as standard_e:
                        handle_model_error(f"Failed to load model with both modern and standard methods. Modern error: {str(modern_e)}. Standard error: {str(standard_e)}")

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

            # Prepare batch processing
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
            model_path = os.path.join(cache_dir, "model.py")
            module_name = f"custom_ben_model_{hash(model_path)}"
            
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            ben_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = ben_module
            spec.loader.exec_module(ben_module)
            
            model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
            self.model = ben_module.BEN_Base()
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
                model_path = os.path.join(cache_dir, "BEN2.py")
                module_name = f"custom_ben2_model_{hash(model_path)}"
                
                spec = importlib.util.spec_from_file_location(module_name, model_path)
                ben2_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = ben2_module
                spec.loader.exec_module(ben2_module)
                
                model_weights_path = os.path.join(cache_dir, "BEN2_Base.pth")
                self.model = ben2_module.BEN_Base()
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
        for c in range(image_np.shape[1]):
            channel = image_np[i, c]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))


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
            "model": "Select the background removal model to use (RMBG-2.0, INSPYRENET, BEN).",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "process_res": "Set the processing resolution (higher values require more VRAM and may increase processing time).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "background": "Choose output type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Pick background color (supports alpha, use color picker).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background"
        }
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(AVAILABLE_MODELS.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": tooltips["refine_foreground"]}),
                "background": (["Alpha", "Green", "Blue"], {"default": "Alpha", "tooltip": tooltips["background"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "VNCCS"

    def process_image(self, image, model, **params):
        try:
            processed_images = []
            processed_masks = []
            
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
                
                if params["background"] == "Green":
                    # Use fixed bright green background when Green option is selected
                    background_color = "#00FF00"
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
                    rgba = hex_to_rgba(background_color)
                    bg_image = Image.new('RGBA', orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                elif params["background"] == "Blue":
                    # Use solid blue background when Blue option is selected
                    background_color = "#0000FF"
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
                    rgba = hex_to_rgba(background_color)
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
            
            mask_image_output = torch.cat(mask_images, dim=0)
            
            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), mask_image_output)
            
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")
            empty_mask = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
            empty_mask_image = empty_mask.reshape((-1, 1, empty_mask.shape[-2], empty_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            return (image, empty_mask, empty_mask_image)

class VNCCS_QuadSplitter:
    """Split a square character sheet into 4 equal square quadrants (2x2) and return them as a list.

    Accepts a single IMAGE (H,W,C) or a batched IMAGE ([B,H,W,C]) or a list containing one IMAGE tensor.
    If the incoming image is not square it will be center-cropped to the largest possible square.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["split", "compose"], {"default": "split"}),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("images",)
    CATEGORY = "VNCCS"
    FUNCTION = "process"
    INPUT_IS_LIST = True

    def _ensure_tensor(self, image):
        # Accept list inputs and batched inputs, return single image tensor H,W,C
        if isinstance(image, list):
            image = image[0]
        if len(image.shape) == 4:
            # If batch dimension present and batch==1, squeeze; otherwise take first element
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image[0]
        return image

    def _normalize_image_list(self, images):
        """Normalize various incoming shapes into a flat list of single-image HWC tensors."""
        out = []
        # If a single tensor with batch dim
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 4:
                for i in range(images.shape[0]):
                    out.append(images[i])
                return out
            else:
                return [images]

        # If it's a list-like object
        if isinstance(images, list):
            for item in images:
                # nested list
                if isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, torch.Tensor) and len(sub.shape) == 4 and sub.shape[0] == 1:
                            out.append(sub[0])
                        elif isinstance(sub, torch.Tensor):
                            out.append(sub)
                        else:
                            out.append(sub)
                elif isinstance(item, torch.Tensor):
                    if len(item.shape) == 4 and item.shape[0] > 1:
                        for i in range(item.shape[0]):
                            out.append(item[i])
                    elif len(item.shape) == 4 and item.shape[0] == 1:
                        out.append(item[0])
                    else:
                        out.append(item)
                else:
                    out.append(item)

        return out

    def _center_crop_square(self, img: torch.Tensor) -> torch.Tensor:
        h, w, c = img.shape
        if h == w:
            return img
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        return img[y0:y0 + size, x0:x0 + size, :]

    def split(self, image):
        img = self._ensure_tensor(image)
        if img is None:
            raise ValueError("No image provided to VNCCS_QuadSplitter")

        img = img.clone()
        # Ensure float in [0,1] for consistency (do not modify dtype if already float)
        if not torch.is_floating_point(img):
            img = img.float()
            if img.max() > 1.5:
                img = img / 255.0

        img = self._center_crop_square(img)
        size, _, _ = img.shape
        half = size // 2

        quads = []
        # Top-left
        q1 = img[0:half, 0:half, :]
        # Top-right
        q2 = img[0:half, half:half * 2, :]
        # Bottom-left
        q3 = img[half:half * 2, 0:half, :]
        # Bottom-right
        q4 = img[half:half * 2, half:half * 2, :]

        quads = [q1, q2, q3, q4]

        # Return as a list of single-item batches for compatibility with the node system
        return ([q.unsqueeze(0) for q in quads],)

    def process(self, mode, image):
        """Dispatcher that accepts keyword args from ComfyUI and routes to split or compose.

        mode: 'split' or 'compose' (may be a list-wrapped value)
        image: image tensor or list as provided by ComfyUI
        """
        # normalize mode (support list inputs)
        if isinstance(mode, list):
            mode_val = mode[0]
        else:
            mode_val = mode

        if mode_val not in ("split", "compose"):
            raise ValueError(f"Unknown mode for VNCCS_QuadSplitter: {mode_val}")

        if mode_val == "split":
            return self.split(image)
        else:
            return self.compose(image)

    def compose(self, images):
        """Compose 4 square images (order: top-left, top-right, bottom-left, bottom-right)
        into a single square sheet (2x2 grid). Accepts a list of 4 IMAGE tensors or a batch.
        """
        # Normalize inputs: accept list of tensors, batched tensor, or single tensor
        imgs = images
        if isinstance(imgs, list):
            imgs = imgs
        elif isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4:
            # batch dimension present
            # if batch has exactly 4 elements, use them; if 1 and each element contains 4 via list, try to unwrap
            if imgs.shape[0] == 4:
                imgs = [imgs[i] for i in range(4)]
            elif imgs.shape[0] == 1:
                # single batch, maybe contains the 4 images as channels? unsupported
                imgs = [imgs[0]]
            else:
                imgs = [imgs[i] for i in range(min(4, imgs.shape[0]))]

        if not isinstance(imgs, list):
            raise ValueError("Compose expects a list or batch of images")

        imgs = self._normalize_image_list(imgs)
        if len(imgs) < 4:
            raise ValueError("Compose expects at least 4 images (top-left, top-right, bottom-left, bottom-right)")

        # Ensure each image is a single image tensor H,W,C
        norm = []
        for im in imgs[:4]:
            if isinstance(im, list):
                im = im[0]
            if len(im.shape) == 4 and im.shape[0] == 1:
                im = im[0]
            # ensure float [0,1]
            if not torch.is_floating_point(im):
                im = im.float()
                if im.max() > 1.5:
                    im = im / 255.0
            norm.append(im)

        # Verify squareness and uniform size; if not, center-crop to the min size
        sizes = [min(im.shape[0], im.shape[1]) for im in norm]
        target = min(sizes)
        cropped = []
        for im in norm:
            h, w, _ = im.shape
            if h != w or h != target:
                # center-crop to target
                y0 = (h - target) // 2
                x0 = (w - target) // 2
                imc = im[y0:y0 + target, x0:x0 + target, :]
            else:
                imc = im
            cropped.append(imc)

        # Compose into big square of size target*2
        big = torch.zeros((target * 2, target * 2, cropped[0].shape[2]), dtype=cropped[0].dtype, device=cropped[0].device)
        # top-left
        big[0:target, 0:target, :] = cropped[0]
        # top-right
        big[0:target, target:target * 2, :] = cropped[1]
        # bottom-left
        big[target:target * 2, 0:target, :] = cropped[2]
        # bottom-right
        big[target:target * 2, target:target * 2, :] = cropped[3]

        return ([big.unsqueeze(0)],)

# Register VNCCS RMBG2
NODE_CLASS_MAPPINGS["VNCCS_RMBG2"] = VNCCS_RMBG2
NODE_DISPLAY_NAME_MAPPINGS["VNCCS_RMBG2"] = "VNCCS RMBG2"
NODE_CATEGORY_MAPPINGS["VNCCS_RMBG2"] = "VNCCS"

# Register the VNCCS Mask Extractor
NODE_CLASS_MAPPINGS["VNCCS_MaskExtractor"] = VNCCS_MaskExtractor
NODE_DISPLAY_NAME_MAPPINGS["VNCCS_MaskExtractor"] = "VNCCS Mask Extractor"
NODE_CATEGORY_MAPPINGS["VNCCS_MaskExtractor"] = "VNCCS"

# Register additional VNCCS utility nodes
NODE_CLASS_MAPPINGS["VNCCS_Resize"] = VNCCS_Resize
NODE_DISPLAY_NAME_MAPPINGS["VNCCS_Resize"] = "VNCCS Resize"
NODE_CATEGORY_MAPPINGS["VNCCS_Resize"] = "VNCCS"

NODE_CLASS_MAPPINGS["VNCCS_ColorFix"] = VNCCS_ColorFix
NODE_DISPLAY_NAME_MAPPINGS["VNCCS_ColorFix"] = "VNCCS Color Fix"
NODE_CATEGORY_MAPPINGS["VNCCS_ColorFix"] = "VNCCS"

# Register VNCCS Quad Splitter
NODE_CLASS_MAPPINGS["VNCCS_QuadSplitter"] = VNCCS_QuadSplitter
NODE_DISPLAY_NAME_MAPPINGS["VNCCS_QuadSplitter"] = "VNCCS Quad Splitter"
NODE_CATEGORY_MAPPINGS["VNCCS_QuadSplitter"] = "VNCCS"
