"""VNCCS Sheet Manager - Sheet splitting and composing nodes.

Contains only sheet-related nodes:
- VNCCSSheetManager - Split sheets into parts or compose images into sheets
- VNCCSSheetExtractor - Extract a single part from a sheet
- VNCCS_QuadSplitter - Split/compose 2x2 grid sheets
"""

import os
import torch
import numpy as np
from typing import List
from PIL import Image


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
        height, width, channels = image_tensor.shape
        
        part_width = width // 6
        part_height = height // 2
        
        parts = []
        
        for row in range(2):
            for col in range(6):
                y_start = row * part_height
                y_end = (row + 1) * part_height
                x_start = col * part_width
                x_end = (col + 1) * part_width
                
                part = image_tensor[y_start:y_end, x_start:x_end, :]
                
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
        num_rows = 2
        num_columns = 6
        expected_batch_size = num_rows * num_columns
        
        batch_size, img_height, img_width, channels = image_tensors.shape
        
        if batch_size < expected_batch_size:
            padding_needed = expected_batch_size - batch_size
            padding = torch.zeros((padding_needed, img_height, img_width, channels), 
                                dtype=image_tensors.dtype, device=image_tensors.device)
            image_tensors = torch.cat([image_tensors, padding], dim=0)
            batch_size = expected_batch_size
        elif batch_size > expected_batch_size:
            image_tensors = image_tensors[:expected_batch_size]
            batch_size = expected_batch_size
        
        margin = 4 if safe_margin else 0

        cell_height = target_height // 2
        cell_width = target_height // 6

        cell_height = max(1, (cell_height // 8) * 8)
        cell_width = max(1, (cell_width // 8) * 8)

        sheet_height = num_rows * cell_height
        sheet_width = num_columns * cell_width

        sheet = torch.zeros((sheet_height, sheet_width, channels), dtype=image_tensors.dtype, device=image_tensors.device)

        if safe_margin:
            if channels >= 3:
                sheet[:, :, 0] = 0.0
                sheet[:, :, 1] = 1.0
                sheet[:, :, 2] = 0.0
            if channels == 4:
                sheet[:, :, 3] = 1.0
        
        for idx, image in enumerate(image_tensors):
            target_inner_height = max(1, cell_height - 2 * margin)
            target_inner_width = max(1, cell_width - 2 * margin)

            resized_image = torch.nn.functional.interpolate(
                image.unsqueeze(0).permute(0, 3, 1, 2), 
                size=(target_inner_height, target_inner_width), 
                mode="bilinear"
            ).squeeze().permute(1, 2, 0)
            
            row = idx // num_columns
            col = idx % num_columns
            
            cell_origin_y = row * cell_height
            cell_origin_x = col * cell_width

            y_start = cell_origin_y + margin
            y_end = y_start + resized_image.shape[0]
            x_start = cell_origin_x + margin
            x_end = x_start + resized_image.shape[1]
            
            sheet[y_start:y_end, x_start:x_end, :] = resized_image
        
        return sheet.unsqueeze(0)

    def process_sheet(self, mode, images, target_width, target_height, safe_margin=False):
        """Main processing function."""
        mode = mode[0] if isinstance(mode, list) else mode
        target_width = target_width[0] if isinstance(target_width, list) else target_width
        target_height = target_height[0] if isinstance(target_height, list) else target_height
        safe_margin = safe_margin[0] if isinstance(safe_margin, list) else safe_margin
        
        if mode == "split":
            if isinstance(images, list):
                images = images[0]
            if len(images.shape) == 4:
                images = images[0]
            parts = self.split_sheet(images, target_width, target_height)
            result_list = [part.unsqueeze(0) for part in parts]
        elif mode == "compose":
            if isinstance(images, list):
                images = torch.cat(images, dim=0)
            elif len(images.shape) == 3:
                images = images.unsqueeze(0)
            result = self.compose_sheet(images, target_height, safe_margin)
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
                "part_index": ("INT", {"default": 1, "min": 1, "max": 12}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 6144, "step": 64}),
                "target_height": ("INT", {"default": 3072, "min": 64, "max": 6144, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "VNCCS"
    FUNCTION = "extract"
    DESCRIPTION = """
    Returns one part of the sheet (2×6 grid) at the given index. Indices 1-6 are the top row left-to-right, 7-12 are the bottom row.
    """

    def extract(self, image, part_index, target_width, target_height):
        """Extract a single part from a 2x6 sheet grid without splitting the entire sheet."""
        part_index = part_index[0] if isinstance(part_index, list) else part_index
        target_width = target_width[0] if isinstance(target_width, list) else target_width
        target_height = target_height[0] if isinstance(target_height, list) else target_height

        if isinstance(image, list):
            image = image[0]
        if len(image.shape) == 4:
            image = image[0]

        # Convert from human-friendly 1-12 to internal 0-11 and clamp
        part_index = max(0, min(11, part_index - 1))

        height, width, channels = image.shape
        cell_width = width // 6
        cell_height = height // 2

        row = part_index // 6
        col = part_index % 6

        y_start = row * cell_height
        y_end = (row + 1) * cell_height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width

        part = image[y_start:y_end, x_start:x_end, :]

        # Resize if needed
        if part.shape[0] != target_height or part.shape[1] != target_width:
            part = torch.nn.functional.interpolate(
                part.unsqueeze(0).permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode="bilinear"
            ).squeeze().permute(1, 2, 0)

        return (part.unsqueeze(0),)


class VNCCS_QuadSplitter:
    """Split a square character sheet into 4 equal square quadrants (2x2) and return them as a list."""

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
        if isinstance(image, list):
            image = image[0]
        if len(image.shape) == 4:
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image[0]
        return image

    def _normalize_image_list(self, images):
        """Flatten incoming images into a list of HWC tensors.
        
        Handles: Tensor (B,H,W,C), List[Tensor], List[List[Tensor]], mixed nesting.
        """
        out = []

        def _add_tensor(t):
            if len(t.shape) == 4:
                for i in range(t.shape[0]):
                    out.append(t[i])
            elif len(t.shape) == 3:
                out.append(t)

        def _process(x):
            if isinstance(x, torch.Tensor):
                _add_tensor(x)
            elif isinstance(x, list):
                for item in x:
                    _process(item)

        _process(images)
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
        if not torch.is_floating_point(img):
            img = img.float()
            if img.max() > 1.5:
                img = img / 255.0

        img = self._center_crop_square(img)
        size, _, _ = img.shape
        half = size // 2

        q1 = img[0:half, 0:half, :]
        q2 = img[0:half, half:half * 2, :]
        q3 = img[half:half * 2, 0:half, :]
        q4 = img[half:half * 2, half:half * 2, :]

        quads = [q1, q2, q3, q4]

        return ([q.unsqueeze(0) for q in quads],)

    def process(self, mode, image):
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
        """Compose 4 square images into a single square sheet (2x2 grid)."""
        imgs = images
        if isinstance(imgs, list):
            imgs = imgs
        elif isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4:
            if imgs.shape[0] == 4:
                imgs = [imgs[i] for i in range(4)]
            elif imgs.shape[0] == 1:
                imgs = [imgs[0]]
            else:
                imgs = [imgs[i] for i in range(min(4, imgs.shape[0]))]

        if not isinstance(imgs, list):
            raise ValueError("Compose expects a list or batch of images")

        imgs = self._normalize_image_list(imgs)
        if len(imgs) < 4:
            raise ValueError("Compose expects at least 4 images")

        norm = []
        for im in imgs[:4]:
            if isinstance(im, list):
                im = im[0]
            if len(im.shape) == 4 and im.shape[0] == 1:
                im = im[0]
            if not torch.is_floating_point(im):
                im = im.float()
                if im.max() > 1.5:
                    im = im / 255.0
            norm.append(im)

        sizes = [min(im.shape[0], im.shape[1]) for im in norm]
        target = min(sizes)
        cropped = []
        for im in norm:
            h, w, _ = im.shape
            if h != w or h != target:
                y0 = (h - target) // 2
                x0 = (w - target) // 2
                imc = im[y0:y0 + target, x0:x0 + target, :]
            else:
                imc = im
            cropped.append(imc)

        big = torch.zeros((target * 2, target * 2, cropped[0].shape[2]), dtype=cropped[0].dtype, device=cropped[0].device)
        big[0:target, 0:target, :] = cropped[0]
        big[0:target, target:target * 2, :] = cropped[1]
        big[target:target * 2, 0:target, :] = cropped[2]
        big[target:target * 2, target:target * 2, :] = cropped[3]

        return ([big.unsqueeze(0)],)


# Node Registration
NODE_CLASS_MAPPINGS = {
    "VNCCSSheetManager": VNCCSSheetManager,
    "VNCCSSheetExtractor": VNCCSSheetExtractor,
    "VNCCS_QuadSplitter": VNCCS_QuadSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS Sheet Manager",
    "VNCCSSheetExtractor": "VNCCS Sheet Extractor",
    "VNCCS_QuadSplitter": "VNCCS Quad Splitter",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS",
    "VNCCSSheetExtractor": "VNCCS",
    "VNCCS_QuadSplitter": "VNCCS",
}
