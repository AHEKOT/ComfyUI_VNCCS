import torch
import numpy as np
from typing import List, Tuple

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

    def compose_sheet(self, image_tensors: torch.Tensor, target_height: int) -> torch.Tensor:
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
        
        # Calculate target cell size for square sheet
        # For 2x6 grid to fit in target_height x target_height square:
        # cell_height = target_height // 2
        # cell_width = target_height // 6
        cell_height = target_height // 2
        cell_width = target_height // 6
        
        # Ensure dimensions are divisible by 8
        cell_height = max(1, (cell_height // 8) * 8)
        cell_width = max(1, (cell_width // 8) * 8)
        
        # Final sheet dimensions (should be target_height x target_height)
        sheet_height = num_rows * cell_height
        sheet_width = num_columns * cell_width
        
        print(f"Cell size: {cell_height}x{cell_width}")
        print(f"Final sheet dimensions: {sheet_height}x{sheet_width}")
        
        # Create the final sheet
        sheet = torch.zeros((sheet_height, sheet_width, channels), dtype=image_tensors.dtype, device=image_tensors.device)
        
        for idx, image in enumerate(image_tensors):
            # Resize image to cell size
            resized_image = torch.nn.functional.interpolate(
                image.unsqueeze(0).permute(0, 3, 1, 2), 
                size=(cell_height, cell_width), 
                mode="bilinear"
            ).squeeze().permute(1, 2, 0)
            
            # Calculate position in grid
            row = idx // num_columns
            col = idx % num_columns
            
            # Place image in sheet
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width
            
            sheet[y_start:y_end, x_start:x_end, :] = resized_image
        
        return sheet.unsqueeze(0)

    def process_sheet(self, mode, images, target_width, target_height):
        """Main processing function."""
        # Handle list inputs since INPUT_IS_LIST = True
        mode = mode[0] if isinstance(mode, list) else mode
        target_width = target_width[0] if isinstance(target_width, list) else target_width
        target_height = target_height[0] if isinstance(target_height, list) else target_height
        
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
            result = self.compose_sheet(images, target_height)
            # Convert single image to list
            result_list = [result]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return (result_list,)


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

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
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
        """Main chroma key function."""
        # Handle batch dimension
        if len(image.shape) == 4:  # [B, H, W, C]
            batch_size = image.shape[0]
            results = []
            masks = []
            for i in range(batch_size):
                img, msk = self.chroma_key_single(image[i], tolerance, despill_strength, despill_kernel_size, despill_color)
                results.append(img)
                masks.append(msk)
            return (torch.stack(results), torch.stack(masks))
        else:  # Single image [H, W, C]
            img, msk = self.chroma_key_single(image, tolerance, despill_strength, despill_kernel_size, despill_color)
            return (img.unsqueeze(0), msk.unsqueeze(0))

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
    "VNCCSChromaKey": VNCCSChromaKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS Sheet Manager",
    "VNCCSChromaKey": "VNCCS Chroma Key"
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSSheetManager": "VNCCS",
    "VNCCSChromaKey": "VNCCS"
}
