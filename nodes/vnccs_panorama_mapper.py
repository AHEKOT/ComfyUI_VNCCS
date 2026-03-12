import os
import json
import torch
import numpy as np
from PIL import Image
import math
import folder_paths

# Import utility from background_generator to reuse logic
from .background_generator import equirect_to_perspective

class VNCCS_PanoramaMapper:
    """
    VNCCS Panorama Mapper Node
    Allows manual room corner marking and distortion correction.
    Outputs 4 images corresponding to the walls of the room.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "widget_data": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "right", "back", "left", "corrected_panorama")
    FUNCTION = "extract_walls"
    CATEGORY = "VNCCS/3D"

    def extract_walls(self, widget_data):
        try:
            data = json.loads(widget_data)
        except Exception as e:
            print(f"❌ [VNCCS] Failed to parse widget_data: {e}")
            empty = torch.zeros((1, 512, 512, 3))
            return (empty, empty, empty, empty, empty)

        # 1. Resolve Panorama Source
        image_path = data.get("image_path")
        if not image_path:
            print("⚠️ [VNCCS] No image path provided in widget_data")
            empty = torch.zeros((1, 512, 512, 3))
            return (empty, empty, empty, empty, empty)
            
        # ComfyUI image paths are usually relative to input dir
        if not os.path.isabs(image_path):
            image_path = os.path.join(folder_paths.get_input_directory(), image_path)

        if not os.path.exists(image_path):
            print(f"⚠️ [VNCCS] Panorama image not found at: {image_path}")
            empty = torch.zeros((1, 512, 512, 3))
            return (empty, empty, empty, empty, empty)
        
        pil_img = Image.open(image_path).convert("RGB")
        W, H = pil_img.size
        
        # 2. Extract Corners and Corrections
        corners = data.get("corners", [45, 135, 225, 315])
        # Convert degrees back to normalized 0..1 for pixel math
        corners_norm = [c / 360.0 for c in corners]
        
        output_size = data.get("output_size", 1024)
        
        # Consistent vertical strip matching the widget's simple crop
        strip_h = H // 3
        strip_y = (H - strip_h) // 2
        
        # 3. Calculate pixel widths and Average for equalization
        pixel_widths = []
        for i in range(4):
            c1 = corners_norm[i] % 1.0
            c2 = corners_norm[(i + 1) % 4] % 1.0
            if c2 >= c1:
                pw = (c2 - c1) * W
            else:
                pw = (1.0 - c1 + c2) * W
            pixel_widths.append(pw)
            
        # Target resolutions: 1&3 match, 2&4 match (AVERAGED)
        # We use output_size as the reference for HEIGHT of individual walls
        scale_factor = output_size / strip_h
        
        # Averages
        avg_13 = (pixel_widths[0] + pixel_widths[2]) / 2.0
        avg_24 = (pixel_widths[1] + pixel_widths[3]) / 2.0
        
        w13_target = int(avg_13 * scale_factor)
        w24_target = int(avg_24 * scale_factor)
        
        target_resolutions = [
            (w13_target, output_size), # Wall 1
            (w24_target, output_size), # Wall 2
            (w13_target, output_size), # Wall 3
            (w24_target, output_size)  # Wall 4
        ]
        
        # 4. Extract individual walls
        walls = []
        for i in range(4):
            c1 = corners_norm[i] % 1.0
            c2 = corners_norm[(i + 1) % 4] % 1.0
            
            x1 = int(round(c1 * W))
            x2 = int(round(c2 * W))
            
            if c2 >= c1:
                wall_pil = pil_img.crop((x1, strip_y, x2, strip_y + strip_h))
            else:
                part1 = pil_img.crop((x1, strip_y, W, strip_y + strip_h))
                part2 = pil_img.crop((0, strip_y, x2, strip_y + strip_h))
                wall_pil = Image.new("RGB", (part1.width + part2.width, strip_h))
                wall_pil.paste(part1, (0, 0))
                wall_pil.paste(part2, (part1.width, 0))
            
            # Apply Equalized Resize
            target_w, target_h = target_resolutions[i]
            wall_pil = wall_pil.resize((target_w, target_h), Image.LANCZOS)
            
            # Convert to torch
            wall_array = np.array(wall_pil).astype(np.float32) / 255.0
            wall_torch = torch.from_numpy(wall_array).unsqueeze(0)
            walls.append(wall_torch)

        # 5. Reconstruct Corrected Panorama
        # We horizontal-scale each segment to its target averaged width
        # The total width should remain exactly W
        
        # Find which segment contains the x=0 seam
        new_pano = Image.new("RGB", (W, H))
        
        # Target widths in original panorama pixel scale
        target_widths_pano = [avg_13, avg_24, avg_13, avg_24]
        
        # We need to find the "first" segment after the seam x=0
        # The corners are in a specific order but can wrap.
        # Let's find the segment that crosses 0.
        seam_idx = -1
        for i in range(4):
            c1 = corners_norm[i] % 1.0
            c2 = corners_norm[(i + 1) % 4] % 1.0
            if c2 < c1: # This segment wraps the seam
                seam_idx = i
                break
        
        if seam_idx == -1:
            # Fallback: maybe no segment wraps (all lines same side of seam)
            # Find the segment closest to 0
            seam_idx = 3 # Default to last wall
            
        # Reconstruct pieces:
        # 1. Any pieces of original image before the first corner
        # 2. Each full segment
        # 3. Any pieces after the last corner
        
        # Actually, it's simpler:
        # We know the 4 Wall segments cover the full 360.
        # Wall i: S1=[c0, c1], S2=[c1, c2], S3=[c2, c3], S4=[c3, c0]
        # One of these wraps. Let's say S4 wraps.
        # Original sequence: [0..c0] part of S4, then S1, S2, S3, then [c3..1] part of S4.
        # We scale each wall i by (avg_i / orig_i)
        
        new_pano = Image.new("RGB", (W, H))
        
        # Current accumulated X in target
        curr_target_x = 0
        
        # Order pieces starting from 0:
        # Pieces sequence:
        # - Part of Wall[seam_idx] from 0 to corners[ (seam_idx+1)%4 ]
        # - Wall[ (seam_idx+1)%4 ]
        # - Wall[ (seam_idx+2)%4 ]
        # - Wall[ (seam_idx+3)%4 ]
        # - Part of Wall[seam_idx] from corners[seam_idx] to 1.0
        
        segments_order = [ (seam_idx + j) % 4 for j in range(1, 4) ]
        
        # Widths calculation for scaling
        def get_scalef(idx):
            return target_widths_pano[idx] / pixel_widths[idx] if pixel_widths[idx] > 0 else 1.0

        # Part 1: Start to first corner (part of seam_idx wall)
        c_first = corners_norm[ (seam_idx + 1) % 4 ] % 1.0
        w_part1 = c_first * W
        target_w_part1 = int(round(w_part1 * get_scalef(seam_idx)))
        if w_part1 > 0:
            piece1 = pil_img.crop((0, 0, int(w_part1), H)).resize((target_w_part1, H), Image.LANCZOS)
            new_pano.paste(piece1, (0, 0))
        curr_target_x = target_w_part1
        
        # Part 2, 3, 4: Full walls
        for idx in segments_order:
            c1 = corners_norm[idx] % 1.0
            c2 = corners_norm[(idx + 1) % 4] % 1.0
            # Width in original
            w_orig = (c2 - c1) * W
            target_w = int(round(w_orig * get_scalef(idx)))
            if w_orig > 0:
                piece = pil_img.crop((int(c1*W), 0, int(c2*W), H)).resize((target_w, H), Image.LANCZOS)
                new_pano.paste(piece, (curr_target_x, 0))
            curr_target_x += target_w
            
        # Part 5: Last corner to end (rest of seam_idx wall)
        c_last = corners_norm[seam_idx] % 1.0
        w_part5 = (1.0 - c_last) * W
        target_w_part5 = W - curr_target_x # Take remaining to avoid rounding gaps
        if w_part5 > 0 and target_w_part5 > 0:
            piece5 = pil_img.crop((int(c_last*W), 0, W, H)).resize((target_w_part5, H), Image.LANCZOS)
            new_pano.paste(piece5, (curr_target_x, 0))
            
        # Fix rounding to exactly W
        if new_pano.size[0] != W:
            new_pano = new_pano.resize((W, H), Image.LANCZOS)

        pano_torch = torch.from_numpy(np.array(new_pano).astype(np.float32) / 255.0).unsqueeze(0)
        
        print(f"✅ [VNCCS] Walls extracted and Panorama corrected.")
        return (*walls, pano_torch)

NODE_CLASS_MAPPINGS = {
    "VNCCS_PanoramaMapper": VNCCS_PanoramaMapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PanoramaMapper": "VNCCS Panorama Mapper"
}
