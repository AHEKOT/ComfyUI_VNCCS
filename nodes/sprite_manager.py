"""VNCCS Sprite Manager - Node with widget for sprite generation."""

import os
import json
import torch
import numpy as np
import cv2
from PIL import Image

from ..utils import (
    base_output_dir, character_dir, list_characters,
    load_character_info, load_character_sheet, list_costumes
)

# --- ComfyUI Server Imports ---
try:
    import server
    from aiohttp import web
except ImportError:
    print("[SpriteManager] Warning: Running outside ComfyUI. API routes will not be registered.")
    server = None
    web = None


class SpriteManager:
    """VNCCS Sprite Manager Node with visual widget."""

    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"
        return {
            "required": {},
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}),
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("sprites", "file_paths", "masks")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_NODE = True
    CATEGORY = "VNCCS"
    FUNCTION = "create_sprites"

    @staticmethod
    def crop_characters_from_sheet(image: torch.Tensor, mask: torch.Tensor, min_size: int = 128):
        """Crop individual characters from a sheet using contour detection.
        
        Based on CharacterSheetCropper logic.
        Returns list of (image_tensor, mask_tensor) tuples.
        """
        results = []
        batch_size = image.shape[0]

        for i in range(batch_size):
            img_item = image[i]
            mask_item = mask[i]

            img_np = img_item.cpu().numpy()

            mask_np_raw = mask_item.cpu().numpy()
            current_mask_np = None
            if mask_np_raw.ndim == 3 and mask_np_raw.shape[0] == 1:
                current_mask_np = np.squeeze(mask_np_raw, axis=0)
            elif mask_np_raw.ndim == 2:
                current_mask_np = mask_np_raw
            else:
                print(f"[SpriteManager] Warning: Mask has unexpected shape {mask_np_raw.shape}. Skipping.")
                continue

            mask_uint8 = (current_mask_np * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print(f"[SpriteManager] No contours found in mask.")
                continue

            for contour in contours:
                char_x, char_y, char_w, char_h = cv2.boundingRect(contour)

                if char_w <= 0 or char_h <= 0:
                    continue
                if char_w < min_size or char_h < min_size:
                    continue

                crop_x_start = char_x
                crop_y_start = char_y
                crop_x_end = char_x + char_w
                crop_y_end = char_y + char_h

                cropped_img_np = img_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]
                cropped_mask_np = current_mask_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                if cropped_img_np.shape[0] == 0 or cropped_img_np.shape[1] == 0:
                    continue

                rgb_part = cropped_img_np[..., :3]
                if img_np.shape[2] == 4:
                    alpha_part = cropped_img_np[..., 3:4]
                else:
                    alpha_part = cropped_mask_np[..., np.newaxis]

                final_cropped_image_np = np.concatenate((rgb_part, alpha_part), axis=-1)

                img_out_tensor = torch.from_numpy(final_cropped_image_np.astype(np.float32)).unsqueeze(0)
                mask_out_tensor = torch.from_numpy(cropped_mask_np.astype(np.float32)).unsqueeze(0)
                
                results.append((img_out_tensor, mask_out_tensor))

        return results

    def create_sprites(self, widget_data="{}", unique_id=None):
        """Main process function for sprite generation.
        
        Creates sprites for ALL costumes and ALL emotions that have sheets.
        The widget_data is only used for character selection.
        """
        try:
            if isinstance(widget_data, str):
                data = json.loads(widget_data)
            else:
                data = widget_data
        except:
            data = {}

        character = data.get("character", "")

        if not character:
            print("[SpriteManager] No character specified.")
            return [], [], []

        print(f"[SpriteManager] Processing ALL sprites for character: {character}")

        all_sprites = []
        all_paths = []
        all_masks = []
        
        # First pass: collect all crops with metadata
        pending_crops = []  # List of (cropped_img, cropped_mask, sprite_path, costume, emotion)

        # Scan Sheets directory to find all costume/emotion combinations
        sheets_path = os.path.join(character_dir(character), "Sheets")
        if not os.path.exists(sheets_path):
            print(f"[SpriteManager] Sheets directory not found: {sheets_path}")
            return [], [], []

        # Iterate through all costumes
        for costume in sorted(os.listdir(sheets_path)):
            costume_path = os.path.join(sheets_path, costume)
            if not os.path.isdir(costume_path):
                continue

            # Iterate through all emotions in this costume
            for emotion in sorted(os.listdir(costume_path)):
                emotion_path = os.path.join(costume_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue

                # Check if has sheet files
                sheet_files = [f for f in os.listdir(emotion_path) 
                              if f.endswith('.png') and f.startswith('sheet_')]
                if not sheet_files:
                    continue

                print(f"[SpriteManager] Processing: {costume}/{emotion}")

                # Load sheet with mask
                result = load_character_sheet(character, costume, emotion, with_mask=True)
                if result is None or result[0] is None:
                    print(f"[SpriteManager] Failed to load sheet for {character}/{costume}/{emotion}")
                    continue

                img_tensor, mask_tensor = result

                if mask_tensor is None:
                    # Create default mask from alpha or full opaque
                    if img_tensor.shape[-1] == 4:
                        mask_tensor = img_tensor[..., 3]
                    else:
                        mask_tensor = torch.ones((img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]))

                # Crop characters from sheet
                crops = self.crop_characters_from_sheet(img_tensor, mask_tensor, min_size=128)

                if not crops:
                    print(f"[SpriteManager] No characters found in sheet for {costume}/{emotion}")
                    continue

                # Prepare output directory
                sprite_base_dir = os.path.join(character_dir(character), "Sprites", costume, emotion)
                os.makedirs(sprite_base_dir, exist_ok=True)

                for idx, (cropped_img, cropped_mask) in enumerate(crops):
                    sprite_filename = f"sprite_{emotion}_{idx:04d}.png"
                    sprite_path = os.path.join(sprite_base_dir, sprite_filename)
                    pending_crops.append((cropped_img, cropped_mask, sprite_path, costume, emotion))

        if not pending_crops:
            print("[SpriteManager] No sprites to process")
            return [], [], []

        # Analyze heights for normalization
        heights = []
        MIN_HEIGHT_THRESHOLD = 200  # Ignore sprites smaller than this (likely broken)
        
        for cropped_img, _, _, _, _ in pending_crops:
            h = cropped_img.shape[1]  # Shape is (batch, H, W, C)
            if h >= MIN_HEIGHT_THRESHOLD:
                heights.append(h)
        
        if not heights:
            print("[SpriteManager] No valid sprites found (all below height threshold)")
            return [], [], []
        
        # Use median height (more robust than min) but with safety
        heights.sort()
        # Use 10th percentile to avoid outliers but not the absolute minimum
        target_idx = max(0, len(heights) // 10)
        target_height = heights[target_idx]
        
        print(f"[SpriteManager] Normalizing all sprites to height: {target_height}")

        # Second pass: resize and save
        for cropped_img, cropped_mask, sprite_path, costume, emotion in pending_crops:
            try:
                img_np = (cropped_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode="RGBA")
                
                # Skip if below threshold (broken crop)
                if pil_img.height < MIN_HEIGHT_THRESHOLD:
                    print(f"[SpriteManager] Skipping broken sprite (height {pil_img.height}): {sprite_path}")
                    continue
                
                # Resize to target height maintaining aspect ratio
                if pil_img.height != target_height:
                    ratio = target_height / pil_img.height
                    new_width = int(pil_img.width * ratio)
                    pil_img = pil_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                
                pil_img.save(sprite_path)
                print(f"[SpriteManager] Saved: {sprite_path} ({pil_img.width}x{pil_img.height})")
                
                # Convert back to tensor for output
                resized_np = np.array(pil_img).astype(np.float32) / 255.0
                resized_tensor = torch.from_numpy(resized_np).unsqueeze(0)
                
                # Create resized mask
                mask_np = (cropped_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np, mode="L")
                if mask_pil.height != target_height:
                    ratio = target_height / mask_pil.height
                    new_width = int(mask_pil.width * ratio)
                    mask_pil = mask_pil.resize((new_width, target_height), Image.Resampling.LANCZOS)
                mask_resized_np = np.array(mask_pil).astype(np.float32) / 255.0
                mask_resized_tensor = torch.from_numpy(mask_resized_np).unsqueeze(0)
                
                all_sprites.append(resized_tensor)
                all_paths.append(sprite_path)
                all_masks.append(mask_resized_tensor)
                
            except Exception as e:
                print(f"[SpriteManager] Failed to save sprite: {e}")
                continue

        print(f"[SpriteManager] Total sprites created: {len(all_sprites)}")

        if not all_sprites:
            return [], [], []

        return all_sprites, all_paths, all_masks


NODE_CLASS_MAPPINGS = {
    "SpriteManager": SpriteManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteManager": "VNCCS Sprite Manager"
}

NODE_CATEGORY_MAPPINGS = {
    "SpriteManager": "VNCCS"
}

# --- API Endpoints ---
if server:
    import glob
    import re
    from PIL import Image
    import io

    @server.PromptServer.instance.routes.get("/vnccs/get_sheet_preview")
    async def get_sheet_preview(request):
        """Get preview image for a specific character/costume/emotion sheet. No fallback."""
        character = request.rel_url.query.get("character", "")
        costume = request.rel_url.query.get("costume", "Naked")
        emotion = request.rel_url.query.get("emotion", "neutral")

        if not character:
            return web.Response(status=404, text="No character specified")

        try:
            sheet_dir_path = os.path.join(character_dir(character), "Sheets", costume, emotion)

            if not os.path.exists(sheet_dir_path):
                return web.Response(status=404, text="Sheet not found")

            # Find best file (highest index)
            pattern = os.path.join(sheet_dir_path, f"sheet_{emotion}_*.png")
            files = glob.glob(pattern)
            
            # Also try sheet_*.png pattern
            if not files:
                pattern = os.path.join(sheet_dir_path, "sheet_*.png")
                files = glob.glob(pattern)

            if not files:
                return web.Response(status=404, text="No sheet images found")

            def get_index(f):
                m = re.search(r'(\d+)', os.path.basename(f))
                return int(m.group(1)) if m else 0

            files.sort(key=get_index)
            best_file = files[-1]

            # Load and crop (same as get_character_sheet_preview)
            img = Image.open(best_file)
            w, h = img.size

            # Layout: 2 Rows, 6 Columns. Get Index 11 (Row 1, Col 5)
            item_w = w // 6
            item_h = h // 2
            left = 5 * item_w
            upper = 1 * item_h
            right = left + item_w
            lower = upper + item_h

            crop = img.crop((left, upper, right, lower))

            img_byte_arr = io.BytesIO()
            crop.save(img_byte_arr, format='PNG')
            return web.Response(body=img_byte_arr.getvalue(), content_type='image/png')

        except Exception as e:
            print(f"[SpriteManager] Error in get_sheet_preview: {e}")
            return web.Response(status=500, text=str(e))

    @server.PromptServer.instance.routes.get("/vnccs/list_characters")
    async def list_characters_api(request):
        """Get list of all characters."""
        try:
            characters = list_characters()
            return web.json_response(characters)
        except Exception as e:
            return web.Response(status=500, text=str(e))

    @server.PromptServer.instance.routes.get("/vnccs/get_costumes_by_emotion")
    async def get_costumes_by_emotion(request):
        """Get costumes that have sheets for the specified emotion."""
        character = request.rel_url.query.get("character", "")
        emotion = request.rel_url.query.get("emotion", "neutral")
        
        if not character:
            return web.json_response([])

        try:
            char_path = character_dir(character)
            sheets_path = os.path.join(char_path, "Sheets")
            
            if not os.path.exists(sheets_path):
                return web.json_response([])

            costumes_with_emotion = []
            
            for costume in os.listdir(sheets_path):
                costume_path = os.path.join(sheets_path, costume)
                if not os.path.isdir(costume_path):
                    continue
                
                emotion_path = os.path.join(costume_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue
                
                # Check if has any sheet files
                files = [f for f in os.listdir(emotion_path) 
                        if f.endswith('.png') and f.startswith('sheet_')]
                if len(files) > 0:
                    costumes_with_emotion.append(costume)
            
            return web.json_response(sorted(costumes_with_emotion))
        except Exception as e:
            print(f"[SpriteManager] Error getting costumes by emotion: {e}")
            return web.json_response([])

    @server.PromptServer.instance.routes.get("/vnccs/get_character_emotions")
    async def get_character_emotions(request):
        """Get emotions that actually exist for a character (have sheet files)."""
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response([])

        try:
            char_path = character_dir(character)
            sheets_path = os.path.join(char_path, "Sheets")
            
            if not os.path.exists(sheets_path):
                return web.json_response(["neutral"])

            found_emotions = set()
            
            # Scan all costumes for emotions with actual files
            for costume in os.listdir(sheets_path):
                costume_path = os.path.join(sheets_path, costume)
                if not os.path.isdir(costume_path):
                    continue
                    
                for emotion in os.listdir(costume_path):
                    emotion_path = os.path.join(costume_path, emotion)
                    if not os.path.isdir(emotion_path):
                        continue
                    
                    # Check if has any sheet files
                    files = [f for f in os.listdir(emotion_path) 
                            if f.endswith('.png') and f.startswith('sheet_')]
                    if len(files) > 0:
                        found_emotions.add(emotion)
            
            if not found_emotions:
                return web.json_response(["neutral"])
            
            # Sort for consistent order
            return web.json_response(sorted(list(found_emotions)))
        except Exception as e:
            print(f"[SpriteManager] Error getting character emotions: {e}")
            return web.json_response(["neutral"])

    @server.PromptServer.instance.routes.get("/vnccs/find_empty_folders")
    async def find_empty_folders(request):
        """Find empty costume and emotion folders for a character."""
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response({"error": "No character specified"}, status=400)

        try:
            char_path = character_dir(character)
            empty_folders = []

            # Check Sheets, Faces, Sprites directories
            for main_dir in ["Sheets", "Faces", "Sprites"]:
                main_path = os.path.join(char_path, main_dir)
                if not os.path.exists(main_path):
                    continue

                for costume in os.listdir(main_path):
                    costume_path = os.path.join(main_path, costume)
                    if not os.path.isdir(costume_path):
                        continue

                    # Check emotion subdirectories
                    has_content = False
                    for emotion in os.listdir(costume_path):
                        emotion_path = os.path.join(costume_path, emotion)
                        if not os.path.isdir(emotion_path):
                            has_content = True
                            continue

                        # Check if emotion folder is empty (no files)
                        files_in_emotion = [f for f in os.listdir(emotion_path) 
                                          if os.path.isfile(os.path.join(emotion_path, f))]
                        if len(files_in_emotion) == 0:
                            rel_path = os.path.join(main_dir, costume, emotion)
                            empty_folders.append({
                                "path": rel_path,
                                "full_path": emotion_path,
                                "type": "emotion",
                                "reason": "No images in folder"
                            })
                        else:
                            has_content = True

                    # Check if costume folder has no content after removing empty emotions
                    remaining_items = [f for f in os.listdir(costume_path) 
                                      if os.path.isdir(os.path.join(costume_path, f))]
                    non_empty_emotions = []
                    for em in remaining_items:
                        em_path = os.path.join(costume_path, em)
                        files = [f for f in os.listdir(em_path) if os.path.isfile(os.path.join(em_path, f))]
                        if len(files) > 0:
                            non_empty_emotions.append(em)
                    
                    if len(non_empty_emotions) == 0 and len(remaining_items) > 0:
                        # All emotions are empty -> costume is effectively empty
                        pass  # Individual emotions already added

            return web.json_response({
                "character": character,
                "empty_folders": empty_folders,
                "count": len(empty_folders)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @server.PromptServer.instance.routes.post("/vnccs/delete_empty_folders")
    async def delete_empty_folders(request):
        """Delete specified empty folders."""
        try:
            data = await request.json()
            folders_to_delete = data.get("folders", [])
            
            deleted = []
            errors = []
            
            for folder_path in folders_to_delete:
                try:
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        # Safety check: only delete if empty or contains only empty subdirs
                        files = [f for f in os.listdir(folder_path) 
                                if os.path.isfile(os.path.join(folder_path, f))]
                        if len(files) == 0:
                            import shutil
                            shutil.rmtree(folder_path)
                            deleted.append(folder_path)
                            
                            # Try to remove parent if it's now empty
                            parent = os.path.dirname(folder_path)
                            if os.path.exists(parent) and len(os.listdir(parent)) == 0:
                                os.rmdir(parent)
                        else:
                            errors.append(f"Folder not empty: {folder_path}")
                except Exception as e:
                    errors.append(f"Failed to delete {folder_path}: {str(e)}")
            
            return web.json_response({
                "deleted": deleted,
                "deleted_count": len(deleted),
                "errors": errors
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

