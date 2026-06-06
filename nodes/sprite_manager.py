"""VNCCS Sprite Manager - Node with widget for sprite generation."""

import os
import json
import io
import torch
import numpy as np
import cv2
from PIL import Image

from ..utils import (
    base_output_dir, character_dir, list_characters,
    load_character_info, list_costumes,
    is_absolute_path_any_os, is_path_under, normalize_filesystem_path,
)
from ._safe_utils import ensure_safe_name, safe_join_under, safe_relative_path

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
        """Return current sprites for the selected character. Runtime no longer reads Sheets."""
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

        print(f"[SpriteManager] Listing current sprites for character: {character}")

        all_sprites = []
        all_paths = []
        all_masks = []

        sprites_root = os.path.join(character_dir(character), "Sprites")
        if not os.path.isdir(sprites_root):
            print(f"[SpriteManager] Sprites directory not found: {sprites_root}. Run migration or generate sprites first.")
            return [], [], []

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = []
        for root, _dirs, filenames in os.walk(sprites_root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in image_exts:
                    paths.append(os.path.join(root, filename))

        for sprite_path in sorted(paths):
            try:
                pil_img = Image.open(sprite_path).convert("RGBA")
                arr = np.array(pil_img).astype(np.float32) / 255.0
                all_sprites.append(torch.from_numpy(arr).unsqueeze(0))
                all_paths.append(sprite_path)
                all_masks.append(torch.from_numpy(1.0 - arr[..., 3]).unsqueeze(0))
            except Exception as e:
                print(f"[SpriteManager] Failed to load sprite '{sprite_path}': {e}")

        print(f"[SpriteManager] Total sprites listed: {len(all_sprites)}")

        if not all_sprites:
            print("[SpriteManager] No sprites found. Run migration or generate sprites first.")
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
    from PIL import Image

    @server.PromptServer.instance.routes.get("/vnccs/get_sheet_preview")
    async def get_sheet_preview(request):
        """Backward-compatible route: get preview image from Sprites only."""
        character = request.rel_url.query.get("character", "")
        costume = request.rel_url.query.get("costume", "Naked")
        emotion = request.rel_url.query.get("emotion", "neutral")

        if not character:
            return web.Response(status=404, text="No character specified")

        try:
            sprite_dir_path = os.path.join(character_dir(character), "Sprites", costume, emotion)
            if not os.path.isdir(sprite_dir_path):
                return web.Response(status=404, text="Sprite folder not found. Run migration or generate sprites first.")
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            files = [
                os.path.join(sprite_dir_path, filename)
                for filename in os.listdir(sprite_dir_path)
                if os.path.splitext(filename)[1].lower() in image_exts
            ]
            if not files:
                return web.Response(status=404, text="No sprite images found")

            best_file = max(files, key=lambda path: (os.path.getmtime(path), path))
            img_byte_arr = io.BytesIO()
            Image.open(best_file).save(img_byte_arr, format='PNG')
            return web.Response(body=img_byte_arr.getvalue(), content_type='image/png')

        except Exception as e:
            print(f"[SpriteManager] Error in get_sheet_preview sprite fallback: {e}")
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
        """Get costumes that have sprites for the specified emotion."""
        character = request.rel_url.query.get("character", "")
        emotion = request.rel_url.query.get("emotion", "neutral")
        
        if not character:
            return web.json_response([])

        try:
            char_path = character_dir(character)
            sprites_path = os.path.join(char_path, "Sprites")
            
            if not os.path.exists(sprites_path):
                return web.json_response([])

            costumes_with_emotion = []
            
            for costume in os.listdir(sprites_path):
                costume_path = os.path.join(sprites_path, costume)
                if not os.path.isdir(costume_path):
                    continue
                
                emotion_path = os.path.join(costume_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue
                
                files = [f for f in os.listdir(emotion_path) 
                        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
                if len(files) > 0:
                    costumes_with_emotion.append(costume)
            
            return web.json_response(sorted(costumes_with_emotion))
        except Exception as e:
            print(f"[SpriteManager] Error getting costumes by emotion: {e}")
            return web.json_response([])

    @server.PromptServer.instance.routes.get("/vnccs/get_character_emotions")
    async def get_character_emotions(request):
        """Get emotions that actually exist for a character (have sprite files)."""
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response([])

        try:
            char_path = character_dir(character)
            sprites_path = os.path.join(char_path, "Sprites")
            
            if not os.path.exists(sprites_path):
                return web.json_response(["neutral"])

            found_emotions = set()
            
            for costume in os.listdir(sprites_path):
                costume_path = os.path.join(sprites_path, costume)
                if not os.path.isdir(costume_path):
                    continue
                    
                for emotion in os.listdir(costume_path):
                    emotion_path = os.path.join(costume_path, emotion)
                    if not os.path.isdir(emotion_path):
                        continue
                    
                    files = [f for f in os.listdir(emotion_path) 
                            if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
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
            character = ensure_safe_name(character, "character")
            char_path = character_dir(character)
            empty_folders = []

            # Runtime cleanup only scans current asset folders.
            for main_dir in ["Faces", "Sprites"]:
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
            character = data.get("character", "")
            
            deleted = []
            errors = []
            base_root = base_output_dir()
            char_root = None
            if character:
                try:
                    char_root = character_dir(ensure_safe_name(character, "character"))
                except ValueError as e:
                    return web.json_response({"error": str(e)}, status=400)
            
            for folder_path in folders_to_delete:
                try:
                    raw_path = normalize_filesystem_path(folder_path)
                    if not raw_path:
                        continue

                    if is_absolute_path_any_os(raw_path):
                        resolved_path = os.path.abspath(raw_path)
                        if not is_path_under(base_root, resolved_path):
                            errors.append(f"Folder outside VNCCS output: {raw_path}")
                            continue
                    else:
                        rel_path = safe_relative_path(raw_path, "folder")
                        root = char_root or base_root
                        resolved_path = safe_join_under(root, rel_path)

                    if os.path.exists(resolved_path) and os.path.isdir(resolved_path):
                        # Safety check: only delete if empty or contains only empty subdirs
                        files = [f for f in os.listdir(resolved_path) 
                                if os.path.isfile(os.path.join(resolved_path, f))]
                        if len(files) == 0:
                            import shutil
                            shutil.rmtree(resolved_path)
                            deleted.append(os.path.relpath(resolved_path, base_root))
                            
                            # Try to remove parent if it's now empty
                            parent = os.path.dirname(resolved_path)
                            if (
                                is_path_under(base_root, parent)
                                and os.path.exists(parent)
                                and len(os.listdir(parent)) == 0
                            ):
                                os.rmdir(parent)
                        else:
                            errors.append(f"Folder not empty: {raw_path}")
                except Exception as e:
                    errors.append(f"Failed to delete {folder_path}: {str(e)}")
            
            return web.json_response({
                "deleted": deleted,
                "deleted_count": len(deleted),
                "errors": errors
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
