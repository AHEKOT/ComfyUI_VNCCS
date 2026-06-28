import os
import torch
import numpy as np
from PIL import Image

from ..utils import (
    base_output_dir, character_dir, list_characters,
    load_character_info
)


class SpriteGenerator:
    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"
        return {
            "required": {
                "character": (characters or ["None"], {"default": default_character}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("images", "file_paths", "masks")
    OUTPUT_IS_LIST = (True, True, True)
    CATEGORY = "VNCCS"
    FUNCTION = "generate_sprites"

    def load_character_config(self, character):
        return load_character_info(character)

    def generate_sprites(self, character):
        character_path = character_dir(character)
        sprites_root = os.path.join(character_path, "Sprites")
        if not os.path.exists(sprites_root):
            print(f"Sprites folder not found: {sprites_root}. Run migration or generate sprites first.")
            return [], [], []

        images_out = []
        file_paths_out = []
        masks_out = []
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        for root, _dirs, filenames in os.walk(sprites_root):
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() not in image_exts:
                    continue
                sprite_path = os.path.join(root, filename)
                try:
                    img = Image.open(sprite_path).convert("RGBA")
                    arr = torch.from_numpy(np.array(img).astype("float32") / 255.0).unsqueeze(0)
                    images_out.append(arr)
                    file_paths_out.append(sprite_path)
                    masks_out.append(1.0 - arr[..., 3])
                except Exception as exc:
                    print(f"Failed to load sprite {sprite_path}: {exc}")

        print(f"Total processed images: {len(images_out)}")
        if not images_out:
            return [], [], []

        return images_out, file_paths_out, masks_out


NODE_CLASS_MAPPINGS = {
    "SpriteGenerator": SpriteGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteGenerator": "VNCCS Sprite Generator"
}

NODE_CATEGORY_MAPPINGS = {
    "SpriteGenerator": "VNCCS"
}
