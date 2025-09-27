import os
import torch
import numpy as np
from PIL import Image, ImageOps

try:
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, load_character_sheet
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, load_character_sheet
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
        sheets_dir = os.path.join(character_path, "Sheets")
        if not os.path.exists(sheets_dir):
            print(f"Sprites folder not found: {sheets_dir}")
            return [], [], []

        images_out = []
        file_paths_out = []
        masks_out = []
        processed_emotions = set()

        costumes = [d for d in os.listdir(sheets_dir) if os.path.isdir(os.path.join(sheets_dir, d))]
        print(f"Found costumes: {len(costumes)}")

        for costume in costumes:
            costume_dir = os.path.join(sheets_dir, costume)
            emotions = [d for d in os.listdir(costume_dir) if os.path.isdir(os.path.join(costume_dir, d))]
            print(f"For costume {costume} found emotions: {len(emotions)}")
            for emotion in emotions:
                emotion_key = f"{costume}_{emotion}"
                if emotion_key in processed_emotions:
                    continue
                processed_emotions.add(emotion_key)
                emotion_dir = os.path.join(costume_dir, emotion)
                # Use the existing load_character_sheet function
                result = load_character_sheet(character, costume, emotion, with_mask=True)
                if result is not None and result[0] is not None:
                    img_tensor, mask_tensor = result
                    images_out.append(img_tensor)
                    masks_out.append(mask_tensor)

                    sprite_dir = os.path.join(self.base_path, character, "Sprites", costume, emotion)
                    os.makedirs(sprite_dir, exist_ok=True)

                    # Generate base sprite filename without number suffix
                    sprite_filename = f"sprite_{emotion}_"
                    sprite_path = os.path.join(sprite_dir, sprite_filename)

                    for _ in range(12):
                        file_paths_out.append(sprite_path)

                    print(f"Processed emotion: {emotion} for costume: {costume}")
                    print(f"Save path: {sprite_path}")
                else:
                    print(f"Failed to load sheet for {character}/{costume}/{emotion}")

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
