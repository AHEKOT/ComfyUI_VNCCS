import os
import shutil

from ..utils import (
    base_output_dir, list_characters,
    load_character_info
)


class DatasetGenerator:
    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"

        return {
            "required": {
                "character": (characters or ["None"], {"default": default_character}),
                "game_name": ("STRING", {"default": "VN"}),
            },
            "optional": {
                "additional_caption": ("STRING", {"default": " ", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    CATEGORY = "VNCCS"
    FUNCTION = "generate_dataset"

    def load_character_config(self, character):
        """Load character configuration file"""
        return load_character_info(character)

    def build_caption_text(self, character_info, costume, emotion, image_type, formatted_character_name, additional_caption=""):
        """Build caption text similar to positive prompt logic in creator/selector nodes."""
        caption_parts = [formatted_character_name]
        
        # Get character attributes
        sex = character_info.get("sex") or character_info.get("gender", "")
        age = character_info.get("age", "")
        race = character_info.get("race", "")
        hair = character_info.get("hair", "")
        eyes = character_info.get("eyes", "")
        face = character_info.get("face", "")
        body = character_info.get("body", "")
        skin_color = character_info.get("skin_color", "")
        additional_details = character_info.get("additional_details", "")
        background_color = character_info.get("background_color", "green")
        
        # Add gender
        if sex == "male":
            caption_parts.append("1boy")
        else:
            caption_parts.append("1girl")
        
        # Add age
        if age:
            caption_parts.append(f"{age}yo")
        
        # Add image type (portrait for faces, full body for sprites)
        caption_parts.append(image_type)
        
        # Add character features
        if race:
            caption_parts.append(f"{race}")
        if hair:
            caption_parts.append(f"{hair} hair")
        if eyes:
            caption_parts.append(f"{eyes} eyes")
        if face:
            caption_parts.append(f"{face}")
        if body and image_type == "full body":
            caption_parts.append(f"{body}")
        if skin_color:
            caption_parts.append(f"{skin_color} skin")
        
        # Add dataset-specific tags
        caption_parts.append("sweatdrop")
        caption_parts.append(emotion)
        
        # Add additional details
        if additional_details:
            caption_parts.append(additional_details)
        
        # Add costume or naked
        if costume == "Naked":
            caption_parts.append("naked")
        else:
            caption_parts.append(f"wear {costume} suit")
        
        # Add background
        caption_parts.extend(["simple background", f"{background_color} background", "solid background"])
        
        # Add sprite-specific tags
        if image_type == "full body":
            caption_parts.extend(["multiple views", "same character in different poses"])
        
        # Join all parts
        caption_text = ", ".join([part for part in caption_parts if part])
        
        # Add additional caption if provided
        if additional_caption and additional_caption.strip():
            caption_text += f", {additional_caption.strip()}"
        
        return caption_text

    def generate_dataset(self, character, game_name, additional_caption=""):
        """Create dataset for Lora training"""
        character_path = os.path.join(self.base_path, character)
        character_config = self.load_character_config(character)

        if not character_config:
            print(f"Failed to load character configuration {character}")
            return [""]

        character_info = character_config.get("character_info", {})
        character_name = character_info.get("name", character)

        formatted_character_name = f"{game_name}_{character_name}"

        lora_dir = os.path.join(character_path, "lora")
        os.makedirs(lora_dir, exist_ok=True)

        processed_files = 0

        faces_dir = os.path.join(character_path, "Faces")
        if os.path.exists(faces_dir):
            costumes = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
            for costume in costumes:
                costume_dir = os.path.join(faces_dir, costume)

                emotions = [d for d in os.listdir(costume_dir) if os.path.isdir(os.path.join(costume_dir, d))]
                for emotion in emotions:
                    emotion_dir = os.path.join(costume_dir, emotion)

                    face_files = [f for f in os.listdir(emotion_dir)
                                  if f.startswith("face_") and os.path.isfile(os.path.join(emotion_dir, f))
                                  and not f.endswith(".txt")]

                    for face_file in face_files:
                        source_path = os.path.join(emotion_dir, face_file)
                        target_filename = f"{costume}_{emotion}_{face_file}"
                        target_path = os.path.join(lora_dir, target_filename)

                        try:
                            shutil.copy2(source_path, target_path)

                            caption_path = os.path.splitext(target_path)[0] + ".txt"
                            caption_text = self.build_caption_text(
                                character_info, costume, emotion, "portrait", 
                                formatted_character_name, additional_caption
                            )

                            with open(caption_path, 'w', encoding='utf-8') as f:
                                f.write(caption_text)

                            processed_files += 1
                        except Exception as e:
                            print(f"Error copying file {source_path}: {e}")

        sprites_dir = os.path.join(character_path, "Sprites")
        if os.path.exists(sprites_dir):
            costumes = [d for d in os.listdir(sprites_dir) if os.path.isdir(os.path.join(sprites_dir, d))]
            for costume in costumes:
                costume_dir = os.path.join(sprites_dir, costume)

                emotions = [d for d in os.listdir(costume_dir) if os.path.isdir(os.path.join(costume_dir, d))]
                for emotion in emotions:
                    emotion_dir = os.path.join(costume_dir, emotion)

                    sprite_files = [f for f in os.listdir(emotion_dir)
                                    if f.startswith("sprite_") and os.path.isfile(os.path.join(emotion_dir, f))
                                    and not f.endswith(".txt")]

                    for sprite_file in sprite_files:
                        source_path = os.path.join(emotion_dir, sprite_file)
                        target_filename = f"{costume}_{emotion}_{sprite_file}"
                        target_path = os.path.join(lora_dir, target_filename)

                        try:
                            shutil.copy2(source_path, target_path)
                            caption_path = os.path.splitext(target_path)[0] + ".txt"

                            caption_text = self.build_caption_text(
                                character_info, costume, emotion, "full body", 
                                formatted_character_name, additional_caption
                            )

                            # Handle multiple character views for sprites
                            sex = character_info.get("sex") or character_info.get("gender", "")
                            if sex == "male" or "1boy" in caption_text:
                                caption_text = caption_text.replace("1boy", "2boys")
                                caption_text += ", multiple boys"
                            else:
                                caption_text = caption_text.replace("1girl", "2girls")  
                                caption_text += ", multiple girls"

                            with open(caption_path, 'w', encoding='utf-8') as f:
                                f.write(caption_text)

                            processed_files += 1
                        except Exception as e:
                            print(f"Error copying file {source_path}: {e}")

        print(f"Dataset created in folder {lora_dir}, processed {processed_files} files")
        return [lora_dir]


NODE_CLASS_MAPPINGS = {
    "DatasetGenerator": DatasetGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DatasetGenerator": "VNCCS Dataset Generator"
}

NODE_CATEGORY_MAPPINGS = {
    "DatasetGenerator": "VNCCS"
}
