import os
import json
from PIL import Image
from ..utils import (
    base_output_dir, character_dir, list_characters,
    load_character_info, ensure_costume_structure, EMOTIONS,
    apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
    sheets_dir, load_costume_info
)

class EmotionPreviewGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": (list_characters(),),
                # Defines width and height inputs separately
                "width": ("INT", {
                    "default": 1024, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 256, 
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 256, 
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = (
        "LIST",  # 0: prompts
        "LIST",  # 1: negative_prompts
        "LIST",  # 2: face prompts
        "LIST",  # 3: paths
        "INT",   # 4: seed
        "INT",   # 5: width
        "INT",   # 6: height
        "INT",   # 7: dataset_size
    )
    RETURN_NAMES = (
        "prompts",
        "negative_prompts",
        "face_prompts",
        "paths",
        "seed",
        "width",
        "height",
        "dataset_size"
    )
    FUNCTION = "generate"
    CATEGORY = "VNCCS"

    def build_emotion_groups(self):
        json_path = os.path.join(os.path.dirname(__file__), "..", "emotions-config", "emotions.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[VNCCS Preview] Failed to load emotion data: {e}")
            return {}

    def get_base_sheet_path(self, character_name):
        sheet_data = load_character_sheet(character_name)
        base_sheet_folder = os.path.join(sheets_dir(character_name), "Naked", "neutral")
        for f in os.listdir(base_sheet_folder):
            if "emotion_previews" not in f:
                if f.startswith("sheet_") and f.endswith(".png"):
                    return os.path.join(base_sheet_folder, f)
        raise FileNotFoundError(f"No sheet_*.png found in Naked/neutral for {character_name}")

    def load_base_sheet_image(self, path):
        try:
            return Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"[VNCCS Preview] Failed to load base sheet image: {e}")
            return Image.new("RGBA", (512, 512), (200, 200, 200, 255))

    def generate(self, character, width, height):
        try:
            load_character_info.cache_clear()
        except AttributeError:
            # Handle case where it's not decorated/cached (optional)
            pass
        info = load_character_info(character)
        emotion_groups = self.build_emotion_groups()
        
        aesthetics = lora_prompt = ""
        config_seed = 0
        negative_prompt = ""

        if info:
            aesthetics = info.get("aesthetics", "")
            background_color = info.get("background_color", "blue")
            sex = info.get("sex", "")
            age = info.get("age", 18)
            race = info.get("race", "")
            eyes = info.get("eyes", "")
            hair = info.get("hair", "")
            face_features = info.get("face", "")
            body = info.get("body", "")
            skin_color = info.get("skin_color", "")
            additional_details = info.get("additional_details", "")
            lora_prompt = info.get("lora_prompt", "")
            negative_prompt = info.get("negative_prompt", "")
            negative_prompt += " (facial droplet), (water drop), (water), (water droplets), (water drops)"
            config_seed = info.get("seed", 0)
            seed = generate_seed(config_seed)
        else:
            print(f"[EmotionPreviewGenerator] Character '{character}' not found")
            aesthetics = background_color = sex = race = eyes = hair = face_features = body = skin_color = additional_details = negative_prompt = lora_prompt = ""
            age = 18
            seed = generate_seed(0)
            
        basedir = base_output_dir()
        output_dir = os.path.join(basedir, "emotion_previews", character)
        prompts = []
        negative_prompts = []
        face_prompts = []
        paths = []
        
        all_emotions = []
        for group, items in emotion_groups.items():
            all_emotions.extend(items)
        
        dataset_size = len(all_emotions) # Total number of items to iterate over

        # Check for square resolution, which triggers a specific tag
        is_square = (width == height)

        for emotion in all_emotions:
            emotion_text = emotion["description"]
            base_prompt = build_face_details(info)
            face_prompt = base_prompt + ", " + emotion_text
            positive_prompt = f"{aesthetics}"
            if background_color:
                positive_prompt += f", {background_color} background"
            if race:
                positive_prompt += f", ({race} race:1.0)"
            if hair:
                positive_prompt += f", ({hair} hair:1.0)"
            if eyes:
                positive_prompt += f", ({eyes} eyes:1.0)"
            if body:
                positive_prompt += f", ({body} body:1.0)"
            if skin_color:
                positive_prompt += f", ({skin_color} skin:1.0)"
            if additional_details:
                positive_prompt += f", ({additional_details})"

            positive_prompt_after_sex, gender_negative = apply_sex(sex, positive_prompt, "")
            positive_prompt = positive_prompt_after_sex
            negative_prompt += f", {gender_negative}"

            positive_prompt = append_age(positive_prompt, age, sex)

            # Line 116 update: Conditionally add ", 1:1 aspect ratio, perfectly centered" tag
            full_prompt = f"{positive_prompt}, detailed facial features, facing viewer directly, portrait composition, "
            
            if is_square:
                full_prompt += f", 1:1 aspect ratio, instagram post, perfectly centered, " # Added tag if the user chose a square composition
            
            full_prompt += f"{base_prompt}, {emotion_text}"

            if lora_prompt:
                full_prompt += f", {lora_prompt}"

            filename = f"{emotion['safe_name']}"
            out_path = os.path.join(output_dir, filename)

            prompts.append(full_prompt)
            negative_prompts.append(negative_prompt)
            face_prompts.append(face_prompt)
            paths.append(out_path)

        # The seed value is a static starting point for the new iterator node
        # UPDATED: Return width and height instead of a single resolution value
        return (prompts, negative_prompts, face_prompts, paths, seed, width, height, dataset_size)


# Node registration
NODE_CLASS_MAPPINGS  = {
    "EmotionPreviewGenerator": EmotionPreviewGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionPreviewGenerator": "VNCCS Emotion Preview Generator"
}

NODE_CATEGORY_MAPPINGS = {
    "EmotionPreviewGenerator": "VNCCS"
}
