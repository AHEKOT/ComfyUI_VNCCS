import os
import torch
from PIL import Image, ImageOps

try:
    from ..utils import (
        base_output_dir, character_dir, list_characters, 
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters, 
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info
    )


class EmotionGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"
        
        emotion_groups = {
            "Emotions": [
                "angry", "anger vein", "annoyed", "clenched teeth", "scowl", "blush", "blush stickers", 
                "embarrassed", "full-face blush", "nose blush", "bored", "closed eyes", "confused", 
                "crazy", "despair", "determined", "disappointed", "disdain", "disgust", "drunk", 
                "envy", "excited", "exhausted", "expressionless", "facepalm", "flustered", 
                "frustrated", "furrowed brow", "grimace", "guilt", "happy", "kubrick stare", 
                "lonely", "nervous", "nosebleed", "one eye closed (winking)", "round mouth", 
                "open mouth", "parted lips", "pain", "pout", "raised eyebrow", "raised inner eyebrows", 
                "rape face", "rolling eyes", "sad", "depressed", "frown", "gloom (expression)", 
                "tears", "scared", "panicking", "worried", "serious", "sigh", "sleepy", "sulking", 
                "surprised", "thinking", "pensive", "v-shaped eyebrows", "wince"
            ],
            "Sexual": [
                "afterglow", "ahegao", "aroused", "in heat", "naughty face", "ohhoai", 
                "seductive smile", "torogao"
            ],
            "Smile": [
                ":d (:D, open-mouthed smile)", "crazy smile", "evil smile", "fingersmile", 
                "forced smile", "glasgow smile", "grin", "evil grin", "light smile", "sad smile", 
                "seductive smile", "smile (smile with mouth close)", "stifled laugh"
            ],
            "Smug": [
                "doyagao (self-satisfaction / smugness)", "smirk", "smug", "troll face"
            ],
            "Surprised / Scared / Sad": [
                "color drain", "depressed", "despair", "gloom (expression)", "horrified", 
                "screaming", "sobbing", "traumatized", "turn pale", "wavy mouth"
            ],
            "Emotes": [
                ";)", ":d", ";d", "xd", "d:", ":3", ";3", "x3", "3:", "0w0", "uwu", ":p", ";p", 
                ":q", ";q", ">:)", ">:(", ":t", ":i", ":/", ":|", "x mouth", ":c", "c:", ":<", 
                ";<", "diamond mouth", ":>", ":>=", ":o", ";o", "o3o", ">3<", "o_o", "0_0", 
                "|_|", "._.", "solid circle eyes", "heart-shaped eyes", "^_^", "\\(^o^)/", 
                "^q^", ">_<", ">o<", "@_@", ">_@", "+_+", "+_-", "=_=", "t_t", "<o>_<o>", "<|>_<|>"
            ]
        }
        
        available_emotions = []
        for group_emotions in emotion_groups.values():
            available_emotions.extend(group_emotions)
        
        return {
            "required": {
                "character": (characters or ["None"], {"default": default_character}),
                "emotions": ("STRING", {"multiline": True, "default": "angry,shy,smile,pout,sad,neutral"}),
                "emotion_selector": (available_emotions, {"default": "happy"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "MASK")
    RETURN_NAMES = ("image", "emotion", "face_output_path", "sheet_output_path", "positive_prompt", "negative_prompt",
                    "seed", "mask")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False, True)
    CATEGORY = "VNCCS"
    FUNCTION = "generate_emotions"

    def generate_emotions(self, character, emotions, emotion_selector):
        character_path = character_dir(character)
        info = load_character_info(character)
        sheets_dir = os.path.join(character_path, "Sheets")
        costumes = [d for d in os.listdir(sheets_dir) if os.path.isdir(os.path.join(sheets_dir, d))] if os.path.exists(
            sheets_dir) else []
        images = []
        emotions_out = []
        face_output_paths = []
        sheet_output_paths = []
        masks = []

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
            negative_prompt = negative_prompt + "(facial droplet), (water drop), (water), (water droplets), (water drops)"
            config_seed = info.get("seed", 0)
            seed = generate_seed(config_seed)
        else:
            print(f"[EmotionGenerator] Character '{character}' not found")
            info = {}
            aesthetics = background_color = sex = race = eyes = hair = face_features = body = skin_color = additional_details = negative_prompt = lora_prompt = ""
            age = 18
            seed = generate_seed(0)

        emotions_list = [e.strip() for e in emotions.split(",") if e.strip()]

        for costume in costumes:
            costume_info = load_costume_info(character, costume)
            head = costume_info.get("head", "")
            face_wear = costume_info.get("face", "")
            top = costume_info.get("top", "")
            bottom = costume_info.get("bottom", "")
            shoes = costume_info.get("shoes", "")
            
            neutral_dir = os.path.join(sheets_dir, costume, "neutral")
            if not os.path.exists(neutral_dir):
                print(f"Folder {neutral_dir} does not exist")
                continue
            
            img_tensor, mask_tensor = load_character_sheet(character, costume, "neutral", with_mask=True)
            
            if img_tensor is None:
                print(f"Failed to load image for costume {costume}")
                continue

            for emotion_key in emotions_list:
                face_dir = os.path.join(character_path, "Faces", costume, emotion_key)
                os.makedirs(face_dir, exist_ok=True)
                sheet_dir = os.path.join(character_path, "Sheets", costume, emotion_key)
                os.makedirs(sheet_dir, exist_ok=True)
                face_output_path = os.path.join(face_dir, f"face_{emotion_key}_")
                sheet_output_path = os.path.join(sheet_dir, f"sheet_{emotion_key}_")
                
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
                positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
                negative_prompt += f", {gender_negative}"
                positive_prompt = append_age(positive_prompt, age, sex)
                if lora_prompt:
                    positive_prompt += f", {lora_prompt}"

                face_details = build_face_details(info)
                emotion_text = f"({emotion_key}), {face_details}"
                if head:
                    emotion_text += f", (wear {head} on head:1.0)"
                if face_wear:
                    emotion_text += f", (wear {face_wear} on face:1.0)"
                if top:
                    emotion_text += f", (wear {top} on top:1.0)"
                if bottom:
                    emotion_text += f", (wear {bottom} on bottom:1.0)"
                if shoes:
                    emotion_text += f", (wear {shoes} on feet:1.0)"

                if mask_tensor is None:
                    h, w = img_tensor.shape[2], img_tensor.shape[3]
                    mask_tensor = torch.ones((1, h, w), dtype=torch.float32)
                
                images.append(img_tensor)
                emotions_out.append(emotion_text)
                masks.append(mask_tensor)
                for _ in range(12):
                    face_output_paths.append(face_output_path)
                sheet_output_paths.append(sheet_output_path)

        print("VNCCS DEBUG SETTINGS:")
        print("----------------------------------")
        print("seed:", seed)
        print("----------------------------------")
        print("positive_prompt:", positive_prompt)
        print("----------------------------------")
        print("negative_prompt:", negative_prompt)
        print("----------------------------------")
        print("emotion prompt:", emotions_out)

        if not images:
            return [], [], [], [], "", "", seed, []

        return images, emotions_out, face_output_paths, sheet_output_paths, positive_prompt, negative_prompt, seed, masks


NODE_CLASS_MAPPINGS = {
    "EmotionGenerator": EmotionGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionGenerator": "VNCCS Emotion Generator"
}

NODE_CATEGORY_MAPPINGS = {
    "EmotionGenerator": "VNCCS"
}
