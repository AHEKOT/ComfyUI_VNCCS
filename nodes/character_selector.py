import os
import torch

try:
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, load_config, generate_seed,
        apply_sex, build_face_details, dedupe_tokens, ensure_costume_structure,
        faces_dir, sheets_dir, load_character_sheet, load_costume_info, save_costume_info, list_costumes,
        append_age
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, load_config, generate_seed,
        apply_sex, build_face_details, dedupe_tokens, ensure_costume_structure,
        faces_dir, sheets_dir, load_character_sheet, load_costume_info, save_costume_info, list_costumes,
        append_age
    )


class CharacterAssetSelector:
    def __init__(self):
        self.base_path = base_output_dir()
        self.character_config = None

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"

        # Collect all costumes from all characters for dynamic list
        all_costumes = ["Naked"]
        for char in characters:
            if char != "None":
                char_costumes = list_costumes(char)
                for c in char_costumes:
                    if c not in all_costumes:
                        all_costumes.append(c)

        return {
            "required": {
                "character": (characters or ["None"], {"default": default_character}),
                "costume": (all_costumes, {"default": "Naked"}),
            },
            "optional": {
                "face": ("STRING", {"default": ""}),
                "head": ("STRING", {"default": ""}),
                "top": ("STRING", {"default": ""}),
                "bottom": ("STRING", {"default": ""}),
                "shoes": ("STRING", {"default": ""}),
                "extra_negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "new_costume_name": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("face_details", "face_path", "sheet_path", "positive_prompt", "seed", "negative_prompt", "character_sheet")
    CATEGORY = "VNCCS"
    FUNCTION = "select"

    def load_character_config(self, character):
        self.character_config = load_config(character)
        return self.character_config is not None

    def select(self, character, costume, face="", head="", top="", bottom="", shoes="",
               extra_negative_prompt="", new_costume_name=""):
        info = load_character_info(character)
        if not info:
            print(f"[CharacterAssetSelector] Character '{character}' not found")
            return "expressionless", "", "", "", 0, "bad quality,worst quality", torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        config_seed = info.get("seed", 0)
        seed = generate_seed(config_seed)

        base_negative = info.get("negative_prompt", "bad quality,worst quality")

        costumes = list_costumes(character)
        if costume not in costumes:
            costume = "Naked"

        existing_costume = load_costume_info(character, costume)
        
        character_path = character_dir(character)
        
        ensure_costume_structure(character, costume)
        
        costume_data = {
            "face": face,
            "head": head,
            "top": top,
            "bottom": bottom,
            "shoes": shoes
        }
        save_costume_info(character, costume, costume_data)
        
        sheet_path = sheets_dir(character, costume, "neutral")
        face_path = faces_dir(character, costume, "neutral")

        positive_prompt = ""
        face_details = "expressionless"

        aesthetics = info.get("aesthetics", "")
        positive_prompt = aesthetics
        positive_prompt += ", simple background"
        background_color = info.get("background_color", "")
        sex = info.get("sex", "")
        age = info.get("age", 18)
        race = info.get("race", "")
        hair = info.get("hair", "")
        eyes = info.get("eyes", "")
        body = info.get("body", "")
        skin_color = info.get("skin_color", "")
        additional_details = info.get("additional_details", "")
        
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
        
        positive_prompt = append_age(positive_prompt, age, sex)
        
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
        if head:
            positive_prompt += f", (wear {head} on head:1.0)"
        if face:
            positive_prompt += f", (wear {face} on face:1.0)"
        if top:
            positive_prompt += f", (wear {top} on top:1.0)"
        if bottom:
            positive_prompt += f", (wear {bottom} on bottom:1.0)"
        if shoes:
            positive_prompt += f", (wear {shoes} on feet:1.0)"
        
        config_style = info.get("style", "")
        if config_style:
            positive_prompt += f", {config_style}"
        lora_prompt = info.get("lora_prompt", "")
        if lora_prompt:
            positive_prompt += f", {lora_prompt}"

        # Build final negative prompt dynamically: base input + internal restrictions + additional input
        negative_prompt = f"{base_negative}, (naked:2.0), (nude:2.0)"
        negative_prompt += f", {gender_negative}"
        if extra_negative_prompt and extra_negative_prompt.strip():
            negative_prompt += f", {extra_negative_prompt.strip()}"
        
        negative_prompt = dedupe_tokens(negative_prompt)

        # Use build_face_details from utils instead of manual construction
        face_details = build_face_details(info)
        face_details += f", (expressionless:1.0)"
        if head:
            face_details += f", (wear {head} on head:1.0)"
        if face:
            face_details += f", (wear {face} on face:1.0)"
    
        print("VNCCS DEBUG SETTINGS:")
        print("----------------------------------")
        print("seed:", seed)
        print("----------------------------------")
        print("positive_prompt:", positive_prompt)
        print("----------------------------------")
        print("negative_prompt:", negative_prompt)
        print("----------------------------------")
        print("face_details:", face_details)

        sheet_image_tensor = load_character_sheet(character, "Naked", "neutral")
        
        if sheet_image_tensor is None:
            sheet_image_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            
        return face_details, face_path, sheet_path, positive_prompt, seed, negative_prompt, sheet_image_tensor


class CharacterAssetSelectorQWEN:
    def __init__(self):
        self.base_path = base_output_dir()
        self.character_config = None

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        default_character = characters[0] if characters else "None"

        # Collect all costumes from all characters for dynamic list
        all_costumes = ["Naked"]
        for char in characters:
            if char != "None":
                char_costumes = list_costumes(char)
                for c in char_costumes:
                    if c not in all_costumes:
                        all_costumes.append(c)

        return {
            "required": {
                "character": (characters or ["None"], {"default": default_character}),
                "costume": (all_costumes, {"default": "Naked"}),
            },
            "optional": {
                "face": ("STRING", {"default": "", "multiline": True}),
                "head": ("STRING", {"default": "", "multiline": True}),
                "top": ("STRING", {"default": "", "multiline": True}),
                "bottom": ("STRING", {"default": "", "multiline": True}),
                "shoes": ("STRING", {"default": "", "multiline": True}),
                "extra_negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "new_costume_name": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("face_details", "face_path", "sheet_path", "positive_prompt", "seed", "negative_prompt", "character_sheet", "costume_prompt")
    CATEGORY = "VNCCS"
    FUNCTION = "select"
    # costume_prompt is returned as a single multiline STRING with each item on its own line
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False, False)

    def load_character_config(self, character):
        self.character_config = load_config(character)
        return self.character_config is not None

    def select(self, character, costume, face="", head="", top="", bottom="", shoes="",
               extra_negative_prompt="", new_costume_name=""):
        info = load_character_info(character)
        if not info:
            print(f"[CharacterAssetSelectorQWEN] Character '{character}' not found")
            return "expressionless", "", "", "", 0, "bad quality,worst quality", torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        config_seed = info.get("seed", 0)
        seed = generate_seed(config_seed)

        base_negative = info.get("negative_prompt", "bad quality,worst quality")

        costumes = list_costumes(character)
        if costume not in costumes:
            costume = "Naked"

        existing_costume = load_costume_info(character, costume)
        
        character_path = character_dir(character)
        
        ensure_costume_structure(character, costume)
        
        costume_data = {
            "face": face,
            "head": head,
            "top": top,
            "bottom": bottom,
            "shoes": shoes
        }
        save_costume_info(character, costume, costume_data)
        
        sheet_path = sheets_dir(character, costume, "neutral")
        face_path = faces_dir(character, costume, "neutral")

        positive_prompt = ""
        face_details = "expressionless"

        aesthetics = info.get("aesthetics", "")
        positive_prompt = aesthetics
        positive_prompt += ", simple background"
        background_color = info.get("background_color", "")
        sex = info.get("sex", "")
        age = info.get("age", 18)
        race = info.get("race", "")
        hair = info.get("hair", "")
        eyes = info.get("eyes", "")
        body = info.get("body", "")
        skin_color = info.get("skin_color", "")
        additional_details = info.get("additional_details", "")
        
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
        
        positive_prompt = append_age(positive_prompt, age, sex)
        
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
        if head:
            positive_prompt += f", (wear {head} on head:1.0)"
        if face:
            positive_prompt += f", (wear {face} on face:1.0)"
        if top:
            positive_prompt += f", (wear {top} on top:1.0)"
        if bottom:
            positive_prompt += f", (wear {bottom} on bottom:1.0)"
        if shoes:
            positive_prompt += f", (wear {shoes} on feet:1.0)"
        
        config_style = info.get("style", "")
        if config_style:
            positive_prompt += f", {config_style}"
        lora_prompt = info.get("lora_prompt", "")
        if lora_prompt:
            positive_prompt += f", {lora_prompt}"

        # Build final negative prompt dynamically: base input + internal restrictions + additional input
        negative_prompt = f"{base_negative}, (naked:2.0), (nude:2.0)"
        negative_prompt += f", {gender_negative}"
        if extra_negative_prompt and extra_negative_prompt.strip():
            negative_prompt += f", {extra_negative_prompt.strip()}"
        
        negative_prompt = dedupe_tokens(negative_prompt)

        # Use build_face_details from utils instead of manual construction
        face_details = build_face_details(info)
        face_details += f", (expressionless:1.0)"
        if head:
            face_details += f", (wear {head} on head:1.0)"
        if face:
            face_details += f", (wear {face} on face:1.0)"
    
        print("VNCCS DEBUG SETTINGS:")
        print("----------------------------------")
        print("seed:", seed)
        print("----------------------------------")
        print("positive_prompt:", positive_prompt)
        print("----------------------------------")
        print("negative_prompt:", negative_prompt)
        print("----------------------------------")
        print("face_details:", face_details)

        sheet_image_tensor = load_character_sheet(character, "Naked", "neutral")
        
        if sheet_image_tensor is None:
            sheet_image_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
        costume_items = []
        if face and face.strip():
            costume_items.append(face.strip())
        if head and head.strip():
            costume_items.append(head.strip())
        if top and top.strip():
            costume_items.append(top.strip())
        if bottom and bottom.strip():
            costume_items.append(bottom.strip())
        if shoes and shoes.strip():
            costume_items.append(shoes.strip())

        costume_items = "\n".join(costume_items)
        return face_details, face_path, sheet_path, positive_prompt, seed, negative_prompt, sheet_image_tensor, costume_items


NODE_CLASS_MAPPINGS = {
    "CharacterAssetSelector": CharacterAssetSelector,
    "CharacterAssetSelectorQWEN": CharacterAssetSelectorQWEN
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterAssetSelector": "VNCCS Character Selector",
    "CharacterAssetSelectorQWEN": "VNCCS Character Selector QWEN"
}

NODE_CATEGORY_MAPPINGS = {
    "CharacterAssetSelector": "VNCCS",
    "CharacterAssetSelectorQWEN": "VNCCS"
}
