import os

# Try relative import first, fallback to absolute if running outside package
try:
    from ..utils import (
        base_output_dir, character_dir, list_characters, generate_seed, age_strength, append_age,
        apply_sex, save_config, ensure_character_structure, build_face_details, 
        dedupe_tokens, faces_dir, sheets_dir, EMOTIONS, MAIN_DIRS, load_config
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters, generate_seed, age_strength, append_age,
        apply_sex, save_config, ensure_character_structure, build_face_details,
        dedupe_tokens, faces_dir, sheets_dir, EMOTIONS, MAIN_DIRS, load_config
    )


class CharacterCreator:
    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        characters_list = characters if characters else ["None"]
        default_character = characters_list[0]

        return {
            "required": {
                "existing_character": (characters_list, {"default": default_character}),
            },
            "optional": {
                "background_color": ("STRING", {"default": "green"}),
                "aesthetics": ("STRING", {"default": "masterpiece,best quality,amazing quality", "multiline": True}),
                "nsfw": ("BOOLEAN", {"default": False}),
                "sex": (["female", "male"], {"default": "female"}),
                "age": ("INT", {"default": 18, "min": 0, "max": 120}),
                "race": ("STRING", {"default": "human"}),
                "eyes": ("STRING", {"default": "blue eyes"}),
                "hair": ("STRING", {"default": "black long"}),
                "face": ("STRING", {"default": "freckles"}),
                "body": ("STRING", {"default": "medium breasts"}),
                "skin_color": ("STRING", {"default": "white"}),
                "additional_details": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "bad quality,worst quality", "multiline": True}),
                "lora_prompt": ("STRING", {"default": ""}),
                "new_character_name": ("STRING", {"default": ""})
            }
        }
    RETURN_TYPES = ("STRING", "INT", "STRING", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "seed", "negative_prompt", "age_lora_strength", "sheets_path", "faces_path", "face_details")
    FUNCTION = "create_character"
    CATEGORY = "VNCCS"

    def create_character(self, existing_character, background_color="green",
                       aesthetics="", nsfw=False, sex="female", age=18, race="",
                       eyes="", hair="", face="", body="", skin_color="",
                       additional_details="", seed=0,
                       negative_prompt="b",
                       lora_prompt="", new_character_name=""):

        character_name = existing_character
        seed_randomize_allowed = True

        ensure_character_structure(character_name, EMOTIONS, MAIN_DIRS)

        if seed_randomize_allowed:
            seed = generate_seed(seed)

        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name)
        faces_path = faces_dir(character_name)
        positive_prompt = f"{aesthetics}, simple background, expressionless"

        if background_color:
            positive_prompt += f", {background_color} background"
        
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")

        if nsfw:
            nude_phrase = "(naked, nude, penis)" if sex == "male" else "(naked, nude, vagina, nipples)"
        else:
            nude_phrase = "(bare chest, wear white boxers)" if sex == "male" else "(wear white bra and panties)"
        
        
        positive_prompt += f", {nude_phrase}"
        positive_prompt = append_age(positive_prompt, age, sex)
        
        if race:
            positive_prompt += f", ({race} race:1.0)"
        if hair:
            positive_prompt += f", ({hair} hair:1.0)"
        if eyes:
            positive_prompt += f", ({eyes} eyes:1.0)"
        if face:
            positive_prompt += f", ({face} face:1.0)"
        if body:
            positive_prompt += f", ({body} body:1.0)"
        if skin_color:
            positive_prompt += f", ({skin_color} skin:1.0)"
        if additional_details:
            positive_prompt += f", ({additional_details})"
        if lora_prompt:
            positive_prompt += f", {lora_prompt}"

        
        age_lora_strength = age_strength(age)

        final_negative_prompt = dedupe_tokens(f"{negative_prompt},{gender_negative}")

        config = load_config(character_name) or {
            "character_info": {},
            "folder_structure": {
                "main_directories": MAIN_DIRS,
                "emotions": EMOTIONS
            },
            "character_path": character_path,
            "config_version": "2.0"
        }

        config["character_info"] = {
            "name": character_name,
            "background_color": background_color,
            "sex": sex,
            "age": age,
            "race": race,
            "aesthetics": aesthetics,
            "eyes": eyes,
            "hair": hair,
            "face": face,
            "body": body,
            "skin_color": skin_color,
            "additional_details": additional_details,
            "negative_prompt": negative_prompt,
            "lora_prompt": lora_prompt,
            "seed": seed
        }

        # Preserve existing costumes if any
        if "costumes" not in config:
            config["costumes"] = {}

        save_config(character_name, config)

        face_details = build_face_details(config["character_info"])
        face_details += f", (expressionless:1.0)"

        print("VNCCS DEBUG SETTINGS:")
        print("----------------------------------")
        print("seed:", seed)
        print("----------------------------------")
        print("positive_prompt:", positive_prompt)
        print("----------------------------------")
        print("negative_prompt:", final_negative_prompt)
        print("----------------------------------")
        print("face_details:", face_details)
        print("----------------------------------")
        print("age_lora_strength:", age_lora_strength)

        return positive_prompt, seed, final_negative_prompt, age_lora_strength, sheets_path, faces_path, face_details


NODE_CLASS_MAPPINGS = {
    "CharacterCreator": CharacterCreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterCreator": "VNCCS Character Creator"
}

NODE_CATEGORY_MAPPINGS = {
    "CharacterCreator": "VNCCS"
}
