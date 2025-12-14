import os
import sys

# Try relative import first, fallback to absolute if running outside package
try:
    from ..utils import (
        base_output_dir, character_dir, list_characters, generate_seed, age_strength, append_age,
        apply_sex, save_config, ensure_character_structure, build_face_details, 
        dedupe_tokens, faces_dir, sheets_dir, EMOTIONS, MAIN_DIRS, load_config
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters, generate_seed, age_strength, append_age,
        apply_sex, save_config, ensure_character_structure, build_face_details,
        dedupe_tokens, faces_dir, sheets_dir, EMOTIONS, MAIN_DIRS, load_config
    )


class CharacterCreator:
    """
    A node for creating and managing characters in VNCCS.
    Handles character configuration, prompt generation, and directory structure setup.
    """
    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
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

    def create_character(self, existing_character: str, background_color: str = "green",
                       aesthetics: str = "", nsfw: bool = False, sex: str = "female", age: int = 18, race: str = "",
                       eyes: str = "", hair: str = "", face: str = "", body: str = "", skin_color: str = "",
                       additional_details: str = "", seed: int = 0,
                       negative_prompt: str = "b",
                       lora_prompt: str = "", new_character_name: str = "") -> tuple:
        """
        Generates character details, configures the character directory, and builds prompts.

        Args:
            existing_character: Name of an existing character to load.
            background_color: Background color for the prompt.
            aesthetics: Aesthetic tags for the prompt.
            nsfw: Whether to generate NSFW content.
            sex: Character sex ('female' or 'male').
            age: Character age.
            race: Character race.
            eyes: Character eye description.
            hair: Character hair description.
            face: Character face description.
            body: Character body description.
            skin_color: Character skin color.
            additional_details: Extra details for the prompt.
            seed: Random seed.
            negative_prompt: Negative prompt tags.
            lora_prompt: LoRA trigger words.
            new_character_name: Name for a new character (overrides existing_character).

        Returns:
            Tuple containing:
            - positive_prompt (str)
            - seed (int)
            - negative_prompt (str)
            - age_lora_strength (float)
            - sheets_path (str)
            - faces_path (str)
            - face_details (str)
        """

        # Determine character name and seed behavior
        if new_character_name.strip():
            character_name = new_character_name.strip()
            seed_randomize_allowed = True
        else:
            character_name = existing_character
            seed_randomize_allowed = True

        # Ensure directory structure exists
        ensure_character_structure(character_name, EMOTIONS, MAIN_DIRS)

        if seed_randomize_allowed:
            seed = generate_seed(seed)

        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name)
        faces_path = faces_dir(character_name)

        # Build Positive Prompt
        prompt_parts = []
        
        if aesthetics:
            prompt_parts.append(aesthetics)
        
        prompt_parts.append("simple background")
        prompt_parts.append("expressionless")

        if background_color:
            prompt_parts.append(f"{background_color} background")
        
        # Join initial parts to pass to apply_sex, as it expects a string
        current_prompt = ", ".join(prompt_parts)
        current_prompt, gender_negative = apply_sex(sex, current_prompt, "")
        
        # Re-split or just append to the string? apply_sex returns a string. 
        # Let's continue building the string or reset the list. 
        # Since apply_sex modifies the prompt significantly, let's work with the string it returns.
        
        if nsfw:
            nude_phrase = "(naked, nude, penis)" if sex == "male" else "(naked, nude, vagina, nipples)"
        else:
            nude_phrase = "(bare chest, wear white boxers)" if sex == "male" else "(wear white bra and panties)"
        
        current_prompt += f", {nude_phrase}"
        current_prompt = append_age(current_prompt, age, sex)
        
        # Add physical attributes
        attributes = [
            (race, "race"),
            (hair, "hair"),
            (eyes, "eyes"),
            (face, "face"),
            (body, "body"),
            (skin_color, "skin"),
        ]

        for value, name in attributes:
            if value:
                current_prompt += f", ({value} {name}:1.0)"

        if additional_details:
            current_prompt += f", ({additional_details})"
        
        if lora_prompt:
            current_prompt += f", {lora_prompt}"

        positive_prompt = current_prompt
        
        age_lora_strength = age_strength(age)

        final_negative_prompt = dedupe_tokens(f"{negative_prompt},{gender_negative}")

        # Load or Create Config
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
