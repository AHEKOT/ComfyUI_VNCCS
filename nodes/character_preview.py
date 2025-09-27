import os

try:
    from ..utils import (
        base_output_dir, character_dir, generate_seed, 
        age_strength, append_age,
        apply_sex, save_config
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, generate_seed, 
        age_strength, append_age,
        apply_sex, save_config
    )


class CharacterPreview:
    """Simplified CharacterCreator version without sheet/face paths.

    Returns only (positive_prompt, seed, negative_prompt).
    Seed: 0 => generates random value (ComfyUI node behavior for generation).
    Directory and config are created for consistency.
    """

    def __init__(self):
        self.base_path = base_output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {"default": "NewCharacter"}),
            },
            "optional": {
                "background_color": ("STRING", {"default": "green"}),
                "aesthetics": ("STRING", {"default": "masterpiece", "multiline": True}),
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
                "negative_prompt": ("STRING", {"default": "bad quality,worst quality,worst detail,sketch,censor, missing arm, missing leg, distorted body", "multiline": True}),
                "lora_prompt": ("STRING", {"default": ""}),
                "styles_list": ("LIST", {"default": []}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("positive_prompt", "seed", "negative_prompt", "age_lora_strength")
    FUNCTION = "preview"
    CATEGORY = "VNCCS"
    
    OUTPUT_IS_LIST = (True, True, True, True)

    def preview(self, character_name, background_color="",
                aesthetics="", sex="female", age=18, race="",
                eyes="", hair="", face="", body="", skin_color="",
                additional_details="", seed=0,
                negative_prompt="",
                lora_prompt="", styles_list=None):
        seed = generate_seed(seed)

        character_path = character_dir(character_name)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
        if not os.path.exists(character_path):
            os.makedirs(character_path, exist_ok=True)

        base_positive_prompt = f"{aesthetics}, simple background, expressionless"
        if background_color:
            base_positive_prompt += f", {background_color} background"
        
        base_positive_prompt, negative_prompt = apply_sex(sex, base_positive_prompt, negative_prompt)
        
        base_positive_prompt = append_age(base_positive_prompt, age, sex)
        
        if race:
            base_positive_prompt += f", ({race} race)"
        if hair:
            base_positive_prompt += f", ({hair} hair)"
        if eyes:
            base_positive_prompt += f", ({eyes} eyes)"
        if face:
            base_positive_prompt += f", ({face} face)"
        if body:
            base_positive_prompt += f", ({body} body)"
        if skin_color:
            base_positive_prompt += f", ({skin_color} skin)"
        if additional_details:
            base_positive_prompt += f", {additional_details}"
        if lora_prompt:
            base_positive_prompt += f", {lora_prompt}"

        age_lora_strength = age_strength(age)

        if styles_list and isinstance(styles_list, list) and len(styles_list) > 0:
            print(f"[VNCCS Character Preview] Batch processing {len(styles_list)} styles...")
            
            positive_prompts = []
            seeds = []
            negative_prompts = []
            age_lora_strengths = []
            
            for i, style in enumerate(styles_list):
                style_seed = generate_seed(0)
                
                if style and style != "None":
                    style_prompt = f"{base_positive_prompt}, artist: {style}"
                else:
                    style_prompt = base_positive_prompt
                
                positive_prompts.append(style_prompt)
                seeds.append(style_seed)
                negative_prompts.append(negative_prompt)
                age_lora_strengths.append(age_lora_strength)
                
                print(f"[VNCCS Character Preview] Style {i+1}/{len(styles_list)}: {style} (seed: {style_seed})")
            
            first_style = styles_list[0] if styles_list else "None"
            config = {
                "character_info": {
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
                    "styles_batch": styles_list,
                    "batch_size": len(styles_list)
                },
                "character_path": character_path,
                "config_version": "2.0"
            }
            save_config(character_name, config)
            
            print(f"[VNCCS Character Preview] Ready {len(positive_prompts)} prompts for generation")
            return positive_prompts, seeds, negative_prompts, age_lora_strengths
        
        else:
            positive_prompt = base_positive_prompt
            
            config = {
                "character_info": {
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
                },
                "character_path": character_path,
                "config_version": "2.0"
            }
            save_config(character_name, config)
            
            print("VNCCS DEBUG SETTINGS:")
            print("----------------------------------")
            print("seed:", seed)
            print("----------------------------------")
            print("positive_prompt:", positive_prompt)
            print("----------------------------------")
            print("negative_prompt:", negative_prompt)
            print("----------------------------------")
            print("age_lora_strength:", age_lora_strength)

            return [positive_prompt], [seed], [negative_prompt], [age_lora_strength]


NODE_CLASS_MAPPINGS = {
    "CharacterPreview": CharacterPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterPreview": "VNCCS Character Preview"
}

NODE_CATEGORY_MAPPINGS = {
    "CharacterPreview": "VNCCS"
}
