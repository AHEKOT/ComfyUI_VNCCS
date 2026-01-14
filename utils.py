"""VNCCS utilities - common functions for character management."""

import os
import json
import random
import re
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


EMOTIONS = ["neutral"]
MAIN_DIRS = ["Sprites", "Faces", "Sheets"]
AGE_CONTROL_POINTS = [
    (0, -5.0),
    (3, -4.0),
    (5, -3.0),
    (7, -2.0),
    (9, -1.0),
    (11, 0.0),
    (14, 1.0),
    (16, 1.5),
    (18, 2.0),
    (30, 2.5),
    (40, 3.0),
    (50, 3.5),
    (60, 3.5),
    (70, 4.0),
    (80, 5.0),
]


def base_output_dir() -> str:
    """Get base output directory path."""
    try:
        from folder_paths import get_output_directory
        return os.path.join(get_output_directory(), "VNCCS", "Characters")
    except ImportError:
        # Fallback for local usage
        current_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(current_dir, "..", "..", "output", "VNCCS", "Characters"))


def get_legacy_output_dir() -> str:
    """Get legacy output directory path for migration."""
    try:
        from folder_paths import get_output_directory
        return os.path.join(get_output_directory(), "VN_CharacterCreatorSuit")
    except ImportError:
        current_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(current_dir, "..", "..", "output", "VN_CharacterCreatorSuit"))


import shutil
import traceback

def migrate_legacy_data() -> dict:
    """Check for legacy data and migrate to new location.
    
    Returns:
        dict: with keys 'migrated' (bool), 'count' (int), 'details' (list of names)
    """
    try:
        print("[VNCCS Migration] Starting migration check...")
        old_dir = get_legacy_output_dir()
        new_dir = base_output_dir()
        
        print(f"[VNCCS Migration] Old Dir: {old_dir}")
        print(f"[VNCCS Migration] New Dir: {new_dir}")
        
        if not os.path.exists(old_dir):
            print("[VNCCS Migration] Legacy folder not found.")
            return {"migrated": False, "count": 0, "message": "No legacy folder found"}
        
        # Check if old folder has content
        try:
            items = os.listdir(old_dir)
        except OSError as e:
            print(f"[VNCCS Migration] Error reading legacy folder: {e}")
            return {"migrated": False, "count": 0, "message": f"Error reading legacy folder: {e}"}
            
        chars_to_move = [i for i in items if os.path.isdir(os.path.join(old_dir, i))]
        print(f"[VNCCS Migration] Found candidates: {chars_to_move}")
        
        if not chars_to_move:
            print("[VNCCS Migration] Legacy folder empty (no subdirs).")
            return {"migrated": False, "count": 0, "message": "Legacy folder empty"}

        # Ensure new dir exists
        if not os.path.exists(new_dir):
            try:
                os.makedirs(new_dir, exist_ok=True)
                print(f"[VNCCS Migration] Created new dir: {new_dir}")
            except Exception as e:
                 print(f"[VNCCS Migration] Failed to create new dir: {e}")
                 return {"migrated": False, "count": 0, "message": f"Failed to create new dir: {e}"}
        
        moved_count = 0
        moved_names = []
        errors = []
        
        for char_name in chars_to_move:
            src = os.path.join(old_dir, char_name)
            dst = os.path.join(new_dir, char_name)
            
            if os.path.exists(dst):
                msg = f"Skipped {char_name}: already exists in new location"
                print(f"[VNCCS Migration] {msg}")
                errors.append(msg)
                continue
                
            try:
                print(f"[VNCCS Migration] Moving {src} -> {dst}")
                shutil.move(src, dst)
                moved_count += 1
                moved_names.append(char_name)
            except Exception as e:
                msg = f"Failed to move {char_name}: {str(e)}"
                print(f"[VNCCS Migration] {msg}")
                errors.append(msg)
                
        # If old dir is empty now, try to remove it
        if not os.listdir(old_dir):
            try:
                print("[VNCCS Migration] Removing empty legacy dir.")
                os.rmdir(old_dir)
            except Exception as e:
                print(f"[VNCCS Migration] Failed to remove empty legacy dir: {e}")
                pass
                
        return {
            "migrated": moved_count > 0,
            "count": moved_count,
            "details": moved_names,
            "errors": errors,
            "all_characters": list_characters()
        }
    except Exception as e:
        trace = traceback.format_exc()
        print(f"[VNCCS Migration] CRITICAL ERROR: {e}\n{trace}")
        return {
            "migrated": False, 
            "error": str(e), 
            "trace": trace
        }


def character_dir(name: str) -> str:
    """Get character directory path."""
    return os.path.join(base_output_dir(), name)


def faces_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get faces directory path."""
    return os.path.join(character_dir(name), "Faces", costume, emotion, "face_neutral")


def sheets_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get sheets directory path."""
    return os.path.join(character_dir(name), "Sheets", costume, emotion, "sheet_neutral")


def sprites_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get sprites directory path."""
    return os.path.join(character_dir(name), "Sprites", costume, emotion, "sprite_neutral")


def ensure_character_structure(name: str, emotions: List[str] = None, main_dirs: List[str] = None) -> None:
    """Create basic character directory structure."""
    if emotions is None:
        emotions = EMOTIONS
    if main_dirs is None:
        main_dirs = MAIN_DIRS
    
    char_path = character_dir(name)
    base_path = base_output_dir()
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    if not os.path.exists(char_path):
        os.makedirs(char_path)
    
    for main_dir in main_dirs:
        main_dir_path = os.path.join(char_path, main_dir)
        if not os.path.exists(main_dir_path):
            os.makedirs(main_dir_path)
        
        naked_path = os.path.join(main_dir_path, "Naked")
        if not os.path.exists(naked_path):
            os.makedirs(naked_path)
        
        for emotion in emotions:
            emotion_path = os.path.join(naked_path, emotion)
            if not os.path.exists(emotion_path):
                os.makedirs(emotion_path)


def ensure_costume_structure(name: str, costume: str, emotions: List[str] = None) -> None:
    """Create costume directory structure."""
    if emotions is None:
        emotions = EMOTIONS
    
    char_path = character_dir(name)
    
    for main_dir in MAIN_DIRS:
        main_dir_path = os.path.join(char_path, main_dir)
        if not os.path.exists(main_dir_path):
            os.makedirs(main_dir_path)
        
        costume_path = os.path.join(main_dir_path, costume)
        if not os.path.exists(costume_path):
            os.makedirs(costume_path)
        
        for emotion in emotions:
            emotion_path = os.path.join(costume_path, emotion)
            if not os.path.exists(emotion_path):
                os.makedirs(emotion_path)


def list_characters() -> List[str]:
    """Get list of existing characters."""
    base_path = base_output_dir()
    try:
        return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    except Exception:
        return []


def generate_seed(value: int) -> int:
    """Generate seed. If value == 0, creates new 64-bit non-zero seed."""
    if value == 0:
        seed = random.getrandbits(64)
        if seed == 0:
            seed = random.getrandbits(64) or 1
        return seed
    return value


def inherit_seed(input_seed: int, upstream_seed: Optional[int]) -> int:
    """Inherit seed from upstream if input_seed == 0."""
    if input_seed != 0:
        return input_seed
    if upstream_seed and upstream_seed != 0:
        return upstream_seed
    return generate_seed(0)


def normalize_sex(raw: Optional[str]) -> str:
    """Normalize gender value."""
    if not raw:
        return "female"
    raw_lower = raw.lower().strip()
    if raw_lower in ["male", "man", "boy", "m"]:
        return "male"
    return "female"


def sex_positive_tokens(sex: str, mode: str = "default") -> List[str]:
    """Get positive tokens for gender."""
    if sex == "male":
        if mode == "creator":
            return ["1boy", "solo:2 male_focus"]
        else:
            return ["1boy", "male_focus"]
    else:
        return ["1girl"]


def sex_negative_tokens(sex: str, mode: str = "default") -> List[str]:
    """Get negative tokens for gender."""
    if sex == "male":
        if mode == "creator":
            return ["1girl", "girl", "woman", "femine", "breasts", "vagina", "boobs", "small breasts", "medium breasts", "big breasts", "erected", "erected_penis", "water_drop", "bra"]
        else:
            return ["1girl", "girl", "woman", "femine", "breasts", "vagina"]
    else:
        return ["1boy", "man", "penis", "dick"]


def apply_sex(sex: str, positive_prompt: str, negative_prompt: str) -> Tuple[str, str]:
    """Apply gender settings to prompts."""
    sex = normalize_sex(sex)
    
    pos_tokens = sex_positive_tokens(sex)
    for token in pos_tokens:
        positive_prompt += f", ({token})"
    
    neg_tokens = sex_negative_tokens(sex)
    if sex == "male":
        negative_prompt += f", (((({', '.join(neg_tokens)}))))"
    else:
        negative_prompt += f", {', '.join(neg_tokens)}"
    
    return positive_prompt, negative_prompt


def age_strength(age: int) -> float:
    """Calculate LoRA strength for age."""
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        age_int = 18
    
    if age_int <= AGE_CONTROL_POINTS[0][0]:
        return AGE_CONTROL_POINTS[0][1]
    if age_int >= AGE_CONTROL_POINTS[-1][0]:
        return AGE_CONTROL_POINTS[-1][1]
    
    for (x0, y0), (x1, y1) in zip(AGE_CONTROL_POINTS, AGE_CONTROL_POINTS[1:]):
        if age_int <= x1:
            t = (age_int - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 3)
    
    return AGE_CONTROL_POINTS[-1][1]


def age_body_descriptor(age: int, sex: str) -> str:
    """Get body type descriptor by age."""
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        return ""
    if sex == "female":
        if age_int <= 3:
            return "(toddler girl:1.0)"
        elif age_int <= 11:
            return "(loli:1.0)"
        elif age_int <= 18:
            return "(teenager girl:1.0)"
        elif age_int <= 24:
            return "(young_adult woman:1.0)"
        elif age_int <= 50:
            return "(adult woman:1.0)"
        elif age_int <= 60:
            return "(old woman:1.0)"
        else:
            return ""
    else:
        if age_int <= 3:
            return "(toddler boy:1.0)"
        elif age_int <= 11:
            return "(shota:1.0)"
        elif age_int <= 16:
            return "(teenager boy:1.0)"
        elif age_int <= 18:
            return "(young_adult man:1.0)"
        elif age_int <= 24:
            return "(young_adult man:1.5)"
        elif age_int <= 50:
            return "(adult man:1.0)"
        elif age_int <= 60:
            return "(old man:1.0)"
        else:
            return ""



def append_age(positive_prompt: str, age: int, sex: str) -> str:
    """Add age descriptors to prompt."""
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        age_int = 18
    
    positive_prompt += f", {age_int}yo"
    
    body_desc = age_body_descriptor(age_int, sex)
    if body_desc:
        positive_prompt += f", {body_desc}"
    
    return positive_prompt


def config_path(character_name: str) -> str:
    """Get character config file path."""
    return os.path.join(character_dir(character_name), f"{character_name}_config.json")


def load_config(character_name: str) -> Optional[Dict[str, Any]]:
    """Load character configuration."""
    config_file = config_path(character_name)
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[VNCCS Utils] Error loading configuration {character_name}: {e}")
    return None


def save_config(character_name: str, data: Dict[str, Any]) -> str:
    """Save character configuration."""
    char_dir = character_dir(character_name)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir, exist_ok=True)
    
    config_file = config_path(character_name)
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return config_file
    except Exception as e:
        print(f"[VNCCS Utils] Error saving configuration {character_name}: {e}")
        return ""


def load_character_info(character_name: str) -> Optional[Dict[str, Any]]:
    """Load character info with sex/gender unification."""
    config = load_config(character_name)
    if not config:
        return None
    
    char_info = config.get("character_info", {})
    
    sex = char_info.get("sex") or char_info.get("gender")
    if sex:
        char_info["sex"] = normalize_sex(sex)
        char_info["gender"] = char_info["sex"]
    
    return char_info


def build_face_details(char_info: Dict[str, Any]) -> str:
    """Build face_details string from character info."""
    details_parts = []
    
    sex = char_info.get("sex") or char_info.get("gender")
    if sex == "male":
        details_parts.append("1boy")
    else:
        details_parts.append("1girl")
    
    if char_info.get("race"):
        details_parts.append(f"{char_info['race']} race")
    
    if char_info.get("eyes"):
        details_parts.append(f"{char_info['eyes']} eyes")
    
    if char_info.get("hair"):
        details_parts.append(f"{char_info['hair']} hair")
    
    if char_info.get("face"):
        details_parts.append(f"{char_info['face']} face")
    
    if char_info.get("skin_color"):
        details_parts.append(f"{char_info['skin_color']} skin")
    
    if char_info.get("additional_details"):
        details_parts.append(char_info['additional_details'])
    
    
    return ",".join([p for p in details_parts if p])


def dedupe_tokens(line: str) -> str:
    """Remove duplicate tokens from prompt string."""
    if not line:
        return line
    
    parts = []
    seen = set()
    
    for segment in line.split(','):
        token = segment.strip()
        if not token:
            continue
        if token not in seen:
            seen.add(token)
            parts.append(token)
    
    return ','.join(parts)


def load_costume_info(character_name: str, costume_name: str) -> Dict[str, Any]:
    """Load character costume info."""
    config = load_config(character_name)
    if not config:
        return {}
    costumes = config.get("costumes", {})
    return costumes.get(costume_name, {})


def save_costume_info(character_name: str, costume_name: str, costume_data: Dict[str, Any]) -> bool:
    """Save character costume info."""
    config = load_config(character_name)
    if not config:
        config = {"character_info": {}, "costumes": {}}
    if "costumes" not in config:
        config["costumes"] = {}
    config["costumes"][costume_name] = costume_data
    return save_config(character_name, config) != ""


def load_character_sheet(character: str, costume: str = "Naked", emotion: str = "neutral", with_mask: bool = False) -> Optional["torch.Tensor"]:
    """Load character sheet image.
    
    Args:
        character (str): Character name
        costume (str): Costume name (default "Naked")
        emotion (str): Emotion (default "neutral")
        with_mask (bool): Whether to return alpha mask separately (default False)
        
    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor] or None: 
        - If with_mask=False: RGBA image tensor [1, H, W, 4] or None on error
        - If with_mask=True: (RGB image [1, H, W, 3], alpha mask [1, H, W]) or (None, None) on error
    """
    try:
        import torch
        from PIL import Image, ImageOps
        import numpy as np
    except ImportError:
        print("[VNCCS Utils] Required libraries (torch, PIL, numpy) not installed")
        return None
    
    try:
        sheet_dir = os.path.join(character_dir(character), "Sheets", costume, emotion)
        
        if not os.path.isdir(sheet_dir):
            print(f"[VNCCS Utils] Directory not found: {sheet_dir}")
            return None
        
        candidates = []
        pattern = f"sheet_{emotion}_*(\\d+)_*\\.png"
        
        for fname in os.listdir(sheet_dir):
            m = re.match(pattern, fname)
            if m:
                idx = int(m.group(1))
                candidates.append((idx, fname))
        
        if not candidates:
            print(f"[VNCCS Utils] Files sheet_{emotion}_*number.png not found in {sheet_dir}")
            if with_mask:
                return None, None
            else:
                return None
        
        candidates.sort(key=lambda x: x[0])
        _, best_name = candidates[-1]
        best_path = os.path.join(sheet_dir, best_name)
        
        img_pil = Image.open(best_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        has_alpha = img_pil.mode == "RGBA" or img_pil.mode == "LA" or img_pil.mode == "P" and "transparency" in img_pil.info
        
        if img_pil.mode != "RGBA":
            img_pil = img_pil.convert("RGBA")
        
        image_np = np.array(img_pil).astype(np.float32) / 255.0
        
        if with_mask:
            if has_alpha:
                img_tensor = torch.from_numpy(image_np[..., :3])[None,]
                mask_alpha_channel = image_np[..., 3]
                mask_tensor = torch.from_numpy(mask_alpha_channel).unsqueeze(0)
                print(f"[VNCCS Utils] Loaded sheet with mask: {best_path}")
                return img_tensor, mask_tensor
            else:
                img_tensor = torch.from_numpy(image_np[..., :3])[None,]
                print(f"[VNCCS Utils] Loaded sheet without mask: {best_path}")
                return img_tensor, None
        else:
            # Return RGBA image for ComfyUI compatibility
            if has_alpha:
                # Keep alpha channel for proper transparency handling
                sheet_image_tensor = torch.from_numpy(image_np)[None,]  # [1, H, W, 4]
                print(f"[VNCCS Utils] Loaded RGBA sheet: {best_path}")
            else:
                # Convert RGB to RGBA by adding opaque alpha channel
                rgb_image = image_np[..., :3]
                alpha_channel = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.float32)
                rgba_image = np.concatenate([rgb_image, alpha_channel], axis=2)
                sheet_image_tensor = torch.from_numpy(rgba_image)[None,]  # [1, H, W, 4]
                print(f"[VNCCS Utils] Loaded RGB sheet (converted to RGBA): {best_path}")
            return sheet_image_tensor
        
    except Exception as e:
        print(f"[VNCCS Utils] Error loading sheet image: {e}")
        if with_mask:
            return None, None
        else:
            return None


def list_costumes(character_name: str) -> List[str]:
    """Get list of available costumes for character."""
    costumes = ["Naked"]
    
    config = load_config(character_name)
    if config:
        costumes_dict = config.get("costumes", {})
        for c in costumes_dict.keys():
            if c not in costumes:
                costumes.append(c)
    
    char_dir = character_dir(character_name)
    sheets_dir_path = os.path.join(char_dir, "Sheets")
    if os.path.exists(sheets_dir_path):
        try:
            for item in os.listdir(sheets_dir_path):
                item_path = os.path.join(sheets_dir_path, item)
                if os.path.isdir(item_path) and item not in costumes:
                    costumes.append(item)
        except OSError:
            pass
    
    return costumes


create_costume_folders = ensure_costume_structure
