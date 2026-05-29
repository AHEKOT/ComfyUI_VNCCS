"""VNCCS utilities - common functions for character management."""

import os
import ntpath
import json
import random
import re
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


EMOTIONS = ["neutral"]
MAIN_DIRS = ["Sprites", "Faces", "Sheets"]
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,120}$")
PRIVILEGED_REQUEST_HEADER = "X-VNCCS-CSRF"
PRIVILEGED_REQUEST_VALUE = "1"
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


def ensure_safe_name(value: str, field: str = "name") -> str:
    """Validate a user-controlled path segment used by VNCCS."""
    if value is None:
        raise ValueError(f"{field} is required")
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field} is required")
    if value in {".", ".."} or ".." in value:
        raise ValueError(f"{field} contains invalid path traversal")
    if not SAFE_NAME_RE.match(value):
        raise ValueError(f"{field} may only contain letters, numbers, spaces, underscores and hyphens")
    return value


def normalize_filesystem_path(value: str) -> str:
    """Normalize slash direction for paths persisted by any OS."""
    return str(value or "").strip().replace("\\", os.sep).replace("/", os.sep)


def _portable_parts(value: str) -> List[str]:
    normalized = str(value or "").strip().replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def is_absolute_path_any_os(value: str) -> bool:
    """Return True for POSIX, Windows drive, UNC, or home-rooted paths."""
    raw = str(value or "").strip()
    normalized = raw.replace("\\", "/")
    return (
        os.path.isabs(raw)
        or ntpath.isabs(raw)
        or bool(ntpath.splitdrive(raw)[0])
        or normalized.startswith("/")
        or normalized.startswith("~")
    )


def is_path_under(base: str, path: str) -> bool:
    """Compare filesystem containment after normalizing slash style."""
    try:
        base_abs = os.path.abspath(normalize_filesystem_path(base))
        path_abs = os.path.abspath(normalize_filesystem_path(path))
        return os.path.commonpath([base_abs, path_abs]) == base_abs
    except Exception:
        return False


def safe_join_under(base: str, *parts: str) -> str:
    """Join path parts and ensure the result remains under base."""
    base_abs = os.path.abspath(normalize_filesystem_path(base))
    normalized_parts = []
    for part in parts:
        raw = str(part)
        if is_absolute_path_any_os(raw):
            raise ValueError("path escapes allowed directory")
        part_items = _portable_parts(raw)
        if any(item in {".", ".."} or "\0" in item for item in part_items):
            raise ValueError("path escapes allowed directory")
        normalized_parts.extend(part_items)
    target = os.path.abspath(os.path.join(base_abs, *normalized_parts))
    if not is_path_under(base_abs, target):
        raise ValueError("path escapes allowed directory")
    return target


def safe_relative_path(value: str, field: str = "path") -> str:
    """Validate a relative path sent by UI for resources below a known root."""
    if value is None:
        raise ValueError(f"{field} is required")
    normalized = str(value).strip().replace("\\", "/")
    if not normalized:
        raise ValueError(f"{field} is required")
    if is_absolute_path_any_os(normalized):
        raise ValueError(f"{field} must be relative")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"{field} contains invalid path traversal")
    if any("\0" in part for part in parts):
        raise ValueError(f"{field} contains invalid characters")
    return "/".join(parts)


def validate_privileged_request(request) -> None:
    """Validate state-changing VNCCS API calls from the same ComfyUI origin."""
    host = (request.headers.get("Host") or "").lower()
    same_origin = False
    for header_name in ("Origin", "Referer"):
        raw = request.headers.get(header_name)
        if not raw:
            continue
        parsed = urlparse(raw)
        if parsed.netloc and host and parsed.netloc.lower() == host:
            same_origin = True
        elif parsed.netloc and host:
            raise ValueError("cross-origin privileged request rejected")

    sec_fetch_site = (request.headers.get("Sec-Fetch-Site") or "").lower()
    if sec_fetch_site == "cross-site":
        raise ValueError("cross-site privileged request rejected")

    if request.headers.get(PRIVILEGED_REQUEST_HEADER) == PRIVILEGED_REQUEST_VALUE:
        return

    if same_origin and sec_fetch_site in {"", "same-origin", "same-site", "none"}:
        return

    raise ValueError(f"missing {PRIVILEGED_REQUEST_HEADER} header")


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

def _migration_archive_path(path: str) -> str:
    """Return a unique archive path for a migrated legacy directory."""
    base = f"{path}_migrated_safe_to_delete"
    if not os.path.exists(base):
        return base
    index = 2
    while True:
        candidate = f"{base}_{index}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


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
        
        migrated_count = 0
        migrated_names = []
        errors = []
        
        for char_name in chars_to_move:
            src = os.path.join(old_dir, char_name)
            dst = os.path.join(new_dir, char_name)

            if os.path.exists(dst):
                # Destination exists: keep legacy data intact and archive old_dir after success.
                dst_config = os.path.join(dst, f"{char_name}_config.json")
                if os.path.exists(dst_config):
                    print(f"[VNCCS Migration] {char_name}: already in new location, keeping legacy copy for archive")
                    migrated_count += 1
                    migrated_names.append(char_name)
                else:
                    # dst exists but is empty/broken: replace dst with a copy, then archive old_dir.
                    print(f"[VNCCS Migration] {char_name}: dst exists but incomplete, replacing from legacy copy")
                    try:
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        migrated_count += 1
                        migrated_names.append(char_name)
                    except Exception as e:
                        errors.append(f"Failed to replace {char_name}: {e}")
                continue

            try:
                print(f"[VNCCS Migration] Copying {src} -> {dst}")
                shutil.copytree(src, dst)
                migrated_count += 1
                migrated_names.append(char_name)
            except Exception as e:
                msg = f"Failed to copy {char_name}: {str(e)}"
                print(f"[VNCCS Migration] {msg}")
                errors.append(msg)

        archive_path = ""
        if migrated_count > 0 and not errors:
            try:
                archive_path = _migration_archive_path(old_dir)
                print(f"[VNCCS Migration] Archiving legacy dir: {old_dir} -> {archive_path}")
                os.rename(old_dir, archive_path)
            except Exception as e:
                msg = f"Failed to archive legacy dir: {e}"
                print(f"[VNCCS Migration] {msg}")
                errors.append(msg)
                
        return {
            "migrated": migrated_count > 0 and not errors,
            "count": migrated_count,
            "details": migrated_names,
            "errors": errors,
            "archive_path": archive_path,
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
    return safe_join_under(base_output_dir(), ensure_safe_name(name, "character"))


def faces_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get faces directory path."""
    return safe_join_under(
        character_dir(name),
        "Faces",
        ensure_safe_name(costume, "costume"),
        ensure_safe_name(emotion, "emotion"),
        "face_neutral",
    )


def sheets_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get sheets directory path."""
    return safe_join_under(
        character_dir(name),
        "Sheets",
        ensure_safe_name(costume, "costume"),
        ensure_safe_name(emotion, "emotion"),
        "sheet_neutral",
    )


def sprites_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    """Get sprites directory path."""
    return safe_join_under(
        character_dir(name),
        "Sprites",
        ensure_safe_name(costume, "costume"),
        ensure_safe_name(emotion, "emotion"),
        "sprite_neutral",
    )


def ensure_character_structure(name: str, emotions: List[str] = None, main_dirs: List[str] = None) -> None:
    """Create basic character directory structure.
    
    Note: Emotion folders are NOT created here. They are created on-demand
    when images are actually saved.
    """
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



def ensure_costume_structure(name: str, costume: str, emotions: List[str] = None) -> None:
    """Create costume directory structure.
    
    Note: Emotion folders are NOT created here. They are created on-demand
    when images are actually saved.
    """
    char_path = character_dir(name)
    
    for main_dir in MAIN_DIRS:
        main_dir_path = os.path.join(char_path, main_dir)
        if not os.path.exists(main_dir_path):
            os.makedirs(main_dir_path)
        
        costume_path = safe_join_under(main_dir_path, ensure_safe_name(costume, "costume"))
        if not os.path.exists(costume_path):
            os.makedirs(costume_path)



def list_characters() -> List[str]:
    """Get list of existing characters from new and (as fallback) legacy paths."""
    base_path = base_output_dir()
    try:
        chars = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    except Exception as exc:
        print(f"[VNCCS] list_characters: failed to list new character path '{base_path}': {exc}")
        chars = []
    if not chars:
        try:
            legacy = get_legacy_output_dir()
            legacy_chars = sorted([d for d in os.listdir(legacy) if os.path.isdir(os.path.join(legacy, d))])
            if legacy_chars:
                print(f"[VNCCS] list_characters: new path empty, falling back to legacy. Run migration to fix.")
            chars = legacy_chars
        except Exception as exc:
            print(f"[VNCCS] list_characters: failed to list legacy character path: {exc}")
    return chars


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


def normalize_hair_tags(hair: str) -> str:
    """Ensure free-form hair input mentions hair."""
    if not hair:
        return ""

    normalized = str(hair).strip()
    if not normalized:
        return ""
    if "hair" not in normalized.lower():
        normalized = f"{normalized} hair"
    return normalized


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
    
    hair = normalize_hair_tags(char_info.get("hair", ""))
    if hair:
        details_parts.append(hair)
    
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
        costume_candidates = [costume]
        if costume == "Naked":
            costume_candidates.append("Original")

        sheet_dir = None
        candidates = []
        pattern = f"sheet_{emotion}_*(\\d+)_*\\.png"

        for candidate_costume in costume_candidates:
            candidate_dir = os.path.join(character_dir(character), "Sheets", candidate_costume, emotion)
            if not os.path.isdir(candidate_dir):
                continue

            candidate_files = []
            for fname in os.listdir(candidate_dir):
                m = re.match(pattern, fname)
                if m:
                    idx = int(m.group(1))
                    candidate_files.append((idx, fname))

            if candidate_files:
                sheet_dir = candidate_dir
                candidates = candidate_files
                break

        best_path = None
        if sheet_dir and candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_name = candidates[-1]
            best_path = os.path.join(sheet_dir, best_name)
        if not best_path:
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            for candidate_costume in costume_candidates:
                sprite_root = os.path.join(character_dir(character), "Sprites", candidate_costume)
                if not os.path.isdir(sprite_root):
                    continue
                neutral_files = []
                for neutral_name in ("Neutral", "neutral"):
                    neutral_root = os.path.join(sprite_root, neutral_name)
                    if not os.path.isdir(neutral_root):
                        continue
                    for root, _dirs, filenames in os.walk(neutral_root):
                        neutral_files.extend(
                            os.path.join(root, filename)
                            for filename in filenames
                            if os.path.splitext(filename)[1].lower() in image_exts
                        )
                if neutral_files:
                    best_path = max(neutral_files, key=lambda path: (os.path.getmtime(path), path))
                    print(f"[VNCCS Utils] Using neutral sprite fallback for sheet load: {best_path}")
                    break

                sprite_files = []
                for root, _dirs, filenames in os.walk(sprite_root):
                    parts = set(os.path.normpath(root).split(os.sep))
                    if "Neutral" in parts or "neutral" in parts:
                        continue
                    sprite_files.extend(
                        os.path.join(root, filename)
                        for filename in filenames
                        if os.path.splitext(filename)[1].lower() in image_exts
                    )
                if sprite_files:
                    best_path = max(sprite_files, key=lambda path: (os.path.getmtime(path), path))
                    print(f"[VNCCS Utils] Using sprite fallback for sheet load: {best_path}")
                    break

        if not best_path:
            checked = [
                os.path.join(character_dir(character), "Sheets", candidate_costume, emotion)
                for candidate_costume in costume_candidates
            ] + [
                os.path.join(character_dir(character), "Sprites", candidate_costume)
                for candidate_costume in costume_candidates
            ]
            print(f"[VNCCS Utils] No sheet/sprite fallback found for {character}/{costume}/{emotion}: {checked}")
            if with_mask:
                return None, None
            else:
                return None
        
        img_pil = Image.open(best_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        has_alpha = img_pil.mode == "RGBA" or img_pil.mode == "LA" or img_pil.mode == "P" and "transparency" in img_pil.info
        
        if img_pil.mode != "RGBA":
            img_pil = img_pil.convert("RGBA")
        
        image_np = np.array(img_pil).astype(np.float32) / 255.0
        
        if with_mask:
            if has_alpha:
                img_tensor = torch.from_numpy(image_np[..., :3])[None,]
                # ComfyUI mask convention: 1.0 = inpaint area, 0.0 = keep.
                # PNG alpha: 1.0 = opaque (keep), 0.0 = transparent (inpaint). Invert.
                mask_alpha_channel = 1.0 - image_np[..., 3]
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
        except OSError as exc:
            print(f"[VNCCS Utils] Failed to scan Sheets costumes for '{character_name}': {exc}")

    sprites_dir_path = os.path.join(char_dir, "Sprites")
    if os.path.exists(sprites_dir_path):
        try:
            for item in os.listdir(sprites_dir_path):
                item_path = os.path.join(sprites_dir_path, item)
                if os.path.isdir(item_path) and item not in costumes:
                    costumes.append(item)
        except OSError as exc:
            print(f"[VNCCS Utils] Failed to scan Sprites costumes for '{character_name}': {exc}")
    
    return costumes


create_costume_folders = ensure_costume_structure
