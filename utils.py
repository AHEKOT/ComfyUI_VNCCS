"""VNCCS utilities - common functions for character management."""

import os
import ntpath
import json
import random
import re
import tempfile
import hashlib
import shutil
import threading
from contextlib import contextmanager
from functools import wraps
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Tuple


EMOTIONS = ["neutral"]
MAIN_DIRS = ["Sprites", "Faces", "Sheets"]
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,120}$")
PRIVILEGED_REQUEST_HEADER = "X-VNCCS-CSRF"
PRIVILEGED_REQUEST_VALUE = "1"
_STORAGE_LOCKS = tuple(threading.RLock() for _ in range(64))
COSTUME_TEXT_FIELDS = {"top", "bottom", "shoes", "head", "face", "negative_prompt"}
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


def _model_path_variants(name: str) -> List[str]:
    raw = str(name or "").strip()
    if not raw:
        return []

    variants = []
    for candidate in (raw, raw.replace("\\", "/"), raw.replace("/", "\\")):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _safe_get_model_folder_paths(folder_paths, category: str) -> List[str]:
    try:
        return folder_paths.get_folder_paths(category) or []
    except Exception:
        return []


def _is_under_any_model_folder(path: str, folders: List[str]) -> bool:
    try:
        path_abs = os.path.abspath(normalize_filesystem_path(path))
        for folder in folders:
            folder_abs = os.path.abspath(normalize_filesystem_path(folder))
            if os.path.commonpath([folder_abs, path_abs]) == folder_abs:
                return True
    except Exception:
        return False
    return False


def get_full_path_agnostic(folder_paths, category: str, name: str, require_exists: bool = False):
    """Find a ComfyUI model path regardless of slash style or host OS."""
    folders = _safe_get_model_folder_paths(folder_paths, category)
    first_match = None

    for candidate in _model_path_variants(name):
        try:
            found = folder_paths.get_full_path(category, candidate)
        except Exception:
            found = None
        if found:
            if os.path.exists(found):
                return found
            if first_match is None:
                first_match = found

        for folder in folders:
            joined = os.path.join(folder, normalize_filesystem_path(candidate))
            if os.path.exists(joined):
                return joined
            if first_match is None:
                first_match = joined

        if is_absolute_path_any_os(candidate) and _is_under_any_model_folder(candidate, folders):
            normalized_candidate = normalize_filesystem_path(candidate)
            if os.path.exists(normalized_candidate):
                return normalized_candidate
            if first_match is None:
                first_match = normalized_candidate

    return None if require_exists else first_match


def basename_agnostic(path: str) -> str:
    """Return a basename for either POSIX or Windows-style paths."""
    normalized = str(path or "").rstrip("\\/")
    if not normalized:
        return ""
    return normalized.replace("\\", "/").rsplit("/", 1)[-1]


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
    if os.path.commonpath([os.path.realpath(base_abs), os.path.realpath(target)]) != os.path.realpath(base_abs):
        raise ValueError("path resolves outside allowed directory")
    return target


def file_fingerprint(path: str) -> str:
    """Hash disk inputs so unchanged widget values cannot reuse stale files."""
    digest = hashlib.sha256()
    digest.update(os.path.abspath(path).encode("utf-8"))
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        digest.update(b"missing")
    except OSError as error:
        digest.update(f"unreadable:{error.errno}".encode("ascii"))
    return digest.hexdigest()


@contextmanager
def character_storage_lock(path):
    """Coordinate publication and maintenance with a bounded lock registry."""
    key = os.path.normcase(os.path.realpath(path))
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(_STORAGE_LOCKS)
    with _STORAGE_LOCKS[index]:
        yield


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


@contextmanager
def staged_image_batch(target_dir, *, version_existing=True, lock_root=None):
    """Publish a complete image directory and restore the old one on failure."""
    parent = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent, exist_ok=True)
    with character_storage_lock(lock_root or target_dir):
        stage = tempfile.mkdtemp(prefix=".vnccs-sprites-", dir=parent)
        backup = None
        try:
            yield stage
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            existing = os.listdir(target_dir) if os.path.isdir(target_dir) else []
            old_images = [name for name in existing if os.path.isfile(os.path.join(target_dir, name))
                          and os.path.splitext(name)[1].lower() in image_exts]
            version_dir = None
            if version_existing and old_images:
                number = 1
                while f"V{number}" in existing:
                    number += 1
                version_dir = os.path.join(stage, f"V{number}")
                os.makedirs(version_dir)
            for name in existing:
                source = os.path.join(target_dir, name)
                destination = os.path.join(stage, name)
                if name in old_images and version_dir:
                    destination = os.path.join(version_dir, name)
                elif os.path.exists(destination):
                    continue
                if os.path.islink(source):
                    os.symlink(os.readlink(source), destination)
                elif os.path.isdir(source):
                    shutil.copytree(source, destination, copy_function=_link_or_copy, symlinks=True)
                else:
                    _link_or_copy(source, destination)
            if os.path.exists(target_dir):
                backup = tempfile.mkdtemp(prefix=".vnccs-rollback-", dir=parent)
                os.rmdir(backup)
                os.replace(target_dir, backup)
            try:
                os.replace(stage, target_dir)
            except BaseException:
                if backup:
                    os.replace(backup, target_dir)
                    backup = None
                raise
            if backup:
                try:
                    shutil.rmtree(backup)
                    backup = None
                except OSError as error:
                    print(f"[VNCCS Storage] Sprites published; could not remove rollback directory '{backup}': {error}")
        finally:
            if os.path.exists(stage):
                shutil.rmtree(stage)
            # A failed rollback leaves its backup in place for recovery.


def validate_costume_info(info):
    """Validate supported costume fields while preserving extension metadata."""
    if not isinstance(info, dict):
        raise ValueError("Costume info must be an object")
    if any(not isinstance(key, str) for key in info):
        raise ValueError("Costume info keys must be strings")
    for field in COSTUME_TEXT_FIELDS:
        if field in info and not isinstance(info[field], str):
            raise ValueError(f"Costume field '{field}' must be a string")
    return dict(info)


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
    sec_fetch_site = (request.headers.get("Sec-Fetch-Site") or "").lower()
    has_marker = request.headers.get(PRIVILEGED_REQUEST_HEADER) == PRIVILEGED_REQUEST_VALUE
    if sec_fetch_site == "cross-site":
        raise ValueError("cross-site privileged request rejected")
    # Fetch Metadata describes the browser-facing origin, before a reverse proxy
    # rewrites Host. Web pages cannot forge Sec-Fetch-* headers. Require our
    # non-simple header as well; never trust arbitrary forwarded host headers.
    proxy_same_origin = sec_fetch_site == "same-origin" and has_marker
    same_origin = False
    # Origin is authoritative; Referer is a fallback, not a second origin check.
    # Launchers and proxies can rewrite one header without rewriting the other.
    raw = request.headers.get("Origin")
    if raw is None:
        raw = request.headers.get("Referer")
    if raw is not None:
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("invalid privileged request origin")
        if host and parsed.netloc.lower() == host:
            same_origin = True
        elif host and not proxy_same_origin:
            raise ValueError("cross-origin privileged request rejected")

    if has_marker:
        return

    if same_origin and sec_fetch_site in {"", "same-origin", "same-site", "none"}:
        return

    raise ValueError(f"missing {PRIVILEGED_REQUEST_HEADER} header")


def privileged_route(handler):
    """Apply the same origin check before any expensive or mutating handler."""
    @wraps(handler)
    async def guarded(request):
        try:
            validate_privileged_request(request)
        except ValueError as error:
            from aiohttp import web
            return web.json_response({"error": str(error)}, status=403)
        return await handler(request)
    return guarded


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
    group_paths = [safe_join_under(char_path, main_dir) for main_dir in main_dirs]
    naked_paths = [safe_join_under(char_path, main_dir, "Naked") for main_dir in main_dirs]
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    if not os.path.exists(char_path):
        os.makedirs(char_path)
    
    for main_dir_path, naked_path in zip(group_paths, naked_paths):
        if not os.path.exists(main_dir_path):
            os.makedirs(main_dir_path)
        
        if not os.path.exists(naked_path):
            os.makedirs(naked_path)


def ensure_costume_structure(name: str, costume: str, emotions: List[str] = None) -> None:
    """Create costume directory structure.
    
    Note: Emotion folders are NOT created here. They are created on-demand
    when images are actually saved.
    """
    char_path = character_dir(name)
    costume = ensure_safe_name(costume, "costume")
    group_paths = [safe_join_under(char_path, main_dir) for main_dir in MAIN_DIRS]
    costume_paths = [safe_join_under(char_path, main_dir, costume) for main_dir in MAIN_DIRS]
    
    for main_dir_path, costume_path in zip(group_paths, costume_paths):
        if not os.path.exists(main_dir_path):
            os.makedirs(main_dir_path)
        
        if not os.path.exists(costume_path):
            os.makedirs(costume_path)


def list_characters() -> List[str]:
    """Get list of existing characters from the current VNCCS character root."""
    base_path = base_output_dir()
    try:
        return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    except Exception as exc:
        print(f"[VNCCS] list_characters: failed to list new character path '{base_path}': {exc}")
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


def load_config(character_name: str, *, strict=False) -> Optional[Dict[str, Any]]:
    """Load character configuration."""
    config_file = config_path(character_name)
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if strict and (not isinstance(data, dict) or any(
            key in data and not isinstance(data[key], dict) for key in ("character_info", "costumes")
        )):
            raise ValueError("Invalid character configuration structure")
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        if strict:
            raise OSError(f"Cannot read configuration for '{character_name}': {e}") from e
        print(f"[VNCCS Utils] Error loading configuration {character_name}: {e}")
    return None


@contextmanager
def atomic_output_path(path):
    """Publish a completed file without truncating the previous version."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_config(character_name: str, data: Dict[str, Any]) -> str:
    """Save character configuration."""
    char_dir = character_dir(character_name)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir, exist_ok=True)
    
    config_file = config_path(character_name)
    try:
        with atomic_output_path(config_file) as temporary:
            with open(temporary, 'w', encoding='utf-8') as f:
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
    
    if not isinstance(config, dict):
        raise ValueError("Character config must be an object")
    char_info = config.get("character_info", {})
    if not isinstance(char_info, dict):
        raise ValueError("Character profile must be an object")
    
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
        eyes = str(char_info["eyes"])
        details_parts.append(eyes if re.search(r"\beyes\b", eyes, re.IGNORECASE) else f"{eyes} eyes")
    
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
    config = load_config(character_name, strict=True)
    if not config:
        return {}
    costumes = config.get("costumes", {})
    return validate_costume_info(costumes.get(costume_name, {}))


def save_costume_info(character_name: str, costume_name: str, costume_data: Dict[str, Any]) -> bool:
    """Save character costume info."""
    costume_data = validate_costume_info(costume_data)
    with character_storage_lock(character_dir(character_name)):
        config = load_config(character_name, strict=True)
        if not config:
            config = {"character_info": {}, "costumes": {}}
        if "costumes" not in config:
            config["costumes"] = {}
        config["costumes"][costume_name] = costume_data
        return save_config(character_name, config) != ""


def delete_costume(character_name: str, costume_name: str) -> Optional[str]:
    """Remove an editable costume, restoring its files if publication fails."""
    character_name = ensure_safe_name(character_name, "character")
    costume_name = ensure_safe_name(costume_name, "costume")
    if costume_name.casefold() in {"naked", "original"}:
        raise ValueError("Cannot delete base sprite set.")
    root = character_dir(character_name)
    with character_storage_lock(root):
        config = load_config(character_name, strict=True)
        costumes = config.get("costumes", {}) if config is not None else {}
        cache_name = costume_name.replace(" ", "_")
        # Legacy preview names can be shared by costumes such as "Red Dress"
        # and "Red_Dress". Keep that cache while another costume uses it.
        shared_cache = any(
            name != costume_name and name.replace(" ", "_").casefold() == cache_name.casefold()
            for name in list_costumes(character_name)
        )
        paths = [safe_join_under(root, group, costume_name) for group in MAIN_DIRS]
        if not shared_cache:
            paths.extend([
                safe_join_under(root, "cache", f"preview_{cache_name}.png"),
                safe_join_under(root, "cache", f"preview_info_{cache_name}.json"),
            ])
        # Reject aliases even when their resolved destination is inside the root.
        # A costume symlink must never let deletion touch a protected base set.
        if any(os.path.islink(path) for path in [root, config_path(character_name), *paths,
                                                *(os.path.dirname(path) for path in paths)]):
            raise ValueError("Cannot delete a costume through symbolic links.")
        existing = [path for path in paths if os.path.lexists(path)]
        if costume_name not in costumes and not existing:
            raise FileNotFoundError(f"Costume '{costume_name}' was not found.")
        staging = tempfile.mkdtemp(prefix=".vnccs-delete-", dir=root)
        moved = []
        try:
            for index, path in enumerate(existing):
                backup = os.path.join(staging, str(index))
                os.replace(path, backup)
                moved.append((path, backup))
            if costume_name in costumes:
                del costumes[costume_name]
                if not save_config(character_name, config):
                    raise OSError(f"Could not save configuration after deleting '{costume_name}'.")
        except BaseException:
            for path, backup in reversed(moved):
                os.replace(backup, path)
            os.rmdir(staging)
            raise
        try:
            shutil.rmtree(staging)
        except OSError as error:
            warning = (
                f"Costume '{costume_name}' was deleted. Some removed files remain at '{staging}': {error}. "
                "Check storage permissions, then remove that temporary directory."
            )
            print(f"[VNCCS] {warning}")
            return warning
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


def create_initial_character(
    existing_character: str,
    background_color: str = "green",
    aesthetics: str = "",
    nsfw: bool = False,
    sex: str = "female",
    age: int = 18,
    race: str = "",
    eyes: str = "",
    hair: str = "",
    face: str = "",
    body: str = "",
    skin_color: str = "",
    additional_details: str = "",
    seed: int = 0,
    negative_prompt: str = "",
    lora_prompt: str = ""
) -> tuple[str, int, str, float, str, str, str]:

    character_name = existing_character

    ensure_character_structure(character_name, EMOTIONS, MAIN_DIRS)

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
    hair = normalize_hair_tags(hair)
    if hair:
        positive_prompt += f", ({hair}:1.0)"
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

    config = load_config(character_name, strict=True) or {
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

    if not save_config(character_name, config):
        raise OSError(f"Could not save character configuration for '{character_name}'. Check storage permissions and free space.")

    face_details = build_face_details(config["character_info"])
    face_details += f", (expressionless:1.0)"

    return positive_prompt, seed, final_negative_prompt, age_lora_strength, sheets_path, faces_path, face_details
