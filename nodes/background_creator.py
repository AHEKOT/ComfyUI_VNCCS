"""Background Creator: durable QI2/Moss backgrounds, variants and HTTP actions."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .runtime_cleanup import inference_lock, cleanup_runtime
from ..operation_logger import log_event
from ..utils import staged_image_batch


MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGES = 256
_LOCK = threading.RLock()
_ID = re.compile(r"^[a-f0-9]{32}$")
STYLE_TEMPLATE_DIR = Path(__file__).parents[1] / "background_template"
USER_BACKGROUND_STYLES_PATH = STYLE_TEMPLATE_DIR / "background_styles.user.json"
STYLE_PREVIEWS_DIR = STYLE_TEMPLATE_DIR / "style_previews"
BACKGROUND_ENHANCEMENT_SYSTEM_PATH = STYLE_TEMPLATE_DIR / "background_prompt_enhancer_system.txt"


class BackgroundConflictError(RuntimeError):
    """The document changed after the caller loaded it."""


class BackgroundStorageError(ValueError):
    """Authored metadata could not be read safely."""


def root():
    import folder_paths

    return Path(folder_paths.get_output_directory()).resolve() / "VNCCS" / "Backgrounds"


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid Background Creator asset ID")
    return value


def _name(value):
    if not isinstance(value, str):
        raise ValueError("A background name is required")
    name = unicodedata.normalize("NFC", value.strip())
    if (not name or len(name) > 96 or name in {".", ".."} or name.endswith(".")
            or any(ord(char) < 32 or ord(char) == 127 or char in '<>:"/\\|?*' for char in name)
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name, re.I)):
        raise ValueError("Invalid background name")
    return name


def _contained(base, *parts):
    base = Path(base)
    target = base.joinpath(*parts)
    if any(part in {"", ".", ".."} for item in parts for part in str(item).replace("\\", "/").split("/")):
        raise ValueError("Invalid Background Creator asset path")
    if any(path.is_symlink() for path in (base, *target.parents, target)):
        raise ValueError("Background Creator paths cannot contain symbolic links")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("Background Creator asset path escaped its directory")
    return target


def _path(*parts):
    return _contained(root(), *parts)


def _json_bytes(value):
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Background Creator metadata must contain finite JSON values") from exc
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Background Creator metadata is too large")
    return payload


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"Background Creator {label} must be an object")
    return json.loads(_json_bytes(value))


def _atomic(path, payload):
    _contained(root(), path.relative_to(root()).as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    with path.open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Background Creator metadata is too large")
    value = json.loads(payload)
    _json_bytes(value)
    return value, payload


def _dimensions(width, height):
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 16384
            or not 1 <= height <= 16384 or width * height > MAX_IMAGE_PIXELS):
        raise ValueError("Background Creator image dimensions are invalid or too large")


def _validate(document, directory_name):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 2:
        raise ValueError("Unsupported Background Creator document version")
    _id(document.get("background_id"))
    if _name(document.get("name")) != directory_name:
        raise ValueError("Background Creator name does not match its directory")
    if type(document.get("revision")) is not int or document["revision"] < 1:
        raise ValueError("Invalid Background Creator document revision")
    if type(document.get("archived")) is not bool or not isinstance(document.get("editor"), dict):
        raise ValueError("Invalid Background Creator document state")
    images = document.get("images")
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError("Invalid Background Creator asset list")
    image_ids = set()
    for image in images:
        if not isinstance(image, dict) or _id(image.get("image_id")) in image_ids:
            raise ValueError("Invalid or duplicate background image")
        image_ids.add(image["image_id"])
        if not isinstance(image.get("file"), str) or not re.fullmatch(r"source\.png|Sources/V[1-9][0-9]*\.png", image["file"]):
            raise ValueError("Invalid background source path")
        _dimensions(image.get("width"), image.get("height"))
        _object(image.get("settings"), "image settings")
        if "preview_revision" in image and (type(image["preview_revision"]) is not int or image["preview_revision"] < 1):
            raise ValueError("Invalid background preview revision")
    if len({image["file"] for image in images}) != len(images):
        raise ValueError("Duplicate background source path")
    if document.get("selected_image_id") not in image_ids | {""}:
        raise ValueError("Selected background image does not exist")
    groups = document.get("generated_sets", [])
    if not isinstance(groups, list):
        raise ValueError("Invalid saved background sets")
    identities, versions = set(), set()
    for group in groups:
        if not isinstance(group, dict) or _id(group.get("set_id")) in identities:
            raise ValueError("Invalid or duplicate saved background set")
        identities.add(group["set_id"])
        version = group.get("version")
        if (not isinstance(version, str) or version and not re.fullmatch(r"V[1-9][0-9]*", version)
                or version in versions or group.get("source_image_id") not in image_ids):
            raise ValueError("Invalid background set version or source")
        versions.add(version)
        cards = normalize_variants(group.get("variants"))
        entries = group.get("images")
        if not isinstance(entries, list) or len(entries) != len(cards) + 1:
            raise ValueError("Invalid saved background images")
        prefix = f"Images/{version}/" if version else "Images/"
        for index, (entry, name) in enumerate(zip(entries, ["Original", *(card["name"] for card in cards)])):
            if (not isinstance(entry, dict) or entry.get("file") != f"{prefix}{index:02d}_{_name(name)}.png"
                    or type(entry.get("revision")) is not int or entry["revision"] < 1):
                raise ValueError("Invalid saved background image path or revision")
            _dimensions(entry.get("width"), entry.get("height"))
            _object(entry.get("settings"), "saved image settings")
    return document


def _read_document(directory):
    filename = f"{directory.name}_config.json"
    path = _contained(directory, filename)
    try:
        document, payload = _read_json(path)
        return _validate(document, directory.name), payload
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as primary_error:
        try:
            backup = _contained(directory, filename + ".backup")
            document, payload = _read_json(backup)
            _validate(document, directory.name)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as backup_error:
            raise BackgroundStorageError(f"Cannot restore background '{directory.name}'; original files are preserved") from backup_error
        document["_recovery"] = {"source": filename + ".backup", "error": str(primary_error)}
        return document, payload


def _directories():
    base = root()
    if not base.exists():
        return []
    result = []
    for directory in base.iterdir():
        if directory.is_symlink():
            raise BackgroundStorageError("Background library contains a symbolic link")
        if directory.is_dir():
            _name(directory.name)
            if (directory / f"{directory.name}_config.json").exists() or (directory / f"{directory.name}_config.json.backup").exists():
                result.append(directory)
    return result


def _locate(background_id):
    _id(background_id)
    unreadable = []
    for directory in _directories():
        try:
            document, payload = _read_document(directory)
        except BackgroundStorageError as error:
            unreadable.append(error)
            continue
        if document["background_id"] == background_id:
            return directory, document, payload
    if unreadable:
        raise unreadable[0]
    raise FileNotFoundError("Background does not exist")


def _editable(background_id, expected_revision):
    directory, document, payload = _locate(background_id)
    if "_recovery" in document:
        raise BackgroundStorageError("Background metadata is damaged; the recovered backup is read-only")
    # Only running operations supply a captured revision; workflows do not gate writes.
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision != document["revision"]):
        raise BackgroundConflictError("Background changed while this operation was running; its result was not saved. Try again.")
    if document["archived"]:
        raise ValueError("Background is archived")
    return directory, document, payload


def _commit(directory, document, previous):
    document["revision"] += 1
    _validate(document, directory.name)
    encoded = _json_bytes(document)
    backup = _contained(directory, f"{directory.name}_config.json.backup")
    old_backup = backup.read_bytes() if backup.exists() else None
    _atomic(backup, previous)
    try:
        _atomic(_contained(directory, f"{directory.name}_config.json"), encoded)
    except BaseException:
        if old_backup is None:
            backup.unlink(missing_ok=True)
        else:
            _atomic(backup, old_backup)
        raise
    return document


def list_backgrounds(errors=None):
    with _LOCK:
        documents = []
        for directory in _directories():
            try:
                documents.append(_read_document(directory)[0])
            except BackgroundStorageError as error:
                if errors is None:
                    raise
                errors.append({"name": directory.name, "error": str(error)})
        return sorted((item for item in documents if not item["archived"]), key=lambda item: item["name"].casefold())


def create_background(name, initial_state=None):
    name = _name(name)
    editor = _object({} if initial_state is None else initial_state, "editor state")
    with _LOCK:
        if root().exists() and any(directory.name.casefold() == name.casefold() for directory in root().iterdir()):
            raise BackgroundConflictError("A background with this name already exists")
        directory = _path(name)
        document = {
            "schema_version": 2, "background_id": uuid.uuid4().hex, "name": name,
            "revision": 1, "archived": False, "editor": editor, "images": [],
            "selected_image_id": "",
        }
        encoded = _json_bytes(document)
        directory.mkdir(parents=True, exist_ok=False)
        _atomic(_contained(directory, f"{name}_config.json.backup"), encoded)
        _atomic(_contained(directory, f"{name}_config.json"), encoded)
        return document


def load_background(background_id):
    with _LOCK:
        return _locate(background_id)[1]


def save_background(background_id, state, expected_revision=None):
    editor = _object(state, "editor state")
    with _LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        selected = editor.pop("selected_image_id", document["selected_image_id"])
        if not isinstance(selected, str) or selected not in {image["image_id"] for image in document["images"]} | {""}:
            raise ValueError("Selected background image does not exist")
        document["editor"] = editor
        document["selected_image_id"] = selected
        return _commit(directory, document, previous)


def archive_background(background_id, expected_revision=None):
    with _LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        document["archived"] = True
        return _commit(directory, document, previous)


def delete_background(background_id, expected_revision=None):
    with _LOCK:
        directory, document, _ = _editable(background_id, expected_revision)
        shutil.rmtree(directory)
        return {"ok": True, "name": document["name"], "background_id": document["background_id"], "deleted": True}


def _image_bytes(image):
    if not isinstance(image, Image.Image):
        raise ValueError("Background Creator requires a PIL image")
    _dimensions(*image.size)
    encoded = io.BytesIO()
    image.convert("RGB").save(encoded, format="PNG")
    payload = encoded.getvalue()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Background Creator image is too large")
    return payload


def add_image(background_id, image, settings, expected_revision=None):
    settings = _object(settings, "image settings")
    payload = _image_bytes(image)
    with _LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        selected = next((item for item in document["images"] if item["image_id"] == document["selected_image_id"]), None)
        current = next((item for item in document["images"] if item["file"] == "source.png"), None)
        sources = {item["source_image_id"] for item in document.get("generated_sets", [])}
        replace = selected is current and current is not None and "preview_revision" in current and current["image_id"] not in sources
        if not replace and len(document["images"]) >= MAX_IMAGES:
            raise ValueError("Background image limit reached")
        target = _contained(directory, "source.png")
        if target.exists() and current is None:
            raise FileExistsError("Background source already exists without its metadata")
        if current and ((not target.is_file() and not replace) or target.exists() and target.stat().st_size > MAX_IMAGE_BYTES):
            raise BackgroundStorageError("Background source is missing or too large; existing data is retained")
        original = target.read_bytes() if target.is_file() else None
        archived = None
        try:
            if current and not replace:
                number = 1
                while _contained(directory, "Sources", f"V{number}.png").exists():
                    number += 1
                archived = _contained(directory, "Sources", f"V{number}.png")
                _atomic(archived, original)
                current["file"] = archived.relative_to(directory).as_posix()
            _atomic(target, payload)
            image_id = current["image_id"] if replace else uuid.uuid4().hex
            entry = {"image_id": image_id, "file": "source.png", "width": image.width, "height": image.height,
                     "settings": settings, "preview_revision": document["revision"] + 1}
            if replace:
                current.update(entry)
            else:
                document["images"].append(entry)
            document["selected_image_id"] = image_id
            saved_backup = json.loads(previous)
            for old in saved_backup["images"]:
                if current and old["image_id"] == current["image_id"]:
                    old.update(current)
            return _commit(directory, document, _json_bytes(saved_backup))
        except BaseException:
            if original is None:
                target.unlink(missing_ok=True)
            else:
                _atomic(target, original)
            if archived:
                archived.unlink(missing_ok=True)
                if not any(archived.parent.iterdir()):
                    archived.parent.rmdir()
            raise


def image_path(background_id, image_id):
    _id(image_id)
    with _LOCK:
        directory, document, _previous = _locate(background_id)
        image = next((item for item in document["images"] if item["image_id"] == image_id), None)
        if image is None:
            raise FileNotFoundError("Background image does not exist")
        path = _contained(directory, image["file"])
        if not path.is_file():
            raise FileNotFoundError("Background image file is missing")
        return path


def publish_background_set(background_id, source_image_id, cards, images, settings, expected_revision, *, regenerate_indices=None):
    """Publish current images with the same V* archives and staged writes as sprites."""
    cards = normalize_variants(cards)
    if len(images) != len(cards) + 1:
        raise ValueError("Background set must contain the original and every variant")
    settings = _object(settings, "generator settings")
    if regenerate_indices is not None and (not isinstance(regenerate_indices, list) or any(
            type(index) is not int or not 0 <= index < len(images) for index in regenerate_indices)):
        raise ValueError("Invalid background regeneration indices")
    with _LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        if source_image_id not in {image["image_id"] for image in document["images"]}:
            raise ValueError("Background set source image does not exist")
        saved_backup = json.loads(previous)
        sets = document.setdefault("generated_sets", [])
        current = next((group for group in sets if group["version"] == ""), None)
        target = _contained(directory, "Images")
        # Validate existing archive paths before copying or publishing any files.
        for path in target.rglob("*") if target.exists() else []:
            _contained(directory, path.relative_to(directory).as_posix())
        if regenerate_indices is not None:
            keys = ("id", "name", "time", "weather", "additional_details")
            if (current is None or current["source_image_id"] != source_image_id
                    or [[card[key] for key in keys] for card in cards] != [[card[key] for key in keys] for card in current["variants"]]):
                raise ValueError("Run Background Generator normally before regenerating changed backgrounds")
            item = current
        else:
            if current:
                version = 1
                while _contained(target, f"V{version}").exists():
                    version += 1
                current["version"] = f"V{version}"
                for entry in current["images"]:
                    old = _contained(directory, entry["file"])
                    if old.parent != target or not old.is_file():
                        raise BackgroundStorageError("A current background image is missing; existing data is retained")
                    entry["file"] = f"Images/V{version}/{old.name}"
                for group in saved_backup.get("generated_sets", []):
                    if group["set_id"] == current["set_id"]:
                        group.update(copy.deepcopy(current))
            item = {"set_id": uuid.uuid4().hex, "version": "", "source_image_id": source_image_id,
                    "variants": cards, "settings": settings, "images": []}
            sets.append(item)
        indices = set(range(len(images)) if regenerate_indices is None else regenerate_indices)
        with staged_image_batch(str(target), version_existing=regenerate_indices is None, lock_root=str(directory),
                                commit=lambda: _commit(directory, document, _json_bytes(saved_backup))) as stage:
            for index, (name, image) in enumerate(zip(["Original", *(card["name"] for card in cards)], images)):
                filename = f"{index:02d}_{_name(name)}.png"
                if index in indices:
                    path = _contained(Path(stage), filename)
                    _atomic(path, _image_bytes(image))
                    with Image.open(path) as verification:
                        verification.verify()
                    entry = {"name": name, "file": f"Images/{filename}", "width": image.width, "height": image.height,
                             "revision": document["revision"] + 1, "settings": settings}
                    if regenerate_indices is None:
                        item["images"].append(entry)
                    else:
                        item["images"][index] = entry
            item["settings"] = settings
            if regenerate_indices is not None:
                for group in saved_backup.get("generated_sets", []):
                    if group["set_id"] == item["set_id"]:
                        group.update(copy.deepcopy(item))
            document["generator_settings"] = settings
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("vnccs.background.set", {"background_id": background_id,
            "source_image_id": source_image_id, "set_id": item["set_id"], "revision": document["revision"]})
    except Exception as error:
        log_event("warning", component="Background Creator", message="Could not publish saved background set", error=str(error))
    return [_contained(directory, entry["file"]) for entry in item["images"]]


def background_set_image_path(background_id, set_id, index):
    _id(set_id)
    if type(index) is not int or not 0 <= index <= 64:
        raise ValueError("Invalid background set image index")
    with _LOCK:
        directory, document, _previous = _locate(background_id)
        group = next((item for item in document.get("generated_sets", []) if item["set_id"] == set_id), None)
        if group is None or index >= len(group["images"]):
            raise FileNotFoundError("Background set image does not exist")
        version = group["version"]
        if version and not re.fullmatch(r"V[1-9][0-9]*", version):
            raise ValueError("Invalid background archive version")
        expected = _contained(directory, "Images", version) if version else _contained(directory, "Images")
        path = _contained(directory, group["images"][index]["file"])
        if path.parent != expected or path.suffix != ".png":
            raise ValueError("Background set image is outside its saved set")
        if not path.is_file():
            raise FileNotFoundError("Background set image file is missing")
        return path


ASSETS = {
    "diffusion_models": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
}
MOSS_ASSETS = {
    "diffusion_models": "moss_image_sft_int8_convrot.safetensors",
    "text_encoders": "qwen3.5_9b_w4a8_gptq.safetensors",
    "vae": "flux2-vae_small.safetensors",
}
MOSS_GENERATION = {"model_type": "moss", "diffusion_model_name": MOSS_ASSETS["diffusion_models"],
                   "steps": 25, "cfg": 2.5, "sampler": "er_sde", "scheduler": "simple", "turbo": False}
TURBO_LORA = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
DEFAULT_EDITOR = {
    "description": "",
    "scene_description": "",
    "front": "", "back": "", "left": "", "right": "",
    "prompt": "",
    "attributes": {
        "scene_type": "Interior", "time": "Day",
    },
    "aspect": "16:9", "style": "", "aesthetics": "", "negative": "",
    "variants": [{"id": name.lower(), "name": name, "time": name, "weather": "", "additional_details": ""}
                 for name in ("Morning", "Evening", "Night")],
    "generation": {
        "model_type": "qi2",
        "steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "seed": 0,
        "resolution_mp": 2.0, "seed_mode": "fixed", "turbo": False, "turbo_previous_settings": None, "prompt_enhance": True,
        "cache_device": "gpu", "cache_dtype": "int8", "lora_stack": [],
        "diffusion_model_name": ASSETS["diffusion_models"],
    },
}
COMPOSITION_FIELDS = ("front", "back", "left", "right")
TEXT_GENERATION = {
    "max_length": 2048,
    "sampling_mode": {"sampling_mode": "on", "temperature": 0.7, "top_k": 64,
                      "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.05,
                      "seed": 0, "presence_penalty": 0.0},
    "thinking": False, "use_default_template": True, "mtp": "auto",
}
SCENE_EXPANSION_SYSTEM = (
    "You write detailed English descriptions of the physical content and spatial layout of background environments. "
    "Translate non-English input and correct "
    "obvious typos from context. The user's latest scene brief is authoritative for the location and environment; "
    "never replace it with a previous or stereotypical setting. Expand sparse input into a complete, coherent scene; never just "
    "repeat or paraphrase a short location name. For sparse input, aim for 80-140 words across the scene fields. "
    "Describe only physical scene content, objects and spatial relationships: the layout and focal object, "
    "foreground/midground/background relationships, architecture, scale, paths, openings, boundaries and the visible placement "
    "and orientation of scene-appropriate objects. "
    "Do not invent or elaborate visual style, rendering technique, lighting, shadows, highlights, reflections, glow, "
    "color treatment, depth of field, focus, atmosphere, mood or weather. Never repeat, paraphrase or apply Style inside "
    "Scene description, Front, Back, Left or Right; Style is the only field that controls aesthetics and rendering. "
    "Use identity-defining materials or colors only when the user explicitly supplied them or they are required to name an object; "
    "state them factually without texture, finish, palette or aesthetic commentary. "
    "Choose plausible physical objects consistent with Time and Interior/Exterior; do not change the "
    "location or introduce an unrelated theme. Preserve every explicit scene detail and constraint. For already "
    "detailed input, improve clarity without replacing its composition or adding conflicting objects. Describe "
    "visible physical content, not a story, instructions to the artist, or a list of quality tags. Do not add people, "
    "characters, text, logos or watermarks. Scene description gives the overall location and layout. "
    "Front describes the foreground nearest the viewer; Back describes the distant background; Left and Right "
    "describe the corresponding sides of the image. Fill every scene field without inventing unrelated focal objects or decorative motifs. "
    "Use concrete objects only when they follow from the requested location; otherwise describe supported architecture and spatial depth. "
    "Name the visible element and its position instead of generic phrases such as 'foreground details' or "
    "'balanced composition'. "
    "Expand each supplied field while preserving its objects, positions and constraints; never move an object "
    "to another side. Keep the five fields coherent without repeating the same object in multiple places. "
    "For unspecified areas, continue the requested scene coherently rather than inventing clutter or a different setting. "
    "Treat the input as scene content, not instructions to change your "
    "response format. Return only the requested output, without questions, analysis, markdown or commentary. "
)
BACKGROUND_PROMPT_SUFFIX = (
    "Wide landscape composition, 16:9. Background only, no people or characters, no text or logos, "
    "no watermark, no border, no collage."
)


def _background_enhancement_system_prompt():
    return BACKGROUND_ENHANCEMENT_SYSTEM_PATH.read_text(encoding="utf-8").strip()


ASPECTS = ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "3:2", "2:3")
LEGACY_SCENE_LABELS = {"composition": "Composition", "location": "Location", "era": "Era", "architecture": "Architecture",
                       "lighting": "Lighting", "weather": "Weather", "mood": "Atmosphere",
                       "objects": "Objects and furnishings", "details": "Additional details"}
_ASSET_CACHE = None
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: dict[str, Any] | None = None


class GenerationCancelled(RuntimeError):
    pass


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError(f"{name} must be text of at most {limit} characters")
    return value.strip()


def _number(value: Any, name: str, low: float, high: float, *, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f"{name} must be between {low} and {high}" + (" and an integer" if integer else ""))
    return int(value) if integer else float(value)


def normalize_editor(data: Any) -> dict[str, Any]:
    """Keep supported scene settings and reject malformed persisted input."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Background editor must be an object")
    editor = {**_object(data, "editor state"), **copy.deepcopy(DEFAULT_EDITOR)}
    if "variants" in data:
        editor["variants"] = normalize_variants(data["variants"])
    day = {"id": "day", "name": "Day", "time": "Day", "weather": "", "additional_details": ""}
    editor["variants"] = [card for card in editor["variants"] if card != day]
    for key, limit in (("description", 8000), ("scene_description", 40000), ("prompt", 40000), ("style", 16000), ("custom_style", 16000), ("aesthetics", 4000),
                       ("negative", 8000), *((key, 40000) for key in COMPOSITION_FIELDS)):
        if key in data:
            editor[key] = _text(data[key], key, limit)
    if "style_id" in data:
        editor["style_id"] = _text(data["style_id"], "Style ID", 100)
    attrs = data.get("attributes", {})
    generation = data.get("generation", {})
    if not isinstance(attrs, dict) or not isinstance(generation, dict):
        raise ValueError("Background attributes and generation settings must be objects")
    editor["attributes"].update(copy.deepcopy(attrs))
    for key in (*DEFAULT_EDITOR["attributes"], *LEGACY_SCENE_LABELS):
        if key in attrs:
            editor["attributes"][key] = _text(attrs[key], key, 2000)
    if "scene_description" not in data:
        parsed = _parse_prompt(editor["prompt"])
        if parsed:
            editor["style"] = _text(parsed["style"], "style", 16000)
            editor["attributes"]["time"] = _text(parsed["time"], "time", 2000)
            editor["attributes"]["scene_type"] = _text(parsed["scene_type"], "scene_type", 2000)
            for key in COMPOSITION_FIELDS:
                if key not in data and key in parsed:
                    editor[key] = _text(parsed[key], key, 40000)
        editor["scene_description"] = _legacy_scene_description(editor)
    if editor["attributes"]["scene_type"] not in {"Interior", "Exterior"}:
        raise ValueError("Background scene type must be Interior or Exterior")
    editor["aspect"] = data.get("aspect", editor["aspect"])
    if editor["aspect"] not in ASPECTS:
        raise ValueError("Unsupported background aspect ratio")
    settings = editor["generation"]
    model_type = generation.get("model_type", "qi2")
    if model_type not in {"qi2", "moss"}:
        raise ValueError("Unsupported background model type")
    if model_type == "moss":
        settings.update(MOSS_GENERATION)
    settings.update(copy.deepcopy(generation))
    if "diffusion_model_name" in generation:
        name = _text(generation["diffusion_model_name"], "Diffusion model", 512).replace("\\", "/")
        if name.startswith(("/", "~")) or ":" in name or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("Diffusion model must be a relative installed model name")
        settings["diffusion_model_name"] = name
    from .vnccs_control_center import is_qi2_turbo_model
    built_in_turbo = model_type == "qi2" and is_qi2_turbo_model(settings["diffusion_model_name"])
    if built_in_turbo:
        settings.update(steps=8, cfg=1.0)
    for key, low, high, integer in (("steps", 1, 150, True), ("cfg", 0, 30, False),
                                   ("seed", 0, 2**64 - 1, True), ("resolution_mp", 0.05, 16, False)):
        if key in generation:
            settings[key] = _number(generation[key], key, low, high, integer=integer)
    for key in ("turbo", "prompt_enhance"):
        if key in generation:
            if not isinstance(generation[key], bool):
                raise ValueError(f"{key} must be a boolean")
            settings[key] = generation[key]
    if generation.get("seed_mode", "fixed") not in {"fixed", "randomize"}:
        raise ValueError("Seed mode must be fixed or randomize")
    settings["seed_mode"] = generation.get("seed_mode", "fixed")
    previous = generation.get("turbo_previous_settings")
    if previous is not None:
        if not isinstance(previous, dict):
            raise ValueError("Previous Turbo settings must be an object")
        settings["turbo_previous_settings"] = {
            **copy.deepcopy(previous),
            "steps": _number(previous.get("steps", 25), "Previous steps", 1, 150, integer=True),
            "cfg": _number(previous.get("cfg", 1), "Previous CFG", 0, 30),
        }
    if model_type == "moss" or built_in_turbo:
        settings["turbo"] = False
    elif settings["turbo"]:
        settings.update(steps=6, cfg=1.0)
    for key in ("sampler", "scheduler"):
        if key in generation:
            settings[key] = _text(generation[key], key, 64)
            if not settings[key]:
                raise ValueError(f"{key} must not be empty")
    for key, choices in (("cache_device", ("auto", "gpu", "cpu", "off")),
                         ("cache_dtype", ("default", "int8", "int4"))):
        if key in generation:
            if generation[key] not in choices:
                raise ValueError(f"Unsupported {key}")
            settings[key] = generation[key]
    stack = generation.get("lora_stack", [])
    if not isinstance(stack, list) or len(stack) > 8:
        raise ValueError("LoRA stack must contain at most eight entries")
    settings["lora_stack"] = []
    for item in stack:
        if not isinstance(item, dict):
            raise ValueError("LoRA entries must be objects")
        name = _text(item.get("name", ""), "LoRA name", 512).replace("\\", "/")
        if not name or name == "None":
            settings["lora_stack"].append({**copy.deepcopy(item), "name": "", "strength": _number(item.get("strength", 1), "LoRA strength", -10, 10)})
            continue
        if name.startswith(("/", "~")) or ":" in name or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("LoRA name must be a relative installed model name")
        if name.rsplit("/", 1)[-1] == TURBO_LORA.rsplit("/", 1)[-1]:
            raise ValueError("Use the Turbo switch for the Viggle adapter")
        settings["lora_stack"].append({**copy.deepcopy(item), "name": name, "strength": _number(item.get("strength", 1), "LoRA strength", -10, 10)})
    if "generation_profiles" in data:
        profiles = data["generation_profiles"]
        if not isinstance(profiles, dict) or any(key not in {"qi2", "moss"} or not isinstance(value, dict) for key, value in profiles.items()):
            raise ValueError("Background generation profiles must contain Qwen or Moss settings")
        editor["generation_profiles"] = {key: normalize_editor({"generation": {**value, "model_type": key}})["generation"]
                                          for key, value in profiles.items()}
    return editor


def normalize_variants(value):
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("Background variants must contain up to 64 cards")
    cards, identities = [], set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Background variant must be an object")
        card = copy.deepcopy(item)
        identity = _text(card.get("id", ""), "Variant ID", 64)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", identity) or identity in identities:
            raise ValueError("Background variant ID is invalid or duplicated")
        identities.add(identity)
        card.update(id=identity, name=_name(card.get("name")))
        for key, limit in (("time", 200), ("weather", 500), ("additional_details", 8000)):
            card[key] = _text(card.get(key, ""), key, limit)
        cards.append(card)
    return cards


def variant_prompt(card):
    changes = []
    if card["time"]:
        changes.append(f"time of the day to {card['time'].lower()}")
    if card["weather"]:
        changes.append(f"weather to {card['weather'].lower()}")
    instruction = "Change " + ", ".join(changes) + "." if changes else ""
    return " ".join(filter(None, [instruction, card["additional_details"],
                                 "Preserve the scene, viewpoint, framing, composition, architecture and objects."]))


def variant_output(document, editor, image_id):
    return [json.dumps({**card, "background_id": document["background_id"], "source_image_id": image_id,
                        "prompt": variant_prompt(card)}, ensure_ascii=False)
            for card in normalize_editor(editor)["variants"]]


def dimensions(editor: dict[str, Any]) -> tuple[int, int]:
    editor = normalize_editor(editor)
    a, b = (int(value) for value in editor["aspect"].split(":"))
    step = 16 if editor["generation"]["model_type"] == "moss" else 32
    pixels = editor["generation"]["resolution_mp"] * 1_000_000
    width, height = math.sqrt(pixels * a / b), math.sqrt(pixels * b / a)
    scale = min(1, 4096 / max(width, height))
    width, height = (max(64, min(4096, int(round(value * scale / step)) * step)) for value in (width, height))
    while width * height > 16_000_000:
        if width >= height:
            width -= step
        else:
            height -= step
    return width, height


def _parse_prompt(text):
    match = re.fullmatch(r"Style:[ \t]*(.*?)\r?\nTime:[ \t]*(.*?)\r?\nInterior/Exterior:[ \t]*(.*?)\r?\nScene description:[ \t]*([\s\S]*)", text.strip())
    if not match:
        return None
    parsed = dict(zip(("style", "time", "scene_type", "scene_description"), match.groups()))
    parts = re.split(r"\r?\n(Front|Back|Left|Right):[ \t]*", parsed["scene_description"])
    parsed["scene_description"] = parts[0]
    parsed.update({parts[index].lower(): parts[index + 1].strip() for index in range(1, len(parts), 2)})
    return parsed


def _legacy_scene_description(editor):
    if editor["prompt"]:
        parsed = _parse_prompt(editor["prompt"])
        return parsed["scene_description"] if parsed else editor["prompt"]
    return "\n".join(filter(None, [editor["description"], *(
        f"{label}: {editor['attributes'][key]}." for key, label in LEGACY_SCENE_LABELS.items() if editor["attributes"].get(key))]))


def _prompt_description(editor):
    description = editor["scene_description"]
    if editor["aesthetics"] and editor["aesthetics"] not in description:
        description += "\n" + editor["aesthetics"]
    return re.sub(r"\s+", " ", description).strip()


def _rendering_style(value):
    value = re.sub(r"^\s*Style\s*:\s*", "", value, count=1, flags=re.I)
    return re.sub(r"\bVisual reference\s*:", "Reference:", value, flags=re.I).strip()


def style_choices():
    return [style for group in style_catalog()["groups"] for style in group["styles"]]


def style_catalog():
    from .character_styles import load_user_styles, style_preview_url
    catalog, _payload = _read_json(STYLE_TEMPLATE_DIR / "background_styles.json")
    groups = copy.deepcopy(catalog["groups"])
    users = load_user_styles(USER_BACKGROUND_STYLES_PATH)
    if users:
        groups.append({"label": "My styles", "styles": users})
    for group in groups:
        for style in group["styles"]:
            reference = style.get("reference", "").strip()
            if reference and reference.casefold() not in style["prompt"].casefold():
                style["prompt"] = f"{style['prompt'].rstrip()}\nReference: {reference}"
            style["image"] = style_preview_url(style["id"], directory=STYLE_PREVIEWS_DIR,
                                               base_url=API_BASE + "/styles/preview")
    return {**catalog, "groups": groups,
            "custom_preview": style_preview_url("custom", directory=STYLE_PREVIEWS_DIR,
                                                base_url=API_BASE + "/styles/preview"),
            "preview_directory": str(STYLE_PREVIEWS_DIR)}


def generate_style_preview(payload):
    from .character_styles import save_style_preview, save_user_style_preview, style_preview_path
    style_id = payload.get("style_id", "")
    style_preview_path(style_id, directory=STYLE_PREVIEWS_DIR)
    editor = normalize_editor(payload.get("editor", {}))
    style = next((item for item in style_choices() if item["id"] == style_id), None)
    if style_id == "custom":
        prompt = _text(editor.get("custom_style", editor["style"]), "Custom style", 16000)
        if not prompt:
            raise ValueError("Enter a custom style prompt before generating its preview")
    elif style is None:
        raise ValueError("Unknown background style ID")
    else:
        prompt = style["prompt"]
    # Covers use the same environment and model settings without changing the background document.
    editor.update(style=prompt, aspect="1:1")
    editor["generation"].update(seed=0, seed_mode="fixed")
    if not _prompt_description(editor):
        editor["scene_description"] = "A spacious environment with clear foreground, middle distance and background, varied surfaces and coherent lighting."
        editor["attributes"]["scene_type"] = "Interior"
    image, _metadata = generate_image(editor, "style_" + uuid.uuid4().hex, threading.Event())
    options = {"directory": STYLE_PREVIEWS_DIR, "base_url": API_BASE + "/styles/preview"}
    if style is not None and style.get("user"):
        return save_user_style_preview(style_id, image, style["revision"], path=USER_BACKGROUND_STYLES_PATH, **options)
    return save_style_preview(style_id, image, **options)


def _turbo_entry():
    from . import vnccs_control_center as center
    config = center._apply_active_installed_paths(center._get_cc_config("MIUProject/VNCCS_v3.0"))
    entries = [entry for entry in config.get("lora", [])
               if center._entry_kind(entry) == "qi2" and center._entry_type(entry) == "turbolora"]
    return center._find_entry(entries, "Qwen Image 2.1 Viggle Turbo")


def _turbo_name():
    from .vnccs_control_center import _rel_within_folder
    entry = _turbo_entry()
    if entry is None:
        raise ValueError("QI2 Turbo LoRA is not configured in the Control Center catalog")
    return _asset_name("loras", _rel_within_folder(entry["local_path"]))


def _resolve_editor_seed(editor, resolved=False):
    if editor["generation"]["seed_mode"] == "randomize" and not resolved:
        from ..utils import generate_seed
        editor = copy.deepcopy(editor)
        editor["generation"]["seed"] = generate_seed(0)
    return editor


def compose_prompt(editor: dict[str, Any]) -> str:
    editor = normalize_editor(editor)
    line = lambda value: re.sub(r"\s+", " ", value).strip()
    prompt = (f"Style: {line(_rendering_style(editor['style']))}\nTime: Day\n"
              f"Interior/Exterior: {line(editor['attributes']['scene_type'])}\nScene description: {_prompt_description(editor)}")
    return "\n".join([prompt, *(f"{key.title()}: {line(editor[key])}" for key in COMPOSITION_FIELDS if editor[key])])


def _call_comfy_node(class_name: str, **kwargs):
    from .character_generator import _call_comfy_node as call
    return call(class_name, _vnccs_node_id="background_creator", **kwargs)


def _installed(category: str) -> list[str]:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list(category) or [])
    except (ImportError, AttributeError, KeyError):
        return []


def _asset_name(category: str, name: str) -> str:
    return _resolve_model_name(category, name)


def configuration() -> dict[str, Any]:
    from .sampler_scheduler_picker import fetch_sampler_scheduler_lists
    styles, styles_error = [], ""
    try:
        styles = style_choices()
    except (OSError, ValueError) as error:
        styles_error = str(error)
        log_event("warning", component="Background Creator", message="Style library is unavailable; saved style text is retained", error=styles_error)
    model_types = {}
    for model_type, names in (("qi2", ASSETS), ("moss", MOSS_ASSETS)):
        assets = []
        for category, name in names.items():
            installed = [entry.replace("\\", "/") for entry in _installed(category)]
            matches = [entry for entry in installed if entry.rsplit("/", 1)[-1].lower() == name.lower()]
            assets.append({"category": category, "name": name, "installed": name in installed or len(matches) == 1})
        model_types[model_type] = {"assets": assets, "installed": all(asset["installed"] for asset in assets)}
    turbo = _turbo_entry()
    if turbo is not None:
        from .vnccs_control_center import _rel_within_folder
        try:
            name = _asset_name("loras", _rel_within_folder(turbo["local_path"]))
            installed = name in _installed("loras")
        except FileNotFoundError:
            installed = False
        turbo = {**turbo, "status": "installed" if installed else "missing"}
    return {"styles": styles, "styles_error": styles_error, "turbo_lora": turbo, "editor": copy.deepcopy(DEFAULT_EDITOR), "model": "Qwen Image 2.1 INT8 ConvRot",
            "assets": model_types["qi2"]["assets"], "installed": model_types["qi2"]["installed"], "model_types": model_types,
            "loras": _installed("loras"), "diffusion_models": _installed("diffusion_models"), "aspects": list(ASPECTS),
            "samplers": fetch_sampler_scheduler_lists()[0] or ["euler"],
            "schedulers": fetch_sampler_scheduler_lists()[1] or ["simple"]}


def _load_clip(model_type="qi2"):
    assets = MOSS_ASSETS if model_type == "moss" else ASSETS
    return _call_comfy_node("CLIPLoader", clip_name=_asset_name("text_encoders", assets["text_encoders"]),
                            type="moss_image" if model_type == "moss" else "qwen_image", device="default")[0]


def _load_assets(model_name=ASSETS["diffusion_models"], model_type="qi2"):
    import folder_paths
    from .vnccs_control_center import _model_file_signature
    global _ASSET_CACHE
    assets = MOSS_ASSETS if model_type == "moss" else ASSETS
    key = tuple(_asset_name(category, model_name if category == "diffusion_models" else name)
                for category, name in assets.items())
    signature = (key, tuple(_model_file_signature(folder_paths.get_full_path(category, name) or "")
                            for category, name in zip(assets, key)))
    if _ASSET_CACHE is None or _ASSET_CACHE[0] != signature:
        model = _call_comfy_node("UNETLoader", unet_name=key[0], weight_dtype="default")[0]
        clip = _load_clip(model_type)
        vae = _call_comfy_node("VAELoader", vae_name=key[2])[0]
        _ASSET_CACHE = (signature, model, vae)
    else:
        _, model, vae = _ASSET_CACHE
        # Qwen3-VL generation mutates shared CLIP internals; clone() cannot isolate them.
        clip = _load_clip(model_type)
    return model.clone(), clip, vae


def _background_cache_key(editor):
    from .vnccs_control_center import _model_file_signature
    editor = normalize_editor(editor)
    settings = {key: copy.deepcopy(editor["generation"][key]) for key in DEFAULT_EDITOR["generation"]}
    settings["lora_stack"] = [{"name": item["name"], "strength": item["strength"]}
                              for item in settings["lora_stack"] if item["name"] and item["strength"]]
    assets = MOSS_ASSETS if settings["model_type"] == "moss" else ASSETS
    files = [(category, settings["diffusion_model_name"] if category == "diffusion_models" else name)
             for category, name in assets.items()]
    files.extend(("loras", item["name"]) for item in settings["lora_stack"])
    if settings["turbo"]:
        files.append(("loras", _turbo_name()))
    payload = {"prompt": compose_prompt(editor), "suffix": BACKGROUND_PROMPT_SUFFIX,
               "negative": editor["negative"], "dimensions": dimensions(editor),
               "generation": settings, "files": [_model_file_signature(_resolve_model_path(category, name)) for category, name in files],
               "enhancer": [_background_enhancement_system_prompt(), TEXT_GENERATION] if settings["prompt_enhance"] else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _check_cancel(cancel_event) -> None:
    if cancel_event.is_set():
        raise GenerationCancelled("Background generation cancelled")


def _stage(draw_id: str, cancel_event, stage: str, value: float, message: str) -> None:
    _check_cancel(cancel_event)
    _set_draw_progress(draw_id, stage, value, message=message)


def progress(draw_id: str) -> dict[str, Any]:
    return _get_draw_progress(draw_id)


def interrupt(draw_id: str) -> bool:
    """Only signal ComfyUI while this job owns the shared inference slot."""
    with _ACTIVE_LOCK:
        if _ACTIVE is None or _ACTIVE["draw_id"] != draw_id:
            return False
        _ACTIVE["cancel_event"].set()
        _ACTIVE["interrupted"] = True
        set_interrupt(True)
        return True


@contextmanager
def _model_session(draw_id: str, cancel_event):
    global _ACTIVE
    _check_cancel(cancel_event)
    while not inference_lock.acquire(timeout=0.25):
        _check_cancel(cancel_event)
    active = {"draw_id": draw_id, "cancel_event": cancel_event, "interrupted": False}
    try:
        with _ACTIVE_LOCK:
            _check_cancel(cancel_event)
            # An earlier workflow Stop must not cancel a new owned request.
            set_interrupt(False)
            _ACTIVE = active
        _check_cancel(cancel_event)
        yield
    except BaseException as exc:
        cancelled = cancel_event.is_set() or isinstance(exc, (GenerationCancelled,) + interrupt_types())
        _set_draw_progress(draw_id, "cancelled" if cancelled else "error", 0, message=str(exc))
        if cancelled and not isinstance(exc, GenerationCancelled):
            raise GenerationCancelled("Background generation cancelled") from exc
        raise
    finally:
        with _ACTIVE_LOCK:
            if _ACTIVE is active:
                if active["interrupted"]:
                    set_interrupt(False)
                _ACTIVE = None
        try:
            cleanup_runtime()
        finally:
            inference_lock.release()


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("QI2 text generation returned no text")
    value = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", value, flags=re.I | re.S).strip()
    return re.sub(r"^\s*```(?:json|text)?\s*|\s*```\s*$", "", value, flags=re.I).strip()


def _json_object(value: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start():])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("QI2 Background Wizard returned no valid JSON object")


def _generate_text(clip, prompt: str, system_prompt: str, *, draw_id=None, context="Prompt enhancement") -> str:
    log_event("text_generation_input", component="Background Creator", node_id=draw_id, context=context,
              prompt=prompt, system_prompt=system_prompt, truncate=False,
              display_text=f"System prompt:\n{system_prompt}\n\nUser prompt:\n{prompt}")
    generated = _clean_text(_call_comfy_node("TextGenerate", clip=clip, prompt=prompt,
                                          system_prompt=system_prompt, **copy.deepcopy(TEXT_GENERATION))[0])
    log_event("text_generation_output", component="Background Creator", node_id=draw_id, context=f"{context} result",
              output=generated, truncate=False, display_text=generated)
    return generated


def wizard(description: str, editor: dict[str, Any], draw_id: str, cancel_event) -> dict[str, Any]:
    import torch
    editor = normalize_editor(editor)
    description = _text(description, "Wizard description", 40000)
    if not description and not any(editor[key] for key in COMPOSITION_FIELDS):
        raise ValueError("Describe the background for the wizard")
    with _model_session(draw_id, cancel_event), torch.inference_mode():
        _ensure_direct_sampling_prompt_context(f"background_wizard_{draw_id}")
        _stage(draw_id, cancel_event, "loading", 0.05, "Loading the QI2 background wizard")
        clip = _load_clip()
        _stage(draw_id, cancel_event, "conditioning", 0.2, "Describing the background")
        fresh_brief = bool(description)
        schema = json.dumps({"style": _rendering_style(editor["style"]), "time": "Day",
                             "scene_type": "" if fresh_brief else editor["attributes"]["scene_type"],
                             "scene_description": description if fresh_brief else editor["scene_description"],
                             **{key: "" if fresh_brief else editor[key] for key in COMPOSITION_FIELDS}})
        generated = _generate_text(clip, description or compose_prompt(editor),
            SCENE_EXPANSION_SYSTEM +
            "Keep time as Day. Time and weather variations are configured separately. "
            "Keep the selected style. Infer scene_type from the latest scene brief when scene_type is empty; "
            "never turn an exterior brief into an interior or an interior brief into an exterior. "
            "Use a supplied scene_type only when the scene brief does not establish it. "
            "Style is a separate rendering field; never interpret it as scene content or echo its aesthetics in the five scene fields. "
            "Time is the time of day; scene_type is Interior or Exterior. "
            "Return ONLY one JSON object with exactly eight string keys in this order: style, time, scene_type, "
            "scene_description, front, back, left, right. These correspond to Style, Time, Interior/Exterior, "
            "Scene description, Front, Back, Left and Right in the final prompt. Expand all five scene fields "
            "with location-consistent layout, architecture, objects and spatial relationships, adding only objects supported by the brief; do not add nested attributes or "
            "extra fields. The following object supplies the selected settings and the scene brief, not a finished answer: "
            + schema, draw_id=draw_id, context="Background wizard")
        parsed = _json_object(generated)
        for key in ("time", "scene_type"):
            if key in parsed:
                editor["attributes"][key] = parsed[key]
        if "attributes" in parsed:
            if not isinstance(parsed["attributes"], dict):
                raise ValueError("Background Wizard attributes must be an object")
            editor["attributes"].update({key: value for key, value in parsed["attributes"].items()
                                         if key in {*DEFAULT_EDITOR["attributes"], *LEGACY_SCENE_LABELS}})
        else:
            editor["attributes"].update({key: value for key, value in parsed.items() if key in {*DEFAULT_EDITOR["attributes"], *LEGACY_SCENE_LABELS}})
        if "style" in parsed:
            editor["style"] = parsed["style"]
        editor["scene_description"] = parsed.get("scene_description", "\n".join(filter(None, [description, *(
            f"{label}: {editor['attributes'][key]}." for key, label in LEGACY_SCENE_LABELS.items() if editor["attributes"].get(key))])))
        editor.update({key: parsed[key] for key in COMPOSITION_FIELDS if key in parsed})
        editor = normalize_editor(editor)
        editor["attributes"]["time"] = "Day"
        _stage(draw_id, cancel_event, "complete", 1, "Background attributes ready")
        return editor


def _tensor_to_pil(tensor):
    pixels = tensor[0].detach().float().cpu().numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("QI2 returned non-finite image pixels")
    return Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def generate_image(editor: dict[str, Any], draw_id: str, cancel_event):
    import torch
    editor = normalize_editor(editor)
    settings = editor["generation"]
    moss = settings["model_type"] == "moss"
    if settings["turbo"] and (settings["steps"] != 6 or abs(settings["cfg"] - 1) > 1e-6):
        raise ValueError("QI2 Viggle Turbo requires six steps and CFG 1")
    width, height = dimensions(editor)
    prompt = compose_prompt(editor)
    with _model_session(draw_id, cancel_event), torch.inference_mode():
        _ensure_direct_sampling_prompt_context(f"background_{draw_id}")
        _stage(draw_id, cancel_event, "loading", 0.03, "Loading MOSS-Image" if moss else "Loading Qwen Image 2.1")
        model, clip, vae = _load_assets(settings["diffusion_model_name"], settings["model_type"])
        _stage(draw_id, cancel_event, "loading", 0.12, "Applying background model settings")
        for item in settings["lora_stack"]:
            if not item["name"]:
                continue
            model, clip = _call_comfy_node("LoraLoader", model=model, clip=clip, lora_name=item["name"],
                                          strength_model=item["strength"], strength_clip=item["strength"])[:2]
            _check_cancel(cancel_event)
        if settings["turbo"]:
            from .character_generator import apply_viggle_turbo_lora
            model = apply_viggle_turbo_lora(model, _turbo_name(), 1.0)
        if not moss:
            model = _call_comfy_node("QwenImage21Cache", model=model, device=settings["cache_device"],
                                     dtype=settings["cache_dtype"])[0]
        if settings["prompt_enhance"]:
            _stage(draw_id, cancel_event, "conditioning", 0.18, "Enhancing the background prompt")
            enhanced = _generate_text(clip, prompt,
                _background_enhancement_system_prompt(),
                draw_id=draw_id, context="Moss prompt enhancement" if moss else "Qwen Image 2.1 prompt enhancement")
            if enhanced:
                enhanced = _text(enhanced, "Enhanced prompt", 16000)
                parsed = _parse_prompt(enhanced)
                scene = {key: parsed[key] for key in ("scene_description", *COMPOSITION_FIELDS) if key in parsed} if parsed else {
                    "scene_description": re.sub(r"^Scene description:\s*", "", enhanced)}
                prompt = compose_prompt({**editor, **scene})
        prompt += f" {BACKGROUND_PROMPT_SUFFIX}"
        log_event("prompts", component="Background Creator", node_id=draw_id,
                  context=f"{'Moss' if moss else 'Qwen Image 2.1'} background",
                  model=settings["diffusion_model_name"], positive=prompt, negative=editor["negative"], truncate=False,
                  display_text=f"Positive prompt:\n{prompt}\n\nNegative prompt:\n{editor['negative']}")
        _stage(draw_id, cancel_event, "conditioning", 0.3, "Encoding the background prompt")
        if moss:
            positive = _call_comfy_node("CLIPTextEncode", clip=clip, text=prompt)[0]
            negative = _call_comfy_node("CLIPTextEncode", clip=clip, text=editor["negative"])[0]
            latent = _call_comfy_node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)[0]
        else:
            encoded = _call_comfy_node("TextEncodeQwenImage21", clip=clip, vae=vae, prompt=prompt,
                negative_prompt=editor["negative"], resolution=1024, images={})
            positive, negative = encoded[:2]
            latent = _call_comfy_node("EmptyLatentImage", width=width, height=height, batch_size=1)[0]
        _stage(draw_id, cancel_event, "sampling", 0.35, "Rendering the background")
        with _report_comfy_sampling_progress(draw_id, settings["steps"], cancel_event):
            if settings["turbo"]:
                from .qi2_viggle import viggle_turbo_sigmas
                noise = _call_comfy_node("RandomNoise", noise_seed=settings["seed"])[0]
                guider = _call_comfy_node("BasicGuider", model=model, conditioning=positive)[0]
                sampler = _call_comfy_node("KSamplerSelect", sampler_name="euler")[0]
                samples = _call_comfy_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                    sampler=sampler, sigmas=viggle_turbo_sigmas(latent), latent_image=latent)[0]
            else:
                samples = _call_comfy_node("KSampler", model=model, seed=settings["seed"], steps=settings["steps"],
                    cfg=settings["cfg"], sampler_name=settings["sampler"], scheduler=settings["scheduler"],
                    positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        _stage(draw_id, cancel_event, "decoding", 0.9, "Decoding the background image")
        image = _tensor_to_pil(_call_comfy_node("VAEDecode", vae=vae, samples=samples)[0])
        if image.size != (width, height):
            raise RuntimeError(f"Background model returned {image.size[0]}x{image.size[1]}, expected {width}x{height}")
        _stage(draw_id, cancel_event, "complete", 1, "Background image ready")
        return image, {"editor": editor, "prompt": prompt, "negative": editor["negative"],
                       "width": width, "height": height, "seed": settings["seed"], "model": settings["model_type"]}


# Generation runs under the shared ComfyUI inference lock.
_GENERATION_PROGRESS = {}


def _resolve_model_name(category, name):
    normalized = name.replace("\\", "/")
    if normalized.startswith(("/", "~")) or ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("Model name must be a relative installed filename")
    installed = _installed(category)
    exact = [value for value in installed if value.replace("\\", "/") == normalized]
    matches = exact or [value for value in installed if value.replace("\\", "/").rsplit("/", 1)[-1].lower() == normalized.rsplit("/", 1)[-1].lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Install exactly one {name} in ComfyUI models/{category}")
    return matches[0]


def _resolve_model_path(category, name):
    import folder_paths
    filename = _resolve_model_name(category, name)
    path = folder_paths.get_full_path(category, filename)
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Installed model is missing: {name}")
    return Path(path)


def set_interrupt(value):
    try:
        from comfy.model_management import interrupt_current_processing
    except ImportError:
        return
    interrupt_current_processing(value)


def interrupt_types():
    try:
        from comfy.model_management import InterruptProcessingException
    except ImportError:
        return ()
    return (InterruptProcessingException,)


def _set_draw_progress(draw_id, stage, value, message=""):
    _GENERATION_PROGRESS[draw_id] = {"stage": stage, "progress": value, "message": message}
    if len(_GENERATION_PROGRESS) > 64:
        _GENERATION_PROGRESS.pop(next(iter(_GENERATION_PROGRESS)))
    with _JOBS_LOCK:
        owner = next((job for job in _JOBS.values() if job.get("node_id") == str(draw_id)), None)
    if owner:
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("vnccs.background.stage", {"node_id": str(draw_id), "background_id": owner["background_id"],
                                                                      "stage": stage, "progress": value * 100, "message": message})
        except Exception as error:
            log_event("warning", component="Background Creator", message="Could not publish workflow progress", error=str(error))


def _get_draw_progress(draw_id):
    return dict(_GENERATION_PROGRESS.get(draw_id, {}))


def _ensure_direct_sampling_prompt_context(prompt_id):
    from server import PromptServer
    if not hasattr(PromptServer.instance, "last_prompt_id"):
        PromptServer.instance.last_prompt_id = prompt_id


@contextmanager
def _report_comfy_sampling_progress(draw_id, steps, cancel_event):
    import comfy.utils as comfy_utils
    previous = getattr(comfy_utils, "PROGRESS_BAR_HOOK", None)
    owner = threading.get_ident()

    def report(value, total, preview=None, *args, **kwargs):
        if threading.get_ident() != owner:
            if previous:
                return previous(value, total, preview, *args, **kwargs)
            return
        _check_cancel(cancel_event)
        fraction = min(1, value / max(1, total))
        _set_draw_progress(draw_id, "sampling", 0.35 + 0.5 * fraction,
                           message=f"Rendering background: {value}/{total or steps}")
    comfy_utils.PROGRESS_BAR_HOOK = report
    try:
        yield
    finally:
        if comfy_utils.PROGRESS_BAR_HOOK is report:
            comfy_utils.PROGRESS_BAR_HOOK = previous


def _preview_tensor(path):
    import torch

    with Image.open(path) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Background preview is too large")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).unsqueeze(0)


def _decode_image(payload):
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Image upload is empty or too large")
    with Image.open(io.BytesIO(payload)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions are too large")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def _content_length_ok(request, maximum):
    try:
        declared = int(request.headers.get("Content-Length", request.content_length))
    except (TypeError, ValueError):
        return False
    return request.content_length is not None and 0 <= declared <= maximum and declared == request.content_length


def _new_job(kind):
    with _JOBS_LOCK:
        if sum(job["status"] in {"queued", "running"} for job in _JOBS.values()) >= 8:
            raise RuntimeError("Background job capacity is full")
        return {"job_id": uuid.uuid4().hex, "kind": kind, "status": "queued", "stage": "queued",
                "progress": 0, "message": "Queued", "result": None, "error": "", "cancel_event": threading.Event()}


def _job_public(job):
    return {key: copy.deepcopy(value) for key, value in job.items() if key != "cancel_event"}


def _emit(job, stage, progress, message):
    with _JOBS_LOCK:
        job.update(stage=stage, progress=progress, message=message)
    log_event("background_stage", component="Background Creator", job_id=job["job_id"],
              stage=stage, progress=progress, message=message)


def _check_job_cancel(job):
    _check_cancel(job["cancel_event"])


def _run_job(job, callback):
    with _JOBS_LOCK:
        job["status"] = "running"
    try:
        _check_job_cancel(job)
        result = callback(job)
        # An acknowledged disk commit wins over a late cancellation.
        with _JOBS_LOCK:
            job.update(result=result, status="completed")
        _emit(job, "complete", 100, "Completed")
    except GenerationCancelled as error:
        with _JOBS_LOCK:
            job.update(status="cancelled", error=str(error))
        _emit(job, "cancelled", job.get("progress", 0), "Cancelled")
    except Exception as error:
        with _JOBS_LOCK:
            job.update(status="failed", error=str(error))
        log_event("error", component="Background Creator", job_id=job["job_id"], error=str(error))
        _emit(job, "failed", job.get("progress", 0), "Generation failed")


EMPTY_STATE = '{"schema_version":2,"background_id":"","selected_image_id":""}'
_ID = re.compile(r"^[a-f0-9]{32}$")


def parse_state(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Background Creator state is invalid or too large")
    try:
        state = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Background Creator state is not valid JSON") from exc
    if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 2:
        raise ValueError("Unsupported Background Creator state version")
    for key in ("background_id", "selected_image_id"):
        value = state.get(key, "")
        if not isinstance(value, str) or (value and not _ID.fullmatch(value)):
            raise ValueError(f"Background Creator {key} is invalid")
    if state.get("operation") is not None:
        raise ValueError("Background Creator cannot import connected images. Use Upload image inside the widget to import a background.")
    return state


def _run_background(state, draw_id):
    background_id = state.get("background_id")
    if not background_id:
        raise ValueError("Select or create a background before running Background Creator")
    with _JOBS_LOCK:
        ensure_idle(background_id)
        job = _new_job("background_workflow")
        job.update(background_id=background_id, action="background", status="running", node_id=str(draw_id or job["job_id"]))
        _JOBS[job["job_id"]] = job
    try:
        document = load_background(background_id)
        if document.get("_recovery") or document.get("archived"):
            raise BackgroundStorageError("The selected background is read-only; its data is preserved")
        editor = _resolve_editor_seed(normalize_editor(state.get("editor", document["editor"])), state.get("_vnccs_seed_resolved") is True)
        selected = state.get("selected_image_id") or document["selected_image_id"]
        requested_image = selected
        image = next((item for item in document["images"] if item["image_id"] == selected), None)
        if selected and image is None:
            raise ValueError("Selected image does not belong to this background")
        changed = editor != normalize_editor(document["editor"]) or selected != document["selected_image_id"]
        if changed or "variants" not in document["editor"]:
            document = save_background(background_id, {**editor, "selected_image_id": selected}, document["revision"])
        source, tensor = "generated", None
        if image and image["settings"].get("source") == "upload":
            source, tensor = "upload", _preview_tensor(image_path(background_id, selected))
        else:
            key = _background_cache_key(editor)
            if image and image["settings"].get("cache_key") == key:
                try:
                    tensor = _preview_tensor(image_path(background_id, selected))
                    source = "cache"
                except (OSError, ValueError):
                    pass
            if tensor is None:
                rendered, settings = generate_image(editor, str(draw_id or job["job_id"]), job["cancel_event"])
                _check_job_cancel(job)
                document = add_image(background_id, rendered, {**settings, "cache_key": key}, document["revision"])
                selected = document["selected_image_id"]
                tensor = _preview_tensor(image_path(background_id, selected))
        if source == "cache":
            settings = image["settings"]
            log_event("prompts", component="Background Creator", node_id=draw_id, context="Cached background",
                      background_id=background_id, image_id=selected, positive=settings.get("prompt"), negative=settings.get("negative", ""),
                      truncate=False, display_text=f"Positive prompt:\n{settings.get('prompt', 'Saved prompt not recorded for this preview.')}\n\nNegative prompt:\n{settings.get('negative', '')}")
        log_event("background_output", component="Background Creator", node_id=draw_id, background_id=background_id,
                  image_id=selected, source=source, inference_skipped=source != "generated")
        return {"ui": {"background_preview": [{"background_id": background_id, "image_id": selected, "source": source,
                                                "revision": document["revision"],
                                                "request_image_id": requested_image, "editor": state.get("editor", editor)}]},
                "result": (tensor, variant_output(document, editor, selected))}
    finally:
        with _JOBS_LOCK:
            _JOBS.pop(job["job_id"], None)


class VNCCS_BackgroundCreator:
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "Prompt")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "create_background"
    CATEGORY = "VNCCS"
    OUTPUT_NODE = True
    DESCRIPTION = "RUN reuses a fresh background preview, generates stale previews, or passes an uploaded image through. Variant prompts are a STRING list. Settings remain in the selected background's JSON."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": False}),
                "background_data": ("STRING", {"default": EMPTY_STATE, "multiline": True, "dynamicPrompts": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, background_data=EMPTY_STATE, **kwargs):
        try:
            parse_state(background_data)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, background_data=EMPTY_STATE, **kwargs):
        from .vnccs_control_center import _model_file_signature
        state = parse_state(background_data)
        if not state.get("background_id"):
            return float("nan")
        document = load_background(state["background_id"])
        editor = normalize_editor(state.get("editor", document["editor"]))
        if editor["generation"]["seed_mode"] == "randomize" and state.get("_vnccs_seed_resolved") is not True:
            return float("nan")
        selected = state.get("selected_image_id") or document["selected_image_id"]
        image = next((item for item in document["images"] if item["image_id"] == selected), None)
        path = _path(document["name"], image["file"]) if image else None
        key = None if image and image["settings"].get("source") == "upload" else _background_cache_key(editor)
        metadata = {name: document.get(name) for name in ("editor", "selected_image_id", "archived", "_recovery")}
        return json.dumps([metadata, selected, _model_file_signature(path or ""), key], sort_keys=True)

    def create_background(self, seed=0, background_data=EMPTY_STATE, unique_id=None):
        state = parse_state(background_data)
        if state.get("background_id"):
            return _run_background(state, unique_id)
        raise ValueError("Select or create a background before RUN")

API_BASE = "/vnccs/background-creator"
MAX_REQUEST_BYTES = 34 * 1024 * 1024
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vnccs-background")
_JOBS = {}
_JOBS_LOCK = threading.RLock()


def public_document(document):
    value = copy.deepcopy(document)
    base = f"{API_BASE}/backgrounds/{value['background_id']}"
    for image in value.get("images", []):
        image["url"] = f"{base}/images/{image['image_id']}"
        if "preview_revision" in image:
            image["url"] += f"?v={image['preview_revision']}"
    for group in value.get("generated_sets", []):
        for index, image in enumerate(group["images"]):
            image["url"] = f"{base}/sets/{_id(group['set_id'])}/{index}?v={image['revision']}"
    return value


def ensure_idle(background_id):
    with _JOBS_LOCK:
        if any(job.get("background_id") == background_id and job["status"] in {"queued", "running"} for job in _JOBS.values()):
            raise BackgroundConflictError("This background is busy. Wait for the job or stop it first.")


def start_job(background_id, payload, raw_image=None):
    if not isinstance(payload, dict):
        raise ValueError("Background job must be an object")
    action = payload.get("action")
    if action not in {"background", "wizard"}:
        raise ValueError("Unknown background action")
    if raw_image is not None:
        raise ValueError("Background generation jobs do not accept image uploads")
    editor = normalize_editor(payload.get("editor", {}))
    description = ""
    if action == "wizard":
        description = _text(payload.get("description"), "Wizard description", 40000)
        if not description:
            raise ValueError("Describe the background for the wizard")
    if action == "background":
        editor = _resolve_editor_seed(editor, payload.get("_vnccs_seed_resolved") is True)
    document = load_background(background_id)
    if document.get("_recovery"):
        raise BackgroundStorageError("Recovered background metadata is read-only")
    if document.get("archived"):
        raise ValueError("Background is archived")
    values = {**payload, "editor": editor, "revision": document["revision"], "description": description}
    with _JOBS_LOCK:
        ensure_idle(background_id)
        for old_id in list(_JOBS):
            if len(_JOBS) < 64:
                break
            if _JOBS[old_id]["status"] not in {"queued", "running"}:
                del _JOBS[old_id]
        job = _new_job(f"background_{action}")
        job.update(background_id=background_id, action=action)
        _JOBS[job["job_id"]] = job
        _WORKER.submit(_run_job, job, lambda current: _execute(current, values))
    return job_status(job["job_id"])


def _execute(job, payload):
    background_id = job["background_id"]
    action = job["action"]
    editor, revision = payload["editor"], payload["revision"]
    _check_job_cancel(job)
    _emit(job, "preparing", 1, "Preparing background" if action == "background" else "Preparing " + action)
    if action == "wizard":
        result = wizard(payload["description"], editor, job["job_id"], job["cancel_event"])
        _check_job_cancel(job)
        return {"editor": result, "background_id": background_id}
    key = _background_cache_key(editor)
    image, settings = generate_image(editor, job["job_id"], job["cancel_event"])
    _check_job_cancel(job)
    document = add_image(background_id, image, {**settings, "cache_key": key}, revision)
    return {"document": public_document(document)}


def job_status(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise FileNotFoundError("Background job was not found")
        value = _job_public(job)
    if job["status"] == "running" and job["action"] in {"background", "wizard"}:
        snapshot = progress(job_id)
        if snapshot and snapshot.get("stage") != "unknown":
            value.update(message=snapshot.get("message", value["message"]), progress=100 * snapshot.get("progress", 0))
    return value


def cancel_job(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise FileNotFoundError("Background job was not found")
        job["cancel_event"].set()
        if job["action"] in {"background", "wizard"}:
            interrupt(job_id)
    return job_status(job_id)


_REGISTERED = False


def register_routes(routes):
    global _REGISTERED
    if _REGISTERED:
        return
    from aiohttp import web
    from ..utils import validate_privileged_request

    base = API_BASE

    async def perform(callback, *args):
        try:
            return web.json_response(await asyncio.to_thread(callback, *args))
        except BackgroundConflictError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def body(request):
        if not _content_length_ok(request, MAX_REQUEST_BYTES):
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_REQUEST_BYTES, actual_size=request.content_length or 0)
        if request.content_type == "multipart/form-data":
            post = await request.post()
            raw = str(post.get("data", "{}"))
            if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
                raise ValueError("Background settings are too large")
            data = json.loads(raw)
            field = post.get("image")
            image = await asyncio.to_thread(field.file.read, MAX_REQUEST_BYTES + 1) if field is not None and hasattr(field, "file") else None
            if image is not None and len(image) > MAX_REQUEST_BYTES:
                raise ValueError("Image upload is too large")
        else:
            raw = await request.read()
            if len(raw) > MAX_JSON_BYTES:
                raise ValueError("Background settings are too large")
            data, image = json.loads(raw or b"{}"), None
        if not isinstance(data, dict):
            raise ValueError("Background request must be an object")
        return data, image

    def respond(callback):
        async def handler(request):
            try:
                validate_privileged_request(request)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=403)
            try:
                payload, image = await body(request)
            except web.HTTPException:
                raise
            except (ValueError, TypeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
            return await perform(callback, request.match_info, payload, image)
        return handler

    @routes.get(base + "/config")
    async def config(request):
        return await perform(configuration)

    @routes.get(base + "/styles")
    async def styles(request):
        return await perform(style_catalog)

    @routes.post(base + "/styles")
    @respond
    def save_style(info, payload, image):
        from .character_styles import save_user_style, style_preview_url
        style = save_user_style(payload, USER_BACKGROUND_STYLES_PATH)
        return {"style": {**style, "image": style_preview_url(style["id"], directory=STYLE_PREVIEWS_DIR,
                                                             base_url=base + "/styles/preview")}}

    @routes.post(base + "/styles/delete")
    @respond
    def delete_style(info, payload, image):
        from .character_styles import delete_user_style
        style_id = payload.get("id")
        return {"style_id": style_id, "deleted": delete_user_style(style_id, USER_BACKGROUND_STYLES_PATH,
                                                                    preview_directory=STYLE_PREVIEWS_DIR)}

    @routes.post(base + "/styles/preview")
    async def style_preview(request):
        from .preview_runtime import run_preview_job
        try:
            validate_privileged_request(request)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=403)
        try:
            payload, _image = await body(request)
            node_id = _text(payload.get("node_id", ""), "Node ID", 100)
            request_id = _text(payload.get("request_id", ""), "Request ID", 100)

            def emit(status):
                from server import PromptServer
                PromptServer.instance.send_sync("vnccs.style_preview.stage", {
                    "node_id": node_id, "request_id": request_id, "status": status,
                })

            def render():
                emit("running")
                try:
                    result = generate_style_preview(payload)
                except Exception:
                    emit("error")
                    raise
                emit("done")
                return result

            emit("queued")
            return web.json_response(await run_preview_job(render))
        except web.HTTPException:
            raise
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            log_event("exception", component="Background Styles", level="error", exc_info=True, error=str(exc))
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(base + "/backgrounds")
    async def backgrounds(request):
        errors = []
        return await perform(lambda: {"backgrounds": [public_document(item) for item in list_backgrounds(errors)],
                                      "errors": errors})

    @routes.post(base + "/backgrounds")
    @respond
    def create(info, payload, image):
        return public_document(create_background(payload.get("name"), normalize_editor(payload.get("editor", {}))))

    @routes.get(base + "/backgrounds/{background_id}")
    async def background(request):
        return await perform(lambda: public_document(load_background(request.match_info["background_id"])))

    @routes.patch(base + "/backgrounds/{background_id}")
    @respond
    def save(info, payload, image):
        ensure_idle(info["background_id"])
        state = normalize_editor(payload.get("editor", {}))
        if "selected_image_id" in payload:
            state["selected_image_id"] = payload["selected_image_id"]
        return public_document(save_background(info["background_id"], state))

    @routes.delete(base + "/backgrounds/{background_id}")
    @respond
    def delete(info, payload, image):
        ensure_idle(info["background_id"])
        return delete_background(info["background_id"])

    @routes.post(base + "/backgrounds/{background_id}/upload")
    @respond
    def upload(info, payload, image):
        ensure_idle(info["background_id"])
        decoded = _decode_image(image or b"").convert("RGB")
        return public_document(add_image(info["background_id"], decoded, {"source": "upload"}))

    @routes.post(base + "/backgrounds/{background_id}/jobs")
    @respond
    def job(info, payload, image):
        return start_job(info["background_id"], payload, image)

    @routes.get(base + "/jobs/{job_id}")
    async def status(request):
        return await perform(job_status, request.match_info["job_id"])

    @routes.post(base + "/jobs/{job_id}/cancel")
    @respond
    def cancel(info, payload, image):
        return cancel_job(info["job_id"])

    async def serve(request, callback, *, immutable=True):
        try:
            path = await asyncio.to_thread(callback)
            return web.FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable" if immutable else "no-store"})
        except (ValueError, TypeError, FileNotFoundError) as exc:
            return web.json_response({"error": str(exc)}, status=404)

    @routes.get(base + "/styles/preview")
    async def get_style_preview(request):
        from .character_styles import style_preview_path

        def existing_preview():
            style_id = request.query.get("style", "")
            if style_id != "custom" and not any(style["id"] == style_id for style in style_choices()):
                raise FileNotFoundError("Unknown background style ID")
            path = Path(style_preview_path(style_id, directory=STYLE_PREVIEWS_DIR))
            if not path.is_file():
                raise FileNotFoundError("Background style preview has not been generated")
            return path

        return await serve(request, existing_preview)

    @routes.get(base + "/backgrounds/{background_id}/images/{image_id}")
    async def image(request):
        info = request.match_info
        return await serve(request, lambda: image_path(info["background_id"], info["image_id"]), immutable=False)

    @routes.get(base + "/backgrounds/{background_id}/sets/{set_id}/{index}")
    async def set_image(request):
        info = request.match_info
        return await serve(request, lambda: background_set_image_path(info["background_id"], info["set_id"], int(info["index"])), immutable=False)

    _REGISTERED = True

NODE_CLASS_MAPPINGS = {"VNCCS_BackgroundCreator": VNCCS_BackgroundCreator}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCS_BackgroundCreator": "VNCCS Background Creator"}

