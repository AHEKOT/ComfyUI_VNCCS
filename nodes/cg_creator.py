"""VNCCS CG Creator: workflow output, refinement, saved assets and HTTP routes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

from PIL import Image

from ..operation_logger import log_event, log_stage, logged_operation
from ..utils import get_full_path_agnostic, _safe_get_model_folder_paths, validate_privileged_request
from . import background_creator as background
from .runtime_cleanup import cleanup_runtime, inference_stage


# Saved background records and immutable CG assets.

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024
MAX_CGS = 512
MAX_MODELS = 512
_ID = re.compile(r"^[a-f0-9]{32}$")
_ASSET_KINDS = {"ply", "splat", "reference", "prepared"}



BackgroundConflictError = background.BackgroundConflictError
BackgroundStorageError = background.BackgroundStorageError


def background_owner():
    return background


def root():
    import folder_paths

    return Path(folder_paths.get_output_directory()).resolve() / "VNCCS" / "Backgrounds"


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid CG Creator asset ID")
    return value


def _request_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value):
        raise ValueError("Invalid CG generation request reference")
    return value


def _name(value):
    if not isinstance(value, str):
        raise ValueError("A background or CG name is required")
    name = unicodedata.normalize("NFC", value.strip())
    if (not name or len(name) > 96 or name in {".", ".."} or name.endswith(".")
            or any(ord(char) < 32 or ord(char) == 127 or char in '<>:"/\\|?*' for char in name)
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name, re.I)):
        raise ValueError("Invalid background or CG name")
    return name


def _contained(base, *parts):
    base = Path(base)
    target = base.joinpath(*parts)
    if any(part in {"", ".", ".."} for item in parts for part in str(item).replace("\\", "/").split("/")):
        raise ValueError("Invalid CG Creator asset path")
    if any(path.is_symlink() for path in (base, *target.parents, target)):
        raise ValueError("CG Creator paths cannot contain symbolic links")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("CG Creator asset path escaped its directory")
    return target


def _json_bytes(value):
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("CG Creator metadata must contain finite JSON values") from exc
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("CG Creator metadata is too large")
    return payload


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"CG Creator {label} must be an object")
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


def _copy_asset(source, target, maximum):
    """Publish a complete copy; retries may reuse only identical immutable files."""
    _contained(root(), target.relative_to(root()).as_posix())
    if not source.is_file() or source.stat().st_size > maximum:
        raise ValueError("CG Creator source asset is missing or too large")
    if target.exists():
        def digest(path):
            value = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    value.update(chunk)
            return value.digest()
        if not target.is_file() or target.stat().st_size != source.stat().st_size or digest(target) != digest(source):
            raise BackgroundConflictError("An existing background asset differs; both copies are preserved")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        count = 0
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                count += len(chunk)
                if count > maximum:
                    raise ValueError("CG Creator source asset is too large")
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    with path.open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("CG Creator metadata is too large")
    value = json.loads(payload)
    _json_bytes(value)
    return value, payload


def _dimensions(width, height):
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 16384
            or not 1 <= height <= 16384 or width * height > MAX_IMAGE_PIXELS):
        raise ValueError("CG Creator image dimensions are invalid or too large")


def _camera(value):
    camera = _object(value, "camera")
    for key in ("position", "target", "up"):
        vector = camera.get(key)
        if (not isinstance(vector, list) or len(vector) != 3
                or any(type(number) not in (float, int) or not math.isfinite(number)
                       or abs(number) > 1e6 for number in vector)):
            raise ValueError("Invalid CG Creator camera vector")
    fov = camera.get("fov")
    if type(fov) not in (int, float) or not math.isfinite(fov) or not 5 <= fov <= 120:
        raise ValueError("Invalid CG Creator camera FOV")
    if math.dist(camera["position"], camera["target"]) < 1e-7 or math.hypot(*camera["up"]) < 1e-7:
        raise ValueError("CG Creator camera is degenerate")
    return camera


def _source_variant(value):
    if value is None:
        return None
    variant = _object(value, "source variant")
    if len(_json_bytes(variant)) > 2048:
        raise ValueError("CG source variant metadata is too large")
    _id(variant.get("set_id"))
    if type(variant.get("index")) is not int or not 1 <= variant["index"] <= 64:
        raise ValueError("CG source variant index must be between 1 and 64")
    if "variant_id" in variant and (not isinstance(variant["variant_id"], str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", variant["variant_id"])):
        raise ValueError("CG source variant ID is invalid")
    if "name" in variant:
        _name(variant["name"])
    return variant


def _variant_key(value):
    variant = _source_variant(value)
    return (variant["set_id"], variant["index"]) if variant else None


def _variant_source(document, image_id, value):
    variant = _source_variant(value)
    if variant is None:
        return None
    groups = document.get("generated_sets", [])
    if not isinstance(groups, list):
        raise ValueError("Saved background sets must be a list")
    group = next((group for group in groups if isinstance(group, dict) and group.get("set_id") == variant["set_id"]), None)
    if group is None or group.get("source_image_id") != image_id:
        raise ValueError("The selected variant belongs to another background source")
    images, cards, index = group.get("images"), group.get("variants"), variant["index"]
    if (not isinstance(images, list) or not isinstance(cards, list) or len(images) > 65 or len(cards) > 64
            or index >= len(images) or index > len(cards)
            or not isinstance(images[index], dict) or not isinstance(cards[index - 1], dict)):
        raise ValueError("The selected background variant does not exist")
    image, card = images[index], cards[index - 1]
    _dimensions(image.get("width"), image.get("height"))
    _source_variant({"set_id": variant["set_id"], "index": index, "variant_id": card.get("id"), "name": card.get("name")})
    if image.get("name") != card["name"]:
        raise ValueError("The saved background variant metadata does not match")
    if ("variant_id" in variant and variant["variant_id"] != card.get("id")
            or "name" in variant and (variant["name"] != card.get("name") or variant["name"] != image.get("name"))):
        raise ValueError("The selected background variant metadata does not match")
    return variant


# Copied from the current VNCCS Background Creator metadata validators.


def _validate(document, directory_name):
    background._validate(document, directory_name)
    image_ids = {image["image_id"] for image in document["images"]}
    cgs = document.get("cgs", [])
    if not isinstance(cgs, list) or len(cgs) > MAX_CGS:
        raise ValueError("Invalid CG Creator asset list")
    cg_ids = set()
    models = document.get("models", [])
    if not isinstance(models, list) or len(models) > MAX_MODELS:
        raise ValueError("Invalid CG Creator model list")
    model_sources, model_variants, asset_paths = {}, {}, set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Invalid CG Creator model")
        model_id = _id(model.get("model_id"))
        if model_id in model_sources:
            raise ValueError("Duplicate CG Creator model")
        if model.get("source_image_id") not in image_ids:
            raise ValueError("Model source image does not exist")
        model_sources[model_id] = model["source_image_id"]
        model_variants[model_id] = _variant_key(model.get("source_variant"))
        _object(model.get("settings"), "model settings")
        if "file" in model:
            relative = _asset_file(model, "models").casefold()
            if relative in asset_paths:
                raise ValueError("Duplicate CG Creator asset path")
            asset_paths.add(relative)
    for cg in cgs:
        if not isinstance(cg, dict) or _id(cg.get("cg_id")) in cg_ids:
            raise ValueError("Invalid or duplicate background CG")
        cg_ids.add(cg["cg_id"])
        if "archived" in cg and type(cg["archived"]) is not bool:
            raise ValueError("Invalid background CG archive state")
        _name(cg.get("name"))
        model_id = _id(cg.get("model_id"))
        if model_id not in model_sources or cg.get("source_image_id") != model_sources[model_id]:
            raise ValueError("CG does not match its background and model variant")
        if _variant_key(cg.get("source_variant")) != model_variants[model_id]:
            raise ValueError("CG does not match its saved source variant")
        _dimensions(cg.get("width"), cg.get("height"))
        _camera(cg.get("camera"))
        if "file" in cg or "raw_file" in cg:
            relative = _asset_file(cg, "cgs").casefold()
            if relative in asset_paths:
                raise ValueError("Duplicate CG Creator asset path")
            asset_paths.add(relative)
    return document


def _read_document(directory):
    try:
        _recover_assets(directory)
        background._recover_set_publication(directory)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise BackgroundStorageError("Cannot recover background publication; original files are preserved") from error
    filename = f"{directory.name}_config.json"
    path = _contained(directory, filename)
    try:
        document, payload = _read_json(path)
        _validate(document, directory.name)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as primary_error:
        try:
            backup = _contained(directory, filename + ".backup")
            document, payload = _read_json(backup)
            _validate(document, directory.name)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as backup_error:
            raise BackgroundStorageError(f"Cannot restore background '{directory.name}'; original files are preserved") from backup_error
        document["_recovery"] = {"source": filename + ".backup", "error": str(primary_error)}
        return document, payload
    return _migrate_assets(directory, document, payload)


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
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision != document["revision"]):
        raise BackgroundConflictError("Background changed in another editor; reload before saving")
    if document["archived"]:
        raise ValueError("Background is archived")
    return directory, document, payload


def _commit(directory, document, previous):
    _validate(document, directory.name)
    return background._commit(directory, document, previous)


def list_backgrounds(errors=None):
    with background._LOCK:
        documents = []
        for directory in _directories():
            try:
                documents.append(_read_document(directory)[0])
            except BackgroundStorageError as error:
                if errors is None:
                    raise
                errors.append({"name": directory.name, "error": str(error)})
        return sorted((item for item in documents if not item["archived"]), key=lambda item: item["name"].casefold())


def load_background(background_id):
    with background._LOCK:
        return _locate(background_id)[1]


def _save_settings(background_id, model_settings, cg_settings, expected_revision):
    model_settings = _object(model_settings, "model settings")
    cg_settings = _object(cg_settings, "CG settings")
    with background._LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        document["model_settings"] = model_settings
        document["cg_settings"] = cg_settings
        return _commit(directory, document, previous)


def _image_bytes(image):
    if not isinstance(image, Image.Image):
        raise ValueError("CG Creator requires a PIL image")
    _dimensions(*image.size)
    encoded = io.BytesIO()
    image.convert("RGB").save(encoded, format="PNG")
    payload = encoded.getvalue()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("CG Creator image is too large")
    return payload


def background_set_image_path(background_id, set_id, index):
    """Resolve a saved set image using the original Background Creator validation."""
    # Source: ComfyUI_VNCCS nodes/background_creator.py, background_set_image_path.
    _id(set_id)
    if type(index) is not int or not 0 <= index <= 64:
        raise ValueError("Invalid background set image index")
    with background._LOCK:
        directory, document, _previous = _locate(background_id)
        group = next((item for item in document.get("generated_sets", []) if isinstance(item, dict) and item.get("set_id") == set_id), None)
        if group is None or index >= len(group["images"]):
            raise FileNotFoundError("Background set image does not exist")
        path = _contained(directory, group["images"][index]["file"])
        version = group["version"]
        if not isinstance(version, str) or version and not re.fullmatch(r"V[1-9][0-9]*", version):
            raise ValueError("Invalid background archive version")
        expected = _contained(directory, "Images", version) if version else _contained(directory, "Images")
        if path.parent != expected or path.suffix != ".png":
            raise ValueError("Background set image is outside its saved set")
        if not path.is_file():
            raise FileNotFoundError("Background set image file is missing")
        return path


def image_path(background_id, image_id, source_variant=None):
    _id(image_id)
    with background._LOCK:
        directory, document, _previous = _locate(background_id)
        image = next((item for item in document["images"] if item["image_id"] == image_id), None)
        if image is None:
            raise FileNotFoundError("Background image does not exist")
        variant = _variant_source(document, image_id, source_variant)
        if variant:
            return background_set_image_path(background_id, variant["set_id"], variant["index"])
        path = _contained(directory, image["file"])
        if not path.is_file():
            raise FileNotFoundError("Background image file is missing")
        return path


def _asset_file(record, kind):
    relative = record.get("file")
    branch, filename = ("model", "scene.json") if kind == "models" else ("CG", "image.png")
    if (not isinstance(relative, str) or "\\" in relative or relative != Path(relative).as_posix() or len(Path(relative).parts) != 3
            or Path(relative).parts[0] != branch or Path(relative).name != filename):
        raise ValueError("Invalid CG Creator saved asset path")
    _name(Path(relative).parts[1])
    if kind == "cgs" and record.get("raw_file") != str(Path(relative).with_name("raw.png")).replace("\\", "/"):
        raise ValueError("Invalid CG Creator raw image path")
    return relative


def _next_asset_directory(directory, branch, name, *, flatten=None):
    base = _contained(directory, branch)
    name = _name(name)
    occupied = {path.name.casefold() for path in base.iterdir()} if base.exists() else set()
    index = 1
    while True:
        suffix = f" ({index})" if index > 1 else ""
        candidate = _name(name[:96 - len(suffix)] + suffix)
        target = _contained(base, candidate)
        if candidate.casefold() not in occupied or target == flatten and not any((target / name).exists() for name in ("image.png", "raw.png")):
            return target
        index += 1


def _source_label(document, image_id, variant):
    if variant:
        return _name(variant.get("name") or f"Variant {variant['index']}")
    return "Original" if len(document["images"]) == 1 else f"Original {next(index for index, image in enumerate(document['images'], 1) if image['image_id'] == image_id)}"


def _recover_assets(directory):
    pending = _contained(directory, ".cg-assets.pending")
    if not pending.exists():
        return
    if not (pending / "plan.json").exists():
        if any(path.is_symlink() or not path.is_file() or path.name not in {"before.json", "before.backup", "scene.before.json"} and not path.name.endswith(".tmp") for path in pending.iterdir()):
            raise ValueError("Incomplete CG asset publication journal")
        shutil.rmtree(pending)
        return
    plan, _ = _read_json(_contained(pending, "plan.json"))
    kind, identity = plan["kind"], _id(plan["id"])
    if kind not in {"models", "cgs"}:
        raise ValueError("Invalid CG asset publication journal")
    before, _ = _read_json(_contained(pending, "before.json"))
    _validate(before, directory.name)
    record = next((item for item in before.get(kind, []) if item[kind[:-1] + "_id"] == identity), None)
    target_record = plan["record"]
    target = _contained(directory, _asset_file(target_record, kind)).parent
    if target_record[kind[:-1] + "_id"] != identity:
        raise ValueError("Invalid CG asset publication identity")
    source = (_contained(directory, "model", identity) if kind == "models" else _contained(directory, "CG", _name(record["name"]), identity)) if record else pending / "data"
    moves = []
    for old, new in plan["moves"]:
        paths = tuple(_contained(directory, relative) for relative in (old, new))
        old, new = paths
        valid = old == source and new == target
        if record and kind == "cgs":
            valid |= old.parent == source and new == target / old.name
        if record and kind == "models":
            valid |= old.parent == target / "objects" and new.parent == old.parent and bool(_ID.fullmatch(old.name)) and bool(re.fullmatch(r"Object [1-9][0-9]*", new.name))
        if not valid:
            raise ValueError("CG asset publication escaped its owner")
        moves.append(paths)
    path = _contained(directory, f"{directory.name}_config.json")
    current, _ = _read_json(path)
    _validate(current, directory.name)
    if current["background_id"] != before["background_id"]:
        raise ValueError("CG asset publication belongs to another background")
    if any(item.get("file") == target_record["file"] and item.get(kind[:-1] + "_id") != identity for item in current.get(kind, [])):
        raise ValueError("CG asset publication belongs to another asset")
    if kind == "models" and (target / "scene.json").exists():
        scene, _ = _read_json(_contained(target, "scene.json"))
        if scene.get("model_id") != identity or scene.get("background_id") != before["background_id"]:
            raise ValueError("CG asset publication model does not match its owner")
    committed = any(item.get(kind[:-1] + "_id") == identity and item.get("file") == target_record["file"] for item in current.get(kind, []))
    if not committed:
        for old, new in reversed(moves):
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new, old)
        if plan.get("manifest"):
            _atomic(_contained(source, "scene.json"), _contained(pending, "scene.before.json").read_bytes())
        backup = _contained(directory, f"{directory.name}_config.json.backup")
        if plan["had_backup"]:
            _atomic(backup, _contained(pending, "before.backup").read_bytes())
        else:
            backup.unlink(missing_ok=True)
    elif source.exists() and source != pending / "data" and source != target and not any(source.iterdir()):
        source.rmdir()
    shutil.rmtree(pending)


@contextmanager
def _asset_publication(directory, previous, kind, record, moves, *, manifest=None):
    pending = _contained(directory, ".cg-assets.pending")
    pending.mkdir(exist_ok=False)
    try:
        _atomic(pending / "before.json", previous)
        backup = _contained(directory, f"{directory.name}_config.json.backup")
        if backup.exists():
            _atomic(pending / "before.backup", backup.read_bytes())
        if manifest is not None:
            _atomic(pending / "scene.before.json", manifest)
        plan = {"kind": kind, "id": record[kind[:-1] + "_id"], "record": record,
                "had_backup": backup.exists(), "manifest": manifest is not None,
                "moves": [[old.relative_to(directory).as_posix(), new.relative_to(directory).as_posix()] for old, new in moves]}
        _atomic(pending / "plan.json", _json_bytes(plan))
    except BaseException:
        shutil.rmtree(pending)
        raise
    try:
        yield pending
    finally:
        _recover_assets(directory)


def _migrate_assets(directory, document, payload):
    for kind in ("models", "cgs"):
        for record in document.get(kind, []):
            if "file" in record:
                continue
            identity = record[kind[:-1] + "_id"]
            source = _contained(directory, "model", identity) if kind == "models" else _contained(directory, "CG", record["name"], identity)
            if not source.exists():
                continue
            variant = record.get("source_variant")
            name = _source_label(document, record["source_image_id"], variant) if kind == "models" else record["name"]
            target = _next_asset_directory(directory, "model" if kind == "models" else "CG", name, flatten=source.parent if kind == "cgs" else None)
            updated = {**record, "file": (target / ("scene.json" if kind == "models" else "image.png")).relative_to(directory).as_posix()}
            if kind == "cgs":
                updated["raw_file"] = (target / "raw.png").relative_to(directory).as_posix()
            moves, scene, original = [(source, target)], None, None
            if kind == "models":
                scene, original = _read_json(_contained(source, "scene.json"))
                if scene.get("model_id") != identity or scene.get("background_id") != document["background_id"]:
                    raise ValueError("Legacy model does not match its saved owner")
                for index, item in enumerate(scene["objects"], 1):
                    old = f"objects/{_id(item['object_id'])}"
                    new = f"objects/Object {index}"
                    _contained(source, old)
                    if _contained(source, new).exists():
                        raise FileExistsError("Legacy object migration target already exists")
                    moves.append((_contained(target, old), _contained(target, new)))
                    for key, relative in item["files"].items():
                        if not isinstance(relative, str) or not relative.startswith(old + "/"):
                            raise ValueError("Legacy model asset does not match its object")
                        _contained(source, relative)
                        item["files"][key] = new + relative[len(old):]
            elif target == source.parent:
                if any(_contained(target, item.name).exists() for item in source.iterdir()):
                    target = _next_asset_directory(directory, "CG", name)
                    updated.update(file=(target / "image.png").relative_to(directory).as_posix(), raw_file=(target / "raw.png").relative_to(directory).as_posix())
                    moves = [(source, target)]
                else:
                    moves = [(item, _contained(target, item.name)) for item in source.iterdir()]
            with _asset_publication(directory, payload, kind, updated, moves, manifest=original):
                for old, new in moves:
                    new.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(old, new)
                if scene is not None:
                    _atomic(_contained(target, "scene.json"), _json_bytes(scene))
                record.update(updated)
                # Recovery backups resolve the renamed assets without requiring old ID folders.
                document = _commit(directory, document, _json_bytes(document))
            payload = _contained(directory, f"{directory.name}_config.json").read_bytes()
    return document, payload


def _model_directory(directory, model_id, document):
    model = next((item for item in document.get("models", []) if item["model_id"] == model_id), None)
    if model is None:
        raise FileNotFoundError("Background model variant does not exist")
    return _contained(directory, _asset_file(model, "models")).parent if "file" in model else _contained(directory, "model", _id(model_id))


def _copy_model(source, scene, target):
    objects = scene.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 32:
        raise ValueError("Background model must contain Gaussian objects")
    copies, total, object_ids = [], 0, set()
    for index, item in enumerate(objects, 1):
        if not isinstance(item, dict) or _id(item.get("object_id")) in object_ids:
            raise ValueError("Background model object is invalid or duplicated")
        object_id = item["object_id"]
        object_ids.add(object_id)
        files = item.get("files")
        if not isinstance(files, dict) or not files.get("ply"):
            raise ValueError("Background model PLY is missing")
        retained = {}
        for kind in _ASSET_KINDS:
            relative = files.get(kind)
            if not relative:
                continue
            if not isinstance(relative, str) or Path(relative).is_absolute() or ":" in relative or relative.startswith("\\"):
                raise ValueError("Invalid background model asset path")
            source_file = _contained(source, relative.replace("\\", "/"))
            if not source_file.is_file():
                raise FileNotFoundError("Background model asset is missing")
            total += source_file.stat().st_size
            if total > MAX_MODEL_BYTES:
                raise ValueError("Background model assets are too large")
            suffix = {"ply": ".ply", "splat": ".splat", "reference": ".png", "prepared": ".png"}[kind]
            target_relative = f"objects/Object {index}/{kind}{suffix}"
            retained[kind] = target_relative
            copies.append((source_file, target_relative))
        item["files"] = retained
        item.pop("urls", None)
    encoded = _json_bytes(scene)
    copied_bytes = 0
    for source_file, relative in copies:
        copied_bytes += source_file.stat().st_size
        if copied_bytes > MAX_MODEL_BYTES:
            raise ValueError("Background model assets are too large")
        _copy_asset(source_file, _contained(target, relative), MAX_MODEL_BYTES)
    _atomic(_contained(target, "scene.json"), encoded)


def commit_model(background_id, image_id, scene_directory, scene, settings, expected_revision, source_variant=None):
    _id(image_id)
    scene = _object(scene, "model scene")
    settings = _object(settings, "model settings")
    with background._LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        if not any(image["image_id"] == image_id for image in document["images"]):
            raise ValueError("Model source image does not exist")
        variant = _variant_source(document, image_id, source_variant)
        if variant:
            image_path(background_id, image_id, variant)
        if len(document.get("models", [])) >= MAX_MODELS:
            raise ValueError("Background model limit reached")
        model_id = uuid.uuid4().hex
        target = _next_asset_directory(directory, "model", _source_label(document, image_id, variant))
        scene.update(background_id=background_id, model_id=model_id, source_image_id=image_id,
                     background_settings=settings)
        scene.pop("source_variant", None)
        if variant:
            scene["source_variant"] = variant
        for key in ("exports", "preview", "capture_set", "reference", "skydome", "textures"):
            scene.pop(key, None)
        metadata = {"model_id": model_id, "source_image_id": image_id,
                    "scene_id": scene.get("scene_id", ""), "settings": settings,
                    "file": (target / "scene.json").relative_to(directory).as_posix()}
        if variant:
            metadata["source_variant"] = variant
        stage = _contained(directory, ".cg-assets.pending", "data")
        with _asset_publication(directory, previous, "models", metadata, [(stage, target)]):
            _copy_model(Path(scene_directory), scene, stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, target)
            document.setdefault("models", []).append(metadata)
            return _commit(directory, document, previous)


def model_scene(background_id, model_id):
    with background._LOCK:
        directory, document, _previous = _locate(background_id)
        model = next((model for model in document.get("models", []) if model["model_id"] == model_id), None)
        if model is None:
            raise FileNotFoundError("Background model variant does not exist")
        path = _contained(_model_directory(directory, model_id, document), "scene.json")
        scene, _payload = _read_json(path)
        if (not isinstance(scene, dict) or scene.get("background_id") != background_id or scene.get("model_id") != model_id
                or scene.get("source_image_id") != model["source_image_id"]
                or _variant_key(scene.get("source_variant")) != _variant_key(model.get("source_variant"))):
            raise BackgroundStorageError("Background model manifest does not match its owner")
        return scene


def model_asset_path(background_id, model_id, object_id, kind):
    _id(object_id)
    if kind not in _ASSET_KINDS:
        raise ValueError("Unsupported background model asset kind")
    with background._LOCK:
        scene = model_scene(background_id, model_id)
        item = next((item for item in scene.get("objects", []) if item.get("object_id") == object_id), None)
        if item is None:
            raise FileNotFoundError("Background model object does not exist")
        directory, document, _previous = _locate(background_id)
        directory = _model_directory(directory, model_id, document)
        relative = item.get("files", {}).get(kind)
        if kind == "splat" and not relative:

            _ensure_cached_splat = provider()["storage"]._ensure_cached_splat

            ply = model_asset_path(background_id, model_id, object_id, "ply")
            return _ensure_cached_splat(ply, ply_sha256=str(item.get("checksums", {}).get("ply_sha256") or ""))
        if not isinstance(relative, str):
            raise FileNotFoundError("Background model asset does not exist")
        path = _contained(directory, relative)
        if not path.is_file():
            raise FileNotFoundError("Background model asset file is missing")
        return path


def commit_cg(background_id, name, raw, finished, camera, settings, model_id, expected_revision):
    name, model_id = _name(name), _id(model_id)
    camera, settings = _camera(camera), _object(settings, "CG settings")
    raw_bytes, finished_bytes = _image_bytes(raw), _image_bytes(finished)
    with background._LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        model = model_scene(background_id, model_id)
        if len(document.get("cgs", [])) >= MAX_CGS:
            raise ValueError("Background CG limit reached")
        cg_id = uuid.uuid4().hex
        target = _next_asset_directory(directory, "CG", name)
        metadata = {"cg_id": cg_id, "name": name, "background_id": background_id,
                    "model_id": model_id, "source_image_id": model["source_image_id"], "camera": camera, "settings": settings,
                    "width": finished.width, "height": finished.height,
                    "file": (target / "image.png").relative_to(directory).as_posix(),
                    "raw_file": (target / "raw.png").relative_to(directory).as_posix()}
        variant = _source_variant(model.get("source_variant"))
        if variant:
            metadata["source_variant"] = variant
        _json_bytes(metadata)
        stage = _contained(directory, ".cg-assets.pending", "data")
        with _asset_publication(directory, previous, "cgs", metadata, [(stage, target)]):
            _atomic(_contained(stage, "raw.png"), raw_bytes)
            _atomic(_contained(stage, "image.png"), finished_bytes)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, target)
            document.setdefault("cgs", []).append(metadata)
            return _commit(directory, document, previous)


def _archive_cg(background_id, cg_id, expected_revision):
    cg_id = _id(cg_id)
    with background._LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        cg = next((item for item in document.get("cgs", []) if item["cg_id"] == cg_id), None)
        if cg is None or cg.get("archived"):
            raise FileNotFoundError("Background CG does not exist")
        # Match Background Creator deletion: retain immutable assets used by saved workflows.
        cg["archived"] = True
        return _commit(directory, document, previous)


def cg_image_path(background_id, cg_id, kind="image"):
    _id(cg_id)
    if kind not in {"image", "raw"}:
        raise ValueError("Unsupported background CG image kind")
    with background._LOCK:
        directory, document, _previous = _locate(background_id)
        cg = next((item for item in document.get("cgs", []) if item["cg_id"] == cg_id), None)
        if cg is None:
            raise FileNotFoundError("Background CG does not exist")
        path = (_contained(directory, cg["file"] if kind == "image" else cg["raw_file"]) if "file" in cg
                else _contained(directory, "CG", cg["name"], cg_id, f"{kind}.png"))
        if not path.is_file():
            raise FileNotFoundError("Background CG image file is missing")
        return path

# QI2 refinement of the captured camera frame.

def model_operation_lock():
    return provider()["runtime"]._model_operation_lock()


# Copied from VNCCS Background Creator.ASSETS and Character Creator V2.QI2_DEFAULTS.
ASSETS = {
    "diffusion_models": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
}
TURBO_LORA = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
TURBO_MODEL = "qwen_image_2.1_turbo_int8_convrot.safetensors"
ANYANGLE_REPO = "lilylilith/QI_2.1_AnyAngle"
ANYANGLE_REVISION = "35b746e4121e1e9fc6a3e334fd8036ba1a9560d0"
ANYANGLE_FILENAME = "QI2.1_AnyAngle.safetensors"
ANYANGLE_LORA = "QI2.1/lilylilith/" + ANYANGLE_FILENAME
ANYANGLE_SHA256 = "e4d7a0d44a45ed62daabd0f2484a1a59046618a4c47e76b1eab5e386a7a35d8a"
ANYANGLE_PROMPT = "Change the camera angle from <image2> to <image1>."
# Copied from VNCCS control_center.json; the Creator uses the same model picker entries.
MODEL_PRESETS = [
    {"name": "Qwen Image 2.1 INT8 ConvRot", "type": "unet", "kind": "QI2",
     "hf_repo": "Comfy-Org/Qwen-Image-2.1", "hf_path": "diffusion_models/" + ASSETS["diffusion_models"],
     "local_path": "models/diffusion_models/" + ASSETS["diffusion_models"], "version": "1.0",
     "description": "Qwen Image 2.1 INT8 ConvRot diffusion model."},
    {"name": "Qwen Image 2.1 Turbo INT8 ConvRot", "type": "unet", "kind": "QI2",
     "hf_repo": "Comfy-Org/Qwen-Image-2.1", "hf_path": "diffusion_models/" + TURBO_MODEL,
     "local_path": "models/diffusion_models/" + TURBO_MODEL, "version": "1.0",
     "description": "Qwen Image 2.1 Turbo INT8 ConvRot diffusion model. Uses standard sampling with 8 steps and CFG 1."},
]
# Copied from VNCCS control_center.json; paths resolve only against this ComfyUI installation.
TURBO_LORA_METADATA = {
    "name": "Qwen Image 2.1 Viggle Turbo", "type": "TurboLora", "kind": "QI2",
    "hf_repo": "Viggle/Qwen-Image-2.1-viggle-turbo",
    "hf_path": TURBO_LORA.rsplit("/", 1)[-1], "local_path": "models/loras/" + TURBO_LORA,
    "version": "0.2.1",
    "description": "Six-step Viggle Turbo LoRA for Qwen Image 2.1; runtime adapter and sigma shift are provided by VNCCS.",
}
DEFAULT_SETTINGS = {
    "prompt": ANYANGLE_PROMPT,
    "aspect": "16:9", "resolution_mp": 2.0,
    "generation": {
        "steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "seed": 0,
        "turbo": False, "prompt_enhance": False, "cache_device": "default", "cache_dtype": "default",
        "lora_stack": [{"name": "", "strength": 1.0} for _ in range(5)],
        "diffusion_model_name": ASSETS["diffusion_models"], "seed_mode": "fixed", "turbo_previous_settings": None,
    },
}
ASPECTS = ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "3:2", "2:3")
_ASSET_CACHE = None
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: dict[str, Any] | None = None
_LORA_LOCK = threading.Lock()
_ANYANGLE_LOCK = threading.Lock()
_PROGRESS = {}


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


def normalize_settings(data: Any) -> dict[str, Any]:
    """Validate QI2 controls while retaining bounded future settings as inert JSON."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("CG settings must be an object")
    try:
        encoded = json.dumps(data, allow_nan=False, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 256 * 1024:
            raise ValueError("CG settings are too large")
        data = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("CG settings must contain finite JSON values within 256 KiB") from exc
    settings = {**copy.deepcopy(DEFAULT_SETTINGS), **data}
    if "prompt" in data:
        settings["prompt"] = _text(data["prompt"], "CG prompt", 8000) or DEFAULT_SETTINGS["prompt"]
    settings["aspect"] = data.get("aspect", settings["aspect"])
    if settings["aspect"] not in ASPECTS:
        raise ValueError("Unsupported CG aspect ratio")
    if "resolution_mp" in data:
        settings["resolution_mp"] = _number(data["resolution_mp"], "Resolution", 0.05, 16)
    incoming = data.get("generation", {})
    if not isinstance(incoming, dict):
        raise ValueError("CG generation settings must be an object")
    generation = {**copy.deepcopy(DEFAULT_SETTINGS["generation"]), **incoming}
    settings["generation"] = generation
    generation["diffusion_model_name"] = _model_name(generation["diffusion_model_name"], "Diffusion model")
    # Copied from VNCCS Background Creator.normalize_editor: built-in Turbo uses standard sampling.
    built_in_turbo = _is_turbo_model(generation["diffusion_model_name"])
    if built_in_turbo:
        generation.update(steps=8, cfg=1.0)
    for key, low, high, integer in (("steps", 1, 150, True), ("cfg", 0, 30, False), ("seed", 0, 2**64 - 1, True)):
        if key in incoming:
            generation[key] = _number(incoming[key], key, low, high, integer=integer)
    for key in ("turbo", "prompt_enhance"):
        if key in incoming:
            if not isinstance(incoming[key], bool):
                raise ValueError(f"{key} must be a boolean")
            generation[key] = incoming[key]
    for key in ("sampler", "scheduler"):
        if key in incoming:
            generation[key] = _text(incoming[key], key, 64)
            if not generation[key]:
                raise ValueError(f"{key} must not be empty")
    for key, choices in (("cache_device", ("default", "auto", "gpu", "cpu", "off")),
                         ("cache_dtype", ("default", "int8", "int4")),
                         ("seed_mode", ("fixed", "randomize"))):
        if key in incoming:
            if incoming[key] not in choices:
                raise ValueError(f"Unsupported {key}")
            generation[key] = incoming[key]
    generation.update(cache_device="default", cache_dtype="default", prompt_enhance=False)
    previous = incoming.get("turbo_previous_settings")
    if previous is not None:
        if not isinstance(previous, dict):
            raise ValueError("Previous Turbo settings must be an object")
        generation["turbo_previous_settings"] = {
            "steps": _number(previous.get("steps", 25), "Previous steps", 1, 150, integer=True),
            "cfg": _number(previous.get("cfg", 1), "Previous CFG", 0, 30),
        }
    if built_in_turbo:
        generation["turbo"] = False
    elif generation["turbo"]:
        generation.update(steps=6, cfg=1.0)
    stack = incoming.get("lora_stack", [])
    if not isinstance(stack, list) or len(stack) > 5:
        raise ValueError("LoRA stack must contain at most five entries")
    generation["lora_stack"] = []
    for item in stack:
        if not isinstance(item, dict):
            raise ValueError("LoRA entries must be objects")
        name = _model_name(item.get("name", ""), "LoRA name")
        if name == "None":
            name = ""
        if name.rsplit("/", 1)[-1].casefold() == TURBO_LORA.rsplit("/", 1)[-1].casefold():
            raise ValueError("Use the Turbo switch for the Viggle adapter")
        generation["lora_stack"].append({"name": name, "strength": _number(item.get("strength", 1), "LoRA strength", -10, 10)})
    generation["lora_stack"].extend({"name": "", "strength": 1.0} for _ in range(5 - len(generation["lora_stack"])))
    return settings


def _model_name(value, label):
    name = _text(value, label, 512).replace("\\", "/")
    if name and (name.startswith(("/", "~")) or ":" in name or any(not part or part.endswith((" ", ".")) for part in name.split("/"))):
        raise ValueError(f"{label} must be a relative installed model name")
    return name



def _is_turbo_model(name):
    # Copied from VNCCS Control Center.is_qi2_turbo_model.
    return name.replace("\\", "/").rsplit("/", 1)[-1] == TURBO_MODEL


@inference_stage()
def _call_comfy_node(class_name: str, **kwargs):
    # Copied from VNCCS CharacterGenerator; resolve classes in this ComfyUI registry.
    import nodes as comfy_nodes
    vnccs_node_id = kwargs.pop("_vnccs_node_id", None)
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if comfy_nodes else {}
    cls = mappings.get(class_name)
    if cls is None:
        raise RuntimeError(f"Required node '{class_name}' is not available")
    # Native V3 nodes such as GetImageSize require a request-local hidden holder.
    if hasattr(cls, "PREPARE_CLASS_CLONE"):
        cls = cls.PREPARE_CLASS_CLONE(None)
    instance = cls()
    method_name = getattr(cls, "FUNCTION", None)
    method = getattr(instance, method_name, None) if method_name else None
    if method is None:
        for candidate in ("execute", "process", "process_image", "load_model", "loadmodel", "load", "sample", "decode"):
            method = getattr(instance, candidate, None)
            if method is not None:
                break
    if method is None:
        raise RuntimeError(f"Node '{class_name}' has no callable FUNCTION")
    signature = inspect.signature(method)
    try:
        signature.bind(**kwargs)
    except TypeError as error:
        raise RuntimeError(f"Required node '{class_name}' has an incompatible interface: {error}") from error
    try:
        result = method(**kwargs)
    except AttributeError as exc:
        if vnccs_node_id is None or "'NoneType' object has no attribute 'node_id'" not in str(exc):
            raise
        module = inspect.getmodule(cls) or inspect.getmodule(method)
        context_getter = getattr(module, "get_executing_context", None) if module is not None else None
        if context_getter is None:
            raise
        context = SimpleNamespace(node_id=str(vnccs_node_id))
        setattr(module, "get_executing_context", lambda: context)
        try:
            result = method(**kwargs)
        finally:
            setattr(module, "get_executing_context", context_getter)
    if hasattr(result, "result") and type(result).__name__ == "NodeOutput":
        block_execution = getattr(result, "block_execution", None)
        if block_execution:
            raise RuntimeError(str(block_execution))
        result = result.result
    return result if isinstance(result, tuple) else (result,)


def _installed(category: str) -> list[str]:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list(category) or [])
    except (ImportError, AttributeError, KeyError):
        return []


def _asset_name(category: str, name: str) -> str:
    # Copied from VNCCS Background Creator._resolve_model_name.
    normalized = _model_name(name, "Model name")
    installed = _installed(category)
    exact = [value for value in installed if value.replace("\\", "/") == normalized]
    matches = exact or [value for value in installed if value.replace("\\", "/").rsplit("/", 1)[-1].lower() == normalized.rsplit("/", 1)[-1].lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Install exactly one {name} in ComfyUI models/{category}")
    return matches[0]


def configuration() -> dict[str, Any]:
    # Copied from Character Creator's context lists.
    try:
        import comfy.samplers
        samplers, schedulers = comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
    except (ImportError, AttributeError):
        samplers, schedulers = ["euler"], ["simple"]
    assets = []
    for category, name in ASSETS.items():
        installed = [entry.replace("\\", "/") for entry in _installed(category)]
        matches = [entry for entry in installed if entry.rsplit("/", 1)[-1].lower() == name.lower()]
        assets.append({"category": category, "name": name, "installed": name in installed or len(matches) == 1})
    models = _installed("diffusion_models")
    try:
        turbo_name = _asset_name("loras", TURBO_LORA)
    except FileNotFoundError:
        turbo_name = ""
    turbo = {**TURBO_LORA_METADATA, "filename": turbo_name or TURBO_LORA,
             "status": "installed" if turbo_name else "missing", "installed": bool(turbo_name)}
    presets = copy.deepcopy(MODEL_PRESETS)
    for entry in presets:
        try:
            filename = _asset_name("diffusion_models", entry["local_path"].split("/", 2)[2])
        except FileNotFoundError:
            entry["status"] = "missing"
        else:
            entry.update(local_path="models/diffusion_models/" + filename, status="installed")
    return {"settings": copy.deepcopy(DEFAULT_SETTINGS), "model": "Qwen Image 2.1 INT8 ConvRot",
            "assets": assets, "installed": all(asset["installed"] for asset in assets),
            "model_presets": presets,
            "loras": _installed("loras"), "diffusion_models": models, "turbo_lora": turbo, "aspects": list(ASPECTS),
            "samplers": list(samplers), "schedulers": list(schedulers)}


def _load_clip():
    return _call_comfy_node("CLIPLoader", clip_name=_asset_name("text_encoders", ASSETS["text_encoders"]),
                            type="qwen_image", device="default")[0]


def _model_file_signature(full_path):
    # Copied from VNCCS Control Center._model_file_signature.
    try:
        stat = os.stat(full_path)
    except OSError:
        return None
    return os.path.realpath(full_path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@inference_stage()
def _load_assets(model_name=ASSETS["diffusion_models"]):
    # Copied from VNCCS Background Creator._load_assets; QI2 is the only CG family.
    import folder_paths
    global _ASSET_CACHE
    key = tuple(_asset_name(category, model_name if category == "diffusion_models" else name) for category, name in ASSETS.items())
    signature = (key, tuple(_model_file_signature(folder_paths.get_full_path(category, name) or "") for category, name in zip(ASSETS, key)))
    if _ASSET_CACHE is None or _ASSET_CACHE[0] != signature:
        model = _call_comfy_node("UNETLoader", unet_name=key[0], weight_dtype="default")[0]
        clip = _load_clip()
        vae = _call_comfy_node("VAELoader", vae_name=key[2])[0]
        _ASSET_CACHE = (signature, model, vae)
    else:
        _, model, vae = _ASSET_CACHE
        # Qwen3-VL generation mutates shared CLIP internals; clone() cannot isolate them.
        clip = _load_clip()
    return model.clone(), clip, vae


# Copied from VNCCS Control Center's four-entry LoRA file cache.
@lru_cache(maxsize=4)
def _cached_lora_file(full_path, signature):
    import comfy.utils
    return comfy.utils.load_torch_file(full_path, safe_load=True, return_metadata=True)


def _load_lora_file(full_path):
    signature = _model_file_signature(full_path)
    if signature is None:
        import comfy.utils
        return comfy.utils.load_torch_file(full_path, safe_load=True, return_metadata=True)
    with _LORA_LOCK:
        return _cached_lora_file(full_path, signature)


def _get_lora_full_path(name):
    import folder_paths
    return get_full_path_agnostic(folder_paths, "loras", name, require_exists=True)


def apply_generation_lora(model, clip, name, strength, clip_strength=None):
    # Copied from VNCCS Character Creator V2.apply_generation_lora; local path/cache bindings only.
    strength = float(strength or 0)
    if not name or name == "None" or strength == 0:
        return model, clip
    path = _get_lora_full_path(name)
    if not path:
        raise FileNotFoundError(f"Selected LoRA is not installed: {name}. Install it or disable it before generating.")
    import comfy.sd
    state, _metadata = _load_lora_file(path)
    result = comfy.sd.load_lora_for_models(model, clip, state, strength,
                                         strength if clip_strength is None else float(clip_strength))
    log_event("lora_applied", component="CGCreator", path=path,
              strength_model=strength, model_patch_keys=len(result[0].patches))
    return result


def _validate_anyangle_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ANYANGLE_SHA256:
        raise ValueError("AnyAngle LoRA is invalid or incomplete; restore the published QI2.1_AnyAngle.safetensors file")


def resolve_anyangle_lora(cancel_event):
    """Install the pinned public adapter into Control Center's QI2.1 family directory."""
    import folder_paths

    # Copied from Control Center's Hub download and atomic model installation path.
    with _ANYANGLE_LOCK:
        _check_cancel(cancel_event)
        try:
            installed = _asset_name("loras", ANYANGLE_LORA)
        except FileNotFoundError:
            installed = ""
        if installed:
            path = _get_lora_full_path(installed)
            if path:
                _validate_anyangle_file(path)
                return installed
        roots = _safe_get_model_folder_paths(folder_paths, "loras")
        if not roots:
            raise RuntimeError("No ComfyUI LoRA folder is configured")
        root = os.path.realpath(roots[0])
        target = os.path.realpath(os.path.join(root, ANYANGLE_LORA))
        if os.path.commonpath((root, target)) != root:
            raise ValueError("AnyAngle installation must stay inside the configured LoRA folder")
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=ANYANGLE_REPO, filename=ANYANGLE_FILENAME,
                                 revision=ANYANGLE_REVISION, token=False)
        _check_cancel(cancel_event)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".anyangle-", suffix=".tmp", dir=os.path.dirname(target))
        try:
            with os.fdopen(descriptor, "wb") as output, open(cached, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    _check_cancel(cancel_event)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            _validate_anyangle_file(temporary)
            _check_cancel(cancel_event)
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return ANYANGLE_LORA


def _resolve_seed(generation):
    # Copied from Character Creator V2.resolve_generation_seed and VNCCS.utils.generate_seed.
    seed = int(generation.get("seed", 0) or 0)
    if generation.get("seed_mode", "fixed") == "randomize":
        seed = random.getrandbits(64)
        if seed == 0:
            seed = random.getrandbits(64) or 1
    return seed


def _viggle_turbo_sigmas(latent):
    # Copied from VNCCS nodes/qi2_viggle.py; uses the actual sampling latent's token count.
    import torch
    samples = latent["samples"]
    spatial_ratio = latent.get("downscale_ratio_spacial", 16) / 16
    tokens = round(samples.shape[-2] * spatial_ratio) * round(samples.shape[-1] * spatial_ratio)
    shift = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
    nodes = torch.tensor((1.0, 0.9375, 0.875, 0.75, 0.5, 0.25), dtype=torch.float64)
    shifted = math.exp(shift) / (math.exp(shift) + (1 / nodes - 1))
    return torch.cat((shifted, shifted.new_zeros(1))).float()


@inference_stage()
def _sample_generation_latent(model, positive, negative, latent, sampler, turbo=False):
    # Copied from VNCCS CharacterGenerator._qi2_sample; no character-specific processing.
    if not turbo:
        return _call_comfy_node("KSampler", model=model, positive=positive, negative=negative,
                                latent_image=latent, **sampler)[0]
    if int(sampler["steps"]) != 6 or abs(float(sampler["cfg"]) - 1.0) > 1e-6:
        raise ValueError("Viggle Turbo requires six steps and CFG 1")
    if float(sampler.get("denoise", 1.0)) != 1.0:
        raise ValueError("Viggle Turbo requires denoise 1.0")
    noise = _call_comfy_node("RandomNoise", noise_seed=sampler["seed"])[0]
    guider = _call_comfy_node("BasicGuider", model=model, conditioning=positive)[0]
    sampler_node = _call_comfy_node("KSamplerSelect", sampler_name="euler")[0]
    sigmas = _viggle_turbo_sigmas(latent)
    return _call_comfy_node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler_node,
                            sigmas=sigmas, latent_image=latent)[0]


def _check_cancel(cancel_event) -> None:
    if cancel_event.is_set():
        raise GenerationCancelled("CG generation cancelled")


def _stage(draw_id: str, cancel_event, stage: str, value: float, message: str) -> None:
    _check_cancel(cancel_event)
    _set_progress(draw_id, stage, value, message)


def _set_progress(draw_id, stage, value, message):
    with _ACTIVE_LOCK:
        previous = _PROGRESS.get(draw_id, {})
        started = previous.get("started", time.monotonic())
        _PROGRESS[draw_id] = {"stage": stage, "progress": value, "message": message, "started": started}
        while len(_PROGRESS) > 64:
            _PROGRESS.pop(next(iter(_PROGRESS)))
    if message != previous.get("message"):
        log_stage(stage, "done" if value == 1 else "running", component="CGCreator",
                  current=value, total=1, job_id=draw_id, message=message)


def progress(draw_id: str) -> dict[str, Any]:
    with _ACTIVE_LOCK:
        return dict(_PROGRESS.get(draw_id, {"stage": "unknown", "progress": 0}))


def _set_interrupt(value):
    try:
        import comfy.model_management
        comfy.model_management.interrupt_current_processing(value)
    except (ImportError, AttributeError):
        pass


def _interrupt_types():
    try:
        from comfy.model_management import InterruptProcessingException
    except ImportError:
        return ()
    return (InterruptProcessingException,)


@contextmanager
def _report_sampling_progress(draw_id, steps):
    import comfy.utils
    original = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    owner = threading.get_ident()

    def hook(current, total, *args, **kwargs):
        if threading.get_ident() != owner:
            if callable(original):
                return original(current, total, *args, **kwargs)
            return None
        total = max(1, int(total or steps)); current = min(max(int(current), 0), total)
        _set_progress(draw_id, "sampling", 0.35 + 0.5 * current / total, f"Sampling {current}/{total}")

    _set_progress(draw_id, "sampling", 0.35, f"Sampling 0/{steps}")
    comfy.utils.PROGRESS_BAR_HOOK = hook
    try:
        yield
    finally:
        comfy.utils.PROGRESS_BAR_HOOK = original


def interrupt(draw_id: str) -> bool:
    """Only signal ComfyUI while this job owns the shared inference slot."""
    with _ACTIVE_LOCK:
        if _ACTIVE is None or _ACTIVE["draw_id"] != draw_id:
            return False
        _ACTIVE["cancel_event"].set()
        _ACTIVE["interrupted"] = True
        _set_interrupt(True)
        return True


@contextmanager
def _model_session(draw_id: str, cancel_event):
    global _ACTIVE
    lock = model_operation_lock()
    _check_cancel(cancel_event)
    while not lock.acquire(timeout=0.25):
        _check_cancel(cancel_event)
    active = {"draw_id": draw_id, "cancel_event": cancel_event, "interrupted": False}
    try:
        with _ACTIVE_LOCK:
            _check_cancel(cancel_event)
            _set_interrupt(False)
            _ACTIVE = active
        _check_cancel(cancel_event)
        yield
    except BaseException as exc:
        cancelled = cancel_event.is_set() or isinstance(exc, (GenerationCancelled,) + _interrupt_types())
        _set_progress(draw_id, "cancelled" if cancelled else "error", 0, str(exc))
        if cancelled and not isinstance(exc, GenerationCancelled):
            raise GenerationCancelled("CG generation cancelled") from exc
        raise
    finally:
        with _ACTIVE_LOCK:
            if _ACTIVE is active:
                if active["interrupted"]:
                    _set_interrupt(False)
                _ACTIVE = None
        try:
            cleanup_runtime()
        finally:
            lock.release()


def _pil_to_tensor(image):
    # Copied from Character Creator.pil2tensor.
    import numpy as np
    import torch
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _tensor_to_pil(tensor):
    # Copied from Character Creator.tensor2pil.
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")


@logged_operation("CGCreator", "refine")
def generate_image(settings: dict[str, Any], draw_id: str, cancel_event, reference, background):
    from PIL import Image
    settings = normalize_settings(settings)
    if not isinstance(reference, Image.Image):
        raise ValueError("CG refinement requires a captured frame")
    if not isinstance(background, Image.Image):
        raise ValueError("AnyAngle requires the original background as image 1")
    width, height = reference.size
    if any(value < 64 or value > 4096 or value % 16 for value in (width, height)) or width * height > 16_000_000:
        raise ValueError("Captured frame dimensions must be multiples of 16, between 64 and 4096, and at most 16 MP")
    generation = settings["generation"]
    import torch
    prompt = ANYANGLE_PROMPT
    with _model_session(draw_id, cancel_event), torch.inference_mode():
        generation["seed"] = _resolve_seed(generation)
        _stage(draw_id, cancel_event, "loading", 0.01, "Installing AnyAngle LoRA if needed")
        anyangle_lora = resolve_anyangle_lora(cancel_event)
        _stage(draw_id, cancel_event, "loading", 0.03, "Loading Qwen Image 2.1")
        model, clip, vae = _load_assets(generation["diffusion_model_name"])
        _stage(draw_id, cancel_event, "loading", 0.12, "Applying CG model settings")
        model, clip = apply_generation_lora(model, clip, anyangle_lora, 1.0)
        for item in generation["lora_stack"]:
            if item["name"].replace("\\", "/").rsplit("/", 1)[-1].casefold() == ANYANGLE_FILENAME.casefold():
                continue
            model, clip = apply_generation_lora(model, clip, item["name"], item["strength"])
            _check_cancel(cancel_event)
        if generation["turbo"]:
            model = _call_comfy_node("LoraLoaderModelOnly", model=model, lora_name=_asset_name("loras", TURBO_LORA), strength_model=1.0)[0]
        # The cache provider calls its default device selection "auto".
        model = _call_comfy_node("QwenImage21Cache", model=model, device="auto", dtype="default")[0]
        _stage(draw_id, cancel_event, "conditioning", 0.3, "Encoding the original background and camera frame")
        background, reference = background.convert("RGB"), reference.convert("RGB")
        for slot, source, image in (("image_1", "original background", background),
                                    ("image_2", "captured camera frame", reference)):
            log_event("reference_image", job_id=draw_id, slot=slot, source=source,
                      width=image.width, height=image.height, rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        log_event("conditioning", job_id=draw_id, prompt=prompt, negative_prompt="",
                  resolution=1024, cache_device="auto", cache_dtype="default", truncate=False)
        background_tensor, reference_tensor = _pil_to_tensor(background), _pil_to_tensor(reference)
        encoded = _call_comfy_node("TextEncodeQwenImage21", clip=clip, vae=vae, prompt=prompt,
            negative_prompt="", resolution=1024,
            images={"image_1": background_tensor, "image_2": reference_tensor})
        if not isinstance(encoded, (tuple, list)) or len(encoded) < 3 or encoded[0] is None or encoded[1] is None:
            raise RuntimeError("QI2 returned invalid image conditioning; update ComfyUI")
        positive, negative = encoded[:2]
        scaled_reference = _call_comfy_node("ImageScaleToTotalPixels", image=reference_tensor,
            upscale_method="nearest-exact", megapixels=settings["resolution_mp"], resolution_steps=1)[0]
        latent_width, latent_height, _ = _call_comfy_node("GetImageSize", image=scaled_reference)
        log_event("latent", job_id=draw_id, width=latent_width, height=latent_height,
                  source="image_2", resolution_mp=settings["resolution_mp"], upscale_method="nearest-exact", resolution_steps=1)
        latent = _call_comfy_node("EmptyLatentImage", width=latent_width, height=latent_height, batch_size=1)[0]
        _stage(draw_id, cancel_event, "sampling", 0.35, "Rendering the CG frame")
        sampler_settings = {
            "seed": generation["seed"], "steps": generation["steps"], "cfg": generation["cfg"],
            "sampler_name": generation["sampler"], "scheduler": generation["scheduler"], "denoise": 1.0,
        }
        sampler_name = "SamplerCustomAdvanced" if generation["turbo"] else "KSampler"
        log_event("sampling", job_id=draw_id, sampler=sampler_name,
                  model=generation["diffusion_model_name"], settings=sampler_settings)
        with _report_sampling_progress(draw_id, generation["steps"]):
            samples = _sample_generation_latent(model, positive, negative, latent, sampler_settings,
                                                turbo=generation["turbo"])
        _stage(draw_id, cancel_event, "decoding", 0.9, "Decoding the CG image")
        image = _tensor_to_pil(_call_comfy_node("VAEDecode", vae=vae, samples=samples)[0])
        _stage(draw_id, cancel_event, "complete", 1, "CG image ready")
        return image, {"settings": settings, "prompt": prompt, "width": image.width, "height": image.height,
                       "seed": generation["seed"], "model": "qi2",
                       "anyangle_lora": {"name": anyangle_lora, "strength": 1.0, "revision": ANYANGLE_REVISION}}

# Workflow node, workspace jobs and ComfyUI HTTP registration.

EMPTY_STATE = '{"schema_version":1,"background_id":"","selected_image_id":"","model_id":"","selected_cg_id":""}'


def parse_state(raw):
    if not isinstance(raw, str) or len(raw) > 256 * 1024:
        raise ValueError("CG Creator state is invalid or too large")
    try:
        state = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("CG Creator state is not valid JSON") from exc
    if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        raise ValueError("Unsupported CG Creator state version")
    for key in ("background_id", "selected_image_id", "model_id", "selected_cg_id"):
        value = state.get(key, "")
        if not isinstance(value, str) or (value and not _ID.fullmatch(value)):
            raise ValueError(f"CG Creator {key} is invalid")
    if state.get("camera") is not None:
        _camera(state["camera"])
    if state.get("selected_variant") is not None:
        _source_variant(state["selected_variant"])
    return state


class VNCCS_CGCreator:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_cg"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True
    DESCRIPTION = "Select a saved background, build its TripoSplat model and compose CG camera views. Workflow execution returns the selected finished CG."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cg_data": ("STRING", {"default": EMPTY_STATE, "multiline": True, "dynamicPrompts": False})}}

    @classmethod
    def VALIDATE_INPUTS(cls, cg_data=EMPTY_STATE):
        try:
            parse_state(cg_data)
        except ValueError as exc:
            return str(exc)
        return True

    def create_cg(self, cg_data=EMPTY_STATE):
        state = parse_state(cg_data)
        if not state.get("background_id") or not state.get("selected_cg_id"):
            raise ValueError("Generate and select a finished CG inside VNCCS CG Creator before running the workflow")
        return (background._preview_tensor(cg_image_path(state["background_id"], state["selected_cg_id"])),)


NODE_CLASS_MAPPINGS = {"VNCCS_CGCreator": VNCCS_CGCreator}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCS_CGCreator": "VNCCS CG Creator"}


def provider():
    import nodes as comfy_nodes

    factory = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("VNCCS_3DFactory")
    accessor = getattr(factory, "cg_provider", None)
    if not callable(accessor):
        module = inspect.getmodule(factory)
        package = sys.modules.get(getattr(module, "__package__", ""))
        runtime, storage = getattr(package, "runtime", None), getattr(package, "storage", None)
        if runtime is None or storage is None or not getattr(module, "__file__", None):
            raise RuntimeError("The installed 3D Factory runtime is unavailable. Enable 3D Factory and restart ComfyUI")
        # Reuse the registered Factory modules and locks without requiring a new Utils bridge.
        node_file = Path(module.__file__).resolve()
        for directory in getattr(comfy_nodes, "EXTENSION_WEB_DIRS", {}).values():
            web_root = Path(directory).resolve()
            if web_root.parent in node_file.parents:
                return {"version": 1, "runtime": runtime, "storage": storage, "web_root": web_root}
        raise RuntimeError("The installed 3D Factory web extension is unavailable. Restart ComfyUI")
    value = accessor()
    if not isinstance(value, dict) or type(value.get("version")) is not int or value.get("version") != 1:
        raise RuntimeError("Unsupported 3D Factory runtime interface")
    return value


def frontend_configuration():
    import nodes as comfy_nodes

    try:
        value = provider()
        web_root = Path(value["web_root"]).resolve()
        for name, directory in getattr(comfy_nodes, "EXTENSION_WEB_DIRS", {}).items():
            if Path(directory).resolve() == web_root:
                return {"version": 1, "module_url": f"/extensions/{quote(name, safe='')}/factory3d/viewer.js"}
        raise RuntimeError("The installed 3D Factory web extension is unavailable. Restart ComfyUI")
    except RuntimeError as error:
        return {"version": 1, "error": str(error)}


API_BASE = "/vnccs/cg-creator"
MAX_REQUEST_BYTES = 34 * 1024 * 1024
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vnccs-cg")
_JOBS = {}
_JOBS_LOCK = threading.RLock()


def public_document(document):
    value = copy.deepcopy(document)
    base = f"{API_BASE}/backgrounds/{value['background_id']}"
    for key in ("images", "models", "cgs", "generated_sets"):
        value.setdefault(key, [])
    value["cgs"] = [cg for cg in value["cgs"] if not cg.get("archived")]
    latest = value["models"][-1] if value["models"] else {}
    value["model_settings"] = model_settings(value.get("model_settings", latest.get("settings", {})))
    legacy_prompt = value.get("editor", {}).get("cg_prompt")
    fallback = {"prompt": legacy_prompt} if isinstance(legacy_prompt, str) and legacy_prompt.strip() else {}
    value["cg_settings"] = normalize_settings(value.get("cg_settings", fallback))
    for image in value["images"]:
        image["url"] = f"{base}/images/{image['image_id']}"
        if "preview_revision" in image:
            image["url"] += f"?v={image['preview_revision']}"
    for group in value["generated_sets"]:
        if not isinstance(group, dict) or not isinstance(group.get("images"), list):
            continue
        try:
            set_id = _id(group.get("set_id"))
        except ValueError:
            continue
        for index, image in enumerate(group["images"][:65]):
            if isinstance(image, dict):
                image["url"] = f"{base}/sets/{set_id}/{index}"
                if "revision" in image:
                    image["url"] += f"?v={image['revision']}"
    for cg in value["cgs"]:
        cg["url"] = f"{base}/cg/{cg['cg_id']}/image"
    return value


def public_model(background_id, model_id):
    scene = model_scene(background_id, model_id)
    for item in scene.get("objects", []):
        item["urls"] = {"splat": f"{API_BASE}/backgrounds/{background_id}/models/{model_id}/objects/{item['object_id']}/splat"}
        item["locked"] = True
        item.pop("files", None)
    return scene


def model_settings(value):
    factory = provider()["storage"]
    if not isinstance(value, dict):
        raise ValueError("TripoSplat settings must be an object")
    return factory._generation_settings({**value, "remove_background": "0", "provider": "triposplat"})


def ensure_idle(background_id):
    with _JOBS_LOCK:
        if any(job["background_id"] == background_id and job["status"] in {"queued", "running"} for job in _JOBS.values()):
            raise RuntimeError("This CG workspace is busy. Wait for the job or stop it first.")
    owner = background_owner()
    if owner is not None and callable(getattr(owner, "ensure_idle", None)):
        owner.ensure_idle(background_id)


@contextmanager
def _inference_slot(job):
    runtime = provider()["runtime"]
    owner = background_owner()
    lock = getattr(owner, "inference_lock", None)
    acquired = False
    try:
        if lock is not None:
            while not lock.acquire(timeout=0.25):
                runtime._check_cancel(job)
            acquired = True
        runtime._check_cancel(job)
        yield
    finally:
        if acquired:
            lock.release()


def save_settings(background_id, payload):
    if not isinstance(payload, dict):
        raise ValueError("CG settings must be an object")
    model = model_settings(payload.get("model_settings", {}))
    settings = normalize_settings(payload.get("cg_settings", {}))
    with _JOBS_LOCK:
        ensure_idle(background_id)
        return public_document(_save_settings(background_id, model, settings, None))


def delete_cg(background_id, cg_id, payload):
    if not isinstance(payload, dict):
        raise ValueError("CG deletion must be an object")
    with _JOBS_LOCK:
        ensure_idle(background_id)
        return public_document(_archive_cg(background_id, cg_id, None))


def start_job(background_id, payload, raw_image=None):
    backend = provider()
    runtime, factory = backend["runtime"], backend["storage"]
    if not isinstance(payload, dict):
        raise ValueError("CG job must be an object")
    action = payload.get("action")
    if action not in {"model", "cg"}:
        raise ValueError("CG Creator supports only model reconstruction and CG refinement")
    request_id = _request_id(payload["request_id"]) if "request_id" in payload else ""
    image_id = payload.get("image_id", "")
    source_variant = copy.deepcopy(_source_variant(payload.get("source_variant")))
    image_path(background_id, image_id, source_variant)
    values = {"image_id": image_id, "source_variant": source_variant,
              "model_settings": model_settings(payload.get("model_settings", {})),
              "settings": normalize_settings(payload.get("settings", {}))}
    if action == "cg":
        values.update(name=_name(payload.get("name")), camera=_camera(payload.get("camera")), model_id=payload.get("model_id", ""))
        scene = model_scene(background_id, values["model_id"])
        if (scene["source_image_id"] != image_id
                or _variant_key(scene.get("source_variant")) != _variant_key(source_variant)):
            raise ValueError("The model belongs to another background variant")
        raw = factory._decode_image(raw_image or b"").convert("RGB")
        if min(raw.size) < 64 or max(raw.size) > 4096 or raw.width % 16 or raw.height % 16 or raw.width * raw.height > 16_000_000:
            raise ValueError("CG frame must be 64–4096 pixels, aligned to 16 pixels, at most 16 MP")
        values["raw"] = raw
    with _JOBS_LOCK:
        ensure_idle(background_id)
        document = load_background(background_id)
        if document.get("_recovery") or document.get("archived"):
            raise ValueError("This saved background is read-only")
        values["revision"] = document["revision"]
        for old_id in list(_JOBS):
            if len(_JOBS) < 64:
                break
            if _JOBS[old_id]["status"] not in {"queued", "running"}:
                del _JOBS[old_id]
        if len(_JOBS) >= 64:
            raise RuntimeError("CG job queue is full. Wait for a job to finish before trying again.")
        job = runtime._new_job(f"cg_creator_{action}")
        job.update(background_id=background_id, action=action, request_id=request_id)
        _JOBS[job["job_id"]] = job
        _WORKER.submit(runtime._run_job, job, lambda current: _execute(current, values))
    return job_status(job["job_id"])


def _execute(job, values):
    backend = provider()
    runtime, factory = backend["runtime"], backend["storage"]
    temporary_scene_id = None
    try:
        with _inference_slot(job):
            if job["action"] == "cg":
                with Image.open(image_path(job["background_id"], values["image_id"], values["source_variant"])) as source:
                    background = source.convert("RGB")
                image, settings = generate_image(values["settings"], job["job_id"], job["cancel_event"], values["raw"], background)
                runtime._check_cancel(job)
                with _JOBS_LOCK:
                    runtime._check_cancel(job)
                    document = commit_cg(job["background_id"], values["name"], values["raw"], image, values["camera"],
                        settings, values["model_id"], values["revision"])
                    job["published"] = True
            else:
                scene = factory.create_scene("CG Creator")
                temporary_scene_id = job["scene_id"] = scene["scene_id"]
                with Image.open(image_path(job["background_id"], values["image_id"], values["source_variant"])) as source:
                    encoded = io.BytesIO()
                    source.convert("RGB").save(encoded, format="PNG")
                runtime._generate_object(job, encoded.getvalue(), uuid.uuid4().hex, "Background", {**values["model_settings"], "capture_source_camera": True})
                runtime._check_cancel(job)
                scene = factory.load_scene(scene["scene_id"])
                scene = factory.update_scene(scene["scene_id"], {"camera": scene["objects"][0]["source_camera"], "lighting": {"preset": "off", "background": "#000000"}})
                with _JOBS_LOCK:
                    runtime._check_cancel(job)
                    document = commit_model(job["background_id"], values["image_id"], factory.resolve_scene_dir(scene["scene_id"]),
                        scene, values["model_settings"], values["revision"], source_variant=values["source_variant"])
                    job["published"] = True
        return {"document": public_document(document)}
    except GenerationCancelled as exc:
        raise runtime.JobCancelled(str(exc)) from exc
    finally:
        if temporary_scene_id:
            job["scene_id"] = ""
            try:
                factory.delete_scene(temporary_scene_id)
            except Exception as error:
                log_event("temporary_scene_cleanup_failed", component="CGCreator",
                          level="error", job_id=job["job_id"], scene_id=temporary_scene_id, error=str(error))


def job_status(job_id):
    runtime = provider()["runtime"]
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise FileNotFoundError("CG job was not found")
        value = runtime._job_public(job)
    if job["status"] == "running" and job["action"] == "cg":
        snapshot = progress(job_id)
        if snapshot.get("stage") != "unknown":
            value.update(message=snapshot.get("message", value["message"]), progress=100 * snapshot.get("progress", 0))
    return value


def request_status(background_id, request_id):
    background_id, request_id = _id(background_id), _request_id(request_id)
    with _JOBS_LOCK:
        job_id = next((job["job_id"] for job in reversed(list(_JOBS.values()))
                       if job["background_id"] == background_id and job.get("request_id") == request_id), None)
    if job_id is None:
        raise FileNotFoundError("CG generation request was not found")
    return job_status(job_id)


def cancel_job(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise FileNotFoundError("CG job was not found")
        if not job.get("published") and job["status"] in {"queued", "running"}:
            job["cancel_event"].set()
            if job["action"] == "cg":
                interrupt(job_id)
    return job_status(job_id)


_REGISTERED = False


def register_routes(routes):
    global _REGISTERED
    if _REGISTERED:
        return
    from aiohttp import web
    base = API_BASE

    async def perform(callback, *args):
        try:
            return web.json_response(await asyncio.to_thread(callback, *args))
        except BackgroundConflictError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    def respond(callback):
        async def handler(request):
            try:
                validate_privileged_request(request)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=403)
            if not background._content_length_ok(request, MAX_REQUEST_BYTES):
                raise web.HTTPRequestEntityTooLarge(max_size=MAX_REQUEST_BYTES, actual_size=request.content_length or 0)
            try:
                if request.content_type == "multipart/form-data":
                    post = await request.post()
                    raw = str(post.get("data", "{}"))
                    field = post.get("image")
                    image = await asyncio.to_thread(field.file.read, MAX_REQUEST_BYTES + 1) if hasattr(field, "file") else None
                    if image is not None and len(image) > MAX_REQUEST_BYTES:
                        raise ValueError("Camera frame upload is too large")
                else:
                    raw, image = await request.read(), None
                if len(raw.encode("utf-8") if isinstance(raw, str) else raw) > 256 * 1024:
                    raise ValueError("CG settings are too large")
                payload = json.loads(raw or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("CG request must be an object")
            except (ValueError, TypeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
            return await perform(callback, request.match_info, payload, image)
        return handler

    @routes.get(base + "/config")
    async def config(request):
        return await perform(lambda: {**configuration(), "factory": frontend_configuration()})

    @routes.get(base + "/backgrounds")
    async def backgrounds(request):
        def inventory():
            errors, documents = [], []
            for item in list_backgrounds(errors):
                try:
                    documents.append(public_document(item))
                except (ValueError, TypeError, KeyError) as error:
                    errors.append({"name": item["name"], "error": str(error)})
            return {"backgrounds": documents, "errors": errors}
        return await perform(inventory)

    @routes.get(base + "/backgrounds/{background_id}")
    async def background_document(request):
        return await perform(lambda: public_document(load_background(request.match_info["background_id"])))

    @routes.patch(base + "/backgrounds/{background_id}")
    @respond
    def save(info, payload, image):
        return save_settings(info["background_id"], payload)

    @routes.post(base + "/backgrounds/{background_id}/jobs")
    @respond
    def job(info, payload, image):
        return start_job(info["background_id"], payload, image)

    @routes.get(base + "/backgrounds/{background_id}/jobs/{request_id}")
    async def submitted_job(request):
        return await perform(request_status, request.match_info["background_id"], request.match_info["request_id"])

    @routes.get(base + "/jobs/{job_id}")
    async def status(request):
        return await perform(job_status, request.match_info["job_id"])

    @routes.post(base + "/jobs/{job_id}/cancel")
    @respond
    def cancel(info, payload, image):
        return cancel_job(info["job_id"])

    async def serve(request, callback, *, immutable=True):
        try:
            return web.FileResponse(await asyncio.to_thread(callback), headers={"Cache-Control": "private, max-age=31536000, immutable" if immutable else "no-store"})
        except (ValueError, TypeError, FileNotFoundError) as exc:
            return web.json_response({"error": str(exc)}, status=404)

    @routes.get(base + "/backgrounds/{background_id}/images/{image_id}")
    async def source_image(request):
        info = request.match_info
        return await serve(request, lambda: image_path(info["background_id"], info["image_id"]), immutable=False)

    @routes.get(base + "/backgrounds/{background_id}/sets/{set_id}/{index}")
    async def variant_image(request):
        info = request.match_info
        return await serve(request, lambda: background_set_image_path(info["background_id"], info["set_id"], int(info["index"])), immutable=False)

    @routes.get(base + "/backgrounds/{background_id}/models/{model_id}")
    async def model(request):
        return await perform(public_model, request.match_info["background_id"], request.match_info["model_id"])

    @routes.get(base + "/backgrounds/{background_id}/models/{model_id}/objects/{object_id}/{kind}")
    async def asset(request):
        info = request.match_info
        return await serve(request, lambda: model_asset_path(info["background_id"], info["model_id"], info["object_id"], info["kind"]))

    @routes.get(base + "/backgrounds/{background_id}/cg/{cg_id}/{kind}")
    async def cg_image(request):
        info = request.match_info
        return await serve(request, lambda: cg_image_path(info["background_id"], info["cg_id"], info["kind"]))

    @routes.delete(base + "/backgrounds/{background_id}/cg/{cg_id}")
    @respond
    def delete_cg_handler(info, payload, image):
        return delete_cg(info["background_id"], info["cg_id"], payload)

    _REGISTERED = True
