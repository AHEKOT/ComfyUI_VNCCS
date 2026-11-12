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
from ..utils import get_full_path_agnostic, _safe_get_model_folder_paths, validate_privileged_request, is_sprite_archive_dir
from . import background_creator as background
from . import background_storage
from .runtime_cleanup import cleanup_runtime, inference_stage


# Saved background records and immutable CG assets.

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024
MAX_CGS = 512
MAX_MODELS = 512
CHARACTER_PROMPT_LIMIT = 8000
CHARACTER_COLORS = ("White", "Pink", "Blue", "Orange", "Green", "Purple", "Yellow", "Cyan", "Red", "Lime",
                    "Brown", "Navy", "Magenta", "Gray")
MAX_CHARACTER_REFERENCES = 16 - 2  # QI2 reserves the first two inputs for background and capture.
_ID = re.compile(r"^[a-f0-9]{32}$")
_ASSET_SUFFIXES = {"ply": ".ply", "splat": ".splat", "model": ".glb", "reference": ".png", "prepared": ".png"}



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


def _model_source(model):
    return model["source_image_id"], _variant_key(model.get("source_variant"))


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
        _id(cg.get("model_id"))
        _id(cg.get("source_image_id"))
        _source_variant(cg.get("source_variant"))
        _dimensions(cg.get("width"), cg.get("height"))
        _camera(cg.get("camera"))
        _characters(cg.get("characters", []))
        if "file" in cg or "raw_file" in cg:
            relative = _asset_file(cg, "cgs").casefold()
            if relative in asset_paths:
                raise ValueError("Duplicate CG Creator asset path")
            asset_paths.add(relative)
    return document


def _read_document(directory):
    document, payload = background._read_document(directory)
    _validate(document, directory.name)
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
        variant = _variant_source(document, image_id, source_variant)
        path = background_storage.source_path(directory, document, image_id, variant)
        if not path.is_file():
            raise FileNotFoundError("Background image file is missing")
        return path


def _asset_file(record, kind):
    if kind == "cgs":
        relative = background_storage.cg_image_file(record)
        parts = Path(relative).parts
        _name(Path(parts[1]).stem if len(parts) == 2 else parts[1])
        return relative
    relative = record.get("file")
    if (not isinstance(relative, str) or "\\" in relative or relative != Path(relative).as_posix() or len(Path(relative).parts) != 3
            or Path(relative).parts[0] not in background_storage.MODEL_BRANCHES or Path(relative).name != "scene.json"):
        raise ValueError("Invalid CG Creator saved asset path")
    _name(Path(relative).parts[1])
    return relative


def _next_asset_path(directory, branch, name, *, extension="", reserved=()):
    base = _contained(directory, branch)
    name = _name(name)
    occupied = {path.name.casefold() for path in base.iterdir()} if base.exists() else set()
    occupied.update(Path(path).name.casefold() for path in reserved)
    index = 1
    while True:
        suffix = f" ({index})" if index > 1 else ""
        candidate = _name(name[:96 - len(suffix)] + suffix)
        filename = candidate + extension
        if filename.casefold() not in occupied:
            return _contained(base, filename)
        index += 1


def _source_label(document, image_id, variant):
    if variant:
        return _name(variant.get("name") or f"Variant {variant['index']}")
    return "Original"


def _recover_assets(directory):
    return background_storage.recover_assets(directory, _atomic)


@contextmanager
def _asset_publication(directory, previous, kind, record, moves):
    pending = _contained(directory, ".cg-assets.pending")
    pending.mkdir(exist_ok=False)
    try:
        saved = json.loads(previous)
        before = background_storage.hydrate(directory, background_storage.unpack(saved), allow_unsigned=saved.get("schema_version") == 2)
        before.pop("cgs", None)
        _atomic(pending / "before.json", _json_bytes(before))
        plan = {"kind": kind, "id": record[kind[:-1] + "_id"], "record": record,
                "moves": [[old.relative_to(directory).as_posix(), new.relative_to(directory).as_posix()] for old, new in moves]}
        _atomic(pending / "plan.json", _json_bytes(plan))
    except BaseException:
        shutil.rmtree(pending)
        raise
    try:
        yield pending
    finally:
        _recover_assets(directory)


def _model_directory(directory, model_id, document):
    model = next((item for item in document.get("models", []) if item["model_id"] == model_id), None)
    if model is None:
        raise FileNotFoundError("Background model variant does not exist")
    return _contained(directory, _asset_file(model, "models")).parent if "file" in model else _contained(directory, "model", _id(model_id))


def _retired_model_moves(directory, document, models):
    moves = []
    for model in models:
        source = _model_directory(directory, model["model_id"], document)
        if not source.exists():
            continue
        scene, _ = _read_json(_contained(source, "scene.json"))
        if scene.get("model_id") != model["model_id"] or scene.get("background_id") != document["background_id"]:
            raise BackgroundStorageError("Retired model does not match its owner")
        moves.append((source, _contained(directory, ".cg-assets.pending", "retired", model["model_id"])))
    return moves


def _replace_model_records(document, model):
    retired = {item["model_id"] for item in document.get("models", []) if _model_source(item) == _model_source(model)}
    document["models"] = [item for item in document.get("models", []) if item["model_id"] not in retired] + [model]


def _copy_model(source, scene, target):
    objects = scene.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 32:
        raise ValueError("Background model must contain scene objects")
    copies, total, object_ids = [], 0, set()
    for index, item in enumerate(objects, 1):
        if not isinstance(item, dict) or _id(item.get("object_id")) in object_ids:
            raise ValueError("Background model object is invalid or duplicated")
        object_id = item["object_id"]
        object_ids.add(object_id)
        files = item.get("files")
        if not isinstance(files, dict) or not files.get(background_storage.model_geometry_key(item)):
            raise ValueError("Background model geometry is missing")
        retained = {}
        for kind in ("ply", "splat", "model"):
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
            suffix = _ASSET_SUFFIXES[kind]
            target_relative = f"{kind}{suffix}" if index == 1 else f"{kind}-{index}{suffix}"
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
    _copy_reconstruction(source, target)
    _atomic(_contained(target, "scene.json"), encoded)


def _copy_reconstruction(source, target):
    folder = _contained(source, "reconstruction")
    if not folder.is_dir():
        return
    total = 0
    for entry in sorted(folder.iterdir()):
        if not re.fullmatch(r"(?:mask-)?[0-9]{6}\.png|settings\.json", entry.name):
            continue
        path = _contained(folder, entry.name)
        total += path.stat().st_size
        if total > MAX_MODEL_BYTES:
            raise ValueError("Reconstruction references are too large")
        _copy_asset(path, _contained(target, "reconstruction", entry.name), MAX_IMAGE_BYTES)


def _save_reconstruction(directory, images, settings, focal_length, capture_holes, cameras=None):
    import numpy as np

    folder = _contained(directory, "reconstruction")
    folder.mkdir()
    for index, image in enumerate(images):
        frame = Image.fromarray((np.asarray(image.cpu()) * 255).round().clip(0, 255).astype(np.uint8))
        (folder / f"{index:06}.png").write_bytes(_image_bytes(frame))
        if capture_holes and index in capture_holes:
            mask = Image.fromarray(capture_holes[index].astype(np.uint8) * 255)
            mask.save(folder / f"mask-{index:06}.png")
    stored = {key: settings[key] for key in ("provider", "target_size", "apply_sky_mask", "resolution_mp",
                                            "precision", "max_splats", "camera_source") if key in settings}
    if cameras is not None:
        stored["cameras"] = cameras
    height, width = images.shape[1:3]
    (folder / "settings.json").write_bytes(_json_bytes({**stored, "focal_length": focal_length, "size": [width, height]}))


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
        old_models = [model for model in document.get("models", [])
                      if _model_source(model) == (image_id, _variant_key(variant))]
        if not old_models and len(document.get("models", [])) >= MAX_MODELS:
            raise ValueError("Background model limit reached")
        model_id = uuid.uuid4().hex
        branch = background_storage.model_branch(directory)
        target = (_model_directory(directory, old_models[-1]["model_id"], document) if old_models
                  else _contained(directory, branch, _source_label(document, image_id, variant)))
        retired_moves = _retired_model_moves(directory, document, old_models)
        if target.exists() and not retired_moves:
            retired, _ = _read_json(_contained(target, "scene.json"))
            if retired.get("background_id") != background_id:
                raise BackgroundStorageError("Existing model belongs to another background; files are preserved")
            if (any(model["model_id"] == retired.get("model_id") for model in document["models"])
                    and _model_source(retired) != (image_id, _variant_key(variant))):
                target = _next_asset_path(directory, branch, _source_label(document, image_id, variant))
            else:
                retired_moves.append((target, _contained(directory, ".cg-assets.pending", "retired", _id(retired["model_id"]))))
        scene.update(background_id=background_id, model_id=model_id, source_image_id=image_id,
                     background_settings=settings,
                     source_signature=background_storage.fingerprint(image_path(background_id, image_id, variant)))
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
        moves = retired_moves + [(stage, target)]
        with _asset_publication(directory, previous, "models", metadata, moves):
            _copy_model(Path(scene_directory), scene, stage)
            for old, new in moves:
                new.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old, new)
            _replace_model_records(document, metadata)
            return _commit(directory, document, _json_bytes(document) if old_models else previous)


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
    if kind not in _ASSET_SUFFIXES:
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


def _character_sprite_files(root):
    from ..utils import safe_join_under
    for name in ("Neutral", "neutral", ""):
        folder = root / name
        if not folder.is_dir():
            continue
        files = [file for file in sorted(folder.iterdir()) if file.is_file()
                 and file.suffix.lower() in {".png", ".webp", ".jpg", ".jpeg", ".bmp"}]
        if files:
            return [Path(safe_join_under(str(root), str(file.relative_to(root)))) for file in files]
    return []


def character_catalog(character=None, costume=None):
    from ..utils import (list_characters, list_costumes, character_dir, load_config,
                         character_storage_lock, ensure_safe_name, safe_join_under)
    if character is None:
        return {"characters": list_characters()}
    character = ensure_safe_name(character, "character")
    with character_storage_lock(character_dir(character)):
        config = load_config(character, strict=True)
        if config is None:
            raise FileNotFoundError("Character not found")
        info = config.get("character_info", {})
        costumes = list_costumes(character)
        costume = ensure_safe_name(costume, "costume") if costume else (costumes[0] if costumes else "")
        sprites = []
        if costume:
            root = Path(safe_join_under(character_dir(character), "Sprites", costume))
            for index, file in enumerate(_character_sprite_files(root), start=1):
                emotions = ["Neutral"]
                for folder in sorted(root.iterdir()):
                    if folder.is_dir() and folder.name.lower() != "neutral" and not is_sprite_archive_dir(folder.name):
                        candidate = Path(safe_join_under(str(root), folder.name, f"sprite_{folder.name}_{index:04d}.png"))
                        if candidate.is_file():
                            emotions.append(folder.name)
                sprites.append({"name": file.name, "emotions": emotions})
        return {"character": character, "age": info.get("age", 25), "gender": info.get("sex") or info.get("gender", ""),
                "costumes": costumes, "costume": costume, "sprites": sprites}


def character_sprite(character, costume, sprite, emotion="Neutral"):
    from ..utils import character_dir, ensure_safe_name, safe_join_under
    root = Path(safe_join_under(character_dir(ensure_safe_name(character, "character")), "Sprites",
                               ensure_safe_name(costume, "costume")))
    if not isinstance(sprite, str) or not re.fullmatch(r"[^/\\\x00-\x1f]{1,255}", sprite) or sprite in {".", ".."}:
        raise ValueError("Invalid character sprite")
    emotion = ensure_safe_name(emotion, "emotion")
    for index, file in enumerate(_character_sprite_files(root), start=1):
        if file.name != sprite:
            continue
        if emotion.lower() == "neutral":
            return file
        candidate = Path(safe_join_under(str(root), emotion, f"sprite_{emotion}_{index:04d}.png"))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Character sprite not found")


def _characters(value):
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("Invalid CG characters")
    def vector(value):
        return (isinstance(value, list) and len(value) == 3
                and all(type(n) in (int, float) and math.isfinite(n) and abs(n) <= 1e6 for n in value))
    ids = set()
    for item in value:
        if not isinstance(item, dict) or _id(item.get("id")) in ids:
            raise ValueError("Invalid or duplicate CG character")
        ids.add(item["id"])
        scale, pose = item.get("scale"), item.get("pose")
        if (not vector(item.get("position")) or type(scale) not in (int, float)
                or not .001 <= scale <= 1000 or not isinstance(pose, dict)
                or not vector(pose.get("modelRotation"))):
            raise ValueError("Invalid CG character transform")
        for key in ("bones", "bonePositions"):
            bones = pose.get(key)
            if (not isinstance(bones, dict) or len(bones) > 256
                    or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", name)
                           or not vector(rotation) for name, rotation in bones.items())):
                raise ValueError("Invalid CG character pose")
        for key, bounds in {"mesh": {"age": (1, 90), "gender": (0, 1), "weight": (0, 1), "muscle": (0, 1), "height": (0, 2)},
                            "light": {"x": (-100, 100), "y": (-100, 100), "z": (-100, 100), "intensity": (0, 5)}}.items():
            if key not in item:
                continue
            settings = item[key]
            if not isinstance(settings, dict):
                raise ValueError("Invalid CG character settings")
            for name, number in settings.items():
                if key == "light" and name == "color":
                    if not isinstance(number, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", number):
                        raise ValueError("Invalid CG character light color")
                elif name not in bounds or type(number) not in (int, float) or not math.isfinite(number) or not bounds[name][0] <= number <= bounds[name][1]:
                    raise ValueError("Invalid CG character settings")
        if "source" in item:
            source = item["source"]
            if (not isinstance(source, dict) or any(key not in {"character", "costume", "sprite", "emotion"}
                    or not isinstance(text, str) or len(text) > 255 or re.search(r"[\\/\x00-\x1f]", text)
                    or text in {".", ".."} for key, text in source.items())):
                raise ValueError("Invalid CG character source")
        if "prompt" in item:
            _text(item["prompt"], "CG character prompt", CHARACTER_PROMPT_LIMIT)
        if item.get("dummyColor") is not None and item["dummyColor"] not in CHARACTER_COLORS:
            raise ValueError("Invalid CG mannequin color")
        if ("origin" in item and not vector(item["origin"]) or "baseScale" in item and
                (type(item["baseScale"]) not in (int, float) or not .001 <= item["baseScale"] <= 1000)):
            raise ValueError("Invalid CG character placement")
    return copy.deepcopy(value)


def commit_cg(background_id, name, raw, finished, camera, model_id, expected_revision, characters=None):
    name, model_id = _name(name), _id(model_id)
    camera = _camera(camera)
    characters = _characters([] if characters is None else characters)
    finished_bytes = _image_bytes(finished)
    with background._LOCK:
        directory, document, previous = _editable(background_id, expected_revision)
        model = model_scene(background_id, model_id)
        if len(document.get("cgs", [])) >= MAX_CGS:
            raise ValueError("Background CG limit reached")
        cg_id = uuid.uuid4().hex
        reserved = [cg[key] for cg in document.get("cgs", []) for key in ("file", "raw_file") if key in cg]
        target = _next_asset_path(directory, "CG", name, extension=".png", reserved=reserved)
        metadata = {"cg_id": cg_id, "name": name, "background_id": background_id,
                    "model_id": model_id, "source_image_id": model["source_image_id"], "camera": camera,
                    "width": finished.width, "height": finished.height, "characters": characters,
                    "file": target.relative_to(directory).as_posix()}
        variant = _source_variant(model.get("source_variant"))
        if variant:
            metadata["source_variant"] = variant
        _json_bytes(metadata)
        stage = _contained(directory, ".cg-assets.pending", "data")
        with _asset_publication(directory, previous, "cgs", metadata, [(stage, target)]):
            _atomic(stage, finished_bytes)
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
        if kind == "raw" and "file" in cg and "raw_file" not in cg:
            raise FileNotFoundError("Background CG raw image is not stored")
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
CG_LORA_NAME = "VNCCS Lucid Dreams QI2"
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
    generation.update(prompt_enhance=False)
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


def resolve_cg_lora() -> str:
    from . import vnccs_control_center as center
    config = center._get_cc_config("MIUProject/VNCCS_v3.0")
    entry = center._find_entry(config.get("lora", []), CG_LORA_NAME)
    if entry is None:
        raise ValueError(f"{CG_LORA_NAME} is not configured in Control Center")
    return _asset_name("loras", center._rel_within_folder(entry["local_path"]))


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
def _report_sampling_progress(draw_id, steps, start=0.35, span=0.5):
    import comfy.utils
    original = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    owner = threading.get_ident()

    def hook(current, total, *args, **kwargs):
        if threading.get_ident() != owner:
            if callable(original):
                return original(current, total, *args, **kwargs)
            return None
        total = max(1, int(total or steps)); current = min(max(int(current), 0), total)
        _set_progress(draw_id, "sampling", start + span * current / total, f"Sampling {current}/{total}")

    _set_progress(draw_id, "sampling", start, f"Sampling 0/{steps}")
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


def _save_dataset(references, image, prompt):
    dataset = _contained(root().parent, "QI2_CG_Dataset")
    sample_id = uuid.uuid4().hex
    files = [(f"control{slot.removeprefix('image_')}", ".png", _image_bytes(control))
             for slot, _source, control in references]
    files.extend([("img", ".txt", prompt.encode("utf-8")), ("img", ".png", _image_bytes(image))])
    written = []
    try:
        for directory, extension, payload in files:
            target = _contained(dataset, directory, sample_id + extension)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                written.append(target)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
        raise


def _qi2_capture_size(width, height):
    width, height = (max(64, min(4096, 32 * round(side / 32))) for side in (width, height))
    while width * height > 16_000_000:
        if width >= height:
            width -= 32
        else:
            height -= 32
    return width, height


@logged_operation("CGCreator", "refine")
def generate_image(settings: dict[str, Any], draw_id: str, cancel_event, reference, background, character_references=(), *,
                   save_dataset=True, progress_range=(0, 1), preserve_capture_grid=False):
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
    progress_start, progress_end = progress_range
    progress_span = progress_end - progress_start

    def stage(name, fraction, message):
        _stage(draw_id, cancel_event, name, progress_start + progress_span * fraction, message)

    megapixels = math.floor(settings["resolution_mp"] * 10 + 0.5) / 10
    resolution = 32 * round(32 * math.sqrt(megapixels))
    if preserve_capture_grid:
        # Calibrated frames and the sampled latent must use the same pixel grid.
        capture_size = _qi2_capture_size(width, height)
        reference = reference.resize(capture_size, Image.Resampling.LANCZOS) if reference.size != capture_size else reference
        background = background.resize(capture_size, Image.Resampling.LANCZOS) if background.size != capture_size else background
        resolution = 0
    prompt = ANYANGLE_PROMPT
    references = [("image_1", "original background", background.convert("RGB")),
                  ("image_2", "captured camera frame", reference.convert("RGB"))]
    character_prompts = []
    for slot, (color, sprite, extra_prompt) in enumerate(character_references, 3):
        if not isinstance(sprite, Image.Image):
            raise ValueError(f"{color} dummy requires a selected character sprite")
        sprite = Image.alpha_composite(Image.new("RGBA", sprite.size, "white"), sprite.convert("RGBA")).convert("RGB")
        references.append((f"image_{slot}", "selected character sprite", sprite))
        prompt += (" " if slot == 3 else "\n") + f"{color} dummy is character from image{slot}"
        if extra_prompt:
            extra_prompt = _text(extra_prompt, "CG character prompt", CHARACTER_PROMPT_LIMIT)
            character_prompts.append(f"{color} dummy: {extra_prompt}" if len(character_references) > 1 else extra_prompt)
    if character_prompts:
        prompt += "\n" + "\n".join(character_prompts)
    with _model_session(draw_id, cancel_event), torch.inference_mode():
        generation["seed"] = _resolve_seed(generation)
        stage("loading", 0.01, "Locating CG LoRA")
        cg_lora = resolve_cg_lora()
        stage("loading", 0.03, "Loading Qwen Image 2.1")
        model, clip, vae = _load_assets(generation["diffusion_model_name"])
        stage("loading", 0.12, "Applying CG model settings")
        model, _ = apply_generation_lora(model, None, cg_lora, 1.0, clip_strength=0)
        cg_filename = cg_lora.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        for item in generation["lora_stack"]:
            filename = item["name"].replace("\\", "/").rsplit("/", 1)[-1].casefold()
            if filename in (ANYANGLE_FILENAME.casefold(), cg_filename):
                continue
            model, clip = apply_generation_lora(model, clip, item["name"], item["strength"])
            _check_cancel(cancel_event)
        if generation["turbo"]:
            model = _call_comfy_node("LoraLoaderModelOnly", model=model, lora_name=_asset_name("loras", TURBO_LORA), strength_model=1.0)[0]
        # The cache provider calls its default device selection "auto".
        cache_device = "auto" if generation["cache_device"] == "default" else generation["cache_device"]
        model = _call_comfy_node("QwenImage21Cache", model=model, device=cache_device, dtype=generation["cache_dtype"])[0]
        stage("conditioning", 0.3, "Encoding the original background and camera frame")
        for slot, source, image in references:
            log_event("reference_image", job_id=draw_id, slot=slot, source=source,
                      width=image.width, height=image.height, rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        log_event("conditioning", job_id=draw_id, prompt=prompt, negative_prompt="",
                  resolution=resolution, cache_device=cache_device, cache_dtype=generation["cache_dtype"], truncate=False)
        images = {slot: _pil_to_tensor(image) for slot, _source, image in references}
        reference_tensor = images["image_2"]
        encoded = _call_comfy_node("TextEncodeQwenImage21", clip=clip, vae=vae, prompt=prompt,
            negative_prompt="", resolution=resolution,
            images=images)
        if not isinstance(encoded, (tuple, list)) or len(encoded) < 3 or encoded[0] is None or encoded[1] is None:
            raise RuntimeError("QI2 returned invalid image conditioning; update ComfyUI")
        positive, negative = encoded[:2]
        if preserve_capture_grid:
            latent_width, latent_height = capture_size
            latent = encoded[2]
            samples = latent.get("samples") if isinstance(latent, dict) else None
            if not torch.is_tensor(samples) or samples.shape[-2:] != (latent_height // 16, latent_width // 16):
                raise RuntimeError("QI2 returned a latent that does not match the calibrated capture")
        else:
            scaled_reference = _call_comfy_node("ImageScaleToTotalPixels", image=reference_tensor,
                upscale_method="nearest-exact", megapixels=megapixels, resolution_steps=1)[0]
            latent_width, latent_height, _ = _call_comfy_node("GetImageSize", image=scaled_reference)
            latent = _call_comfy_node("EmptyLatentImage", width=latent_width, height=latent_height, batch_size=1)[0]
        log_event("latent", job_id=draw_id, width=latent_width, height=latent_height,
                  source="image_2", resolution_mp=megapixels,
                  upscale_method="lanczos" if preserve_capture_grid else "nearest-exact",
                  resolution_steps=32 if preserve_capture_grid else 1)
        stage("sampling", 0.35, "Rendering the CG frame")
        sampler_settings = {
            "seed": generation["seed"], "steps": generation["steps"], "cfg": generation["cfg"],
            "sampler_name": generation["sampler"], "scheduler": generation["scheduler"], "denoise": 1.0,
        }
        sampler_name = "SamplerCustomAdvanced" if generation["turbo"] else "KSampler"
        log_event("sampling", job_id=draw_id, sampler=sampler_name,
                  model=generation["diffusion_model_name"], settings=sampler_settings)
        with _report_sampling_progress(draw_id, generation["steps"], start=progress_start + 0.35 * progress_span,
                                       span=0.5 * progress_span):
            samples = _sample_generation_latent(model, positive, negative, latent, sampler_settings,
                                                turbo=generation["turbo"])
        stage("decoding", 0.9, "Decoding the CG image")
        image = _tensor_to_pil(_call_comfy_node("VAEDecode", vae=vae, samples=samples)[0])
        if preserve_capture_grid and image.size != capture_size:
            raise RuntimeError(f"QI2 returned a {image.width}x{image.height} frame for the calibrated "
                               f"{latent_width}x{latent_height} capture")
        _check_cancel(cancel_event)
        if save_dataset:
            _save_dataset(references, image, prompt)
        stage("complete" if save_dataset else "refining", 1,
              "CG image ready" if save_dataset else "Refined camera reference ready")
        return image, {"settings": settings, "prompt": prompt, "width": image.width, "height": image.height,
                       "seed": generation["seed"], "model": "qi2",
                       "cg_lora": {"name": cg_lora, "strength": 1.0}}

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
    if "characters" in state:
        scenes = state["characters"]
        if not isinstance(scenes, dict):
            raise ValueError("Invalid CG character scenes")
        for model_id, characters in scenes.items():
            _id(model_id)
            _characters(characters)
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
    DESCRIPTION = "Select a saved background, build a TripoSplat, Pixal3D, WorldMirror or MoGe model and compose CG camera views. Workflow execution returns the selected finished CG."

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
_JOBS_LOCK = background_storage.JOBS_LOCK
background_storage.register_jobs("cg", lambda: _JOBS)


def public_document(document):
    value = copy.deepcopy(document)
    base = f"{API_BASE}/backgrounds/{value['background_id']}"
    for key in ("images", "models", "cgs", "generated_sets"):
        value.setdefault(key, [])
    value["cgs"] = [cg for cg in value["cgs"] if not cg.get("archived")]
    value.pop("model_settings", None)
    value.pop("cg_settings", None)
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


def _public_scene(scene, url):
    for item in scene.get("objects", []):
        kind = "model" if background_storage.model_geometry_key(item) == "model" else "splat"
        if kind == "splat" and scene.get("background_settings", {}).get("provider") == "worldmirror":
            # Factory rotates every SPLAT as TripoSplat. Cancel that child rotation in the scene transform.
            item["transform"] = {**item.get("transform", {}), "rotation": [180, 90, 0]}
        item["urls"] = {kind: f"{url}/objects/{item['object_id']}/{kind}"}
        item["locked"] = True
        item.pop("files", None)
    return scene


def public_model(background_id, model_id):
    with background._LOCK:
        scene = model_scene(background_id, model_id)
        directory, document, _ = _locate(background_id)
        model = _model_directory(directory, model_id, document)
        manifest = _contained(model, "reconstruction", "settings.json")
        if manifest.is_file():
            settings, _ = _read_json(manifest)
            scene["can_expand"] = True
            scene["reconstruction_size"] = settings["size"]
        pending = _contained(model, ".expansion", "scene.json")
        if pending.is_file():
            preview, _ = _read_json(pending)
            scene["expansion_id"] = _id(preview["expansion_id"])
        else:
            frame = _contained(model, ".expansion-frame", "preview.json")
            if frame.is_file():
                preview, _ = _read_json(frame)
                scene["expansion_frame_id"] = _id(preview["frame_id"])
    return _public_scene(scene, f"{API_BASE}/backgrounds/{background_id}/models/{model_id}")


def _expansion_scene(background_id, model_id, expansion_id):
    base = model_scene(background_id, model_id)
    directory, document, _ = _locate(background_id)
    folder = _contained(_model_directory(directory, model_id, document), ".expansion")
    scene, _ = _read_json(_contained(folder, "scene.json"))
    if (scene.get("expansion_id") != _id(expansion_id) or scene.get("model_id") != model_id
            or scene.get("background_id") != background_id):
        raise FileNotFoundError("Scene expansion is no longer available")
    return base, folder, scene


def public_expansion(background_id, model_id, expansion_id):
    with background._LOCK:
        _, _, scene = _expansion_scene(background_id, model_id, expansion_id)
        url = f"{API_BASE}/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}"
        return _public_scene(scene, url)


def _expansion_frame(model, frame_id):
    folder = _contained(model, ".expansion-frame")
    frame, _ = _read_json(_contained(folder, "preview.json"))
    if frame["frame_id"] != _id(frame_id):
        raise FileNotFoundError("Expansion frame is no longer available")
    return folder, frame


def _expansion_diagnostics(frame_id):
    return _contained(root().parent, "CG_Expansion_Diagnostics", _id(frame_id))


def expansion_frame_image(background_id, model_id, expansion_id, kind):
    with background._LOCK:
        directory, document, _ = _locate(background_id)
        folder, _ = _expansion_frame(_model_directory(directory, model_id, document), expansion_id)
        if kind == "before":
            return _contained(_expansion_diagnostics(expansion_id), "capture.png")
        if kind == "after":
            return _contained(folder, "frame.png")
        raise FileNotFoundError("Expansion frame image is unavailable")


def discard_expansion_frame(background_id, model_id, expansion_id):
    with _JOBS_LOCK, background._LOCK:
        ensure_idle(background_id)
        directory, document, _ = _editable(background_id, None)
        folder, _ = _expansion_frame(_model_directory(directory, model_id, document), expansion_id)
        shutil.rmtree(folder)
    return {}


def expansion_asset(background_id, model_id, expansion_id, object_id, kind):
    with background._LOCK:
        _, folder, scene = _expansion_scene(background_id, model_id, expansion_id)
        item = next((item for item in scene["objects"] if item["object_id"] == _id(object_id)), None)
        key = "ply" if kind == "splat" else "model" if kind == "model" else None
        relative = item and key and item["files"].get(key)
        if not relative:
            raise FileNotFoundError("Expansion geometry is unavailable")
        path = _contained(folder, relative)
        return provider()["storage"]._ensure_cached_splat(path, ply_sha256="") if kind == "splat" else path


def finish_expansion(background_id, model_id, expansion_id, save):
    with _JOBS_LOCK, background._LOCK:
        ensure_idle(background_id)
        _editable(background_id, None)
        base, folder, scene = _expansion_scene(background_id, model_id, expansion_id)
        if save:
            if background_storage.fingerprint(image_path(background_id, base["source_image_id"], base.get("source_variant"))) != scene["source_signature"]:
                raise BackgroundConflictError("Background source changed while expanding its scene")
            scene.pop("expansion_id")
            document = commit_model(background_id, base["source_image_id"], folder, scene, base["background_settings"],
                                    None, source_variant=base.get("source_variant"))
        else:
            shutil.rmtree(folder)
            document = load_background(background_id)
    return public_document(document)


def model_settings(value, saved=None):
    if not isinstance(value, dict):
        raise ValueError("Background model settings must be an object")
    saved = saved or {}
    provider_name = "moge" if (saved.get("provider") or value.get("provider")) == "moge" else "worldmirror"
    resolution = _number(value.get(f"{provider_name}_resolution_mp", saved.get("resolution_mp", 1)),
                         "Reconstruction resolution", 0.05, 16)
    camera_source = value.get(f"{provider_name}_camera_source", saved.get("camera_source", "camera_inputs"))
    if camera_source not in ("camera_inputs", "colmap" if provider_name == "moge" else "predicted"):
        raise ValueError("Invalid reconstruction camera source")
    if provider_name == "moge":
        return {"provider": "moge", "resolution_mp": resolution, "camera_source": camera_source}
    size = value.get("worldmirror_target_size", saved.get("target_size", 2044))
    if type(size) not in (int, float) or not -math.inf < size < math.inf:
        raise ValueError("WorldMirror resolution must be a finite number")
    size = 14 * math.floor(max(252, min(4088, size)) / 14 + 0.5)
    sky_mask = value.get("worldmirror_apply_sky_mask", saved.get("apply_sky_mask", True))
    if type(sky_mask) is not bool:
        raise ValueError("WorldMirror sky mask must be a boolean")
    precision = value.get("worldmirror_precision", saved.get("precision", "convrot4"))
    if precision not in ("convrot4", "convrot8", "bf16"):
        raise ValueError("Invalid WorldMirror model precision")
    max_splats = _number(value.get("worldmirror_max_splats", saved.get("max_splats", 2_000_000)),
                         "Maximum Gaussians", 1, 2**53 - 1, integer=True)
    return {"provider": "worldmirror", "resolution_mp": resolution, "target_size": size,
            "apply_sky_mask": sky_mask, "precision": precision, "max_splats": max_splats,
            "camera_source": camera_source}


def _reconstruction_generation(settings, model_settings):
    return {**settings, "resolution_mp": model_settings.get("resolution_mp", 1),
            "generation": {**settings["generation"], "cache_device": "off"}}


def _worldmirror_camera(poses, intrinsics, height):
    pose, intrinsic = poses[0, 0], intrinsics[0, 0]
    focal_y = float(intrinsic[1, 1])
    if not math.isfinite(focal_y) or focal_y <= 0:
        raise ValueError("WorldMirror returned an invalid camera focal length")
    position = pose[:3, 3]
    # OpenCV cameras look along +Z with +Y down.
    return _camera({"position": position.tolist(), "target": (position + pose[:3, 2]).tolist(),
                    "up": (-pose[:3, 1]).tolist(), "fov": math.degrees(2 * math.atan(height / (2 * focal_y)))})


def _write_worldmirror_ply(path, splats, cancel_event):
    import numpy as np

    if not splats:
        raise ValueError("WorldMirror returned no Gaussian splats")
    means, sh, opacity, scales, quats = (
        splats[key][0].detach().float().cpu().numpy()
        for key in ("means", "sh", "opacities", "scales", "quats")
    )
    colors = sh[:, 0, :]
    valid = (np.isfinite(means).all(axis=1) & np.isfinite(colors).all(axis=1)
             & np.isfinite(opacity) & np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1)
             & np.isfinite(quats).all(axis=1) & (np.linalg.norm(quats, axis=1) > 0))
    count = int(valid.sum())
    if not count:
        raise ValueError("WorldMirror returned no usable Gaussian splats")
    properties = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(header.encode("ascii"))
        for start in range(0, len(means), 65536):
            _check_cancel(cancel_event)
            section = slice(start, start + 65536)
            keep = valid[section]
            size = int(keep.sum())
            if not size:
                continue
            rows = np.zeros((size, len(properties)), dtype="<f4")
            rows[:, :3] = means[section][keep]
            rows[:, 6:9] = colors[section][keep]
            alpha = opacity[section][keep].clip(1e-6, 1 - 1e-6)
            rows[:, 9] = np.log(alpha / (1 - alpha))
            rows[:, 10:13] = np.log(scales[section][keep])
            rows[:, 13:] = quats[section][keep]
            stream.write(memoryview(rows).cast("B"))


def _generate_worldmirror_scene(job, images, scene, scene_directory, settings, cameras):
    import torch
    from .world_mirror_v2 import VNCCS_WorldMirrorV2Loader, VNCCS_WorldMirrorV2Reconstruction

    job_id, cancel = job["job_id"], job["cancel_event"]
    camera_options = {"splat_camera_source": settings["camera_source"]}
    if cameras is not None:
        world_to_camera = torch.tensor(cameras["world_to_camera"], dtype=torch.float32)
        camera_to_world = torch.linalg.inv(world_to_camera)
        camera_options.update(camera_poses=camera_to_world,
                              camera_intrinsics=torch.tensor(cameras["intrinsics"], dtype=torch.float32))
    with _model_session(job_id, cancel), torch.inference_mode():
        precision = settings.get("precision", "convrot4")
        _stage(job_id, cancel, "loading", 0.60, f"Loading WorldMirror V2 ({precision})")
        model, = VNCCS_WorldMirrorV2Loader().load_model(precision=precision)
        _stage(job_id, cancel, "reconstructing", 0.70, f"Reconstructing {len(images)} views with WorldMirror V2")
        ply, depth, _, poses, intrinsics, *_ = VNCCS_WorldMirrorV2Reconstruction().run_inference(
            model, images, target_size=settings["target_size"],
            offload_scheme="model_cpu_offload", low_vram_mode=True, debug_log=False,
            apply_sky_mask=settings["apply_sky_mask"], enable_splat_upsample=False,
            splat_max_points=settings.get("max_splats", 2_000_000), **camera_options,
        )
        if cameras is None:
            # Store intrinsics in reconstruction image pixels, before WorldMirror's internal resize.
            height, width = images.shape[1:3]
            resized_height, resized_width = depth.shape[1:3]
            calibration = intrinsics[0] * intrinsics.new_tensor([width / resized_width, height / resized_height, 1])[:, None]
            cameras = {"intrinsics": calibration.tolist(), "world_to_camera": torch.linalg.inv(poses[0]).tolist()}
        elif not torch.allclose(poses[0], camera_to_world):
            raise ValueError("WorldMirror could not align its depth estimates with the supplied cameras")
        _stage(job_id, cancel, "saving", 0.90, "Saving WorldMirror splats and camera")
        camera = _worldmirror_camera(poses, intrinsics, depth.shape[1])
        object_id = uuid.uuid4().hex
        relative = f"objects/{object_id}/model.ply"
        _write_worldmirror_ply(_contained(scene_directory, relative), ply.get("splats"), cancel)
    _set_reconstruction_object(scene, object_id, camera, "ply", relative)
    return cameras


def _worldmirror_expansion_cameras(saved, camera, image_shape):
    import numpy as np

    cameras = copy.deepcopy(saved)
    _, height, width = image_shape[:3]
    position = np.asarray(camera["position"], dtype=np.float64)
    forward = np.asarray(camera["target"]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera["up"])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("The WorldMirror capture camera has an invalid up direction")
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, -rotation @ position
    focal = height / (2 * math.tan(math.radians(camera["fov"]) / 2))
    cameras["world_to_camera"].append(pose.tolist())
    cameras["intrinsics"].append([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return cameras


def _infer_moge(images, fov_x=0.0):
    from .preview_runtime import _run_preview_job

    model_name = _asset_name("geometry_estimation", "moge_2_vitl_normal_fp16.safetensors")
    model, = _call_comfy_node("LoadMoGeModel", model_name=model_name)
    geometry, = _run_preview_job(
        _call_comfy_node, "MoGeInference", moge_model=model, image=images, resolution_level=9, fov_x_degrees=fov_x,
        batch_size=min(4, len(images)), force_projection=True, apply_mask=True, refine_steps=3,
    )
    return geometry


def _generate_refined_moge_views(job, source, settings, diagnostics, *, preserve_capture=False):
    import torch
    from .cg_moge_multiview import REFINED_MOGE_VIEWS, composite_capture, render_shifted_views

    job_id, cancel = job["job_id"], job["cancel_event"]
    diagnostics.mkdir(parents=True)
    (diagnostics / "image_1.png").write_bytes(_image_bytes(source))
    log_event("moge_diagnostics", component="CGCreator", job_id=job_id, directory=str(diagnostics))
    megapixels = math.floor(settings["resolution_mp"] * 10 + 0.5) / 10
    scale = min(math.sqrt(megapixels * 1e6 / (source.width * source.height)), 4000 / max(source.size))
    size = _qi2_capture_size(source.width * scale, source.height * scale)
    original = source if source.size == size else source.resize(size, Image.Resampling.LANCZOS)
    images = _pil_to_tensor(original)
    with _model_session(job_id, cancel), torch.inference_mode():
        _stage(job_id, cancel, "reconstructing", 0.03, "Building the initial MoGe scene")
        geometry = _infer_moge(images)
        _stage(job_id, cancel, "rendering", 0.08, "Capturing shifted MoGe views")
        mesh, = _call_comfy_node("MoGePointMapToMesh", moge_geometry=geometry, batch_index=0,
                                decimation=max(1, math.ceil(max(size) / 1536)), discontinuity_threshold=0.04, texture=True)
        intrinsics = geometry["intrinsics"][0]
        frames, calibration, poses = render_shifted_views(mesh, intrinsics, lambda: _check_cancel(cancel))
        source_depth = geometry["depth"][0].detach().cpu().float()
        focal_length = float(intrinsics[1, 1]) * size[1]
        del geometry, mesh

    (diagnostics / "cameras.json").write_bytes(_json_bytes({
        "intrinsics": calibration.tolist(), "world_to_camera": poses.tolist(), "size": list(size),
    }))
    generation = settings["generation"]
    settings = {**settings, "generation": {**generation, "seed": _resolve_seed(generation), "seed_mode": "fixed"}}
    raw_views = []
    capture_holes = {}
    for index, ((direction, _), (frame, holes, observed)) in enumerate(zip(REFINED_MOGE_VIEWS, frames, strict=True), 1):
        view_dir = diagnostics / f"{index:02}-{direction}"
        view_dir.mkdir()
        raw = _tensor_to_pil(frame)
        if raw.size != size or holes.shape != size[::-1] or observed.shape != size[::-1]:
            raise RuntimeError("Camera capture and its depth and mask must use the calibrated frame dimensions")
        (view_dir / "image_2.png").write_bytes(_image_bytes(raw))
        raw_views.append((view_dir, raw, observed))
        capture_holes[index] = holes
    references = [images]
    view_progress = 0.46 / len(raw_views)
    for index, (view_dir, raw, observed) in enumerate(raw_views):
        start = 0.12 + view_progress * index
        log_event("moge_refine_view", component="CGCreator", job_id=job_id, view=view_dir.name)
        refined, _ = generate_image(settings, job_id, cancel, raw, original, save_dataset=False,
                                    progress_range=(start, start + view_progress), preserve_capture_grid=True)
        (view_dir / "qi2_result.png").write_bytes(_image_bytes(refined))
        if preserve_capture:
            refined = composite_capture(raw, refined, observed)
            (view_dir / "reconstruction_frame.png").write_bytes(_image_bytes(refined))
        references.append(_pil_to_tensor(refined))
    return torch.cat(references), focal_length, capture_holes, (calibration, poses, source_depth, frames)


def _generate_moge_scene(job, images, scene, scene_directory, *, capture_holes, known_views=None, colmap_views=None):
    import numpy as np
    import torch
    from .cg_moge_multiview import REFINED_MOGE_VIEWS, capture_depth_tracks, depth_interior, depth_transform, fuse_moge_views

    job_id, cancel = job["job_id"], job["cancel_event"]
    def check_cancel():
        _check_cancel(cancel)

    if colmap_views is None:
        calibration, poses, source_depth, captures = known_views
    else:
        calibration, poses, tracks = colmap_views
    height, width = images.shape[1:3]
    fov_x = math.degrees(2 * math.atan(width / (2 * calibration[0, 0])))
    fov_y = math.degrees(2 * math.atan(height / (2 * calibration[1, 1])))
    camera = _camera({"position": [0, 0, 0], "target": [0, 0, -1], "up": [0, 1, 0],
                      "fov": max(5.0, min(120.0, fov_y))})
    with _model_session(job_id, cancel), torch.inference_mode():
        _stage(job_id, cancel, "reconstructing", 0.70, f"Estimating geometry for {len(images)} views with MoGe")
        if colmap_views is None:
            depths = _infer_moge(images[1:], fov_x)["depth"].detach().cpu().float()
            transforms = [(1., 0.)]
            retained = [0]
            retained_depths = [source_depth]
            for index, ((name, _), (raw, holes, observed), depth) in enumerate(
                    zip(REFINED_MOGE_VIEWS, captures, depths, strict=True), 1):
                check_cancel()
                matches = capture_depth_tracks((raw[0].numpy() * 255).round().astype(np.uint8),
                    (images[index].numpy() * 255).round().astype(np.uint8), observed, ~holes & depth_interior(observed))
                try:
                    scale, offset = depth_transform(depth.numpy(), matches)
                except ValueError:
                    log_event("moge_view_skipped", component="CGCreator", job_id=job_id, view=name,
                              matches=len(matches), reason="depth_could_not_be_aligned_to_capture")
                    continue
                log_event("moge_view_aligned", component="CGCreator", job_id=job_id, view=name, matches=len(matches),
                          depth_scale=scale, depth_offset=offset)
                retained.append(index)
                retained_depths.append(depth)
                transforms.append((scale, offset))
            images, poses = images[retained], poses[retained]
            capture_holes = {index: capture_holes[old] for index, old in enumerate(retained) if old}
            depths = torch.stack(retained_depths)
        else:
            depths = _infer_moge(images, fov_x)["depth"].detach().cpu().float()
            transforms = [depth_transform(view.numpy(), match) for view, match in zip(depths, tracks, strict=True)]
        _stage(job_id, cancel, "meshing", 0.86, "Fusing all calibrated MoGe views")
        scales, offsets = np.asarray(transforms).T
        normalized = poses.copy()
        normalized[:, :3, 3] /= scales[0]
        cameras = {"intrinsics": [calibration.tolist()] * len(images),
                   "world_to_camera": (normalized @ np.linalg.inv(normalized[0])).tolist(),
                   "depth_scales": (scales / scales[0]).tolist(), "depth_offsets": (offsets / scales[0]).tolist()}
        # Only initial captures have masks for every supplementary view.
        mesh = fuse_moge_views(depths, images, cameras, _call_comfy_node, check_cancel,
                               capture_holes=capture_holes if known_views is not None else None)
        _save_moge_mesh(job, scene, scene_directory, mesh, camera)
    return images, capture_holes, cameras


def _save_moge_mesh(job, scene, scene_directory, mesh, camera):
    cancel = job["cancel_event"]
    _stage(job["job_id"], cancel, "saving", 0.94, "Saving the MoGe mesh and camera")
    file, = _call_comfy_node("MeshToFile3D", mesh=mesh)
    payload = file.get_bytes()
    if not payload.startswith(b"glTF") or len(payload) > MAX_MODEL_BYTES:
        raise ValueError("MoGe returned an invalid or oversized GLB model")
    _check_cancel(cancel)
    object_id = uuid.uuid4().hex
    relative = f"objects/{object_id}/model.glb"
    path = _contained(scene_directory, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    _set_reconstruction_object(scene, object_id, camera, "model", relative)


def _recover_initial_moge_cameras(images, depths, geometry, check_cancel):
    import numpy as np
    from .cg_moge_multiview import capture_depth_tracks, depth_transform, render_shifted_views

    size = images.shape[1:3]
    mesh, = _call_comfy_node("MoGePointMapToMesh", moge_geometry=geometry, batch_index=0,
                            decimation=max(1, math.ceil(max(size) / 1536)), discontinuity_threshold=0.04, texture=True)
    # Reconstructions saved before the camera turns used parallel shifted views.
    captures, calibration, poses = render_shifted_views(mesh, geometry["intrinsics"][0], check_cancel, turn_camera=False)
    scales = [1.]
    offsets = [0.]
    for index, (raw, holes, observed) in enumerate(captures, 1):
        check_cancel()
        tracks = capture_depth_tracks((raw[0].numpy() * 255).round().astype(np.uint8),
            (images[index].numpy() * 255).round().astype(np.uint8), observed, ~holes)
        try:
            scale, offset = depth_transform(depths[index].numpy(), tracks)
            scales.append(scale)
            offsets.append(offset)
        except ValueError as error:
            raise ValueError("An older reconstruction frame cannot be aligned. Recreate the base model with Refine views") from error
    return {"intrinsics": [calibration.tolist()] * len(poses), "world_to_camera": poses.tolist(),
            "depth_scales": scales, "depth_offsets": offsets}


def _expand_moge_scene(job, values, images, scene, scene_directory):
    import numpy as np
    import torch
    from .cg_moge_multiview import align_expansion_depth, fuse_moge_views

    job_id, cancel = job["job_id"], job["cancel_event"]
    base = model_scene(job["background_id"], values["model_id"])
    item = base["objects"][0]
    path = _contained(values["model_directory"], item["files"]["model"])
    camera = values["model_camera"]
    height, width = images.shape[1:3]
    focal = height / (2 * math.tan(math.radians(camera["fov"]) / 2))
    calibration = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    cameras = copy.deepcopy(values["model_settings"].get("cameras"))
    check_cancel = lambda: _check_cancel(cancel)
    with _model_session(job_id, cancel), torch.inference_mode():
        _stage(job_id, cancel, "reconstructing", 0.70, f"Re-estimating MoGe depth for all {len(images)} views")
        if cameras is None:
            if len(images) != 6 or values["model_settings"]["focal_length"] is None:
                raise ValueError("This older model has no saved reconstruction cameras. Recreate its base model once")
            seed = _infer_moge(images[:1])
            original_fx = float(seed["intrinsics"][0, 0, 0]) * width
            intrinsics = [[[original_fx, 0, width / 2], [0, original_fx, height / 2], [0, 0, 1]]] * 5
        else:
            intrinsics = cameras["intrinsics"]
            if len(intrinsics) != len(images) - 1:
                raise ValueError("Saved camera calibration does not match the reconstruction frames")
        by_focal = {}
        for index, matrix in enumerate([*intrinsics, calibration]):
            by_focal.setdefault(float(matrix[0][0]), []).append(index)
        depths = torch.empty(images.shape[:3], dtype=torch.float32)
        for fx, indices in by_focal.items():
            check_cancel()
            fov_x = math.degrees(2 * math.atan(width / (2 * fx)))
            depths[indices] = _infer_moge(images[indices], fov_x)["depth"].detach().cpu().float()
        if cameras is None:
            cameras = _recover_initial_moge_cameras(images, depths, seed, check_cancel)
        _stage(job_id, cancel, "meshing", 0.86, "Fusing all calibrated MoGe views")
        try:
            pose, scale, offset = align_expansion_depth(path, camera, values["raw"], images[-1], depths[-1], focal, check_cancel)
            cameras["intrinsics"].append(calibration)
            cameras["world_to_camera"].append(pose.tolist())
            cameras["depth_scales"].append(scale)
            cameras.setdefault("depth_offsets", [0.] * (len(images) - 1)).append(offset)
            mesh = fuse_moge_views(depths, images, cameras, _call_comfy_node, check_cancel)
        except ValueError as error:
            raise ValueError(f"{error}. See {values['expansion_diagnostics']}.") from error
        _save_moge_mesh(job, scene, scene_directory, mesh, base["camera"])
    return cameras


def _set_reconstruction_object(scene, object_id, camera, kind, relative):
    scene["camera"] = camera
    item = {"object_id": object_id, "name": "Background", "source_camera": camera, "visible": True,
            "locked": True, "files": {kind: relative},
            "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": 1}}
    if kind == "model":
        item.update(asset_kind="mesh", source={"format": "glb"})
    scene["objects"] = [item]


def ensure_idle(background_id):
    background.ensure_idle(background_id)


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
    if action not in {"model", "cg", "expand"}:
        raise ValueError("Unsupported CG Creator action")
    request_id = _request_id(payload["request_id"]) if "request_id" in payload else ""
    image_id = payload.get("image_id", "")
    source_variant = copy.deepcopy(_source_variant(payload.get("source_variant")))
    image_path(background_id, image_id, source_variant)
    values = {"image_id": image_id, "source_variant": source_variant,
              "source_signature": background_storage.fingerprint(image_path(background_id, image_id, source_variant)),
              "model_settings": model_settings(payload.get("model_settings", {})) if action != "expand" else {},
              "settings": normalize_settings(payload.get("settings", {}))}
    if action in {"cg", "expand"}:
        values["model_id"] = _id(payload.get("model_id"))
        if action == "cg":
            values.update(characters=_characters(payload.get("characters", [])), name=_name(payload.get("name")),
                          camera=_camera(payload.get("camera")))
        scene = model_scene(background_id, values["model_id"])
        if (scene["source_image_id"] != image_id
                or _variant_key(scene.get("source_variant")) != _variant_key(source_variant)):
            raise ValueError("The model belongs to another background variant")
        if action == "expand" and "expansion_frame_id" in payload:
            values["expansion_frame_id"] = _id(payload["expansion_frame_id"])
        else:
            raw = factory._decode_image(raw_image or b"").convert("RGB")
            if min(raw.size) < 64 or max(raw.size) > 4096 or raw.width % 16 or raw.height % 16 or raw.width * raw.height > 16_000_000:
                raise ValueError("CG frame must be 64–4096 pixels, aligned to 16 pixels, at most 16 MP")
            values["raw"] = raw
        if action == "cg":
            characters = values["characters"]
            if len(characters) > MAX_CHARACTER_REFERENCES:
                raise ValueError(f"QI2 supports up to {MAX_CHARACTER_REFERENCES} character sprites per CG")
            assigned_colors = [item["dummyColor"] for item in characters if item.get("dummyColor") is not None]
            if len(assigned_colors) != len(set(assigned_colors)):
                raise ValueError("Each CG character needs a distinct mannequin color")
            free_colors = iter(color for color in CHARACTER_COLORS if color not in assigned_colors)
            references = values["character_references"] = []
            for item in characters:
                color = item.get("dummyColor") or next(free_colors)
                source = item.get("source", {})
                if not all(source.get(key) for key in ("character", "costume", "sprite")):
                    raise ValueError(f"Select a character, costume and sprite for the {color.lower()} dummy before generating CG")
                path = character_sprite(source["character"], source["costume"], source["sprite"], source.get("emotion") or "Neutral")
                with Image.open(path) as sprite:
                    _dimensions(*sprite.size)
                    references.append((color, sprite.copy(), item.get("prompt", "")))
    with _JOBS_LOCK:
        ensure_idle(background_id)
        document = load_background(background_id)
        if document.get("_recovery") or document.get("archived"):
            raise ValueError("This saved background is read-only")
        if action == "expand":
            directory, _, _ = _locate(background_id)
            model = _model_directory(directory, values["model_id"], document)
            if _contained(model, ".expansion").exists():
                raise BackgroundConflictError("Save or discard the current scene expansion first")
            manifest = _contained(model, "reconstruction", "settings.json")
            if not manifest.is_file():
                raise ValueError("Create this model again to save its reconstruction frames")
            settings, _ = _read_json(manifest)
            provider_name = settings.get("provider")
            if provider_name not in {"moge", "worldmirror"}:
                raise ValueError("Scene expansion requires a MoGe or WorldMirror model")
            if scene.get("source_signature") != values["source_signature"]:
                raise BackgroundConflictError("The background source changed. Recreate its model before expanding")
            settings.update(model_settings(payload.get("model_settings", {}), settings))
            if provider_name == "worldmirror" and settings["camera_source"] == "camera_inputs" and not settings.get("cameras"):
                raise ValueError("This WorldMirror model has no saved reconstruction cameras. Recreate its base model once")
            if "expansion_frame_id" in values:
                folder, frame = _expansion_frame(model, values["expansion_frame_id"])
                if frame["source_signature"] != values["source_signature"]:
                    raise BackgroundConflictError("Background source changed. Discard the expansion frame and try again")
                values.update(expansion_frame=folder, model_camera=_camera(frame.get("camera")),
                              expansion_diagnostics=_expansion_diagnostics(frame["frame_id"]))
            elif _contained(model, ".expansion-frame").exists():
                raise BackgroundConflictError("Confirm or discard the current expansion frame first")
            else:
                values["model_camera"] = _camera(payload.get("model_camera"))
            values.update(model_settings=settings, model_directory=model)
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
        if action in {"model", "expand"}:
            job["model_provider"] = values["model_settings"]["provider"]
        _JOBS[job["job_id"]] = job
        _WORKER.submit(runtime._run_job, job, lambda current: _execute(current, values))
    return job_status(job["job_id"])


def _expand_references(job, values):
    import numpy as np
    import torch

    folder = _contained(values["model_directory"], "reconstruction")
    paths = sorted(entry.name for entry in folder.iterdir() if re.fullmatch(r"[0-9]{6}\.png", entry.name))
    if not paths:
        raise ValueError("The scene has no reconstruction frames")
    images, holes = [], {}
    for index, name in enumerate(paths):
        _check_cancel(job["cancel_event"])
        with Image.open(_contained(folder, name)) as stored:
            frame = stored.convert("RGB")
        if index == 0:
            original = frame
        if frame.size != original.size:
            raise ValueError("Reconstruction frames have inconsistent dimensions")
        images.append(_pil_to_tensor(frame))
        mask = _contained(folder, f"mask-{name}")
        if mask.is_file():
            with Image.open(mask) as stored:
                holes[index] = np.asarray(stored.convert("L")) != 0
    with Image.open(_contained(values["expansion_frame"], "frame.png")) as stored:
        refined = stored.convert("RGB")
    with Image.open(_contained(values["expansion_diagnostics"], "capture.png")) as stored:
        values["raw"] = stored.convert("RGB")
    if refined.size != original.size:
        raise ValueError("The approved expansion frame does not match the reconstruction dimensions")
    return torch.cat((*images, _pil_to_tensor(refined))), holes


def _preview_expansion_frame(job, values):
    diagnostics = _expansion_diagnostics(job["job_id"])
    diagnostics.mkdir(parents=True)
    (diagnostics / "capture.png").write_bytes(_image_bytes(values["raw"]))
    if "model_camera" in values:
        (diagnostics / "camera.json").write_bytes(_json_bytes(values["model_camera"]))
    log_event("expansion_diagnostics", component="CGCreator", job_id=job["job_id"], directory=str(diagnostics))
    model = values["model_directory"]
    with Image.open(_contained(model, "reconstruction", "000000.png")) as stored:
        original = stored.convert("RGB")
    if values["raw"].size != original.size:
        raise ValueError("Expansion capture must match the saved reconstruction dimensions")
    settings = _reconstruction_generation(values["settings"], values["model_settings"])
    refined, _ = generate_image(settings, job["job_id"], job["cancel_event"], values["raw"], original,
                                save_dataset=False, preserve_capture_grid=True)
    (diagnostics / "qi2_result.png").write_bytes(_image_bytes(refined))
    if refined.size != original.size:
        # Older reconstructions store their cameras in a 16-pixel grid.
        refined = refined.resize(original.size, Image.Resampling.LANCZOS)
    if values["model_settings"]["provider"] == "moge":
        from .cg_moge_multiview import composite_capture, rasterize_expansion_depth

        base = model_scene(job["background_id"], values["model_id"])
        path = _contained(model, base["objects"][0]["files"]["model"])
        camera = values["model_camera"]
        focal = refined.height / (2 * math.tan(math.radians(camera["fov"]) / 2))
        depth, _ = rasterize_expansion_depth(path, camera, (refined.height, refined.width), focal,
                                            lambda: _check_cancel(job["cancel_event"]))
        refined = composite_capture(values["raw"], refined, depth)
    with _JOBS_LOCK, background._LOCK:
        provider()["runtime"]._check_cancel(job)
        directory, document, _ = _editable(job["background_id"], None)
        if (_model_directory(directory, values["model_id"], document) != model
                or background_storage.fingerprint(image_path(job["background_id"], values["image_id"], values["source_variant"]))
                != values["source_signature"]):
            raise BackgroundConflictError("Background source changed while preparing its expansion frame")
        target = _contained(model, ".expansion-frame")
        with tempfile.TemporaryDirectory(prefix=".expansion-frame-", dir=model) as temporary:
            stage = Path(temporary) / "preview"
            stage.mkdir()
            (stage / "frame.png").write_bytes(_image_bytes(refined))
            (stage / "preview.json").write_bytes(_json_bytes({
                "frame_id": job["job_id"], "source_signature": values["source_signature"],
                "camera": values.get("model_camera"),
            }))
            os.replace(stage, target)
        job["published"] = True
    return {"expansion_frame_id": job["job_id"]}


def _execute(job, values):
    backend = provider()
    runtime, factory = backend["runtime"], backend["storage"]
    temporary_scene_id = None
    try:
        with _inference_slot(job):
            if job["action"] == "expand" and "expansion_frame" not in values:
                return _preview_expansion_frame(job, values)
            if job["action"] == "cg":
                with Image.open(image_path(job["background_id"], values["image_id"], values["source_variant"])) as source:
                    background_image = source.convert("RGB")
                characters = values.get("characters", [])
                image, _ = generate_image(values["settings"], job["job_id"], job["cancel_event"], values["raw"], background_image,
                    character_references=values["character_references"])
                runtime._check_cancel(job)
                with _JOBS_LOCK:
                    runtime._check_cancel(job)
                    with background._LOCK:
                        if background_storage.fingerprint(image_path(job["background_id"], values["image_id"], values["source_variant"])) != values["source_signature"]:
                            raise BackgroundConflictError("Background source changed while generating CG; the result was not published")
                        document = commit_cg(job["background_id"], values["name"], values["raw"], image, values["camera"],
                            values["model_id"], None, characters=characters)
                    job["published"] = True
            else:
                settings = values["model_settings"]
                use_colmap = settings["provider"] == "moge" and settings["camera_source"] == "colmap"
                if use_colmap:
                    from .cg_moge_multiview import register_views, require_colmap

                    require_colmap()
                scene = factory.create_scene("CG Creator")
                temporary_scene_id = job["scene_id"] = scene["scene_id"]
                scene_directory = factory.resolve_scene_dir(scene["scene_id"])
                expanding = job["action"] == "expand"
                diagnostics = _contained(root().parent, "CG_MoGe_Diagnostics", _id(job["job_id"]))
                cameras = None
                if expanding:
                    images, capture_holes = _expand_references(job, values)
                    focal_length = settings["focal_length"]
                else:
                    with Image.open(image_path(job["background_id"], values["image_id"], values["source_variant"])) as source:
                        source_image = source.convert("RGB")
                    images, focal_length, capture_holes, known_views = _generate_refined_moge_views(
                        job, source_image, _reconstruction_generation(values["settings"], settings), diagnostics,
                        preserve_capture=settings["provider"] == "moge")
                if settings["provider"] == "worldmirror":
                    if settings["camera_source"] == "camera_inputs":
                        if expanding:
                            cameras = _worldmirror_expansion_cameras(settings["cameras"], values["model_camera"], images.shape)
                        else:
                            calibration, poses, *_ = known_views
                            cameras = {"intrinsics": [calibration.tolist()] * len(images), "world_to_camera": poses.tolist()}
                    cameras = _generate_worldmirror_scene(job, images, scene, scene_directory, settings, cameras)
                elif use_colmap:
                    _stage(job["job_id"], job["cancel_event"], "calibrating", 0.60,
                           f"Estimating {len(images)} camera poses with COLMAP")
                    diagnostics.mkdir(parents=True, exist_ok=True)
                    colmap_views = register_views(images, lambda: runtime._check_cancel(job), diagnostics=diagnostics,
                                                  focal_length=None if expanding else focal_length, capture_holes=capture_holes)
                    images, capture_holes, cameras = _generate_moge_scene(job, images, scene, scene_directory,
                        capture_holes=capture_holes, known_views=None if expanding else known_views, colmap_views=colmap_views)
                elif expanding:
                    cameras = _expand_moge_scene(job, values, images, scene, scene_directory)
                else:
                    images, capture_holes, cameras = _generate_moge_scene(job, images, scene, scene_directory,
                        capture_holes=capture_holes, known_views=known_views)
                _save_reconstruction(scene_directory, images, settings, focal_length, capture_holes, cameras)
                scene["lighting"] = {"preset": "off", "background": "#000000"}
                with _JOBS_LOCK:
                    runtime._check_cancel(job)
                    with background._LOCK:
                        if background_storage.fingerprint(image_path(job["background_id"], values["image_id"], values["source_variant"])) != values["source_signature"]:
                            raise BackgroundConflictError("Background source changed while building its model; the result was not published")
                        if job["action"] == "expand":
                            directory, document, _ = _editable(job["background_id"], None)
                            model = _model_directory(directory, values["model_id"], document)
                            base = model_scene(job["background_id"], values["model_id"])
                            scene.update({key: base[key] for key in ("background_id", "model_id", "source_image_id",
                                                                     "background_settings", "source_signature")})
                            if values["source_variant"]:
                                scene["source_variant"] = values["source_variant"]
                            scene["expansion_id"] = job["job_id"]
                            with tempfile.TemporaryDirectory(prefix=".expansion-", dir=model) as temporary:
                                stage = Path(temporary) / "preview"
                                _copy_model(scene_directory, scene, stage)
                                target = _contained(model, ".expansion")
                                if target.exists():
                                    raise BackgroundConflictError("Save or discard the current scene expansion first")
                                os.replace(stage, target)
                            job["published"] = True
                            shutil.rmtree(values["expansion_frame"], ignore_errors=True)
                            return {"expansion_id": job["job_id"]}
                        document = commit_model(job["background_id"], values["image_id"], scene_directory,
                            scene, {"provider": settings["provider"]}, None, source_variant=values["source_variant"])
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
    if job["status"] == "running" and (job["action"] == "cg" or job.get("model_provider") in {"worldmirror", "moge"}):
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
            if job["action"] == "cg" or job.get("model_provider") in {"worldmirror", "moge"}:
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

    @routes.get(base + "/characters")
    async def characters(request):
        return await perform(lambda: character_catalog())

    @routes.get(base + "/characters/{character}")
    async def character_options(request):
        return await perform(lambda: character_catalog(request.match_info["character"], request.query.get("costume")))

    @routes.get(base + "/characters/{character}/sprite")
    async def sprite(request):
        return await serve(request, lambda: character_sprite(request.match_info["character"],
            request.query.get("costume", ""), request.query.get("sprite", ""), request.query.get("emotion", "Neutral")), immutable=False)

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

    expansion = base + "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}"

    @routes.get(expansion)
    async def expansion_preview(request):
        info = request.match_info
        return await perform(public_expansion, info["background_id"], info["model_id"], info["expansion_id"])

    @routes.get(expansion + "/objects/{object_id}/{kind}")
    async def expansion_geometry(request):
        return await serve(request, lambda: expansion_asset(**dict(request.match_info)))

    @routes.get(expansion + "/frame/{kind}")
    async def expansion_frame(request):
        return await serve(request, lambda: expansion_frame_image(**dict(request.match_info)), immutable=False)

    @routes.delete(expansion + "/frame")
    @respond
    def reject_expansion_frame(info, _payload, _image):
        return discard_expansion_frame(info["background_id"], info["model_id"], info["expansion_id"])

    @routes.post(expansion)
    @respond
    def resolve_expansion(info, payload, _image):
        if type(payload.get("save")) is not bool:
            raise ValueError("Choose whether to save the scene expansion")
        return finish_expansion(info["background_id"], info["model_id"], info["expansion_id"], payload["save"])

    @routes.get(base + "/backgrounds/{background_id}/cg/{cg_id}/{kind}")
    async def cg_image(request):
        info = request.match_info
        return await serve(request, lambda: cg_image_path(info["background_id"], info["cg_id"], info["kind"]))

    @routes.delete(base + "/backgrounds/{background_id}/cg/{cg_id}")
    @respond
    def delete_cg_handler(info, payload, image):
        return delete_cg(info["background_id"], info["cg_id"], payload)

    _REGISTERED = True
