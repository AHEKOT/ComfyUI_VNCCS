"""Pose sets owned by this project, independent of the Utils pose library."""

import base64
import hashlib
import io
import json
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from ..utils import (atomic_output_path, character_dir, character_storage_lock,
                     ensure_safe_name, list_characters, list_costumes,
                     list_pose_preview_files, safe_join_under)

ROOT = Path(__file__).resolve().parents[1] / "PoseSetLibrary"
MAX_BYTES = 32 * 1024 * 1024
MAX_POSES = 600


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid library identifier")
    return value


def library_path(group, name):
    return Path(safe_join_under(str(ROOT), group, identifier(name) + ".json"))


def read_json(path):
    with open(path, "rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Pose library file exceeds 32 MiB")
    return json.loads(data)


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("Pose set exceeds 32 MiB; the previous file was preserved")
    with atomic_output_path(str(path)) as temporary:
        Path(temporary).write_bytes(data)


def text(value, label, limit, required=False):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"Invalid {label}")
    return value.strip()


def thumbnail(image):
    if image.width * image.height > 64 * 1024 * 1024:
        raise ValueError("Preview image is too large")
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image.thumbnail((240, 360), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="WEBP", quality=86)
    return base64.b64encode(output.getvalue()).decode("ascii")


def preview_bytes(value):
    if not isinstance(value, str) or len(value) > 2 * 1024 * 1024:
        raise ValueError("Invalid pose preview")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return thumbnail(image)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid pose preview") from error


def validate_record(data):
    if not isinstance(data, dict):
        raise ValueError("Pose set must be an object")
    asset = data.get("asset")
    if not isinstance(asset, dict) or asset.get("type") != "pose_set":
        raise ValueError("Invalid pose set asset")
    poses = asset.get("poses")
    if not isinstance(poses, list) or not 1 <= len(poses) <= MAX_POSES or any(not isinstance(p, dict) for p in poses):
        raise ValueError("Pose set must contain 1 to 600 poses")
    json.dumps(asset, allow_nan=False)
    previews = data.get("previews")
    if not isinstance(previews, list) or len(previews) != len(poses):
        raise ValueError("Each pose needs a preview")
    return {"id": identifier(data.get("id")), "name": text(data.get("name"), "name", 160, True),
            "description": text(data.get("description", ""), "description", 4000),
            "asset": asset, "previews": [preview_bytes(p) for p in previews]}


def sprite_paths(character, costume):
    root = character_dir(character)
    costume = ensure_safe_name(costume, "costume")
    allowed = Path(safe_join_under(root, "Sprites", costume)).resolve()
    indexed = {}
    for path in list_pose_preview_files(character, costume=costume):
        file = Path(path)
        if not file.resolve().is_relative_to(allowed):
            continue
        match = re.fullmatch(r"sprite_pose_(\d+)\.(png|webp|jpe?g)", file.name, re.IGNORECASE)
        if match:
            indexed[int(match[1])] = file
    return indexed


def preview_sources():
    return [{"character": character, "costume": costume, "count": len(paths)}
            for character in list_characters() for costume in list_costumes(character)
            if (paths := sprite_paths(character, costume))]


def save_set(data):
    if not isinstance(data, dict):
        raise ValueError("Request must be an object")
    record = {**data, "id": uuid.uuid4().hex}
    selection = data.get("preview_source")
    if selection:
        if not isinstance(selection, dict):
            raise ValueError("Invalid preview character")
        character = ensure_safe_name(selection.get("character"), "character")
        costume = ensure_safe_name(selection.get("costume"), "costume")
        poses = data.get("asset", {}).get("poses", [])
        if not isinstance(poses, list) or not 1 <= len(poses) <= MAX_POSES:
            raise ValueError("Invalid pose count")
        with character_storage_lock(character_dir(character)):
            paths = sprite_paths(character, costume)
            record["previews"] = []
            for index in range(1, len(poses) + 1):
                if index not in paths:
                    raise ValueError(f"The selected character is missing sprite {index}")
                with Image.open(paths[index]) as image:
                    record["previews"].append(thumbnail(image))
    record = validate_record(record)
    with character_storage_lock(str(ROOT)):
        write_json(library_path("local", record["id"]), record)
    return record


def get_set(source, name):
    identifier(name)
    if source == "local":
        return read_json(library_path("local", name))
    repository = read_json(library_path("repositories", source))
    return next((item for item in repository["sets"] if item["id"] == name), None) or _missing()


def _missing():
    raise FileNotFoundError("Pose set not found")


def local_sets():
    directory = Path(safe_join_under(str(ROOT), "local"))
    return [read_json(library_path("local", p.stem)) for p in sorted(directory.glob("*.json"))]


def catalog():
    entries = [("local", item) for item in local_sets()]
    repositories = []
    directory = Path(safe_join_under(str(ROOT), "repositories"))
    for file in sorted(directory.glob("*.json")):
        repo = read_json(library_path("repositories", file.stem))
        repositories.append({**{key: repo[key] for key in ("id", "repo_id", "enabled", "revision")},
                             "count": len(repo["sets"])})
        if repo["enabled"]:
            entries.extend((repo["id"], item) for item in repo["sets"])
    return {"sets": [{"source": source, "id": item["id"], "name": item["name"],
                      "description": item["description"], "count": len(item["asset"]["poses"])}
                     for source, item in entries], "repositories": repositories}


def delete_set(name):
    with character_storage_lock(str(ROOT)):
        library_path("local", name).unlink()


def repository_action(data):
    action = data.get("action")
    if action in ("remove", "toggle"):
        with character_storage_lock(str(ROOT)):
            path = library_path("repositories", data.get("id"))
            if action == "remove":
                path.unlink()
            else:
                record = read_json(path)
                if not isinstance(data.get("enabled"), bool):
                    raise ValueError("Invalid enabled state")
                record["enabled"] = data["enabled"]
                write_json(path, record)
        return
    if action not in ("download", "refresh"):
        raise ValueError("Invalid repository action")
    repo_id = data.get("repo_id", "")
    if not isinstance(repo_id, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id, re.ASCII) or ".." in repo_id:
        raise ValueError("Use a public Hugging Face repository in owner/name format")
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).model_info(repo_id, files_metadata=True)
    asset = next((item for item in info.siblings if item.rfilename == "pose_sets.json"), None)
    if asset is None or not isinstance(asset.size, int) or not 0 < asset.size <= MAX_BYTES:
        raise ValueError("Repository needs pose_sets.json smaller than 32 MiB")
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha or ""):
        raise ValueError("Repository revision is unavailable")
    path = hf_hub_download(repo_id=repo_id, filename="pose_sets.json", revision=info.sha, token=False)
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("sets"), list):
        raise ValueError("Invalid pose set repository")
    records = [validate_record(item) for item in manifest["sets"]]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate set identifiers")
    key = hashlib.sha256(repo_id.encode()).hexdigest()[:32]
    with character_storage_lock(str(ROOT)):
        destination = library_path("repositories", key)
        enabled = read_json(destination)["enabled"] if destination.exists() else True
        write_json(destination, {"id": key, "repo_id": repo_id, "revision": info.sha,
                                 "enabled": enabled, "sets": records})
