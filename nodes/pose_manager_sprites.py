"""Manager-owned cache for the public sprite pose repository."""

import hashlib
import io
import json
import re
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from PIL import Image

from .pose_sets import MAX_BYTES, MAX_POSES, read_json, write_json
from ..utils import character_storage_lock, safe_join_under

ROOT = Path(__file__).resolve().parents[1] / "PoseSetLibrary" / "sprites"
REPOSITORY = "MIUProject/VNCCS_PoseLibrary_Sprites"


def path(name):
    return Path(safe_join_under(str(ROOT), name))


def repository_id(value):
    if not isinstance(value, str):
        raise ValueError("Invalid pose repository")
    value = value.strip()
    if value.startswith("https://"):
        url = urlparse(value)
        if url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("Use a public Hugging Face repository")
        value = url.path.strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value, re.ASCII) or ".." in value:
        raise ValueError("Use a repository in owner/name format")
    return value


def download(filename, revision, repo_id=REPOSITORY):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache = path("hub")
    options = dict(repo_id=repo_id, filename=filename, revision=revision, cache_dir=str(cache))
    try:
        file = Path(hf_hub_download(**options, token=False, local_files_only=True))
    except LocalEntryNotFoundError:
        file = Path(hf_hub_download(**options, token=False))
    if not file.resolve().is_relative_to(cache.resolve()):
        raise ValueError("Sprite cache path escapes the library")
    return file


def validate_manifest(manifest, files):
    entries = manifest.get("poses") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_POSES:
        raise ValueError("Invalid sprite pose manifest")
    records, ids = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("asset_type", "pose") != "pose":
            raise ValueError("Sprite library requires single poses")
        for key in ("name", "category"):
            if not isinstance(entry.get(key), str) or not 0 < len(entry[key]) <= 160:
                raise ValueError(f"Invalid sprite {key}")
        for key, extensions in (("json", {".json"}), ("preview", {".webp", ".png", ".jpg", ".jpeg"})):
            name = entry.get(f"{key}_path")
            if (not isinstance(name, str) or "\\" in name or ":" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or PurePosixPath(name).suffix.lower() not in extensions):
                raise ValueError("Unsafe sprite asset path")
            size = files.get(name)
            if not isinstance(size, int) or not 0 < size <= 8 * 1024 * 1024:
                raise ValueError("Sprite asset is missing or exceeds 8 MiB")
            if not re.fullmatch(r"[a-fA-F0-9]{64}", entry.get(f"{key}_sha256", "")):
                raise ValueError("Invalid sprite asset checksum")
        tags = entry.get("tags", [])
        if not isinstance(tags, list) or len(tags) > 100 or any(not isinstance(tag, str) or len(tag) > 160 for tag in tags):
            raise ValueError("Invalid sprite tags")
        key = hashlib.sha256(entry["json_path"].encode()).hexdigest()[:32]
        if key in ids:
            raise ValueError("Duplicate sprite pose")
        ids.add(key)
        records.append({field: entry[field] for field in (
            "name", "category", "json_path", "preview_path", "json_sha256", "preview_sha256")}
            | {"id": key, "tags": tags})
    return records


def repositories():
    if path("repositories.json").exists():
        return read_json(path("repositories.json"))["repositories"]
    # Preserve the previous single-repository cache and its enabled setting.
    data = read_json(path("catalog.json")) if path("catalog.json").exists() else {"poses": []}
    settings = read_json(path("settings.json")) if path("settings.json").exists() else {"enabled": True}
    return [{**data, "repo_id": REPOSITORY, "enabled": settings["enabled"]}]


def refreshed(repo):
    from huggingface_hub import HfApi

    repo_id = repository_id(repo["repo_id"])
    info = HfApi(token=False).model_info(repo_id, files_metadata=True)
    files = {item.rfilename: item.size for item in info.siblings}
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha or ""):
        raise ValueError("Pose repository revision is unavailable")
    if not 0 < (files.get("pose_library.json") or 0) <= MAX_BYTES:
        raise ValueError("Pose manifest is missing or exceeds 32 MiB")
    records = validate_manifest(read_json(download("pose_library.json", info.sha, repo_id)), files)
    return {**repo, "revision": info.sha, "poses": records, "last_checked": time.time()}


def ensure(refresh=False):
    with character_storage_lock(str(ROOT)):
        repos = repositories()
        repos = [refreshed(repo) if refresh or (repo["enabled"] and not repo.get("revision")) else repo for repo in repos]
        write_json(path("repositories.json"), {"repositories": repos})
        return catalog()


def catalog():
    repos = repositories()
    summaries = [{"repo_id": repo["repo_id"], "title": repo["repo_id"], "builtin": False,
                  "enabled": repo["enabled"], "pose_count": len(repo["poses"]), "animation_count": 0,
                  "status": "ok" if repo.get("revision") else "not downloaded", "last_checked": repo.get("last_checked", 0)}
                 for repo in repos]
    return {"repositories": summaries, "repository": next((repo for repo in summaries if repo["repo_id"] == REPOSITORY), None),
            "poses": [{"id": entry["id"], "manager_sprite_id": entry["id"],
             "name": entry["name"], "category": entry["category"], "tags": entry["tags"],
             "repository": repo["repo_id"], "asset_type": "pose", "has_preview": True,
             "preview_type": "image/webp", "preview_mtime": repo["revision"]}
             for repo in repos if repo["enabled"] for entry in repo["poses"]]}


def repository_action(action, repo_id, enabled=None):
    repo_id = repository_id(repo_id)
    with character_storage_lock(str(ROOT)):
        repos = repositories()
        index = next((i for i, repo in enumerate(repos) if repo["repo_id"] == repo_id), None)
        if action == "add":
            if index is not None:
                raise ValueError("Repository already exists")
            repos.append(refreshed({"repo_id": repo_id, "enabled": True, "poses": []}))
        elif index is None:
            raise FileNotFoundError("Pose repository not found")
        elif action == "toggle":
            if not isinstance(enabled, bool):
                raise ValueError("Invalid enabled state")
            repos[index]["enabled"] = enabled
        elif action == "refresh":
            repos[index] = refreshed(repos[index])
        elif action == "remove":
            repos.pop(index)
        else:
            raise ValueError("Invalid repository action")
        write_json(path("repositories.json"), {"repositories": repos})
        return catalog()


def toggle(enabled):
    return repository_action("toggle", REPOSITORY, enabled)


def asset(identifier, preview=False, repo_id=REPOSITORY):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid sprite identifier")
    data = next((repo for repo in repositories() if repo["repo_id"] == repository_id(repo_id)), None)
    if data is None:
        raise FileNotFoundError("Pose repository not found")
    entry = next((item for item in data["poses"] if item["id"] == identifier), None)
    if entry is None:
        raise FileNotFoundError("Sprite pose not found")
    key = "preview" if preview else "json"
    file = download(entry[f"{key}_path"], data["revision"], data["repo_id"])
    with file.open("rb") as handle:
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != entry[f"{key}_sha256"].lower():
        raise ValueError("Sprite asset checksum mismatch")
    if preview:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 64 * 1024 * 1024:
                raise ValueError("Sprite preview is too large")
            content_type = Image.MIME.get(image.format)
            if content_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Unsupported sprite preview")
            image.verify()
        return raw, content_type
    pose = json.loads(raw)
    if not isinstance(pose, dict) or pose.get("type") in {"pose_set", "pose_animation"}:
        raise ValueError("Sprite library requires a single pose")
    return pose
