"""Current background state, independently owned CGs, and recoverable file writes."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from functools import lru_cache
from pathlib import Path


FORMAT_VERSION = 3
MAX_JSON_BYTES = 4 * 1024 * 1024
_ID = re.compile(r"[a-f0-9]{32}")
_PENDING = ".background-write.pending"
JOBS_LOCK = threading.RLock()
_JOB_OWNERS = {}


def register_jobs(owner, get_jobs):
    with JOBS_LOCK:
        _JOB_OWNERS[owner] = get_jobs


def ensure_idle(background_id):
    with JOBS_LOCK:
        if any(job.get("background_id") == background_id and job.get("status") in {"queued", "running"}
               for get_jobs in _JOB_OWNERS.values() for job in get_jobs().values()):
            raise RuntimeError("This background is busy. Wait for the job or stop it first.")


def contained(directory, relative):
    directory = Path(directory)
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Invalid background storage path")
    target = directory / relative
    if any(path.is_symlink() for path in (directory, *target.parents, target)):
        raise ValueError("Background storage paths cannot contain symbolic links")
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Background storage path escaped its owner")
    return target


def json_bytes(value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Background metadata is too large")
    return payload


def read_json(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Background metadata is too large")
    value = json.loads(payload)
    json_bytes(value)
    return value, payload


def atomic(path, payload):
    path = Path(path)
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


def _discard_journal(directory, name, *, completed=False):
    pending = contained(directory, name)
    finished = contained(directory, name + ".completed")
    try:
        if completed and pending.exists():
            os.replace(pending, finished)
        if finished.exists():
            shutil.rmtree(finished)
    except OSError:
        if not completed:
            raise
        # A committed callback must not trigger its caller's asset rollback.
        logging.getLogger("VNCCS").warning("Background publication completed; journal cleanup will retry: %s", pending)


@lru_cache(maxsize=512)
def _fingerprint(path, size, modified, changed):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return _fingerprint(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def unpack(value):
    """Keep the existing node/API representation separate from the disk format."""
    if not isinstance(value, dict):
        raise ValueError("Background configuration must be an object")
    document = copy.deepcopy(value)
    if document.get("schema_version") == FORMAT_VERSION:
        source = document.pop("source")
        outputs = document.pop("variant_outputs")
        document.update(schema_version=2, images=[source] if source else [],
                        selected_image_id=source["image_id"] if source else "", generated_sets=[outputs] if outputs else [])
    return document


def pack(document):
    value = copy.deepcopy(document)
    sources = value.pop("images", [])
    selected = value.pop("selected_image_id", "")
    source = next((item for item in sources if item["image_id"] == selected), None)
    if selected and source is None:
        raise ValueError("Current background source does not exist")
    groups = value.pop("generated_sets", [])
    outputs = next((item for item in reversed(groups) if item["version"] == "" and item["source_image_id"] == selected), None)
    for key in ("models", "model", "cgs", "_recovery", "asset_errors"):
        value.pop(key, None)
    if source:
        source.pop("url", None)
    if outputs:
        for image in outputs["images"]:
            image.pop("url", None)
    value.update(schema_version=FORMAT_VERSION, source=source, variant_outputs=outputs)
    return value


def current(document):
    value = unpack(pack(document))
    value["models"] = copy.deepcopy(document.get("models", []))
    value["cgs"] = copy.deepcopy(document.get("cgs", []))
    return value


def source_path(directory, document, image_id, variant=None):
    if image_id != document.get("selected_image_id"):
        raise FileNotFoundError("The background source has been replaced")
    if variant:
        group = next((group for group in document.get("generated_sets", [])
                      if group["version"] == "" and group["source_image_id"] == image_id and group["set_id"] == variant["set_id"]), None)
        index = variant.get("index")
        if group is None or type(index) is not int or not 1 <= index < len(group["images"]):
            raise FileNotFoundError("The background variant has been replaced")
        return contained(directory, group["images"][index]["file"])
    source = next((image for image in document.get("images", []) if image["image_id"] == image_id), None)
    if source is None:
        raise FileNotFoundError("The background source does not exist")
    return contained(directory, source["file"])


def cg_config_path(directory, record):
    relative = record.get("file")
    if not isinstance(relative, str):
        relative = f"CG/{record['name']}/{record['cg_id']}/image.png"
    image = contained(directory, relative)
    if image.suffix != ".png" or image.relative_to(directory).parts[0] != "CG":
        raise ValueError("CG image escaped its owner")
    return image.parent / f"{image.parent.name}_config.json"


def model_geometry_key(item):
    return "model" if item.get("asset_kind") == "mesh" else "ply"


def hydrate(directory, document, *, allow_unsigned=False):
    """Discover actual assets; loading never migrates or renames their files."""
    value = copy.deepcopy(document)
    models, cgs, errors, latest = {}, {}, [], {}
    paths = set()
    for kind, branch, filename in (("models", "model", "scene.json"), ("cgs", "CG", "image.png")):
        records = value.get(kind, [])
        if not isinstance(records, list) or len(records) > (128 if kind == "models" else 512):
            raise ValueError("Invalid background asset list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Invalid background asset record")
            relative = record.get("file")
            if relative is None:
                continue
            path = contained(directory, relative)
            parts = path.relative_to(directory).parts
            if (len(parts) not in ({3} if kind == "models" else {3, 4}) or parts[0] != branch or parts[-1] != filename
                    or len(parts) == 4 and parts[2] != record.get("cg_id")):
                raise ValueError("Invalid CG Creator saved asset path")
            if relative.casefold() in paths:
                raise ValueError("Duplicate CG Creator asset path")
            paths.add(relative.casefold())
    candidates = [(record, contained(directory, record.get("file", f"model/{record['model_id']}/scene.json")))
                  for record in value.get("models", [])]
    model_root = contained(directory, "model")
    for path in sorted(model_root.glob("*/scene.json")) if model_root.exists() else []:
        contained(directory, path.relative_to(directory).as_posix())
        if all(path != known for _record, known in candidates):
            candidates.append((None, path))
    for record, path in candidates:
        if not path.is_file():
            continue
        try:
            scene, _ = read_json(path)
            identity = scene.get("model_id")
            if not isinstance(identity, str) or not _ID.fullmatch(identity) or scene.get("background_id") != value["background_id"]:
                raise ValueError("Model scene does not match its owner")
            if record and record.get("model_id") != identity:
                raise ValueError("Model record does not match its scene")
            source = source_path(directory, value, scene["source_image_id"], scene.get("source_variant"))
            signature = fingerprint(source)
            if scene.get("source_signature") != signature and not (allow_unsigned and "source_signature" not in scene):
                continue
            objects = scene.get("objects")
            if not isinstance(objects, list) or not 1 <= len(objects) <= 32:
                raise ValueError("Background model must contain scene objects")
            for item in objects:
                geometry = item.get("files", {}).get(model_geometry_key(item))
                if not geometry or not contained(path.parent, geometry).is_file():
                    raise FileNotFoundError("Model geometry is missing")
            model = copy.deepcopy(record) if record else {
                "model_id": identity, "source_image_id": scene["source_image_id"], "scene_id": scene.get("scene_id", ""),
                "settings": scene.get("background_settings", {}), "file": path.relative_to(directory).as_posix(),
            }
            if scene.get("source_variant"):
                model["source_variant"] = copy.deepcopy(scene["source_variant"])
            if identity in models and models[identity].get("file") != model.get("file"):
                raise ValueError("Duplicate background model identity")
            models[identity] = model
            variant = model.get("source_variant")
            key = (model["source_image_id"], variant["set_id"], variant["index"]) if variant else (model["source_image_id"],)
            if key not in latest or path.stat().st_mtime_ns >= latest[key][0]:
                latest[key] = (path.stat().st_mtime_ns, identity)
        except FileNotFoundError:
            continue
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"{path.relative_to(directory)}: {error}")
    for record in value.get("cgs", []):
        image = contained(directory, record.get("file", f"CG/{record['name']}/{record['cg_id']}/image.png"))
        if image.is_file():
            cgs[record["cg_id"]] = copy.deepcopy(record)
    cg_root = contained(directory, "CG")
    cg_paths = sorted([*cg_root.glob("*/*_config.json"), *cg_root.glob("*/*/*_config.json")]) if cg_root.exists() else []
    for path in cg_paths:
        contained(directory, path.relative_to(directory).as_posix())
        record, _ = read_json(path)
        if (not isinstance(record, dict) or record.get("background_id") != value["background_id"]
                or not isinstance(record.get("cg_id"), str) or not _ID.fullmatch(record["cg_id"])
                or cg_config_path(directory, record) != path):
            raise ValueError("CG configuration does not match its owner")
        image = contained(directory, record["file"])
        if image.is_file():
            if record["cg_id"] in cgs and cgs[record["cg_id"]] != record:
                raise ValueError("Duplicate CG identity")
            cgs[record["cg_id"]] = record
    active_ids = {identity for _modified, identity in latest.values()}
    value["models"] = [model for identity, model in models.items() if identity in active_ids]
    value["cgs"] = list(cgs.values())
    if errors:
        value["asset_errors"] = errors
    return value


def recover(directory, writer=atomic):
    _discard_journal(directory, _PENDING)
    pending = contained(directory, _PENDING)
    if not pending.exists():
        return
    marker = pending / "transaction.json"
    if not marker.exists():
        shutil.rmtree(pending)
        return
    plan, _ = read_json(marker)
    if not isinstance(plan, list) or len(plan) > 2048:
        raise ValueError("Invalid background write journal")
    records = []
    for index, item in enumerate(plan):
        target = contained(directory, item["file"])
        relative = target.relative_to(directory)
        allowed = relative.as_posix() in {f"{directory.name}_config.json", f"{directory.name}_config.json.backup", "source.png"}
        allowed |= len(relative.parts) in {3, 4} and relative.parts[0] == "CG" and relative.name == f"{relative.parts[-2]}_config.json" and (len(relative.parts) == 3 or bool(_ID.fullmatch(relative.parts[2])))
        allowed |= len(relative.parts) == 3 and relative.parts[0] == "model" and relative.name == "scene.json"
        if (not allowed or not isinstance(item.get("after"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["after"])
                or item.get("before") is not None and not re.fullmatch(r"[a-f0-9]{64}", item["before"])):
            raise ValueError("Background write journal escaped its owner")
        records.append((target, item, pending / "before" / str(index)))
    complete = all(target.is_file() and fingerprint(target) == item["after"] for target, item, _backup in records)
    if not complete:
        for target, item, backup in reversed(records):
            if item["before"] is None:
                target.unlink(missing_ok=True)
            elif not target.is_file() or fingerprint(target) != item["before"]:
                if not backup.is_file() or fingerprint(backup) != item["before"]:
                    raise ValueError("Background rollback copy is missing or damaged")
                writer(target, backup.read_bytes())
    _discard_journal(directory, _PENDING, completed=True)


def _writes(directory, document, files=None):
    directory = Path(directory)
    writes = dict(files or {})
    for model in document.get("models", []):
        relative = model.get("file", f"model/{model['model_id']}/scene.json")
        if relative in writes:
            continue
        scene, _ = read_json(contained(directory, relative))
        if "source_signature" not in scene:
            scene["source_signature"] = fingerprint(source_path(directory, document, scene["source_image_id"], scene.get("source_variant")))
            writes.setdefault(relative, json_bytes(scene))
    for record in document.get("cgs", []):
        metadata = copy.deepcopy(record)
        metadata.pop("url", None)
        metadata["background_id"] = document["background_id"]
        metadata.setdefault("file", f"CG/{record['name']}/{record['cg_id']}/image.png")
        path = cg_config_path(directory, metadata)
        writes[path.relative_to(directory).as_posix()] = json_bytes(metadata)
    encoded = json_bytes(pack(document))
    writes[f"{directory.name}_config.json"] = encoded
    writes[f"{directory.name}_config.json.backup"] = encoded
    return {relative: payload for relative, payload in writes.items()
            if not contained(directory, relative).is_file() or contained(directory, relative).read_bytes() != payload}


def publish(directory, document, writer=atomic, files=None):
    """Commit current metadata and owned files together, retaining rollback on failure."""
    directory = Path(directory)
    writes = _writes(directory, document, files)
    if not writes:
        return
    pending = contained(directory, _PENDING)
    pending.mkdir(exist_ok=False)
    plan = []
    try:
        for index, (relative, payload) in enumerate(writes.items()):
            target = contained(directory, relative)
            before = None
            if target.exists():
                if not target.is_file():
                    raise ValueError("Background write target is not a file")
                before = fingerprint(target)
                backup = pending / "before" / str(index)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            plan.append({"file": relative, "before": before, "after": hashlib.sha256(payload).hexdigest()})
        atomic(pending / "transaction.json", json_bytes(plan))
    except BaseException:
        shutil.rmtree(pending)
        raise
    try:
        for relative, payload in writes.items():
            writer(contained(directory, relative), payload)
    except BaseException:
        recover(directory, writer)
        raise
    _discard_journal(directory, _PENDING, completed=True)


def migrate_directory(directory, backup_directory=None, *, apply=False):
    """Inspect by default; explicit application retains one exact, reversible backup."""
    directory = Path(directory)
    for reserved in (_PENDING, ".cg-assets.pending", ".vnccs-background-publication"):
        if contained(directory, reserved).exists():
            raise ValueError("Finish background publication recovery before migrating")
    raw, _ = read_json(contained(directory, f"{directory.name}_config.json"))
    if (not isinstance(raw, dict) or raw.get("schema_version") not in {2, FORMAT_VERSION}
            or raw.get("name") != directory.name or not isinstance(raw.get("background_id"), str)
            or not _ID.fullmatch(raw["background_id"]) or type(raw.get("revision")) is not int or raw["revision"] < 1
            or not isinstance(raw.get("editor"), dict) or type(raw.get("archived")) is not bool):
        raise ValueError("Background configuration cannot be migrated safely")
    legacy = unpack(raw)
    document = hydrate(directory, current(legacy), allow_unsigned=raw["schema_version"] == 2)
    if document.get("asset_errors"):
        raise ValueError("Repair damaged model metadata before migrating: " + "; ".join(document["asset_errors"]))
    writes = {}
    for model in document["models"]:
        relative = model.get("file", f"model/{model['model_id']}/scene.json")
        scene, _ = read_json(contained(directory, relative))
        source = source_path(directory, document, scene["source_image_id"], scene.get("source_variant"))
        signature = fingerprint(source)
        if scene.get("source_signature") != signature:
            scene["source_signature"] = signature
            writes[relative] = json_bytes(scene)
    if document["images"]:
        image = document["images"][0]
        if not _ID.fullmatch(image["image_id"]):
            raise ValueError("Invalid background source identity")
        source = source_path(directory, document, image["image_id"])
        if not source.is_file():
            raise FileNotFoundError("Current background source is missing; migration stopped")
        if source != contained(directory, "source.png"):
            writes["source.png"] = source.read_bytes()
            image["file"] = "source.png"
        image.setdefault("preview_revision", document["revision"])
    writes = _writes(directory, document, writes)
    report = {"name": directory.name, "background_id": raw["background_id"], "applied": False,
              "sources_before": len(legacy.get("images", [])), "sources_after": len(document["images"]),
              "models_before": len(legacy.get("models", [])), "models_after": len(document["models"]),
              "cgs_after": len(document["cgs"]), "changes": list(writes)}
    if not apply or not writes:
        return report
    if backup_directory is None:
        raise ValueError("An explicit migration backup directory is required")
    backup_directory = Path(backup_directory)
    if backup_directory.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Migration backup must be outside the background directory")
    backup_directory.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, payload in writes.items():
        target = contained(directory, relative)
        before = fingerprint(target) if target.is_file() else None
        if before:
            backup = contained(backup_directory, relative)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        records.append({"file": relative, "before": before, "after": hashlib.sha256(payload).hexdigest()})
    atomic(backup_directory / "migration.json", json_bytes({"background_id": raw["background_id"], "files": records}))
    publish(directory, document, files=writes)
    report["applied"] = True
    report["backup"] = str(backup_directory)
    return report


def restore_migration(directory, backup_directory):
    directory, backup_directory = Path(directory), Path(backup_directory)
    manifest, _ = read_json(contained(backup_directory, "migration.json"))
    current, _ = read_json(contained(directory, f"{directory.name}_config.json"))
    if current.get("background_id") != manifest.get("background_id"):
        raise ValueError("Migration backup belongs to another background")
    for item in manifest["files"]:
        target = contained(directory, item["file"])
        present = fingerprint(target) if target.is_file() else None
        if present not in {item["before"], item["after"]}:
            raise ValueError("Background changed after migration; rollback stopped")
        if item["before"]:
            backup = contained(backup_directory, item["file"])
            if not backup.is_file() or fingerprint(backup) != item["before"]:
                raise ValueError("Migration backup is missing or damaged")
    for item in reversed(manifest["files"]):
        target = contained(directory, item["file"])
        if (fingerprint(target) if target.is_file() else None) == item["before"]:
            continue
        if item["before"]:
            atomic(target, contained(backup_directory, item["file"]).read_bytes())
        else:
            target.unlink(missing_ok=True)


def recover_assets(directory, writer=atomic):
    _discard_journal(directory, ".cg-assets.pending")
    pending = contained(directory, ".cg-assets.pending")
    if not pending.exists():
        return
    marker = pending / "plan.json"
    if not marker.exists():
        allowed = {"before.json", "before.backup", "scene.before.json"}
        if any(path.is_symlink() or not path.is_file() or path.name not in allowed and not path.name.endswith(".tmp") for path in pending.iterdir()):
            raise ValueError("Incomplete CG asset publication journal")
        shutil.rmtree(pending)
        return
    plan, _ = read_json(marker)
    before, _ = read_json(contained(pending, "before.json"))
    kind, identity, target_record = plan["kind"], plan["id"], plan["record"]
    if kind not in {"models", "cgs"} or not isinstance(identity, str) or not _ID.fullmatch(identity):
        raise ValueError("Invalid CG asset publication journal")
    key = kind[:-1] + "_id"
    if target_record.get(key) != identity:
        raise ValueError("Invalid CG asset publication identity")
    target = contained(directory, target_record["file"]).parent
    branch = "model" if kind == "models" else "CG"
    expected = "scene.json" if kind == "models" else "image.png"
    parts = Path(target_record["file"]).parts
    if len(parts) != 3 or parts[0] != branch or parts[-1] != expected:
        raise ValueError("CG asset publication escaped its owner")
    record = next((item for item in before.get(kind, []) if item[key] == identity), None)
    if record:
        relative = record.get("file", f"model/{identity}/scene.json" if kind == "models" else f"CG/{record['name']}/{identity}/image.png")
        source = contained(directory, relative).parent
    else:
        source = pending / "data"
    def source_key(item):
        variant = item.get("source_variant")
        return item["source_image_id"], (variant["set_id"], variant["index"]) if variant else None
    retired = {}
    if kind == "models":
        for item in before.get("models", []):
            if item["model_id"] != identity and source_key(item) == source_key(target_record):
                path = contained(directory, item.get("file", f"model/{item['model_id']}/scene.json")).parent
                retired[path] = pending / "retired" / item["model_id"]
        for old, new in plan["moves"]:
            old, new = contained(directory, old), contained(directory, new)
            if old == target and new.parent == pending / "retired" and _ID.fullmatch(new.name):
                manifest = (new if new.exists() else old) / "scene.json"
                scene, _ = read_json(manifest)
                if scene.get("model_id") != new.name or scene.get("background_id") != before.get("background_id"):
                    raise ValueError("Retired model does not match its owner")
                retired[old] = new
    moves = []
    for old, new in plan["moves"]:
        old, new = contained(directory, old), contained(directory, new)
        valid = old == source and new == target or old in retired and new == retired[old]
        if record and kind == "cgs":
            valid |= old.parent == source and new == target / old.name
        if record and kind == "models":
            valid |= old.parent == target / "objects" and new.parent == old.parent and bool(_ID.fullmatch(old.name)) and bool(re.fullmatch(r"Object [1-9][0-9]*", new.name))
        if not valid:
            raise ValueError("CG asset publication escaped its owner")
        moves.append((old, new))
    current, _ = read_json(contained(directory, f"{directory.name}_config.json"))
    if current.get("background_id") != before.get("background_id"):
        raise ValueError("CG asset publication belongs to another background")
    committed = current.get("revision") == before.get("revision", 0) + 1
    if committed:
        manifest = target / "scene.json" if kind == "models" else cg_config_path(directory, target_record)
        published, _ = read_json(manifest)
        if published.get(key) != identity or published.get("background_id") != current["background_id"]:
            raise ValueError("CG asset publication does not match its owner")
    else:
        if current.get("revision") != before.get("revision"):
            raise ValueError("Background changed during asset recovery; files are preserved")
        for old, new in reversed(moves):
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new, old)
        if plan.get("manifest"):
            writer(contained(source, "scene.json"), contained(pending, "scene.before.json").read_bytes())
        backup = contained(directory, f"{directory.name}_config.json.backup")
        if plan["had_backup"]:
            writer(backup, contained(pending, "before.backup").read_bytes())
        else:
            backup.unlink(missing_ok=True)
    _discard_journal(directory, ".cg-assets.pending", completed=True)
