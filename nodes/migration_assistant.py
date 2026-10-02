"""VNCCS Migration Assistant - legacy sheet to sprite migration widget."""

import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import server
    from aiohttp import web
except Exception:
    server = None
    web = None

try:
    from ..utils import (
        base_output_dir,
        ensure_safe_name,
        get_legacy_output_dir,
        safe_join_under,
        validate_privileged_request,
        character_storage_lock,
        staged_image_batch,
    )
except Exception:
    from utils import (
        base_output_dir,
        ensure_safe_name,
        get_legacy_output_dir,
        safe_join_under,
        validate_privileged_request,
        character_storage_lock,
        staged_image_batch,
    )


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_SPRITE_HEIGHT = 128
LEGACY_SHEET_COLS = 6
LEGACY_SHEET_ROWS = 2
RUNS: Dict[str, dict] = OrderedDict()
MAX_RUNS = 64
MAX_LOG_LINES = 500
RUN_TTL_SECONDS = 24 * 60 * 60
_RUNS_LOCK = threading.RLock()


def _prune_runs(*, reserve=False):
    cutoff = time.time() - RUN_TTL_SECONDS
    with _RUNS_LOCK:
        for key, run in list(RUNS.items()):
            if run.get("status") not in {"queued", "running"} and run.get("updated_at", 0) < cutoff:
                del RUNS[key]
        for key, run in list(RUNS.items()):
            if len(RUNS) <= MAX_RUNS - int(reserve):
                break
            if run.get("status") not in {"queued", "running"}:
                del RUNS[key]


def _start_job(callback, args, total, message):
    with _RUNS_LOCK:
        _prune_runs(reserve=True)
        if any(run.get("status") in {"queued", "running"} for run in RUNS.values()):
            return None
        run_id = uuid.uuid4().hex
        RUNS[run_id] = {"id": run_id, "status": "queued", "message": message, "log": [],
                        "created_at": time.time(), "updated_at": time.time(), "total": total, "current": 0}
        try:
            threading.Thread(target=callback, args=(run_id, *args), daemon=True).start()
        except Exception as error:
            _set_status(RUNS[run_id], status="error", error=str(error))
            raise
        return run_id


def _safe_legacy_name(name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9 _-]+", "_", str(name or "")).strip(" _-")
    candidate = re.sub(r"\s+", " ", candidate)
    if not candidate:
        candidate = "Migrated Character"
    return ensure_safe_name(candidate[:120], "character")


def _unique_character_name(base: str, used: set) -> str:
    candidate = _safe_legacy_name(base)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while True:
        stem = candidate[: max(1, 120 - len(f" {suffix}"))].rstrip()
        unique = f"{stem} {suffix}"
        if unique not in used:
            used.add(unique)
            return unique
        suffix += 1


def _has_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.array(image.getchannel("A"))
    return bool(np.any(alpha < 250))


def _estimate_foreground_mask(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]
    border = max(4, min(h, w) // 80)
    samples = np.concatenate([
        rgb[:border, :, :].reshape(-1, 3),
        rgb[-border:, :, :].reshape(-1, 3),
        rgb[:, :border, :].reshape(-1, 3),
        rgb[:, -border:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(samples, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    mask = (dist > 34).astype(np.uint8) * 255
    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _rgba_with_alpha(image: Image.Image) -> Tuple[Image.Image, np.ndarray, str]:
    rgba = image.convert("RGBA")
    if _has_alpha(rgba):
        alpha = np.array(rgba.getchannel("A"))
        return rgba, alpha, "existing alpha"
    mask = _estimate_foreground_mask(rgba)
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    return rgba, mask, "background removed"


def _has_visible_pixels(image: Image.Image) -> bool:
    if not _has_alpha(image):
        return True
    return image.getchannel("A").getbbox() is not None


def _legacy_grid_sprites(rgba: Image.Image, min_size: int) -> List[Image.Image]:
    width, height = rgba.size
    if width < LEGACY_SHEET_COLS * min_size or height < LEGACY_SHEET_ROWS * min_size:
        return []
    item_w, item_h = width // LEGACY_SHEET_COLS, height // LEGACY_SHEET_ROWS
    if item_w < min_size or item_h < min_size:
        return []
    usable_w, usable_h = item_w * LEGACY_SHEET_COLS, item_h * LEGACY_SHEET_ROWS
    if usable_w <= 0 or usable_h <= 0:
        return []

    sprites = []
    for row in range(LEGACY_SHEET_ROWS):
        for col in range(LEGACY_SHEET_COLS):
            crop = rgba.crop((col * item_w, row * item_h, (col + 1) * item_w, (row + 1) * item_h))
            if _has_visible_pixels(crop):
                sprites.append(crop)
    return sprites


def _pad_sprites_to_uniform_canvas(sprites: List[Image.Image]) -> List[Image.Image]:
    if not sprites:
        return sprites
    max_w = max(sprite.width for sprite in sprites)
    max_h = max(sprite.height for sprite in sprites)
    uniform = []
    for sprite in sprites:
        if sprite.width == max_w and sprite.height == max_h:
            uniform.append(sprite)
            continue
        canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        x = (max_w - sprite.width) // 2
        y = max_h - sprite.height
        canvas.alpha_composite(sprite.convert("RGBA"), (x, y))
        uniform.append(canvas)
    return uniform


def _crop_sprites(image: Image.Image, min_size: int = MIN_SPRITE_HEIGHT) -> List[Image.Image]:
    rgba, alpha, _ = _rgba_with_alpha(image)
    grid_sprites = _legacy_grid_sprites(rgba, min_size)
    if grid_sprites:
        return grid_sprites

    mask_uint8 = (alpha > 10).astype(np.uint8) * 255
    boxes = []
    if cv2 is not None:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                boxes.append((x, y, w, h))
    else:
        boxes = _connected_component_boxes(mask_uint8, min_size)

    if not boxes:
        w, h = rgba.size
        cols, rows = 6, 2
        item_w, item_h = w // cols, h // rows
        boxes = [
            (col * item_w, row * item_h, item_w, item_h)
            for row in range(rows)
            for col in range(cols)
            if item_w >= min_size and item_h >= min_size
        ]

    boxes.sort(key=lambda box: (box[1] // max(1, min_size), box[0]))
    sprites = []
    for x, y, w, h in boxes:
        crop = rgba.crop((x, y, x + w, y + h))
        if crop.width >= min_size and crop.height >= min_size:
            sprites.append(crop)
    return _pad_sprites_to_uniform_canvas(sprites)


def _connected_component_boxes(mask_uint8: np.ndarray, min_size: int) -> List[Tuple[int, int, int, int]]:
    binary = mask_uint8 > 0
    visited = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    boxes = []
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not binary[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            while stack:
                x, y = stack.pop()
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1
            if box_w >= min_size and box_h >= min_size:
                boxes.append((min_x, min_y, box_w, box_h))
    return boxes


def _scan_sheet_files(character_dir: str) -> List[dict]:
    sheets_root = os.path.join(character_dir, "Sheets")
    files = []
    if not os.path.isdir(sheets_root):
        return files
    for root, _dirs, filenames in os.walk(sheets_root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            rel = os.path.relpath(os.path.join(root, filename), sheets_root)
            parts = rel.split(os.sep)
            costume = parts[0] if len(parts) >= 3 else "Naked"
            emotion = parts[1] if len(parts) >= 3 else "neutral"
            files.append({
                "path": os.path.join(root, filename),
                "relative": rel,
                "costume": _safe_legacy_name(costume),
                "emotion": _safe_legacy_name(emotion),
            })
    # Legacy generation kept only the latest sheet per costume/emotion valid.
    latest = {}
    for sheet in files:
        key = (sheet["costume"], sheet["emotion"])
        rank = (os.path.getmtime(sheet["path"]), sheet["relative"].lower())
        previous = latest.get(key)
        if previous is None or rank > previous[0]:
            latest[key] = (rank, sheet)
    return sorted((item[1] for item in latest.values()), key=lambda item: item["relative"].lower())


def _sprite_files(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
        and os.path.splitext(name)[1].lower() == ".png"
        and name.startswith("sprite_")
    )


def _sprite_dirs(root: str) -> List[str]:
    dirs = []
    if not os.path.isdir(root):
        return dirs
    for walk_root, _dirs, filenames in os.walk(root):
        if any(
            os.path.isfile(os.path.join(walk_root, name))
            and os.path.splitext(name)[1].lower() == ".png"
            and name.startswith("sprite_")
            for name in filenames
        ):
            dirs.append(walk_root)
    return sorted(dirs)


def _sprite_size_groups(paths: List[str]) -> Dict[Tuple[int, int], List[str]]:
    groups: Dict[Tuple[int, int], List[str]] = {}
    for path in paths:
        try:
            with Image.open(path) as image:
                groups.setdefault(image.size, []).append(path)
        except Exception:
            continue
    return groups


def _scan_sprite_canvas_issues() -> List[dict]:
    output_root = base_output_dir()
    issues = []
    for directory in _sprite_dirs(output_root):
        paths = _sprite_files(directory)
        if len(paths) < 2:
            continue
        groups = _sprite_size_groups(paths)
        if len(groups) <= 1:
            continue
        rel = os.path.relpath(directory, output_root)
        issues.append({
            "directory": directory,
            "relative": rel,
            "count": len(paths),
            "sizes": [
                {"width": size[0], "height": size[1], "count": len(group)}
                for size, group in sorted(groups.items())
            ],
        })
    return issues


def _repair_sprite_canvas_folder(directory: str, backup: bool = True) -> dict:
    relative = os.path.relpath(directory, base_output_dir())
    root = safe_join_under(base_output_dir(), relative.split(os.sep)[0])
    with character_storage_lock(root):
        return _repair_sprite_canvas_folder_locked(directory, backup)


def _repair_sprite_canvas_folder_locked(directory: str, backup: bool = True) -> dict:
    paths = _sprite_files(directory)
    groups = _sprite_size_groups(paths)
    if len(groups) <= 1:
        return {"directory": directory, "changed": 0, "skipped": 0, "target_size": None}

    max_w = max(size[0] for size in groups)
    max_h = max(size[1] for size in groups)
    backup_dir = ""
    if backup:
        backup_dir = os.path.join(directory, f".vnccs_canvas_repair_backup_{int(time.time())}")
        os.makedirs(backup_dir, exist_ok=True)

    changed = skipped = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
            if rgba.size == (max_w, max_h):
                continue
            if not _has_alpha(rgba):
                skipped += 1
                continue
            if backup_dir:
                shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
            canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
            x = (max_w - rgba.width) // 2
            y = max_h - rgba.height
            canvas.alpha_composite(rgba, (x, y))
            canvas.save(path, format="PNG")
            changed += 1
        except Exception:
            skipped += 1

    return {
        "directory": directory,
        "changed": changed,
        "skipped": skipped,
        "target_size": {"width": max_w, "height": max_h},
        "backup_dir": backup_dir,
    }


def _run_canvas_repair(run_id: str, backup: bool = True) -> None:
    run = RUNS[run_id]
    try:
        issues = _scan_sprite_canvas_issues()
        _set_status(run, status="running", total=len(issues), current=0)
        if not issues:
            _set_status(run, status="done", current=0, total=0, results=[], message="No mismatched sprite canvases found")
            _log(run, "No mismatched sprite canvases found")
            return

        results = []
        for index, issue in enumerate(issues, start=1):
            _set_status(run, current=index, current_character=issue["relative"], current_sheet="")
            size_text = ", ".join(f"{item['width']}x{item['height']} ({item['count']})" for item in issue["sizes"])
            _log(run, f"Repairing {issue['relative']}: {size_text}")
            result = _repair_sprite_canvas_folder(issue["directory"], backup=backup)
            results.append(result)
            target = result.get("target_size") or {}
            _log(
                run,
                f"Repaired {issue['relative']}: padded {result.get('changed', 0)} sprite(s) "
                f"to {target.get('width')}x{target.get('height')}; skipped {result.get('skipped', 0)}",
            )

        _set_status(run, status="done", current=len(issues), results=results, message="Sprite canvas repair complete")
        _log(run, "Sprite canvas repair complete")
    except Exception as exc:
        _set_status(run, status="error", error=str(exc), message=str(exc))
        _log(run, f"Sprite canvas repair failed: {exc}")


def _ensure_sprite_alpha(path: str) -> bool:
    image = Image.open(path)
    if _has_alpha(image.convert("RGBA")):
        return False
    rgba, _mask, _source = _rgba_with_alpha(image)
    rgba.save(path)
    return True


def _copy_config(legacy_char_dir: str, new_char_dir: str, old_name: str, new_name: str) -> bool:
    dst = os.path.join(new_char_dir, f"{new_name}_config.json")
    # force applies to sprite conversion, never to an existing character profile.
    if os.path.exists(dst):
        return False
    candidates = [
        os.path.join(legacy_char_dir, f"{old_name}_config.json"),
        os.path.join(legacy_char_dir, f"{new_name}_config.json"),
    ]
    candidates.extend(
        os.path.join(legacy_char_dir, name)
        for name in os.listdir(legacy_char_dir)
        if name.endswith("_config.json")
    )
    for src in candidates:
        if not os.path.isfile(src):
            continue
        os.makedirs(new_char_dir, exist_ok=True)
        with open(src, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except Exception:
                data = None
        if isinstance(data, dict):
            info = data.setdefault("character_info", {})
            if isinstance(info, dict):
                info["name"] = new_name
            try:
                with open(dst, "x", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=4)
            except FileExistsError:
                return False
        else:
            try:
                with open(src, "rb") as in_handle, open(dst, "xb") as out_handle:
                    out_handle.write(in_handle.read())
            except FileExistsError:
                return False
        return True
    return False


def scan_legacy_characters() -> dict:
    legacy_root = get_legacy_output_dir()
    new_root = base_output_dir()
    used = set()

    characters = []
    if os.path.isdir(legacy_root):
        for name in sorted(os.listdir(legacy_root), key=str.lower):
            legacy_char_dir = os.path.join(legacy_root, name)
            if not os.path.isdir(legacy_char_dir):
                continue
            new_name = _unique_character_name(name, used)
            sheet_files = _scan_sheet_files(legacy_char_dir)
            existing_sprites = 0
            missing_targets = 0
            for sheet in sheet_files:
                target_dir = os.path.join(new_root, new_name, "Sprites", sheet["costume"], sheet["emotion"])
                count = len(_sprite_files(target_dir))
                existing_sprites += count
                if count == 0:
                    missing_targets += 1
            status = "migrated" if sheet_files and missing_targets == 0 else "needs_migration"
            characters.append({
                "legacy_name": name,
                "new_name": new_name,
                "sheet_count": len(sheet_files),
                "existing_sprite_count": existing_sprites,
                "missing_sprite_targets": missing_targets,
                "status": status,
                "config_exists": any(
                    os.path.isfile(os.path.join(legacy_char_dir, file_name))
                    and file_name.endswith("_config.json")
                    for file_name in os.listdir(legacy_char_dir)
                ),
            })
    return {
        "legacy_root": legacy_root,
        "new_root": new_root,
        "characters": characters,
    }


def _set_status(run: dict, **updates) -> None:
    with _RUNS_LOCK:
        run.update(updates)
        run["updated_at"] = time.time()


def _log(run: dict, message: str) -> None:
    with _RUNS_LOCK:
        run.setdefault("log", []).append(message)
        run["log"] = run["log"][-MAX_LOG_LINES:]
        run["message"] = message
        run["updated_at"] = time.time()
    print(f"[VNCCS Migration Assistant] {message}")


def _migrate_character(run: dict, legacy_name: str, new_name: str, force: bool, retry_sheets=None) -> dict:
    root = safe_join_under(base_output_dir(), new_name)
    with character_storage_lock(root):
        return _migrate_character_locked(run, legacy_name, new_name, force, retry_sheets)


def _migrate_character_locked(run: dict, legacy_name: str, new_name: str, force: bool, retry_sheets=None) -> dict:
    legacy_root = get_legacy_output_dir()
    new_root = base_output_dir()
    legacy_char_dir = safe_join_under(legacy_root, legacy_name)
    new_char_dir = safe_join_under(new_root, new_name)
    os.makedirs(new_char_dir, exist_ok=True)
    config_copied = _copy_config(legacy_char_dir, new_char_dir, legacy_name, new_name)
    sheet_files = _scan_sheet_files(legacy_char_dir)
    if retry_sheets is not None:
        known = {sheet["relative"] for sheet in sheet_files}
        if not retry_sheets or not set(retry_sheets).issubset(known):
            raise ValueError("Retry includes an unknown legacy sheet")
        sheet_files = [sheet for sheet in sheet_files if sheet["relative"] in retry_sheets]
    saved = skipped = alpha_fixed = failed = 0
    failed_paths = []

    for index, sheet in enumerate(sheet_files, start=1):
        _set_status(run, current_sheet=sheet["relative"])
        try:
            target_dir = safe_join_under(new_char_dir, "Sprites", sheet["costume"], sheet["emotion"])
            existing = _sprite_files(target_dir)
            if existing and not force:
                for sprite_path in existing:
                    if _ensure_sprite_alpha(sprite_path):
                        alpha_fixed += 1
                skipped += len(existing)
                _log(run, f"{new_name}: skipped {sheet['relative']} because sprites already exist")
                continue

            os.makedirs(target_dir, exist_ok=True)
            with Image.open(sheet["path"]) as image:
                sprites = _crop_sprites(image)
            if not sprites:
                failed += 1
                failed_paths.append(sheet["relative"])
                _log(run, f"{new_name}: no sprites detected in {sheet['relative']}")
                continue
            with staged_image_batch(target_dir, version_existing=force, lock_root=new_char_dir) as stage:
                for sprite_index, sprite in enumerate(sprites):
                    filename = f"sprite_{sheet['emotion']}_{sprite_index:04d}.png"
                    sprite.save(os.path.join(stage, filename))
            saved += len(sprites)
            _log(run, f"{new_name}: {index}/{len(sheet_files)} saved {len(sprites)} sprite(s) from {sheet['relative']}")
        except Exception as exc:
            failed += 1
            failed_paths.append(sheet["relative"])
            _log(run, f"{new_name}: failed {sheet['relative']}: {exc}")

    return {
        "legacy_name": legacy_name,
        "new_name": new_name,
        "sheet_count": len(sheet_files),
        "sprites_saved": saved,
        "sprites_skipped": skipped,
        "sprites_alpha_fixed": alpha_fixed,
        "failed_sheets": failed,
        "failed_sheet_paths": failed_paths,
        "config_copied": config_copied,
    }


def _run_migration(run_id: str, characters: List[dict], force: bool) -> None:
    run = RUNS[run_id]
    results = []
    try:
        _set_status(run, status="running", total=len(characters), current=0)
        for index, item in enumerate(characters, start=1):
            legacy_name = item.get("legacy_name") or item.get("name")
            new_name = _safe_legacy_name(item.get("new_name") or legacy_name)
            _set_status(run, current=index, current_character=legacy_name, current_sheet="")
            _log(run, f"Processing {legacy_name} -> {new_name}")
            if item.get("retry_sheets") is not None:
                results.append(_migrate_character(run, legacy_name, new_name, force, item["retry_sheets"]))
            else:
                results.append(_migrate_character(run, legacy_name, new_name, force))
        failures = sum(result.get("failed_sheets", 0) for result in results)
        message = f"Migration incomplete: {failures} sheet(s) failed" if failures else "Migration complete"
        _set_status(run, status="partial" if failures else "done", current=len(characters),
                    results=results, failed_sheets=failures,
                    failed_characters=[result["legacy_name"] for result in results if result.get("failed_sheets", 0)],
                    message=message)
        _log(run, message)
    except Exception as exc:
        failed_characters = [result.get("legacy_name") for result in results if result.get("failed_sheets", 0)]
        failed_characters.extend(item.get("legacy_name") or item.get("name") for item in characters[len(results):])
        _set_status(run, status="error", error=str(exc), message=str(exc), results=results,
                    failed_sheets=sum(result.get("failed_sheets", 0) for result in results),
                    failed_characters=list(dict.fromkeys(name for name in failed_characters if name)))
        _log(run, f"Migration failed: {exc}")


class VNCCS_MigrationAssistant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "VNCCS"

    def run(self):
        return ()


if server and web:
    @server.PromptServer.instance.routes.get("/vnccs/migration/characters")
    async def vnccs_migration_characters(request):
        try:
            return web.json_response(scan_legacy_characters())
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post("/vnccs/migration/start")
    async def vnccs_migration_start(request):
        try:
            validate_privileged_request(request)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=403)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(data, dict) or not isinstance(data.get("characters", []), list) or any(not isinstance(name, str) for name in data.get("characters", [])):
            return web.json_response({"error": "Characters must be a list of names"}, status=400)
        scan = scan_legacy_characters()
        by_legacy = {item["legacy_name"]: item for item in scan["characters"]}
        selected = data.get("characters") or list(by_legacy)
        characters = [by_legacy[name] for name in selected if name in by_legacy]
        retries = data.get("retry_sheets", {})
        if not isinstance(retries, dict) or any(not isinstance(paths, list) or any(not isinstance(path, str) for path in paths) for paths in retries.values()):
            return web.json_response({"error": "Invalid retry sheet list"}, status=400)
        characters = [{**item, "retry_sheets": retries[item["legacy_name"]]} if item["legacy_name"] in retries else item for item in characters]
        if not characters:
            return web.json_response({"error": "No legacy characters selected"}, status=400)
        run_id = _start_job(_run_migration, (characters, bool(data.get("force", False))), len(characters), "Queued")
        if run_id is None:
            return web.json_response({"error": "A migration or repair job is already running. Retry after it completes."}, status=409)
        return web.json_response({"run_id": run_id})

    @server.PromptServer.instance.routes.post("/vnccs/migration/repair-sprites")
    async def vnccs_migration_repair_sprites(request):
        try:
            validate_privileged_request(request)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=403)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(data, dict):
            return web.json_response({"error": "Request body must be an object"}, status=400)
        run_id = _start_job(_run_canvas_repair, (bool(data.get("backup", True)),), 0, "Queued sprite canvas repair")
        if run_id is None:
            return web.json_response({"error": "A migration or repair job is already running. Retry after it completes."}, status=409)
        return web.json_response({"run_id": run_id})

    @server.PromptServer.instance.routes.get("/vnccs/migration/status/{run_id}")
    async def vnccs_migration_status(request):
        run_id = request.match_info.get("run_id", "")
        with _RUNS_LOCK:
            _prune_runs()
            run = RUNS.get(run_id)
            if not run:
                return web.json_response({"error": "Run not found"}, status=404)
            snapshot = {**run, "log": list(run.get("log", [])), "results": list(run.get("results", []))}
        return web.json_response(snapshot)


NODE_CLASS_MAPPINGS = {
    "VNCCS_MigrationAssistant": VNCCS_MigrationAssistant,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_MigrationAssistant": "VNCCS Migration Assistant",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_MigrationAssistant": "VNCCS",
}
