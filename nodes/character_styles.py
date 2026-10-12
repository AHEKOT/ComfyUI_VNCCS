"""User-owned style library; packaged updates never write this file."""
from ..operation_logger import log_event

import json
import math
import os
import re
import threading
import time
import uuid
import numpy as np
from PIL import Image

from ..utils import atomic_output_path, safe_join_under


USER_CHARACTER_STYLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "character_template", "character_styles.user.json",
)
_STYLE_LOCK = threading.RLock()
_TEXT_LIMITS = {"label": 100, "description": 240, "reference": 500, "prompt": 16000}
STYLE_PREVIEWS_DIR = os.path.join(os.path.dirname(USER_CHARACTER_STYLES_PATH), "style_previews")


def style_preview_path(style_id):
    if not isinstance(style_id, str) or not re.fullmatch(r"[a-z0-9_]{1,100}", style_id):
        raise ValueError("Invalid style ID")
    return safe_join_under(STYLE_PREVIEWS_DIR, style_id + ".webp")


def style_preview_url(style_id):
    path = style_preview_path(style_id)
    if not os.path.isfile(path):
        return ""
    stat = os.stat(path)
    return f"/vnccs/character_styles/preview?style={style_id}&v={stat.st_mtime_ns}_{stat.st_size}"


def square_style_resolution(target_size):
    # Creator resolution scale stores megapixels multiplied by 1024.
    target_size = max(1024, min(4096, float(target_size)))
    side = int(round(math.sqrt(target_size * 1024) / 16) * 16)
    return side, side


def flatten_style_preview(image):
    """Composite alpha onto the style card's 145-degree dark gradient."""
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")
    image = image.convert("RGBA")
    width, height = image.size
    dx, dy = math.sin(math.radians(145)), -math.cos(math.radians(145))
    weight = ((np.arange(height)[:, None] + .5) * dy
              + (np.arange(width)[None, :] + .5) * dx) / (width * dx + height * dy)
    start, end = np.array([41, 32, 52]), np.array([23, 19, 31])
    pixels = np.rint(start + (end - start) * weight[:, :, None]).astype(np.uint8)
    background = Image.fromarray(pixels)
    background.paste(image, mask=image.getchannel("A"))
    return background


def save_style_preview(style_id, image):
    if image.width != image.height:
        raise ValueError("Style preview must be generated as a square image")
    path = style_preview_path(style_id)
    mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
    image = image.convert(mode)
    if image.width > 1024:
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image = flatten_style_preview(image)
    started = time.perf_counter()
    with atomic_output_path(path) as temporary:
        image.save(temporary, format="WEBP", quality=90, method=6)
        encoded = time.perf_counter()
        with Image.open(temporary) as stored:
            stored.load()
            if stored.format != "WEBP" or stored.size != image.size:
                raise OSError("Style preview could not be verified on disk")
        verified = time.perf_counter()
    log_event("preview_saved", component="Styles", style=style_id, width=image.width, height=image.height, encode_s=round(encoded - started, 3), verify_s=round(verified - encoded, 3), publish_s=round(time.perf_counter() - verified, 3))
    return {"style_id": style_id, "image": style_preview_url(style_id), "width": image.width, "height": image.height, "saved": True, "path": path}


def validate_user_style(value):
    if not isinstance(value, dict):
        raise ValueError("Style must be a JSON object")
    result = {}
    for key, limit in _TEXT_LIMITS.items():
        text = value.get(key, "")
        if not isinstance(text, str) or len(text) > limit or "\x00" in text:
            raise ValueError(f"Invalid {key}; maximum length is {limit}")
        result[key] = text.strip()
    if not result["label"] or not result["prompt"]:
        raise ValueError("Name and style prompt are required")
    style_id = value.get("id", "")
    if not isinstance(style_id, str) or not re.fullmatch(r"user_[a-f0-9]{32}", style_id):
        raise ValueError("Invalid user style ID")
    result.update(id=style_id, image="", user=True)
    return result


def load_user_styles(path=None):
    path = path or USER_CHARACTER_STYLES_PATH
    with _STYLE_LOCK:
        if not os.path.exists(path):
            return []
        if os.path.getsize(path) > 18000000:
            raise ValueError("User style library is too large")
        with open(path, encoding="utf-8") as source:
            library = json.load(source)
        if not isinstance(library, dict) or not isinstance(library.get("styles"), list):
            raise ValueError("Invalid user style library; the file has been preserved")
        if len(library["styles"]) > 1000:
            raise ValueError("User style library supports up to 1000 styles")
        styles = [validate_user_style(style) for style in library["styles"]]
        if len({style["id"] for style in styles}) != len(styles):
            raise ValueError("Duplicate user style IDs")
        return styles


def save_user_style(value, path=None):
    path = path or USER_CHARACTER_STYLES_PATH
    if not isinstance(value, dict):
        raise ValueError("Style must be a JSON object")
    supplied_id = value.get("id")
    style_id = "user_" + uuid.uuid4().hex if supplied_id is None or supplied_id == "" else supplied_id
    style = validate_user_style({**value, "id": style_id})
    with _STYLE_LOCK:
        styles = load_user_styles(path)
        index = next((i for i, item in enumerate(styles) if item["id"] == style["id"]), None)
        if supplied_id and index is None:
            raise ValueError("User style no longer exists; save it as a new style")
        if index is None:
            if len(styles) >= 1000:
                raise ValueError("User style library supports up to 1000 styles")
            styles.append(style)
        else:
            styles[index] = style
        with atomic_output_path(path) as temporary:
            with open(temporary, "w", encoding="utf-8") as output:
                json.dump({"version": 1, "styles": styles}, output, ensure_ascii=False, indent=2)
                output.write("\n")
    return style


def delete_user_style(style_id, path=None):
    if not isinstance(style_id, str) or not re.fullmatch(r"user_[a-f0-9]{32}", style_id):
        raise ValueError("Only user styles can be deleted")
    path = path or USER_CHARACTER_STYLES_PATH
    with _STYLE_LOCK:
        styles = load_user_styles(path)
        remaining = [style for style in styles if style["id"] != style_id]
        if len(remaining) == len(styles):
            return False
        preview = style_preview_path(style_id)
        if os.path.islink(preview):
            raise ValueError("Cannot delete a linked style preview")
        previous = None
        if os.path.isfile(preview):
            with open(preview, "rb") as source:
                previous = source.read()
        removed = False
        try:
            with atomic_output_path(path) as temporary:
                with open(temporary, "w", encoding="utf-8") as output:
                    json.dump({"version": 1, "styles": remaining}, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                if previous is not None:
                    os.unlink(preview)
                    removed = True
        except OSError:
            if removed:
                with atomic_output_path(preview) as temporary:
                    with open(temporary, "wb") as output:
                        output.write(previous)
            raise
        return True


def save_user_style_preview(style_id, image):
    # A render finishing after deletion must not recreate an orphan preview.
    with _STYLE_LOCK:
        if not any(style["id"] == style_id for style in load_user_styles()):
            raise ValueError("User style no longer exists")
        return save_style_preview(style_id, image)
