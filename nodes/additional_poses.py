"""Select an existing sprite as the reference for additional poses."""

import json
import os
from pathlib import Path

from PIL import Image

from ..utils import (
    base_output_dir, character_dir, character_storage_lock, config_path,
    ensure_safe_name, file_fingerprint, is_sprite_archive_dir, list_characters,
    list_costumes, load_config, normalize_filesystem_path, safe_join_under, sheets_dir,
)

IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".bmp"}


def _files(directory):
    return sorted((path for path in directory.iterdir()
                   if not path.is_symlink() and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda path: path.name)


def _config(character):
    config = load_config(character, strict=True)
    if config is None:
        raise FileNotFoundError(f"Character '{character}' has no saved configuration.")
    return config


def _costume_root(character, costume):
    root = Path(safe_join_under(character_dir(character), "Sprites", ensure_safe_name(costume, "costume")))
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError("The selected costume must be in the character's Sprites folder.")
    return root


def _sprite_groups(root):
    if not root.is_dir():
        return {}
    groups = {}
    for name in ("Neutral", "neutral", ""):
        directory = root / name
        if not directory.is_symlink() and directory.is_dir() and (files := _files(directory)):
            groups["Neutral"] = files
            break
    for directory in sorted(root.iterdir()):
        name = directory.name
        if (directory.is_symlink() or not directory.is_dir() or name.casefold() == "neutral"
                or is_sprite_archive_dir(name)):
            continue
        try:
            if ensure_safe_name(name, "emotion") != name:
                continue
        except ValueError:
            continue
        if files := _files(directory):
            groups[name] = files
    return groups


def catalog(character="", costume=""):
    if not character:
        characters = []
        for name in list_characters():
            try:
                if ensure_safe_name(name, "character") == name and os.path.isfile(config_path(name)):
                    characters.append(name)
            except ValueError:
                continue
        return {"characters": characters}
    with character_storage_lock(character_dir(character)):
        _config(character)
        if not costume:
            return {"costumes": list_costumes(character)}
        return {"emotions": _sprite_groups(_costume_root(character, costume))}


def destination(sheets_path):
    base = base_output_dir()
    path = normalize_filesystem_path(sheets_path)
    parts = os.path.relpath(path, base).split(os.sep) if os.path.isabs(path) else []
    if len(parts) != 5 or parts[1] != "Sheets" or parts[4] != "sheet_neutral":
        raise ValueError("Connect sheets_path from VNCCS Additional Poses.")
    character, _, costume, emotion, _ = parts
    if path != sheets_dir(character, costume, emotion):
        raise ValueError("Invalid Additional Poses destination.")
    return character, costume, emotion


def sprite_paths(character, costume, emotion):
    emotion = ensure_safe_name(emotion, "emotion")
    name = "Neutral" if emotion.casefold() == "neutral" else emotion
    paths = _sprite_groups(_costume_root(character, costume)).get(name, [])
    if not paths:
        raise FileNotFoundError(f"No sprites found for {character} / {costume} / {emotion}.")
    return paths


class VNCCS_AdditionalPoses:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"widget_data": ("STRING", {"default": "{}", "dynamicPrompts": False})}}

    RETURN_TYPES = ("IMAGE", "STRING", "*")
    RETURN_NAMES = ("character", "sheets_path", "background")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @staticmethod
    def _selection(widget_data):
        data = json.loads(widget_data)
        if not isinstance(data, dict):
            raise ValueError("Invalid Additional Poses selection.")
        return tuple(ensure_safe_name(data.get(field), field) for field in ("character", "costume", "emotion")), data.get("sprite")

    @staticmethod
    def _selected_path(character, costume, emotion, sprite):
        paths = sprite_paths(character, costume, emotion)
        if not sprite:
            return paths[0]
        for path in paths:
            if path.name == sprite:
                return path
        raise FileNotFoundError(f"Sprite '{sprite}' is missing from {character} / {costume} / {emotion}.")

    @classmethod
    def IS_CHANGED(cls, widget_data="{}"):
        try:
            selection, sprite = cls._selection(widget_data)
            with character_storage_lock(character_dir(selection[0])):
                path = cls._selected_path(*selection, sprite)
                return (file_fingerprint(config_path(selection[0])), file_fingerprint(str(path)))
        except (ValueError, OSError):
            return float("NaN")

    def process(self, widget_data="{}"):
        from .clothes_designer import ClothesDesigner

        selection, sprite = self._selection(widget_data)
        with character_storage_lock(character_dir(selection[0])):
            config = _config(selection[0])
            path = self._selected_path(*selection, sprite)
            with Image.open(path) as image:
                tensor = ClothesDesigner._pil_image_tensor(image)
            background = config.get("character_info", {}).get("background_color") or "Green"
            return tensor, sheets_dir(*selection), ClothesDesigner._background_output_value(background)


NODE_CLASS_MAPPINGS = {"VNCCS_AdditionalPoses": VNCCS_AdditionalPoses}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCS_AdditionalPoses": "VNCCS Additional Poses"}
