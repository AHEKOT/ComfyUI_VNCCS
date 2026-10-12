"""Preserve legacy custom emotions separately from bundled defaults."""

import json
import os
import shutil

from ..utils import atomic_output_path, character_storage_lock, ensure_safe_name, safe_join_under


def _read_catalog(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if (not isinstance(data, dict) or any(not isinstance(entries, list)
            or any(not isinstance(entry, dict) or not isinstance(entry.get("safe_name"), str)
                   or not isinstance(entry.get("key"), str) or not isinstance(entry.get("description"), str)
                   for entry in entries) for entries in data.values())):
        raise ValueError("Invalid emotion catalog; repair it before saving")
    return data


def save_emotion_library(path, data):
    with atomic_output_path(path) as temporary:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
            handle.write("\n")


def load_emotion_library(default_path, user_path):
    with character_storage_lock(user_path):
        defaults = _read_catalog(default_path)
        try:
            users = _read_catalog(user_path)
        except FileNotFoundError:
            users = {}
        known = {entry["safe_name"] for entries in users.values() for entry in entries}
        legacy = [entry for entry in defaults.get("Custom", []) if entry["safe_name"] not in known]
        if legacy:
            # Copy before publication; old files remain recoverable on any failure.
            for entry in legacy:
                name = ensure_safe_name(entry["safe_name"], "emotion")
                source = safe_join_under(os.path.dirname(default_path), "images", f"{name}.webp")
                target = safe_join_under(os.path.dirname(user_path), "images", f"{name}.webp")
                if os.path.isfile(source) and not os.path.exists(target):
                    with atomic_output_path(target) as temporary:
                        shutil.copyfile(source, temporary)
            users.setdefault("Custom", []).extend(legacy)
            save_emotion_library(user_path, users)
        merged = {**defaults}
        for category, entries in users.items():
            user_names = {entry["safe_name"] for entry in entries}
            merged[category] = [entry for entry in defaults.get(category, [])
                                if entry["safe_name"] not in user_names] + entries
        return merged
