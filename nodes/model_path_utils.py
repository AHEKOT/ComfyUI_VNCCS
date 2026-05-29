"""OS-agnostic model path lookup helpers for ComfyUI folder_paths."""

import os
import ntpath


def _normalize_path(value):
    return str(value or "").strip().replace("\\", os.sep).replace("/", os.sep)


def _is_absolute_any_os(value):
    raw = str(value or "").strip()
    return os.path.isabs(raw) or ntpath.isabs(raw) or bool(ntpath.splitdrive(raw)[0])


def path_variants(name):
    raw = str(name or "").strip()
    if not raw:
        return []

    variants = []
    for candidate in (raw, raw.replace("\\", "/"), raw.replace("/", "\\")):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _safe_get_folder_paths(folder_paths, category):
    try:
        return folder_paths.get_folder_paths(category) or []
    except Exception:
        return []


def _is_under_any_folder(path, folders):
    try:
        path_abs = os.path.abspath(_normalize_path(path))
        for folder in folders:
            folder_abs = os.path.abspath(_normalize_path(folder))
            if os.path.commonpath([folder_abs, path_abs]) == folder_abs:
                return True
    except Exception:
        return False
    return False


def get_full_path_agnostic(folder_paths, category, name, require_exists=False):
    folders = _safe_get_folder_paths(folder_paths, category)
    first_match = None

    for candidate in path_variants(name):
        try:
            found = folder_paths.get_full_path(category, candidate)
        except Exception:
            found = None
        if found:
            if os.path.exists(found):
                return found
            if first_match is None:
                first_match = found

        for folder in folders:
            joined = os.path.join(folder, _normalize_path(candidate))
            if os.path.exists(joined):
                return joined
            if first_match is None:
                first_match = joined

        if _is_absolute_any_os(candidate) and _is_under_any_folder(candidate, folders):
            normalized_candidate = _normalize_path(candidate)
            if os.path.exists(normalized_candidate):
                return normalized_candidate
            if first_match is None:
                first_match = normalized_candidate

    return None if require_exists else first_match


def basename_agnostic(path):
    normalized = str(path or "").rstrip("\\/")
    if not normalized:
        return ""
    return normalized.replace("\\", "/").rsplit("/", 1)[-1]
