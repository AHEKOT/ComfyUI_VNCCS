"""Compatibility wrappers for safety helpers.

These helpers live in utils.py in current VNCCS builds. Keeping fallback copies
here prevents a partially updated install from failing all node registration.
"""

import os
import ntpath
import re

try:
    from ..utils import (
        ensure_safe_name,
        is_absolute_path_any_os,
        is_path_under,
        normalize_filesystem_path,
        safe_join_under,
        safe_relative_path,
    )
except Exception:
    _SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,120}$")

    def normalize_filesystem_path(value):
        return str(value or "").strip().replace("\\", os.sep).replace("/", os.sep)

    def is_absolute_path_any_os(value):
        raw = str(value or "").strip()
        normalized = raw.replace("\\", "/")
        return (
            os.path.isabs(raw)
            or ntpath.isabs(raw)
            or bool(ntpath.splitdrive(raw)[0])
            or normalized.startswith("/")
            or normalized.startswith("~")
        )

    def is_path_under(base, path):
        try:
            base_abs = os.path.abspath(normalize_filesystem_path(base))
            path_abs = os.path.abspath(normalize_filesystem_path(path))
            return os.path.commonpath([base_abs, path_abs]) == base_abs
        except Exception:
            return False

    def ensure_safe_name(value, field="name"):
        if value is None:
            raise ValueError(f"{field} is required")
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field} is required")
        if value in {".", ".."} or ".." in value:
            raise ValueError(f"{field} contains invalid path traversal")
        if not _SAFE_NAME_RE.match(value):
            raise ValueError(f"{field} may only contain letters, numbers, spaces, underscores and hyphens")
        return value

    def safe_join_under(base, *parts):
        base_abs = os.path.abspath(normalize_filesystem_path(base))
        normalized_parts = []
        for part in parts:
            raw = str(part)
            if is_absolute_path_any_os(raw):
                raise ValueError("path escapes allowed directory")
            items = [item for item in raw.replace("\\", "/").split("/") if item]
            if any(item in {".", ".."} or "\0" in item for item in items):
                raise ValueError("path escapes allowed directory")
            normalized_parts.extend(items)
        target = os.path.abspath(os.path.join(base_abs, *normalized_parts))
        if not is_path_under(base_abs, target):
            raise ValueError("path escapes allowed directory")
        return target

    def safe_relative_path(value, field="path"):
        if value is None:
            raise ValueError(f"{field} is required")
        normalized = str(value).strip().replace("\\", "/")
        if not normalized:
            raise ValueError(f"{field} is required")
        if is_absolute_path_any_os(normalized):
            raise ValueError(f"{field} must be relative")
        parts = [part for part in normalized.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"{field} contains invalid path traversal")
        if any("\0" in part for part in parts):
            raise ValueError(f"{field} contains invalid characters")
        return "/".join(parts)
