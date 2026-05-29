"""Compatibility wrappers for safety helpers.

These helpers live in utils.py in current VNCCS builds. Keeping fallback copies
here prevents a partially updated install from failing all node registration.
"""

import os
import re

try:
    from ..utils import ensure_safe_name, safe_join_under, safe_relative_path
except Exception:
    _SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,120}$")

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
        base_abs = os.path.abspath(base)
        target = os.path.abspath(os.path.join(base_abs, *[str(part) for part in parts]))
        if os.path.commonpath([base_abs, target]) != base_abs:
            raise ValueError("path escapes allowed directory")
        return target

    def safe_relative_path(value, field="path"):
        if value is None:
            raise ValueError(f"{field} is required")
        normalized = str(value).strip().replace("\\", "/")
        if not normalized:
            raise ValueError(f"{field} is required")
        if normalized.startswith("/") or normalized.startswith("~"):
            raise ValueError(f"{field} must be relative")
        parts = [part for part in normalized.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"{field} contains invalid path traversal")
        if any("\0" in part for part in parts):
            raise ValueError(f"{field} contains invalid characters")
        return "/".join(parts)
