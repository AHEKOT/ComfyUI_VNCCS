"""Bounded snapshots for restoring generator UI after transport interruptions."""

from collections import OrderedDict
from copy import deepcopy
import re
import threading
import time
import uuid
import hashlib


_states = OrderedDict()
_lock = threading.Lock()
_revision = 0
MAX_STATES = 64
MAX_IMAGES = 512
MAX_IMAGE_CHARS = 2_000_000
TTL_SECONDS = 24 * 60 * 60
PROGRESS_EPOCH = uuid.uuid4().hex


def expire_cache_progress(scope_digest):
    """Do not offer deleted transient preview files after disk cache eviction."""
    global _revision
    with _lock:
        for scope, state in _states.items():
            if hashlib.sha256(scope.encode("utf-8")).hexdigest() == scope_digest:
                _revision += 1
                state.update(revision=_revision, stages={}, active_stage=None,
                             error={"message": "Cached results expired. Run this generator again.", "stage": None})


def _valid_scope(scope):
    return isinstance(scope, str) and re.fullmatch(r"[A-Za-z0-9:_-]{1,200}", scope)


def begin_progress(scope, node_id, run_id=None, start_stage=None, preserve_images=False, request_id=None):
    if not _valid_scope(scope):
        return None
    global _revision
    with _lock:
        _revision += 1
        previous = _states.pop(scope, None)
        stages = {}
        if previous and start_stage in previous["stages"]:
            resetting = False
            for key, value in previous["stages"].items():
                resetting = resetting or key == start_stage
                if not resetting:
                    stages[key] = deepcopy(value)
                elif preserve_images:
                    stages[key] = {"status": "waiting", "images": list(value.get("images") or []), "message": ""}
        _states[scope] = {"node_id": str(node_id), "scope": scope, "revision": _revision,
                          "epoch": PROGRESS_EPOCH, "run_id": run_id or uuid.uuid4().hex,
                          "request_id": request_id, "updated": time.monotonic(), "stages": stages,
                          "active_stage": start_stage, "error": None}
        while len(_states) > MAX_STATES:
            _states.popitem(last=False)
    return scope


def record_progress(scope, event, run_id=None):
    global _revision
    with _lock:
        state = _states.get(scope)
        if not state or state["node_id"] != str(event.get("node_id")):
            return None
        if run_id is not None and state["run_id"] != run_id:
            return None
        _revision += 1
        state["revision"] = _revision
        state["updated"] = time.monotonic()
        stage = str(event.get("stage", ""))
        if stage == "error":
            state["error"] = {"message": str(event.get("message", ""))[:4000], "stage": state["active_stage"]}
            for value in state["stages"].values():
                if value["status"] == "running":
                    value.update(status="error", message=str(event.get("message", ""))[:4000])
        else:
            state["active_stage"] = stage
            previous = state["stages"].get(stage, {})
            images = previous.get("images")
            if "images" in event:
                incoming = list(event.get("images") or [])[:MAX_IMAGES]
                if event.get("replace_images"):
                    images = list(images or [])
                    start = max(0, min(MAX_IMAGES, int(event.get("preview_start", 0))))
                    while len(images) < start:
                        images.append("")
                    images[start:start + len(incoming)] = incoming
                elif event.get("append_images"):
                    images = list(images or []) + incoming
                else:
                    images = incoming
                # Prefer stable file URLs. Bound base64 fallbacks as well as count.
                bounded, size = [], 0
                for image in images[:MAX_IMAGES]:
                    image = image if isinstance(image, str) else ""
                    size += len(image)
                    bounded.append(image if size <= MAX_IMAGE_CHARS else "")
                images = bounded
            state["stages"][stage] = {
                "status": event.get("status", "waiting"), "images": images,
                "message": str(event.get("message", ""))[:4000],
                "current": event.get("current"), "total": event.get("total"),
            }
            # The byte budget covers the whole node, not each stage separately.
            remaining = MAX_IMAGE_CHARS
            for value in reversed(list(state["stages"].values())):
                for index, image in enumerate(value.get("images") or []):
                    if len(image) > remaining:
                        value["images"][index] = ""
                    else:
                        remaining -= len(image)
        return _revision


def progress_snapshot(scope):
    if not _valid_scope(scope):
        return None
    with _lock:
        state = _states.get(scope)
        if state and time.monotonic() - state["updated"] > TTL_SECONDS:
            del _states[scope]
            return None
        if not state:
            return None
        return {key: deepcopy(value) for key, value in state.items() if key != "updated"}
