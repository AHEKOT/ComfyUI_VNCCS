"""Bounded, workflow-scoped live inputs for regenerating without submitting a graph."""

from collections import OrderedDict
import hashlib
import os
import threading
import time
import inspect
import json
import re
import shutil
from contextlib import contextmanager
from functools import wraps
from .progress_state import _valid_scope, expire_cache_progress

_LIVE_GENERATOR_CONTEXTS = OrderedDict()
_lock = threading.RLock()
MAX_CONTEXTS = 8
TTL_SECONDS = 60 * 60
MAX_DISK_CACHES = 8
DISK_TTL_SECONDS = 24 * 60 * 60
_execution_locks = {}


@contextmanager
def generator_execution_lock(unique_id, scope=None):
    """Serialize normal execution and Regenerate sharing the same disk cache."""
    if isinstance(unique_id, list):
        unique_id = unique_id[0] if unique_id else None
    scope = scope if _valid_scope(scope) else None
    key = _generator_context_key(unique_id, scope)
    with _lock:
        entry = _execution_locks.setdefault(key, [threading.RLock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _lock:
            entry[1] -= 1
            if not entry[1]:
                _execution_locks.pop(key, None)
                context = _LIVE_GENERATOR_CONTEXTS.get(key)
                if context is not None:
                    context["updated_at"] = time.monotonic()
                _prune_contexts()
                if context is not None:
                    try:
                        prune_workflow_caches(context["cache_dir"])
                    except OSError as error:
                        # Retry on the next run; cleanup must not mask its result.
                        print(f"[VNCCS] Could not prune inactive generator caches: {error}")


def serialized_generator(process):
    names = tuple(inspect.signature(process).parameters)

    @wraps(process)
    def run(*args, **kwargs):
        arguments = {**dict(zip(names, args)), **kwargs}
        payload = arguments.get("widget_data", {})
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                payload = {}
        ui = payload.get("ui") if isinstance(payload, dict) else None
        scope = ui.get("progress_scope") if isinstance(ui, dict) else None
        with generator_execution_lock(arguments.get("unique_id"), scope):
            return process(*args, **kwargs)
    return run


def _forget_generator_context(unique_id, scope=None):
    with _lock:
        _LIVE_GENERATOR_CONTEXTS.pop(_generator_context_key(unique_id, scope), None)


def _generator_context_key(unique_id, scope=None):
    node_id = str(unique_id or "").strip()
    return (str(scope), node_id) if scope else node_id


def scoped_cache_dir(cache_dir, scope=None):
    """Keep tensors and previews isolated as well as the in-memory pipe."""
    if not cache_dir or not scope:
        return cache_dir
    identity = hashlib.sha256(str(scope).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, "workflows", identity)


def prune_workflow_caches(cache_dir, *, touch=False):
    """Bound only hashed transient caches, never character outputs or legacy caches."""
    if not cache_dir:
        return
    directory = os.path.abspath(cache_dir)
    workflows = os.path.dirname(directory)
    node_root = os.path.dirname(workflows)
    poses = os.path.dirname(node_root)
    cache_root = os.path.dirname(poses)
    if (not re.fullmatch(r"[a-f0-9]{64}", os.path.basename(directory))
            or os.path.basename(workflows) != "workflows"
            or os.path.basename(poses) != "poses" or os.path.basename(cache_root) != "cache"
            or any(os.path.islink(path) for path in (directory, workflows, node_root, poses, cache_root))):
        return
    # Hold the registry lock during removal so a new execution cannot start
    # using an inactive directory between the protection check and deletion.
    with _lock:
        if touch:
            os.makedirs(directory, exist_ok=True)
            os.utime(directory, None)
        if not os.path.isdir(workflows):
            return
        protected = {hashlib.sha256(key[0].encode("utf-8")).hexdigest()
                     for key in set(_LIVE_GENERATOR_CONTEXTS) | set(_execution_locks)
                     if isinstance(key, tuple)}
        protected.add(os.path.basename(directory))
        with os.scandir(workflows) as entries:
            candidates = sorted(
                [(entry.stat(follow_symlinks=False).st_mtime, entry.name, entry.path)
                 for entry in entries if re.fullmatch(r"[a-f0-9]{64}", entry.name)
                 and entry.is_dir(follow_symlinks=False)],
                reverse=True,
            )
        retained = sum(name in protected for _, name, _ in candidates)
        for updated, name, path in candidates:
            if name in protected:
                continue
            if time.time() - updated <= DISK_TTL_SECONDS and retained < MAX_DISK_CACHES:
                retained += 1
                continue
            shutil.rmtree(path)
            expire_cache_progress(name)


def _prune_contexts():
    now = time.monotonic()
    for key, context in list(_LIVE_GENERATOR_CONTEXTS.items()):
        if key not in _execution_locks and now - context["updated_at"] > TTL_SECONDS:
            del _LIVE_GENERATOR_CONTEXTS[key]
    while len(_LIVE_GENERATOR_CONTEXTS) > MAX_CONTEXTS:
        key = next((key for key in _LIVE_GENERATOR_CONTEXTS if key not in _execution_locks), None)
        if key is None:
            break
        del _LIVE_GENERATOR_CONTEXTS[key]


def _get_generator_context(unique_id, scope=None):
    # The Regenerate request holds the execution lock before this lookup. That
    # lock protects disk access, but must not resurrect an already expired pipe.
    key = _generator_context_key(unique_id, scope)
    with _lock:
        previous = _LIVE_GENERATOR_CONTEXTS.get(key)
        if previous is not None and time.monotonic() - previous["updated_at"] > TTL_SECONDS:
            del _LIVE_GENERATOR_CONTEXTS[key]
        _prune_contexts()
        context = _LIVE_GENERATOR_CONTEXTS.get(key)
        if context is not None:
            context["updated_at"] = time.monotonic()
            _LIVE_GENERATOR_CONTEXTS.move_to_end(key)
        return context


def _remember_generator_context(unique_id, generator_type, cache_dir, pipe, scope=None):
    if not str(unique_id or "").strip() or pipe is None:
        return None
    key = _generator_context_key(unique_id, scope)
    with _lock:
        _LIVE_GENERATOR_CONTEXTS[key] = {
            "generator_type": generator_type,
            "cache_dir": cache_dir,
            "pipe": pipe,
            "updated_at": time.monotonic(),
        }
        _LIVE_GENERATOR_CONTEXTS.move_to_end(key)
        _prune_contexts()
        prune_workflow_caches(cache_dir, touch=True)
        return _LIVE_GENERATOR_CONTEXTS.get(key)
