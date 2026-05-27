#!/usr/bin/env python3
"""
Standalone verification tests for ComfyUI_VNCCS bug fixes.
Run from repo root: python3 tests/verify_vnccs_fixes.py
No ComfyUI installation required.
"""
import sys
import os

PASS = 0
FAIL = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"PASS  {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"FAIL  {name}: {reason}")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

def read_file(rel_path):
    return open(os.path.join(REPO_ROOT, rel_path)).read()

# Bug #3 + threading — _PREVIEW_LOCK must be threading.Lock, not asyncio.Lock
def test_preview_lock_is_threading():
    src = read_file("nodes/character_creator_v2.py")
    if "_PREVIEW_LOCK = threading.Lock()" in src:
        ok("Bug #3: _PREVIEW_LOCK is threading.Lock")
    else:
        fail("Bug #3: _PREVIEW_LOCK is threading.Lock", "Not found or wrong type")

def test_preview_lock_used_in_sync():
    src = read_file("nodes/character_creator_v2.py")
    if "with _PREVIEW_LOCK:" in src:
        ok("Bug #3: _PREVIEW_LOCK used (with statement)")
    else:
        fail("Bug #3: _PREVIEW_LOCK used", "'with _PREVIEW_LOCK:' not found")

def test_no_async_with_preview_lock():
    src = read_file("nodes/character_creator_v2.py")
    if "async with _PREVIEW_LOCK" not in src:
        ok("Bug #3: no async with _PREVIEW_LOCK (threading.Lock correct)")
    else:
        fail("Bug #3: no async with _PREVIEW_LOCK", "Found 'async with _PREVIEW_LOCK' — should be threading.Lock used in sync context")

# Bug #1 — preview_generate uses run_in_executor
def test_run_in_executor_in_preview():
    src = read_file("nodes/character_creator_v2.py")
    if "run_in_executor" in src and "_preview_generate_sync" in src:
        ok("Bug #1: preview_generate uses run_in_executor + sync helper")
    else:
        fail("Bug #1: preview_generate uses run_in_executor", "run_in_executor or _preview_generate_sync not found")

# Bug #2 — cc_nunchaku_apply_fix uses run_in_executor
def test_nunchaku_fix_uses_executor():
    src = read_file("nodes/vnccs_control_center.py")
    if "run_in_executor" in src:
        ok("Bug #2: vnccs_control_center uses run_in_executor")
    else:
        fail("Bug #2: vnccs_control_center uses run_in_executor", "run_in_executor not found")

# Bug #4 — _NODE_CACHE in process()
def test_node_cache_exists():
    src = read_file("nodes/character_creator_v2.py")
    if "_NODE_CACHE" in src and "_NODE_CACHE_LOCK" in src:
        ok("Bug #4: _NODE_CACHE and _NODE_CACHE_LOCK defined")
    else:
        fail("Bug #4: _NODE_CACHE defined", "_NODE_CACHE or _NODE_CACHE_LOCK not found")

# Bug #6 — no bare except:
def test_no_bare_except():
    src = read_file("nodes/character_creator_v2.py")
    lines = src.splitlines()
    bare = [f"line {i+1}" for i, l in enumerate(lines)
            if l.strip() == "except:" or (l.strip().startswith("except:") and len(l.strip()) < 10)]
    if not bare:
        ok("Bug #6: no bare except: in character_creator_v2.py")
    else:
        fail("Bug #6: no bare except:", f"Found: {bare}")

# Bug #8 — no top-level import cv2
def test_cv2_not_toplevel():
    src = read_file("nodes/vnccs_utils.py")
    lines = src.splitlines()
    # Check module-level lines (before any class/def)
    for i, l in enumerate(lines[:40]):
        stripped = l.strip()
        if stripped == "import cv2" and not l.startswith(" ") and not l.startswith("\t"):
            fail("Bug #8: cv2 not top-level", f"Found top-level 'import cv2' at line {i+1}")
            return
    ok("Bug #8: no top-level import cv2")

def test_import_cv2_lazy_helper():
    src = read_file("nodes/vnccs_utils.py")
    if "_import_cv2" in src:
        ok("Bug #8: _import_cv2 lazy helper defined")
    else:
        fail("Bug #8: _import_cv2 lazy helper", "_import_cv2 not found in vnccs_utils.py")

if __name__ == "__main__":
    test_preview_lock_is_threading()
    test_preview_lock_used_in_sync()
    test_no_async_with_preview_lock()
    test_run_in_executor_in_preview()
    test_nunchaku_fix_uses_executor()
    test_node_cache_exists()
    test_no_bare_except()
    test_cv2_not_toplevel()
    test_import_cv2_lazy_helper()

    print(f"\n{PASS}/{PASS+FAIL} pass")
    sys.exit(0 if FAIL == 0 else 1)
