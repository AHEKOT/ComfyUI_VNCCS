"""
Unit tests for VNCCSBatchSaver — validates the fix for the emotion face
save cross-contamination (emotion explosion) bug.

These tests do NOT require a running ComfyUI instance or CUDA.
A minimal mock for torch.Tensor is injected before import since the
devcontainer has no GPU shared libraries.

Run with:
    python tests/test_batch_saver.py
"""

import importlib.util
import os
import re
import sys
import tempfile
import traceback
import types

# ---------------------------------------------------------------------------
# 1. Inject a minimal torch mock BEFORE anything imports torch
# ---------------------------------------------------------------------------
import numpy as np
from PIL import Image as PilImage


class _MockTensor:
    """Numpy-backed drop-in for the torch.Tensor subset used in batch_saver."""

    def __init__(self, array):
        self._a = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self._a.shape

    def dim(self):
        return self._a.ndim

    def unsqueeze(self, dim):
        return _MockTensor(np.expand_dims(self._a, axis=dim))

    def clamp(self, lo, hi):
        return _MockTensor(np.clip(self._a, lo, hi))

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __getitem__(self, idx):
        return _MockTensor(self._a[idx])

    def __repr__(self):
        return f"MockTensor(shape={self.shape})"


_torch_mock = types.ModuleType("torch")
_torch_mock.Tensor = _MockTensor


def _full(size, fill_value, dtype=None):
    return _MockTensor(np.full(size, fill_value, dtype=np.float32))


def _cat(tensors, dim=0):
    arrays = [t._a if isinstance(t, _MockTensor) else np.asarray(t) for t in tensors]
    return _MockTensor(np.concatenate(arrays, axis=dim))


def _zeros(*args, **kwargs):
    shape = args if len(args) > 1 else args[0]
    return _MockTensor(np.zeros(shape, dtype=np.float32))


_torch_mock.full = _full
_torch_mock.cat = _cat
_torch_mock.zeros = _zeros
sys.modules["torch"] = _torch_mock

# ---------------------------------------------------------------------------
# 2. Load batch_saver.py directly (bypasses nodes/__init__.py import chain)
# ---------------------------------------------------------------------------
_BATCH_SAVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "nodes", "batch_saver.py"
)
_spec = importlib.util.spec_from_file_location("batch_saver", _BATCH_SAVER_PATH)
_batch_saver_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch_saver_mod)

VNCCSBatchSaver = _batch_saver_mod.VNCCSBatchSaver
_find_next_counter = _batch_saver_mod._find_next_counter

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_tensor(B, H, W, fill: float = 0.5):
    """Return a float [B, H, W, C=3] mock tensor."""
    return _MockTensor(np.full((B, H, W, 3), fill, dtype=np.float32))


def _make_mask(B, H, W, fill: float = 1.0):
    """Return a float [B, H, W] mock mask tensor."""
    return _MockTensor(np.full((B, H, W), fill, dtype=np.float32))


def _list_pngs(directory: str):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".png"))


# ---------------------------------------------------------------------------
# Lightweight test runner
# ---------------------------------------------------------------------------
PASSED: list = []
FAILED: list = []


def test(fn):
    """Decorator: run fn(tmp_dir) and record pass/fail."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fn(tmp)
            PASSED.append(fn.__name__)
            print(f"  PASS  {fn.__name__}")
        except Exception:
            FAILED.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    return fn


# ===========================================================================
# _find_next_counter
# ===========================================================================

@test
def test_counter_empty_dir(tmp):
    assert _find_next_counter(tmp, "face_angry_") == 0


@test
def test_counter_nonexistent_dir(tmp):
    assert _find_next_counter(os.path.join(tmp, "nope"), "face_angry_") == 0


@test
def test_counter_after_existing(tmp):
    open(os.path.join(tmp, "face_angry_00000.png"), "w").close()
    open(os.path.join(tmp, "face_angry_00001.png"), "w").close()
    assert _find_next_counter(tmp, "face_angry_") == 2


@test
def test_counter_with_gaps(tmp):
    open(os.path.join(tmp, "face_angry_00000.png"), "w").close()
    open(os.path.join(tmp, "face_angry_00005.png"), "w").close()
    assert _find_next_counter(tmp, "face_angry_") == 6


@test
def test_counter_ignores_other_prefix(tmp):
    open(os.path.join(tmp, "face_smile_00000.png"), "w").close()
    assert _find_next_counter(tmp, "face_angry_") == 0


# ===========================================================================
# Node class attributes (critical for ComfyUI integration)
# ===========================================================================

@test
def test_input_is_list_attribute(tmp):
    """CRITICAL: INPUT_IS_LIST=True is the mechanism that fixes the bug."""
    assert getattr(VNCCSBatchSaver, "INPUT_IS_LIST", False) is True, (
        "INPUT_IS_LIST must be True — this bypasses ComfyUI's broadcasting logic"
    )


@test
def test_output_node_attribute(tmp):
    assert getattr(VNCCSBatchSaver, "OUTPUT_NODE", False) is True


@test
def test_return_types_empty(tmp):
    assert VNCCSBatchSaver.RETURN_TYPES == (), (
        f"RETURN_TYPES should be () (terminal node), got {VNCCSBatchSaver.RETURN_TYPES}"
    )


@test
def test_category_is_vnccs(tmp):
    assert VNCCSBatchSaver.CATEGORY == "VNCCS"


@test
def test_input_types_declared(tmp):
    types_ = VNCCSBatchSaver.INPUT_TYPES()
    assert "images" in types_["required"]
    assert "filename_prefix" in types_["required"]
    assert "masks" in types_.get("optional", {})


# ===========================================================================
# Registration in nodes/__init__.py (parse statically — no import)
# ===========================================================================

@test
def test_batch_saver_imported_in_init(tmp):
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nodes", "__init__.py"
    )
    with open(init_path) as f:
        src = f.read()
    assert "from .batch_saver import NODE_CLASS_MAPPINGS" in src, (
        "nodes/__init__.py does not import batch_saver NODE_CLASS_MAPPINGS"
    )
    assert "from .batch_saver import NODE_DISPLAY_NAME_MAPPINGS" in src, (
        "nodes/__init__.py does not import batch_saver NODE_DISPLAY_NAME_MAPPINGS"
    )
    assert "BATCH_SAVER_MAPPINGS" in src, (
        "BATCH_SAVER_MAPPINGS not merged into NODE_CLASS_MAPPINGS"
    )
    assert "BATCH_SAVER_DISPLAY_MAPPINGS" in src, (
        "BATCH_SAVER_DISPLAY_MAPPINGS not merged into NODE_DISPLAY_NAME_MAPPINGS"
    )


# ===========================================================================
# Core save_batch behaviour
# ===========================================================================

@test
def test_three_emotions_one_face_each(tmp):
    """
    PRIMARY REGRESSION TEST — the emotion explosion bug.
    3 emotions → each face must go to its OWN folder only.
    """
    saver = VNCCSBatchSaver()
    emotions = ["angry", "smile", "sad"]
    fills = (0.2, 0.5, 0.8)
    prefixes = [os.path.join(tmp, e, f"face_{e}_") for e in emotions]
    images = [_make_tensor(1, 64, 64, f) for f in fills]

    saver.save_batch(images=images, filename_prefix=prefixes)

    for emotion, fill in zip(emotions, fills):
        folder = os.path.join(tmp, emotion)
        files = _list_pngs(folder)
        assert len(files) == 1, (
            f"Expected 1 face in {emotion}/, got {len(files)}: {files}\n"
            f"(This would be > 1 if the broadcast bug were present)"
        )
        img = np.array(PilImage.open(os.path.join(folder, files[0])))
        expected = int(fill * 255)
        actual = int(img[0, 0, 0])
        assert abs(actual - expected) <= 1, (
            f"{emotion}: pixel {actual} != expected ~{expected} "
            f"(cross-contamination check failed)"
        )


@test
def test_batch_faces_per_emotion(tmp):
    """Each emotion can produce a batch of faces (e.g. 6 crops from a sheet)."""
    saver = VNCCSBatchSaver()
    prefixes = [os.path.join(tmp, e, f"face_{e}_") for e in ["angry", "smile"]]
    images = [_make_tensor(6, 32, 32, 0.3), _make_tensor(6, 32, 32, 0.7)]

    saver.save_batch(images=images, filename_prefix=prefixes)

    for emotion, expected_count in zip(["angry", "smile"], [6, 6]):
        folder = os.path.join(tmp, emotion)
        files = _list_pngs(folder)
        assert len(files) == expected_count, (
            f"Expected {expected_count} in {emotion}/, got {len(files)}"
        )


@test
def test_single_emotion(tmp):
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "neutral")
    prefix = os.path.join(folder, "face_neutral_")
    saver.save_batch(images=[_make_tensor(1, 32, 32, 0.5)], filename_prefix=[prefix])
    assert len(_list_pngs(folder)) == 1


@test
def test_counter_continues_after_existing_files(tmp):
    """New saves don't overwrite existing files — counter picks up where left off."""
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "happy")
    os.makedirs(folder)
    PilImage.new("RGB", (32, 32)).save(os.path.join(folder, "face_happy_00000.png"))
    PilImage.new("RGB", (32, 32)).save(os.path.join(folder, "face_happy_00001.png"))

    saver.save_batch(
        images=[_make_tensor(2, 32, 32, 0.4)],
        filename_prefix=[os.path.join(folder, "face_happy_")],
    )

    files = _list_pngs(folder)
    assert len(files) == 4, f"Expected 4 files, got {len(files)}"
    assert "face_happy_00002.png" in files
    assert "face_happy_00003.png" in files


@test
def test_rgba_save_with_mask(tmp):
    """Mask is used as alpha channel → saved file is RGBA."""
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "masked")
    prefix = os.path.join(folder, "face_masked_")
    img = _make_tensor(1, 32, 32, 0.6)
    mask = _make_mask(1, 32, 32, 0.9)  # ~229 alpha

    saver.save_batch(images=[img], filename_prefix=[prefix], masks=[mask])

    files = _list_pngs(folder)
    assert len(files) == 1
    saved = PilImage.open(os.path.join(folder, files[0]))
    assert saved.mode == "RGBA", f"Expected RGBA mode, got {saved.mode}"
    alpha = np.array(saved)[:, :, 3]
    assert abs(int(alpha[0, 0]) - 229) <= 1, f"Alpha {alpha[0,0]} != ~229"


@test
def test_rgb_save_without_mask(tmp):
    """Without mask → saved file is RGB (no spurious alpha channel)."""
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "rgb")
    prefix = os.path.join(folder, "face_rgb_")
    saver.save_batch(images=[_make_tensor(1, 32, 32, 0.5)], filename_prefix=[prefix])
    saved = PilImage.open(os.path.join(folder, _list_pngs(folder)[0]))
    assert saved.mode == "RGB", f"Expected RGB, got {saved.mode}"


@test
def test_directory_auto_created(tmp):
    saver = VNCCSBatchSaver()
    deep = os.path.join(tmp, "a", "b", "c")
    prefix = os.path.join(deep, "face_x_")
    saver.save_batch(images=[_make_tensor(1, 16, 16, 0.5)], filename_prefix=[prefix])
    assert os.path.isdir(deep)
    assert len(_list_pngs(deep)) == 1


@test
def test_scalar_inputs_coerced(tmp):
    """Plain tensor + plain string (not wrapped in lists) are handled."""
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "coerced")
    prefix = os.path.join(folder, "face_c_")
    saver.save_batch(
        images=_make_tensor(1, 16, 16, 0.5),   # not a list
        filename_prefix=prefix,                  # not a list
    )
    assert len(_list_pngs(folder)) == 1


@test
def test_zip_stops_at_shortest(tmp):
    """Extra images beyond the prefix count are silently ignored (zip semantics)."""
    saver = VNCCSBatchSaver()
    prefixes = [os.path.join(tmp, "x", "face_x_")]
    images = [_make_tensor(1, 16, 16, 0.5), _make_tensor(1, 16, 16, 0.9)]
    saver.save_batch(images=images, filename_prefix=prefixes)
    assert len(_list_pngs(os.path.join(tmp, "x"))) == 1


@test
def test_five_digit_counter_format(tmp):
    """Filenames use zero-padded 5-digit counters."""
    saver = VNCCSBatchSaver()
    folder = os.path.join(tmp, "fmt")
    prefix = os.path.join(folder, "face_fmt_")
    saver.save_batch(images=[_make_tensor(3, 16, 16, 0.5)], filename_prefix=[prefix])
    for fname in _list_pngs(folder):
        stem = fname.replace("face_fmt_", "").replace(".png", "")
        assert re.fullmatch(r"\d{5}", stem), f"Bad counter format in: {fname}"


@test
def test_pixel_conversion_accuracy(tmp):
    """Float 0→0, 0.5→127, 1→255 pixel conversion."""
    saver = VNCCSBatchSaver()
    for fill, expected in [(0.0, 0), (1.0, 255), (0.5, 127)]:
        folder = os.path.join(tmp, f"pix_{fill}")
        prefix = os.path.join(folder, "f_")
        saver.save_batch(images=[_make_tensor(1, 8, 8, fill)], filename_prefix=[prefix])
        files = _list_pngs(folder)
        assert files
        img = np.array(PilImage.open(os.path.join(folder, files[0])))
        actual = int(img[0, 0, 0])
        assert abs(actual - expected) <= 1, (
            f"fill={fill}: got {actual}, expected ~{expected}"
        )


# ===========================================================================
# Bug-vs-fix simulation — the definitive regression proof
# ===========================================================================

@test
def test_broadcast_bug_vs_fix(tmp):
    """
    Demonstrates the emotion explosion bug and confirms the fix avoids it.

    BUG (simulated): ComfyUI's slice_dict broadcasts images[0] (the full
    batch of ALL emotions' faces) to every path iteration → every folder
    receives all emotions' faces.

    FIX: VNCCSBatchSaver with INPUT_IS_LIST=True receives one tensor per
    emotion (the raw list from merge_result_data) and zips them with paths
    → each folder receives only its own emotion's faces.
    """
    # --- Bug simulation ---
    bug_dir = os.path.join(tmp, "bug")
    all_faces_tensor = _make_tensor(3, 32, 32, 0.5)  # 3 emotions concatenated
    paths = [os.path.join(bug_dir, e, f"face_{e}_") for e in ["angry", "smile", "sad"]]

    # Simulate slice_dict broadcast: all 3 paths get the identical batch
    for path in paths:
        folder = os.path.dirname(path)
        base = os.path.basename(path)
        os.makedirs(folder, exist_ok=True)
        for i in range(all_faces_tensor.shape[0]):
            frame = (
                np.clip(all_faces_tensor[i].numpy(), 0, 1) * 255
            ).astype(np.uint8)
            PilImage.fromarray(frame).save(
                os.path.join(folder, f"{base}{i:05d}.png")
            )

    for e in ["angry", "smile", "sad"]:
        files = _list_pngs(os.path.join(bug_dir, e))
        assert len(files) == 3, (
            f"Bug simulation: {e}/ should have 3 contaminated files, "
            f"got {len(files)}"
        )

    # --- Fix: VNCCSBatchSaver with per-emotion tensors ---
    fix_dir = os.path.join(tmp, "fix")
    saver = VNCCSBatchSaver()
    paths = [os.path.join(fix_dir, e, f"face_{e}_") for e in ["angry", "smile", "sad"]]
    # merge_result_data wraps N non-LIST outputs as [tensor1, tensor2, tensor3]
    per_emotion_tensors = [_make_tensor(1, 32, 32, f) for f in (0.2, 0.5, 0.8)]

    saver.save_batch(images=per_emotion_tensors, filename_prefix=paths)

    for e, fill in zip(["angry", "smile", "sad"], (0.2, 0.5, 0.8)):
        folder = os.path.join(fix_dir, e)
        files = _list_pngs(folder)
        assert len(files) == 1, (
            f"Fix: {e}/ should have exactly 1 face, got {len(files)}"
        )
        img = np.array(PilImage.open(os.path.join(folder, files[0])))
        expected = int(fill * 255)
        actual = int(img[0, 0, 0])
        assert abs(actual - expected) <= 1, (
            f"Fix pixel check {e}: got {actual}, expected ~{expected} "
            f"(cross-contamination if wrong emotion)"
        )


# ===========================================================================
# Summary
# ===========================================================================

def main():
    total = len(PASSED) + len(FAILED)
    print(f"\n{'='*60}")
    print(f"Results: {len(PASSED)}/{total} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed tests:")
        for name in FAILED:
            print(f"  - {name}")
    return len(FAILED) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
