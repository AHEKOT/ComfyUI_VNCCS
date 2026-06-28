"""Tests for nodes/vnccs_api.py."""

import base64
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes.vnccs_api import VNCCS_Variable, VNCCS_ImageList, VNCCS_LoraBypass, VNCCS_LoraStack


# ── VNCCS_Variable ────────────────────────────────────────────────────────────

class TestVNCCSVariable:
    def test_int_string(self):
        assert VNCCS_Variable().execute("42") == (42,)
        assert isinstance(VNCCS_Variable().execute("42")[0], int)

    def test_negative_int(self):
        assert VNCCS_Variable().execute("-10") == (-10,)

    def test_float_string(self):
        result = VNCCS_Variable().execute("3.14")
        assert result == (3.14,)
        assert isinstance(result[0], float)

    def test_plain_string(self):
        assert VNCCS_Variable().execute("hello") == ("hello",)

    def test_whitespace_stripped(self):
        assert VNCCS_Variable().execute("  100  ") == (100,)

    def test_empty_string_returned_as_string(self):
        result = VNCCS_Variable().execute("")
        assert result == ("",)

    def test_scientific_notation_becomes_float(self):
        result = VNCCS_Variable().execute("1e5")
        assert isinstance(result[0], float)

    def test_zero_string(self):
        assert VNCCS_Variable().execute("0") == (0,)

    def test_float_zero(self):
        assert VNCCS_Variable().execute("0.0") == (0.0,)


# ── VNCCS_ImageList ───────────────────────────────────────────────────────────

def _make_b64_png(w=4, h=4):
    """Create a tiny valid PNG as base64."""
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_data_uri(w=4, h=4):
    return f"data:image/png;base64,{_make_b64_png(w, h)}"


class TestVNCCSImageList:
    def test_decodes_base64_image(self):
        torch = pytest.importorskip("torch")
        b64 = _make_b64_png()
        import json
        result, = VNCCS_ImageList().execute(json.dumps([b64]))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape[-1] == 3  # RGB

    def test_decodes_data_uri(self):
        torch = pytest.importorskip("torch")
        import json
        uri = _make_data_uri()
        result, = VNCCS_ImageList().execute(json.dumps([uri]))
        assert len(result) == 1

    def test_multiple_images(self):
        torch = pytest.importorskip("torch")
        import json
        images = [_make_b64_png(), _make_b64_png(8, 8)]
        result, = VNCCS_ImageList().execute(json.dumps(images))
        assert len(result) == 2

    def test_empty_array_raises(self):
        import json
        with pytest.raises(ValueError, match="no images"):
            VNCCS_ImageList().execute(json.dumps([]))

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            VNCCS_ImageList().execute("NOT JSON")

    def test_non_array_raises(self):
        import json
        with pytest.raises(ValueError, match="must be a JSON array"):
            VNCCS_ImageList().execute(json.dumps({"key": "val"}))

    def test_output_tensor_has_batch_dim(self):
        torch = pytest.importorskip("torch")
        import json
        result, = VNCCS_ImageList().execute(json.dumps([_make_b64_png()]))
        assert result[0].shape[0] == 1  # batch dim

    def test_output_values_normalized_01(self):
        torch = pytest.importorskip("torch")
        import json
        result, = VNCCS_ImageList().execute(json.dumps([_make_b64_png()]))
        assert result[0].max() <= 1.0
        assert result[0].min() >= 0.0


# ── VNCCS_LoraBypass — bypass paths ──────────────────────────────────────────

class TestVNCCSLoraBypassBypass:
    def test_none_bypasses(self):
        m, c = object(), object()
        result = VNCCS_LoraBypass().execute(m, c, "None", 1.0, 1.0)
        assert result == (m, c)

    def test_empty_string_bypasses(self):
        m, c = object(), object()
        result = VNCCS_LoraBypass().execute(m, c, "", 1.0, 1.0)
        assert result == (m, c)

    def test_whitespace_bypasses(self):
        m, c = object(), object()
        result = VNCCS_LoraBypass().execute(m, c, "  none  ", 1.0, 1.0)
        assert result == (m, c)

    def test_case_insensitive_bypass(self):
        m, c = object(), object()
        result = VNCCS_LoraBypass().execute(m, c, "NONE", 1.0, 1.0)
        assert result == (m, c)


# ── VNCCS_LoraStack — skip/parse logic ───────────────────────────────────────

class TestVNCCSLoraStack:
    def test_empty_array_passthrough(self):
        import json
        m, c = object(), object()
        result = VNCCS_LoraStack().execute(m, c, "[]")
        assert result == (m, c)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            VNCCS_LoraStack().execute(object(), object(), "NOT JSON")

    def test_non_array_raises(self):
        import json
        with pytest.raises(ValueError, match="must be a JSON array"):
            VNCCS_LoraStack().execute(object(), object(), json.dumps({"a": 1}))

    def test_slot_with_none_name_skipped(self):
        import json
        m, c = object(), object()
        slots = [{"name": "None", "strength": 1.0}]
        result = VNCCS_LoraStack().execute(m, c, json.dumps(slots))
        assert result == (m, c)

    def test_slot_with_empty_name_skipped(self):
        import json
        m, c = object(), object()
        slots = [{"name": "", "strength": 1.0}]
        result = VNCCS_LoraStack().execute(m, c, json.dumps(slots))
        assert result == (m, c)

    def test_zero_strength_slot_skipped(self):
        import json
        m, c = object(), object()
        slots = [{"name": "lora.safetensors", "strength_model": 0, "strength_clip": 0}]
        result = VNCCS_LoraStack().execute(m, c, json.dumps(slots))
        assert result == (m, c)

    def test_non_dict_slot_raises(self):
        import json
        with pytest.raises(ValueError, match="must be an object"):
            VNCCS_LoraStack().execute(object(), object(), json.dumps(["not_a_dict"]))
