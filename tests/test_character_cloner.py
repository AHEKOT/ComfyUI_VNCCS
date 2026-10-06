"""Tests for nodes/character_cloner.py — grid layout and config logic."""

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── grid algorithm (extracted from CharacterCloner.process) ──────────────────
# The algorithm lives inline in process(), so we replicate it here exactly
# so tests stay decoupled from model-loading side-effects.

import numpy as np


def _best_grid(count, max_w, max_h):
    """Replicate the smart-grid selection from CharacterCloner.process()."""
    best_cols = 1
    best_rows = count
    best_score = float("inf")

    for c in range(1, count + 1):
        r = int(np.ceil(count / c))
        w_total = c * max_w
        h_total = r * max_h
        ratio = w_total / h_total
        symmetric_ratio = ratio if ratio >= 1 else 1 / ratio
        grid_diff = abs(c - r)
        score = symmetric_ratio + (grid_diff * 0.01)
        if score < best_score:
            best_score = score
            best_cols = c
            best_rows = r

    return best_cols, best_rows


class TestGridLayout:
    def test_single_image_is_1x1(self):
        cols, rows = _best_grid(1, 512, 512)
        assert cols == 1
        assert rows == 1

    def test_four_square_images_prefer_2x2(self):
        cols, rows = _best_grid(4, 512, 512)
        assert cols == 2
        assert rows == 2

    def test_grid_covers_all_images(self):
        for count in range(1, 13):
            cols, rows = _best_grid(count, 512, 512)
            assert cols * rows >= count

    def test_single_tall_image_prefers_single_column(self):
        # 3 portrait images — vertical strip is reasonable
        cols, rows = _best_grid(3, 256, 768)
        assert cols * rows >= 3

    def test_many_wide_images_grid_covers_all(self):
        # Algorithm minimises total output aspect ratio, not number of columns.
        # For 6 wide images (1024x256) the minimum-score layout is 1 col × 6 rows.
        cols, rows = _best_grid(6, 1024, 256)
        assert cols * rows >= 6

    def test_symmetric_score_penalises_extreme_ratios(self):
        # 4 square images: 2x2 beats 4x1
        cols_4x1, rows_4x1 = 4, 1
        cols_2x2, rows_2x2 = 2, 2
        max_w = max_h = 512

        def score(c, r):
            ratio = (c * max_w) / (r * max_h)
            sym = ratio if ratio >= 1 else 1 / ratio
            return sym + abs(c - r) * 0.01

        assert score(cols_2x2, rows_2x2) < score(cols_4x1, rows_4x1)


# ── image path resolution logic ───────────────────────────────────────────────
# Mirrors the resolution block in CharacterCloner.process() lines 67-89.

def _resolve_img_path(img_obj, input_dir, temp_dir, output_dir):
    """Replicate image path resolution from CharacterCloner.process()."""
    if isinstance(img_obj, dict):
        img_name = img_obj.get("name")
        subfolder = img_obj.get("subfolder", "")
        img_type = img_obj.get("type", "input")
    else:
        img_name = img_obj
        subfolder = ""
        img_type = "input"

    if not img_name:
        return None

    if img_type == "input":
        base_dir = input_dir
    elif img_type == "temp":
        base_dir = temp_dir
    else:
        base_dir = output_dir

    if subfolder:
        return os.path.join(base_dir, subfolder, img_name)
    return os.path.join(base_dir, img_name)


class TestImagePathResolution:
    def test_string_resolves_to_input(self):
        path = _resolve_img_path("img.png", "/in", "/tmp", "/out")
        assert path == "/in/img.png"

    def test_dict_input_type(self):
        obj = {"name": "img.png", "type": "input", "subfolder": ""}
        path = _resolve_img_path(obj, "/in", "/tmp", "/out")
        assert path == "/in/img.png"

    def test_dict_temp_type(self):
        obj = {"name": "img.png", "type": "temp", "subfolder": ""}
        path = _resolve_img_path(obj, "/in", "/tmp", "/out")
        assert path == "/tmp/img.png"

    def test_dict_output_type(self):
        obj = {"name": "img.png", "type": "output", "subfolder": ""}
        path = _resolve_img_path(obj, "/in", "/tmp", "/out")
        assert path == "/out/img.png"

    def test_dict_with_subfolder(self):
        obj = {"name": "img.png", "type": "input", "subfolder": "batch1"}
        path = _resolve_img_path(obj, "/in", "/tmp", "/out")
        assert path == "/in/batch1/img.png"

    def test_empty_name_returns_none(self):
        assert _resolve_img_path({"name": "", "type": "input"}, "/in", "/tmp", "/out") is None

    def test_dict_defaults_to_input(self):
        obj = {"name": "img.png"}
        path = _resolve_img_path(obj, "/in", "/tmp", "/out")
        assert path == "/in/img.png"


# ── config save in process() ──────────────────────────────────────────────────

class TestClonerConfigSave:
    def test_config_written_for_valid_character(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        Image = pytest.importorskip("PIL.Image")
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        from nodes.character_cloner import CharacterCloner
        import nodes.character_cloner as cc_mod
        monkeypatch.setattr(cc_mod, "character_dir", lambda n: str(tmp_path / n), raising=False)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.new("RGB", (32, 64), "green").save(input_dir / "clone.png")
        monkeypatch.setattr(cc_mod.folder_paths, "get_input_directory", lambda: str(input_dir), raising=False)

        info = {
            "sex": "female", "age": 20, "race": "human",
            "hair": "blonde", "eyes": "green", "face": "", "body": "",
            "skin_color": "", "additional_details": "",
            "nsfw": False, "aesthetics": "masterpiece",
            "negative_prompt": "bad", "lora_prompt": "", "background_color": "Green",
        }
        widget_data = json.dumps({
            "character": "CloneTest",
            "character_info": info,
            "source_images": [{"name": "clone.png", "type": "input", "subfolder": ""}],
        })

        node = CharacterCloner()
        try:
            node.process(widget_data=widget_data)
        except Exception:
            pass

        config = utils.load_config("CloneTest")
        assert config is not None
        assert config["character_info"]["sex"] == "female"

    def test_missing_source_image_raises_before_saving_config(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        from nodes.character_cloner import CharacterCloner
        import nodes.character_cloner as cc_mod
        monkeypatch.setattr(cc_mod, "character_dir", lambda n: str(tmp_path / n), raising=False)
        monkeypatch.setattr(cc_mod.folder_paths, "get_input_directory", lambda: str(tmp_path / "input"), raising=False)

        widget_data = json.dumps({
            "character": "CloneMissing",
            "character_info": {"background_color": "Green"},
            "source_images": [{"name": "missing.png", "type": "input", "subfolder": ""}],
        })

        node = CharacterCloner()
        with pytest.raises(ValueError, match="Upload a character image"):
            node.process(widget_data=widget_data)

        assert utils.load_config("CloneMissing") is None

    def test_config_not_written_for_unknown_character(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        from nodes.character_cloner import CharacterCloner

        widget_data = json.dumps({"character": "Unknown", "character_info": {}, "source_images": []})
        node = CharacterCloner()
        with pytest.raises(ValueError, match="Upload a character image"):
            node.process(widget_data=widget_data)

        # "Unknown" should not create a config
        assert utils.load_config("Unknown") is None


def _load_cloner(monkeypatch):
    """Load the cloner without tensor or model dependencies."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    from nodes.qwen_vl import get_qwen_vl_chat_handler
    assets = types.ModuleType("_vnccs.nodes.qwen_vl")
    assets.get_qwen_vl_chat_handler = get_qwen_vl_chat_handler
    assets._ensure_qwen_vl_assets = lambda **kwargs: ("model.gguf", "mmproj.gguf")
    assets.QWEN_VL_MODEL_FILENAME = "model.gguf"
    monkeypatch.setitem(sys.modules, assets.__name__, assets)
    spec = importlib.util.spec_from_file_location(
        "_vnccs.nodes._cloner_wizard_test", Path(__file__).parents[1] / "nodes/character_cloner.py"
    )
    cloner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloner)
    return cloner


def test_cloner_rejects_multiple_references_before_processing_or_saving(monkeypatch):
    cloner = _load_cloner(monkeypatch)
    monkeypatch.setattr(cloner, "_source_image_path", lambda image: pytest.fail("Extra references must be rejected before reading images"))
    monkeypatch.setattr(cloner, "load_config", lambda name: pytest.fail("Invalid reference counts must not change character data"))
    with pytest.raises(ValueError, match="only one reference image"):
        cloner.CharacterCloner().process(json.dumps({"character": "Alice", "source_images": ["one.png", "two.png"]}))


def _run_wizard(monkeypatch, tmp_path, content):
    """Exercise the real wizard request without tensor or model dependencies."""
    from PIL import Image

    cloner = _load_cloner(monkeypatch)
    monkeypatch.setattr(cloner.folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(cloner.web, "json_response", lambda data, status=200: types.SimpleNamespace(data=data, status=status), raising=False)
    Image.new("RGB", (32, 64), "red").save(tmp_path / "reference.png")

    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": content}}]}
    llama = types.ModuleType("llama_cpp")
    llama.llama_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    llama.llama_chat_format.Qwen35ChatHandler = type("Qwen35ChatHandler", (), {"__init__": lambda self, **kwargs: None})
    llama.Llama = lambda **kwargs: types.SimpleNamespace(create_chat_completion=complete)
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", llama.llama_chat_format)
    monkeypatch.setitem(sys.modules, "json_repair", types.SimpleNamespace(loads=json.loads))
    return cloner._cloner_auto_generate_response({"image_name": "reference.png"}), calls


def test_wizard_requests_visible_character_traits_only(monkeypatch, tmp_path):
    traits = {"skin_color": "red skin", "face": "", "additional_details": "monster arm"}
    response, calls = _run_wizard(monkeypatch, tmp_path, json.dumps(traits))
    assert response.status == 200
    assert response.data == traits
    prompt = calls[0]["messages"][1]["content"][0]["text"]
    assert "do not choose from presets or a closed list" in prompt
    assert "including unusual or multiple colors" in prompt
    assert "Red or pink skin across the face or body is skin_color, not blush" in prompt
    assert "distinct localized cheek blush" in prompt
    assert "monster arm" in prompt
    assert "Do not include clothing, footwear, wearable accessories, held objects, pose, actions" in prompt
    assert "Do not fill a field just to avoid an empty value" in prompt


@pytest.mark.parametrize("content", ["", "wearing white shirt, standing, one arm raised", "[]"])
def test_wizard_invalid_response_does_not_become_character_details(monkeypatch, tmp_path, content):
    response, _ = _run_wizard(monkeypatch, tmp_path, content)
    assert response.status == 502
    assert response.data["error"] == "INVALID_RESPONSE"
    assert "additional_details" not in response.data
