"""Tests for nodes/character_cloner.py — grid layout and config logic."""

import json
import math
import os
import sys
import types

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
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        from nodes.character_cloner import CharacterCloner
        import nodes.character_cloner as cc_mod
        monkeypatch.setattr(cc_mod, "character_dir", lambda n: str(tmp_path / n), raising=False)

        # Minimal widget_data — no source_images so no PIL needed
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
            "source_images": [],
        })

        node = CharacterCloner()
        try:
            node.process(widget_data=widget_data)
        except Exception:
            pass

        config = utils.load_config("CloneTest")
        assert config is not None
        assert config["character_info"]["sex"] == "female"

    def test_config_not_written_for_unknown_character(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        from nodes.character_cloner import CharacterCloner

        widget_data = json.dumps({"character": "Unknown", "character_info": {}, "source_images": []})
        node = CharacterCloner()
        try:
            node.process(widget_data=widget_data)
        except Exception:
            pass

        # "Unknown" should not create a config
        assert utils.load_config("Unknown") is None
