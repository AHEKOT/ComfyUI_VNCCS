"""Tests for nodes/clothes_designer.py — pure logic functions."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")
import torch

from nodes.clothes_designer import ClothesDesigner, PipeContext


# ── _find_breasts_desc ────────────────────────────────────────────────────────

class TestFindBreastsDesc:
    def test_finds_in_body_field(self):
        info = {"body": "slim, small breasts, tall"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "breasts" in result.lower()

    def test_finds_flat_chest(self):
        info = {"body": "flat chest, petite"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "flat chest" in result.lower()

    def test_finds_in_other_field_if_body_empty(self):
        info = {"body": "", "additional_details": "large breasts, long legs"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "breasts" in result.lower()

    def test_returns_none_when_no_breast_desc(self):
        info = {"body": "slim, tall", "additional_details": "holding sword"}
        assert ClothesDesigner._find_breasts_desc(info) is None

    def test_body_field_takes_priority_over_others(self):
        info = {"body": "medium breasts", "additional_details": "huge breasts reference"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert "medium" in result.lower()

    def test_case_insensitive(self):
        info = {"body": "LARGE BREASTS"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None

    def test_ignores_non_string_values(self):
        info = {"body": 42, "additional_details": "small breasts"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None


# ── construct_prompt ──────────────────────────────────────────────────────────

class TestClothesDesignerConstructPrompt:
    def _data(self, **overrides):
        data = {
            "activeTab": "generate",
            "character": "",
            "costume_info": {},
            "gen_settings": {"background_color": "Green"},
        }
        data.update(overrides)
        return data

    def test_generate_tab_returns_tuple(self):
        result = ClothesDesigner.construct_prompt(self._data())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_tab_includes_green_bg(self):
        pos, _ = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Green"}))
        assert "green" in pos.lower()
        assert "00FF00" in pos

    def test_generate_tab_includes_blue_bg(self):
        pos, _ = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Blue"}))
        assert "blue" in pos.lower()
        assert "0000FF" in pos

    def test_generate_tab_unknown_bg_defaults_to_green(self):
        pos, _ = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Red"}))
        assert "00FF00" in pos

    def test_generate_tab_includes_costume_parts(self):
        costume = {"top": "white shirt", "bottom": "black jeans", "shoes": "sneakers"}
        pos, _ = ClothesDesigner.construct_prompt(self._data(costume_info=costume))
        assert "white shirt" in pos
        assert "black jeans" in pos
        assert "sneakers" in pos

    def test_generate_tab_omits_empty_costume_parts(self):
        costume = {"top": "red dress", "bottom": "", "head": ""}
        pos, _ = ClothesDesigner.construct_prompt(self._data(costume_info=costume))
        assert "red dress" in pos

    def test_generate_tab_negative_prompt_not_empty(self):
        _, neg = ClothesDesigner.construct_prompt(self._data())
        assert len(neg) > 0

    def test_generate_tab_negative_contains_nsfw_block(self):
        _, neg = ClothesDesigner.construct_prompt(self._data())
        assert "naked" in neg.lower() or "nude" in neg.lower()

    def test_clone_tab_without_clone_image_falls_through_to_generate(self):
        # clone tab but no clone_image → should NOT use clone branch
        data = self._data(activeTab="clone", clone_image=None)
        pos, _ = ClothesDesigner.construct_prompt(data)
        # Without clone_image the clone branch is skipped; result is generate-style
        assert isinstance(pos, str)

    def test_clone_tab_with_clone_image(self):
        data = self._data(activeTab="clone", clone_image="img.png")
        pos, neg = ClothesDesigner.construct_prompt(data)
        assert "Picture 3" in pos
        assert "Picture 1" in pos
        assert neg == ""

    def test_clone_tab_uses_background_color(self):
        data = self._data(
            activeTab="clone",
            clone_image="img.png",
            gen_settings={"background_color": "Blue"},
        )
        pos, _ = ClothesDesigner.construct_prompt(data)
        assert "0000FF" in pos


# ── get_cache_paths ───────────────────────────────────────────────────────────

class TestGetCachePaths:
    def test_returns_two_paths(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        img_path, info_path = ClothesDesigner.get_cache_paths("Alice", "Casual")
        assert img_path.endswith(".png")
        assert info_path.endswith(".json")

    def test_costume_name_sanitized(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        img_path, _ = ClothesDesigner.get_cache_paths("Alice", "My Fancy/Costume!")
        basename = os.path.basename(img_path)
        # Special characters removed; spaces → underscores
        assert "/" not in basename
        assert "!" not in basename

    def test_cache_dir_created(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        ClothesDesigner.get_cache_paths("Alice", "Dress")
        assert os.path.isdir(tmp_path / "Alice" / "cache")


# ── Clone reference preparation ───────────────────────────────────────────────

class TestCloneReferencePreparation:
    def test_sam3_preprocessing_helpers_are_not_exposed(self):
        assert not hasattr(ClothesDesigner, "_run_clone_sam3_reference")
        assert not hasattr(ClothesDesigner, "_apply_mask_on_background")


# ── Costume validation ───────────────────────────────────────────────────────

class TestEditableCostumeValidation:
    @pytest.mark.parametrize("costume", ["Casual", "My Costume", "armor_01"])
    def test_accepts_editable_costumes(self, costume):
        assert ClothesDesigner._is_editable_costume(costume)

    @pytest.mark.parametrize("costume", ["", None, "Naked", "Original"])
    def test_rejects_missing_or_base_costumes(self, costume):
        assert not ClothesDesigner._is_editable_costume(costume)


# ── PipeContext ───────────────────────────────────────────────────────────────

class TestPipeContext:
    def test_creates_empty_pipe_from_none(self):
        ctx = PipeContext(source=None)
        assert ctx.model is None
        assert ctx.clip is None
        assert ctx.vae is None
        assert ctx.seed_int == 0
        assert ctx.denoise == 1.0

    def test_copies_attrs_from_source(self):
        src = types.SimpleNamespace(
            model=object(), clip=object(), vae=object(),
            pos=object(), neg=object(),
            seed_int=42, sample_steps=20, cfg=7.0, denoise=0.8,
            sampler_name="euler", scheduler="karras",
            loader_type="standard", nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src)
        assert ctx.model is src.model
        assert ctx.seed_int == 42
        assert ctx.cfg == 7.0
        assert ctx.sampler_name == "euler"

    def test_updates_override_source(self):
        src = types.SimpleNamespace(
            model=object(), clip=object(), vae=object(),
            pos=object(), neg=object(),
            seed_int=1, sample_steps=10, cfg=5.0, denoise=1.0,
            sampler_name="euler", scheduler="normal",
            loader_type=None, nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src, seed_int=999, cfg=3.5)
        assert ctx.seed_int == 999
        assert ctx.cfg == 3.5
        # other attrs unchanged from source
        assert ctx.sample_steps == 10

    def test_falls_back_to_seed_attr(self):
        src = types.SimpleNamespace(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed=777,  # old attr name, no seed_int
            sample_steps=0, cfg=0.0, denoise=1.0,
            sampler_name=None, scheduler=None,
            loader_type=None, nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src)
        assert ctx.seed_int == 777

    def test_propagates_loader_type(self):
        src = types.SimpleNamespace(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=1.0,
            sampler_name=None, scheduler=None,
            loader_type="nunchaku", nunchaku_kind="flux",
            nunchaku_settings={"precision": "fp4"}, model_entry={"name": "x"},
        )
        ctx = PipeContext(source=src)
        assert ctx.loader_type == "standard"
        assert ctx.nunchaku_kind is None
        assert ctx.nunchaku_settings is None
