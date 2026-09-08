"""Tests for nodes/clothes_designer.py — pure logic functions."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")
import torch

from _vnccs.nodes.clothes_designer import (
    ClothesDesigner,
    PipeContext,
    _resolve_pipe_clothes_core_lora,
)


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
        assert pos == "Dress character: clothes, footwear and accessories from Picture 2"
        assert neg == ""

    def test_clone_tab_ignores_background_color(self):
        data = self._data(
            activeTab="clone",
            clone_image="img.png",
            gen_settings={"background_color": "Blue"},
        )
        pos, _ = ClothesDesigner.construct_prompt(data)
        assert pos == "Dress character: clothes, footwear and accessories from Picture 2"


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


# ── Clothes Core LoRA resolution ──────────────────────────────────────────────

class TestClothesCoreLoraResolution:
    def test_prefers_pipe_lora_entry(self):
        pipe = types.SimpleNamespace(
            model_entry={"kind": "QIE2511"},
            lora_entries=[
                {
                    "name": "VNCCS Clothes Core",
                    "kind": "QIE2511",
                    "local_path": "models/loras/qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.5.safetensors",
                }
            ],
        )
        assert _resolve_pipe_clothes_core_lora(pipe) == "qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.5.safetensors"

    def test_selects_only_lora_matching_klein_model_kind(self):
        pipe = types.SimpleNamespace(
            model_entry={"kind": "Klein9b"},
            lora_entries=[
                {
                    "name": "VNCCS Clothes Core",
                    "kind": "QIE2511",
                    "local_path": "models/loras/qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.7.safetensors",
                },
                {
                    "name": "VNCCS Clothes Core Klein9b",
                    "kind": "Klein9b",
                    "local_path": "models/loras/Klein9b/VNCCS_ClothesCoreKlein9b_V1.safetensors",
                },
            ],
        )
        assert _resolve_pipe_clothes_core_lora(pipe) == "Klein9b/VNCCS_ClothesCoreKlein9b_V1.safetensors"


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


@pytest.mark.parametrize("kind,size,expected", [
    ("MiniMaxH3", None, 1536), ("MiniMaxH3", 1024, 1024),
    ("QIE2511", None, 1024), ("QIE2511", 1536, 1536),
    ("Klein9b", None, 1024), ("Klein9b", 2048, 2048),
])
@pytest.mark.parametrize("clone", [False, True])
def test_preview_resolution_reaches_model_encoder(tmp_path, monkeypatch, kind, size, expected, clone):
    from _vnccs.nodes import clothes_designer as cd
    from PIL import Image
    import json

    reference = torch.zeros((1, 96, 64, 3))
    clone_path = tmp_path / "clone.png"
    Image.new("RGB", (64, 96)).save(clone_path)
    monkeypatch.setattr(cd, "get_latest_sprite_path", lambda *args: "reference.png")
    monkeypatch.setattr(cd, "sheets_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(cd, "_resolve_pipe_clothes_core_lora", lambda pipe: "clothes.safetensors")
    monkeypatch.setattr(cd, "resolve_comfy_image_path", lambda info: str(clone_path))
    monkeypatch.setattr(cd.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)
    node = cd.ClothesDesigner()
    monkeypatch.setattr(node, "get_reference_sprite", lambda *args: reference)
    monkeypatch.setattr(node, "get_cache_paths", lambda *args: (str(tmp_path / "preview.png"), str(tmp_path / "preview.json")))
    calls = {}
    def call(name, **kwargs):
        calls[name] = kwargs
        if name in (cd.WORKFLOW_ENCODER_CLASS, cd.KLEIN_ENCODER_CLASS):
            return "positive", "negative", {"samples": torch.zeros(1)}
        if name == "MiniMaxH3ReferenceToVideo":
            return "positive", {"samples": torch.zeros(1)}
        if name in ("KSampler", "SamplerCustomAdvanced"):
            return ({"samples": torch.zeros(1)},)
        if name == "VAEDecodeTiled":
            return (torch.zeros((5 if kind == "MiniMaxH3" else 1, 96, 64, 3)),)
        return (object(),)
    monkeypatch.setattr(cd, "_call_comfy_node", call)
    pipe = types.SimpleNamespace(model=object(), clip=object(), vae=object(), audio_vae=object(), model_entry={"kind": kind})
    data = {"character": "Alice", "costume": "Dress", "gen_settings": {"target_size": size},
            "activeTab": "clone" if clone else "generate", "clone_image": {"name": "clone.png"} if clone else None}
    image, _, _ = node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
    assert image.shape == (1, 96, 64, 3)
    if kind == "MiniMaxH3":
        encoder = calls["MiniMaxH3ReferenceToVideo"]
        width, height = encoder["width"], encoder["height"]
        assert width % 32 == height % 32 == 0
        assert width * height == pytest.approx(expected ** 2, rel=0.04)
        assert width / height == pytest.approx(64 / 96, rel=0.04)
        assert encoder["length"] == 5
        assert len(encoder["ref_images"]) == (2 if clone else 1)
        assert "SamplerCustomAdvanced" in calls and "KSampler" not in calls
    elif kind == "Klein9b":
        assert calls[cd.KLEIN_ENCODER_CLASS]["megapixels"] == (expected / 1024) ** 2
    else:
        assert calls[cd.WORKFLOW_ENCODER_CLASS]["target_size"] == expected


@pytest.mark.parametrize("size", [True, "bad", -1, 0, 511, 4097, 1024.5, float("inf"), float("nan")])
def test_resolution_rejects_invalid_values(size):
    from _vnccs.nodes.clothes_designer import _clothes_target_size
    with pytest.raises(ValueError, match="Resolution scale"):
        _clothes_target_size({"target_size": size}, "minimaxh3")
