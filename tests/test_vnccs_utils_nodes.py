"""Tests for nodes/vnccs_utils.py — image helpers and processing nodes."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nodes.vnccs_utils as vnccs_utils
from nodes.vnccs_utils import (
    tensor2pil, pil2tensor, _ensure_float01,
    _unwrap_node_result,
    VNCCS_ClothesTemplates,
    VNCCS_VLAnalyzer,
    _build_vl_analyzer_prompt,
    VNCCS_ColorFix,
    VNCCSChromaKey,
    VNCCS_Resize,
    VNCCS_MaskExtractor,
    VNCCS_RMBG2,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


# ── tensor2pil ────────────────────────────────────────────────────────────────

def test_rmbg2_return_type_tolerates_legacy_high_slot_validation():
    assert VNCCS_RMBG2.RETURN_TYPES[5] == "IMAGE"


def test_registered_node_result_unwraps_comfy_node_output():
    NodeOutput = type("NodeOutput", (), {})
    output = NodeOutput()
    output.result = ({"processor": object()},)
    output.args = ()

    assert _unwrap_node_result(output) is output.result[0]


def test_chroma_key_exposes_sam3_recovery_checkbox():
    required = VNCCSChromaKey.INPUT_TYPES()["required"]
    assert required["use_sam3_recovery_mask"][0] == "BOOLEAN"
    assert required["use_sam3_recovery_mask"][1]["default"] is False


def test_chroma_key_disabled_sam3_recovery_does_not_call_recovery(monkeypatch):
    node = VNCCSChromaKey()

    def fail_recovery(*args, **kwargs):
        raise AssertionError("SAM3 recovery path should not run when disabled")

    monkeypatch.setattr(node, "_chroma_key_with_sam3_recovery", fail_recovery)
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    rgba, matte, debug = node.chroma_key(
        image,
        0.2,
        0.16,
        0.5,
        3,
        0.2,
        0.35,
        0.7,
        0.2,
        "guided_edge",
        "auto",
        "straight_rgba",
        False,
    )

    assert rgba.shape == (1, 8, 8, 4)
    assert matte.shape == (1, 8, 8)
    assert debug.shape == (1, 8, 8, 3)


def test_sam3_recovery_restores_only_shrunk_mask_area():
    node = VNCCSChromaKey()
    original = torch.zeros((12, 12, 3), dtype=torch.float32)
    original[..., 0] = 1.0
    rgba = torch.zeros((12, 12, 4), dtype=torch.float32)
    alpha = torch.zeros((12, 12), dtype=torch.float32)
    debug = torch.zeros((12, 12, 3), dtype=torch.float32)
    recovery_mask = torch.zeros((12, 12), dtype=torch.float32)
    recovery_mask[1:11, 1:11] = 1.0

    restored_rgba, restored_alpha, _ = node._restore_recovery_details(
        original=original,
        rgba=rgba,
        alpha=alpha,
        debug=debug,
        recovery_mask=recovery_mask,
        output_mode="straight_rgba",
    )

    assert restored_alpha[5, 5].item() == pytest.approx(1.0)
    assert restored_rgba[5, 5, 0].item() == pytest.approx(1.0)
    assert restored_alpha[2, 2].item() == pytest.approx(0.0)
    assert restored_rgba[2, 2, 0].item() == pytest.approx(0.0)


def test_sam3_recovery_rejects_background_objects_without_clipping_kept_masks():
    node = VNCCSChromaKey()
    alpha = torch.zeros((40, 40), dtype=torch.float32)
    alpha[10:30, 10:30] = 1.0

    foreground_object = torch.zeros((40, 40), dtype=torch.float32)
    foreground_object[8:32, 8:32] = 1.0
    background_object = torch.zeros((40, 40), dtype=torch.float32)
    background_object[0:8, :] = 1.0
    whole_image_false_positive = torch.ones((40, 40), dtype=torch.float32)

    selected = node._select_sam3_recovery_mask(
        torch.stack(
            [
                foreground_object,
                background_object,
                whole_image_false_positive,
            ]
        ),
        alpha,
    )

    assert selected[20, 20].item() == pytest.approx(1.0)
    assert selected[8, 20].item() == pytest.approx(1.0)
    assert selected[2, 20].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "raw_masks",
    [
        torch.ones((2, 1, 6, 8), dtype=torch.float32),
        torch.ones((2, 6, 8, 1), dtype=torch.float32),
        torch.ones((1, 2, 6, 8), dtype=torch.float32),
    ],
)
def test_sam3_recovery_normalizes_individual_object_mask_layouts(raw_masks):
    node = VNCCSChromaKey()
    combined = torch.ones((1, 6, 8), dtype=torch.float32)

    candidates = node._sam3_recovery_candidates_from_result(
        (combined, None, raw_masks, [], []),
        target_hw=(6, 8),
    )

    assert candidates.shape == (2, 6, 8)


def test_sam3_recovery_supports_legacy_combined_mask_output():
    node = VNCCSChromaKey()
    combined = torch.ones((1, 6, 8), dtype=torch.float32)

    candidates = node._sam3_recovery_candidates_from_result(
        combined,
        target_hw=(6, 8),
    )

    assert candidates.shape == (1, 6, 8)


def test_sam3_recovery_segments_batch_one_image_at_a_time(monkeypatch):
    node = VNCCSChromaKey()
    image = torch.zeros((3, 6, 8, 3), dtype=torch.float32)
    model = object()
    loader_calls = []
    segment_calls = []

    monkeypatch.setattr(vnccs_utils, "_ensure_sam3_model_available", lambda: "sam3.safetensors")

    def fake_call(class_names, method_names=None, **kwargs):
        if class_names[0] == "LoadSam3Model":
            loader_calls.append(kwargs["model"])
            return model

        assert kwargs["sam3_model"] is model
        if kwargs["images"].shape[0] != 1:
            raise RuntimeError(
                "stack expects each tensor to be equal size, "
                "but got [3, 4] at entry 0 and [6, 4] at entry 2"
            )
        assert kwargs["images"].shape == (1, 6, 8, 3)
        segment_calls.append(kwargs["keep_model_loaded"])
        mask_value = len(segment_calls) / 10.0
        combined = torch.full((1, 6, 8), mask_value, dtype=torch.float32)
        segmented_image = torch.zeros((1, 6, 8, 4), dtype=torch.float32)
        object_masks = torch.stack(
            [
                torch.full((6, 8), mask_value, dtype=torch.float32),
                torch.full((6, 8), mask_value + 0.05, dtype=torch.float32),
            ]
        )
        return combined, segmented_image, object_masks, [], []

    monkeypatch.setattr(vnccs_utils, "_call_registered_node", fake_call)

    candidates = node._run_sam3_recovery_masks(image, target_hw=(6, 8))

    assert len(candidates) == 3
    assert all(mask.shape == (2, 6, 8) for mask in candidates)
    assert [candidates[index][0, 0, 0].item() for index in range(3)] == pytest.approx([0.1, 0.2, 0.3])
    assert loader_calls == ["sam3.safetensors"]
    assert segment_calls == [True, True, False]


def test_sam3_recovery_forwards_generator_settings(monkeypatch):
    node = VNCCSChromaKey()
    image = torch.zeros((1, 6, 8, 3), dtype=torch.float32)
    model = object()
    seen = {}

    def fake_call(class_names, method_names=None, **kwargs):
        if class_names[0] == "LoadSam3Model":
            seen["loader"] = kwargs
            return model
        seen["segment"] = kwargs
        combined = torch.ones((1, 6, 8), dtype=torch.float32)
        return combined, None, combined, [], []

    monkeypatch.setattr(vnccs_utils, "_call_registered_node", fake_call)
    candidates = node._run_sam3_recovery_masks(
        image,
        target_hw=(6, 8),
        settings={
            "sam3_model": "custom-sam3.safetensors",
            "sam3_segmentor": "image",
            "sam3_device": "cpu",
            "sam3_precision": "fp32",
            "sam3_prompt": "face, hair",
            "sam3_threshold": 0.62,
            "sam3_add_background": "black",
            "sam3_detection_limit": 7,
        },
    )

    assert candidates[0].shape == (1, 6, 8)
    assert seen["loader"] == {
        "model": "custom-sam3.safetensors",
        "segmentor": "image",
        "device": "cpu",
        "precision": "fp32",
    }
    assert seen["segment"]["prompt"] == "face, hair"
    assert seen["segment"]["threshold"] == pytest.approx(0.62)
    assert seen["segment"]["add_background"] == "black"
    assert seen["segment"]["detection_limit"] == 7


def test_sam3_recovery_filter_and_erode_are_configurable():
    node = VNCCSChromaKey()
    alpha = torch.zeros((12, 12), dtype=torch.float32)
    alpha[4:8, 4:8] = 1.0
    candidate = torch.zeros((12, 12), dtype=torch.float32)
    candidate[3:9, 3:9] = 1.0

    rejected = node._select_sam3_recovery_mask(
        candidate.unsqueeze(0),
        alpha,
        {"sam3_min_foreground_overlap": 0.9},
    )
    accepted = node._select_sam3_recovery_mask(
        candidate.unsqueeze(0),
        alpha,
        {"sam3_min_foreground_overlap": 0.4},
    )

    assert rejected.max().item() == pytest.approx(0.0)
    assert accepted.max().item() == pytest.approx(1.0)

    original = torch.ones((12, 12, 3), dtype=torch.float32)
    rgba = torch.zeros((12, 12, 4), dtype=torch.float32)
    debug = torch.zeros((12, 12, 3), dtype=torch.float32)
    no_erode_rgba, no_erode_alpha, _ = node._restore_recovery_details(
        original,
        rgba,
        torch.zeros_like(alpha),
        debug,
        accepted,
        "straight_rgba",
        erode_radius=0,
    )

    assert no_erode_alpha[3, 3].item() == pytest.approx(1.0)
    assert no_erode_rgba[3, 3, 0].item() == pytest.approx(1.0)


def test_connected_key_fringe_suppression_is_not_used_by_chroma_key(monkeypatch):
    node = VNCCSChromaKey()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("connected fringe suppression must not run in base chroma key")

    monkeypatch.setattr(node, "_suppress_connected_key_fringe", fail_if_called)
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    image[..., 1] = 1.0

    rgba, matte, debug = node.chroma_key(
        image,
        0.2,
        0.16,
        0.5,
        3,
        0.2,
        0.35,
        0.7,
        0.2,
        "guided_edge",
        "green",
        "straight_rgba",
        False,
    )

    assert rgba.shape == (1, 8, 8, 4)
    assert matte.shape == (1, 8, 8)
    assert debug.shape == (1, 8, 8, 3)


class TestClothesTemplates:
    def test_registered_with_display_name(self):
        assert NODE_CLASS_MAPPINGS["VNCCS_ClothesTemplates"] is VNCCS_ClothesTemplates
        assert NODE_DISPLAY_NAME_MAPPINGS["VNCCS_ClothesTemplates"] == "VNCCS Clothes Templates"

    def test_aesthetic_choices_include_all_and_json_values(self):
        choices = VNCCS_ClothesTemplates.INPUT_TYPES()["required"]["aesthetic"][0]
        assert choices[0] == "ВСЕ"
        assert "Techwear" in choices

    def test_random_template_filters_by_aesthetic_and_explicit(self, monkeypatch):
        sample = [
            {"aesthetic": "Techwear", "content": "techwear, jacket", "is_explicit": True},
            {"aesthetic": "Casual", "content": "hoodie, jeans", "is_explicit": False},
        ]
        monkeypatch.setattr(VNCCS_ClothesTemplates, "_load_outfits", classmethod(lambda cls: sample))

        result, = VNCCS_ClothesTemplates().random_template("Casual", False)

        assert result == "hoodie, jeans"

    def test_random_template_errors_when_no_match(self, monkeypatch):
        sample = [{"aesthetic": "Techwear", "content": "techwear, jacket", "is_explicit": True}]
        monkeypatch.setattr(VNCCS_ClothesTemplates, "_load_outfits", classmethod(lambda cls: sample))

        with pytest.raises(RuntimeError, match="no outfits found"):
            VNCCS_ClothesTemplates().random_template("Techwear", False)


class TestVLAnalyzer:
    def test_registered_with_display_name(self):
        assert NODE_CLASS_MAPPINGS["VNCCS_VLAnalyzer"] is VNCCS_VLAnalyzer
        assert NODE_DISPLAY_NAME_MAPPINGS["VNCCS_VLAnalyzer"] == "VNCCS VL analyzer"

    def test_input_contract(self):
        required = VNCCS_VLAnalyzer.INPUT_TYPES()["required"]
        assert set(required.keys()) == {"image", "clothing_tags"}
        assert VNCCS_VLAnalyzer.RETURN_TYPES == ("STRING",)
        assert VNCCS_VLAnalyzer.RETURN_NAMES == ("description",)

    def test_prompt_uses_clothing_tags_as_mandatory_hints(self):
        prompt = _build_vl_analyzer_prompt("techwear, black_jacket, thighhighs")
        assert "Clothing tags that must be used as mandatory hints" in prompt
        assert "techwear, black_jacket, thighhighs" in prompt
        assert "Do not output raw comma-separated tags" in prompt


class TestTensor2Pil:
    def test_returns_pil_image(self):
        t = torch.rand(32, 32, 3)
        result = tensor2pil(t)
        assert isinstance(result, Image.Image)

    def test_values_scaled_to_0_255(self):
        t = torch.ones(4, 4, 3)  # all 1.0
        result = tensor2pil(t)
        arr = np.array(result)
        assert arr.max() == 255

    def test_zero_tensor_gives_black(self):
        t = torch.zeros(4, 4, 3)
        result = tensor2pil(t)
        arr = np.array(result)
        assert arr.max() == 0

    def test_clips_above_1(self):
        t = torch.full((4, 4, 3), 2.0)
        result = tensor2pil(t)
        arr = np.array(result)
        assert arr.max() == 255


# ── pil2tensor ────────────────────────────────────────────────────────────────

class TestPil2Tensor:
    def test_returns_tensor(self):
        img = Image.new("RGB", (8, 8), (128, 64, 32))
        result = pil2tensor(img)
        assert isinstance(result, torch.Tensor)

    def test_has_batch_dim(self):
        img = Image.new("RGB", (8, 8))
        result = pil2tensor(img)
        assert result.shape[0] == 1

    def test_normalized_to_0_1(self):
        img = Image.new("RGB", (4, 4), (255, 255, 255))
        result = pil2tensor(img)
        assert result.max().item() <= 1.0
        assert result.min().item() >= 0.0

    def test_shape_hwc(self):
        img = Image.new("RGB", (16, 8))
        result = pil2tensor(img)
        assert result.shape == (1, 8, 16, 3)

    def test_roundtrip_close(self):
        img = Image.new("RGB", (4, 4), (100, 150, 200))
        t = pil2tensor(img)
        back = tensor2pil(t[0])
        arr = np.array(back)
        assert np.allclose(arr[0, 0], [100, 150, 200], atol=1)


# ── _ensure_float01 ───────────────────────────────────────────────────────────

class TestEnsureFloat01:
    def test_uint8_normalized(self):
        t = torch.tensor([0, 128, 255], dtype=torch.uint8)
        result = _ensure_float01(t)
        assert torch.is_floating_point(result)
        assert result.max().item() <= 1.0

    def test_float_above_1_normalized(self):
        t = torch.tensor([0.0, 128.0, 255.0])
        result = _ensure_float01(t)
        assert result.max().item() <= 1.0

    def test_float_already_01_unchanged(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        result = _ensure_float01(t)
        assert torch.allclose(result, t)

    def test_clamps_above_1(self):
        t = torch.tensor([0.5, 1.5, 2.0])
        result = _ensure_float01(t)
        assert result.max().item() == 1.0

    def test_clamps_below_0(self):
        t = torch.tensor([-0.5, 0.5])
        result = _ensure_float01(t)
        assert result.min().item() == 0.0


# ── VNCCS_ColorFix ────────────────────────────────────────────────────────────

class TestColorFix:
    def _rgb(self, h=8, w=8):
        return torch.rand(h, w, 3)

    def test_neutral_params_identity(self):
        rgb = self._rgb()
        result = VNCCS_ColorFix()._apply_to_rgb(rgb, contrast=1.0, saturation=1.0)
        # Should be very close to input
        assert torch.allclose(result, rgb, atol=1e-4)

    def test_zero_saturation_makes_grayscale(self):
        rgb = torch.rand(4, 4, 3)
        result = VNCCS_ColorFix()._apply_to_rgb(rgb, contrast=1.0, saturation=0.0)
        # All channels should be equal (grayscale)
        assert torch.allclose(result[:, :, 0], result[:, :, 1], atol=1e-4)
        assert torch.allclose(result[:, :, 1], result[:, :, 2], atol=1e-4)

    def test_output_clamped_to_01(self):
        rgb = torch.rand(4, 4, 3)
        result = VNCCS_ColorFix()._apply_to_rgb(rgb, contrast=5.0, saturation=3.0)
        assert result.max().item() <= 1.0
        assert result.min().item() >= 0.0

    def test_high_contrast_increases_variance(self):
        rgb = torch.rand(8, 8, 3)
        low = VNCCS_ColorFix()._apply_to_rgb(rgb.clone(), contrast=0.5, saturation=1.0)
        high = VNCCS_ColorFix()._apply_to_rgb(rgb.clone(), contrast=2.0, saturation=1.0)
        assert high.var().item() >= low.var().item()

    def test_color_fix_preserves_alpha(self):
        image = torch.rand(1, 8, 8, 4)
        result, = VNCCS_ColorFix().color_fix(image, contrast=1.0, saturation=1.0)
        assert result.shape[-1] == 4
        assert torch.allclose(result[:, :, :, 3], image[:, :, :, 3], atol=1e-4)

    def test_color_fix_rgb_image(self):
        image = torch.rand(1, 8, 8, 3)
        result, = VNCCS_ColorFix().color_fix(image, contrast=1.2, saturation=0.8)
        assert result.shape == image.shape


# ── VNCCS_Resize ──────────────────────────────────────────────────────────────

class TestResize:
    def test_resize_smaller(self):
        img = torch.rand(64, 64, 3)
        result = VNCCS_Resize()._resize_single(img, 32, 32, "bilinear")
        assert result.shape == (32, 32, 3)

    def test_resize_larger(self):
        img = torch.rand(16, 16, 3)
        result = VNCCS_Resize()._resize_single(img, 64, 64, "bilinear")
        assert result.shape == (64, 64, 3)

    def test_resize_preserves_alpha(self):
        img = torch.rand(32, 32, 4)
        result = VNCCS_Resize()._resize_single(img, 16, 16, "bilinear")
        assert result.shape[2] == 4

    def test_resize_non_square(self):
        img = torch.rand(32, 64, 3)
        result = VNCCS_Resize()._resize_single(img, 16, 48, "bilinear")
        assert result.shape == (48, 16, 3)

    def test_lanczos_method(self):
        img = torch.rand(32, 32, 3)
        result = VNCCS_Resize()._resize_single(img, 16, 16, "lanczos")
        assert result.shape == (16, 16, 3)


# ── VNCCS_MaskExtractor ───────────────────────────────────────────────────────

class TestMaskExtractor:
    def test_rgba_extracts_rgb(self):
        image = torch.rand(1, 8, 8, 4)
        result, = VNCCS_MaskExtractor().fill_alpha_with_color(image)
        assert result.shape[-1] == 3

    def test_rgb_passes_through(self):
        image = torch.rand(1, 8, 8, 3)
        result, = VNCCS_MaskExtractor().fill_alpha_with_color(image)
        assert result.shape[-1] == 3
