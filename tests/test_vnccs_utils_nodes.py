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


def test_removed_background_model_is_not_exposed():
    removed_model = "BE" + "N2"
    assert removed_model not in vnccs_utils.AVAILABLE_MODELS
    assert removed_model not in VNCCS_RMBG2().models


def test_qwen_download_disables_hub_credentials(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\0" * (1024 * 1024))
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(model_path)

    monkeypatch.setattr(vnccs_utils, "hf_hub_download", fake_download)

    result = vnccs_utils._download_qwen_vl_file(
        "public/repository",
        "model.gguf",
        str(tmp_path),
        revision="pinned-revision",
    )

    assert result == str(model_path)
    assert captured == {
        "repo_id": "public/repository",
        "filename": "model.gguf",
        "revision": "pinned-revision",
        "local_dir": str(tmp_path),
        "token": False,
    }


def test_registered_node_result_unwraps_comfy_node_output():
    NodeOutput = type("NodeOutput", (), {})
    output = NodeOutput()
    output.result = ({"processor": object()},)
    output.args = ()

    assert _unwrap_node_result(output) is output.result[0]


def test_chroma_key_defaults_match_balanced_profile_and_expose_sam3_checkbox():
    required = VNCCSChromaKey.INPUT_TYPES()["required"]
    expected_defaults = {
        "tolerance": 0.15,
        "softness": 0.12,
        "despill_strength": 0.65,
        "edge_width": 3,
        "matte_cleanup": 0.10,
        "foreground_recover": 0.35,
        "edge_decontaminate": 0.75,
        "edge_choke": 0.08,
        "matte_method": "guided_edge",
        "output_mode": "straight_rgba",
    }

    actual_defaults = {
        name: required[name][1]["default"]
        for name in expected_defaults
    }

    assert actual_defaults == expected_defaults
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
        torch.ones((1, 2, 1, 6, 8), dtype=torch.float32),
        torch.ones((1, 1, 2, 1, 6, 8, 1), dtype=torch.float32),
        torch.ones((6, 8, 2), dtype=torch.float32),
        torch.ones((2, 8, 6), dtype=torch.float32),
        torch.ones((1, 2, 1, 3, 4), dtype=torch.float32),
        torch.ones((2, 3, 4, 1), dtype=torch.float32),
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


def test_sam3_recovery_preserves_candidates_across_arbitrary_wrapper_axes():
    node = VNCCSChromaKey()
    first = torch.full((6, 8), 0.25, dtype=torch.float32)
    second = torch.full((6, 8), 0.75, dtype=torch.float32)
    raw_masks = torch.stack((first, second), dim=0).reshape(1, 2, 1, 6, 8, 1)

    candidates = node._sam3_recovery_candidates_from_result(
        (torch.ones((1, 6, 8)), None, raw_masks, [], []),
        target_hw=(6, 8),
    )

    assert candidates.shape == (2, 6, 8)
    assert candidates[:, 0, 0].tolist() == pytest.approx([0.25, 0.75])


def test_sam3_recovery_falls_back_to_combined_mask_for_uninterpretable_candidates(capsys):
    node = VNCCSChromaKey()
    combined = torch.full((1, 6, 8), 0.6, dtype=torch.float32)
    invalid_candidates = torch.ones((7,), dtype=torch.float32)

    candidates = node._sam3_recovery_candidates_from_result(
        (combined, None, invalid_candidates, [], []),
        target_hw=(6, 8),
        stage="SAM3 fallback test",
    )

    assert candidates.shape == (1, 6, 8)
    assert candidates[0, 0, 0].item() == pytest.approx(0.6)
    assert "individual mask shape (7,)" in capsys.readouterr().out


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


def test_chroma_key_clears_border_connected_shifted_screen_color():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.31, 0.77, 0.56], dtype=torch.float32)
    shifted_screen = torch.tensor([0.49, 0.75, 0.66], dtype=torch.float32)
    foreground = torch.tensor([0.90, 0.25, 0.35], dtype=torch.float32)
    image = key_color.expand(1, 40, 40, 3).clone()
    image[:, :, 18:22, :] = shifted_screen
    image[:, 14:26, 14:26, :] = foreground

    _, low_tolerance_matte, _ = node.chroma_key(
        image,
        0.0,
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

    assert rgba.shape == (1, 40, 40, 4)
    assert matte.shape == (1, 40, 40)
    assert debug.shape == (1, 40, 40, 3)
    assert low_tolerance_matte[0, 5, 19].item() > 0.5
    assert matte[0, 5, 19].item() == pytest.approx(0.0)
    assert matte[0, 20, 20].item() > 0.95


def test_connected_screen_cleanup_requires_chroma_and_rgb_similarity():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.31, 0.77, 0.56], dtype=torch.float32)
    image = key_color.expand(16, 16, 3).clone()
    pale_foreground = torch.tensor([1.0, 0.80, 0.80], dtype=torch.float32)
    image[:, 7:9, :] = pale_foreground
    alpha = torch.full((16, 16), 0.4, dtype=torch.float32)
    alpha[:, 7:9] = 1.0

    cleaned = node._suppress_connected_key_fringe(
        image=image,
        alpha=alpha,
        key_color=key_color,
        tolerance=0.20,
        softness=0.16,
        amount=1.0,
    )

    assert cleaned[2, 2].item() == pytest.approx(0.0)
    assert cleaned[2, 7].item() == pytest.approx(1.0)


def test_connected_screen_cleanup_preserves_confident_same_hue_foreground():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.23, 0.44, 0.37], dtype=torch.float32)
    image = key_color.expand(24, 24, 3).clone()
    dark_same_hue_foreground = key_color * 0.45
    image[:, 10:14] = dark_same_hue_foreground
    alpha = torch.zeros((24, 24), dtype=torch.float32)
    alpha[:, 10:14] = 1.0

    cleaned = node._suppress_connected_key_fringe(
        image=image,
        alpha=alpha,
        key_color=key_color,
        tolerance=0.15,
        softness=0.16,
        amount=1.0,
    )

    assert cleaned[12, 12].item() == pytest.approx(1.0)


def test_screen_cleanup_handles_dark_border_and_enclosed_background():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.30, 0.70, 0.60], dtype=torch.float32)
    image = key_color.expand(32, 32, 3).clone()
    alpha = torch.full((32, 32), 0.4, dtype=torch.float32)

    dark_border = key_color * 0.3
    image[:, -1] = dark_border
    alpha[:, -1] = 0.5

    foreground = torch.tensor([0.90, 0.20, 0.30], dtype=torch.float32)
    image[9:23, 9:23] = foreground
    alpha[9:23, 9:23] = 1.0
    enclosed_screen = torch.tensor([0.38, 0.68, 0.61], dtype=torch.float32)
    image[12:20, 12:20] = enclosed_screen
    alpha[12:20, 12:20] = 0.2

    cleaned = node._suppress_connected_key_fringe(
        image=image,
        alpha=alpha,
        key_color=key_color,
        tolerance=0.20,
        softness=0.16,
        amount=1.0,
    )

    assert cleaned[16, -1].item() == pytest.approx(0.0)
    assert cleaned[16, 16].item() == pytest.approx(0.0)
    assert cleaned[10, 10].item() == pytest.approx(1.0)


def test_despill_strength_controls_edge_decontamination():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.05, 0.95, 0.10], dtype=torch.float32)
    image = key_color.expand(32, 32, 3).clone()
    image[8:24, 8:24] = torch.tensor([0.80, 0.15, 0.20])
    image[8:24, 8] = torch.tensor([0.25, 0.75, 0.20])

    outputs = []
    for despill_strength in (0.0, 1.0):
        rgba, _, _ = node._process_single(
            image,
            tolerance=0.15,
            softness=0.16,
            despill_strength=despill_strength,
            edge_width=3,
            matte_cleanup=0.20,
            foreground_recover=0.0,
            edge_decontaminate=0.70,
            edge_choke=0.0,
            matte_method="guided_edge",
            screen_mode="green",
            output_mode="straight_rgba",
        )
        outputs.append(rgba)

    no_despill, full_despill = outputs
    assert full_despill[12, 8, 1].item() < no_despill[12, 8, 1].item()
    assert not torch.allclose(no_despill[..., :3], full_despill[..., :3])


def test_edge_color_bleed_removes_hidden_key_color_without_changing_alpha():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.2, 0.8, 0.3], dtype=torch.float32)
    foreground = torch.tensor([0.9, 0.2, 0.3], dtype=torch.float32)
    image = key_color.expand(16, 16, 3).clone()
    image[6:10, 6:10] = foreground
    alpha = torch.zeros((16, 16), dtype=torch.float32)
    alpha[6:10, 6:10] = 1.0
    alpha[5:11, 5:11] = torch.maximum(alpha[5:11, 5:11], torch.full((6, 6), 0.5))
    edge = torch.zeros_like(alpha)
    edge[4:12, 4:12] = 1.0
    original_alpha = alpha.clone()

    cleaned = node._bleed_clean_edge_colors(
        image=image,
        alpha=alpha,
        edge=edge,
        key_color=key_color,
        dominant_idx=1,
        other_indices=[0, 2],
        radius=3,
        amount=1.0,
    )

    assert torch.equal(alpha, original_alpha)
    assert cleaned[5, 7, 1].item() < image[5, 7, 1].item()
    assert torch.allclose(cleaned[4, 7], foreground)
    assert torch.allclose(cleaned[0, 0], key_color)


@pytest.mark.parametrize(
    ("key_rgb", "foreground_rgb", "screen_mix"),
    [
        ([0.23, 0.44, 0.37], [0.05, 0.05, 0.50], 0.10),  # Teal screen into blue foreground.
        ([0.23, 0.44, 0.37], [0.05, 0.05, 0.50], 0.50),
        ([0.08, 0.15, 0.95], [0.75, 0.08, 0.10], 0.10),  # Blue screen into red foreground.
        ([0.08, 0.15, 0.95], [0.75, 0.08, 0.10], 0.50),
        ([0.95, 0.12, 0.08], [0.08, 0.10, 0.75], 0.10),  # Red screen into blue foreground.
        ([0.95, 0.12, 0.08], [0.08, 0.10, 0.75], 0.50),
    ],
)
def test_edge_color_bleed_removes_full_rgb_key_contamination(key_rgb, foreground_rgb, screen_mix):
    node = VNCCSChromaKey()
    key_color = torch.tensor(key_rgb, dtype=torch.float32)
    foreground = torch.tensor(foreground_rgb, dtype=torch.float32)
    contaminated = foreground * (1.0 - screen_mix) + key_color * screen_mix
    image = key_color.expand(16, 16, 3).clone()
    image[5:11, 5:11] = contaminated
    image[6:10, 6:10] = foreground
    alpha = torch.zeros((16, 16), dtype=torch.float32)
    alpha[5:11, 5:11] = 0.5
    alpha[6:10, 6:10] = 1.0
    edge = torch.zeros_like(alpha)
    edge[4:12, 4:12] = 1.0
    dominant_idx = int(torch.argmax(key_color).item())
    other_indices = [index for index in range(3) if index != dominant_idx]

    cleaned = node._bleed_clean_edge_colors(
        image=image,
        alpha=alpha,
        edge=edge,
        key_color=key_color,
        dominant_idx=dominant_idx,
        other_indices=other_indices,
        radius=3,
        amount=1.0,
    )

    before = torch.linalg.vector_norm(image[5, 7] - foreground)
    after = torch.linalg.vector_norm(cleaned[5, 7] - foreground)
    assert after.item() < before.item() * 0.1
    assert torch.allclose(cleaned[5, 7], foreground, atol=1e-4)


def test_edge_color_bleed_does_not_trust_high_alpha_fringe_as_foreground():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.10, 0.70, 0.35], dtype=torch.float32)
    foreground = torch.tensor([0.08, 0.12, 0.72], dtype=torch.float32)
    contaminated = foreground * 0.55 + key_color * 0.45
    image = key_color.expand(32, 32, 3).clone()
    image[6:26, 6:26] = foreground
    image[6:26, 6] = contaminated
    image[6:26, 25] = contaminated

    alpha = torch.zeros((32, 32), dtype=torch.float32)
    alpha[6:26, 6:26] = 1.0
    # These pixels are visually blended even though guided refinement made
    # their matte nearly opaque and placed them outside the hard edge band.
    alpha[10:22, 6] = 0.985
    edge = torch.zeros_like(alpha)
    original_alpha = alpha.clone()

    cleaned = node._bleed_clean_edge_colors(
        image=image,
        alpha=alpha,
        edge=edge,
        key_color=key_color,
        dominant_idx=1,
        other_indices=[0, 2],
        radius=5,
        amount=1.0,
    )

    assert torch.equal(alpha, original_alpha)
    assert torch.allclose(cleaned[14, 6], foreground, atol=1e-4)


def test_chroma_key_falls_back_when_sam3_nodes_are_unavailable(monkeypatch):
    node = VNCCSChromaKey()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    expected = (
        torch.zeros((4, 4, 4), dtype=torch.float32),
        torch.zeros((4, 4), dtype=torch.float32),
        torch.zeros((4, 4, 3), dtype=torch.float32),
    )

    monkeypatch.setattr(
        node,
        "_chroma_key_with_sam3_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Required node 'easy sam3ModelLoader' is not available")),
    )
    monkeypatch.setattr(vnccs_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(node, "_process_single", lambda *_args, **_kwargs: expected)

    rgba, matte, debug = node.chroma_key(
        image=image,
        tolerance=0.15,
        softness=0.12,
        despill_strength=0.65,
        edge_width=3,
        matte_cleanup=0.1,
        foreground_recover=0.35,
        edge_decontaminate=0.75,
        edge_choke=0.08,
        matte_method="guided_edge",
        screen_mode="auto",
        output_mode="straight_rgba",
        use_sam3_recovery_mask=True,
    )

    assert rgba.shape == (1, 4, 4, 4)
    assert matte.shape == (1, 4, 4)
    assert debug.shape == (1, 4, 4, 3)


def test_chroma_key_never_calls_sam3_recovery_on_macos(monkeypatch):
    node = VNCCSChromaKey()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    expected = (
        torch.zeros((4, 4, 4), dtype=torch.float32),
        torch.zeros((4, 4), dtype=torch.float32),
        torch.zeros((4, 4, 3), dtype=torch.float32),
    )

    monkeypatch.setattr(vnccs_utils.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        node,
        "_chroma_key_with_sam3_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("SAM3 must be skipped on macOS")),
    )
    monkeypatch.setattr(node, "_process_single", lambda *_args, **_kwargs: expected)

    rgba, matte, debug = node.chroma_key(
        image=image,
        tolerance=0.15,
        softness=0.12,
        despill_strength=0.65,
        edge_width=3,
        matte_cleanup=0.1,
        foreground_recover=0.35,
        edge_decontaminate=0.75,
        edge_choke=0.08,
        matte_method="guided_edge",
        screen_mode="auto",
        output_mode="straight_rgba",
        use_sam3_recovery_mask=True,
    )

    assert rgba.shape == (1, 4, 4, 4)
    assert matte.shape == (1, 4, 4)
    assert debug.shape == (1, 4, 4, 3)


@pytest.mark.parametrize("error", [ImportError("triton unavailable"), ValueError("unsupported device")])
def test_chroma_key_falls_back_for_any_optional_sam3_failure(monkeypatch, error):
    node = VNCCSChromaKey()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    expected = (
        torch.zeros((4, 4, 4), dtype=torch.float32),
        torch.zeros((4, 4), dtype=torch.float32),
        torch.zeros((4, 4, 3), dtype=torch.float32),
    )

    monkeypatch.setattr(vnccs_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        node,
        "_chroma_key_with_sam3_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(node, "_process_single", lambda *_args, **_kwargs: expected)

    result = node.chroma_key(
        image=image,
        tolerance=0.15,
        softness=0.12,
        despill_strength=0.65,
        edge_width=3,
        matte_cleanup=0.1,
        foreground_recover=0.35,
        edge_decontaminate=0.75,
        edge_choke=0.08,
        matte_method="guided_edge",
        screen_mode="auto",
        output_mode="straight_rgba",
        use_sam3_recovery_mask=True,
    )

    assert tuple(tensor.shape for tensor in result) == ((1, 4, 4, 4), (1, 4, 4), (1, 4, 4, 3))


def test_edge_color_bleed_uses_interior_anchor_for_opaque_boundary_spill():
    node = VNCCSChromaKey()
    key_color = torch.tensor([0.10, 0.70, 0.35], dtype=torch.float32)
    foreground = torch.tensor([0.08, 0.12, 0.72], dtype=torch.float32)
    contaminated = foreground * 0.55 + key_color * 0.45
    image = key_color.expand(32, 32, 3).clone()
    image[6:26, 6:26] = foreground
    image[6:26, 6] = contaminated

    alpha = torch.zeros((32, 32), dtype=torch.float32)
    alpha[6:26, 6:26] = 1.0
    edge = torch.zeros_like(alpha)
    edge[6:26, 6] = 1.0

    cleaned = node._bleed_clean_edge_colors(
        image=image,
        alpha=alpha,
        edge=edge,
        key_color=key_color,
        dominant_idx=1,
        other_indices=[0, 2],
        radius=5,
        amount=1.0,
    )

    assert torch.allclose(cleaned[14, 6], foreground, atol=1e-4)


class TestClothesTemplates:
    def test_registered_with_display_name(self):
        assert NODE_CLASS_MAPPINGS["VNCCS_ClothesTemplates"] is VNCCS_ClothesTemplates
        assert NODE_DISPLAY_NAME_MAPPINGS["VNCCS_ClothesTemplates"] == "VNCCS Clothes Templates"

    def test_aesthetic_choices_include_all_and_json_values(self):
        choices = VNCCS_ClothesTemplates.INPUT_TYPES()["required"]["aesthetic"][0]
        assert choices[0] == "ALL"
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
