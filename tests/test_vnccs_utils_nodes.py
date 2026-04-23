"""Tests for nodes/vnccs_utils.py — image helpers and processing nodes."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.vnccs_utils import (
    tensor2pil, pil2tensor, _ensure_float01,
    VNCCS_QuadSplitter as _QS,  # also tested in sheet_manager; skip here
    VNCCS_ColorFix,
    VNCCS_Resize,
    VNCCS_MaskExtractor,
)


# ── tensor2pil ────────────────────────────────────────────────────────────────

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
