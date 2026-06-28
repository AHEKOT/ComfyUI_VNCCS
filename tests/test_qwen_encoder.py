import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

torch = pytest.importorskip("torch")

from nodes.vnccs_qwen_encoder import VNCCS_QWEN_Encoder


def test_encoder_flattens_rgba_on_white_by_default():
    image = torch.tensor([[[[0.2, 0.4, 0.6, 0.5]]]], dtype=torch.float32)

    result = VNCCS_QWEN_Encoder()._prepare_encoder_image(image)

    assert result.shape == (1, 1, 1, 3)
    assert torch.allclose(result[0, 0, 0], torch.tensor([0.6, 0.7, 0.8]))


def test_encoder_flattens_rgba_on_named_background():
    image = torch.tensor([[[[0.2, 0.4, 0.6, 0.5]]]], dtype=torch.float32)

    result = VNCCS_QWEN_Encoder()._prepare_encoder_image(image, "Green")

    assert torch.allclose(result[0, 0, 0], torch.tensor([0.1, 0.7, 0.3]))


def test_encoder_flattens_rgba_on_hex_background():
    image = torch.tensor([[[[0.2, 0.4, 0.6, 0.5]]]], dtype=torch.float32)

    result = VNCCS_QWEN_Encoder()._prepare_encoder_image(image, "#0000FF")

    assert torch.allclose(result[0, 0, 0], torch.tensor([0.1, 0.2, 0.8]))


def test_encoder_leaves_rgb_untouched():
    image = torch.tensor([[[[0.2, 0.4, 0.6]]]], dtype=torch.float32)

    result = VNCCS_QWEN_Encoder()._prepare_encoder_image(image)

    assert torch.equal(result, image)
