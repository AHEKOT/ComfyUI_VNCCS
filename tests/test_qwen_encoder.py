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


@pytest.mark.parametrize("slots", [(2,), (3,), (1, 3), (1, 2, 3), ()])
@pytest.mark.parametrize("selected", [1, 2, 3])
def test_sparse_image_slots_keep_their_weights_and_selected_latent(monkeypatch, slots, selected):
    from types import SimpleNamespace
    from nodes import vnccs_qwen_encoder as module

    monkeypatch.setattr(module.node_helpers, "conditioning_set_values",
                        lambda conditioning, values, **kwargs: [(item[0], {**item[1], **values}) for item in conditioning], raising=False)
    tokenized = []
    clip = SimpleNamespace(tokenize=lambda prompt, **kwargs: tokenized.append((prompt, kwargs)),
                           encode_from_tokens_scheduled=lambda tokens: [(torch.ones(1), {})])
    vae = SimpleNamespace(encode=lambda image: image.movedim(-1, 1))
    encoder = VNCCS_QWEN_Encoder()
    monkeypatch.setattr(encoder, "_process_image", lambda image, *args: image)
    images = {slot: torch.full((1, 2, 3, 3), slot / 3) for slot in slots}
    weights = [0, 0.5, 2]
    positive, negative, latent = encoder.encode(clip, "prompt", vae, latent_image_index=selected,
        weight1=weights[0], weight2=weights[1], weight3=weights[2],
        **{f"image{slot}": image for slot, image in images.items()})
    expected = [(weights[slot - 1] ** 2) * image.movedim(-1, 1)
                for slot, image in images.items() if weights[slot - 1] > 0]
    references = positive[0][1].get("reference_latents", [])
    assert len(references) == len(expected)
    assert all(torch.equal(actual, wanted) for actual, wanted in zip(references, expected))
    assert len(tokenized[0][1]["images"]) == len(slots)
    assert [f"Picture {slot}" for slot in slots] == [name for name in ("Picture 1", "Picture 2", "Picture 3") if name in tokenized[0][0]]
    assert torch.count_nonzero(negative[0][0]) == 0
    if selected in images:
        assert torch.equal(latent["samples"], images[selected].movedim(-1, 1))
    else:
        assert latent["samples"].shape == (1, 4, 128, 128)
        assert torch.count_nonzero(latent["samples"]) == 0
