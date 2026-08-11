"""Tests for the Flux Klein encoder's native-node graph."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes import vnccs_flux_klein_encoder as encoder_module
from nodes.vnccs_flux_klein_encoder import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    VNCCS_Flux_Klein_Encoder,
)


def _install_fake_nodes(monkeypatch):
    calls = []

    class FakeScaledImage:
        shape = (1, 1152, 768, 3)

        def __init__(self, source):
            self.source = source

        def __str__(self):
            return f"scaled-{self.source}"

    def node(name, function, callback):
        return type(
            name,
            (),
            {
                "FUNCTION": function,
                function: lambda self, **kwargs: callback(kwargs),
            },
        )

    reference_index = {"value": 0}

    def record(name, result):
        def callback(kwargs):
            calls.append((name, kwargs))
            return result(kwargs) if callable(result) else result

        return callback

    def reference_result(kwargs):
        reference_index["value"] += 1
        return (f"conditioned-{reference_index['value']}",)

    mappings = {
        "CLIPTextEncode": node(
            "FakeCLIPTextEncode",
            "encode",
            record("CLIPTextEncode", ("base-positive",)),
        ),
        "ConditioningZeroOut": node(
            "FakeConditioningZeroOut",
            "zero_out",
            record("ConditioningZeroOut", ("negative",)),
        ),
        "ImageScaleToTotalPixels": node(
            "FakeImageScaleToTotalPixels",
            "upscale",
            record("ImageScaleToTotalPixels", lambda kwargs: (FakeScaledImage(kwargs["image"]),)),
        ),
        "VAEEncode": node(
            "FakeVAEEncode",
            "encode",
            record("VAEEncode", lambda kwargs: (f"latent-{kwargs['pixels']}",)),
        ),
        "ReferenceLatent": node(
            "FakeReferenceLatent",
            "append",
            record("ReferenceLatent", reference_result),
        ),
        "EmptyFlux2LatentImage": node(
            "FakeEmptyFlux2LatentImage",
            "generate",
            record(
                "EmptyFlux2LatentImage",
                lambda kwargs: ({
                    "samples": (kwargs["width"], kwargs["height"], kwargs["batch_size"]),
                },),
            ),
        ),
    }
    monkeypatch.setattr(encoder_module.comfy_nodes, "NODE_CLASS_MAPPINGS", mappings, raising=False)
    return calls


def test_registration_and_optional_image_inputs():
    assert NODE_CLASS_MAPPINGS["VNCCS_Flux_Klein_Encoder"] is VNCCS_Flux_Klein_Encoder
    assert NODE_DISPLAY_NAME_MAPPINGS["VNCCS_Flux_Klein_Encoder"] == "VNCCS Flux Klein Encoder"
    assert {"image1", "image2", "image3"} <= set(VNCCS_Flux_Klein_Encoder.INPUT_TYPES()["optional"])


def test_no_images_skips_every_reference_processing_block(monkeypatch):
    calls = _install_fake_nodes(monkeypatch)

    positive, negative, latent = VNCCS_Flux_Klein_Encoder().encode(
        clip="clip",
        prompt="prompt",
        vae="vae",
        empty_width=640,
        empty_height=960,
        batch_size=2,
    )

    assert positive == "base-positive"
    assert negative == "negative"
    assert latent == {"samples": (640, 960, 2)}
    assert [name for name, _ in calls] == [
        "CLIPTextEncode",
        "ConditioningZeroOut",
        "EmptyFlux2LatentImage",
    ]


@pytest.mark.parametrize("image_key", ["image1", "image2", "image3"])
def test_sparse_image_inputs_only_run_their_own_blocks(monkeypatch, image_key):
    calls = _install_fake_nodes(monkeypatch)

    inputs = {
        "clip": "clip",
        "prompt": "prompt",
        "vae": "vae",
        image_key: f"{image_key}-value",
        "megapixels": 1.5,
        "resolution_steps": 4,
    }
    positive, negative, latent = VNCCS_Flux_Klein_Encoder().encode(
        **inputs,
    )

    assert positive == "conditioned-1"
    assert negative == "negative"
    assert latent == {"samples": (768, 1152, 1)}
    assert [name for name, _ in calls].count("ImageScaleToTotalPixels") == 1
    assert [name for name, _ in calls].count("VAEEncode") == 1
    assert [name for name, _ in calls].count("ReferenceLatent") == 1
    scale_call = next(kwargs for name, kwargs in calls if name == "ImageScaleToTotalPixels")
    assert scale_call == {
        "image": f"{image_key}-value",
        "upscale_method": "lanczos",
        "megapixels": 1.5,
        "resolution_steps": 4,
    }


def test_three_images_chain_reference_conditioning_in_input_order(monkeypatch):
    calls = _install_fake_nodes(monkeypatch)

    positive, negative, latent = VNCCS_Flux_Klein_Encoder().encode(
        clip="clip",
        prompt="change outfit",
        vae="vae",
        image1="image-1",
        image2="image-2",
        image3="image-3",
    )

    assert positive == "conditioned-3"
    assert negative == "negative"
    assert latent == {"samples": (768, 1152, 1)}

    reference_calls = [kwargs for name, kwargs in calls if name == "ReferenceLatent"]
    assert [call["conditioning"] for call in reference_calls] == [
        "base-positive",
        "conditioned-1",
        "conditioned-2",
    ]
    assert [call["latent"] for call in reference_calls] == [
        "latent-scaled-image-1",
        "latent-scaled-image-2",
        "latent-scaled-image-3",
    ]
    zero_call = next(kwargs for name, kwargs in calls if name == "ConditioningZeroOut")
    assert zero_call["conditioning"] == "base-positive"
    assert "GetImageSize" not in [name for name, _ in calls]
    latent_call = next(kwargs for name, kwargs in calls if name == "EmptyFlux2LatentImage")
    assert (latent_call["width"], latent_call["height"]) == (768, 1152)
