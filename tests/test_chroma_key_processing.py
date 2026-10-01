"""Chroma key equivalence, transparent output and edge compositing regressions."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from nodes import vnccs_utils as utils


def settings(**overrides):
    defaults = {
        name: spec[1]["default"]
        for name, spec in utils.VNCCSChromaKey.INPUT_TYPES()["required"].items()
        if name != "image"
    }
    return {**defaults, **overrides}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (24, 32)])
@pytest.mark.parametrize("radius", [0, 1, 3, 32])
@pytest.mark.parametrize("mode", ["dilate", "erode"])
def test_cpu_morphology_matches_pooling_at_borders(dtype, shape, radius, mode):
    generator = torch.Generator().manual_seed(37)
    masks = [
        torch.zeros(shape, dtype=dtype), torch.ones(shape, dtype=dtype),
        torch.rand(shape[::-1], dtype=dtype, generator=generator).T,
    ]
    for mask in masks:
        before = mask.clone()
        source = mask[None, None] if mode == "dilate" else -mask[None, None]
        expected = F.max_pool2d(source, 2 * radius + 1, stride=1, padding=radius)[0, 0]
        if mode == "erode":
            expected = -expected
        actual = utils._morph(mask, radius, mode)
        assert torch.equal(actual, expected)
        assert torch.equal(mask, before)
        assert actual.dtype == dtype


def test_morphology_preserves_autograd():
    mask = torch.arange(9, dtype=torch.float32).reshape(3, 3).requires_grad_()
    utils._morph(mask, 1, "dilate").sum().backward()
    assert mask.grad is not None
    assert mask.grad.sum() == 9


def test_morphology_keeps_unsupported_cpu_dtype_on_torch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Half precision must not enter OpenCV morphology")
    monkeypatch.setattr(utils.cv2, "dilate", forbidden)
    mask = torch.arange(9, dtype=torch.float16).reshape(3, 3)
    expected = F.max_pool2d(mask[None, None], 3, stride=1, padding=1)[0, 0]
    assert torch.equal(utils._morph(mask, 1, "dilate"), expected)


@pytest.mark.parametrize("mode", ["straight_rgba", "premultiplied_rgba"])
@pytest.mark.parametrize("despill", [0.0, 0.65])
def test_fully_transparent_output_has_zero_rgb(mode, despill):
    image = torch.tensor([0.1, 0.7, 0.35]).expand(1, 32, 32, 3).clone()
    image[:, 8:24, 8:24] = torch.tensor([0.8, 0.1, 0.2])
    rgba, matte, debug = utils.VNCCSChromaKey().chroma_key(
        image, **settings(output_mode=mode, despill_strength=despill),
    )
    transparent = matte == 0
    assert transparent.any()
    assert torch.count_nonzero(rgba[..., :3][transparent]) == 0
    assert torch.equal(rgba[..., 3], matte)
    assert torch.equal(debug[..., 1], matte)


@pytest.mark.parametrize("key,foreground", [
    ([0.1, 0.7, 0.35], [0.08, 0.12, 0.72]),
    ([0.08, 0.15, 0.95], [0.75, 0.08, 0.10]),
    ([0.95, 0.12, 0.08], [0.08, 0.10, 0.75]),
])
@pytest.mark.parametrize("mode", ["straight_rgba", "premultiplied_rgba"])
def test_soft_edges_recompose_without_old_screen_halo(key, foreground, mode):
    key, foreground = torch.tensor(key), torch.tensor(foreground)
    coverage = torch.zeros(32, 64)
    coverage[8:24, 20:44] = 1
    for offset, value in enumerate([0.95, 0.8, 0.65, 0.5, 0.35]):
        coverage[8:24, 19 - offset] = value
    source = foreground * coverage[..., None] + key * (1 - coverage[..., None])
    rgba, matte, _ = utils.VNCCSChromaKey().chroma_key(source, **settings(output_mode=mode))
    edge = (0, 16, slice(15, 20))
    assert torch.allclose(matte[edge], coverage[16, 15:20], atol=1e-4)
    assert torch.all((matte[edge] > 0) & (matte[edge] < 1))
    for background in ([0., 0., 0.], [1., 1., 1.], [1., 0., 1.], [0., 1., 0.]):
        background = torch.tensor(background)
        rgb = rgba[..., :3]
        if mode == "straight_rgba":
            rgb = rgb * matte[..., None]
        composite = rgb + background * (1 - matte[..., None])
        expected = foreground * coverage[..., None] + background * (1 - coverage[..., None])
        assert torch.allclose(composite[edge], expected[16, 15:20], atol=1e-4)
    if mode == "straight_rgba":
        filled, = utils.VNCCS_MaskExtractor().fill_alpha_with_color(rgba)
        assert torch.allclose(filled[edge], expected[16, 15:20], atol=1e-4)


def test_unmix_preserves_unrelated_edge_colors_and_distant_foreground():
    node = utils.VNCCSChromaKey()
    nearest = torch.tensor([0.08, 0.12, 0.72]).expand(3, 3, 3)
    key = torch.tensor([0.1, 0.7, 0.35])
    source = torch.tensor([0.9, 0.7, 0.1]).expand(3, 3, 3).clone()
    source[1, 1] = nearest[1, 1] * 0.5 + key * 0.5
    distance = torch.ones(3, 3)
    distance[1, 1] = 10
    alpha = torch.full((3, 3), 0.8)
    rgb, refined = node._unmix_screen_edges(source, source, alpha, key, (nearest, distance), 5)
    assert torch.equal(rgb, source)
    assert torch.equal(refined, alpha)


def test_no_opaque_anchor_and_empty_foreground_are_safe():
    node = utils.VNCCSChromaKey()
    assert node._nearest_opaque_colors(torch.rand(3, 3, 3), torch.full((3, 3), 0.5), 3) is None
    rgba, matte, _ = node.chroma_key(torch.tensor([0., 1., 0.]).expand(1, 16, 16, 3), **settings())
    assert torch.count_nonzero(rgba) == 0
    assert torch.count_nonzero(matte) == 0


def test_unmix_does_not_restore_removed_background():
    node = utils.VNCCSChromaKey()
    foreground = torch.tensor([0.08, 0.12, 0.72]).expand(3, 3, 3)
    key = torch.tensor([0.1, 0.7, 0.35])
    shifted_screen = foreground * 0.1 + key * 0.9
    alpha = torch.zeros(3, 3)
    _, refined = node._unmix_screen_edges(
        shifted_screen, shifted_screen, alpha, key, (foreground, torch.ones(3, 3)), 5,
    )
    assert torch.equal(refined, alpha)


def test_sam3_recovery_does_not_multiply_premultiplied_edges_twice():
    node = utils.VNCCSChromaKey()
    alpha = torch.full((16, 16), 0.5)
    alpha[0, 0] = 0
    color = torch.tensor([0.8, 0.2, 0.1]).expand(16, 16, 3)
    original = torch.full_like(color, 0.7)
    mask = torch.zeros_like(alpha)
    mask[6:10, 6:10] = 1
    debug = torch.stack([torch.zeros_like(alpha), alpha, 1 - alpha], -1)
    outputs = []
    for mode in ("straight_rgba", "premultiplied_rgba"):
        rgba = node._pack_rgba(color, alpha, mode)
        restored, matte, _ = node._restore_recovery_details(original, rgba, alpha, debug, mask, mode, erode_radius=0)
        assert torch.count_nonzero(restored[0, 0, :3]) == 0
        assert torch.equal(restored[2, 2], rgba[2, 2])
        rgb = restored[..., :3]
        outputs.append(rgb * matte[..., None] if mode == "straight_rgba" else rgb)
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("mode", ["straight_rgba", "premultiplied_rgba"])
def test_balanced_cleans_real_hair_without_removing_green_leg(mode):
    # Frozen RGB source: the user's working output file can change independently.
    source = np.asarray(Image.open(Path(__file__).parent / "fixtures/chroma_green_character.png").convert("RGB")).copy()
    image = torch.from_numpy(source).float() / 255
    with torch.inference_mode():
        rgba, matte, _ = utils.VNCCSChromaKey().chroma_key(image, **settings(output_mode=mode))
    if mode == "straight_rgba":
        from nodes import character_generator as cg
        with torch.inference_mode():
            generated = cg.VNCCS_CharacterGenerator()._run_bg_remove(
                image, dict(cg.DEFAULT_WIDGET_DATA["bg_remove"]), background="Green",
            )
        assert torch.equal(generated, rgba)
    rgb, alpha = rgba[0, ..., :3], matte[0]
    straight = rgb / alpha.clamp(min=1e-6)[..., None] if mode == "premultiplied_rgba" else rgb

    # Annotated hair region excludes the intentionally blue sleeve and green leg.
    hair = torch.zeros_like(alpha, dtype=torch.bool)
    hair[:980, 280:1025] = True
    hair[550:690, 830:1160] = False
    hair[910:980, 750:875] = False
    hair[640:690, 865:910] = True
    r, g, b = straight.unbind(-1)
    visible_hair = hair & (alpha >= 1 / 255)
    assert not torch.any(visible_hair & (g > r + 1 / 255) & (g > b + 1 / 255))
    assert not torch.any(visible_hair & (g > r * 1.2 + 1 / 255) & (b > r))

    leg = np.zeros(source.shape[:2], dtype=bool)
    leg[910:1370, 750:880] = True
    leg &= np.linalg.norm(source.astype(float) - [76, 209, 96], axis=-1) < 15
    assert leg.sum() > 26000
    assert torch.all(alpha[leg] > 0)
    # Antialiasing is allowed at the contour; the interior must be unchanged.
    interior = utils.cv2.erode(leg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert torch.all(alpha[interior] == 1)
    assert torch.equal(straight[interior], image[interior])
    # Blue fingers are real foreground, including small shaded details.
    hand = np.zeros_like(leg)
    hand[620:680, 1080:1140] = True
    hand &= np.linalg.norm(source.astype(float) - [48, 174, 231], axis=-1) < 20
    assert hand.sum() > 2600
    assert alpha[hand].mean() > 0.99
    color_error = (straight[hand] - image[hand]).abs().amax(dim=-1)
    assert torch.quantile(color_error, 0.99) < 4 / 255
    assert torch.count_nonzero(rgb[alpha == 0]) == 0
