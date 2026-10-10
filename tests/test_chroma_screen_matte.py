"""Known-color mattes and connectivity regressions, independent of real assets."""

import pytest

torch = pytest.importorskip("torch")

from nodes.chroma_screen_matte import screen_matte, _reconstruct_detail


def process(image, **kwargs):
    return screen_matte(image, processing_device="cpu", **kwargs)


@pytest.mark.parametrize("key", [(0, 1, 0), (0.25, 0.6, 0.8), (0.8, 0.8, 0.8)])
def test_empty_plate_has_zero_alpha_and_zero_hidden_rgb(key):
    image = torch.tensor(key, dtype=torch.float32).expand(65, 49, 3).clone()
    rgba, alpha, debug = process(image)
    assert torch.count_nonzero(rgba) == 0
    assert alpha.shape == (65, 49)
    assert debug.shape == (65, 49, 3)


def test_green_foreground_on_teal_screen_and_enclosed_hole():
    image = torch.tensor([0.18, 0.9, 0.55]).expand(160, 128, 3).clone()
    image[20:145, 25:105] = torch.tensor([0.28, 0.8, 0.27])
    image[40:65, 50:65] = torch.tensor([0.18, 0.9, 0.55])
    rgba, alpha, _ = process(image)
    assert torch.all(alpha[80:130, 40:90] == 1)
    assert torch.all(alpha[44:61, 54:61] == 0)
    torch.testing.assert_close(rgba[100, 60, :3], image[100, 60])


def test_same_hue_clothing_inside_silhouette_survives_blue_plate():
    image = torch.tensor([0.28, 0.6, 0.81]).expand(180, 140, 3).clone()
    image[20:165, 25:115] = torch.tensor([0.9, 0.72, 0.55])
    image[65:125, 45:95] = torch.tensor([0.55, 0.82, 0.95])
    rgba, alpha, _ = process(image)
    assert torch.all(alpha[70:120, 50:90] == 1)
    torch.testing.assert_close(rgba[90, 60, :3], image[90, 60])


def test_pale_green_skin_highlight_is_not_a_hole():
    image = torch.tensor([0.01, 0.94, 0.01]).expand(180, 140, 3).clone()
    image[20:165, 25:115] = torch.tensor([0.9, 0.72, 0.55])
    image[60:110, 45:95] = torch.tensor([0.8, 0.94, 0.8])
    rgba, alpha, _ = process(image)
    assert torch.all(alpha[65:105, 50:90] == 1)
    torch.testing.assert_close(rgba[90, 60, :3], image[90, 60])


def test_dark_internal_line_does_not_make_green_hair_transparent():
    image = torch.tensor([0.36, 0.72, 0.40]).expand(160, 128, 3).clone()
    image[20:145, 25:105] = torch.tensor([0.13, 0.88, 0.63])
    image[35:125, 60:63] = torch.tensor([0.14, 0.61, 0.44])
    rgba, alpha, _ = process(image)
    assert torch.all(alpha[30:135, 35:95] == 1)
    torch.testing.assert_close(rgba[30:135, 35:95, :3], image[30:135, 35:95])


@pytest.mark.parametrize("foreground", [(0.9, 0.15, 0.2), (0, 0, 0)])
def test_half_coverage_edge_unmixes_screen_color(foreground):
    key = torch.tensor([0.0, 1.0, 0.0])
    foreground = torch.tensor(foreground, dtype=torch.float32)
    image = key.expand(120, 100, 3).clone()
    image[20:100, 30:70] = foreground
    image[20:100, 29] = (foreground + key) * 0.5
    rgba, alpha, _ = process(image, matte_cleanup=0)
    assert float(alpha[60, 29]) == pytest.approx(0.5, abs=0.035)
    torch.testing.assert_close(rgba[60, 29, :3], foreground, atol=0.06, rtol=0)


def test_dark_ink_edge_next_to_light_hair_keeps_coverage_without_green():
    image = torch.tensor([0.0, 1.0, 0.0]).expand(120, 100, 3).clone()
    image[20:100, 30:70] = torch.tensor([0.8, 0.8, 0.85])
    image[20:100, 29] = torch.tensor([0.0, 0.3, 0.0])
    rgba, alpha, _ = process(image)
    assert float(alpha[60, 29]) == pytest.approx(0.7, abs=0.01)
    torch.testing.assert_close(rgba[60, 29, :3], torch.zeros(3), atol=0.01, rtol=0)


def test_sparse_reconstruction_keeps_long_hair_without_bridging_gap():
    allowed = torch.zeros(1, 1, 90, 100, dtype=torch.bool)
    allowed[:, :, 10:80, 10:40] = True
    allowed[:, :, 10, 40:90] = True
    allowed[:, :, 12, 50:90] = True  # Detached line, one empty row between them.
    anchor = torch.zeros_like(allowed)
    anchor[:, :, 30:60, 15:35] = True
    anchor[:, :, 11, 70] = True  # Invalid coarse seed must not bridge the gap.
    result = _reconstruct_detail(anchor, allowed)
    assert torch.all(result[:, :, 10, 40:90] == 1)
    assert torch.all(result[:, :, 12, 50:90] == 0)
    assert torch.all(result[~allowed] == 0)


def test_cleanup_preserves_a_long_one_pixel_strand_beyond_coarse_support():
    image = torch.tensor([0.0, 1.0, 0.0]).expand(130, 220, 3).clone()
    foreground = torch.tensor([0.9, 0.2, 0.15])
    image[20:110, 20:70] = foreground
    image[50, 70:195] = foreground
    image[53, 90:195] = foreground  # Detached noise near the genuine strand.
    _, alpha, _ = process(image)
    assert torch.all(alpha[50, 70:195] == 1)
    assert torch.all(alpha[53, 90:195] == 0)


def test_rgba_input_and_premultiplied_output_preserve_existing_transparency():
    image = torch.tensor([0.0, 1.0, 0.0, 1.0]).expand(120, 100, 4).clone()
    image[20:100, 25:75] = torch.tensor([0.9, 0.2, 0.15, 0.5])
    image[50:60, 45:55, 3] = 0
    straight, alpha, _ = process(image)
    premultiplied, premult_alpha, _ = process(image, output_mode="premultiplied_rgba")
    torch.testing.assert_close(premultiplied[..., :3], straight[..., :3] * alpha[..., None])
    torch.testing.assert_close(alpha, premult_alpha)
    assert alpha[70, 50] == 0.5
    assert torch.count_nonzero(straight[alpha == 0]) == 0


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_gpu_matches_cpu_and_returns_to_callers_device(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable")
    image = torch.tensor([0.0, 1.0, 0.0]).expand(181, 141, 3).clone()
    image[30:160, 30:110] = torch.tensor([0.85, 0.3, 0.2])
    image[80:95, 60:75] = torch.tensor([0.0, 1.0, 0.0])
    cpu = process(image)
    gpu = screen_matte(image, processing_device=device)
    for expected, actual in zip(cpu, gpu):
        assert actual.device == image.device
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)


def test_internal_screen_matte_handles_batches_without_changing_generator_defaults():
    from nodes.vnccs_utils import VNCCSChromaKey
    from nodes.character_generator import CHROMA_KEY_PRESETS

    node = VNCCSChromaKey()
    settings = dict(CHROMA_KEY_PRESETS["balanced"])
    assert settings["matte_method"] == "guided_edge"
    settings["matte_method"] = "screen_matte"
    settings["screen_mode"] = "auto"
    image = torch.tensor([0.0, 1.0, 0.0]).expand(2, 100, 80, 3).clone()
    image[:, 20:80, 20:60] = torch.tensor([0.85, 0.3, 0.2])
    rgba, alpha, debug = node.chroma_key(image, **settings)
    assert rgba.shape == (2, 100, 80, 4)
    assert alpha.shape == (2, 100, 80)
    assert debug.shape == (2, 100, 80, 3)
    assert torch.all(alpha[:, 30:70, 30:50] == 1)
    assert torch.count_nonzero(rgba[alpha == 0]) == 0
