import pytest

from conftest import _preload_node


pytest.importorskip("torch")

character_creator_v2 = _preload_node("character_creator_v2")
get_generation_resolution = character_creator_v2.get_generation_resolution
normalize_gen_settings = character_creator_v2.normalize_gen_settings


@pytest.mark.parametrize(
    ("target_size", "expected"),
    [
        (1024, (792, 1408)),
        (1344, (864, 1536)),
        (1536, (936, 1664)),
        (4096, (1512, 2688)),
    ],
)
def test_resolution_scale_creates_strict_nine_by_sixteen_latent(target_size, expected):
    settings = normalize_gen_settings({
        "generation_mode": "qi2",
        "target_size": target_size,
    })

    assert settings["target_size"] == target_size
    width, height = get_generation_resolution(settings)
    assert (width, height) == expected
    assert width * 16 == height * 9
    assert (width // 8) * 16 == (height // 8) * 9


def test_resolution_scale_is_clamped_to_one_to_four_megapixels():
    low = normalize_gen_settings({"generation_mode": "illustrious", "target_size": 100})
    high = normalize_gen_settings({"generation_mode": "anima", "target_size": 9000})

    assert low["target_size"] == 1024
    assert high["target_size"] == 4096


def test_legacy_anima_resolution_preset_migrates_to_scale():
    settings = normalize_gen_settings({
        "generation_mode": "anima",
        "resolution_preset": "maximum",
    })

    assert "resolution_preset" not in settings
    assert settings["target_size"] == 2458
    assert get_generation_resolution(settings) == (1224, 2176)


@pytest.mark.parametrize("mode", ["illustrious", "anima", "qi2"])
def test_resolution_scale_applies_to_every_generation_mode(mode):
    settings = normalize_gen_settings({"generation_mode": mode, "target_size": 1536})

    assert get_generation_resolution(settings) == (936, 1664)
