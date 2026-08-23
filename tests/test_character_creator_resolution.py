import pytest

from conftest import _preload_node


pytest.importorskip("torch")

character_creator_v2 = _preload_node("character_creator_v2")
get_generation_resolution = character_creator_v2.get_generation_resolution
normalize_gen_settings = character_creator_v2.normalize_gen_settings


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("normal", (640, 1536)),
        ("high", (856, 2048)),
        ("maximum", (1024, 2456)),
    ],
)
def test_anima_resolution_presets(preset, expected):
    settings = normalize_gen_settings({
        "generation_mode": "anima",
        "resolution_preset": preset,
    })

    assert settings["resolution_preset"] == preset
    assert get_generation_resolution(settings) == expected


def test_unknown_anima_resolution_falls_back_to_normal():
    settings = normalize_gen_settings({
        "generation_mode": "anima",
        "resolution_preset": "unsupported",
    })

    assert settings["resolution_preset"] == "normal"
    assert get_generation_resolution(settings) == (640, 1536)


def test_illustrious_ignores_anima_resolution_preset():
    settings = normalize_gen_settings({
        "generation_mode": "illustrious",
        "resolution_preset": "maximum",
    })

    assert get_generation_resolution(settings) == (640, 1536)
