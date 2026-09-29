import json
from pathlib import Path

import pytest

from conftest import _preload_node


pytest.importorskip("torch")

creator = _preload_node("character_creator_v2")
CATALOG_PATH = Path(__file__).parents[1] / "character_template" / "character_styles.json"
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _base_info(**overrides):
    info = {
        "sex": "female",
        "age": 18,
        "race": "human",
        "background_color": "Green",
    }
    info.update(overrides)
    return info


def test_style_catalog_contains_requested_range():
    styles = creator.CHARACTER_STYLE_PROMPTS
    anime_and_manga_styles = {
        "classic_anime",
        "modern_clean_anime",
        "soft_pastel_anime",
        "bold_cel_anime",
        "cinematic_anime",
        "anime_key_visual",
        "light_novel_illustration",
        "visual_novel_character",
        "game_character_illustration",
        "painterly_anime",
        "manga_ink",
        "chibi_anime",
        "retro_80s_anime",
        "retro_90s_anime",
        "digital_2000s_anime",
        "glossy_anime",
        "fashion_anime",
        "anime_3d_cel",
    }

    assert creator.DEFAULT_CHARACTER_STYLE == "classic_anime"
    assert len(styles) >= 40
    assert anime_and_manga_styles.issubset(styles)
    assert {
        "american_superhero_comic",
        "classic_western_cel",
        "ukiyoe_woodblock",
        "pixel_art",
        "photoreal_studio",
        "cinematic_live_action",
        "editorial_fashion_photo",
        "documentary_photo",
        "realist_oil_portrait",
    }.issubset(styles)
    assert {
        "cut_paper_animation",
        "expressionist_painting",
        "cubist_geometric",
        "surreal_dreamscape",
        "stained_glass",
        "paper_collage",
        "risograph_print",
        "low_poly_3d",
        "synthwave_neon",
    }.isdisjoint(styles)


def test_style_catalog_is_loaded_from_character_template_json():
    json_styles = {
        style["id"]: style["prompt"]
        for group in CATALOG["groups"]
        for style in group["styles"]
    }

    assert creator.CHARACTER_STYLE_CATALOG_PATH == str(CATALOG_PATH)
    assert creator.CHARACTER_STYLE_CATALOG == CATALOG
    assert creator.CHARACTER_STYLE_PROMPTS == json_styles
    assert creator.CHARACTER_STYLE_ALIASES == CATALOG["aliases"]
    assert creator.DEFAULT_CHARACTER_STYLE == CATALOG["default_style"]


def test_default_style_is_classic_anime():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info())
    assert "classic hand-drawn Japanese cel character art" in positive


def test_selected_style_template_is_added():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(
        _base_info(style="noir_graphic_novel")
    )
    assert "noir graphic novel" in positive
    assert "deep angular shadows" in positive


def test_custom_style_replaces_template():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info(
        style="custom",
        custom_style="soft felt puppet illustration",
    ))
    assert "soft felt puppet illustration" in positive
    assert "classic hand-drawn Japanese cel character art" not in positive


@pytest.mark.parametrize(
    ("legacy_key", "expected_fragment"),
    [
        ("shoujo_anime", "soft pastel anime character illustration"),
        ("shonen_anime", "bold cel-shaded anime character art"),
        ("seinen_anime", "cinematic anime character illustration"),
        ("monochrome_manga", "black-and-white manga character art"),
    ],
)
def test_legacy_style_keys_migrate_to_drawing_styles(legacy_key, expected_fragment):
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(
        _base_info(style=legacy_key)
    )
    assert expected_fragment in positive
