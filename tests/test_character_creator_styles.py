import json
import re
from pathlib import Path

import pytest

from conftest import _preload_node


pytest.importorskip("torch")

creator = _preload_node("character_creator_v2")
CATALOG_PATH = Path(__file__).parents[1] / "character_template" / "character_styles.json"
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
STYLES = [style for group in CATALOG["groups"] for style in group["styles"]]


def _base_info(**overrides):
    info = {
        "sex": "female",
        "age": 30,
        "race": "human",
        "background_color": "Green",
    }
    info.update(overrides)
    return info


def test_catalog_contains_the_approved_reference_styles():
    assert {g["label"]: len(g["styles"]) for g in CATALOG["groups"]} == {
        "Anime Meta Styles": 10,
        "Anime": 16, "Animation": 10, "Artistic": 2, "Realistic": 2,
    }
    assert {s["id"] for s in STYLES} == {
        "shonen_anime", "shojo_anime", "seinen_anime", "josei_anime",
        "anime_1970s", "anime_1980s", "anime_1990s", "anime_2000s", "anime_2010s", "anime_2020s",
        "ghibli_miyazaki", "makoto_shinkai", "kyoto_violet_evergarden", "kyoto_k_on",
        "trigger_imaishi", "yoshiyuki_sadamoto", "clamp", "sailor_moon_90s",
        "akira_toriyama", "rumiko_takahashi", "katsuhiro_otomo", "satoshi_kon",
        "ghost_in_the_shell_1995", "toshihiro_kawamoto", "yoshitoshi_abe", "nobuteru_yuki",
        "disney_renaissance", "don_bluth", "bruce_timm_dcau", "genndy_tartakovsky",
        "cartoon_saloon", "disney_tangled", "pixar_incredibles", "fortiche_arcane",
        "spider_verse", "coraline_selick", "art_nouveau_mucha", "art_deco_lempicka",
        "academic_realism", "photorealism",
    }
    assert len({s["id"] for s in STYLES}) == len(STYLES)
    assert len({s["label"] for s in STYLES}) == len(STYLES)
    assert len({s["prompt"] for s in STYLES}) == len(STYLES)
    assert all(re.fullmatch(r"[a-z0-9_]+", s["id"]) for s in STYLES)


def test_style_catalog_is_loaded_without_legacy_aliases():
    assert creator.CHARACTER_STYLE_CATALOG_PATH == str(CATALOG_PATH)
    assert creator.CHARACTER_STYLE_CATALOG == CATALOG
    assert creator.CHARACTER_STYLE_PROMPTS == {s["id"]: s["prompt"] for s in STYLES}
    assert "aliases" not in CATALOG
    assert creator.CHARACTER_STYLE_ALIASES == {}
    assert creator.DEFAULT_CHARACTER_STYLE == CATALOG["default_style"] == "ghibli_miyazaki"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s["id"])
def test_every_style_contains_a_name_reference_and_visual_description(style):
    lines = style["prompt"].splitlines()
    assert len(lines) == 3
    assert lines[0] == f"Style: {style['label']}."
    assert lines[1].startswith("Reference: ") and len(lines[1]) > len("Reference: ")
    assert lines[2].startswith("Visual description: ")
    assert "preserve all specified character details" in lines[2]
    assert "eye colors" in lines[2]
    assert style["prompt"].isascii()


def test_default_style_is_ghibli_miyazaki():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info())
    assert positive.startswith("Style: Hayao Miyazaki / Studio Ghibli.")
    assert "Princess Mononoke; Howl's Moving Castle" in positive


def test_selected_style_template_is_added():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info(style="fortiche_arcane"))
    assert positive.startswith("Style: Fortiche / Arcane.\nReference: Arcane.")


def test_custom_style_replaces_template():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info(
        style="custom", custom_style="soft felt puppet illustration",
    ))
    assert positive.startswith("soft felt puppet illustration")
    assert "Hayao Miyazaki" not in positive


@pytest.mark.parametrize("retired_key", ["classic_anime", "monochrome_manga", "chibi_anime", "pixel_art"])
def test_retired_style_uses_existing_default_fallback(retired_key):
    assert retired_key not in creator.CHARACTER_STYLE_PROMPTS
    assert creator.CharacterCreatorV2.construct_prompt(_base_info(style=retired_key)) == (
        creator.CharacterCreatorV2.construct_prompt(_base_info())
    )


@pytest.mark.parametrize("style", [s["id"] for s in STYLES])
@pytest.mark.parametrize("mode", ["illustrious", "anima", "qi2"])
def test_style_can_be_separated_without_changing_character_fields(style, mode):
    info = _base_info(
        style=style, framing="Full_body", hair="black hair, long hair", eyes="blue eyes",
        face="freckles", body="broad shoulders", skin_color="dark skin",
        additional_details="silver ear studs", negative_prompt="blurry, missing facial features",
        background_color="Transparent" if mode == "qi2" else "Green",
    )
    body, body_negative = creator.CharacterCreatorV2.construct_prompt(info, mode, include_style=False)
    positive, negative = creator.CharacterCreatorV2.construct_prompt(info, mode)
    assert positive == creator.CHARACTER_STYLE_PROMPTS[style] + ", " + body
    assert negative == body_negative
    for detail in ("blue eyes", "black hair", "freckles", "broad shoulders",
                   "dark skin", "silver ear studs", "wear white bra and panties"):
        assert detail in body
