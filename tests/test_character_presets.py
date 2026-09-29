"""Creator V2 catalog compatibility and species expansion contracts."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import _preload_node

presets = _preload_node("character_presets")
ROOT = Path(__file__).parents[1]
TAGS = presets.CHARACTER_PRESETS["tags"]


@pytest.fixture(scope="module")
def creator():
    pytest.importorskip("torch")
    return _preload_node("character_creator_v2")


def test_breast_presets_remain_identical_to_legacy_catalog():
    legacy = json.loads((ROOT / "character_template/character_tags.json").read_text())
    assert TAGS["breast_size"] == legacy["tags"]["breast_size"]


def test_catalog_has_distinct_choices_and_unambiguous_aliases():
    owners = {}
    for category, items in TAGS.items():
        assert items
        for item in items:
            key = presets.preset_key(item["tag"])
            assert key not in owners, (category, key)
            owners[key] = category
        alias_owners = {}
        for item in items:
            for alias in [item["tag"], item["label"], *item.get("synonyms", [])]:
                key = presets.preset_key(alias)
                assert alias_owners.get(key, item["tag"]) == item["tag"]
                alias_owners[key] = item["tag"]
    assert len(TAGS["races"]) >= 50
    assert not {"animal ears", "kemonomimi", "demon horns", "furry female"} & {
        item["tag"] for item in TAGS["races"]
    }


@pytest.mark.parametrize("race", TAGS["races"], ids=lambda race: race["tag"])
def test_every_species_has_a_natural_language_hint(race):
    expanded = presets.race_prompt(race["tag"])
    assert expanded.startswith(race["tag"] + ";")
    assert race["prompt"] in expanded
    assert race["prompt"].endswith(".") and "_" not in race["prompt"]
    assert "override these defaults" in expanded


def test_aliases_and_weighted_hybrid_species_are_explained_once():
    value = "cat_girl, CAT BOY, (elf:1.2), no tail, My Custom Trait"
    expanded = presets.race_prompt(value)
    assert expanded.startswith(value)
    assert expanded.count(presets.RACE_PRESETS["catfolk"]["prompt"]) == 1
    assert expanded.count(presets.RACE_PRESETS["elf"]["prompt"]) == 1


@pytest.mark.parametrize("value", ["", "My Unknown Species", "fox tattoo", "elf-like ears"])
def test_freeform_values_are_not_reinterpreted(value):
    assert presets.race_prompt(value) == value


@pytest.mark.parametrize("mode", ["illustrious", "anima", "qi2"])
def test_species_hints_reach_generation_without_rewriting_saved_info(creator, mode):
    info = {"race": "naga, no scales", "body": "small_breasts", "sex": "male", "age": 30}
    original = copy.deepcopy(info)
    prompt, _ = creator.CharacterCreatorV2.construct_prompt(info, mode)
    assert presets.RACE_PRESETS["naga"]["prompt"] in prompt
    assert "naga, no scales" in prompt
    assert "small_breasts" in prompt
    assert info == original
    fields = creator._qi2_character_fields(info)
    assert fields["race"] == info["race"]
    assert fields["race_features"] == presets.race_features(info["race"])
    assert fields["body"] == "small_breasts"
    # Invalid PE output must not lose species hints or explicit overrides.
    fallback = creator._qi2_expanded_field_prompt("invalid JSON", fields)
    assert presets.RACE_PRESETS["naga"]["prompt"] in fallback
    assert "no scales" in fallback


def test_wizard_uses_all_sections_and_accepts_curated_races(creator):
    options = creator._extract_character_tag_options(presets.CHARACTER_PRESETS)
    assert "broad shoulders" in options["body"]
    assert "small_breasts" in options["body"]
    assert "olive skin" in options["skin_color"]
    assert "oval face" in options["face"]
    assert "oval face" not in options["eyes"]
    for race in TAGS["races"]:
        assert creator._normalize_wizard_race(race["tag"], "adventurer") == race["tag"]


def test_catalog_route_preserves_default_response_for_cloner(creator, monkeypatch):
    monkeypatch.setattr(creator.web, "json_response", lambda data: data, raising=False)
    request = lambda query: SimpleNamespace(rel_url=SimpleNamespace(query=query))
    modern = asyncio.run(creator.get_tags(request({"catalog": "creator_v2"})))
    legacy = asyncio.run(creator.get_tags(request({})))
    assert modern == presets.CHARACTER_PRESETS
    assert legacy == json.loads((ROOT / "character_template/character_tags.json").read_text())
