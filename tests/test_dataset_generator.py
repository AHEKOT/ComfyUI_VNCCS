"""Tests for nodes/dataset_generator.py — build_caption_text logic."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.dataset_generator import DatasetGenerator


def _info(**overrides):
    base = {
        "sex": "female", "age": 25, "race": "elf",
        "hair": "silver long", "eyes": "green", "face": "freckles",
        "body": "slim", "skin_color": "pale", "additional_details": "",
        "background_color": "green",
    }
    base.update(overrides)
    return base


class TestBuildCaptionText:
    def setup_method(self):
        self.gen = DatasetGenerator.__new__(DatasetGenerator)

    def test_starts_with_character_name(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_Alice")
        assert result.startswith("Game_Alice")

    def test_female_adds_1girl(self):
        result = self.gen.build_caption_text(_info(sex="female"), "Naked", "neutral", "portrait", "Game_A")
        assert "1girl" in result

    def test_male_adds_1boy(self):
        result = self.gen.build_caption_text(_info(sex="male"), "Naked", "neutral", "portrait", "Game_A")
        assert "1boy" in result

    def test_age_added(self):
        result = self.gen.build_caption_text(_info(age=22), "Naked", "neutral", "portrait", "Game_A")
        assert "22yo" in result

    def test_image_type_added(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A")
        assert "portrait" in result

    def test_full_body_image_type(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "full body", "Game_A")
        assert "full body" in result

    def test_race_added(self):
        result = self.gen.build_caption_text(_info(race="orc"), "Naked", "neutral", "portrait", "Game_A")
        assert "orc" in result

    def test_hair_added(self):
        result = self.gen.build_caption_text(_info(hair="blue short"), "Naked", "neutral", "portrait", "Game_A")
        assert "blue short hair" in result

    def test_eyes_added(self):
        result = self.gen.build_caption_text(_info(eyes="red"), "Naked", "neutral", "portrait", "Game_A")
        assert "red eyes" in result

    def test_skin_color_added(self):
        result = self.gen.build_caption_text(_info(skin_color="dark"), "Naked", "neutral", "portrait", "Game_A")
        assert "dark skin" in result

    def test_body_added_for_full_body(self):
        result = self.gen.build_caption_text(_info(body="muscular"), "Naked", "neutral", "full body", "Game_A")
        assert "muscular" in result

    def test_body_not_added_for_portrait(self):
        result = self.gen.build_caption_text(_info(body="muscular"), "Naked", "neutral", "portrait", "Game_A")
        assert "muscular" not in result

    def test_emotion_added(self):
        result = self.gen.build_caption_text(_info(), "Naked", "happy", "portrait", "Game_A")
        assert "happy" in result

    def test_naked_costume(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A")
        assert "naked" in result

    def test_non_naked_costume(self):
        result = self.gen.build_caption_text(_info(), "Dress", "neutral", "portrait", "Game_A")
        assert "wear Dress suit" in result

    def test_background_color_added(self):
        result = self.gen.build_caption_text(_info(background_color="blue"), "Naked", "neutral", "portrait", "Game_A")
        assert "blue background" in result

    def test_additional_caption_appended(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A", "best quality")
        assert "best quality" in result

    def test_empty_additional_caption_not_added(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A", "   ")
        # Trailing whitespace-only caption should not leave a trailing comma
        assert not result.endswith(", ")

    def test_sweatdrop_tag_always_present(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A")
        assert "sweatdrop" in result

    def test_full_body_has_multiple_views_tag(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "full body", "Game_A")
        assert "multiple views" in result

    def test_portrait_no_multiple_views_tag(self):
        result = self.gen.build_caption_text(_info(), "Naked", "neutral", "portrait", "Game_A")
        assert "multiple views" not in result

    def test_gender_fallback_uses_gender_key(self):
        info = _info()
        del info["sex"]
        info["gender"] = "male"
        result = self.gen.build_caption_text(info, "Naked", "neutral", "portrait", "Game_A")
        assert "1boy" in result
