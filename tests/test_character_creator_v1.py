"""Tests for nodes/character_creator.py — VALIDATE_INPUTS and prompt building."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.character_creator import CharacterCreator


class TestValidateInputs:
    def test_none_returns_error(self):
        result = CharacterCreator.VALIDATE_INPUTS(existing_character=None)
        assert result is not True
        assert isinstance(result, str)

    def test_string_none_returns_error(self):
        result = CharacterCreator.VALIDATE_INPUTS(existing_character="None")
        assert result is not True

    def test_empty_string_returns_error(self):
        result = CharacterCreator.VALIDATE_INPUTS(existing_character="")
        assert result is not True

    def test_missing_dir_returns_error(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        result = CharacterCreator.VALIDATE_INPUTS(existing_character="Ghost")
        assert result is not True
        assert "Ghost" in result

    def test_existing_dir_returns_true(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()
        result = CharacterCreator.VALIDATE_INPUTS(existing_character="Alice")
        assert result is True


class TestCreateCharacterPrompt:
    def _run(self, tmp_path, monkeypatch, **kwargs):
        import utils
        import nodes.character_creator as cc_mod
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        monkeypatch.setattr(cc_mod, "base_output_dir", lambda: str(tmp_path), raising=False)
        (tmp_path / "Alice").mkdir()

        defaults = dict(
            existing_character="Alice",
            background_color="green",
            aesthetics="masterpiece",
            nsfw=False,
            sex="female",
            age=18,
            race="human",
            eyes="blue",
            hair="black",
            face="oval",
            body="slim",
            skin_color="fair",
            additional_details="",
            seed=0,
            negative_prompt="bad quality",
            lora_prompt="",
            new_character_name="",
        )
        defaults.update(kwargs)
        return CharacterCreator().create_character(**defaults)

    def test_returns_7_values(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch)
        assert len(result) == 7

    def test_positive_prompt_contains_aesthetics(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, aesthetics="cinematic quality")
        assert "cinematic quality" in pos

    def test_positive_prompt_contains_hair(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, hair="silver long")
        assert "silver long" in pos

    def test_positive_prompt_contains_eyes(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, eyes="red")
        assert "red" in pos

    def test_positive_prompt_female_has_1girl(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, sex="female")
        assert "1girl" in pos

    def test_positive_prompt_male_has_1boy(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, sex="male")
        assert "1boy" in pos

    def test_nsfw_false_female_underwear(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, sex="female", nsfw=False)
        assert "bra" in pos or "panties" in pos

    def test_nsfw_true_female_nude(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, sex="female", nsfw=True)
        assert "nude" in pos or "naked" in pos

    def test_positive_contains_age(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, age=30)
        assert "30yo" in pos

    def test_lora_prompt_appended(self, tmp_path, monkeypatch):
        pos, *_ = self._run(tmp_path, monkeypatch, lora_prompt="trigger_word")
        assert "trigger_word" in pos

    def test_age_lora_strength_is_float(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch)
        age_strength = result[3]
        assert isinstance(age_strength, float)

    def test_negative_prompt_deduped(self, tmp_path, monkeypatch):
        _, _, neg, *_ = self._run(tmp_path, monkeypatch, sex="female", negative_prompt="bad quality")
        # dedupe_tokens should remove exact duplicates
        tokens = [t.strip() for t in neg.split(",")]
        assert len(tokens) == len(set(tokens))

    def test_sheets_path_contains_character_name(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch)
        sheets_path = result[4]
        assert "Alice" in sheets_path

    def test_faces_path_contains_character_name(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch)
        faces_path = result[5]
        assert "Alice" in faces_path

    def test_face_details_contains_expressionless(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch)
        face_details = result[6]
        assert "expressionless" in face_details

    def test_config_saved_to_disk(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        import nodes.character_creator as cc_mod
        monkeypatch.setattr(cc_mod, "base_output_dir", lambda: str(tmp_path), raising=False)
        (tmp_path / "Alice").mkdir()

        CharacterCreator().create_character(
            existing_character="Alice", background_color="green",
            aesthetics="masterpiece", nsfw=False, sex="female", age=18,
            race="human", eyes="blue", hair="black", face="oval",
            body="slim", skin_color="fair", additional_details="",
            seed=0, negative_prompt="bad", lora_prompt="", new_character_name="",
        )
        config = utils.load_config("Alice")
        assert config is not None
        assert config["character_info"]["sex"] == "female"
