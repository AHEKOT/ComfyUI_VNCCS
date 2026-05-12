"""Tests for nodes/character_creator_v2.py — construct_prompt and config logic."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes.character_creator_v2 import CharacterCreatorV2, normalize_gen_settings


def _base_info(**overrides):
    info = {
        "sex": "female",
        "age": 18,
        "race": "human",
        "skin_color": "fair",
        "hair": "long black",
        "eyes": "blue",
        "face": "oval",
        "body": "slim",
        "additional_details": "",
        "nsfw": False,
        "aesthetics": "masterpiece, best quality",
        "negative_prompt": "bad quality",
        "lora_prompt": "",
        "background_color": "Green",
    }
    info.update(overrides)
    return info


# ── construct_prompt ──────────────────────────────────────────────────────────

class TestConstructPrompt:
    def test_returns_tuple_of_two_strings(self):
        pos, neg = CharacterCreatorV2.construct_prompt(_base_info())
        assert isinstance(pos, str)
        assert isinstance(neg, str)

    def test_positive_contains_aesthetics(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(aesthetics="cinematic"))
        assert "cinematic" in pos

    def test_positive_contains_age(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(age=25))
        assert "25yo" in pos

    def test_positive_female_adds_1girl(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(sex="female"))
        assert "1girl" in pos

    def test_positive_male_adds_1boy(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(sex="male"))
        assert "1boy" in pos

    def test_positive_includes_race(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(race="elf"))
        assert "elf" in pos

    def test_positive_includes_hair(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(hair="silver short"))
        assert "silver short" in pos

    def test_positive_includes_eyes(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(eyes="red"))
        assert "red" in pos

    def test_positive_includes_background_color_background(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(background_color="Green"))
        assert "Green background" in pos

    def test_positive_includes_lora_prompt(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(lora_prompt="trigger_word"))
        assert "trigger_word" in pos

    def test_nsfw_false_female_includes_underwear(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(sex="female", nsfw=False))
        assert "bra" in pos or "panties" in pos

    def test_nsfw_true_female_includes_nude(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(sex="female", nsfw=True))
        assert "nude" in pos or "naked" in pos

    def test_nsfw_true_male_includes_nude(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(sex="male", nsfw=True))
        assert "nude" in pos or "naked" in pos

    def test_nsfw_string_true_treated_as_nsfw(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(nsfw="true"))
        assert "nude" in pos or "naked" in pos

    def test_negative_contains_user_negative_prompt(self):
        _, neg = CharacterCreatorV2.construct_prompt(_base_info(negative_prompt="blurry, ugly"))
        assert "blurry" in neg

    def test_no_lora_prompt_not_in_positive(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(lora_prompt=""))
        # empty lora_prompt should not add a trailing comma artifact
        assert not pos.endswith(", ")

    def test_positive_contains_cowboy_shot(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info())
        assert "cowboy_shot" in pos

    def test_empty_additional_details_not_appended(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(additional_details=""))
        # Should not produce double commas from empty field
        assert ",," not in pos

    def test_additional_details_included(self):
        pos, _ = CharacterCreatorV2.construct_prompt(_base_info(additional_details="freckles"))
        assert "freckles" in pos


# ── config save/load via process() inputs ─────────────────────────────────────

class TestProcessConfigSave:
    """Test the config-building logic in process() without model loading."""

    def test_info_fields_saved_to_config(self, tmp_path, monkeypatch):
        import utils as U
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))

        # We import inside to pick up the monkeypatch via the module reference used by character_creator_v2
        from nodes import character_creator_v2 as cc_mod
        monkeypatch.setattr(cc_mod, "base_output_dir", lambda: str(tmp_path), raising=False)

        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        info = _base_info(hair="pink curly", eyes="green")
        widget_data = json.dumps({
            "character": "TestChar",
            "character_info": info,
            "gen_settings": {
                "ckpt_name": "fake.safetensors",
                "sampler": "euler",
                "scheduler": "normal",
                "steps": 20,
                "cfg": 7.0,
                "seed": 12345,
            },
            "preview_valid": False,
            "preview_source": "gen",
        })

        # process() will fail when it tries to load a checkpoint — we only test up to config save.
        # Patch the model loading to raise before it's called by checking config was already written.
        original_load = cc_mod.comfy.sd.load_checkpoint_guess_config if hasattr(cc_mod, 'comfy') else None

        node = CharacterCreatorV2()

        # We expect a ValueError/AttributeError when checkpoint loading is attempted.
        # The config should already be written before that point.
        try:
            node.process(widget_data=widget_data)
        except Exception:
            pass

        config = utils.load_config("TestChar")
        assert config is not None, "Config was not saved before model loading"
        assert config["character_info"]["hair"] == "pink curly"
        assert config["character_info"]["eyes"] == "green"
        assert config["character_info"]["name"] == "TestChar"
        assert config["character_info"]["seed"] == 12345
        assert "costumes" in config


class TestGenerationModes:
    def test_normalize_gen_settings_uses_anima_defaults(self):
        settings = normalize_gen_settings({"generation_mode": "anima"})

        assert settings["generation_mode"] == "anima"
        assert settings["steps"] == 30
        assert settings["cfg"] == 4.0
        assert settings["sampler"] == "er_sde"
        assert settings["scheduler"] == "simple"

    def test_process_uses_anima_loader(self, tmp_path, monkeypatch):
        import utils as U
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))

        from nodes import character_creator_v2 as cc_mod
        monkeypatch.setattr(cc_mod, "base_output_dir", lambda: str(tmp_path), raising=False)

        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))

        class DummyClip:
            def tokenize(self, text):
                return text

            def encode_from_tokens(self, tokens, return_pooled=True):
                return object(), object()

        class DummyModel:
            load_device = "cpu"

        class DummyVAE:
            pass

        captured = {}

        def fake_load_generation_assets(gen_settings):
            captured.update(gen_settings)
            return ("anima", "diffusion.safetensors", "clip.safetensors", "vae.safetensors"), DummyModel(), DummyClip(), DummyVAE()

        monkeypatch.setattr(cc_mod, "load_generation_assets", fake_load_generation_assets)

        node = CharacterCreatorV2()
        widget_data = json.dumps({
            "character": "AnimaChar",
            "character_info": _base_info(),
            "gen_settings": {
                "generation_mode": "anima",
                "diffusion_model_name": "diffusion.safetensors",
                "clip_name": "clip.safetensors",
                "vae_name": "vae.safetensors",
                "seed": 123,
            },
            "preview_valid": True,
            "preview_source": "gen",
        })

        with pytest.raises(Exception):
            node.process(widget_data=widget_data)

        assert captured["generation_mode"] == "anima"
        assert captured["diffusion_model_name"] == "diffusion.safetensors"
        assert captured["clip_name"] == "clip.safetensors"
        assert captured["vae_name"] == "vae.safetensors"
        assert captured["steps"] == 30
        assert captured["cfg"] == 4.0
        assert captured["sampler"] == "er_sde"
        assert captured["scheduler"] == "simple"
