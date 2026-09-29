import json

import pytest

from conftest import _preload_node


pytest.importorskip("torch")

creator = _preload_node("character_creator_v2")
from nodes import character_generator as generator


class _CloneableAsset:
    def __init__(self, name):
        self.name = name
        self.clone_calls = 0

    def clone(self):
        self.clone_calls += 1
        return _CloneableAsset(f"{self.name}-clone")


def test_qi2_preview_cache_never_reuses_text_encoder(monkeypatch):
    model = _CloneableAsset("model")
    first_clip = _CloneableAsset("first-clip")
    fresh_clip = _CloneableAsset("fresh-clip")
    vae = object()
    settings = {
        "generation_mode": "qi2",
        "diffusion_model_name": "qwen-model.safetensors",
        "clip_name": "qwen-clip.safetensors",
        "vae_name": "qwen-vae.safetensors",
    }
    creator.PREVIEW_CACHE.update({"asset_key": None, "asset_obj": None})
    monkeypatch.setattr(
        creator,
        "load_generation_assets",
        lambda _settings: (("qi2", "model", "clip", "vae"), model, first_clip, vae),
    )
    fresh_loads = []

    def load_fresh(_settings):
        fresh_loads.append(True)
        return fresh_clip

    monkeypatch.setattr(creator, "load_generation_clip", load_fresh)

    _model_one, clip_one, _vae_one = creator.acquire_preview_assets(settings)
    assert clip_one is first_clip
    assert creator.PREVIEW_CACHE["asset_obj"] == (model, None, vae)
    assert first_clip.clone_calls == 0

    _model_two, clip_two, _vae_two = creator.acquire_preview_assets(settings)
    assert clip_two is fresh_clip
    assert fresh_loads == [True]
    assert fresh_clip.clone_calls == 0


def test_qi2_settings_normalize_cache_and_turbo_defaults():
    normal = creator.normalize_gen_settings({"generation_mode": "qi2"})
    assert normal["steps"] == 25
    assert normal["cfg"] == 3.0
    assert normal["clip_type"] == "qwen_image"
    assert normal["qi2_cache"] == {"device": "gpu", "dtype": "int8"}

    turbo = creator.normalize_gen_settings({
        "generation_mode": "qi2",
        "turbo_enabled": True,
        "steps": 25,
        "cfg": 3.0,
        "qi2_cache": {"device": "cpu", "dtype": "int4"},
    })
    assert turbo["steps"] == 6
    assert turbo["cfg"] == 1.0
    assert turbo["qi2_cache"] == {"device": "cpu", "dtype": "int4"}


def test_qi2_prompt_uses_text_generate_without_media_then_system_encoder(monkeypatch):
    calls = []

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "TextGenerate":
            return (json.dumps({"rewritten_prompt": "A rewritten character portrait.", "wh_ratio": "2:3"}),)
        if name == "TextEncodeQwenImage21":
            return ("positive", "negative", "encoder latent")
        raise AssertionError(name)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    positive, negative, rewritten = creator.encode_generation_conditioning(
        "clip",
        "vae",
        "anime character prompt",
        "negative prompt",
        {"generation_mode": "qi2"},
    )

    assert (positive, negative, rewritten) == (
        "positive",
        "negative",
        "A rewritten character portrait.",
    )
    assert [name for name, _ in calls] == ["TextGenerate", "TextEncodeQwenImage21"]
    text_generate = calls[0][1]
    assert "# Image Prompt Rewriting Expert" in text_generate["prompt"]
    assert text_generate["prompt"].endswith("User image request:\nanime character prompt")
    assert text_generate["max_length"] == 512
    assert text_generate["sampling_mode"] == {
        "sampling_mode": "on",
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "seed": 0,
        "presence_penalty": 0.0,
    }
    assert text_generate["thinking"] is True
    assert text_generate["use_default_template"] is True
    assert text_generate["mtp"] == "auto"
    assert not {"image", "video", "audio"}.intersection(text_generate)

    encoder = calls[1][1]
    assert encoder["prompt"] == "A rewritten character portrait."
    assert encoder["negative_prompt"] == "negative prompt"
    assert encoder["resolution"] == 1024
    assert encoder["images"] == {}


def test_qi2_prompt_rewriter_has_builtin_fallback(monkeypatch):
    monkeypatch.setattr(creator.os.path, "isfile", lambda _path: False)

    prompt = creator._qi2_prompt_rewriter_system_prompt()

    assert prompt.startswith("# Image Prompt Rewriting Expert")
    assert '"rewritten_prompt"' in prompt


def test_generate_text_initializes_preview_progress_context(monkeypatch):
    prompt_server = creator.server.PromptServer.instance
    monkeypatch.delattr(prompt_server, "last_prompt_id", raising=False)

    def fake_node(name, **_kwargs):
        assert name == "TextGenerate"
        assert prompt_server.last_prompt_id == "vnccs_character_creator_v2"
        return (json.dumps({"rewritten_prompt": "Character portrait."}),)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    assert creator.generate_qi2_prompt("clip", "character") == "Character portrait."


def test_qi2_rewriter_extracts_json_after_thinking_text():
    generated = '<think>Internal reasoning.</think>\n{"rewritten_prompt":"Final description.","wh_ratio":"2:3"}'
    assert creator._qi2_rewritten_prompt(generated, "fallback") == "Final description."


def test_qi2_rewriter_preserves_required_alpha_instruction(monkeypatch):
    def fake_node(name, **_kwargs):
        assert name == "TextGenerate"
        return (json.dumps({"rewritten_prompt": "A clean character cutout:1.0"}),)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    rewritten = creator.generate_qi2_prompt(
        "clip",
        "anime character, transparent background with alpha channel",
    )

    assert rewritten == (
        "A clean character cutout\n"
        "transparent background with alpha channel"
    )


def test_qi2_character_prompt_uses_alpha_and_natural_framing():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt({
        "sex": "female",
        "age": 24,
        "framing": "cowboy_shot",
        "background_color": "Transparent",
        "race": "human",
    }, "qi2")

    assert "transparent background with alpha channel" in positive
    assert "draw the character from the head to slightly below the waist" in positive
    assert "cowboy_shot" not in positive


def test_character_prompt_removes_unit_weights_for_every_model():
    info = {
        "sex": "female",
        "age": 24,
        "background_color": "Green",
        "race": "(elf:1.0)",
        "lora_prompt": "trigger:1.0",
    }

    for mode in ("illustrious", "anima", "qi2"):
        positive, negative = creator.CharacterCreatorV2.construct_prompt({
            **info,
            "negative_prompt": "artifact:1.0",
        }, mode)
        assert ":1.0" not in positive
        assert ":1.0" not in negative
