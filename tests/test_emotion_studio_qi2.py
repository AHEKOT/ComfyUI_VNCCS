import json
from pathlib import Path

import pytest

pytest.importorskip("torch", exc_type=ImportError)

from nodes import emotion_generator_v2 as emotion


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = (ROOT / "web" / "vnccs_emotion_v2.js").read_text(encoding="utf-8")


def test_emotion_studio_builds_qi2_pipe_with_cache_and_viggle_state(monkeypatch):
    monkeypatch.setattr(emotion, "load_anima_assets", lambda settings: ("model", "clip", "vae"))
    settings = {
        "generation_mode": "qi2",
        "mode_settings": {
            "qi2": {
                "seed": 721,
                "seed_mode": "fixed",
                "sampler": "euler",
                "scheduler": "simple",
                "diffusion_model_name": "qwen.safetensors",
                "clip_name": "qwen3vl.safetensors",
                "vae_name": "qwen_vae.safetensors",
                "turbo_enabled": True,
                "qi2_cache": {"device": "cpu", "dtype": "int4"},
            }
        },
    }

    pipe, _seed = emotion.build_emotion_pipe("QI2", json.dumps(settings))

    assert pipe.model_kind == "qi2"
    assert pipe.model_entry["kind"] == "QI2"
    assert pipe.qi2_cache == {"device": "cpu", "dtype": "int4"}
    assert pipe.sample_steps == 6
    assert pipe.cfg == 1.0
    assert pipe.lora_entries[0]["name"] == "Qwen Image 2.1 Viggle Turbo"
    assert pipe.lora_states == [{
        "name": "Qwen Image 2.1 Viggle Turbo",
        "auto_apply": True,
        "strength": 1.0,
    }]
    from nodes.character_generator import VNCCS_EmotionsGenerator

    original = vars(pipe).copy()
    values = VNCCS_EmotionsGenerator()._extract_pipe(pipe)
    assert (values["model"], values["clip"], values["vae"]) == ("model", "clip", "vae")
    assert (values["seed"], values["steps"], values["cfg"], values["denoise"]) == (721, 6, 1.0, 1.0)
    assert (values["sampler"], values["scheduler"]) == ("euler", "simple")
    assert values["model_kind"] == "qi2"
    assert values["qi2_cache"] == {"device": "cpu", "dtype": "int4"}
    assert vars(pipe) == original


def test_emotion_studio_ui_exposes_qi2_model_cache_and_turbo_controls():
    assert 'tabQi2.innerText = "Qwen Image 2.1"' in UI_SOURCE
    assert 'tabQi2.onclick = () => setGenerationMode("qi2")' in UI_SOURCE
    assert 'qi2CacheTitle.innerText = "Qwen Image 2.1 Cache"' in UI_SOURCE
    assert '["auto", "gpu", "cpu", "off"]' in UI_SOURCE
    assert '["default", "int8", "int4"]' in UI_SOURCE
    assert 'state.gen.steps = 6;' in UI_SOURCE
    assert 'state.gen.cfg = 1.0;' in UI_SOURCE
    assert 'mode === "qi2" ? "QI2"' in UI_SOURCE


@pytest.mark.parametrize("nested", [False, True])
def test_emotion_builtin_qi2_turbo_uses_standard_sampler(monkeypatch, nested):
    monkeypatch.setattr(emotion, "load_anima_assets", lambda settings: ("model", "clip", "vae"))
    profile = {
        "diffusion_model_name": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "turbo_enabled": True,
    }
    settings = {"mode_settings": {"qi2": profile}} if nested else profile
    pipe, _seed = emotion.build_emotion_pipe("QI2", json.dumps(settings))
    assert (pipe.sample_steps, pipe.cfg) == (8, 1.0)
    assert not any(state["auto_apply"] for state in pipe.lora_states)

    from nodes import character_generator as generator
    monkeypatch.setattr(generator.VNCCS_EmotionsGenerator, "_qi2_cache_model", lambda self, model, values: model)
    node = generator.VNCCS_EmotionsGenerator()
    model, turbo = node._qi2_prepare_model(pipe.model, pipe, node._extract_pipe(pipe))
    assert (model, turbo) == ("model", False)
    calls = []
    monkeypatch.setattr(generator, "_call_comfy_node", lambda name, **kwargs: calls.append((name, kwargs)) or ("latent",))
    assert node._qi2_sample(model, "positive", "negative", "empty", {
        "seed": 2, "steps": pipe.sample_steps, "cfg": pipe.cfg,
        "denoise": 1, "sampler_name": "euler", "scheduler": "simple",
    }, turbo=turbo) == "latent"
    assert [name for name, _ in calls] == ["KSampler"]

    profile.update(steps=10, cfg=1.5)
    pipe, _seed = emotion.build_emotion_pipe("QI2", json.dumps(settings))
    assert (pipe.sample_steps, pipe.cfg) == (10, 1.5)


def test_qi2_emotion_card_uses_natural_prompt_and_description_tags():
    source = (ROOT / "nodes" / "emotion_generator_v2.py").read_text(encoding="utf-8")
    assert 'if mode == "qi2":' in source
    assert """emotion_text = build_anima_emotion_prompt(
                        natural_prompt,
                        emotion_description,
                        emotion_key,
                    )""" in source


def test_emotion_prompt_combines_natural_prompt_with_description_tags():
    prompt = emotion.build_anima_emotion_prompt(
        "The character gives a warm, relaxed smile.",
        "soft smile, relaxed eyes, raised cheeks",
        "happy",
    )

    assert prompt == (
        "The character gives a warm, relaxed smile.\n\n"
        "Emotion Tags: soft smile, relaxed eyes, raised cheeks"
    )
