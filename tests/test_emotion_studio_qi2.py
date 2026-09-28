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


def test_emotion_studio_ui_exposes_qi2_model_cache_and_turbo_controls():
    assert 'tabQi2.innerText = "Qwen Image 2.1"' in UI_SOURCE
    assert 'tabQi2.onclick = () => setGenerationMode("qi2")' in UI_SOURCE
    assert 'qi2CacheTitle.innerText = "Qwen Image 2.1 Cache"' in UI_SOURCE
    assert '["auto", "gpu", "cpu", "off"]' in UI_SOURCE
    assert '["default", "int8", "int4"]' in UI_SOURCE
    assert 'state.gen.steps = 6;' in UI_SOURCE
    assert 'state.gen.cfg = 1.0;' in UI_SOURCE
    assert 'mode === "qi2" ? "QI2"' in UI_SOURCE


def test_qi2_emotion_card_uses_natural_prompt_without_sdxl_wrapping():
    source = (ROOT / "nodes" / "emotion_generator_v2.py").read_text(encoding="utf-8")
    assert 'if mode == "qi2":' in source
    assert 'emotion_text = str(natural_prompt or emotion_description or emotion_key).strip()' in source
