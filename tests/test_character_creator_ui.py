from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "web" / "vnccs_character_creator_v2.js").read_text()


def test_anima_resolution_selector_exposes_supported_presets():
    assert 'createCompactSelectField("Resolution", "resolution_preset", state.gen_settings)' in SOURCE
    assert '["normal", "Normal · 640 × 1536"]' in SOURCE
    assert '["high", "High · 856 × 2048"]' in SOURCE
    assert '["maximum", "Maximum · 1024 × 2456"]' in SOURCE


def test_anima_resolution_selector_is_mode_scoped_and_persisted():
    assert 'animaResolutionWrap.style.display = "none"' in SOURCE
    assert 'els.animaResolutionWrap.style.display = isAnima ? "flex" : "none"' in SOURCE
    assert 'anima: ["diffusion_model_name", "clip_name", "vae_name", "resolution_preset"' in SOURCE
    assert 'resolution_preset: "normal"' in SOURCE
