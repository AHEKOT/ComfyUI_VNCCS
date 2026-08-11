from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "web" / "vnccs_character_generator.js"
).read_text(encoding="utf-8")


def test_emotions_generator_exposes_face_denoise_slider():
    assert "face_denoise: 0.55" in SOURCE
    assert 'slider.type = "range"' in SOURCE
    assert 'this.set("emotion_generation", "face_denoise", next)' in SOURCE
    assert 'this.block("Emotion Strength", [' in SOURCE
    assert "this.faceDenoiseSlider()" in SOURCE


def test_seedvr_upscaler_exposes_resolution_controls():
    assert 'number("upscaler", "resolution", "target short edge", 16, 16384, 2)' in SOURCE
    assert 'number("upscaler", "max_resolution", "maximum edge (0 = unlimited)", 0, 16384, 2)' in SOURCE
    assert 'this.field("upscaler", "resolution", "target short edge", "number", { min: 16, max: 16384, step: 2 })' in SOURCE
    assert 'this.field("upscaler", "max_resolution", "maximum edge", "number", { min: 0, max: 16384, step: 2 })' in SOURCE


def test_seedvr_model_card_uses_persistent_widget_setter():
    assert 'this.set("upscaler", "model", rel);' in SOURCE
    assert "this.data.upscaler.model = rel;" not in SOURCE
