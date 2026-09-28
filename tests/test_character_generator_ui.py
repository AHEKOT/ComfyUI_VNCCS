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


def test_sam_defaults_are_disabled_and_native_hides_recovery_controls():
    assert "use_sam: false" in SOURCE
    assert "use_sam3_details_recovery: false" in SOURCE
    assert "if (!this.isNativeBgRemove())" in SOURCE
    assert 'this.block("BG Remove", this.bgRemoveFields())' in SOURCE


def test_seedvr_upscaler_exposes_resolution_controls():
    assert 'number("upscaler", "resolution", "target short edge", 16, 16384, 2)' in SOURCE
    assert 'number("upscaler", "max_resolution", "maximum edge (0 = unlimited)", 0, 16384, 2)' in SOURCE
    assert 'this.field("upscaler", "resolution", "target short edge", "number", { min: 16, max: 16384, step: 2 })' in SOURCE
    assert 'this.field("upscaler", "max_resolution", "maximum edge", "number", { min: 0, max: 16384, step: 2 })' in SOURCE


def test_pose_resolution_control_uses_clear_label():
    assert 'caption.textContent = "resolution scale"' in SOURCE
    assert 'slider.type = "range"' in SOURCE
    assert "RESOLUTION_SCALE_MIN_MP = 1" in SOURCE
    assert "RESOLUTION_SCALE_MAX_MP = 4" in SOURCE
    assert "RESOLUTION_SCALE_STEP_MP = 0.1" in SOURCE
    assert "[1.3, 1344]" in SOURCE
    assert "[1.5, 1536]" in SOURCE
    assert "resolutionScaleValue(slider.value)" in SOURCE
    assert '"target_size", "scale area", "select"' not in SOURCE


def test_seedvr_model_card_uses_persistent_widget_setter():
    assert 'this.set("upscaler", "model", rel);' in SOURCE
    assert "this.data.upscaler.model = rel;" not in SOURCE
