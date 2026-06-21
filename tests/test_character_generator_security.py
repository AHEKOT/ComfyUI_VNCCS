"""Security-focused tests for character generator path handling."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes import character_generator as cg


def test_character_root_ignores_external_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    external = tmp_path / "elsewhere" / "Sheets" / "Bad"
    char_root.mkdir(parents=True)
    external.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._character_root_from_sheets_path(str(external), "Alice") == str(char_root)


def test_character_root_accepts_windows_style_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    sheets = char_root / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    windows_style = str(sheets).replace(os.sep, "\\")
    assert cg._character_root_from_sheets_path(windows_style, "Alice") == str(char_root)
    assert cg._costume_name_from_sheets_path(windows_style) == "Naked"


def test_cache_tensor_path_rejects_external_cache(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    outside = tmp_path / "outside" / "cache"
    outside.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._cache_tensor_path(str(outside), "stage") == ""


def test_emotion_output_prefix_must_stay_under_character_sprites(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    safe_prefix = char_root / "Sprites" / "Happy" / "Neutral" / "sprite_"
    unsafe_prefix = tmp_path / "outside" / "sprite_"
    char_root.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._safe_emotion_output_prefix(str(safe_prefix), "Alice") == str(safe_prefix)
    assert cg._safe_emotion_output_prefix(str(unsafe_prefix), "Alice") == ""


def test_bg_remove_disabled_skips_chroma_key(monkeypatch):
    torch = pytest.importorskip("torch")

    class FailingChromaKey:
        def chroma_key(self, *args, **kwargs):
            raise AssertionError("chroma key should not run")

    monkeypatch.setattr(cg, "VNCCSChromaKey", FailingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    result = cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "disabled"},
        background="Green",
    )

    assert torch.equal(result, images)


def test_bg_remove_uses_sam3_details_recovery_by_default(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class CapturingChromaKey:
        def chroma_key(self, *args, **kwargs):
            seen["use_sam3_recovery_mask"] = args[12]
            return (args[0], None, None)

    monkeypatch.setattr(cg, "VNCCSChromaKey", CapturingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "balanced"},
        background="Green",
    )

    assert seen["use_sam3_recovery_mask"] is True


def test_bg_remove_can_disable_sam3_details_recovery(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class CapturingChromaKey:
        def chroma_key(self, *args, **kwargs):
            seen["use_sam3_recovery_mask"] = args[12]
            return (args[0], None, None)

    monkeypatch.setattr(cg, "VNCCSChromaKey", CapturingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "balanced", "use_sam3_details_recovery": False},
        background="Green",
    )

    assert seen["use_sam3_recovery_mask"] is False


def test_emotions_generator_bg_remove_uses_character_background_color(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    monkeypatch.setattr(cg, "_character_cache_dir_from_sheets_path", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(cg, "_rotate_preview_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(cg, "_save_run_inputs", lambda *args, **kwargs: None)

    node = cg.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_load_source_sprite_from_path", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(node, "_pad_alpha_sources_to_uniform_canvas", lambda data, items: (items, (4, 4)))
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda image, *args, **kwargs: (image, image))

    def capture_bg_remove(images, settings, background="Green", **kwargs):
        seen["background"] = background
        return images

    monkeypatch.setattr(node, "_run_bg_remove", capture_bg_remove)

    images = torch.rand(1, 4, 4, 3)
    emotion_data = json.dumps([{"emotion_prompt": "angry", "sprite_output_path": "", "background_color": "Green"}])
    widget_data = json.dumps({
        "character_name": "Alice",
        "bg_remove": {"preset": "balanced", "use_sam3_details_recovery": True},
    })

    node.process(images, object(), emotion_data, widget_data=widget_data, unique_id="test-node")

    assert seen["background"] == "Green"


def test_emotions_generator_single_bg_regenerate_slices_cached_raw_batch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    monkeypatch.setattr(cg, "_character_cache_dir_from_sheets_path", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(cg, "_save_run_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cg, "_load_cached_tensor", lambda *args, **kwargs: torch.zeros(4, 4, 4, 3))

    node = cg.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_load_source_sprite_from_path", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(node, "_pad_alpha_sources_to_uniform_canvas", lambda data, items: (items, (4, 4)))
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda image, *args, **kwargs: (image, image))

    cached_raw = torch.stack([
        torch.full((4, 4, 3), float(index) / 10.0)
        for index in range(4)
    ])

    def load_cached_stage(cache_dir, stage, unique_id=None, message=""):
        if stage == "emotion_0001":
            return cached_raw
        return None

    def capture_bg_remove(images, settings, background="Green", **kwargs):
        seen["shape"] = tuple(images.shape)
        seen["value"] = float(images[0, 0, 0, 0].item())
        return images

    monkeypatch.setattr(node, "_load_cached_stage", load_cached_stage)
    monkeypatch.setattr(node, "_run_bg_remove", capture_bg_remove)

    images = torch.rand(4, 4, 4, 3)
    emotion_data = [
        json.dumps({"emotion_prompt": "angry", "sprite_output_path": "same", "background_color": "Green"})
        for _ in range(4)
    ]
    widget_data = json.dumps({
        "character_name": "Alice",
        "regenerate_from": "emotion_0001_bg_remove",
        "regenerate_index": 2,
        "bg_remove": {"preset": "balanced", "use_sam3_details_recovery": True},
    })

    node.process(images, object(), emotion_data, widget_data=widget_data, unique_id="test-node")

    assert seen["shape"] == (1, 4, 4, 3)
    assert seen["value"] == pytest.approx(0.2)


def test_list_to_batch_normalizes_mixed_image_sizes():
    torch = pytest.importorskip("torch")

    small = torch.rand(1, 12, 8, 3)
    large = torch.rand(1, 24, 16, 3)

    result = cg.VNCCS_CharacterGenerator()._list_to_batch([small, large])

    assert result.shape == (2, 12, 8, 3)


def test_emotion_detailer_prompt_orders_emotion_then_face_details():
    generator = cg.VNCCS_EmotionsGenerator()

    result = generator._detailer_positive_prompt(
        "The character is furious.\n\nEmotion Tags: angry, open_mouth",
        "1girl, blue eyes, long black hair, (wear glasses on face:1.0), (wear hood on head:1.0)",
    )

    assert result.index("The character is furious.") < result.index("Character face details:")
    assert "blue eyes" in result
    assert "long black hair" in result
    assert "(wear glasses on face:1.0)" in result
    assert "(wear hood on head:1.0)" in result
    assert "The character is furious." in result
    assert "Emotion Tags: angry, open_mouth" in result
    assert "masterpiece" not in result


def test_pose_generation_decode_preserves_encoder_aspect(monkeypatch):
    torch = pytest.importorskip("torch")

    decoded = torch.rand(1, 1584, 664, 3)

    class FakeMaskExtractor:
        def fill_alpha_with_color(self, image):
            return (image,)

    class TestGenerator(cg.VNCCS_CharacterGenerator):
        def _extract_pipe(self, pipe):
            return {
                "clip": object(),
                "vae": object(),
                "model": object(),
                "seed": 1,
                "steps": 1,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
            }

        def _run_list_mapped(self, class_name, list_kwargs, **kwargs):
            if class_name == "VNCCS_QWEN_Encoder":
                return ([object()], [object()], [{"samples": torch.rand(1, 4, 198, 83)}])
            if class_name == "KSampler":
                return ([{"samples": torch.rand(1, 4, 198, 83)}],)
            if class_name == "VAEDecodeTiled":
                return ([decoded],)
            raise AssertionError(f"Unexpected node call: {class_name}")

        def _apply_pose_lora_to_model(self, model, clip, pipe, lora_info):
            return model

        def _validate_conditioning_for_model(self, pipe_values, positive, negative, stage_label):
            return None

    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", FakeMaskExtractor)

    result = TestGenerator()._run_pose_generation(
        torch.rand(1, 1536, 640, 3),
        torch.rand(1, 1536, 640, 3),
        object(),
        "prompt",
        {"target_size": 1024},
        background="Green",
    )

    assert result.shape == (1, 1584, 664, 3)


def test_seedvr_loader_cleans_vram_and_uses_settings(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class FakeModelManagement:
        def __init__(self):
            self.unloaded = 0
            self.emptied = 0

        def unload_all_models(self):
            self.unloaded += 1

        def soft_empty_cache(self):
            self.emptied += 1

    fake_mm = FakeModelManagement()
    monkeypatch.setattr(cg, "model_management", fake_mm)

    def fake_call(class_name, **kwargs):
        calls.append((class_name, kwargs))
        return (f"{class_name}_out",)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_call)

    settings = cg.VNCCS_CharacterGenerator()._settings("{}")["upscaler"]
    settings.update(
        {
            "model": "custom_dit.gguf",
            "vae": "custom_vae.safetensors",
            "offload_device": "cpu",
            "cache_dit": True,
            "cache_vae": False,
            "resolution": 4096,
            "max_resolution": 3840,
        }
    )

    generator = cg.VNCCS_CharacterGenerator()
    dit, vae = generator._run_upscaler_models(settings)
    generator._run_seedvr_upscale_one(torch.rand(1, 1584, 664, 3), dit, vae, settings, seed=42)

    assert fake_mm.unloaded == 1
    assert fake_mm.emptied == 1
    assert calls[0][0] == "SeedVR2LoadDiTModel"
    assert calls[0][1]["model"] == "custom_dit.gguf"
    assert calls[0][1]["cache_model"] is True
    assert calls[1][0] == "SeedVR2LoadVAEModel"
    assert calls[1][1]["model"] == "custom_vae.safetensors"
    assert calls[2][0] == "SeedVR2VideoUpscaler"
    assert calls[2][1]["resolution"] == 4096
    assert calls[2][1]["max_resolution"] == 3840
    assert calls[2][1]["offload_device"] == "cpu"


def test_seedvr_upscaler_runs_whole_batch_once(monkeypatch):
    torch = pytest.importorskip("torch")
    generator = cg.VNCCS_CharacterGenerator()
    calls = []

    monkeypatch.setattr(generator, "_run_upscaler_models", lambda settings: ("dit", "vae"))

    def fake_seedvr(image, dit, vae, settings, seed):
        calls.append(image)
        assert image.shape == (4, 1584, 664, 3)
        return image

    monkeypatch.setattr(generator, "_run_seedvr_upscale_one", fake_seedvr)

    images = torch.rand(4, 1584, 664, 3)
    result = generator._run_upscaler(
        images,
        "Green",
        generator._settings("{}")["upscaler"],
        seed=42,
        use_internal_rmbg=False,
    )

    assert len(calls) == 1
    assert result.shape == images.shape


def test_seedvr_attention_auto_detects_until_manual(monkeypatch):
    generator = cg.VNCCS_CharacterGenerator()
    monkeypatch.setattr(cg, "_detect_seedvr_attention_mode", lambda: "flash_attn_3")

    assert generator._resolve_seedvr_attention_mode({"attention_mode": "sdpa"}) == "flash_attn_3"
    assert generator._resolve_seedvr_attention_mode({"attention_mode": "sdpa", "attention_mode_manual": True}) == "sdpa"
    assert generator._resolve_seedvr_attention_mode({"attention_mode": "flash_attn_2", "attention_mode_manual": True}) == "flash_attn_2"
