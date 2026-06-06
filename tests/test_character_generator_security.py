"""Security-focused tests for character generator path handling."""

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


def test_list_to_batch_normalizes_mixed_image_sizes():
    torch = pytest.importorskip("torch")

    small = torch.rand(1, 12, 8, 3)
    large = torch.rand(1, 24, 16, 3)

    result = cg.VNCCS_CharacterGenerator()._list_to_batch([small, large])

    assert result.shape == (2, 12, 8, 3)


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
        {"target_size": "1024"},
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
