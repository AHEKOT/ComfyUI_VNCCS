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
