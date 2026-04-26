"""Tests for nodes/vnccs_control_center.py — pure helper functions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.vnccs_control_center import (
    _detect_nunchaku_model_kind,
    _find_entry,
    _rel_within_folder,
    _find_model_on_disk,
    _build_dynamic_paths,
    VNCCSPipeProxy,
)


# ── _detect_nunchaku_model_kind ───────────────────────────────────────────────

class TestDetectNunchakuModelKind:
    def test_detects_qwen_from_entry_name(self):
        entry = {"name": "Nunchaku Qwen Image Edit", "local_path": "models/unet/x.safetensors"}
        assert _detect_nunchaku_model_kind(model_entry=entry, full_path="") == "qwen-image"

    def test_detects_qwen_from_path(self):
        assert _detect_nunchaku_model_kind(model_entry=None, full_path="/models/nunchaku_qwen_image_edit.safetensors") == "qwen-image"

    def test_defaults_to_flux(self):
        entry = {"name": "Nunchaku FLUX", "local_path": "models/unet/flux.safetensors"}
        assert _detect_nunchaku_model_kind(model_entry=entry, full_path="") == "flux"

    def test_none_entry_uses_path(self):
        assert _detect_nunchaku_model_kind(model_entry=None, full_path="/models/flux_dev.safetensors") == "flux"

    def test_case_insensitive(self):
        entry = {"name": "QWEN MODEL", "local_path": ""}
        assert _detect_nunchaku_model_kind(model_entry=entry, full_path="") == "qwen-image"


# ── _find_entry ───────────────────────────────────────────────────────────────

class TestFindEntry:
    def test_finds_exact_match(self):
        entries = [{"name": "ModelA"}, {"name": "ModelB"}]
        assert _find_entry(entries, "ModelA") == {"name": "ModelA"}

    def test_case_insensitive(self):
        entries = [{"name": "ModelA"}]
        assert _find_entry(entries, "modela") is not None

    def test_strips_whitespace(self):
        entries = [{"name": " ModelA "}]
        assert _find_entry(entries, "ModelA") is not None

    def test_returns_none_when_not_found(self):
        entries = [{"name": "ModelA"}]
        assert _find_entry(entries, "ModelX") is None

    def test_empty_name_returns_none(self):
        entries = [{"name": "ModelA"}]
        assert _find_entry(entries, "") is None

    def test_empty_list_returns_none(self):
        assert _find_entry([], "ModelA") is None


# ── _rel_within_folder ────────────────────────────────────────────────────────

class TestRelWithinFolder:
    def test_standard_models_path(self):
        result = _rel_within_folder("models/checkpoints/mymodel.safetensors")
        assert result == "mymodel.safetensors"

    def test_subfolder_within_models(self):
        result = _rel_within_folder("models/loras/subdir/lora.safetensors")
        assert result == "subdir/lora.safetensors"

    def test_non_models_path_returns_basename(self):
        result = _rel_within_folder("somefile.safetensors")
        assert result == "somefile.safetensors"

    def test_backslash_normalized(self):
        result = _rel_within_folder("models\\checkpoints\\mymodel.safetensors")
        assert result == "mymodel.safetensors"


# ── _find_model_on_disk ───────────────────────────────────────────────────────

class TestFindModelOnDisk:
    def test_empty_path_returns_false(self):
        path, exists = _find_model_on_disk("")
        assert path == ""
        assert exists is False

    def test_finds_real_file(self, tmp_path, monkeypatch):
        import folder_paths as fp
        f = tmp_path / "mymodel.safetensors"
        f.write_bytes(b"data")

        monkeypatch.setattr(fp, "get_full_path", lambda key, name: str(f) if name == "mymodel.safetensors" else None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [str(tmp_path)])

        path, exists = _find_model_on_disk("models/checkpoints/mymodel.safetensors")
        assert exists is True

    def test_missing_file_falls_back_to_resolve(self, monkeypatch):
        import folder_paths as fp
        monkeypatch.setattr(fp, "get_full_path", lambda *a: None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda *a: [])

        path, exists = _find_model_on_disk("models/checkpoints/ghost.safetensors")
        assert exists is False


# ── _build_dynamic_paths ──────────────────────────────────────────────────────

class TestBuildDynamicPaths:
    def test_empty_slot_names(self):
        assert _build_dynamic_paths({}, []) == []

    def test_unknown_entry_returns_empty_string(self):
        config = {"controlnet": [], "other": []}
        result = _build_dynamic_paths(config, ["UnknownModel"])
        assert result == [""]

    def test_known_entry_found_on_disk(self, tmp_path, monkeypatch):
        import folder_paths as fp
        f = tmp_path / "ctrl.safetensors"
        f.write_bytes(b"x")

        monkeypatch.setattr(fp, "get_full_path", lambda key, name: str(f) if "ctrl" in name else None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [str(tmp_path)])

        config = {
            "controlnet": [{"name": "MyCtrl", "local_path": "models/controlnet/ctrl.safetensors"}],
            "other": [],
        }
        result = _build_dynamic_paths(config, ["MyCtrl"])
        assert result == ["ctrl.safetensors"]

    def test_known_entry_not_on_disk_returns_empty(self, monkeypatch):
        import folder_paths as fp
        monkeypatch.setattr(fp, "get_full_path", lambda *a: None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda *a: [])

        config = {
            "controlnet": [{"name": "Missing", "local_path": "models/controlnet/ghost.safetensors"}],
            "other": [],
        }
        result = _build_dynamic_paths(config, ["Missing"])
        assert result == [""]


# ── VNCCSPipeProxy ────────────────────────────────────────────────────────────

class TestVNCCSPipeProxy:
    def test_stores_model_clip_vae(self):
        m, c, v = object(), object(), object()
        proxy = VNCCSPipeProxy(m, c, v)
        assert proxy.model is m
        assert proxy.clip is c
        assert proxy.vae is v

    def test_optional_attrs_initialized(self):
        proxy = VNCCSPipeProxy(None, None, None)
        assert proxy.pos is None
        assert proxy.neg is None
        assert proxy.seed_int == 0
        assert proxy.sample_steps == 0
        assert proxy.cfg == 0.0
        assert proxy.denoise == 0.0
        assert proxy.sampler_name is None
        assert proxy.scheduler is None
        assert proxy.loader_type is None
        assert proxy.nunchaku_kind is None
        assert proxy.nunchaku_settings is None
        assert proxy.model_entry is None

