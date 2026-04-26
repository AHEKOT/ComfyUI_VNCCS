"""Tests for nodes/vnccs_control_center.py — pure helper functions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.vnccs_control_center import (
    _build_control_center_pipe,
    _detect_nunchaku_model_kind,
    _find_entry,
    _rel_within_folder,
    _find_model_on_disk,
    _build_dynamic_paths,
    _build_custom_lora_name,
    _merge_custom_loras,
    _remove_custom_lora,
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


# ── custom LoRA helpers ──────────────────────────────────────────────────────

class TestCustomLoraHelpers:
    def test_build_custom_lora_name_disambiguates_parent_folder(self):
        result = _build_custom_lora_name("portraits/my_style.safetensors", {"my_style"})
        assert result == "my_style (portraits)"

    def test_merge_custom_loras_appends_non_duplicate_entries(self, monkeypatch):
        monkeypatch.setattr(
            "nodes.vnccs_control_center._load_custom_loras",
            lambda: [
                {
                    "name": "custom_one",
                    "local_path": "models/loras/custom_one.safetensors",
                    "description": "Custom LoRA",
                    "custom": True,
                },
                {
                    "name": "duplicate_path",
                    "local_path": "models/loras/base.safetensors",
                    "description": "Duplicate",
                    "custom": True,
                },
            ],
        )

        merged = _merge_custom_loras({
            "lora": [
                {"name": "base", "local_path": "models/loras/base.safetensors"},
            ]
        })

        assert [entry["name"] for entry in merged["lora"]] == ["base", "custom_one"]

    def test_remove_custom_lora_by_path(self, monkeypatch):
        stored = [
            {"name": "keep", "local_path": "models/loras/keep.safetensors", "custom": True},
            {"name": "drop", "local_path": "models/loras/drop.safetensors", "custom": True},
        ]
        saved = {}

        monkeypatch.setattr("nodes.vnccs_control_center._load_custom_loras", lambda: stored)
        monkeypatch.setattr("nodes.vnccs_control_center._get_custom_loras_path", lambda: "/tmp/vnccs_custom_loras.json")
        monkeypatch.setattr("nodes.vnccs_control_center.os.makedirs", lambda *args, **kwargs: None)

        class _FakeFile:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def write(self, text):
                saved.setdefault("text", "")
                saved["text"] += text

        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _FakeFile())

        removed = _remove_custom_lora(local_path="models/loras/drop.safetensors")

        assert removed is True
        assert "drop.safetensors" not in saved["text"]
        assert "keep.safetensors" in saved["text"]


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


class TestControlCenterCustomModel:
    def test_custom_type_uses_model_input_and_standard_loader(self, monkeypatch):
        custom_model = object()
        custom_clip = object()
        custom_vae = object()

        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [],
            "clip": [{"name": "clip_a"}],
            "vae": [{"name": "vae_a"}],
            "lora": [],
        })

        def fake_load_model_block(model_entry, selected_type, type_settings, config, selected_clips, selected_vae, custom_model=None):
            assert selected_type == "custom"
            assert model_entry is None
            assert custom_model is not None
            return custom_model, custom_clip, custom_vae

        monkeypatch.setattr("nodes.vnccs_control_center._load_model_block", fake_load_model_block)
        monkeypatch.setattr(
            "nodes.vnccs_control_center._apply_loras",
            lambda model, clip, lora_states, config, model_type, **kwargs: (model, clip),
        )

        pipe = _build_control_center_pipe(
            "demo/repo",
            {"selected_type": "custom", "loras": [], "type_settings": {}, "model_params": {}},
            custom_model=custom_model,
        )

        assert pipe.model is custom_model
        assert pipe.clip is custom_clip
        assert pipe.vae is custom_vae
        assert pipe.loader_type == "standard"
        assert pipe.nunchaku_kind is None
        assert pipe.nunchaku_settings is None
        assert pipe.model_entry is None

