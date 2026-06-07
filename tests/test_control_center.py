"""Tests for nodes/vnccs_control_center.py — pure helper functions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.vnccs_control_center import (
    _build_control_center_pipe,
    _find_entry,
    _rel_within_folder,
    _find_model_on_disk,
    _resolve_model_download_path,
    _validate_downloaded_model_file,
    _validate_download_response,
    _validate_https_url,
    _apply_lora_standard,
    _filter_entries_by_kind,
    _build_dynamic_paths,
    _build_custom_lora_name,
    _dedupe_config_by_name,
    _enrich_config_entries,
    _merge_custom_loras,
    _remove_custom_lora,
    VNCCSPipeProxy,
)


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

    def test_finds_windows_style_subpath_via_folder_paths(self, tmp_path, monkeypatch):
        import folder_paths as fp
        f = tmp_path / "packs" / "mymodel.safetensors"
        f.parent.mkdir()
        f.write_bytes(b"data")

        def fake_get_full_path(key, name):
            return str(f) if name == "packs/mymodel.safetensors" else None

        monkeypatch.setattr(fp, "get_full_path", fake_get_full_path)
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [str(tmp_path)])

        path, exists = _find_model_on_disk("models\\checkpoints\\packs\\mymodel.safetensors")
        assert path == str(f)
        assert exists is True

    def test_finds_windows_style_subpath_from_folder_scan(self, tmp_path, monkeypatch):
        import folder_paths as fp
        f = tmp_path / "packs" / "mymodel.safetensors"
        f.parent.mkdir()
        f.write_bytes(b"data")

        monkeypatch.setattr(fp, "get_full_path", lambda *a: None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [str(tmp_path)])

        path, exists = _find_model_on_disk("models\\checkpoints\\packs\\mymodel.safetensors")
        assert path == str(f)
        assert exists is True

    def test_missing_file_falls_back_to_resolve(self, monkeypatch):
        import folder_paths as fp
        monkeypatch.setattr(fp, "get_full_path", lambda *a: None)
        monkeypatch.setattr(fp, "get_folder_paths", lambda *a: [])

        path, exists = _find_model_on_disk("models/checkpoints/ghost.safetensors")
        assert exists is False


class TestDownloadSafety:
    def test_https_url_required(self):
        with pytest.raises(ValueError):
            _validate_https_url("http://example.com/model.safetensors")

    def test_https_url_rejects_private_ip_literal(self):
        with pytest.raises(ValueError):
            _validate_https_url("https://127.0.0.1/model.safetensors")

    def test_download_response_rejects_oversized_content_length(self, monkeypatch):
        class Response:
            url = "https://example.com/model.safetensors"
            headers = {"content-length": str(11)}

        monkeypatch.setenv("VNCCS_MAX_MODEL_DOWNLOAD_BYTES", "10")

        with pytest.raises(ValueError):
            _validate_download_response(Response(), "model.safetensors")

    def test_resolve_model_download_path_rejects_absolute(self):
        with pytest.raises(ValueError):
            _resolve_model_download_path("/tmp/model.safetensors")

    def test_resolve_model_download_path_rejects_traversal(self):
        with pytest.raises(ValueError):
            _resolve_model_download_path("models/checkpoints/../evil.safetensors")

    def test_resolve_model_download_path_rejects_non_model_extension(self):
        with pytest.raises(ValueError):
            _resolve_model_download_path("models/checkpoints/readme.txt")

    def test_resolve_model_download_path_accepts_models_relative_path(self, tmp_path, monkeypatch):
        import folder_paths as fp
        monkeypatch.setattr(fp, "models_dir", str(tmp_path), raising=False)
        path = _resolve_model_download_path("models/checkpoints/model.safetensors")
        assert path == os.path.join(str(tmp_path), "checkpoints", "model.safetensors")

    def test_validate_downloaded_model_rejects_html(self, tmp_path):
        f = tmp_path / "fake.safetensors"
        f.write_bytes(b"<html>" + b"x" * 2048)
        with pytest.raises(ValueError):
            _validate_downloaded_model_file(str(f), "fake.safetensors")

    def test_validate_downloaded_model_accepts_gguf_magic(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF" + b"\0" * 2048)
        assert _validate_downloaded_model_file(str(f), "model.gguf") is True

    def test_apply_lora_standard_wraps_invalid_safetensors(self, tmp_path, monkeypatch):
        import comfy.utils

        f = tmp_path / "broken.safetensors"
        f.write_bytes((16).to_bytes(8, "little") + b"not-json" + b"x" * 2048)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("invalid LoRA should be rejected before torch load")

        monkeypatch.setattr(comfy.utils, "load_torch_file", fail_if_called, raising=False)

        with pytest.raises(RuntimeError, match="Failed to load LoRA 'broken.safetensors'"):
            _apply_lora_standard(object(), None, str(f), 1.0)


class TestKindFiltering:
    def test_filters_exact_kind_match(self):
        entries = [
            {"name": "QIE", "kind": "QIE2511"},
            {"name": "Anima", "kind": "Anima"},
        ]
        assert _filter_entries_by_kind(entries, "QIE2511") == [entries[0]]

    def test_uses_generic_entries_when_no_exact_match(self):
        entries = [
            {"name": "Generic"},
            {"name": "Anima", "kind": "Anima"},
        ]
        assert _filter_entries_by_kind(entries, "QIE2511") == [entries[0]]

    def test_rejects_mismatched_kinded_entries(self):
        entries = [
            {"name": "Anima", "kind": "Anima"},
        ]
        with pytest.raises(RuntimeError):
            _filter_entries_by_kind(entries, "QIE2511")


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


class TestEnrichConfigEntries:
    def test_dedupes_same_name_to_newest_version(self):
        deduped = _dedupe_config_by_name({
            "lora": [
                {
                    "name": "VNCCS Clothes Core",
                    "hf_path": "models/loras/old.safetensors",
                    "local_path": "models/loras/old.safetensors",
                    "version": "0.3.0",
                },
                {
                    "name": "VNCCS Clothes Core",
                    "hf_path": "models/loras/new.safetensors",
                    "local_path": "models/loras/new.safetensors",
                    "version": "0.3.5",
                },
            ],
        })

        assert len(deduped["lora"]) == 1
        assert deduped["lora"][0]["version"] == "0.3.5"
        assert deduped["lora"][0]["local_path"] == "models/loras/new.safetensors"

    def test_marks_installed_file_as_outdated_when_registry_version_is_old(self, monkeypatch):
        monkeypatch.setattr(
            "nodes.vnccs_control_center._find_model_on_disk",
            lambda local_path: ("/models/loras/model.safetensors", True),
        )

        result = _enrich_config_entries(
            [
                {
                    "name": "Model",
                    "local_path": "models/loras/model.safetensors",
                    "version": "0.3.5",
                }
            ],
            "lora",
            {"cc_lora_Model": "0.3.0"},
        )

        assert result[0]["status"] == "outdated"
        assert result[0]["active_version"] == "0.3.0"

    def test_unregistered_existing_file_uses_catalog_version(self, monkeypatch):
        monkeypatch.setattr(
            "nodes.vnccs_control_center._find_model_on_disk",
            lambda local_path: ("/models/loras/model.safetensors", True),
        )

        result = _enrich_config_entries(
            [
                {
                    "name": "Model",
                    "local_path": "models/loras/model.safetensors",
                    "version": "0.3.5",
                }
            ],
            "lora",
            {},
        )

        assert result[0]["status"] == "installed"
        assert result[0]["active_version"] == "0.3.5"


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
        assert proxy.sample_steps == 4
        assert proxy.cfg == 1.0
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
        context_model = {"name": "Qwen GGUF", "type": "gguf", "kind": "QIE2511"}

        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [context_model],
            "clip": [{"name": "clip_a", "kind": "QIE2511"}],
            "vae": [{"name": "vae_a", "kind": "QIE2511"}],
            "lora": [],
        })

        def fake_load_model_block(model_entry, selected_type, type_settings, config, selected_clips, selected_vae, custom_model=None):
            assert selected_type == "custom"
            assert model_entry == context_model
            assert custom_model is not None
            assert selected_clips == ["clip_a"]
            assert selected_vae == "vae_a"
            return custom_model, custom_clip, custom_vae

        monkeypatch.setattr("nodes.vnccs_control_center._load_model_block", fake_load_model_block)
        captured = {}
        def fake_apply_loras(model, clip, lora_states, config, model_type, **kwargs):
            captured["model_entry"] = kwargs.get("model_entry")
            return model, clip
        monkeypatch.setattr(
            "nodes.vnccs_control_center._apply_loras",
            fake_apply_loras,
        )

        pipe = _build_control_center_pipe(
            "demo/repo",
            {
                "selected_type": "custom",
                "selected_models": {"gguf": "Qwen GGUF"},
                "loras": [],
                "type_settings": {},
                "model_params": {},
            },
            custom_model=custom_model,
        )

        assert pipe.model is custom_model
        assert pipe.clip is custom_clip
        assert pipe.vae is custom_vae
        assert pipe.loader_type == "standard"
        assert pipe.nunchaku_kind is None
        assert pipe.nunchaku_settings is None
        assert pipe.model_entry == context_model
        assert captured["model_entry"] == context_model
        assert pipe.sample_steps == 4
        assert pipe.cfg == 1.0
