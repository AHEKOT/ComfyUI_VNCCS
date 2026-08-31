"""Tests for nodes/vnccs_control_center.py — pure helper functions."""

import json
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
    _control_center_data_path,
    _purge_legacy_download_credentials,
    _create_download_staging_file,
    _custom_nodes_roots,
    _registered_node_class,
    _validate_downloaded_model_file,
    _max_download_bytes,
    _apply_lora_standard,
    _filter_entries_by_kind,
    _build_dynamic_paths,
    _build_custom_lora_name,
    _dedupe_config_by_name,
    _enrich_config_entries,
    _merge_custom_loras,
    _remove_custom_lora,
    _sync_packaged_cc_config,
    _get_cc_config,
    _CC_CONFIG_CACHE,
    _describe_gguf_loader,
    _load_gguf,
    _enable_manager_personal_cloud,
    _get_manager_install_policy,
    _manager_config_path,
    VNCCSPipeProxy,
)

_CONTROL_CENTER_MODULE = sys.modules[_sync_packaged_cc_config.__module__]


class TestManagerInstallPolicy:
    def test_uses_comfyui_system_manager_directory(self, monkeypatch, tmp_path):
        manager_dir = tmp_path / "manager"
        monkeypatch.setattr(
            _CONTROL_CENTER_MODULE.folder_paths,
            "get_system_user_directory",
            lambda name: str(manager_dir) if name == "manager" else None,
            raising=False,
        )

        assert _manager_config_path() == str(manager_dir / "config.ini")

    def test_nonlocal_default_policy_requires_personal_cloud(self, tmp_path):
        policy = _get_manager_install_policy(
            path=str(tmp_path / "missing.ini"),
            listen_address="0.0.0.0",
        )

        assert policy["network_mode"] == "public"
        assert policy["security_level"] == "normal"
        assert policy["security_level_allows_install"] is True
        assert policy["requires_personal_cloud"] is True
        assert policy["install_allowed"] is False

    def test_loopback_default_policy_is_allowed(self, tmp_path):
        policy = _get_manager_install_policy(
            path=str(tmp_path / "missing.ini"),
            listen_address="127.0.0.1",
        )

        assert policy["listener_is_loopback"] is True
        assert policy["install_allowed"] is True
        assert policy["requires_personal_cloud"] is False

    def test_unknown_listener_defers_to_manager_instead_of_relaxing_policy(self, tmp_path):
        policy = _get_manager_install_policy(
            path=str(tmp_path / "missing.ini"),
            listen_address="",
        )

        assert policy["listener_is_loopback"] is None
        assert policy["install_allowed"] is True
        assert policy["requires_personal_cloud"] is False

    def test_enables_personal_cloud_without_rewriting_other_settings(self, tmp_path):
        path = tmp_path / "config.ini"
        original = (
            "; keep this comment\n"
            "[default]\n"
            "security_level = normal\n"
            "network_mode = public ; keep inline comment\n"
            "channel_url = https://example.invalid/channel\n"
            "\n"
            "[other]\n"
            "network_mode = private\n"
        )
        path.write_text(original, encoding="utf-8")

        result = _enable_manager_personal_cloud(str(path))
        updated = path.read_text(encoding="utf-8")

        assert result == {"changed": True, "restart_required": True}
        assert "; keep this comment" in updated
        assert "security_level = normal" in updated
        assert "network_mode = personal_cloud\n; keep inline comment" in updated
        assert "channel_url = https://example.invalid/channel" in updated
        assert "[other]\nnetwork_mode = private" in updated
        assert (tmp_path / "config.ini.vnccs-backup").read_text(encoding="utf-8") == original
        policy = _get_manager_install_policy(str(path), listen_address="0.0.0.0")
        assert policy["network_mode"] == "personal_cloud"
        assert policy["security_level"] == "normal"
        assert policy["install_allowed"] is True

        second_result = _enable_manager_personal_cloud(str(path))
        assert second_result == {"changed": False, "restart_required": True}
        assert (tmp_path / "config.ini.vnccs-backup").read_text(encoding="utf-8") == original

    def test_creates_default_section_for_missing_config(self, tmp_path):
        path = tmp_path / "manager" / "config.ini"

        result = _enable_manager_personal_cloud(str(path))

        assert result["changed"] is True
        assert path.read_text(encoding="utf-8") == "[default]\nnetwork_mode = personal_cloud\n"

    def test_never_weakens_strong_security_level(self, tmp_path):
        path = tmp_path / "config.ini"
        original = "[default]\nsecurity_level = strong\nnetwork_mode = public\n"
        path.write_text(original, encoding="utf-8")

        with pytest.raises(RuntimeError, match="will not weaken"):
            _enable_manager_personal_cloud(str(path))

        assert path.read_text(encoding="utf-8") == original
        assert not (tmp_path / "config.ini.vnccs-backup").exists()

    def test_refuses_symlinked_manager_config(self, tmp_path):
        target = tmp_path / "real.ini"
        target.write_text("[default]\nnetwork_mode = public\n", encoding="utf-8")
        link = tmp_path / "config.ini"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable")

        with pytest.raises(RuntimeError, match="symlinked"):
            _enable_manager_personal_cloud(str(link))

        assert "network_mode = public" in target.read_text(encoding="utf-8")


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
    def test_download_limit_is_fixed_in_source(self):
        assert _max_download_bytes() == 50 * 1024 * 1024 * 1024

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
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [])
        path = _resolve_model_download_path("models/checkpoints/model.safetensors")
        assert path == os.path.join(str(tmp_path), "checkpoints", "model.safetensors")

    def test_resolve_model_download_path_uses_configured_desktop_model_folder(self, tmp_path, monkeypatch):
        import folder_paths as fp
        shared_checkpoints = tmp_path / "shared-models" / "checkpoints"
        monkeypatch.setattr(
            fp,
            "get_folder_paths",
            lambda key: [str(shared_checkpoints)] if key == "checkpoints" else [],
        )

        path = _resolve_model_download_path("models/checkpoints/packs/model.safetensors")

        assert path == os.path.join(str(shared_checkpoints), "packs", "model.safetensors")

    def test_download_staging_file_is_hidden_and_beside_target(self, tmp_path):
        target = tmp_path / "models" / "checkpoints" / "model.safetensors"
        fd, staging_path = _create_download_staging_file(str(target), "cc_models_Test Model")
        os.close(fd)
        try:
            assert os.path.dirname(staging_path) == str(target.parent)
            assert os.path.basename(staging_path).startswith(".vnccs_ccmodelsTestModel_")
            assert staging_path.endswith(".part")
        finally:
            os.unlink(staging_path)

    def test_control_center_state_writes_to_user_directory(self, tmp_path, monkeypatch):
        import folder_paths as fp
        monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path), raising=False)

        path = _control_center_data_path("vnccs_user_config.json", for_write=True)

        assert path == os.path.join(str(tmp_path), "VNCCS", "vnccs_user_config.json")

    def test_obsolete_secret_stores_are_deleted(self, tmp_path, monkeypatch):
        user_store = tmp_path / "user" / "vnccs_user_config.json"
        portable_store = tmp_path / "portable" / "vnccs_user_config.json"
        user_store.parent.mkdir()
        portable_store.parent.mkdir()
        user_store.write_text("{}", encoding="utf-8")
        portable_store.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            _CONTROL_CENTER_MODULE,
            "_control_center_data_path",
            lambda *args, **kwargs: str(user_store),
        )
        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "resolve_path", lambda *args: str(portable_store))

        _purge_legacy_download_credentials()

        assert not user_store.exists()
        assert not portable_store.exists()


class TestModuleStatusHelpers:
    def test_custom_node_roots_use_all_registered_paths(self, tmp_path, monkeypatch):
        import folder_paths as fp
        first = tmp_path / "desktop-custom-nodes"
        second = tmp_path / "extra-custom-nodes"
        monkeypatch.setattr(fp, "get_folder_paths", lambda key: [str(first), str(second), str(first)])

        assert _custom_nodes_roots() == [str(first), str(second)]

    def test_registered_node_class_ignores_unregistered_python_classes(self):
        class FaceDetailer:
            pass

        spec = {"class_names": ["FaceDetailer"]}

        assert _registered_node_class(spec, mappings={}) is None
        assert _registered_node_class(spec, mappings={"FaceDetailer": FaceDetailer}) is FaceDetailer

    def test_registered_node_class_supports_v3_node_id(self):
        class EasySam3Loader:
            pass

        spec = {
            "node_id": "easy sam3ModelLoader",
            "class_names": ["LoadSam3Model"],
        }

        assert _registered_node_class(
            spec,
            mappings={"easy sam3ModelLoader": EasySam3Loader},
        ) is EasySam3Loader

    def test_dependency_status_exposes_registry_ids_for_manager_installation(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "nodes",
            "vnccs_control_center.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert '"manager_id": "ComfyUI-GGUF"' in source
        assert '"manager_id": "comfyui-impact-pack"' in source
        assert '"manager_id": "comfyui-impact-subpack"' in source
        assert '"manager_id": "comfyui-easy-sam3"' in source
        assert '"manager_version": "latest"' in source
        assert '"darwin_compatibility_warning"' in source


class TestDownloadedModelValidation:
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


class TestPackagedConfigSync:
    def test_updates_packaged_catalog_atomically(self, tmp_path, monkeypatch):
        target = tmp_path / "control_center.json"
        target.write_text('{"name": "old"}\n', encoding="utf-8")
        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "_get_packaged_cc_path", lambda: str(target))

        updated = {
            "name": "current",
            "lora": [
                {
                    "name": "VNCCS Clothes Core",
                    "version": "0.3.7",
                    "local_path": "models/loras/qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.7.safetensors",
                }
            ],
        }

        assert _sync_packaged_cc_config("MIUProject/VNCCS_v3.0", updated) is True
        assert target.read_text(encoding="utf-8").endswith("\n")
        assert json.loads(target.read_text(encoding="utf-8")) == updated
        assert list(tmp_path.glob("control_center.json.tmp.*")) == []
        assert _sync_packaged_cc_config("MIUProject/VNCCS_v3.0", updated) is False

    def test_ignores_unrelated_repositories(self, tmp_path, monkeypatch):
        target = tmp_path / "control_center.json"
        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "_get_packaged_cc_path", lambda: str(target))

        assert _sync_packaged_cc_config("someone/else", {"name": "remote"}) is False
        assert not target.exists()

    def test_remote_config_refresh_replaces_local_copy(self, tmp_path, monkeypatch):
        target = tmp_path / "control_center.json"
        remote = tmp_path / "remote_control_center.json"
        target.write_text(json.dumps({
            "name": "old",
            "lora": [{
                "name": "VNCCS Pose Studio Klein9b",
                "version": "3.0",
                "kind": "Klein9b",
            }],
        }), encoding="utf-8")
        remote_data = {
            "name": "current",
            "models": [],
            "clip": [],
            "vae": [],
            "lora": [{
                "name": "VNCCS Pose Studio Klein9b",
                "version": "2.2",
                "kind": "Klein9b",
            }],
            "controlnet": [],
            "other": [],
        }
        remote.write_text(json.dumps(remote_data), encoding="utf-8")

        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "_get_packaged_cc_path", lambda: str(target))
        download_args = {}

        def fake_hf_download(**kwargs):
            download_args.update(kwargs)
            return str(remote)

        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "hf_hub_download", fake_hf_download)
        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "_load_custom_loras", lambda: [])
        _CC_CONFIG_CACHE.clear()

        try:
            loaded = _get_cc_config("MIUProject/VNCCS_v3.0", prefer_remote=True)
        finally:
            _CC_CONFIG_CACHE.clear()

        assert loaded["name"] == "current"
        assert loaded["lora"][0]["version"] == "2.2"
        assert json.loads(target.read_text(encoding="utf-8")) == remote_data
        assert download_args["force_download"] is True

    def test_remote_refresh_failure_does_not_return_stale_local_config(self, tmp_path, monkeypatch):
        target = tmp_path / "control_center.json"
        local_data = {"name": "stale", "lora": [{"name": "Removed LoRA", "version": "3.0"}]}
        target.write_text(json.dumps(local_data), encoding="utf-8")
        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "_get_packaged_cc_path", lambda: str(target))

        def fail_hf_download(**kwargs):
            raise RuntimeError("HF unavailable")

        monkeypatch.setattr(_CONTROL_CENTER_MODULE, "hf_hub_download", fail_hf_download)
        _CC_CONFIG_CACHE.clear()

        try:
            with pytest.raises(RuntimeError, match="HF unavailable"):
                _get_cc_config("MIUProject/VNCCS_v3.0", prefer_remote=True)
        finally:
            _CC_CONFIG_CACHE.clear()

        assert json.loads(target.read_text(encoding="utf-8")) == local_data

    def test_packaged_catalog_uses_current_clothes_core(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "control_center.json")
        with open(path, "r", encoding="utf-8") as handle:
            config = _dedupe_config_by_name(json.load(handle))

        clothes_core = next(
            entry for entry in config["lora"]
            if entry["name"] == "VNCCS Clothes Core"
        )

        assert clothes_core["version"] == "0.3.7"
        assert clothes_core["local_path"].endswith("VNCCS_QIE2511_ClothesCore-RC3.7.safetensors")
        assert all(entry["name"] != "VNCCS Emotion Core" for entry in config["lora"])

    def test_packaged_catalog_contains_complete_klein_family(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "control_center.json")
        with open(path, "r", encoding="utf-8") as handle:
            config = _dedupe_config_by_name(json.load(handle))

        klein_models = [entry for entry in config["models"] if entry.get("kind") == "Klein9b"]
        klein_clips = [entry for entry in config["clip"] if entry.get("kind") == "Klein9b"]
        klein_vaes = [entry for entry in config["vae"] if entry.get("kind") == "Klein9b"]
        klein_loras = [entry for entry in config["lora"] if entry.get("kind") == "Klein9b"]

        assert [entry["hf_path"] for entry in klein_models] == ["flux-2-klein-9b-fp8.safetensors"]
        assert [entry["hf_repo"] for entry in klein_models] == ["MIUProject/FLUX.2-klein-9b-fp8"]
        assert [entry["clip_type"] for entry in klein_clips] == ["flux2"]
        assert [entry["hf_repo"] for entry in klein_clips] == [
            "Comfy-Org/vae-text-encorder-for-flux-klein-9b"
        ]
        assert [entry["hf_path"] for entry in klein_clips] == [
            "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
        ]
        assert [entry["hf_repo"] for entry in klein_vaes] == [
            "Comfy-Org/vae-text-encorder-for-flux-klein-9b"
        ]
        assert [entry["hf_path"] for entry in klein_vaes] == [
            "split_files/vae/flux2-vae.safetensors"
        ]
        assert [entry["local_path"] for entry in klein_vaes] == ["models/vae/flux2-vae.safetensors"]
        assert {entry["type"] for entry in klein_loras} == {"Helper"}
        assert {entry["name"] for entry in klein_loras} == {
            "VNCCS Clothes Core Klein9b",
            "VNCCS Pose Studio Klein9b",
        }


class TestControlCenterFamilyState:
    def test_builds_klein_pipe_from_family_scoped_state(self, monkeypatch):
        model = object()
        clip = object()
        vae = object()
        model_entry = {"name": "Flux Klein 9B FP8", "type": "unet", "kind": "Klein9b"}
        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [model_entry],
            "clip": [{"name": "klein_clip", "kind": "Klein9b"}],
            "vae": [{"name": "klein_vae", "kind": "Klein9b"}],
            "lora": [],
        })
        captured = {}

        def fake_load_model_block(entry, selected_type, settings, config, clips, vae_name, **kwargs):
            captured.update(entry=entry, selected_type=selected_type, clips=clips, vae_name=vae_name)
            return model, clip, vae

        monkeypatch.setattr("nodes.vnccs_control_center._load_model_block", fake_load_model_block)
        monkeypatch.setattr("nodes.vnccs_control_center._apply_loras", lambda model, clip, *args, **kwargs: (model, clip))

        pipe = _build_control_center_pipe("demo/repo", {
            "active_kind": "Klein9b",
            "selected_types_by_kind": {"QIE2511": "gguf", "Klein9b": "unet"},
            "selected_models": {"Klein9b:unet": "Flux Klein 9B FP8"},
            "model_params_by_kind": {
                "QIE2511": {"steps": 8, "cfg": 2},
                "Klein9b": {"steps": 4, "cfg": 1, "sampler": "euler", "scheduler": "simple"},
            },
        })

        assert captured == {
            "entry": model_entry,
            "selected_type": "unet",
            "clips": ["klein_clip"],
            "vae_name": "klein_vae",
        }
        assert pipe.model_entry == model_entry
        assert pipe.sample_steps == 4
        assert pipe.cfg == 1.0
        assert pipe.sampler_name == "euler"

    def test_custom_klein_pipe_keeps_klein_model_context(self, monkeypatch):
        custom_model = object()
        custom_clip = object()
        custom_vae = object()
        qie_entry = {"name": "Qwen GGUF", "type": "gguf", "kind": "QIE2511"}
        klein_entry = {"name": "Flux Klein 9B FP8", "type": "unet", "kind": "Klein9b"}
        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [qie_entry, klein_entry],
            "clip": [],
            "vae": [],
            "lora": [],
        })
        captured = {}

        def fake_load_model_block(entry, selected_type, *args, **kwargs):
            captured.update(entry=entry, selected_type=selected_type)
            return kwargs["custom_model"], kwargs["custom_clip"], kwargs["custom_vae"]

        monkeypatch.setattr("nodes.vnccs_control_center._load_model_block", fake_load_model_block)
        monkeypatch.setattr("nodes.vnccs_control_center._apply_loras", lambda model, clip, *args, **kwargs: (model, clip))

        pipe = _build_control_center_pipe(
            "demo/repo",
            {
                "active_kind": "Klein9b",
                "selected_types_by_kind": {"Klein9b": "custom"},
                "selected_models": {"Klein9b:unet": "Flux Klein 9B FP8"},
                "model_params_by_kind": {"Klein9b": {"steps": 4, "cfg": 1}},
            },
            custom_model=custom_model,
            custom_clip=custom_clip,
            custom_vae=custom_vae,
        )

        assert captured == {"entry": klein_entry, "selected_type": "custom"}
        assert pipe.model_entry == klein_entry


class TestControlCenterFrontendFamilies:
    def test_frontend_has_family_tabs_and_kind_filters(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "vnccs_control_center.js")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert '{ kind: "QIE2511", label: "QIE2511", defaultType: "gguf" }' in source
        assert '{ kind: "Klein9b", label: "Flux Klein9b", defaultType: "unet" }' in source
        assert 'activeKind === "Klein9b" ? ["unet", "custom"]' in source
        assert 'const contextType = this._familyDefinition().defaultType;' in source
        assert '.vnccs-cc-twocol-left > .vnccs-cc-model-card' in source
        assert 'this.scrollArea.appendChild(this._renderFamilyTabs())' in source
        assert "_exactKind(entry, kind = this._selectedKind())" in source
        assert "entryKind && kind && entryKind.toLowerCase() === kind.toLowerCase()" in source
        assert "!entry.custom && !this._isTurboLora(entry) && this._exactKind(entry, selectedKind)" in source
        assert "if (!this._exactKind(entry, selectedKind)) continue;" in source
        assert "(this._isHelperLora(entry) || this._sameKind(entry, selectedKind))" not in source

    def test_download_errors_are_visible_to_desktop_users(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "vnccs_control_center.js")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        download_source = source.split("async _downloadEntry(cat, entry)", 1)[1].split(
            "async _downloadAllMissing()", 1
        )[0]
        assert "if (!r.ok || d.error)" in download_source
        assert "this.showMessage(message, true);" in download_source
        assert "detail: { repo_id: repoId }" in download_source

    def test_missing_dependencies_install_through_comfyui_manager(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "vnccs_control_center.js")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert 'api.fetchApi("/manager/queue/install"' in source
        assert 'api.fetchApi("/manager/queue/start"' in source
        assert 'api.fetchApi("/v2/manager/queue/task"' in source
        assert 'api.fetchApi("/v2/manager/queue/start"' in source
        assert 'api.addEventListener("cm-task-completed"' in source
        assert 'api.addEventListener("cm-queue-status"' in source
        assert 'api.fetchApi("/manager/reboot"' in source
        assert 'api.fetchApi("/v2/manager/reboot"' in source
        assert 'api.fetchApi("/vnccs/manager/install_policy"' in source
        assert 'api.fetchApi("/vnccs/manager/enable_personal_cloud"' in source
        assert 'confirmation: "enable_personal_cloud"' in source
        assert '"X-VNCCS-CSRF": "1"' in source
        assert 'sessionStorage.setItem(PENDING_DEPENDENCY_INSTALLS_KEY' in source
        assert 'sessionStorage.removeItem(PENDING_DEPENDENCY_INSTALLS_KEY)' in source
        assert "window.location.reload();" in source
        assert 'this._btn("Enable & restart"' in source
        assert "security_level will not be changed" in source
        assert 'selected_version: "latest"' in source
        assert 'kind: "install"' in source
        assert "skip_post_install: false" in source
        assert 'this._btn("Install all"' in source
        assert 'this._btn("Restart server"' in source
        assert "this._dependencyRestartRequired && this._dependencyInstallTasks.size === 0" in source
        assert 'info.status === "unsupported"' in source
        assert 'info.status !== "unsupported"' in source
        assert source.index('api.fetchApi("/manager/queue/install"') < source.index(
            'api.fetchApi("/v2/manager/queue/task"'
        )
        queue_source = source.split("async _queueDependencyInstall", 1)[1].split(
            "async _installDependency", 1
        )[0]
        assert queue_source.index("this._dependencyInstallTasks.set(uiId, tracked)") < queue_source.index(
            "await this._queueLegacyDependencyInstall(item, uiId)"
        )
        assert "this._dependencyInstallTasks.delete(uiId);" in queue_source
        assert "git clone" not in source.split("async _queueDependencyInstall", 1)[1].split(
            "_handleManagerTaskCompleted", 1
        )[0]

    def test_custom_model_inputs_follow_the_active_custom_tab(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "vnccs_control_center.js")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        sync_source = source.split("_syncCustomModelInput()", 1)[1].split("_setSelectedType", 1)[0]
        assert "this.config ? this._getSelectedType() : this._getStoredSelectedType()" in sync_source
        assert 'const isCustom = selectedType === "custom"' in sync_source
        assert 'sync("model", "MODEL"' in sync_source
        assert 'sync("clip", "CLIP"' in sync_source
        assert 'sync("vae", "VAE"' in sync_source
        assert source.count("this._syncCustomModelInput();") >= 7

class TestClothesPreviewFrontendContract:
    def test_custom_preview_uses_partial_graph_execution(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "web",
            "vnccs_clothes_designer.js",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert 'controlCenter.selected_type === "custom"' in source
        assert "app.queuePrompt(0, 1, [targetId])" in source
        assert 'api.addEventListener("vnccs.preview.updated", onPreview)' in source
        assert 'api.addEventListener("execution_cached", onCached)' in source
        assert "cachedNodes.some(nodeId => String(nodeId) === targetId)" in source

    def test_clothes_designer_is_partial_execution_output(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "nodes",
            "clothes_designer.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        class_source = source.split("class ClothesDesigner:", 1)[1]
        assert "OUTPUT_NODE = True" in class_source

    def test_generated_preview_preserves_cache_input_signature(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "web",
            "vnccs_clothes_designer.js",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        assert 'if (url.includes("force_cache=true")) return;' in source
        force_cache_branch = source.split("if (forceCache) {", 1)[1].split("} else {", 1)[0]
        assert "selected_preview_sprite = null" not in force_cache_branch
        custom_preview_branch = source.split(
            'if (controlCenter.selected_type === "custom") {',
            1,
        )[1].split("} else {", 1)[0]
        assert "if (previewResult?.cached)" in custom_preview_branch
        assert custom_preview_branch.count("updatePreviewImage(true)") == 1


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
        monkeypatch.setattr("nodes.vnccs_control_center._get_custom_loras_path", lambda *args, **kwargs: "/tmp/vnccs_custom_loras.json")
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
    def test_custom_type_uses_external_model_clip_and_vae_inputs(self, monkeypatch):
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

        def fake_load_model_block(
            model_entry,
            selected_type,
            type_settings,
            config,
            selected_clips,
            selected_vae,
            custom_model=None,
            custom_clip=None,
            custom_vae=None,
        ):
            assert selected_type == "custom"
            assert model_entry == context_model
            assert custom_model is not None
            assert custom_clip is not None
            assert custom_vae is not None
            assert selected_clips == []
            assert selected_vae == ""
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
            custom_clip=custom_clip,
            custom_vae=custom_vae,
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
        assert pipe.scheduler == "simple"

    def test_custom_type_requires_external_clip_and_vae_inputs(self, monkeypatch):
        custom_model = object()
        custom_clip = object()
        context_model = {"name": "Qwen GGUF", "type": "gguf", "kind": "QIE2511"}

        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [context_model],
            "clip": [{"name": "clip_a", "kind": "QIE2511"}],
            "vae": [{"name": "vae_a", "kind": "QIE2511"}],
            "lora": [],
        })

        base_state = {
            "selected_type": "custom",
            "selected_models": {"gguf": "Qwen GGUF"},
            "loras": [],
            "type_settings": {},
            "model_params": {},
        }

        with pytest.raises(RuntimeError, match="Custom CLIP input is not connected"):
            _build_control_center_pipe(
                "demo/repo",
                base_state,
                custom_model=custom_model,
            )

        with pytest.raises(RuntimeError, match="Custom VAE input is not connected"):
            _build_control_center_pipe(
                "demo/repo",
                base_state,
                custom_model=custom_model,
                custom_clip=custom_clip,
            )


class TestControlCenterRequiredTurboLora:
    def test_qwen_four_step_cfg_one_forces_lightning_lora_for_process(self, monkeypatch):
        model = object()
        clip = object()
        vae = object()
        model_entry = {"name": "Qwen-Image-Edit-2511-GGUF-Q5", "type": "gguf", "kind": "QIE2511"}
        lightning_entry = {
            "name": "Qwen Image Edit 2511 Lightning",
            "type": "TurboLora",
            "kind": "QIE2511",
            "local_path": "models/loras/qwen/Qwen-Image-Edit-2511-Lightning.safetensors",
        }

        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [model_entry],
            "clip": [{"name": "clip_a", "kind": "QIE2511"}],
            "vae": [{"name": "vae_a", "kind": "QIE2511"}],
            "lora": [lightning_entry],
        })
        monkeypatch.setattr(
            "nodes.vnccs_control_center._load_model_block",
            lambda *args, **kwargs: (model, clip, vae),
        )
        captured = {}

        def fake_apply_loras(model_arg, clip_arg, lora_states, config, model_type, **kwargs):
            captured["lora_states"] = lora_states
            return model_arg, clip_arg

        monkeypatch.setattr("nodes.vnccs_control_center._apply_loras", fake_apply_loras)

        pipe = _build_control_center_pipe(
            "demo/repo",
            {
                "selected_type": "gguf",
                "selected_model": "Qwen-Image-Edit-2511-GGUF-Q5",
                "loras": [],
                "model_params": {"steps": 4, "cfg": 1},
            },
        )

        assert pipe.model is model
        assert captured["lora_states"] == [
            {"name": "Qwen Image Edit 2511 Lightning", "auto_apply": True, "strength": 1.0}
        ]
        assert pipe.lora_states == captured["lora_states"]

    def test_qwen_non_four_step_does_not_force_lightning_lora(self, monkeypatch):
        model = object()
        clip = object()
        vae = object()
        model_entry = {"name": "Qwen-Image-Edit-2511-GGUF-Q5", "type": "gguf", "kind": "QIE2511"}
        lightning_entry = {
            "name": "Qwen Image Edit 2511 Lightning",
            "type": "TurboLora",
            "kind": "QIE2511",
            "local_path": "models/loras/qwen/Qwen-Image-Edit-2511-Lightning.safetensors",
        }

        monkeypatch.setattr("nodes.vnccs_control_center._get_cc_config", lambda repo_id: {
            "models": [model_entry],
            "clip": [{"name": "clip_a", "kind": "QIE2511"}],
            "vae": [{"name": "vae_a", "kind": "QIE2511"}],
            "lora": [lightning_entry],
        })
        monkeypatch.setattr(
            "nodes.vnccs_control_center._load_model_block",
            lambda *args, **kwargs: (model, clip, vae),
        )
        captured = {}

        def fake_apply_loras(model_arg, clip_arg, lora_states, config, model_type, **kwargs):
            captured["lora_states"] = lora_states
            return model_arg, clip_arg

        monkeypatch.setattr("nodes.vnccs_control_center._apply_loras", fake_apply_loras)

        _build_control_center_pipe(
            "demo/repo",
            {
                "selected_type": "gguf",
                "selected_model": "Qwen-Image-Edit-2511-GGUF-Q5",
                "loras": [],
                "model_params": {"steps": 8, "cfg": 1},
            },
        )

        assert captured["lora_states"] == []


class TestGGUFLoaderDiagnostics:
    def test_describes_classic_loader_without_warning(self, monkeypatch):
        class OfficialLoader:
            pass

        monkeypatch.setattr(
            "nodes.vnccs_control_center.inspect.getfile",
            lambda cls: "/tmp/ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py",
        )

        info = _describe_gguf_loader(OfficialLoader)

        assert info["available"] is True
        assert info["is_classic"] is True
        assert info["folder"] == "ComfyUI-GGUF"
        assert info["warning"] is None

    def test_describes_forked_loader_with_warning(self, monkeypatch):
        class ForkedLoader:
            pass

        monkeypatch.setattr(
            "nodes.vnccs_control_center.inspect.getfile",
            lambda cls: "/tmp/ComfyUI/custom_nodes/ComfyUI-GGUF_Forked/nodes.py",
        )

        info = _describe_gguf_loader(ForkedLoader)

        assert info["available"] is True
        assert info["is_classic"] is False
        assert info["folder"] == "ComfyUI-GGUF_Forked"
        assert "non-standard GGUF loader" in info["warning"]

    def test_load_gguf_rewrites_qwen_image_architecture_error(self, monkeypatch):
        import nodes as comfy_nodes

        class ForkedLoader:
            def load_unet(self, _name):
                raise ValueError(
                    "Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder "
                    "but got 'qwen_image'"
                )

        monkeypatch.setattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {"UnetLoaderGGUF": ForkedLoader}, raising=False)
        monkeypatch.setattr(
            "nodes.vnccs_control_center.inspect.getfile",
            lambda cls: "/tmp/ComfyUI/custom_nodes/ComfyUI-GGUF_Forked/nodes.py",
        )

        with pytest.raises(RuntimeError, match="does not support Qwen Image GGUF"):
            _load_gguf("/tmp/Qwen-Image.gguf")
