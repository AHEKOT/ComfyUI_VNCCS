"""Control Center path and registry preservation without model dependencies."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import ntpath
import queue
import threading
from types import SimpleNamespace

import pytest
import utils

from nodes import vnccs_control_center as cc


@pytest.mark.parametrize("category,folder,filename", [
    ("models", "diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"),
    ("clip", "text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),
    ("vae", "vae", "qwen_image_2.1_vae_bf16.safetensors"),
])
def test_catalog_finds_installed_models_after_another_windows_drive(
    monkeypatch, category, folder, filename,
):
    preferred = ntpath.join("E:\\", "models", folder)
    local = ntpath.join("F:\\", "ComfyUI", "models", folder)
    model = ntpath.join(local, filename)
    windows_os = SimpleNamespace(path=SimpleNamespace(**vars(ntpath)), sep="\\")
    windows_os.path.exists = windows_os.path.isfile = lambda path: ntpath.normcase(path) == ntpath.normcase(model)
    monkeypatch.setattr(utils, "os", windows_os)
    monkeypatch.setattr(cc, "os", windows_os)
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [preferred, local])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda key, name: model if name == filename else None)
    entry = {"name": "Installed model", "local_path": f"models/{folder}/{filename}"}

    assert cc._find_model_on_disk(entry["local_path"]) == (model, True)
    assert cc._enrich_config_entries([entry], category)[0]["status"] == "installed"
    assert not utils._is_under_any_model_folder(ntpath.join("G:\\", filename), [preferred, local])


@pytest.mark.parametrize("shadow", [False, True])
@pytest.mark.parametrize("linked_root", [False, True])
def test_gguf_loader_receives_the_selected_nested_registered_model(monkeypatch, tmp_path, shadow, linked_root):
    real_root = tmp_path / "models"
    (real_root / "pack").mkdir(parents=True)
    target = real_root / "pack" / "model.gguf"
    target.write_bytes(b"selected nested weights")
    if shadow:
        (real_root / target.name).write_bytes(b"unselected root weights")
    root = tmp_path / "linked-models" if linked_root else real_root
    if linked_root:
        root.symlink_to(real_root, target_is_directory=True)
    unrelated = tmp_path / "other-models"
    unrelated.mkdir()
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [str(unrelated), str(root)]
                        if key in {"unet", "diffusion_models"} else [])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda *args: None)
    received = []

    class Loader:
        def load_unet(self, name):
            received.append(name)
            path = cc.get_full_path_agnostic(cc.folder_paths, "unet", name, require_exists=True)
            if not path:
                raise FileNotFoundError(name)
            with open(path, "rb") as handle:
                return (handle.read(),)

    monkeypatch.setattr(cc, "_get_gguf_loader_class", lambda: Loader)
    monkeypatch.setattr(cc, "_describe_gguf_loader", lambda *args: {})
    monkeypatch.setattr(cc, "_MODEL_ASSET_CACHE", {})
    path, found = cc._find_model_on_disk("models/unet/pack/model.gguf")
    assert found
    assert cc._load_gguf(path) == target.read_bytes()
    assert received == [str(target.relative_to(real_root))]


def test_slower_catalog_refresh_cannot_replace_newer_cache_or_packaged_catalog(monkeypatch, tmp_path):
    packaged = tmp_path / "packaged.json"
    packaged.write_text('{"name": "Packaged"}')
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    old_path.write_text('{"name": "Old"}')
    new_path.write_text('{"name": "New"}')
    monkeypatch.setattr(cc, "_get_packaged_cc_path", lambda: str(packaged))
    monkeypatch.setattr(cc, "_load_custom_loras", lambda: [])
    monkeypatch.setattr(cc, "_CC_CONFIG_CACHE", {})
    old_started, release_old = threading.Event(), threading.Event()

    def download(**kwargs):
        if not old_started.is_set():
            old_started.set()
            assert release_old.wait(5), "Old request was not released"
            return str(old_path)
        return str(new_path)

    monkeypatch.setattr(cc, "hf_hub_download", download)
    repo_id = "MIUProject/VNCCS_v3.0"
    with ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(cc._get_cc_config, repo_id, True)
        try:
            assert old_started.wait(5), "Old request did not start"
            current = executor.submit(cc._get_cc_config, repo_id, True).result(timeout=5)
            assert current["name"] == "New"
        finally:
            release_old.set()
        old.result(timeout=5)
    assert cc._get_cc_config(repo_id)["name"] == "New"
    assert json.loads(packaged.read_text())["name"] == "New"


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_catalog_lookup_rejects_traversal_before_basename_fallback(monkeypatch, tmp_path, separator):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "private.safetensors"
    outside.write_bytes(b"private weights")
    (root / outside.name).write_bytes(b"different weights")
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [str(root)])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda *args: None)
    path = separator.join(["models", "unet", "..", outside.name])
    assert cc._find_model_on_disk(path) == ("", False)


def test_model_lookup_and_download_reject_nested_symlink_escape(monkeypatch, tmp_path):
    root, outside = tmp_path / "models", tmp_path / "private"
    root.mkdir()
    outside.mkdir()
    target = outside / "model.safetensors"
    target.write_bytes(b"private weights")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [str(root)])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda *args: None)
    path = "models/unet/escape/model.safetensors"
    assert cc._find_model_on_disk(path) == ("", False)
    with pytest.raises(ValueError, match="outside allowed directory"):
        cc._resolve_model_download_path(path)
    assert target.read_bytes() == b"private weights"


def test_registered_symlink_root_keeps_nested_model_lookup_and_download(monkeypatch, tmp_path):
    real, root = tmp_path / "real-models", tmp_path / "registered-models"
    (real / "pack").mkdir(parents=True)
    root.symlink_to(real, target_is_directory=True)
    target = root / "pack" / "model.safetensors"
    target.write_bytes(b"weights")
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [str(root)])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda *args: None)
    path = "models/unet/pack/model.safetensors"
    assert cc._find_model_on_disk(path) == (str(target), True)
    assert cc._resolve_model_download_path(path) == str(target)


@pytest.mark.parametrize("content", [b'{"existing": "1.0",', b'["existing"]'])
def test_invalid_installed_registry_is_preserved(monkeypatch, tmp_path, content):
    registry = tmp_path / "installed.json"
    registry.write_bytes(content)
    monkeypatch.setattr(cc, "_control_center_data_path", lambda *args, **kwargs: str(registry))
    with pytest.raises(ValueError):
        cc.update_installed_version("new", "2.0")
    assert registry.read_bytes() == content


def test_failed_registry_publish_preserves_existing_versions(monkeypatch, tmp_path):
    registry = tmp_path / "installed.json"
    registry.write_text('{"existing": "1.0"}')
    original = registry.read_bytes()
    monkeypatch.setattr(cc, "_control_center_data_path", lambda *args, **kwargs: str(registry))
    def fail_replace(*args):
        raise OSError("Disk unavailable")
    monkeypatch.setattr(cc.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Disk unavailable"):
        cc.update_installed_version("new", "2.0")
    assert registry.read_bytes() == original
    assert list(tmp_path.iterdir()) == [registry]


def test_concurrent_registry_updates_retain_all_versions(monkeypatch, tmp_path):
    registry = tmp_path / "installed.json"
    monkeypatch.setattr(cc, "_control_center_data_path", lambda *args, **kwargs: str(registry))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda number: cc.update_installed_version(str(number), "1.0"), range(12)))
    assert json.loads(registry.read_text()) == {str(number): "1.0" for number in range(12)}


@pytest.mark.parametrize("state", ['{"selected_model":', '[]', 'null'])
def test_invalid_workflow_state_does_not_load_default_models(monkeypatch, state):
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args: pytest.fail("Invalid state must stop before loading models"))
    with pytest.raises(RuntimeError, match="Invalid node_state"):
        cc._build_control_center_pipe("test/repo", state)


@pytest.mark.parametrize("state", [
    {"selected_models": []}, {"selected_types_by_kind": {"QI2": []}},
    {"model_params": ["authored"]}, {"model_params_by_kind": {"QI2": ["authored"]}},
    {"type_settings": {"unet": "authored"}}, {"loras": [None]},
    {"qi2_cache": []}, {"collapsed": []},
    {"model_params": {"steps": []}}, {"model_params": {"cfg": {}}},
    {"model_params_by_kind": {"QI2": {"steps": "broken"}}},
    {"model_params": {"sampler": []}}, {"model_params": {"scheduler": {}}},
    {"model_params": {"cfg": "nan"}}, {"model_params": {"steps": float("inf")}},
    {"model_params": {"steps": "2.5"}}, {"model_params": {"steps": "1e2"}},
    {"model_params": {"steps": "0x10"}}, {"model_params": {"steps": "1__0"}},
    {"model_params": {"cfg": "0x10"}}, {"model_params": {"cfg": "1__0"}},
    {"loras": [{"name": "Style", "strength": "0x10"}]},
    {"qi2_cache": {"device": []}}, {"qi2_cache": {"dtype": {}}},
    {"loras": [{"name": "Style", "auto_apply": True, "strength": []}]},
])
def test_invalid_nested_workflow_state_stops_before_catalog_or_models(monkeypatch, state):
    original = json.dumps(state)
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args: pytest.fail("Invalid state must stop before loading models"))
    with pytest.raises(RuntimeError, match="Invalid node_state"):
        cc._build_control_center_pipe("test/repo", state)
    assert json.dumps(state) == original


def test_numeric_strings_and_null_parameter_fallbacks_remain_valid():
    state = {"model_params": {"steps": "25", "cfg": "3.5", "sampler": None},
             "model_params_by_kind": {"QI2": {"steps": None, "cfg": None, "scheduler": None}},
             "qi2_cache": {"device": None, "dtype": None}, "loras": [{"strength": "-0.5"}]}
    assert cc._parse_control_center_state(state) == state


def test_decimal_strings_and_boolean_parameters_remain_valid_without_mutating_state():
    state = {"model_params": {"steps": " +2_5 ", "cfg": "3_0.5e-1"},
             "model_params_by_kind": {"QI2": {"steps": True, "cfg": ".5"}},
             "loras": [{"strength": "-1_0.5e-1"}]}
    original = json.dumps(state)
    assert cc._parse_control_center_state(state) == state
    assert json.dumps(state) == original


@pytest.mark.parametrize("kind,steps,cfg,expected_steps,expected_cfg", [
    ("QI2", " +2_5 ", "3_0.0e-1", 25, 3),
    ("QI2", True, True, 1, 1),
    ("QI2", None, None, 25, 3),
    ("QI2", False, False, 25, 0),
    ("QI2", "", 0, 25, 0),
    ("QI2", 2.9, " +.5 ", 2, 0.5),
    ("Klein9b", None, None, 4, 1),
    ("MiniMaxH3", None, None, 8, 1),
])
def test_legacy_sampling_display_contract_matches_custom_pipe(monkeypatch, kind, steps, cfg, expected_steps, expected_cfg):
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [], "lora": []})
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    state = {"active_kind": kind, "selected_type": "custom", "model_params": {"steps": steps, "cfg": cfg}}
    original = json.dumps(state)
    pipe = cc._build_control_center_pipe("test/repo", state, object(), object(), object(), object())
    assert (pipe.sample_steps, pipe.cfg) == (expected_steps, expected_cfg)
    assert json.dumps(state) == original


@pytest.mark.parametrize("kind", ["QI2", "Klein9b", "MiniMaxH3"])
@pytest.mark.parametrize("empty", [None, ""])
def test_legacy_empty_sampler_settings_keep_backend_defaults(monkeypatch, kind, empty):
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [], "lora": []})
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    state = {"active_kind": kind, "selected_type": "custom", "model_params": {"sampler": empty, "scheduler": empty}}
    original = json.dumps(state)
    pipe = cc._build_control_center_pipe("test/repo", state, object(), object(), object(), object())
    assert pipe.sampler_name == ("res_multistep" if kind == "MiniMaxH3" else None)
    assert pipe.scheduler == "simple"
    assert json.dumps(state) == original


def test_legacy_lora_enabled_state_is_normalized_without_mutating_workflow(monkeypatch):
    entry = {"name": "native", "kind": "QI2", "type": "unet"}
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [entry], "lora": []})
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    captured = []
    def apply(model, clip, states, *args, **kwargs):
        captured.extend(states)
        return model, clip
    monkeypatch.setattr(cc, "_apply_loras", apply)
    state = {"active_kind": "QI2", "selected_type": "custom", "loras": [{"name": "legacy", "enabled": True}]}
    pipe = cc._build_control_center_pipe("test/repo", state, object(), object(), object())
    assert captured == [{"name": "legacy", "enabled": True, "auto_apply": True}]
    assert state["loras"] == [{"name": "legacy", "enabled": True}]
    assert pipe.lora_states == captured


@pytest.mark.parametrize("strength,expected", [(4, 4.0), ("-4", -4.0)])
def test_legacy_lora_strength_outside_slider_defaults_is_applied_unchanged(monkeypatch, strength, expected):
    entry = {"name": "Style", "kind": "QI2", "custom": True, "local_path": "models/loras/style.safetensors"}
    captured = []
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, True))
    def apply(model, clip, path, value):
        captured.append(value)
        return model, clip
    monkeypatch.setattr(cc, "_apply_lora_standard", apply)
    state = {"loras": [{"name": entry["name"], "auto_apply": True, "strength": strength}]}
    original = json.dumps(state)
    cc._parse_control_center_state(state)
    cc._apply_loras(object(), object(), state["loras"], {"lora": [entry]}, "custom", model_entry={"kind": "QI2"})
    assert captured == [expected]
    assert json.dumps(state) == original


@pytest.mark.parametrize("name", ["ILFlatMix", "Newgrounds Mix v2.0", "WAI Illustrious SDXL v1.70"])
@pytest.mark.parametrize("selected_type", ["checkpoint", "", None])
def test_checkpoint_pipe_uses_its_embedded_clip_and_vae(monkeypatch, name, selected_type):
    with open(cc._get_packaged_cc_path(), encoding="utf-8") as handle:
        config = json.load(handle)
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, True))
    model, clip, vae = object(), object(), object()
    loaded_paths = []
    def load_checkpoint(path):
        loaded_paths.append(path)
        return model, clip, vae
    monkeypatch.setattr(cc, "_load_checkpoint", load_checkpoint)
    state = {"selected_model": name}
    if selected_type is not None:
        state["selected_type"] = selected_type
    pipe = cc._build_control_center_pipe("test/repo", state)
    assert (pipe.model, pipe.clip, pipe.vae) == (model, clip, vae)
    assert loaded_paths == [pipe.model_entry["local_path"]]
    assert pipe.model_kind == "illustrious"
    assert pipe.model_cache_key["clips"] == []
    assert pipe.model_cache_key["vaes"] == []


def test_custom_lora_names_follow_catalog_case_insensitive_identity():
    name = cc._build_custom_lora_name("packB/style.safetensors", {"Style"})
    entries = [{"name": "Style"}, {"name": name}]
    assert len(cc._dedupe_entries_by_name(entries)) == 2


@pytest.mark.parametrize("selected_type,kind", [("unet", "QI2"), ("custom", "QI2"),
                                               ("custom", "Klein9b"), ("custom", "MiniMaxH3")])
def test_legacy_per_type_model_selection_keeps_the_backend_model_context(monkeypatch, selected_type, kind):
    entries = [{"name": name, "type": "unet", "kind": kind, "local_path": f"models/unet/{index}.safetensors"}
               for index, name in enumerate(["QI2", "QI2 Alternative"])]
    config = {"models": entries, "lora": [],
              "clip": [{"name": "CLIP", "kind": "QI2"}], "vae": [{"name": "VAE", "kind": "QI2"}]}
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    monkeypatch.setattr(cc, "_load_model_block", lambda *args, **kwargs: (object(), object(), object()))
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, False))
    state = {"active_kind": kind, "selected_type": selected_type, "selected_models": {"unet": " qi2 alternative " }}
    original = json.dumps(state)
    assert cc._build_control_center_pipe("test/repo", state, custom_audio_vae=object()).model_entry["name"] == "QI2 Alternative"
    assert json.dumps(state) == original


@pytest.mark.parametrize("selected_type", ["unet", "custom"])
@pytest.mark.parametrize("selected_model,family_name,legacy_name", [
    ("QI2", "", "QI2 Alternative"), ("Removed", "", "QI2 Alternative"),
    ("QI2 Alternative", "QI2", "QI2 Alternative"),
    ("QI2 Alternative", "Removed", "QI2 Alternative"), ("QI2 Alternative", "", ""),
])
def test_legacy_model_selection_priority_matches_execution(monkeypatch, selected_type, selected_model, family_name, legacy_name):
    entries = [{"name": name, "type": "unet", "kind": "QI2", "local_path": f"models/unet/{index}.safetensors"}
               for index, name in enumerate(["QI2", "QI2 Alternative"])]
    config = {"models": entries, "lora": [],
              "clip": [{"name": "CLIP", "kind": "QI2"}], "vae": [{"name": "VAE", "kind": "QI2"}]}
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    monkeypatch.setattr(cc, "_load_model_block", lambda *args, **kwargs: (object(), object(), object()))
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, False))
    state = {"active_kind": "QI2", "selected_type": selected_type, "selected_model": selected_model,
             "selected_models": {**({"QI2:unet": family_name} if family_name else {}),
                                 **({"unet": legacy_name} if legacy_name else {})}}
    original = json.dumps(state)
    if family_name:
        expected = family_name if family_name != "Removed" else "QI2"
    elif selected_type == "custom":
        expected = legacy_name or selected_model
    else:
        expected = selected_model if selected_model != "Removed" else legacy_name
    assert cc._build_control_center_pipe("test/repo", state).model_entry["name"] == expected
    assert json.dumps(state) == original


@pytest.mark.parametrize("name", ["klein b", "  KLEIN B  "])
@pytest.mark.parametrize("layout", ["legacy", "family"])
def test_model_identity_uses_canonical_catalog_name_without_mutating_workflow(monkeypatch, name, layout):
    entries = [{"name": f"Klein {suffix}", "kind": "Klein9b", "type": "unet", "local_path": f"models/unet/{suffix}.safetensors"}
               for suffix in ("A", "B")]
    config = {"models": entries, "lora": [],
              "clip": [{"name": "Klein CLIP", "kind": "Klein9b", "local_path": "models/clip/clip.safetensors"}],
              "vae": [{"name": "Klein VAE", "kind": "Klein9b", "local_path": "models/vae/vae.safetensors"}]}
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_apply_active_installed_paths", lambda config: config)
    monkeypatch.setattr(cc, "_load_model_block", lambda *args, **kwargs: (object(), object(), object()))
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, False))
    state = {"active_kind": "Klein9b", "selected_type": "unet"}
    state.update({"selected_model": name} if layout == "legacy" else {"selected_models": {"Klein9b:unet": name}})
    original = json.dumps(state)
    assert cc._build_control_center_pipe("test/repo", state).model_entry["name"] == "Klein B"
    assert json.dumps(state) == original


def test_custom_lora_delete_by_path_preserves_other_entry_with_same_legacy_name(monkeypatch, tmp_path):
    registry = tmp_path / "loras.json"
    entries = [{"name": "Style", "local_path": f"models/loras/pack{letter}/style.safetensors"}
               for letter in ("A", "B")]
    registry.write_text(json.dumps({"lora": entries}))
    monkeypatch.setattr(cc, "_get_custom_loras_path", lambda **kwargs: str(registry))
    assert cc._remove_custom_lora(local_path=entries[0]["local_path"], name="Style")
    assert cc._load_custom_loras() == entries[1:]


@pytest.mark.parametrize("payload", [[], None, {"repo_id": 123},
    {"repo_id": "test/repo", "node_state": {"model_params": []}},
    {"repo_id": "test/repo", "node_state": {"model_params": {"steps": []}}},
    {"repo_id": "test/repo", "node_state": {"qi2_cache": {"device": []}}},
    {"repo_id": "test/repo", "clothes_state": []}])
def test_clothes_preview_rejects_malformed_payload_before_loading_models(monkeypatch, payload):
    monkeypatch.setattr(cc.web, "Response", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    monkeypatch.setattr(cc, "_build_control_center_pipe", lambda *args: pytest.fail("Invalid preview must not load models"))
    response = cc._clothes_preview_response(payload)
    assert response.status == 400


def test_clothes_preview_rejects_invalid_json_before_preview_job(monkeypatch):
    async def invalid_json():
        raise ValueError("Invalid JSON")
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(cc, "run_preview_job", lambda *args: pytest.fail("Invalid JSON must not start a preview job"))
    request = SimpleNamespace(json=invalid_json, headers={"X-VNCCS-CSRF": "1"})
    assert asyncio.run(cc.cc_clothes_preview(request)).status == 400


@pytest.mark.parametrize("payload", [[], None, {"repo_id": 123}, {"repo_id": "test/repo", "name": []},
                                          {"repo_id": "test/repo", "category": "name"}])
def test_malformed_download_requests_stop_before_catalog_access(monkeypatch, payload):
    async def read():
        return payload
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args, **kwargs: pytest.fail("Invalid request must not read catalog"))
    response = asyncio.run(cc.cc_download(SimpleNamespace(json=read)))
    assert response.status == 400
    assert response.data["error"]


def test_download_aliases_share_the_canonical_catalog_identity(monkeypatch):
    entry = {"name": "Style", "local_path": "models/loras/style.safetensors", "hf_path": "style.safetensors"}
    tasks = queue.Queue()
    monkeypatch.setattr(cc, "_DOWNLOAD_QUEUE", tasks)
    monkeypatch.setattr(cc, "_DOWNLOAD_STATUS", {})
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args, **kwargs: {"lora": [entry]})
    monkeypatch.setattr(cc, "_resolve_model_download_path", lambda path: path)
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    for name in ("style", " STYLE ", "Style"):
        async def read():
            return {"repo_id": "test/repo", "category": "lora", "name": name}
        assert asyncio.run(cc.cc_download(SimpleNamespace(json=read))).status == 200
    assert tasks.qsize() == 1
    assert tasks.get_nowait() == ("test/repo", "cc_lora_Style", entry)
    assert list(cc._DOWNLOAD_STATUS) == ["cc_lora_Style"]


@pytest.mark.parametrize("active_status", ["queued", "downloading"])
def test_same_named_catalog_download_conflict_is_visible_and_can_be_retried(monkeypatch, active_status):
    entries = {
        repo: {"name": "Style", "local_path": f"models/loras/{suffix}/style.safetensors",
               "hf_path": "style.safetensors"}
        for repo, suffix in (("first/catalog", "first"), ("second/catalog", "second"))
    }
    tasks = queue.Queue()
    monkeypatch.setattr(cc, "_DOWNLOAD_QUEUE", tasks)
    monkeypatch.setattr(cc, "_DOWNLOAD_STATUS", {})
    monkeypatch.setattr(cc, "_DOWNLOAD_TARGETS", {})
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo, **kwargs: {"lora": [entries[repo]]})
    monkeypatch.setattr(cc, "_resolve_model_download_path", lambda path: path)
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    for index, repo in enumerate(entries):
        async def read():
            return {"repo_id": repo, "category": "lora", "name": "Style"}
        response = asyncio.run(cc.cc_download(SimpleNamespace(json=read)))
        assert response.status == (200 if index == 0 else 409)
        if index == 0:
            cc._DOWNLOAD_STATUS["cc_lora_Style"]["status"] = active_status
        else:
            assert "retry" in response.data["error"]
    assert tasks.qsize() == 1
    assert tasks.get_nowait() == ("first/catalog", "cc_lora_Style", entries["first/catalog"])
    cc._DOWNLOAD_STATUS["cc_lora_Style"] = {"status": "success"}
    assert asyncio.run(cc.cc_download(SimpleNamespace(json=read))).status == 200
    assert tasks.get_nowait() == ("second/catalog", "cc_lora_Style", entries["second/catalog"])


@pytest.mark.parametrize("repo_id,expected", [
    ("first/catalog", {"same": {"status": "downloading"}, "legacy": {"status": "queued"}}),
    ("second/catalog", {"legacy": {"status": "queued"}}),
    ("", {"same": {"status": "downloading"}, "legacy": {"status": "queued"}}),
])
def test_download_status_keeps_public_keys_and_filters_known_catalog_origins(monkeypatch, repo_id, expected):
    statuses = {"same": {"status": "downloading"}, "legacy": {"status": "queued"}}
    monkeypatch.setattr(cc, "_DOWNLOAD_STATUS", statuses)
    monkeypatch.setattr(cc, "_DOWNLOAD_TARGETS", {"same": ("first/catalog", "models/loras/first/style.safetensors")})
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"repo_id": repo_id}))
    response = asyncio.run(cc.get_download_status(request))
    assert response.data == expected
    assert cc._DOWNLOAD_STATUS == statuses


@pytest.mark.parametrize("valid", [True, False])
def test_download_origin_survives_worker_success_and_failure(monkeypatch, tmp_path, valid):
    cached = tmp_path / "cache.gguf"
    cached.write_bytes((b"GGUF" if valid else b"<html>") + b"x" * 1024)
    target = tmp_path / "models" / "first" / "style.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing weights")
    entry = {"name": "Style", "local_path": "models/unet/first/style.gguf",
             "hf_path": "style.gguf", "version": "1"}
    tasks = queue.Queue()
    versions = []
    monkeypatch.setattr(cc, "_DOWNLOAD_QUEUE", tasks)
    monkeypatch.setattr(cc, "_DOWNLOAD_STATUS", {})
    monkeypatch.setattr(cc, "_DOWNLOAD_TARGETS", {})
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args, **kwargs: {"models": [entry]})
    monkeypatch.setattr(cc, "_resolve_model_download_path", lambda path: str(target))
    monkeypatch.setattr(cc, "_download_with_progress", lambda *args, **kwargs: str(cached))
    monkeypatch.setattr(cc, "update_installed_version", lambda *args: versions.append(args))
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    async def read():
        return {"repo_id": "first/catalog", "category": "models", "name": "Style"}
    assert asyncio.run(cc.cc_download(SimpleNamespace(json=read))).status == 200
    tasks.put(None)
    cc._download_worker_loop()
    expected_status = "success" if valid else "error"
    for repo in ("first/catalog", "second/catalog", ""):
        request = SimpleNamespace(rel_url=SimpleNamespace(query={"repo_id": repo}))
        data = asyncio.run(cc.get_download_status(request)).data
        if repo == "second/catalog":
            assert data == {}, "Completed and failed jobs keep their origin rather than becoming unscoped legacy jobs"
        else:
            assert data["cc_models_Style"]["status"] == expected_status
    assert target.read_bytes() == (cached.read_bytes() if valid else b"existing weights")
    assert versions == ([("cc_models_Style", "1")] if valid else []), "Installed registry keys stay compatible"


@pytest.mark.parametrize("handler", [cc.cc_dependencies, cc.vnccs_manager_enable_personal_cloud])
def test_policy_and_dependency_requests_reject_non_objects(monkeypatch, handler):
    async def read():
        return []
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    response = asyncio.run(handler(SimpleNamespace(json=read)))
    assert response.status == 400


@pytest.mark.parametrize("field,value", [
    ("model_type", ["nunchaku"]), ("model_name", []),
    ("model_path", None), ("model_path", {"path": "model.safetensors"}),
])
def test_dependency_request_rejects_invalid_identity_fields(monkeypatch, field, value):
    payload = {"model_type": "nunchaku", "model_name": "Qwen", "model_path": "models/unet/qwen.safetensors", field: value}
    async def read():
        return payload
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    response = asyncio.run(cc.cc_dependencies(SimpleNamespace(json=read)))
    assert response.status == 400


@pytest.mark.parametrize("payload,expected", [
    ({}, {"ok": True, "message": ""}),
    ({"model_type": "unet"}, {"ok": True, "message": ""}),
    ({"model_type": "nunchaku", "model_name": "Qwen", "model_path": "models/unet/qwen.safetensors", "has_enabled_loras": True},
     {"model_kind": "qwen-image", "disabled": True, "ok": False, "has_enabled_loras": True}),
])
def test_dependency_request_preserves_disabled_legacy_contract(monkeypatch, payload, expected):
    async def read():
        return payload
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    response = asyncio.run(cc.cc_dependencies(SimpleNamespace(json=read)))
    assert response.status == 200
    assert expected.items() <= response.data.items()
