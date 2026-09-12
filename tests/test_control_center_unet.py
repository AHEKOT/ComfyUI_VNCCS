"""QIE2511 defaults, legacy state migration, and native diffusion model loading."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nodes import vnccs_control_center as cc


MODEL = {
    "name": cc.DEFAULT_QIE_MODEL,
    "type": "unet",
    "kind": "QIE2511",
    "local_path": "models/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
}
ALTERNATE = {**MODEL, "name": "Other native Qwen", "local_path": "models/unet/other.safetensors"}
LEGACY = {**MODEL, "name": "Qwen-Image-Edit-2511-GGUF-Q5", "type": "gguf"}


@pytest.mark.parametrize("state,expected", [
    ({}, MODEL["name"]),
    ({"selected_type": "gguf", "selected_model": LEGACY["name"]}, MODEL["name"]),
    ({"active_kind": "QIE2511", "selected_types_by_kind": {"QIE2511": "gguf"},
      "selected_models": {"QIE2511:gguf": LEGACY["name"]}}, MODEL["name"]),
    ({"selected_type": "unet", "selected_model": ALTERNATE["name"]}, ALTERNATE["name"]),
    ({"selected_type": "gguf", "selected_model": LEGACY["name"],
      "selected_models": {"QIE2511:unet": ALTERNATE["name"]}}, ALTERNATE["name"]),
    ({"selected_type": "gguf", "selected_model": MODEL["name"]}, MODEL["name"]),
])
def test_qie_loads_native_unet_without_gguf(monkeypatch, state, expected):
    # The default dtype path imports torch but does not use tensor operations.
    # Keep this loader-routing test runnable in the lightweight CI environment.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    config = {
        "models": [LEGACY, ALTERNATE, MODEL],
        "clip": [{"name": "QIE clip", "kind": "QIE2511"}],
        "vae": [{"name": "QIE vae", "kind": "QIE2511"}],
        "lora": [],
    }
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cc, "_load_clips", lambda *args: "clip")
    monkeypatch.setattr(cc, "_load_vae", lambda *args: "vae")
    monkeypatch.setattr(cc, "_apply_loras", lambda model, clip, *args, **kwargs: (model, clip))
    loaded = []
    monkeypatch.setattr(cc.comfy.sd, "load_diffusion_model", lambda path, model_options: loaded.append((path, model_options)) or "model", raising=False)

    def forbidden(*args):
        raise AssertionError("QIE2511 must not call a GGUF loader")

    monkeypatch.setattr(cc, "_load_gguf", forbidden)
    pipe = cc._build_control_center_pipe("test/repo", json.dumps(state))
    assert (pipe.model, pipe.clip, pipe.vae) == ("model", "clip", "vae")
    assert pipe.model_entry["name"] == expected
    assert pipe.model_entry["type"] == "unet"
    assert loaded == [(pipe.model_entry["local_path"], {})]


def test_legacy_only_catalog_reports_missing_unet_instead_of_loading_gguf(monkeypatch):
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [LEGACY]})
    with pytest.raises(RuntimeError, match="No native QIE2511 UNet model"):
        cc._build_control_center_pipe("test/repo", {"selected_type": "gguf", "selected_model": LEGACY["name"]})


def test_custom_qie_context_prefers_native_default_over_stale_gguf():
    context = cc._custom_context_model_entry(
        {"models": [LEGACY, ALTERNATE, MODEL]},
        {"selected_type": "custom", "selected_model": LEGACY["name"]},
    )
    assert context == MODEL


def test_diffusion_model_paths_also_search_configured_unet_folders(monkeypatch, tmp_path):
    folder = tmp_path / "custom-unet"
    folder.mkdir()
    target = folder / Path(MODEL["local_path"]).name
    target.write_bytes(b"test")
    monkeypatch.setattr(cc.folder_paths, "get_folder_paths", lambda key: [str(folder)] if key == "unet" else [])
    monkeypatch.setattr(cc.folder_paths, "get_full_path", lambda *args: None)
    assert cc._find_model_on_disk(MODEL["local_path"]) == (str(target), True)
    assert cc._resolve_model_download_path(MODEL["local_path"]) == str(target)


def test_module_status_no_longer_tracks_or_installs_comfyui_gguf(monkeypatch):
    monkeypatch.setattr(cc, "_custom_nodes_roots", lambda: [])
    monkeypatch.setattr(cc.web, "json_response", lambda payload: payload, raising=False)
    result = asyncio.run(cc.vnccs_module_status(None))
    dependencies = result["dependencies"]
    assert "gguf" not in dependencies
    assert all(item.get("manager_id") != "ComfyUI-GGUF" for item in dependencies.values())
    assert "impact_pack" in dependencies


def test_packaged_catalog_uses_native_qie_models():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "control_center.json").read_text())
    qie = [entry for entry in catalog["models"] if entry.get("kind") == "QIE2511"]
    assert qie[0]["name"] == MODEL["name"]
    assert qie[0]["local_path"] == MODEL["local_path"]
    assert qie[0]["hf_repo"] == "MIUProject/Qwen-Image-Edit-2511-int8-convrot"
    assert all(entry["type"] != "gguf" for entry in qie)
