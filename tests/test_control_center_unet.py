"""QI2 catalog selection and native diffusion model loading."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nodes import vnccs_control_center as cc


MODEL = {
    "name": cc.DEFAULT_QI2_MODEL,
    "type": "unet",
    "kind": "QI2",
    "local_path": "models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
}
ALTERNATE = {**MODEL, "name": "Other native Qwen", "local_path": "models/unet/other.safetensors"}
LEGACY = {**MODEL, "name": "Qwen-Image-Edit-2511-GGUF-Q5", "type": "gguf", "kind": "QIE2511"}


@pytest.mark.parametrize("state,expected", [
    ({}, MODEL["name"]),
    ({"active_kind": "QI2", "selected_type": "unet", "selected_model": ALTERNATE["name"]}, ALTERNATE["name"]),
    ({"active_kind": "QI2", "selected_type": "gguf", "selected_model": LEGACY["name"]}, MODEL["name"]),
])
def test_qi2_loads_native_unet(monkeypatch, state, expected):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    config = {
        "models": [LEGACY, ALTERNATE, MODEL],
        "clip": [{"name": "QI2 clip", "kind": "QI2"}],
        "vae": [{"name": "QI2 vae", "kind": "QI2"}],
        "lora": [],
    }
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cc, "_load_clips", lambda *args: "clip")
    monkeypatch.setattr(cc, "_load_vae", lambda *args: "vae")
    monkeypatch.setattr(cc, "_apply_loras", lambda model, clip, *args, **kwargs: (model, clip))
    loaded = []
    monkeypatch.setattr(cc.comfy.sd, "load_diffusion_model", lambda path, model_options: loaded.append((path, model_options)) or "model", raising=False)
    monkeypatch.setattr(cc, "_load_gguf", lambda *args: pytest.fail("QI2 must not call a GGUF loader"))

    pipe = cc._build_control_center_pipe("test/repo", json.dumps(state))
    assert (pipe.model, pipe.clip, pipe.vae) == ("model", "clip", "vae")
    assert pipe.model_entry["name"] == expected
    assert pipe.model_entry["type"] == "unet"
    assert loaded == [(pipe.model_entry["local_path"], {})]
    assert pipe.sample_steps == 25
    assert pipe.cfg == 3


def test_legacy_selection_requires_explicit_qi2_choice(monkeypatch):
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [MODEL]})
    with pytest.raises(RuntimeError, match="QIE2511 is no longer supported"):
        cc._build_control_center_pipe("test/repo", {"active_kind": "QIE2511"})


def test_catalog_without_qi2_native_model_reports_missing_unet(monkeypatch):
    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: {"models": [LEGACY]})
    with pytest.raises(RuntimeError, match="No native QI2 UNet model"):
        cc._build_control_center_pipe("test/repo", {"active_kind": "QI2"})


def test_custom_qi2_context_prefers_native_default():
    context = cc._custom_context_model_entry(
        {"models": [LEGACY, ALTERNATE, MODEL]},
        {"selected_type": "custom", "active_kind": "QI2", "selected_model": LEGACY["name"]},
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


def test_packaged_catalog_has_qi2_and_viggle_without_qie():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "control_center.json").read_text())
    assert not any(entry.get("kind") == "QIE2511" for entries in catalog.values() if isinstance(entries, list) for entry in entries if isinstance(entry, dict))
    qi2 = [entry for entry in catalog["models"] if entry.get("kind") == "QI2"]
    assert qi2[0]["name"] == MODEL["name"]
    assert qi2[0]["local_path"] == MODEL["local_path"]
    assert qi2[0]["hf_repo"] == "Comfy-Org/Qwen-Image-2.1"
    turbo = next(entry for entry in catalog["lora"] if entry.get("type") == "TurboLora" and entry.get("kind") == "QI2")
    assert turbo["hf_repo"] == "Viggle/Qwen-Image-2.1-viggle-turbo"
    assert turbo["hf_path"].endswith("v0.2.1-6step-lora-r128.safetensors")


def test_qi2_six_step_preset_selects_viggle_lora():
    turbo = {"name": "Qwen Image 2.1 Viggle Turbo", "type": "TurboLora", "kind": "QI2"}
    config = {"lora": [turbo]}
    assert cc._ensure_required_turbo_lora_state([], config, MODEL, {"steps": 25, "cfg": 3}) == []
    selected = cc._ensure_required_turbo_lora_state([], config, MODEL, {"steps": 6, "cfg": 1})
    assert selected == [{"name": turbo["name"], "auto_apply": True, "strength": 1.0}]
