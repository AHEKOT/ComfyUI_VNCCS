"""Preview and workflow weight reuse without PyTorch or model downloads."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from nodes import vnccs_control_center as cc


class Asset:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else object()
        self.options = {}

    def clone(self):
        result = Asset(self.weights)
        result.options = self.options.copy()
        return result


def setup(monkeypatch, tmp_path, kind="QI2", model_type="unet"):
    monkeypatch.setattr(cc, "_MODEL_ASSET_CACHE", {})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, float8_e4m3fn="fp8", float8_e5m2="fp8_e5m2",
    ))
    config = {
        "models": [{"name": "model", "kind": kind, "type": model_type,
                    "local_path": "models/unet/model.safetensors"}],
        "clip": [{"name": "clip", "kind": kind, "local_path": "models/clip/clip.safetensors"}],
        "vae": [{"name": "vae", "kind": kind, "local_path": "models/vae/vae.safetensors"}],
        "lora": [],
    }
    if kind == "MiniMaxH3":
        config["vae"].append({"name": "audio", "kind": kind, "type": "AudioVAE",
                              "local_path": "models/vae/audio.safetensors"})
    paths = {}
    for entries in config.values():
        for entry in entries:
            path = tmp_path / entry["name"]
            path.write_bytes(b"weights")
            paths[entry["local_path"]] = str(path)
    calls = []

    def loaded(name, result):
        calls.append(name)
        return result

    monkeypatch.setattr(cc, "_get_cc_config", lambda repo: config)
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (paths.get(path, ""), path in paths))
    monkeypatch.setattr(cc.comfy.sd, "CLIPType", SimpleNamespace(STABLE_DIFFUSION="sd"), raising=False)
    monkeypatch.setattr(cc.comfy.sd, "load_diffusion_model", lambda *args, **kwargs: loaded("model", Asset()), raising=False)
    monkeypatch.setattr(cc.comfy.sd, "load_clip", lambda **kwargs: loaded("clip", Asset()), raising=False)
    monkeypatch.setattr(cc.comfy.sd, "load_checkpoint_guess_config", lambda *args, **kwargs: loaded(
        "checkpoint", (Asset(), Asset(), Asset()),
    ), raising=False)
    monkeypatch.setattr(cc.comfy.utils, "load_torch_file", lambda path, **kwargs: loaded(
        path, ({"source": path}, {}),
    ), raising=False)
    monkeypatch.setattr(cc.comfy.sd, "VAE", lambda **kwargs: loaded("vae", Asset()), raising=False)
    state = {"active_kind": kind, "selected_type": model_type, "selected_model": "model"}
    return config, paths, state, calls


@pytest.mark.parametrize("workflow_first", [False, True])
def test_preview_and_run_share_weights_but_keep_separate_request_state(monkeypatch, tmp_path, workflow_first):
    _config, _paths, state, calls = setup(monkeypatch, tmp_path)
    observed = []
    module = ModuleType(f"{cc.__package__}.clothes_designer")

    class Designer:
        def process(self, pipe, **kwargs):
            observed.append(pipe)
            image = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((1, 2, 2, 3))))
            return image, "sheets", "Blue"

    module.ClothesDesigner = Designer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cc.web, "json_response", lambda data: data, raising=False)
    monkeypatch.setattr(cc.web, "Response", lambda **data: pytest.fail(str(data)), raising=False)
    center = cc.VNCCS_ControlCenter()
    if workflow_first:
        observed.append(center.execute("test/repo", json.dumps(state))[0])
    for seed in (1, 2):
        result = cc._clothes_preview_response({
            "repo_id": "test/repo", "node_state": json.dumps(state),
            "clothes_state": {"gen_settings": {"seed": seed}},
        })
        assert result["image"]
    first = observed[0]
    first.pos = "request conditioning"
    first.model.options["patch"] = True
    first.clip.options["template"] = "request template"
    state.update(model_params={"steps": 31, "cfg": 4}, qi2_cache={"device": "cpu", "dtype": "int4"})
    run = center.execute("test/repo", json.dumps(state))[0]
    assert calls.count("model") == calls.count("clip") == calls.count("vae") == 1
    assert run is not first and run.pos is None
    assert run.model is not first.model and run.model.weights is first.model.weights
    assert run.clip is not first.clip and run.clip.weights is first.clip.weights
    assert run.model.options == run.clip.options == {}
    assert run.vae is first.vae
    assert run.sample_steps == 31 and run.cfg == 4
    assert run.qi2_cache == {"device": "cpu", "dtype": "int4"}


def test_file_dtype_and_model_changes_reload_only_affected_assets(monkeypatch, tmp_path):
    config, paths, state, calls = setup(monkeypatch, tmp_path)
    original = cc._build_control_center_pipe("test/repo", state)
    (tmp_path / "model").write_bytes(b"replacement weights")
    replaced = cc._build_control_center_pipe("test/repo", state)
    assert replaced.model.weights is not original.model.weights
    assert replaced.model_cache_key != original.model_cache_key
    assert calls.count("model") == 2 and calls.count("clip") == calls.count("vae") == 1
    state["type_settings"] = {"unet": {"weight_dtype": "fp8_e4m3fn"}}
    cc._build_control_center_pipe("test/repo", state)
    assert calls.count("model") == 3 and calls.count("clip") == calls.count("vae") == 1
    (tmp_path / "clip").write_bytes(b"replacement CLIP")
    cc._build_control_center_pipe("test/repo", state)
    assert calls.count("model") == 3 and calls.count("clip") == 2 and calls.count("vae") == 1
    (tmp_path / "vae").write_bytes(b"replacement VAE")
    cc._build_control_center_pipe("test/repo", state)
    assert calls.count("model") == 3 and calls.count("clip") == calls.count("vae") == 2
    alternate = {**config["models"][0], "name": "other", "local_path": "models/unet/other.safetensors"}
    config["models"].append(alternate)
    (tmp_path / "other").write_bytes(b"other model")
    paths[alternate["local_path"]] = str(tmp_path / "other")
    for selected in ("other", "model"):
        state["selected_model"] = selected
        cc._build_control_center_pipe("test/repo", state)
    assert calls.count("model") == 5 and calls.count("clip") == calls.count("vae") == 2
    assert set(cc._MODEL_ASSET_CACHE) == {"model", "clip", "vae"}


@pytest.mark.parametrize("kind,model_type", [("MiniMaxH3", "unet"), ("Klein9b", "checkpoint"), ("Klein9b", "gguf")])
def test_supported_loaders_and_both_h3_vaes_are_reused(monkeypatch, tmp_path, kind, model_type):
    _config, _paths, state, calls = setup(monkeypatch, tmp_path, kind, model_type)
    loader = type("GGUF", (), {"load_unet": lambda self, name: calls.append("gguf") or (Asset(),)})
    monkeypatch.setattr(cc, "_get_gguf_loader_class", lambda: loader)
    monkeypatch.setattr(cc, "_describe_gguf_loader", lambda cls: {})
    first = cc._build_control_center_pipe("test/repo", state)
    second = cc.VNCCS_ControlCenter().execute("test/repo", json.dumps(state))[0]
    assert first.model.weights is second.model.weights
    assert first.clip.weights is second.clip.weights and first.vae is second.vae
    if kind == "MiniMaxH3":
        assert first.audio_vae is second.audio_vae and first.audio_vae is not first.vae
        assert calls.count("model") == calls.count("clip") == 1 and calls.count("vae") == 2
    else:
        assert calls.count(model_type) == 1


def test_simultaneous_loads_reuse_one_asset_and_failures_are_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_MODEL_ASSET_CACHE", {})
    path = tmp_path / "model"
    path.write_bytes(b"weights")
    started, release = threading.Event(), threading.Event()
    calls = []
    asset = object()

    def load():
        calls.append(True)
        started.set()
        assert release.wait(3)
        return asset

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cc._cached_model_asset, "model", (str(path),), (), load)
        assert started.wait(3)
        second = executor.submit(cc._cached_model_asset, "model", (str(path),), (), load)
        release.set()
        assert first.result(timeout=3) is second.result(timeout=3) is asset
    assert len(calls) == 1
    path.write_bytes(b"new weights")

    def fail():
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        cc._cached_model_asset("model", (str(path),), (), fail)
    assert "model" not in cc._MODEL_ASSET_CACHE
    assert cc._cached_model_asset("model", (str(path),), (), lambda: asset) is asset


def test_lora_weights_reuse_and_file_replacement(monkeypatch, tmp_path):
    cc._cached_lora_file.cache_clear()
    path = tmp_path / "adapter.safetensors"
    contents = (2).to_bytes(8, "little") + b"{}" + b"\0" * 1024
    path.write_bytes(contents)
    calls = []
    monkeypatch.setattr(cc.comfy.utils, "load_torch_file", lambda name, **kwargs:
        calls.append(kwargs) or ({"weight": object()}, {"alpha": 1}), raising=False)
    def apply(model, clip, weights, strength, clip_strength):
        patched = model.clone()
        patched.options["lora"] = weights, strength
        return patched, clip.clone()
    monkeypatch.setattr(cc.comfy.sd, "load_lora_for_models", apply, raising=False)
    model, clip = Asset(), Asset()
    full, _ = cc._apply_lora_standard(model, clip, str(path), 1.0)
    half, _ = cc._apply_lora_standard(model, clip, str(path), 0.5)
    assert full.options["lora"][1] == 1 and half.options["lora"][1] == 0.5
    assert full.options["lora"][0] is half.options["lora"][0]
    assert model.options == clip.options == {}
    first = cc._load_lora_file(str(path))
    assert cc._load_lora_file(str(path)) is first
    assert calls == [{"safe_load": True, "return_metadata": True}]
    path.write_bytes(contents + b"new adapter weights")
    assert cc._load_lora_file(str(path)) is not first
    assert len(calls) == 2
    cc._cached_lora_file.cache_clear()
