"""H3 direct node calls must observe ComfyUI's dynamic VRAM boundaries."""

import sys
import types
import weakref

import pytest
import torch

from nodes import character_generator as cg
from nodes import runtime_cleanup as runtime


def install_node_calls(monkeypatch, call):
    """Keep the production dispatcher and its cleanup active in model-free tests."""
    def node_class(name):
        def run(**kwargs):
            return call(name, **kwargs)
        return type(name, (), {"FUNCTION": "run", "run": staticmethod(run)})
    names = [
        "MiniMaxH3ReferenceToVideo", "SamplerCustomAdvanced", "VAEDecode",
        "BasicGuider", "RandomNoise", "KSamplerSelect", "BasicScheduler",
        "VNCCS_Flux_Klein_Encoder", "ProbeEncode", "KSampler", "VAEDecodeTiled",
        "ImageScale", "SeedVR2Preprocess", "VAEEncodeTiled", "SeedVR2Conditioning",
        "SeedVR2PostProcessing",
    ]
    monkeypatch.setattr(cg, "comfy_nodes", types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={name: node_class(name) for name in names},
    ))


@pytest.fixture
def dynamic_runtime(monkeypatch):
    events = []
    pending = []
    memory = types.ModuleType("comfy.memory_management")
    memory.aimdo_enabled = True
    prefetch = types.ModuleType("comfy.model_prefetch")
    def cleanup():
        events.append("prefetch")
        pending.clear()
    prefetch.cleanup_prefetch_queues = cleanup
    aimdo = types.ModuleType("comfy_aimdo")
    vbar = types.ModuleType("comfy_aimdo.model_vbar")
    vbar.vbars_reset_watermark_limits = lambda: events.append("watermarks")
    aimdo.model_vbar = vbar
    for name, module in {
        "comfy.memory_management": memory, "comfy.model_prefetch": prefetch,
        "comfy_aimdo": aimdo, "comfy_aimdo.model_vbar": vbar,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["comfy"], "memory_management", memory, raising=False)
    monkeypatch.setattr(sys.modules["comfy"], "model_prefetch", prefetch, raising=False)
    management = types.ModuleType("comfy.model_management")
    management.reset_cast_buffers = lambda: events.append("cast_buffers")
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", management, raising=False)
    return memory, events, pending


@pytest.mark.parametrize("fail", [False, True])
def test_h3_stage_cleans_runtime_on_success_and_failure(dynamic_runtime, monkeypatch, fail):
    _, events, pending = dynamic_runtime
    output = ({"samples": torch.ones(1, 2, 2)},)
    def call(name, **kwargs):
        pending.append(torch.ones(4))
        if fail:
            raise RuntimeError("sampling interrupted")
        return output
    install_node_calls(monkeypatch, call)
    if fail:
        with pytest.raises(RuntimeError, match="sampling interrupted"):
            cg._call_comfy_node("SamplerCustomAdvanced")
    else:
        assert cg._call_comfy_node("SamplerCustomAdvanced") is output
    assert events == ["prefetch", "cast_buffers", "watermarks"]
    assert pending == []


def test_cleanup_is_noop_without_dynamic_allocator(dynamic_runtime):
    memory, events, _ = dynamic_runtime
    memory.aimdo_enabled = False
    runtime.cleanup_runtime()
    assert events == []


def test_cleanup_is_compatible_with_older_comfyui(dynamic_runtime, monkeypatch):
    _, events, _ = dynamic_runtime
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", None)
    runtime.cleanup_runtime()
    assert events == []


def test_long_pose_list_does_not_accumulate_executor_resources(dynamic_runtime, monkeypatch):
    _, events, pending = dynamic_runtime
    generator = cg.VNCCS_CharacterGenerator()
    model = object()
    monkeypatch.setattr(generator, "_apply_pose_lora_to_model", lambda *args: model)
    monkeypatch.setattr(generator, "_log_stage", lambda *args, **kwargs: None)
    calls, guiders, videos = [], [], []
    count = 48

    class Guider:
        pass

    def call(name, **kwargs):
        calls.append(name)
        if name in {"MiniMaxH3ReferenceToVideo", "SamplerCustomAdvanced", "VAEDecode"}:
            assert not pending, "Previous stage still retains dynamic VRAM resources"
            pending.append(torch.zeros(1024))
        if name == "MiniMaxH3ReferenceToVideo":
            assert kwargs["length"] == 5
            return ([torch.ones(1)], {"samples": torch.zeros(1)})
        if name == "BasicGuider":
            assert kwargs["model"] is model
            assert all(ref() is None for ref in guiders)
            guider = Guider()
            guiders.append(weakref.ref(guider))
            return (guider,)
        if name == "SamplerCustomAdvanced":
            return ({"samples": torch.ones(1)}, None)
        if name == "VAEDecode":
            assert set(kwargs) == {"samples", "vae"}
            assert all(ref() is None for ref in videos)
            decoded = torch.ones(5, 8, 8, 3)
            videos.append(weakref.ref(decoded))
            return (decoded,)
        return (object(),)

    install_node_calls(monkeypatch, call)
    result = generator._run_h3_pose_generation(
        [torch.ones(1, 32, 32, 3) for _ in range(count)],
        torch.ones(1, 32, 32, 3), None,
        {"model": model, "clip": object(), "vae": object(), "audio_vae": object()},
        "prompt", {"target_size": 1024}, {},
        {"sampler_name": "euler", "scheduler": "simple", "steps": 4, "denoise": 1, "seed": 77}, {},
    )
    assert result.shape == (count, 8, 8, 3)
    assert torch.all(result == 1)
    assert events == ["prefetch", "cast_buffers", "watermarks"] * len(calls)
    assert calls.count("MiniMaxH3ReferenceToVideo") == count
    assert calls.count("SamplerCustomAdvanced") == count
    assert calls.count("VAEDecode") == count
    assert calls.index("SamplerCustomAdvanced") > max(i for i, name in enumerate(calls) if name == "MiniMaxH3ReferenceToVideo")
    assert calls.index("VAEDecode") > max(i for i, name in enumerate(calls) if name == "SamplerCustomAdvanced")
    assert not pending
    assert all(ref() is None for ref in guiders + videos)
