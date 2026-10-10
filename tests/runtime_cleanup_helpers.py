"""Model-free runtime fixtures; optional generator imports stay inside helpers."""

import sys
import types

import pytest


def install_node_calls(monkeypatch, call):
    """Keep the production dispatcher and its cleanup active in model-free tests."""
    def node_class(name):
        def run(**kwargs):
            return call(name, **kwargs)
        return type(name, (), {"FUNCTION": "run", "run": staticmethod(run)})
    from nodes import character_generator as cg

    names = [
        "VNCCS_Flux_Klein_Encoder",
        "MiniMaxH3ReferenceToVideo", "SamplerCustomAdvanced", "VAEDecode",
        "BasicGuider", "RandomNoise", "KSamplerSelect", "BasicScheduler",
        "ProbeEncode", "KSampler", "VAEDecodeTiled",
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


