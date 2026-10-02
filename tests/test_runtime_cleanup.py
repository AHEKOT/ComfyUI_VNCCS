"""Runtime cleanup contracts that must run without PyTorch or ComfyUI."""

import sys

import pytest

from nodes import runtime_cleanup as runtime
from runtime_cleanup_helpers import dynamic_runtime


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


def test_nested_stages_clean_once_and_reset_after_failure(dynamic_runtime):
    _, events, pending = dynamic_runtime
    with pytest.raises(RuntimeError, match='interrupted'):
        with runtime.inference_stage():
            with runtime.inference_stage():
                pending.append(object())
                raise RuntimeError('interrupted')
    assert events == ['prefetch', 'cast_buffers', 'watermarks']
    assert not pending
    with runtime.inference_stage():
        pending.append(object())
    assert events == ['prefetch', 'cast_buffers', 'watermarks'] * 2
    assert not pending

