"""Tests for nodes/vnccs_pipe.py — inheritance logic and process_pipe."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.vnccs_pipe import VNCCS_Pipe, PIPE_INHERIT
from nodes.sampler_scheduler_picker import VNCCSSamplerSchedulerPicker


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pipe(**kwargs):
    """Create a minimal pipe-like object with given attributes."""
    p = types.SimpleNamespace(
        model=object(),
        clip=object(),
        vae=object(),
        pos=object(),
        neg=object(),
        seed_int=1000,
        sample_steps=20,
        cfg=7.0,
        denoise=1.0,
        sampler_name="euler",
        scheduler="normal",
        loader_type="standard",
        nunchaku_kind=None,
        nunchaku_settings=None,
        model_entry=None,
    )
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


# ── _inherit ──────────────────────────────────────────────────────────────────

class TestInherit:
    def test_none_inherits_from_pipe(self):
        pipe = _make_pipe(cfg=7.5)
        assert VNCCS_Pipe._inherit(None, pipe, "cfg", zero_is_empty=False) == 7.5

    def test_zero_inherits_when_zero_is_empty(self):
        pipe = _make_pipe(sample_steps=30)
        assert VNCCS_Pipe._inherit(0, pipe, "sample_steps") == 30

    def test_zero_does_not_inherit_when_zero_is_not_empty(self):
        pipe = _make_pipe(cfg=7.5)
        assert VNCCS_Pipe._inherit(0, pipe, "cfg", zero_is_empty=False) == 0

    def test_nonzero_value_wins_over_pipe(self):
        pipe = _make_pipe(sample_steps=30)
        assert VNCCS_Pipe._inherit(10, pipe, "sample_steps") == 10

    def test_none_no_pipe_returns_none(self):
        assert VNCCS_Pipe._inherit(None, None, "cfg", zero_is_empty=False) is None

    def test_missing_pipe_attr_returns_value(self):
        pipe = types.SimpleNamespace()  # no cfg attr
        result = VNCCS_Pipe._inherit(None, pipe, "cfg", zero_is_empty=False)
        assert result is None


# ── process_pipe — inheritance ─────────────────────────────────────────────────

class TestProcessPipeInheritance:
    def _run(self, **kwargs):
        node = VNCCS_Pipe()
        defaults = dict(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=None,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        defaults.update(kwargs)
        return node.process_pipe(**defaults)

    def test_inherits_model_from_pipe(self):
        pipe = _make_pipe()
        result = self._run(pipe=pipe)
        assert result[0] is pipe.model

    def test_explicit_model_overrides_pipe(self):
        my_model = object()
        pipe = _make_pipe()
        result = self._run(model=my_model, pipe=pipe)
        assert result[0] is my_model

    def test_inherits_seed_from_pipe_when_zero(self):
        pipe = _make_pipe(seed_int=9999)
        result = self._run(seed_int=0, pipe=pipe)
        assert result[5] == 9999

    def test_explicit_seed_overrides_pipe(self):
        pipe = _make_pipe(seed_int=9999)
        result = self._run(seed_int=42, pipe=pipe)
        assert result[5] == 42

    def test_inherits_steps_from_pipe_when_zero(self):
        pipe = _make_pipe(sample_steps=50)
        result = self._run(sample_steps=0, pipe=pipe)
        assert result[6] == 50

    def test_inherits_cfg_from_pipe_when_zero(self):
        pipe = _make_pipe(cfg=6.5)
        result = self._run(cfg=0.0, pipe=pipe)
        assert result[7] == 6.5

    def test_inherits_denoise_from_pipe_when_zero(self):
        pipe = _make_pipe(denoise=0.8)
        result = self._run(denoise=0.0, pipe=pipe)
        assert result[8] == 0.8

    def test_pipe_inherit_sentinel_uses_pipe_sampler(self):
        pipe = _make_pipe(sampler_name="heun")
        result = self._run(pipe=pipe, sampler_name=PIPE_INHERIT)
        assert result[10] == "heun"

    def test_pipe_inherit_sentinel_uses_pipe_scheduler(self):
        pipe = _make_pipe(scheduler="karras")
        result = self._run(pipe=pipe, scheduler=PIPE_INHERIT)
        assert result[11] == "karras"

    def test_explicit_sampler_overrides_pipe(self):
        pipe = _make_pipe(sampler_name="heun")
        result = self._run(pipe=pipe, sampler_name="euler")
        assert result[10] == "euler"

    def test_returns_self_as_pipe_output(self):
        node = VNCCS_Pipe()
        pipe = _make_pipe()
        defaults = dict(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=pipe,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        result = node.process_pipe(**defaults)
        assert result[9] is node


# ── process_pipe — sampler/scheduler fallback ─────────────────────────────────

class TestProcessPipeSamplerFallback:
    def _run(self, **kwargs):
        node = VNCCS_Pipe()
        defaults = dict(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=None,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        defaults.update(kwargs)
        return node.process_pipe(**defaults)

    def test_invalid_sampler_passes_through(self):
        # No compatibility fallback — unknown value passes through unchanged
        result = self._run(sampler_name="no_such_sampler_xyz")
        assert isinstance(result[10], str)
        assert result[10] == "no_such_sampler_xyz"

    def test_invalid_scheduler_passes_through(self):
        result = self._run(scheduler="no_such_scheduler_xyz")
        assert isinstance(result[11], str)
        assert result[11] == "no_such_scheduler_xyz"

    def test_no_pipe_no_sampler_gets_default(self):
        result = self._run()
        assert isinstance(result[10], str)
        assert len(result[10]) > 0

    def test_no_pipe_no_scheduler_gets_default(self):
        result = self._run()
        assert isinstance(result[11], str)
        assert len(result[11]) > 0


# ── process_pipe — loader context propagation ─────────────────────────────────

class TestProcessPipeLoaderContext:
    def _run(self, pipe):
        node = VNCCS_Pipe()
        return node.process_pipe(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=pipe,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )

    def test_propagates_loader_type(self):
        pipe = _make_pipe(loader_type="nunchaku")
        result = self._run(pipe)
        assert result[9].loader_type == "nunchaku"

    def test_propagates_nunchaku_kind(self):
        pipe = _make_pipe(nunchaku_kind="qwen-image")
        result = self._run(pipe)
        assert result[9].nunchaku_kind == "qwen-image"

    def test_propagates_model_entry(self):
        entry = {"name": "mymodel", "local_path": "models/unet/x.safetensors"}
        pipe = _make_pipe(model_entry=entry)
        result = self._run(pipe)
        assert result[9].model_entry is entry


# ── return tuple shape ────────────────────────────────────────────────────────

class TestProcessPipeReturnShape:
    def test_declares_sampler_outputs_as_strings(self):
        assert VNCCS_Pipe.RETURN_TYPES[10] == "STRING"
        assert VNCCS_Pipe.RETURN_TYPES[11] == "STRING"
        assert VNCCSSamplerSchedulerPicker.RETURN_TYPES == ("STRING", "STRING")

    def test_accepts_legacy_non_pipe_input_without_inheriting(self):
        node = VNCCS_Pipe()
        result = node.process_pipe(
            model="model", clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe="legacy-string-link",
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        assert result[0] == "model"
        assert result[9] is node

    def test_returns_12_values(self):
        node = VNCCS_Pipe()
        pipe = _make_pipe()
        result = node.process_pipe(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=pipe,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        assert len(result) == 12

    def test_seed_never_none_in_output(self):
        node = VNCCS_Pipe()
        result = node.process_pipe(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=None,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        assert result[5] is not None

    def test_steps_never_none_in_output(self):
        node = VNCCS_Pipe()
        result = node.process_pipe(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=0.0,
            pipe=None,
            sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
            lora_name="none", lora_strength=1.0, lora_options_json='["none"]',
        )
        assert result[6] is not None
