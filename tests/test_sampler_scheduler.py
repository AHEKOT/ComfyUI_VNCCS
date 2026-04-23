"""Tests for nodes/sampler_scheduler_picker.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.sampler_scheduler_picker import (
    fetch_sampler_scheduler_lists,
    _DynamicReturnTypes,
    DEFAULT_SAMPLERS,
    DEFAULT_SCHEDULERS,
    VNCCSSamplerSchedulerPicker,
)


# ── fetch_sampler_scheduler_lists ─────────────────────────────────────────────

class TestFetchSamplerSchedulerLists:
    def test_returns_two_lists(self):
        samplers, schedulers = fetch_sampler_scheduler_lists()
        assert isinstance(samplers, list)
        assert isinstance(schedulers, list)

    def test_samplers_not_empty(self):
        samplers, _ = fetch_sampler_scheduler_lists()
        assert len(samplers) > 0

    def test_schedulers_not_empty(self):
        _, schedulers = fetch_sampler_scheduler_lists()
        assert len(schedulers) > 0

    def test_returns_strings(self):
        samplers, schedulers = fetch_sampler_scheduler_lists()
        assert all(isinstance(s, str) for s in samplers)
        assert all(isinstance(s, str) for s in schedulers)

    def test_includes_euler(self):
        samplers, _ = fetch_sampler_scheduler_lists()
        assert "euler" in samplers

    def test_includes_karras(self):
        _, schedulers = fetch_sampler_scheduler_lists()
        assert "karras" in schedulers

    def test_falls_back_to_defaults_on_comfy_failure(self, monkeypatch):
        import importlib
        monkeypatch.setattr(importlib, "import_module", lambda *a, **kw: (_ for _ in ()).throw(ImportError()))
        samplers, schedulers = fetch_sampler_scheduler_lists()
        assert samplers == DEFAULT_SAMPLERS
        assert schedulers == DEFAULT_SCHEDULERS


# ── _DynamicReturnTypes descriptor ───────────────────────────────────────────

class TestDynamicReturnTypes:
    def test_returns_tuple(self):
        drt = _DynamicReturnTypes(("MODEL", None, None), sampler_idx=1, scheduler_idx=2)

        class Holder:
            RETURN_TYPES = drt

        rt = Holder.RETURN_TYPES
        assert isinstance(rt, tuple)

    def test_fills_sampler_slot(self):
        drt = _DynamicReturnTypes(("MODEL", None, None), sampler_idx=1, scheduler_idx=2)

        class Holder:
            RETURN_TYPES = drt

        rt = Holder.RETURN_TYPES
        samplers, _ = fetch_sampler_scheduler_lists()
        assert rt[1] == samplers

    def test_fills_scheduler_slot(self):
        drt = _DynamicReturnTypes(("MODEL", None, None), sampler_idx=1, scheduler_idx=2)

        class Holder:
            RETURN_TYPES = drt

        rt = Holder.RETURN_TYPES
        _, schedulers = fetch_sampler_scheduler_lists()
        assert rt[2] == schedulers

    def test_non_sampler_slots_preserved(self):
        drt = _DynamicReturnTypes(("MODEL", None, None), sampler_idx=1, scheduler_idx=2)

        class Holder:
            RETURN_TYPES = drt

        assert Holder.RETURN_TYPES[0] == "MODEL"

    def test_reads_current_list_after_extension(self, monkeypatch):
        import comfy.samplers
        original = list(comfy.samplers.KSampler.SAMPLERS)
        comfy.samplers.KSampler.SAMPLERS = original + ["my_custom_sampler"]

        drt = _DynamicReturnTypes((None, None), sampler_idx=0, scheduler_idx=1)

        class Holder:
            RETURN_TYPES = drt

        rt = Holder.RETURN_TYPES
        assert "my_custom_sampler" in rt[0]

        # Restore
        comfy.samplers.KSampler.SAMPLERS = original


# ── VNCCSSamplerSchedulerPicker.pick ─────────────────────────────────────────

class TestSamplerSchedulerPickerPick:
    def test_returns_passed_values(self):
        picker = VNCCSSamplerSchedulerPicker()
        result = picker.pick("euler", "karras")
        assert result == ("euler", "karras")

    def test_any_string_passthrough(self):
        picker = VNCCSSamplerSchedulerPicker()
        result = picker.pick("dpm_2", "normal")
        assert result[0] == "dpm_2"
        assert result[1] == "normal"
