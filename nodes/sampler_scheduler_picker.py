"""Sampler and scheduler picker node with dynamic ComfyUI enumeration lookup."""

from __future__ import annotations

import importlib
from typing import List, Tuple

DEFAULT_SAMPLERS = ["euler", "euler_a", "heun"]
DEFAULT_SCHEDULERS = ["normal", "karras", "exponential"]


def fetch_sampler_scheduler_lists() -> Tuple[List[str], List[str]]:
    """Always reads current KSampler lists — safe to call after all extensions load."""
    try:
        comfy_mod = importlib.import_module("comfy")
        samplers_mod = getattr(comfy_mod, "samplers", None)
        ksampler_cls = getattr(samplers_mod, "KSampler", None) if samplers_mod else None
        samplers = list(getattr(ksampler_cls, "SAMPLERS", [])) if ksampler_cls else []
        schedulers = list(getattr(ksampler_cls, "SCHEDULERS", [])) if ksampler_cls else []
        return (samplers or DEFAULT_SAMPLERS, schedulers or DEFAULT_SCHEDULERS)
    except Exception:
        return DEFAULT_SAMPLERS, DEFAULT_SCHEDULERS


class _DynamicReturnTypes:
    """Descriptor: reads KSampler enum lists at access time so RETURN_TYPES stays
    current even after extensions like RES4LYF extend the lists post-import."""
    def __init__(self, template: tuple, sampler_idx: int, scheduler_idx: int):
        self._template = list(template)
        self._si = sampler_idx
        self._sci = scheduler_idx

    def __get__(self, obj, objtype=None):
        samplers, schedulers = fetch_sampler_scheduler_lists()
        rt = list(self._template)
        rt[self._si] = samplers
        rt[self._sci] = schedulers
        return tuple(rt)


# Static fallback used only where a fixed reference is needed (e.g. VALIDATE_INPUTS defaults)
SAMPLER_ENUM, SCHEDULER_ENUM = fetch_sampler_scheduler_lists()


class VNCCSSamplerSchedulerPicker:
    """Expose ComfyUI sampler and scheduler lists as selectable outputs."""

    CATEGORY = "VNCCS"
    RETURN_TYPES = _DynamicReturnTypes(
        (None, None), sampler_idx=0, scheduler_idx=1
    )
    RETURN_NAMES = ("sampler_name", "scheduler")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        sampler_enum, scheduler_enum = fetch_sampler_scheduler_lists()
        default_sampler = sampler_enum[0] if sampler_enum else "euler"
        default_scheduler = scheduler_enum[0] if scheduler_enum else "normal"

        return {
            "required": {
                "sampler_name": (sampler_enum, {"default": default_sampler}),
                "scheduler": (scheduler_enum, {"default": default_scheduler}),
            }
        }

    def pick(self, sampler_name: str, scheduler: str):
        return sampler_name, scheduler


NODE_CLASS_MAPPINGS = {
    "VNCCSSamplerSchedulerPicker": VNCCSSamplerSchedulerPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSSamplerSchedulerPicker": "VNCCS Sampler Scheduler Picker",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSSamplerSchedulerPicker": "VNCCS",
}
