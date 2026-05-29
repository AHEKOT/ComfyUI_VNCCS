"""Sampler and scheduler picker node with dynamic ComfyUI enumeration lookup."""

from __future__ import annotations

import importlib
from typing import List, Tuple

DEFAULT_SAMPLERS = ["euler", "euler_a", "heun"]
DEFAULT_SCHEDULERS = ["normal", "karras", "exponential"]


def fetch_sampler_scheduler_lists() -> Tuple[List[str], List[str]]:
    try:
        comfy_mod = importlib.import_module("comfy")
        samplers_mod = getattr(comfy_mod, "samplers", None)
        ksampler_cls = getattr(samplers_mod, "KSampler", None) if samplers_mod else None
        samplers = list(getattr(ksampler_cls, "SAMPLERS", [])) if ksampler_cls else []
        schedulers = list(getattr(ksampler_cls, "SCHEDULERS", [])) if ksampler_cls else []
        return (samplers or DEFAULT_SAMPLERS, schedulers or DEFAULT_SCHEDULERS)
    except Exception:
        return DEFAULT_SAMPLERS, DEFAULT_SCHEDULERS


SAMPLER_ENUM, SCHEDULER_ENUM = fetch_sampler_scheduler_lists()


class FlexibleEnumOutput(str):
    """Output type compatible with ComfyUI enum-list inputs.

    ComfyUI validates links by comparing output_type != input_type. KSampler
    sampler/scheduler inputs are declared as lists of allowed strings, so a
    plain "STRING" output is rejected. A str subclass with permissive __ne__
    keeps the output serializable while allowing links to dynamic enum inputs.
    """

    def __ne__(self, other):
        return False


SAMPLER_OUTPUT_TYPE = FlexibleEnumOutput("SAMPLER_NAME")
SCHEDULER_OUTPUT_TYPE = FlexibleEnumOutput("SCHEDULER_NAME")


class VNCCSSamplerSchedulerPicker:
    """Expose ComfyUI sampler and scheduler lists as selectable outputs."""

    CATEGORY = "VNCCS"
    RETURN_TYPES = (SAMPLER_OUTPUT_TYPE, SCHEDULER_OUTPUT_TYPE)
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
