"""Aggregation pipe node for VNCCS with sampler/scheduler support.

Seed logic:
 - incoming pipe & seed==0 -> inherit
 - incoming pipe & seed!=0 -> override
 - no pipe & seed==0 -> keep 0 (no auto randomness)

Sampler / scheduler:
 - Accept rich objects (SAMPLER, SCHEDULER) if connected
 - Fallback to string names when objects not provided
 - Persist both object and name for downstream usage
"""


"""Prepare enumeration lists once for input validation."""
import os
import folder_paths
from .sampler_scheduler_picker import (
    fetch_sampler_scheduler_lists,
    DEFAULT_SAMPLERS,
    DEFAULT_SCHEDULERS,
)
from .vnccs_control_center import (
    _apply_lora_standard,
    _apply_lora_nunchaku,
)


SAMPLER_ENUM, SCHEDULER_ENUM = fetch_sampler_scheduler_lists()

# Sentinel value shown in the widget to mean "inherit this value from the incoming pipe"
PIPE_INHERIT = "(← pipe)"


class VNCCS_Pipe:
    CATEGORY = "VNCCS"
    # NOTE: Outputs must declare type names, not enumeration lists.
    # Using the enumeration lists directly caused UI duplication (width/height twice)
    # because ComfyUI iterated over list elements as separate types. We return STRING
    # while still constraining the input via enumerations so connections remain valid.
    RETURN_TYPES = (
        "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "VNCCS_PIPE",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "model", "clip", "vae", "pos", "neg",
        "seed_int", "steps", "cfg", "denoise", "pipe",
        "sampler_name", "scheduler"
    )
    FUNCTION = "process_pipe"

    @staticmethod
    def _inherit(value, pipe, attr_name, zero_is_empty=True):
        """Inherit value from pipe if current value is the sentinel (None or 0)."""
        if value is None or (zero_is_empty and value == 0):
            return getattr(pipe, attr_name, value) if pipe else value
        return value

    @classmethod
    def INPUT_TYPES(cls):
        sampler_enum, scheduler_enum = fetch_sampler_scheduler_lists()
        # Prepend the inherit sentinel so the widget defaults to "pass through"
        sampler_input = [PIPE_INHERIT] + list(sampler_enum)
        scheduler_input = [PIPE_INHERIT] + list(scheduler_enum)

        return {
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                # 0 = inherit from pipe; any non-zero value overrides
                "seed_int": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sample_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Wildcard keeps old saved workflows from failing validation if
                # this optional passthrough was accidentally linked to a string.
                "pipe": ("*",),
                # PIPE_INHERIT sentinel = pass the pipe value through unchanged
                "sampler_name": (sampler_input, {"default": PIPE_INHERIT}),
                "scheduler": (scheduler_input, {"default": PIPE_INHERIT}),
                "lora_name": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lora_options_json": ("STRING", {"default": '["none"]'}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT, **kwargs):
        valid_samplers = [PIPE_INHERIT] + list(SAMPLER_ENUM)
        valid_schedulers = [PIPE_INHERIT] + list(SCHEDULER_ENUM)
        if sampler_name not in valid_samplers:
            return f"sampler_name '{sampler_name}' is not valid"
        if scheduler not in valid_schedulers:
            return f"scheduler '{scheduler}' is not valid"
        return True

    def process_pipe(self, model=None, clip=None, vae=None, pos=None, neg=None, seed_int=0,
                     sample_steps=0, cfg=0.0, denoise=0.0, pipe=None,
                     sampler_name=PIPE_INHERIT, scheduler=PIPE_INHERIT,
                     lora_name="none", lora_strength=1.0, lora_options_json='["none"]', **_):
        """Aggregate pipe values, inheriting from upstream pipe if not provided."""
        if pipe is not None and not any(hasattr(pipe, attr) for attr in ("model", "clip", "vae", "pos", "neg")):
            pipe = None

        # Inherit object references when not connected
        model = self._inherit(model, pipe, "model", zero_is_empty=False)
        clip = self._inherit(clip, pipe, "clip", zero_is_empty=False)
        vae = self._inherit(vae, pipe, "vae", zero_is_empty=False)
        pos = self._inherit(pos, pipe, "pos", zero_is_empty=False)
        neg = self._inherit(neg, pipe, "neg", zero_is_empty=False)

        # Numeric fields: 0 / 0.0 means "inherit from pipe"
        sample_steps = self._inherit(sample_steps, pipe, "sample_steps")
        cfg = self._inherit(cfg, pipe, "cfg")
        denoise = self._inherit(denoise, pipe, "denoise")

        # seed: 0 means inherit
        if seed_int in (None, 0) and pipe is not None:
            seed_int = getattr(pipe, "seed_int", getattr(pipe, "seed", 0)) or 0

        # sampler/scheduler: PIPE_INHERIT sentinel means pass the pipe value through
        if sampler_name in (None, "", PIPE_INHERIT):
            sampler_name = getattr(pipe, "sampler_name", None) if pipe else None
        if scheduler in (None, "", PIPE_INHERIT):
            scheduler = getattr(pipe, "scheduler", None) if pipe else None

        # Final fallback to first valid value if pipe had nothing
        if not sampler_name or sampler_name == PIPE_INHERIT:
            sampler_name = SAMPLER_ENUM[0] if SAMPLER_ENUM else DEFAULT_SAMPLERS[0]
        if not scheduler or scheduler == PIPE_INHERIT:
            scheduler = SCHEDULER_ENUM[0] if SCHEDULER_ENUM else DEFAULT_SCHEDULERS[0]

        if lora_name and lora_name != "none" and model is not None:
            loader_type = getattr(pipe, "loader_type", "standard") if pipe else "standard"
            full_path = folder_paths.get_full_path("loras", lora_name)
            if full_path and os.path.exists(full_path):
                print(f"[VNCCS Pipe] Applying LoRA: {lora_name} (strength={lora_strength}, loader={loader_type})")
                if loader_type == "nunchaku":
                    model = _apply_lora_nunchaku(
                        model, full_path, lora_strength,
                        settings=getattr(pipe, "nunchaku_settings", None) or {},
                        model_entry=getattr(pipe, "model_entry", None),
                    )
                else:
                    model, clip = _apply_lora_standard(model, clip, full_path, lora_strength)
            else:
                print(f"[VNCCS Pipe] LoRA not found on disk: '{lora_name}', skipping.")

        # Store on self for return
        self.model = model
        self.clip = clip
        self.vae = vae
        self.pos = pos
        self.neg = neg
        self.seed_int = seed_int
        self.sample_steps = sample_steps
        self.cfg = cfg
        self.denoise = denoise
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.loader_type = getattr(pipe, "loader_type", None)
        self.nunchaku_kind = getattr(pipe, "nunchaku_kind", None)
        self.nunchaku_settings = getattr(pipe, "nunchaku_settings", None)
        self.model_entry = getattr(pipe, "model_entry", None)
        self.repo_id = getattr(pipe, "repo_id", None)
        self.lora_entries = getattr(pipe, "lora_entries", [])
        self.lora_states = getattr(pipe, "lora_states", [])

        return (
            self.model,
            self.clip,
            self.vae,
            self.pos,
            self.neg,
            self.seed_int if self.seed_int is not None else 0,
            self.sample_steps if self.sample_steps is not None else 0,
            self.cfg,
            self.denoise,
            self,
            self.sampler_name or "",
            self.scheduler or "",
        )


# Registration mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VNCCS_Pipe": VNCCS_Pipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Pipe": "VNCCS Pipe",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_Pipe": "VNCCS",
}
