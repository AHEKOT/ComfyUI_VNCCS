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


"""Prepare enumeration lists once so both inputs and outputs share identical types."""
from .sampler_scheduler_picker import (
    fetch_sampler_scheduler_lists,
    DEFAULT_SAMPLERS,
    DEFAULT_SCHEDULERS,
)


SAMPLER_ENUM, SCHEDULER_ENUM = fetch_sampler_scheduler_lists()


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
        SAMPLER_ENUM,
        SCHEDULER_ENUM,
    )
    RETURN_NAMES = (
        "model", "clip", "vae", "pos", "neg",
        "seed_int", "steps", "cfg", "denoise", "pipe",
        "sampler_name", "scheduler"
    )
    FUNCTION = "process_pipe"

    @staticmethod
    def _inherit(value, pipe, attr_name, zero_is_empty=True):
        """Inherit value from pipe if current value is empty."""
        if value is None or (zero_is_empty and value == 0):
            return getattr(pipe, attr_name, value) if pipe else value
        return value

    @classmethod
    def INPUT_TYPES(cls):
        sampler_enum, scheduler_enum = fetch_sampler_scheduler_lists()

        return {
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "seed_int": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sample_steps": ("INT", {"default": None}),
                "cfg": ("FLOAT", {"default": None}),
                "denoise": ("FLOAT", {"default": None}),
                "pipe": ("VNCCS_PIPE",),
                "sampler_name": (sampler_enum, {"default": sampler_enum[0] if sampler_enum else DEFAULT_SAMPLERS[0]}),
                "scheduler": (scheduler_enum, {"default": scheduler_enum[0] if scheduler_enum else DEFAULT_SCHEDULERS[0]}),
            }
        }

    def process_pipe(self, model=None, clip=None, vae=None, pos=None, neg=None, seed_int=0,
                     sample_steps=None, cfg=None, denoise=None, pipe=None,
                     sampler_name=None, scheduler=None):
        """Aggregate pipe values, inheriting from upstream pipe if not provided."""
        # Inherit from upstream pipe if values are empty
        model = self._inherit(model, pipe, "model", zero_is_empty=False)
        clip = self._inherit(clip, pipe, "clip", zero_is_empty=False)
        vae = self._inherit(vae, pipe, "vae", zero_is_empty=False)
        pos = self._inherit(pos, pipe, "pos", zero_is_empty=False)
        neg = self._inherit(neg, pipe, "neg", zero_is_empty=False)
        sample_steps = self._inherit(sample_steps, pipe, "sample_steps")
        cfg = self._inherit(cfg, pipe, "cfg")
        denoise = self._inherit(denoise, pipe, "denoise")

        # Handle seed with fallback to legacy "seed" attribute
        if seed_int in (None, 0) and pipe is not None:
            seed_int = getattr(pipe, "seed_int", getattr(pipe, "seed", 0)) or 0

        # Handle sampler/scheduler with default comparison
        default_sampler = SAMPLER_ENUM[0] if SAMPLER_ENUM else None
        default_scheduler = SCHEDULER_ENUM[0] if SCHEDULER_ENUM else None
        if sampler_name in (None, "", default_sampler) and pipe:
            sampler_name = getattr(pipe, "sampler_name", sampler_name)
        if scheduler in (None, "", default_scheduler) and pipe:
            scheduler = getattr(pipe, "scheduler", scheduler)

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
