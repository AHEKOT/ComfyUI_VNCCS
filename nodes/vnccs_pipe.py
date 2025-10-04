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

    def __init__(self):
        self.model = None
        self.clip = None
        self.vae = None
        self.pos = None
        self.neg = None
        self.seed_int = None
        self.sample_steps = None
        self.cfg = None
        self.denoise = None
        self.sampler_name = None
        self.scheduler = None

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
        # 3. Otherwise remain 0 (caller downstream may randomize later).
        if pipe is not None:
            if model is None: model = getattr(pipe, "model", None)
            if clip is None: clip = getattr(pipe, "clip", None)
            if vae is None: vae = getattr(pipe, "vae", None)
            if pos is None: pos = getattr(pipe, "pos", None)
            if neg is None: neg = getattr(pipe, "neg", None)
            if sample_steps is None or sample_steps == 0: sample_steps = getattr(pipe, "sample_steps", None)
            if cfg is None or cfg == 0: cfg = getattr(pipe, "cfg", None)
            if denoise is None or denoise == 0: denoise = getattr(pipe, "denoise", None)
            default_sampler_name = SAMPLER_ENUM[0] if SAMPLER_ENUM else None
            default_scheduler_name = SCHEDULER_ENUM[0] if SCHEDULER_ENUM else None
            if (sampler_name in (None, "", default_sampler_name)):
                sampler_name = getattr(pipe, "sampler_name", sampler_name)
            if (scheduler in (None, "", default_scheduler_name)):
                scheduler = getattr(pipe, "scheduler", scheduler)
            if seed_int in (None, 0):
                upstream_val = getattr(pipe, "seed_int", getattr(pipe, "seed", 0))
                if upstream_val not in (None, 0):
                    seed_int = upstream_val
            else:
                # Propagate override downstream pipe immediately so deeper chains see it
                try:
                    pipe.seed_int = seed_int
                except Exception:
                    try:
                        pipe.seed = seed_int
                    except Exception:
                        pass

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

        if pipe is not None:
            pipe.model = self.model
            pipe.clip = self.clip
            pipe.vae = self.vae
            pipe.pos = self.pos
            pipe.neg = self.neg
            pipe.sample_steps = self.sample_steps
            pipe.cfg = self.cfg
            pipe.denoise = self.denoise
            pipe.sampler_name = self.sampler_name
            pipe.scheduler = self.scheduler

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
