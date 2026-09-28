"""Viggle Turbo adapter for Qwen Image 2.1.

The schedule and unmerged LoRA application follow Viggle's ComfyUI reference:
https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py
"""

import json
import math

import torch
import torch.nn.functional as F


VIGGLE_TURBO_NODES = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)


def viggle_turbo_sigmas(latent):
    """Shift the six raw student nodes using the sampler latent's token count."""
    samples = latent["samples"]
    spatial_ratio = latent.get("downscale_ratio_spacial", 16) / 16
    tokens = round(samples.shape[-2] * spatial_ratio) * round(samples.shape[-1] * spatial_ratio)
    shift = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
    nodes = torch.tensor(VIGGLE_TURBO_NODES, dtype=torch.float64)
    shifted = math.exp(shift) / (math.exp(shift) + (1 / nodes - 1))
    return torch.cat((shifted, shifted.new_zeros(1))).float()


class ViggleDetailerSchedule:
    """Bridge Viggle's latent-aware schedule into Impact Pack FaceDetailer.

    Impact Pack asks a scheduler function for sigmas after its detailer hook has
    encoded the detected crop.  Keeping that latent on the hook lets us reuse
    the exact Viggle shift calculation for every face crop instead of guessing
    its dimensions from the source image.
    """

    def __init__(self):
        self.latent = None

    def scheduler_func(self, _model, _sampler_name, _steps):
        if self.latent is None:
            raise RuntimeError("Viggle FaceDetailer schedule was requested before the crop latent was encoded.")
        return viggle_turbo_sigmas(self.latent)

    def touch_scaled_size(self, width, height):
        return width, height

    def post_upscale(self, image, _noise_mask):
        return image

    def get_skip_sampling(self):
        return False

    def post_encode(self, latent):
        self.latent = latent
        return latent

    def get_custom_sampler(self):
        return None

    def set_steps(self, _steps):
        return None

    def cycle_latent(self, latent):
        return latent

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise):
        self.latent = latent
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise

    def get_custom_noise(self, _seed, noise, is_touched=False):
        return noise, is_touched

    def pre_decode(self, latent):
        return latent

    def post_decode(self, image):
        return image

    def post_crop_region(self, _width, _height, _bbox, crop_region):
        return crop_region

    def post_detection(self, segs):
        return segs


def _lora_linear(tensor, pair):
    down, up = pair
    return F.linear(F.linear(tensor, down.to(tensor.dtype)), up.to(tensor.dtype))


def _attach_fused_mlp_hooks(mlp, gate, up, down):
    intermediate = {}

    def gate_up_hook(_module, inputs, output):
        combined = output + torch.cat(
            (_lora_linear(inputs[0], gate), _lora_linear(inputs[0], up)), dim=-1,
        )
        intermediate["gate_up"] = combined
        return combined

    def mlp_hook(_module, _inputs, output):
        gate_value, up_value = intermediate.pop("gate_up").chunk(2, dim=-1)
        return output + _lora_linear(F.silu(gate_value) * up_value, down)

    return [
        mlp.gate_up.register_forward_hook(gate_up_hook),
        mlp.register_forward_hook(mlp_hook),
    ]


def _run_with_viggle_lora(weights, executor, *args, **kwargs):
    diffusion_model = executor.class_obj
    device = args[0].device
    for pair in weights.values():
        if pair[0].device != device:
            pair[0], pair[1] = pair[0].to(device), pair[1].to(device)

    hooks = []
    try:
        for name, pair in weights.items():
            parent_name, _, leaf = name.rpartition(".")
            parent = diffusion_model.get_submodule(parent_name)
            if not getattr(parent, "fused", False):
                hooks.append(diffusion_model.get_submodule(name).register_forward_hook(
                    lambda _module, inputs, output, adapter=pair:
                    output + _lora_linear(inputs[0], adapter)
                ))
            elif leaf == "out":
                hooks.extend(_attach_fused_mlp_hooks(
                    parent,
                    weights[parent_name + ".gate_layer"],
                    weights[parent_name + ".proj"],
                    pair,
                ))
        return executor(*args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()


def apply_viggle_turbo_lora(model, lora_name, strength=1.0):
    """Attach the adapter as a model execution wrapper without merging weights."""
    import comfy.patcher_extension
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    adapter_metadata = json.loads((metadata or {}).get("lora_adapter_metadata", "{}"))
    alpha = adapter_metadata.get("transformer.lora_alpha", 1)
    rank = adapter_metadata.get("transformer.r", 1)
    scale = float(strength) * float(alpha) / float(rank)
    weights = {}
    for key, down in state_dict.items():
        if not key.endswith(".lora_A.weight"):
            continue
        up_key = key.replace(".lora_A.weight", ".lora_B.weight")
        name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        weights[name] = [down, state_dict[up_key] * scale]
    if not weights:
        raise ValueError("Viggle Turbo LoRA contains no compatible transformer adapter weights.")

    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "vnccs_qi2_viggle_turbo",
        lambda executor, *args, **kwargs: _run_with_viggle_lora(weights, executor, *args, **kwargs),
    )
    return patched
