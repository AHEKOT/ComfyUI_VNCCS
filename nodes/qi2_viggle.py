"""Viggle Turbo sigma schedule for Qwen Image 2.1."""

import math

import torch


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
