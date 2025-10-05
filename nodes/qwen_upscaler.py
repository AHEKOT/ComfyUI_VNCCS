"""VNCCS QWEN Upscaler node.

This node is inspired by the Ultimate SD Upscale workflow but targets the
Qwen2 Image Edit (2509) pipeline. It prepares QWEN-friendly conditioning for
individual tiles, runs the sampler per tile, and recombines the results into a
single upscaled image.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

import comfy.utils
import node_helpers
from nodes import VAEDecode, common_ksampler

from .sampler_scheduler_picker import (
    DEFAULT_SAMPLERS,
    DEFAULT_SCHEDULERS,
    fetch_sampler_scheduler_lists,
)

DEFAULT_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture,"
    " objects, background), then explain how the user's text instruction should"
    " alter or modify the image. Generate a new image that meets the user's"
    " requirements while maintaining consistency with the original input where"
    " appropriate."
)

DEFAULT_UPSCALE_METHOD = "lanczos"
DEFAULT_CROP_MODE = "disabled"

QUALITY_PRESETS = (1024, 1344, 1536, 2048)
SCALE_PRESETS = (1.0, 2.0, 4.0, 8.0)
DEFAULT_SEAM_PROMPT = "fix seam line and blend. Draw image without seam lines."


def _snap_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _snap_down_to_multiple(value: int, multiple: int, minimum: int) -> int:
    if value <= 0:
        return minimum
    snapped = (value // multiple) * multiple
    return max(minimum, snapped)


def _sanitize_instruction(raw: str) -> str:
    """Ensure the system instruction fits the template expected by QWEN."""
    template_prefix = "<|im_start|>system\n"
    template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    if not raw:
        cleaned = DEFAULT_SYSTEM_PROMPT
    else:
        cleaned = raw
        if template_prefix in cleaned:
            cleaned = cleaned.split(template_prefix)[-1]
        if template_suffix in cleaned:
            cleaned = cleaned.split(template_suffix)[0]
        cleaned = cleaned.replace("{}", "").strip()
        if not cleaned:
            cleaned = DEFAULT_SYSTEM_PROMPT

    return template_prefix + cleaned + template_suffix


def _resize_for_latent(
    tensor_nhwc: torch.Tensor,
    target_area: int,
    snap_multiple: int = 8,
    max_dim: int | None = None,
) -> torch.Tensor:
    """Resize a tile tensor to the desired latent resolution."""
    samples = tensor_nhwc.movedim(-1, 1)
    current_total = samples.shape[2] * samples.shape[3]
    if current_total == 0:
        raise ValueError("Tile has zero area; cannot upscale")
    scale_by = math.sqrt(target_area / current_total)
    width = max(snap_multiple, round(samples.shape[3] * scale_by / snap_multiple) * snap_multiple)
    height = max(snap_multiple, round(samples.shape[2] * scale_by / snap_multiple) * snap_multiple)
    if max_dim is not None and max_dim > 0:
        max_dim = max(snap_multiple, min(int(max_dim), 4096))
        max_side = max(width, height)
        if max_side > max_dim:
            ratio = max_dim / float(max_side)
            width = max(snap_multiple, round(width * ratio / snap_multiple) * snap_multiple)
            height = max(snap_multiple, round(height * ratio / snap_multiple) * snap_multiple)
            width = min(width, max_dim)
            height = min(height, max_dim)
    upscaled = comfy.utils.common_upscale(
        samples,
        width,
        height,
        DEFAULT_UPSCALE_METHOD,
        DEFAULT_CROP_MODE,
    )
    return upscaled.movedim(1, -1)


def _build_qwen_conditioning(
    clip,
    vae,
    tile: torch.Tensor,
    prompt: str,
    system_prompt: str,
    pass_prompt: str | None,
    quality: int,
    vision_size: int,
) -> Tuple[List, torch.Tensor]:
    """Create QWEN conditioning and obtain the starting latent for a tile."""
    llama_template = _sanitize_instruction(system_prompt)

    # Prepare the images for CLIP + reference latent attachment
    target_area = quality * quality
    vl_area = vision_size * vision_size

    latent_ready = _resize_for_latent(tile, target_area, snap_multiple=8, max_dim=quality)
    vl_ready = _resize_for_latent(tile, vl_area, snap_multiple=28, max_dim=vision_size)

    ref_latent = vae.encode(latent_ready[:, :, :, :3])
    ref_latents = [ref_latent]

    image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    full_prompt = image_prompt + (pass_prompt if pass_prompt else prompt)
    tokens = clip.tokenize(full_prompt, images=[vl_ready], llama_template=llama_template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": ref_latents},
        append=True,
    )

    # vae.encode returns a latent tensor; wrap in dict later for samplers
    return conditioning, ref_latent


def _build_negative_conditioning(clip, prompt: str) -> List:
    tokens = clip.tokenize(prompt or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _apply_mask_blur(mask: torch.Tensor, blur: int) -> torch.Tensor:
    if blur <= 0:
        return mask
    kernel = blur * 2 + 1
    if kernel < 3:
        return mask
    if mask.shape[-2] <= blur or mask.shape[-1] <= blur:
        return mask
    padded = F.pad(mask, (blur, blur, blur, blur), mode="reflect")
    blurred = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return blurred


def _create_mask(
    height: int,
    width: int,
    mode: str | None,
    seam_width: int,
    blur: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    if mode == "feather" and seam_width > 0:
        y = torch.arange(height, device=device, dtype=dtype)
        x = torch.arange(width, device=device, dtype=dtype)
        dist_top = y
        dist_bottom = (height - 1) - y
        dist_left = x
        dist_right = (width - 1) - x
        min_vertical = torch.minimum(dist_top, dist_bottom)[:, None]
        min_horizontal = torch.minimum(dist_left, dist_right)[None, :]
        edge_distance = torch.minimum(min_vertical, min_horizontal)
        mask = torch.clamp(edge_distance / float(max(seam_width, 1)), min=0.0, max=1.0)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mode in {"horizontal", "vertical", "radial"} and seam_width > 0:
        if mode == "horizontal":
            center = (height - 1) / 2.0
            y = torch.arange(height, device=device, dtype=dtype)
            dist = torch.abs(y - center)[None, None, :, None]
            falloff = torch.clamp(1.0 - dist / float(seam_width), min=0.0, max=1.0)
            mask = falloff.expand(1, 1, height, width)
        elif mode == "vertical":
            center = (width - 1) / 2.0
            x = torch.arange(width, device=device, dtype=dtype)
            dist = torch.abs(x - center)[None, None, None, :]
            falloff = torch.clamp(1.0 - dist / float(seam_width), min=0.0, max=1.0)
            mask = falloff.expand(1, 1, height, width)
        elif mode == "radial":
            cy = (height - 1) / 2.0
            cx = (width - 1) / 2.0
            y = torch.arange(height, device=device, dtype=dtype) - cy
            x = torch.arange(width, device=device, dtype=dtype) - cx
            yy = y.view(-1, 1)
            xx = x.view(1, -1)
            dist = torch.sqrt(yy * yy + xx * xx)[None, None, :, :]
            falloff = torch.clamp(1.0 - dist / float(seam_width), min=0.0, max=1.0)
            mask = falloff

    mask = _apply_mask_blur(mask, blur)
    return mask.clamp(0.0, 1.0)


def _stable_tile_seed(base_seed: int, x: int, y: int, pass_order: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    value = (base_seed ^ 0x9E3779B97F4A7C15) & mask
    value ^= ((x & 0xFFFFFFFF) * 0x100000001B3) & mask
    value = (value + ((y & 0xFFFFFFFF) * 0x1B873593)) & mask
    value ^= ((pass_order + 1) * 0x85EBCA6B) & mask
    return value & mask


def _compute_global_stats(
    tensor_cf: torch.Tensor,
    valid_height: int,
    valid_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    region = tensor_cf[:, :, :valid_height, :valid_width]
    mean = region.mean(dim=(2, 3), keepdim=True)
    std = region.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
    return mean, std


def _normalize_tile_to_global(
    tile_cf: torch.Tensor,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
) -> torch.Tensor:
    tile_mean = tile_cf.mean(dim=(2, 3), keepdim=True)
    tile_std = tile_cf.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
    adjusted = (tile_cf - tile_mean) * (global_std / tile_std) + global_mean
    return adjusted.clamp(0.0, 1.0)


def _compute_low_frequency_guidance(
    tensor_cf: torch.Tensor,
    levels: int = 2,
) -> torch.Tensor:
    guidance = tensor_cf
    for _ in range(max(1, levels)):
        pooled = F.avg_pool2d(guidance, kernel_size=5, stride=2, padding=2)
        guidance = F.interpolate(pooled, size=tensor_cf.shape[-2:], mode="bilinear", align_corners=False)
    return guidance


def _build_seam_grid_mask(
    height: int,
    width: int,
    tile_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    stripe_width = max(1, tile_size // 16)
    for y in range(tile_size, height, tile_size):
        y0 = max(0, y - stripe_width)
        y1 = min(height, y + stripe_width)
        mask[:, :, y0:y1, :] = 1.0
    for x in range(tile_size, width, tile_size):
        x0 = max(0, x - stripe_width)
        x1 = min(width, x + stripe_width)
        mask[:, :, :, x0:x1] = 1.0
    return mask.clamp(0.0, 1.0)


def _apply_feedback_consistency(
    canvas_cf: torch.Tensor,
    reference_cf: torch.Tensor,
    scale_factor: float,
    tile_size: int,
) -> torch.Tensor:
    if scale_factor <= 1.0:
        return canvas_cf
    ref_h = reference_cf.shape[-2]
    ref_w = reference_cf.shape[-1]
    downscaled = F.interpolate(canvas_cf, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
    diff = reference_cf - downscaled
    seam_mask_low = _build_seam_grid_mask(ref_h, ref_w, tile_size, reference_cf.device, reference_cf.dtype)
    seam_weight = seam_mask_low.sum()
    if float(seam_weight.item()) <= 1e-6:
        return canvas_cf
    correction = (diff * seam_mask_low).sum(dim=(-2, -1), keepdim=True) / (seam_weight + 1e-6)
    seam_mask_high = F.interpolate(seam_mask_low, size=canvas_cf.shape[-2:], mode="bilinear", align_corners=False)
    canvas_cf = canvas_cf + correction * seam_mask_high
    return canvas_cf


class VNCCSQWENUpscaler:
    CATEGORY = "VNCCS"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        sampler_enum, scheduler_enum = fetch_sampler_scheduler_lists()
        default_sampler = sampler_enum[0] if sampler_enum else DEFAULT_SAMPLERS[0]
        default_scheduler = scheduler_enum[0] if scheduler_enum else DEFAULT_SCHEDULERS[0]

        return {
            "required": {
                "scale": (SCALE_PRESETS, {"default": 2.0}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 256}),
                "quality": (QUALITY_PRESETS, {"default": 1024}),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "prompt": ("STRING", {"multiline": True, "default": "upscale image to 4k ultrasharp quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (sampler_enum, {"default": default_sampler}),
                "scheduler": (scheduler_enum, {"default": default_scheduler}),
                "seam_fix": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "vision_tile_size": ("INT", {"default": 384, "min": 256, "max": 512, "step": 8}),
                "denoise": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    def upscale(
        self,
        scale: float,
        image: torch.Tensor,
        model,
        clip,
        vae,
        system_prompt: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        quality: int,
    tile_size: int,
    seam_fix: int,
        vision_tile_size: int,
        denoise: float,
        debug_logging: bool,
        fix_seams: bool = False,
        seam_prompt: str = DEFAULT_SEAM_PROMPT,
        **deprecated_inputs,
    ) -> Tuple[torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("Expected image tensor with shape [B, H, W, C]")
        if image.shape[-1] < 3:
            raise ValueError("QWEN upscaler expects RGB images")

        try:
            quality = int(quality)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid quality level: {quality}") from None
        if quality not in QUALITY_PRESETS:
            raise ValueError(f"Quality must be one of {QUALITY_PRESETS}, got {quality}")

        try:
            scale = float(scale)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid scale factor: {scale}") from None
        if not any(abs(scale - preset) < 1e-6 for preset in SCALE_PRESETS):
            raise ValueError(f"Scale must be one of {SCALE_PRESETS}, got {scale}")

        tile_multiple = 8
        try:
            tile_size = int(tile_size)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid tile size: {tile_size}") from None
        tile_size = max(512, tile_size)
        if tile_size % tile_multiple != 0:
            tile_size = _snap_to_multiple(tile_size, tile_multiple)

        try:
            seam_fix = int(seam_fix)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid seam fix padding: {seam_fix}") from None
        seam_fix = max(0, seam_fix)
        seam_fix = min(seam_fix, tile_size)
        tile_context = seam_fix

        def log(*parts):
            if debug_logging:
                print("[VNCCS QWEN Upscaler]", *parts)

        batch_size = image.shape[0]
        original_sizes: List[Tuple[int, int]] = []
        for batch_index in range(batch_size):
            sample = image[batch_index]
            original_sizes.append((int(sample.shape[0]), int(sample.shape[1])))
        image_min = float(image.min().item())
        image_max = float(image.max().item())
        log(
            "received input tensor:",
            f"batch={batch_size}",
            f"shape={tuple(image.shape)}",
            f"dtype={image.dtype}",
            f"range=[{image_min:.4f}, {image_max:.4f}]",
        )

        if deprecated_inputs:
            log(
                "deprecated inputs ignored:",
                ", ".join(sorted(deprecated_inputs.keys())),
            )

        prompt_clean = (prompt or "").strip()
        seam_prompt_effective = (seam_prompt or "").strip() or DEFAULT_SEAM_PROMPT

        seam_fix_requested = bool(fix_seams)
        seam_fix_enabled = False
        log("seam repair disabled (deprecated)")
        if seam_fix_requested:
            log("deprecated input 'fix_seams' was requested but is disabled")
        if seam_prompt and seam_prompt_effective != DEFAULT_SEAM_PROMPT:
            log("deprecated input 'seam_prompt' ignored")

        vae_decoder = VAEDecode()
        negative = _build_negative_conditioning(clip, negative_prompt)

        if abs(scale - 1.0) < 1e-6:
            pass_scales = [1.0]
        elif abs(scale - 2.0) < 1e-6:
            pass_scales = [2.0]
        elif abs(scale - 4.0) < 1e-6:
            pass_scales = [2.0, 2.0]
        elif abs(scale - 8.0) < 1e-6:
            pass_scales = [2.0, 2.0, 2.0]
        else:
            raise ValueError(f"Unsupported scale factor for QWEN upscaler: {scale}")

        current_image = image
        total_scale = 1.0
        for s in pass_scales:
            total_scale *= s

        with torch.no_grad():
            for pass_index, scale_factor in enumerate(pass_scales, start=1):
                log(
                    f"=== upscale pass {pass_index}/{len(pass_scales)} ===",
                    f"scale={scale_factor}",
                )

                log(
                    "tile geometry:",
                    f"selected={tile_size}",
                    f"multiple={tile_multiple}",
                )
                if tile_context > 0:
                    log("seam fix padding:", f"{tile_context} px")

                pass_outputs: List[torch.Tensor] = []
                batch_size = current_image.shape[0]

                for batch_index in range(batch_size):
                    base = current_image[batch_index : batch_index + 1, :, :, :3]
                    base_cf = base.movedim(-1, 1)
                    base_height = base_cf.shape[-2]
                    base_width = base_cf.shape[-1]

                    orig_height = base_height
                    orig_width = base_width

                    tiles_y = math.ceil(orig_height / tile_size)
                    tiles_x = math.ceil(orig_width / tile_size)
                    work_height = tiles_y * tile_size
                    work_width = tiles_x * tile_size
                    total_tiles = tiles_x * tiles_y
                    pad_bottom = work_height - orig_height
                    pad_right = work_width - orig_width
                    if pad_bottom > 0 or pad_right > 0:
                        base_cf = F.pad(base_cf, (0, pad_right, 0, pad_bottom), mode="replicate")
                        log(
                            "applied padding:",
                            f"right={pad_right}",
                            f"bottom={pad_bottom}",
                            f"work_size={work_width}x{work_height}",
                            f"tensor_shape={tuple(base_cf.shape)}",
                        )

                    global_mean = None
                    global_std = None
                    low_freq_cf = None
                    seam_guidance_mask = None
                    noise_reproj_alpha = 0.18 if seam_fix_enabled else 0.0
                    if seam_fix_enabled:
                        global_mean, global_std = _compute_global_stats(base_cf, orig_height, orig_width)
                        low_freq_cf = _compute_low_frequency_guidance(base_cf)
                        seam_guidance_mask = _build_seam_grid_mask(
                            base_cf.shape[-2],
                            base_cf.shape[-1],
                            tile_size,
                            base_cf.device,
                            base_cf.dtype,
                        )
                        log(
                            "seam mask prepared:",
                            f"shape={tuple(seam_guidance_mask.shape)}",
                            f"device={seam_guidance_mask.device}",
                        )
                        log(
                            "ctc conditioning:",
                            "global stats + low-freq guidance active",
                            f"alpha={noise_reproj_alpha:.2f}",
                        )

                    final_height = max(1, int(round(work_height * scale_factor)))
                    final_width = max(1, int(round(work_width * scale_factor)))
                    target_height = max(1, int(round(orig_height * scale_factor)))
                    target_width = max(1, int(round(orig_width * scale_factor)))
                    if final_height != target_height or final_width != target_width:
                        log(
                            "canvas clamp to target:",
                            f"requested={final_width}x{final_height}",
                            f"target={target_width}x{target_height}",
                        )
                    final_height = target_height
                    final_width = target_width
                    log("processing batch index", batch_index)
                    log("base input size:", f"{orig_width}x{orig_height}")
                    if work_height != orig_height or work_width != orig_width:
                        log("working size:", f"{work_width}x{work_height}")
                    log(
                        "target resolution:",
                        f"{final_width}x{final_height} (scale={scale_factor:.2f})",
                    )
                    log(
                        "tile grid:",
                        f"{tiles_x} columns x {tiles_y} rows = {total_tiles} tiles",
                    )

                    canvas = torch.zeros(
                        (1, base_cf.shape[1], final_height, final_width),
                        dtype=base_cf.dtype,
                        device=base_cf.device,
                    )
                    weights = torch.zeros(
                        (1, 1, final_height, final_width),
                        dtype=base_cf.dtype,
                        device=base_cf.device,
                    )

                    base_tiles: List[dict] = []
                    for y in range(0, work_height, tile_size):
                        for x in range(0, work_width, tile_size):
                            crop_h = min(tile_size, work_height - y)
                            crop_w = min(tile_size, work_width - x)
                            if crop_h <= 0 or crop_w <= 0:
                                continue
                            if debug_logging:
                                log(
                                    "tile window prepared:",
                                    f"origin=({x},{y})",
                                    f"size={crop_w}x{crop_h}",
                                    f"seam_fix={tile_context}px",
                                )
                            base_tiles.append(
                                {
                                    "x": x,
                                    "y": y,
                                    "w": crop_w,
                                    "h": crop_h,
                                    "padding": tile_context,
                                    "orientation": None,
                                    "mask_width": 0,
                                    "mask_blur": 0,
                                    "source": "base",
                                }
                            )

                    passes: List[dict] = [
                        {
                            "name": "base",
                            "denoise": denoise,
                            "tiles": base_tiles,
                        }
                    ]
                    log("base pass prepared:", f"{len(base_tiles)} tiles")

                    seam_tiles_total = 0
                    if seam_fix_enabled:
                        seam_denoise = 1
                        seam_width_px = max(32, min(tile_size // 2, 128))
                        seam_padding_px = max(16, min(tile_size // 4, 64))
                        seam_blur_px = max(8, min(tile_size // 4, 48))
                        log(
                            "seam settings:",
                            f"denoise={seam_denoise}",
                            f"width={seam_width_px}",
                            f"padding={seam_padding_px}",
                            f"mask_blur={seam_blur_px}",
                        )

                        band_tiles: List[dict] = []
                        stripe_w = min(seam_width_px, work_width)
                        stripe_h = min(seam_width_px, work_height)

                        for seam_x in range(tile_size, work_width, tile_size):
                            x0 = seam_x - stripe_w // 2
                            x0 = max(0, min(x0, work_width - stripe_w))
                            crop_w = min(stripe_w, work_width - x0)
                            if crop_w <= 0:
                                continue
                            band_tiles.append(
                                {
                                    "x": x0,
                                    "y": 0,
                                    "w": crop_w,
                                    "h": work_height,
                                    "padding": seam_padding_px,
                                    "orientation": "vertical",
                                    "mask_width": seam_width_px,
                                    "mask_blur": 0,
                                    "source": "canvas",
                                }
                            )

                        for seam_y in range(tile_size, work_height, tile_size):
                            y0 = seam_y - stripe_h // 2
                            y0 = max(0, min(y0, work_height - stripe_h))
                            crop_h = min(stripe_h, work_height - y0)
                            if crop_h <= 0:
                                continue
                            band_tiles.append(
                                {
                                    "x": 0,
                                    "y": y0,
                                    "w": work_width,
                                    "h": crop_h,
                                    "padding": seam_padding_px,
                                    "orientation": "horizontal",
                                    "mask_width": seam_width_px,
                                    "mask_blur": 0,
                                    "source": "canvas",
                                }
                            )

                        if band_tiles:
                            passes.append(
                                {
                                    "name": "seam_band",
                                    "denoise": seam_denoise,
                                    "tiles": band_tiles,
                                }
                            )
                            seam_tiles_total += len(band_tiles)
                            log("band seam tiles prepared:", len(band_tiles))

                        half_tiles: List[dict] = []
                        offset = tile_size // 2 if tile_size >= 2 else 1

                        for seam_y in range(tile_size, work_height, tile_size):
                            y0 = seam_y - offset
                            y0 = max(0, min(y0, work_height - tile_size))
                            crop_h = min(tile_size, work_height - y0)
                            if crop_h <= 0:
                                continue
                            for x0 in range(0, work_width, tile_size):
                                crop_w = min(tile_size, work_width - x0)
                                if crop_w <= 0:
                                    continue
                                half_tiles.append(
                                    {
                                        "x": x0,
                                        "y": y0,
                                        "w": crop_w,
                                        "h": crop_h,
                                        "padding": seam_padding_px,
                                        "orientation": "horizontal",
                                        "mask_width": seam_width_px,
                                        "mask_blur": seam_blur_px,
                                        "source": "canvas",
                                    }
                                )

                        for seam_x in range(tile_size, work_width, tile_size):
                            x0 = seam_x - offset
                            x0 = max(0, min(x0, work_width - tile_size))
                            crop_w = min(tile_size, work_width - x0)
                            if crop_w <= 0:
                                continue
                            for y0 in range(0, work_height, tile_size):
                                crop_h = min(tile_size, work_height - y0)
                                if crop_h <= 0:
                                    continue
                                half_tiles.append(
                                    {
                                        "x": x0,
                                        "y": y0,
                                        "w": crop_w,
                                        "h": crop_h,
                                        "padding": seam_padding_px,
                                        "orientation": "vertical",
                                        "mask_width": seam_width_px,
                                        "mask_blur": seam_blur_px,
                                        "source": "canvas",
                                    }
                                )

                        if half_tiles:
                            passes.append(
                                {
                                    "name": "seam_half_tile",
                                    "denoise": seam_denoise,
                                    "tiles": half_tiles,
                                }
                            )
                            seam_tiles_total += len(half_tiles)
                            log("half-tile seam tiles prepared:", len(half_tiles))

                        corner_tiles: List[dict] = []
                        radial_width = max(seam_width_px, tile_size // 2)

                        for seam_y in range(tile_size, work_height, tile_size):
                            y0 = seam_y - offset
                            y0 = max(0, min(y0, work_height - tile_size))
                            crop_h = min(tile_size, work_height - y0)
                            if crop_h <= 0:
                                continue
                            for seam_x in range(tile_size, work_width, tile_size):
                                x0 = seam_x - offset
                                x0 = max(0, min(x0, work_width - tile_size))
                                crop_w = min(tile_size, work_width - x0)
                                if crop_w <= 0:
                                    continue
                                corner_tiles.append(
                                    {
                                        "x": x0,
                                        "y": y0,
                                        "w": crop_w,
                                        "h": crop_h,
                                        "padding": 0,
                                        "orientation": "radial",
                                        "mask_width": radial_width,
                                        "mask_blur": seam_blur_px,
                                        "source": "canvas",
                                    }
                                )

                        if corner_tiles:
                            passes.append(
                                {
                                    "name": "seam_corners",
                                    "denoise": seam_denoise,
                                    "tiles": corner_tiles,
                                }
                            )
                            seam_tiles_total += len(corner_tiles)
                            log("corner seam tiles prepared:", len(corner_tiles))

                    if seam_tiles_total > 0:
                        log(
                            "seam repair summary:",
                            f"additional_tiles={seam_tiles_total}",
                        )
                    else:
                        log("no seam passes added")

                    for order_index, tile_pass in enumerate(passes):
                        tile_pass["order"] = order_index

                    tile_idx = 0

                    def process_tile(spec: dict, tile_denoise: float, pass_name: str, pass_order: int):
                        nonlocal tile_idx
                        x0 = spec["x"]
                        y0 = spec["y"]
                        crop_w = spec["w"]
                        crop_h = spec["h"]
                        padding = spec["padding"]
                        orientation = spec["orientation"]
                        mask_width = spec["mask_width"]
                        mask_blur = spec["mask_blur"]
                        source_type = spec.get("source", "base")

                        if crop_w <= 0 or crop_h <= 0:
                            log(
                                "skip tile with non-positive dimensions:",
                                f"pass={pass_name}",
                                f"w={crop_w}",
                                f"h={crop_h}",
                            )
                            return

                        pad_left = min(padding, x0)
                        pad_top = min(padding, y0)
                        pad_right = min(padding, max(0, work_width - (x0 + crop_w)))
                        pad_bottom = min(padding, max(0, work_height - (y0 + crop_h)))

                        src_y0 = y0 - pad_top
                        src_y1 = y0 + crop_h + pad_bottom
                        src_x0 = x0 - pad_left
                        src_x1 = x0 + crop_w + pad_right

                        if source_type == "canvas":
                            canvas_height = final_height
                            canvas_width = final_width
                            scaled_src_y0 = max(0, min(canvas_height, int(round(src_y0 * scale_factor))))
                            scaled_src_y1 = max(0, min(canvas_height, int(round(src_y1 * scale_factor))))
                            scaled_src_x0 = max(0, min(canvas_width, int(round(src_x0 * scale_factor))))
                            scaled_src_x1 = max(0, min(canvas_width, int(round(src_x1 * scale_factor))))
                            if scaled_src_y1 <= scaled_src_y0 or scaled_src_x1 <= scaled_src_x0:
                                log(
                                    "skip tile after scaling produced empty slice:",
                                    f"pass={pass_name}",
                                    f"scaled_src=({scaled_src_x0},{scaled_src_y0})-({scaled_src_x1},{scaled_src_y1})",
                                )
                                tile_idx += 1
                                return
                            tile_cf = canvas[:, :, scaled_src_y0:scaled_src_y1, scaled_src_x0:scaled_src_x1]
                        else:
                            tile_cf = base_cf[:, :, src_y0:src_y1, src_x0:src_x1]
                            if seam_fix_enabled and global_mean is not None and global_std is not None:
                                tile_cf = _normalize_tile_to_global(tile_cf, global_mean, global_std)
                                if low_freq_cf is not None and noise_reproj_alpha > 0.0:
                                    guidance_patch = low_freq_cf[:, :, src_y0:src_y1, src_x0:src_x1]
                                    if guidance_patch.shape[-2:] != tile_cf.shape[-2:]:
                                        guidance_patch = F.interpolate(
                                            guidance_patch,
                                            size=tile_cf.shape[-2:],
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                    if seam_guidance_mask is not None:
                                        seam_weights = seam_guidance_mask[:, :, src_y0:src_y1, src_x0:src_x1]
                                        if seam_weights.shape[-2:] != tile_cf.shape[-2:]:
                                            seam_weights = F.interpolate(
                                                seam_weights,
                                                size=tile_cf.shape[-2:],
                                                mode="bilinear",
                                                align_corners=False,
                                            )
                                        tile_cf = (
                                            tile_cf * (1.0 - noise_reproj_alpha * seam_weights)
                                            + guidance_patch * (noise_reproj_alpha * seam_weights)
                                        )
                                    else:
                                        tile_cf = tile_cf * (1.0 - noise_reproj_alpha) + guidance_patch * noise_reproj_alpha
                                    tile_cf = tile_cf.clamp(0.0, 1.0)

                        if tile_cf.shape[-2] <= 0 or tile_cf.shape[-1] <= 0:
                            log(
                                "skip tile after padding produced empty slice:",
                                f"pass={pass_name}",
                                f"src=({src_x0},{src_y0})-({src_x1},{src_y1})",
                            )
                            tile_idx += 1
                            return

                        if debug_logging:
                            log(
                                "tile source slice:",
                                f"pass={pass_name}",
                                f"index={tile_idx}",
                                f"shape={tuple(tile_cf.shape)}",
                                f"padding=({pad_left},{pad_right},{pad_top},{pad_bottom})",
                            )

                        log(
                            "tile start:",
                            f"pass={pass_name}",
                            f"index={tile_idx}",
                            f"origin=({x0},{y0})",
                            f"size={crop_w}x{crop_h}",
                            f"padding=({pad_left},{pad_right},{pad_top},{pad_bottom})",
                            f"orientation={orientation}",
                            f"mask_width={mask_width}",
                            f"mask_blur={mask_blur}",
                            f"denoise={tile_denoise}",
                        )

                        if source_type == "canvas":
                            scaled_total_h = max(1, tile_cf.shape[-2])
                            scaled_total_w = max(1, tile_cf.shape[-1])
                            tile_scaled_cf = tile_cf
                        else:
                            scaled_total_h = max(1, int(round(tile_cf.shape[-2] * scale_factor)))
                            scaled_total_w = max(1, int(round(tile_cf.shape[-1] * scale_factor)))
                            tile_scaled_cf = F.interpolate(
                                tile_cf,
                                size=(scaled_total_h, scaled_total_w),
                                mode="bilinear",
                                align_corners=False,
                            )
                        if debug_logging:
                            ratio_h = scaled_total_h / max(tile_cf.shape[-2], 1)
                            ratio_w = scaled_total_w / max(tile_cf.shape[-1], 1)
                            log(
                                "tile rescaled:",
                                f"pass={pass_name}",
                                f"index={tile_idx}",
                                f"scaled_shape={tuple(tile_scaled_cf.shape)}",
                                f"scale_ratio=({ratio_w:.5f},{ratio_h:.5f})",
                            )
                        tile_scaled = tile_scaled_cf.movedim(1, -1)

                        output_h = int(tile_scaled_cf.shape[-2])
                        output_w = int(tile_scaled_cf.shape[-1])
                        quality_cap = QUALITY_PRESETS[-1]
                        quality_for_tile = min(quality, quality_cap)
                        quality_for_tile = max(tile_multiple, quality_for_tile)
                        quality_for_tile = _snap_to_multiple(quality_for_tile, tile_multiple)
                        if (output_h > quality_for_tile or output_w > quality_for_tile) and debug_logging:
                            log(
                                "quality clamp active:",
                                f"tile_output={output_w}x{output_h}",
                                f"latent_size={quality_for_tile}",
                            )
                        vision_limit = min(vision_tile_size, output_h, output_w)
                        vision_for_tile = _snap_down_to_multiple(max(vision_limit, 28), 28, 28)

                        if pass_name == "base":
                            pass_prompt_local = prompt
                        else:
                            if prompt_clean:
                                pass_prompt_local = f"{prompt_clean} {seam_prompt_effective}"
                            else:
                                pass_prompt_local = seam_prompt_effective

                        effective_prompt_text = pass_prompt_local if pass_prompt_local else prompt
                        if tile_idx == 0:
                            summary = effective_prompt_text.replace("\n", " ")
                            if len(summary) > 160:
                                summary = summary[:157] + "..."
                            log(
                                "effective prompt:",
                                f"pass={pass_name}",
                                f"text='{summary}'",
                            )
                            log(
                                "tile targets:",
                                f"latent={quality_for_tile}",
                                f"vision={vision_for_tile}",
                            )

                        conditioning, start_latent = _build_qwen_conditioning(
                            clip,
                            vae,
                            tile_scaled,
                            prompt,
                            system_prompt,
                            pass_prompt_local,
                            quality_for_tile,
                            vision_for_tile,
                        )

                        latent_dict = {"samples": start_latent}
                        tile_seed = _stable_tile_seed(seed, x0, y0, pass_order)
                        log(
                            "sampling tile:",
                            f"pass={pass_name}",
                            f"index={tile_idx}",
                            f"seed={tile_seed}",
                            f"latent_shape={tuple(start_latent.shape)}",
                        )
                        samples, = common_ksampler(
                            model,
                            tile_seed,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            conditioning,
                            negative,
                            latent_dict,
                            denoise=tile_denoise,
                        )

                        decoded_tile, = vae_decoder.decode(vae, samples)
                        decoded_cf = decoded_tile.movedim(-1, 1)
                        if decoded_cf.shape[-2] != scaled_total_h or decoded_cf.shape[-1] != scaled_total_w:
                            decoded_cf = F.interpolate(
                                decoded_cf,
                                size=(scaled_total_h, scaled_total_w),
                                mode="bilinear",
                                align_corners=False,
                            )
                        log(
                            "decoded tile:",
                            f"pass={pass_name}",
                            f"index={tile_idx}",
                            f"decoded_shape={tuple(decoded_cf.shape)}",
                        )

                        src_total_h = max(1, tile_cf.shape[-2])
                        src_total_w = max(1, tile_cf.shape[-1])
                        scale_ratio_h = max(1e-6, scaled_total_h / float(src_total_h))
                        scale_ratio_w = max(1e-6, scaled_total_w / float(src_total_w))

                        pad_top_scaled = int(round(pad_top * scale_ratio_h))
                        pad_bottom_scaled = int(round(pad_bottom * scale_ratio_h))
                        pad_left_scaled = int(round(pad_left * scale_ratio_w))
                        pad_right_scaled = int(round(pad_right * scale_ratio_w))

                        def _ensure_positive(total: int, before: int, after: int) -> Tuple[int, int]:
                            interior = total - before - after
                            if interior >= 1:
                                return before, after
                            deficit = 1 - interior
                            reduce_after = min(after, deficit)
                            after -= reduce_after
                            deficit -= reduce_after
                            if deficit > 0:
                                reduce_before = min(before, deficit)
                                before -= reduce_before
                                deficit -= reduce_before
                            return max(before, 0), max(after, 0)

                        pad_top_scaled, pad_bottom_scaled = _ensure_positive(scaled_total_h, pad_top_scaled, pad_bottom_scaled)
                        pad_left_scaled, pad_right_scaled = _ensure_positive(scaled_total_w, pad_left_scaled, pad_right_scaled)

                        context_mask = torch.ones(
                            (1, 1, scaled_total_h, scaled_total_w),
                            dtype=decoded_cf.dtype,
                            device=decoded_cf.device,
                        )

                        if pad_left_scaled > 0:
                            ramp = torch.linspace(0.0, 1.0, pad_left_scaled + 1, device=decoded_cf.device, dtype=decoded_cf.dtype)[1:]
                            context_mask[:, :, :, :pad_left_scaled] *= ramp.view(1, 1, 1, -1)
                        if pad_right_scaled > 0:
                            ramp = torch.linspace(1.0, 0.0, pad_right_scaled + 1, device=decoded_cf.device, dtype=decoded_cf.dtype)[1:]
                            context_mask[:, :, :, -pad_right_scaled:] *= ramp.view(1, 1, 1, -1)
                        if pad_top_scaled > 0:
                            ramp = torch.linspace(0.0, 1.0, pad_top_scaled + 1, device=decoded_cf.device, dtype=decoded_cf.dtype)[1:]
                            context_mask[:, :, :pad_top_scaled, :] *= ramp.view(1, 1, -1, 1)
                        if pad_bottom_scaled > 0:
                            ramp = torch.linspace(1.0, 0.0, pad_bottom_scaled + 1, device=decoded_cf.device, dtype=decoded_cf.dtype)[1:]
                            context_mask[:, :, -pad_bottom_scaled:, :] *= ramp.view(1, 1, -1, 1)

                        dst_y = int(round(y0 * scale_factor)) - pad_top_scaled
                        dst_x = int(round(x0 * scale_factor)) - pad_left_scaled
                        src_y0 = 0
                        src_x0 = 0
                        src_y1 = scaled_total_h
                        src_x1 = scaled_total_w

                        if dst_y < 0:
                            src_y0 = min(scaled_total_h, -dst_y)
                            dst_y = 0
                        if dst_x < 0:
                            src_x0 = min(scaled_total_w, -dst_x)
                            dst_x = 0
                        if dst_y + (src_y1 - src_y0) > final_height:
                            src_y1 = src_y0 + max(0, final_height - dst_y)
                        if dst_x + (src_x1 - src_x0) > final_width:
                            src_x1 = src_x0 + max(0, final_width - dst_x)

                        paste_cf = decoded_cf[:, :, src_y0:src_y1, src_x0:src_x1]
                        if paste_cf.shape[-2] <= 0 or paste_cf.shape[-1] <= 0:
                            log(
                                "skip tile because clipped region is empty:",
                                f"pass={pass_name}",
                                f"index={tile_idx}",
                            )
                            tile_idx += 1
                            return

                        context_mask = context_mask[:, :, src_y0:src_y1, src_x0:src_x1]

                        if mask_width > 0 or orientation is not None:
                            scaled_mask_width = max(0, int(round(mask_width * scale_factor)))
                        else:
                            scaled_mask_width = 0
                        scaled_mask_blur = max(0, int(round(mask_blur * scale_factor))) if mask_blur > 0 else 0
                        mask_full = _create_mask(
                            paste_cf.shape[-2],
                            paste_cf.shape[-1],
                            orientation,
                            scaled_mask_width,
                            scaled_mask_blur,
                            paste_cf.device,
                            paste_cf.dtype,
                        )
                        mask = mask_full * context_mask

                        dst_y_end = dst_y + paste_cf.shape[-2]
                        dst_x_end = dst_x + paste_cf.shape[-1]
                        if dst_y_end > final_height or dst_x_end > final_width:
                            overlap_h = max(0, final_height - dst_y)
                            overlap_w = max(0, final_width - dst_x)
                            paste_cf = paste_cf[:, :, :overlap_h, :overlap_w]
                            mask = mask[:, :, :overlap_h, :overlap_w]
                            dst_y_end = min(dst_y_end, final_height)
                            dst_x_end = min(dst_x_end, final_width)
                        paste_h = dst_y_end - dst_y
                        paste_w = dst_x_end - dst_x
                        if paste_h <= 0 or paste_w <= 0:
                            log(
                                "skip tile because destination has zero area:",
                                f"pass={pass_name}",
                                f"index={tile_idx}",
                            )
                            tile_idx += 1
                            return

                        if debug_logging:
                            mask_min = float(mask.min().item()) if mask.numel() > 0 else 0.0
                            mask_max = float(mask.max().item()) if mask.numel() > 0 else 0.0
                            log(
                                "tile paste geometry:",
                                f"pass={pass_name}",
                                f"index={tile_idx}",
                                f"dst=({dst_x},{dst_y})-({dst_x_end},{dst_y_end})",
                                f"paste_size={paste_w}x{paste_h}",
                                f"mask_minmax=({mask_min:.4f},{mask_max:.4f})",
                            )

                        log(
                            "compositing tile:",
                            f"pass={pass_name}",
                            f"index={tile_idx}",
                            f"dst=({dst_x},{dst_y})",
                            f"size={paste_w}x{paste_h}",
                            f"mask_sum={float(mask.sum().item()):.4f}",
                        )

                        target_slice = canvas[:, :, dst_y:dst_y_end, dst_x:dst_x_end]
                        weight_slice = weights[:, :, dst_y:dst_y_end, dst_x:dst_x_end]

                        if seam_fix_enabled and pass_name != "base":
                            diff_metric = torch.mean(torch.abs(target_slice - paste_cf), dim=1, keepdim=True)
                            attenuation = torch.clamp(torch.exp(-diff_metric * 12.0), min=0.25, max=1.0)
                            mask = mask * attenuation

                        if pass_name == "base":
                            target_slice += paste_cf * mask
                            weight_slice += mask
                        else:
                            weight_slice.fill_(1.0)
                            target_slice.mul_(1.0 - mask)
                            target_slice += paste_cf * mask

                        tile_idx += 1

                    for tile_pass in passes:
                        log(
                            "starting pass:",
                            f"name={tile_pass['name']}",
                            f"tiles={len(tile_pass['tiles'])}",
                            f"denoise={tile_pass['denoise']}",
                        )
                        for spec in tile_pass["tiles"]:
                            process_tile(spec, tile_pass["denoise"], tile_pass["name"], tile_pass.get("order", 0))
                        log("pass complete:", tile_pass["name"])

                        if tile_pass["name"] == "base":
                            weights_safe = weights.clamp(min=1e-6)
                            canvas = canvas / weights_safe
                            weights.fill_(1.0)

                    weights_safe = weights.clamp(min=1e-6)
                    canvas = canvas / weights_safe
                    if seam_fix_enabled:
                        canvas = _apply_feedback_consistency(canvas, base_cf, scale_factor, tile_size)
                    if canvas.shape[-2] != target_height or canvas.shape[-1] != target_width:
                        canvas = canvas[:, :, :target_height, :target_width]
                    output = canvas.movedim(1, -1).clamp(0.0, 1.0)
                    pass_outputs.append(output)
                    log("batch complete:", f"output_shape={tuple(output.shape)}")

                current_image = torch.cat(pass_outputs, dim=0)
                log(
                    f"=== pass {pass_index}/{len(pass_scales)} complete ===",
                    f"output_shape={tuple(current_image.shape)}",
                )

        final_image = current_image
        final_slices: List[torch.Tensor] = []
        for batch_index, (orig_h, orig_w) in enumerate(original_sizes):
            target_h = max(1, int(round(orig_h * total_scale)))
            target_w = max(1, int(round(orig_w * total_scale)))
            available_h = final_image.shape[1]
            available_w = final_image.shape[2]
            slice_h = min(target_h, available_h)
            slice_w = min(target_w, available_w)
            cropped = final_image[batch_index : batch_index + 1, :slice_h, :slice_w, :]
            final_slices.append(cropped)
            log(
                "final crop:",
                f"batch={batch_index}",
                f"target={slice_w}x{slice_h}",
            )
        if final_slices:
            final_image = torch.cat(final_slices, dim=0)
        log("node finished:", f"final_shape={tuple(final_image.shape)}")
        return (final_image,)


NODE_CLASS_MAPPINGS = {
    "VNCCSQWENUpscaler": VNCCSQWENUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSQWENUpscaler": "VNCCS QWEN Upscaler",
}
