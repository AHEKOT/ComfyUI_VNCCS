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

UPSCALE_METHODS = ("lanczos", "bicubic", "area")
CROP_METHODS = ("center", "disabled")
SEAM_FIX_MODES = (
    "None",
    "Band Pass",
    "Half Tile",
    "Half Tile + Intersections",
)


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
    upscale_method: str,
    crop_mode: str,
    snap_multiple: int = 8,
) -> torch.Tensor:
    """Resize a tile tensor to the desired latent resolution."""
    samples = tensor_nhwc.movedim(-1, 1)
    current_total = samples.shape[2] * samples.shape[3]
    if current_total == 0:
        raise ValueError("Tile has zero area; cannot upscale")
    scale_by = math.sqrt(target_area / current_total)
    width = max(snap_multiple, round(samples.shape[3] * scale_by / snap_multiple) * snap_multiple)
    height = max(snap_multiple, round(samples.shape[2] * scale_by / snap_multiple) * snap_multiple)
    upscaled = comfy.utils.common_upscale(samples, width, height, upscale_method, crop_mode)
    return upscaled.movedim(1, -1)


def _build_qwen_conditioning(
    clip,
    vae,
    tile: torch.Tensor,
    prompt: str,
    system_prompt: str,
    quality: int,
    vision_size: int,
    upscale_method: str,
    crop_mode: str,
) -> Tuple[List, torch.Tensor]:
    """Create QWEN conditioning and obtain the starting latent for a tile."""
    llama_template = _sanitize_instruction(system_prompt)

    # Prepare the images for CLIP + reference latent attachment
    target_area = quality * quality
    vl_area = vision_size * vision_size

    latent_ready = _resize_for_latent(tile, target_area, upscale_method, crop_mode, snap_multiple=8)
    vl_ready = _resize_for_latent(tile, vl_area, upscale_method, crop_mode, snap_multiple=28)

    ref_latent = vae.encode(latent_ready[:, :, :, :3])
    ref_latents = [ref_latent]

    image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    tokens = clip.tokenize(image_prompt + prompt, images=[vl_ready], llama_template=llama_template)
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
    if mode in {"horizontal", "vertical", "radial"} and seam_width > 0:
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
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "prompt": ("STRING", {"multiline": True, "default": "upscale image to 4k ultrasharp quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (sampler_enum, {"default": default_sampler}),
                "scheduler": (scheduler_enum, {"default": default_scheduler}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "quality": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "vision_tile_size": ("INT", {"default": 384, "min": 256, "max": 512, "step": 8}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos"}),
                "crop_mode": (CROP_METHODS, {"default": "center"}),
                "seam_fix_mode": (SEAM_FIX_MODES, {"default": "None"}),
                "seam_fix_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    def upscale(
        self,
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
        tile_size: int,
        quality: int,
        vision_tile_size: int,
        scale: float,
        denoise: float,
        upscale_method: str,
        crop_mode: str,
        seam_fix_mode: str,
        seam_fix_denoise: float,
        seam_fix_width: int,
        seam_fix_mask_blur: int,
        seam_fix_padding: int,
        debug_logging: bool,
    ) -> Tuple[torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("Expected image tensor with shape [B, H, W, C]")
        if image.shape[-1] < 3:
            raise ValueError("QWEN upscaler expects RGB images")

        def log(*parts):
            if debug_logging:
                print("[VNCCS QWEN Upscaler]", *parts)

        batch_size = image.shape[0]
        image_min = float(image.min().item())
        image_max = float(image.max().item())
        log(
            "received input tensor:",
            f"batch={batch_size}",
            f"shape={tuple(image.shape)}",
            f"dtype={image.dtype}",
            f"range=[{image_min:.4f}, {image_max:.4f}]",
        )
        log(
            "runtime parameters:",
            f"tile_size={tile_size}",
            f"scale={scale}",
            f"quality={quality}",
            f"vision_tile_size={vision_tile_size}",
            f"steps={steps}",
            f"cfg={cfg}",
            f"sampler={sampler_name}/{scheduler}",
            f"denoise={denoise}",
        )
        if seam_fix_mode != "None":
            log(
                "seam fix enabled:",
                f"mode={seam_fix_mode}",
                f"denoise={seam_fix_denoise}",
                f"width={seam_fix_width}",
                f"padding={seam_fix_padding}",
                f"mask_blur={seam_fix_mask_blur}",
            )
        else:
            log("seam fix disabled")

        outputs: List[torch.Tensor] = []
        vae_decoder = VAEDecode()
        negative = _build_negative_conditioning(clip, negative_prompt)
        seam_fix_enabled = seam_fix_mode != "None"

        with torch.no_grad():
            for batch_index in range(batch_size):
                base = image[batch_index : batch_index + 1, :, :, :3]
                base_cf = base.movedim(-1, 1)
                base_height = base_cf.shape[-2]
                base_width = base_cf.shape[-1]

                tiles_y = math.ceil(base_height / tile_size)
                tiles_x = math.ceil(base_width / tile_size)
                total_tiles = tiles_x * tiles_y
                final_height = max(1, int(round(base_height * scale)))
                final_width = max(1, int(round(base_width * scale)))
                log("processing batch index", batch_index)
                log("base input size:", f"{base_width}x{base_height}")
                log(
                    "target resolution:",
                    f"{final_width}x{final_height} (scale={scale:.2f})",
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
                for y in range(0, base_height, tile_size):
                    for x in range(0, base_width, tile_size):
                        crop_h = min(tile_size, base_height - y)
                        crop_w = min(tile_size, base_width - x)
                        if crop_h <= 0 or crop_w <= 0:
                            continue
                        base_tiles.append(
                            {
                                "x": x,
                                "y": y,
                                "w": crop_w,
                                "h": crop_h,
                                "padding": 0,
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
                    seam_width_px = max(0, seam_fix_width)
                    seam_padding_px = max(0, seam_fix_padding)
                    seam_blur_px = max(0, seam_fix_mask_blur)

                    if seam_fix_mode == "Band Pass" and seam_width_px > 0:
                        band_tiles: List[dict] = []
                        stripe_w = min(seam_width_px, base_width)
                        stripe_h = min(seam_width_px, base_height)

                        for seam_x in range(tile_size, base_width, tile_size):
                            x0 = seam_x - stripe_w // 2
                            x0 = max(0, min(x0, base_width - stripe_w))
                            crop_w = min(stripe_w, base_width - x0)
                            if crop_w <= 0:
                                continue
                            band_tiles.append(
                                {
                                    "x": x0,
                                    "y": 0,
                                    "w": crop_w,
                                    "h": base_height,
                                    "padding": seam_padding_px,
                                    "orientation": "vertical",
                                    "mask_width": seam_width_px,
                                        "mask_blur": 0,
                                        "source": "canvas",
                                }
                            )

                        for seam_y in range(tile_size, base_height, tile_size):
                            y0 = seam_y - stripe_h // 2
                            y0 = max(0, min(y0, base_height - stripe_h))
                            crop_h = min(stripe_h, base_height - y0)
                            if crop_h <= 0:
                                continue
                            band_tiles.append(
                                {
                                    "x": 0,
                                    "y": y0,
                                    "w": base_width,
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
                                    "name": "seam_band_pass",
                                    "denoise": seam_fix_denoise,
                                    "tiles": band_tiles,
                                }
                            )
                            seam_tiles_total += len(band_tiles)
                            log("band pass tiles prepared:", len(band_tiles))

                    if seam_fix_mode in {"Half Tile", "Half Tile + Intersections"}:
                        half_tiles: List[dict] = []
                        offset = tile_size // 2

                        for seam_y in range(tile_size, base_height, tile_size):
                            y0 = seam_y - offset
                            y0 = max(0, min(y0, base_height - tile_size))
                            crop_h = min(tile_size, base_height - y0)
                            if crop_h <= 0:
                                continue
                            for x0 in range(0, base_width, tile_size):
                                crop_w = min(tile_size, base_width - x0)
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

                        for seam_x in range(tile_size, base_width, tile_size):
                            x0 = seam_x - offset
                            x0 = max(0, min(x0, base_width - tile_size))
                            crop_w = min(tile_size, base_width - x0)
                            if crop_w <= 0:
                                continue
                            for y0 in range(0, base_height, tile_size):
                                crop_h = min(tile_size, base_height - y0)
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
                                    "denoise": seam_fix_denoise,
                                    "tiles": half_tiles,
                                }
                            )
                            seam_tiles_total += len(half_tiles)
                            log("half-tile seam tiles prepared:", len(half_tiles))

                        if seam_fix_mode == "Half Tile + Intersections":
                            corner_tiles: List[dict] = []
                            radial_width = seam_width_px if seam_width_px > 0 else max(1, tile_size // 2)

                            for seam_y in range(tile_size, base_height, tile_size):
                                y0 = seam_y - offset
                                y0 = max(0, min(y0, base_height - tile_size))
                                crop_h = min(tile_size, base_height - y0)
                                if crop_h <= 0:
                                    continue
                                for seam_x in range(tile_size, base_width, tile_size):
                                    x0 = seam_x - offset
                                    x0 = max(0, min(x0, base_width - tile_size))
                                    crop_w = min(tile_size, base_width - x0)
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
                                        "denoise": seam_fix_denoise,
                                        "tiles": corner_tiles,
                                    }
                                )
                                seam_tiles_total += len(corner_tiles)
                                log("corner seam tiles prepared:", len(corner_tiles))

                if seam_tiles_total > 0:
                    log(
                        "seam fix summary:",
                        f"mode={seam_fix_mode}",
                        f"additional_tiles={seam_tiles_total}",
                        f"denoise={seam_fix_denoise:.2f}",
                    )
                else:
                    log("no seam passes added")

                tile_idx = 0

                def process_tile(spec: dict, tile_denoise: float, pass_name: str):
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
                    pad_right = min(padding, max(0, base_width - (x0 + crop_w)))
                    pad_bottom = min(padding, max(0, base_height - (y0 + crop_h)))

                    src_y0 = y0 - pad_top
                    src_y1 = y0 + crop_h + pad_bottom
                    src_x0 = x0 - pad_left
                    src_x1 = x0 + crop_w + pad_right

                    if source_type == "canvas":
                        canvas_height = final_height
                        canvas_width = final_width
                        scaled_src_y0 = max(0, min(canvas_height, int(round(src_y0 * scale))))
                        scaled_src_y1 = max(0, min(canvas_height, int(round(src_y1 * scale))))
                        scaled_src_x0 = max(0, min(canvas_width, int(round(src_x0 * scale))))
                        scaled_src_x1 = max(0, min(canvas_width, int(round(src_x1 * scale))))
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

                    if tile_cf.shape[-2] <= 0 or tile_cf.shape[-1] <= 0:
                        log(
                            "skip tile after padding produced empty slice:",
                            f"pass={pass_name}",
                            f"src=({src_x0},{src_y0})-({src_x1},{src_y1})",
                        )
                        tile_idx += 1
                        return

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
                        scaled_total_h = max(1, int(round(tile_cf.shape[-2] * scale)))
                        scaled_total_w = max(1, int(round(tile_cf.shape[-1] * scale)))
                        tile_scaled_cf = F.interpolate(
                            tile_cf,
                            size=(scaled_total_h, scaled_total_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    tile_scaled = tile_scaled_cf.movedim(1, -1)

                    conditioning, start_latent = _build_qwen_conditioning(
                        clip,
                        vae,
                        tile_scaled,
                        prompt,
                        system_prompt,
                        quality,
                        vision_tile_size,
                        upscale_method,
                        crop_mode,
                    )

                    latent_dict = {"samples": start_latent}
                    tile_seed = (seed + tile_idx) & 0xFFFFFFFFFFFFFFFF
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

                    pad_top_scaled = int(round(pad_top * scale))
                    pad_left_scaled = int(round(pad_left * scale))
                    inner_h = max(1, int(round(crop_h * scale)))
                    inner_w = max(1, int(round(crop_w * scale)))

                    max_h = decoded_cf.shape[-2]
                    max_w = decoded_cf.shape[-1]
                    if pad_top_scaled + inner_h > max_h:
                        inner_h = max(0, max_h - pad_top_scaled)
                    if pad_left_scaled + inner_w > max_w:
                        inner_w = max(0, max_w - pad_left_scaled)
                    if inner_h <= 0 or inner_w <= 0:
                        log(
                            "skip tile due to invalid inner size after padding:",
                            f"pass={pass_name}",
                            f"index={tile_idx}",
                        )
                        tile_idx += 1
                        return

                    inner_cf = decoded_cf[:, :, pad_top_scaled : pad_top_scaled + inner_h, pad_left_scaled : pad_left_scaled + inner_w]

                    dst_y = int(round(y0 * scale))
                    dst_x = int(round(x0 * scale))
                    dst_y_end = min(dst_y + inner_h, final_height)
                    dst_x_end = min(dst_x + inner_w, final_width)
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

                    inner_cf = inner_cf[:, :, :paste_h, :paste_w]

                    if mask_width > 0 or orientation is not None:
                        scaled_mask_width = max(0, int(round(mask_width * scale)))
                    else:
                        scaled_mask_width = 0
                    scaled_mask_blur = max(0, int(round(mask_blur * scale))) if mask_blur > 0 else 0
                    mask_full = _create_mask(
                        inner_cf.shape[-2],
                        inner_cf.shape[-1],
                        orientation,
                        scaled_mask_width,
                        scaled_mask_blur,
                        inner_cf.device,
                        inner_cf.dtype,
                    )
                    mask = mask_full[:, :, :paste_h, :paste_w]

                    log(
                        "compositing tile:",
                        f"pass={pass_name}",
                        f"index={tile_idx}",
                        f"dst=({dst_x},{dst_y})",
                        f"size={paste_w}x{paste_h}",
                        f"mask_sum={float(mask.sum().item()):.4f}",
                    )

                    target_slice = canvas[:, :, dst_y:dst_y_end, dst_x:dst_x_end]
                    target_slice += inner_cf * mask
                    weights[:, :, dst_y:dst_y_end, dst_x:dst_x_end] += mask

                    tile_idx += 1

                for tile_pass in passes:
                    log(
                        "starting pass:",
                        f"name={tile_pass['name']}",
                        f"tiles={len(tile_pass['tiles'])}",
                        f"denoise={tile_pass['denoise']}",
                    )
                    for spec in tile_pass["tiles"]:
                        process_tile(spec, tile_pass["denoise"], tile_pass["name"])
                    log("pass complete:", tile_pass["name"])

                weights_safe = weights.clamp(min=1e-6)
                canvas = canvas / weights_safe
                output = canvas.movedim(1, -1).clamp(0.0, 1.0)
                outputs.append(output)
                log("batch complete:", f"output_shape={tuple(output.shape)}")

        final_image = torch.cat(outputs, dim=0)
        log("node finished:", f"final_shape={tuple(final_image.shape)}")
        return (final_image,)


NODE_CLASS_MAPPINGS = {
    "VNCCSQWENUpscaler": VNCCSQWENUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSQWENUpscaler": "VNCCS QWEN Upscaler",
}
