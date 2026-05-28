"""Experimental VNCCS chroma key node.

This module intentionally lives outside vnccs_utils.py so the production
VNCCSChromaKey node can stay stable while we test more aggressive keying ideas.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .vnccs_utils import _ensure_float01
except ImportError:
    from vnccs_utils import _ensure_float01


def _box_blur_2d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    src = mask.unsqueeze(0).unsqueeze(0)
    blurred = F.avg_pool2d(src, kernel_size=kernel_size, stride=1, padding=radius)
    return blurred.squeeze(0).squeeze(0)


def _morph(mask: torch.Tensor, radius: int, mode: str) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    src = mask.unsqueeze(0).unsqueeze(0)
    if mode == "dilate":
        out = F.max_pool2d(src, kernel_size=kernel_size, stride=1, padding=radius)
    elif mode == "erode":
        out = -F.max_pool2d(-src, kernel_size=kernel_size, stride=1, padding=radius)
    else:
        raise ValueError(f"Unsupported morph mode: {mode}")
    return out.squeeze(0).squeeze(0)


def _remove_small_islands(mask: torch.Tensor, min_neighbors: int) -> torch.Tensor:
    if min_neighbors <= 0:
        return mask
    hard = (mask > 0.5).float()
    src = hard.unsqueeze(0).unsqueeze(0)
    neighbors = F.conv2d(src, torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype), padding=1)
    keep = (neighbors.squeeze(0).squeeze(0) >= float(min_neighbors)).float()
    return torch.where(keep > 0.0, mask, torch.zeros_like(mask))


class VNCCSChromaKeyExperimental:
    """Experimental chroma key with soft alpha and foreground reconstruction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 0.14, "min": 0.0, "max": 1.0, "step": 0.01}),
                "softness": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "matte_cleanup": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_recover": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "matte_method": (["chroma_soft", "guided_edge", "pymatting_if_available"], {"default": "guided_edge"}),
                "screen_mode": (["auto", "green", "blue", "red"], {"default": "auto"}),
                "output_mode": (["straight_rgba", "premultiplied_rgba"], {"default": "straight_rgba"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "matte", "edge_debug")
    CATEGORY = "VNCCS/Experimental"
    FUNCTION = "chroma_key"
    DESCRIPTION = """
    Experimental VNCCS chroma key. Uses chroma-difference soft alpha, optional
    edge-guided refinement, edge-only despill, and foreground reconstruction.
    """

    def chroma_key(
        self,
        image,
        tolerance,
        softness,
        despill_strength,
        edge_width,
        matte_cleanup,
        foreground_recover,
        matte_method,
        screen_mode,
        output_mode,
    ):
        if len(image.shape) == 4:
            rgba_list = []
            matte_list = []
            debug_list = []
            for frame in image:
                rgba, alpha, debug = self._process_single(
                    frame,
                    tolerance,
                    softness,
                    despill_strength,
                    edge_width,
                    matte_cleanup,
                    foreground_recover,
                    matte_method,
                    screen_mode,
                    output_mode,
                )
                rgba_list.append(rgba)
                matte_list.append(alpha)
                debug_list.append(debug)
            return (torch.stack(rgba_list), torch.stack(matte_list), torch.stack(debug_list))

        rgba, alpha, debug = self._process_single(
            image,
            tolerance,
            softness,
            despill_strength,
            edge_width,
            matte_cleanup,
            foreground_recover,
            matte_method,
            screen_mode,
            output_mode,
        )
        return (rgba.unsqueeze(0), alpha.unsqueeze(0), debug.unsqueeze(0))

    def _process_single(
        self,
        image,
        tolerance,
        softness,
        despill_strength,
        edge_width,
        matte_cleanup,
        foreground_recover,
        matte_method,
        screen_mode,
        output_mode,
    ):
        image = _ensure_float01(image)[..., :3]
        height, width, _ = image.shape
        key_color = self._detect_key_color(image)
        dominant_idx = self._dominant_channel(key_color, screen_mode)
        other_indices = [idx for idx in range(3) if idx != dominant_idx]

        alpha = self._build_soft_alpha(
            image=image,
            key_color=key_color,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            tolerance=float(tolerance),
            softness=float(softness),
        )
        alpha = self._cleanup_alpha(alpha, int(edge_width), float(matte_cleanup))

        if matte_method == "guided_edge":
            alpha = self._guided_edge_refine(image, alpha, int(edge_width), float(matte_cleanup))
        elif matte_method == "pymatting_if_available":
            alpha = self._pymatting_refine_if_available(image, alpha, int(edge_width))

        alpha = alpha.clamp(0.0, 1.0)
        edge = self._edge_band(alpha, int(edge_width))

        recovered = self._recover_foreground(
            image=image,
            alpha=alpha,
            edge=edge,
            key_color=key_color,
            amount=float(foreground_recover),
        )
        despilled = self._edge_despill(
            image=recovered,
            alpha=alpha,
            edge=edge,
            dominant_idx=dominant_idx,
            other_indices=other_indices,
            strength=float(despill_strength),
        )

        if output_mode == "premultiplied_rgba":
            rgb_out = despilled * alpha.unsqueeze(-1)
        else:
            rgb_out = despilled

        rgba = torch.cat([rgb_out.clamp(0.0, 1.0), alpha.unsqueeze(-1)], dim=-1)
        debug = torch.stack([edge, alpha, 1.0 - alpha], dim=-1).clamp(0.0, 1.0)

        if rgba.shape[:2] != (height, width):
            raise RuntimeError("Experimental chroma key changed image dimensions unexpectedly.")

        return rgba, alpha, debug

    def _detect_key_color(self, image: torch.Tensor) -> torch.Tensor:
        height, width, _ = image.shape
        ch = max(1, height // 20)
        cw = max(1, width // 20)
        patches = [
            image[0:ch, 0:cw, :3],
            image[0:ch, width - cw : width, :3],
            image[height - ch : height, 0:cw, :3],
            image[height - ch : height, width - cw : width, :3],
        ]

        stable_colors = []
        for patch in patches:
            pixels = patch.reshape(-1, 3)
            if pixels.std(dim=0).mean() < 0.02:
                stable_colors.append(pixels.median(dim=0)[0])

        if stable_colors:
            return torch.stack(stable_colors).median(dim=0)[0]

        y_margin = max(1, height // 10)
        x_margin = max(1, width // 10)
        border_pixels = torch.cat(
            [
                image[0:y_margin, :, :3].reshape(-1, 3),
                image[height - y_margin : height, :, :3].reshape(-1, 3),
                image[:, 0:x_margin, :3].reshape(-1, 3),
                image[:, width - x_margin : width, :3].reshape(-1, 3),
            ],
            dim=0,
        )
        return border_pixels.median(dim=0)[0]

    def _dominant_channel(self, key_color: torch.Tensor, screen_mode: str) -> int:
        if screen_mode == "red":
            return 0
        if screen_mode == "green":
            return 1
        if screen_mode == "blue":
            return 2
        return int(torch.argmax(key_color).item())

    def _build_soft_alpha(
        self,
        image: torch.Tensor,
        key_color: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        tolerance: float,
        softness: float,
    ) -> torch.Tensor:
        eps = 1e-6
        chroma = image / (image.sum(dim=-1, keepdim=True) + eps)
        key_chroma = key_color / (key_color.sum() + eps)
        chroma_dist = torch.sqrt(((chroma - key_chroma) ** 2).sum(dim=-1))

        rgb_dist = torch.sqrt(((image - key_color) ** 2).sum(dim=-1))
        dom = image[..., dominant_idx]
        other_max = torch.maximum(image[..., other_indices[0]], image[..., other_indices[1]])
        other_avg = (image[..., other_indices[0]] + image[..., other_indices[1]]) * 0.5
        screen_excess = dom - (other_max * 0.65 + other_avg * 0.35)

        key_dom = key_color[dominant_idx]
        key_other_max = torch.maximum(key_color[other_indices[0]], key_color[other_indices[1]])
        key_other_avg = (key_color[other_indices[0]] + key_color[other_indices[1]]) * 0.5
        key_excess = torch.clamp(key_dom - (key_other_max * 0.65 + key_other_avg * 0.35), min=0.05)

        hue_similarity = 1.0 - self._smoothstep(tolerance, tolerance + softness, chroma_dist)
        rgb_similarity = 1.0 - self._smoothstep(tolerance * 1.5, tolerance * 1.5 + softness * 2.0, rgb_dist)
        screen_affinity = self._smoothstep(key_excess * 0.2, key_excess * 0.85 + 1e-6, screen_excess)

        # Background must look like the screen and also share the screen channel bias.
        # Chroma similarity alone is unsafe for anime whites, grays, and skin.
        background = (hue_similarity * 0.55 + rgb_similarity * 0.45) * screen_affinity

        strong_screen = self._smoothstep(key_excess * 0.75, key_excess * 1.25 + 1e-6, screen_excess)
        background = torch.maximum(background, strong_screen * hue_similarity * 0.85).clamp(0.0, 1.0)
        return 1.0 - background

    def _smoothstep(self, edge0: float | torch.Tensor, edge1: float | torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = ((value - edge0) / (edge1 - edge0 + 1e-6)).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _cleanup_alpha(self, alpha: torch.Tensor, edge_width: int, amount: float) -> torch.Tensor:
        if amount <= 0.0:
            return alpha
        cleaned = _remove_small_islands(alpha, min_neighbors=max(1, int(2 + amount * 5)))
        if edge_width > 0:
            opened = _morph(_morph(cleaned, 1, "erode"), 1, "dilate")
            edge = self._edge_band(cleaned, max(1, edge_width))
            cleaned = torch.lerp(cleaned, opened, edge * amount * 0.35)
            blurred = _box_blur_2d(cleaned, max(1, edge_width // 2))
            cleaned = torch.lerp(cleaned, blurred, edge * amount * 0.25)
        return cleaned.clamp(0.0, 1.0)

    def _guided_edge_refine(self, image: torch.Tensor, alpha: torch.Tensor, edge_width: int, amount: float) -> torch.Tensor:
        if edge_width <= 0 or amount <= 0.0:
            return alpha
        luminance = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        radius = max(1, edge_width)
        mean_i = _box_blur_2d(luminance, radius)
        mean_a = _box_blur_2d(alpha, radius)
        corr_i = _box_blur_2d(luminance * luminance, radius)
        corr_ia = _box_blur_2d(luminance * alpha, radius)
        var_i = corr_i - mean_i * mean_i
        cov_ia = corr_ia - mean_i * mean_a
        linear_a = cov_ia / (var_i + 0.01)
        linear_b = mean_a - linear_a * mean_i
        refined = _box_blur_2d(linear_a, radius) * luminance + _box_blur_2d(linear_b, radius)
        edge = self._edge_band(alpha, edge_width)
        return torch.lerp(alpha, refined.clamp(0.0, 1.0), edge * amount).clamp(0.0, 1.0)

    def _pymatting_refine_if_available(self, image: torch.Tensor, alpha: torch.Tensor, edge_width: int) -> torch.Tensor:
        try:
            import numpy as np
            from pymatting import estimate_alpha_cf
        except Exception:
            return self._guided_edge_refine(image, alpha, edge_width, 0.5)

        trimap = torch.full_like(alpha, 0.5)
        trimap = torch.where(alpha > 0.98, torch.ones_like(trimap), trimap)
        trimap = torch.where(alpha < 0.02, torch.zeros_like(trimap), trimap)

        image_np = image.detach().cpu().numpy().astype("float64")
        trimap_np = trimap.detach().cpu().numpy().astype("float64")
        try:
            matte_np = estimate_alpha_cf(image_np, trimap_np)
        except Exception:
            return self._guided_edge_refine(image, alpha, edge_width, 0.5)

        matte = torch.from_numpy(np.asarray(matte_np)).to(device=image.device, dtype=image.dtype)
        return matte.clamp(0.0, 1.0)

    def _edge_band(self, alpha: torch.Tensor, edge_width: int) -> torch.Tensor:
        if edge_width <= 0:
            return ((alpha > 0.01) & (alpha < 0.99)).float()
        hard = (alpha > 0.5).float()
        dilated = _morph(hard, edge_width, "dilate")
        eroded = _morph(hard, edge_width, "erode")
        return (dilated - eroded).clamp(0.0, 1.0)

    def _recover_foreground(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        key_color: torch.Tensor,
        amount: float,
    ) -> torch.Tensor:
        if amount <= 0.0:
            return image
        safe_alpha = alpha.unsqueeze(-1).clamp(0.08, 1.0)
        reconstructed = (image - (1.0 - safe_alpha) * key_color) / safe_alpha
        reconstructed = reconstructed.clamp(0.0, 1.0)
        edge_weight = (1.0 - (alpha - 0.5).abs() * 2.0).clamp(0.0, 1.0).unsqueeze(-1)
        edge_weight = edge_weight * edge.unsqueeze(-1)
        return torch.lerp(image, reconstructed, edge_weight * amount).clamp(0.0, 1.0)

    def _edge_despill(
        self,
        image: torch.Tensor,
        alpha: torch.Tensor,
        edge: torch.Tensor,
        dominant_idx: int,
        other_indices: list[int],
        strength: float,
    ) -> torch.Tensor:
        if strength <= 0.0:
            return image

        result = image.clone()
        dom = image[..., dominant_idx]
        other1 = image[..., other_indices[0]]
        other2 = image[..., other_indices[1]]
        limit = torch.maximum(other1, other2) * 0.75 + ((other1 + other2) * 0.5) * 0.25
        corrected_dom = torch.minimum(dom, limit)

        spill = (dom - limit).clamp(0.0, 1.0)
        neutral = image.clone()
        neutral[..., dominant_idx] = corrected_dom
        neutral[..., other_indices[0]] = (neutral[..., other_indices[0]] + spill * 0.25).clamp(0.0, 1.0)
        neutral[..., other_indices[1]] = (neutral[..., other_indices[1]] + spill * 0.25).clamp(0.0, 1.0)

        edge_weight = torch.maximum(edge, ((alpha > 0.0) & (alpha < 0.98)).float() * 0.5).unsqueeze(-1)
        return torch.lerp(result, neutral, edge_weight * strength).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {
    "VNCCSChromaKeyExperimental": VNCCSChromaKeyExperimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSChromaKeyExperimental": "VNCCS Chroma Key Experimental",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCSChromaKeyExperimental": "VNCCS/Experimental",
}
