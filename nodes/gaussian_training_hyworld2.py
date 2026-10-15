"""In-process HYWorld2-style 3DGS training for VNCCS orbit scenes.

Adapted from ComfyUI_HYWorld2's world_gs_trainer.py (MIT, MiuProject).
The orbit dataset supplies RGB, alpha, COLMAP poses and sparse points; it does
not supply HYWorld2 depth, normal, sky or anchor data. Only ordinary gsplat's
public Python APIs are required; the HYWorld2 package is never imported.
"""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gaussian_reconstruction import (
    GAUSSIAN_SCENE_TYPE,
    SH_C0,
    _comfy_splat,
    _output_path,
    _runtime_helpers,
    _scene_data,
    _ssim_loss,
)
from ..utils import atomic_output_path


def _hyworld2_scene_scale(frames):
    views = np.stack([frame["view"] for frame in frames]).astype(np.float64)
    centers = np.linalg.inv(views)[:, :3, 3]
    radius = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
    if not np.isfinite(radius) or radius < 1e-6:
        raise ValueError("HYWorld2 training requires non-degenerate camera positions")
    return float(radius) * 1.1


def _hyworld2_initial_scales(points):
    import cv2

    points = np.ascontiguousarray(points, dtype=np.float32)
    index = cv2.flann_Index(points, {"algorithm": 1, "trees": 4})
    neighbors, _ = index.knnSearch(points, 4, params={"checks": 64})
    offsets = points[neighbors[:, 1:]] - points[:, None]
    mean_squared_distance = np.square(offsets).sum(axis=-1).mean(axis=-1)
    return (0.5 * np.log(np.maximum(mean_squared_distance, 1e-12))).astype(np.float32)


def _hyworld2_strategy(
    training_steps, refine_start_iter, refine_stop_iter, refine_every,
    reset_every, grow_grad2d, prune_opacity,
):
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    class BudgetedDefaultStrategy(DefaultStrategy):
        """HYWorld2 ADC: clone small splats, split large ones, respect the cap."""

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            grads = state["grad2d"] / state["count"].clamp_min(1)
            high_gradient = grads > self.grow_grad2d
            small = (
                torch.exp(params["scales"]).amax(dim=-1)
                <= self.grow_scale3d * state["scene_scale"]
            )
            clone_mask = high_gradient & small
            split_mask = high_gradient & ~small
            if step < self.refine_scale2d_stop_iter:
                split_mask |= state["radii"] > self.grow_scale2d

            available = max(0, state["max_gaussians"] - len(params["means"]))
            clone_ids = torch.where(clone_mask)[0]
            split_ids = torch.where(split_mask)[0]
            candidates = torch.cat((clone_ids, split_ids))
            if len(candidates) > available:
                selected = torch.zeros(len(candidates), dtype=torch.bool, device=grads.device)
                if available:
                    selected[torch.topk(grads[candidates], available, sorted=False).indices] = True
                clone_mask = torch.zeros_like(clone_mask)
                split_mask = torch.zeros_like(split_mask)
                clone_mask[clone_ids[selected[:len(clone_ids)]]] = True
                split_mask[split_ids[selected[len(clone_ids):]]] = True

            cloned = int(clone_mask.sum().item())
            divided = int(split_mask.sum().item())
            if cloned:
                duplicate(params=params, optimizers=optimizers, state=state, mask=clone_mask)
            if divided:
                split_mask = torch.cat((
                    split_mask,
                    torch.zeros(cloned, dtype=torch.bool, device=grads.device),
                ))
                split(
                    params=params, optimizers=optimizers, state=state,
                    mask=split_mask, revised_opacity=self.revised_opacity,
                )
            return cloned, divided

    stop = min(refine_stop_iter, training_steps)
    return BudgetedDefaultStrategy(
        refine_start_iter=refine_start_iter,
        refine_stop_iter=stop,
        refine_every=refine_every,
        refine_scale2d_stop_iter=stop,
        reset_every=reset_every,
        grow_grad2d=grow_grad2d,
        prune_opa=prune_opacity,
        prune_scale3d=0.1,
        verbose=True,
    )


def _hyworld2_loss(rendered, target, alpha, target_alpha, kernel, alpha_weight,
                   log_scales, scale_regularization):
    l1 = F.l1_loss(rendered, target)
    ssim = _ssim_loss(rendered, target, kernel) if kernel is not None else l1.new_zeros(())
    alpha_loss = F.l1_loss(alpha, target_alpha)
    scale_loss = l1.new_zeros(())
    if scale_regularization:
        # PhysGaussian ratio regularization, evaluated every ten steps in HYWorld2.
        log_ratio = log_scales.amax(dim=-1) - log_scales.amin(dim=-1)
        scale_loss = 0.1 * (log_ratio.exp() - 5.0).clamp_min(0.0).mean()
    return l1, ssim, alpha_loss * alpha_weight, scale_loss


@torch.inference_mode(False)
@torch.enable_grad()
def _train_hyworld2_gaussians(
    reconstruction, training_frames, output_path, training_steps, max_splats,
    sh_degree, ssim_weight, alpha_weight, use_scale_regularization, antialiased,
    optimizer_mode, refine_start_iter, refine_stop_iter, refine_every,
    reset_every, grow_grad2d, prune_opacity, seed, progress, interrupt,
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("HYWorld2 Gaussian training requires an NVIDIA CUDA build of PyTorch")
    try:
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy.ops import remove
        if optimizer_mode == "visible_adam":
            from gsplat.optimizers import SelectiveAdam
    except ImportError as error:
        raise RuntimeError("gsplat 1.5.3 is required for HYWorld2 Gaussian training") from error

    device = torch.device("cuda", torch.cuda.current_device())
    points, rgb, frames = _scene_data(reconstruction, training_frames)
    scene_scale = _hyworld2_scene_scale(frames)
    torch.manual_seed(seed)
    initial_scales = _hyworld2_initial_scales(points)
    if len(points) > max_splats:
        selected = torch.randperm(len(points))[:max_splats].numpy()
        points, rgb = points[selected], rgb[selected]
        initial_scales = initial_scales[selected]
    count = len(points)
    colors = torch.zeros((count, (sh_degree + 1) ** 2, 3), dtype=torch.float32)
    colors[:, 0] = (torch.from_numpy(rgb) - 0.5) / SH_C0
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(points)),
        "scales": torch.nn.Parameter(
            torch.from_numpy(initial_scales)[:, None].repeat(1, 3)
        ),
        "quats": torch.nn.Parameter(torch.rand((count, 4))),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((count,), 0.1))),
        "sh0": torch.nn.Parameter(colors[:, :1]),
        "shN": torch.nn.Parameter(colors[:, 1:], requires_grad=sh_degree > 0),
    }).to(device)
    learning_rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    optimizer_class = SelectiveAdam if optimizer_mode == "visible_adam" else torch.optim.Adam
    optimizer_options = {"eps": 1e-15, "betas": (0.9, 0.999)}
    if optimizer_mode == "fused_adam":
        optimizer_options["fused"] = True
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": rate, "name": name}],
            **optimizer_options,
        )
        for name, rate in learning_rates.items()
        if splats[name].requires_grad
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / training_steps)
    )
    strategy = _hyworld2_strategy(
        training_steps, refine_start_iter, refine_stop_iter, refine_every,
        reset_every, grow_grad2d, prune_opacity,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    state["max_gaussians"] = max_splats
    frames = [
        {
            "rgba": torch.from_numpy(frame["rgba"]).contiguous().pin_memory(),
            "view": torch.from_numpy(frame["view"]).to(device)[None],
            "K": torch.from_numpy(frame["K"]).to(device)[None],
        }
        for frame in frames
    ]
    kernel = None
    if ssim_weight > 0:
        coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
        kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
        kernel /= kernel.sum()

    # Shuffled epochs give every registered view equal coverage, as in HYWorld2.
    generator = torch.Generator().manual_seed(seed)
    order = iter(())
    transfer_stream = torch.cuda.Stream(device=device)
    training_stream = torch.cuda.current_stream(device)

    def prefetch_frame():
        nonlocal order
        try:
            index = next(order)
        except StopIteration:
            order = iter(torch.randperm(len(frames), generator=generator).tolist())
            index = next(order)
        with torch.cuda.stream(transfer_stream):
            rgba = frames[index]["rgba"].to(
                device=device, dtype=torch.float32, non_blocking=True
            ).div_(255.0)
        return index, rgba

    prefetched = prefetch_frame()
    started = time.perf_counter()
    print(
        f"[VNCCS HYWorld2] Training: frames={len(frames)}, initial_splats={count}, "
        f"steps={training_steps}, cap={max_splats}, scene_scale={scene_scale:.6g}, "
        f"means_lr={learning_rates['means']:.6g}, optimizer={optimizer_mode}"
    )
    print(
        f"[VNCCS HYWorld2] ADC: start={strategy.refine_start_iter}, "
        f"stop={strategy.refine_stop_iter}, every={strategy.refine_every}, "
        f"grow_grad2d={strategy.grow_grad2d:g}, reset_every={strategy.reset_every}; "
        f"SSIM={ssim_weight:g}, alpha={alpha_weight:g}, "
        f"scale_regularization={use_scale_regularization}, antialiased={antialiased}, "
        f"SH={sh_degree} from step 1; full-resolution shuffled frames"
    )
    print(f"[VNCCS HYWorld2] Output: {output_path}")
    for step in range(training_steps):
        interrupt()
        training_stream.wait_stream(transfer_stream)
        frame_index, rgba = prefetched
        rgba.record_stream(training_stream)
        frame = frames[frame_index]
        if step + 1 < training_steps:
            prefetched = prefetch_frame()
        background = torch.rand((1, 1, 1, 3), device=device)
        target_alpha = rgba[None, ..., 3:4]
        target = rgba[None, ..., :3] * target_alpha + background * (1 - target_alpha)
        height, width = rgba.shape[:2]
        rendered, alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat((splats["sh0"], splats["shN"]), dim=1),
            viewmats=frame["view"], Ks=frame["K"], width=width, height=height,
            packed=False, absgrad=strategy.absgrad, sh_degree=sh_degree,
            near_plane=0.01, far_plane=1e10,
            rasterize_mode="antialiased" if antialiased else "classic",
        )
        rendered = rendered + background * (1 - alpha)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1, ssim, alpha_loss, scale_loss = _hyworld2_loss(
            rendered, target, alpha, target_alpha, kernel, alpha_weight,
            splats["scales"], use_scale_regularization and step % 10 == 0,
        )
        loss = (1 - ssim_weight) * l1 + ssim_weight * ssim + alpha_loss + scale_loss
        loss.backward()
        visibility = (info["radii"] > 0).all(dim=-1).any(dim=0)
        for optimizer in optimizers.values():
            if optimizer_mode == "visible_adam":
                optimizer.step(visibility)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        completed = step + 1
        if completed == 1 or completed % 25 == 0 or completed == training_steps:
            phase = "densify" if step < strategy.refine_stop_iter else "finetune"
            print(
                f"[VNCCS HYWorld2] {completed}/{training_steps} ({phase}): "
                f"loss={loss.detach().item():.6f}, l1={l1.detach().item():.6f}, "
                f"ssim={ssim.detach().item():.6f}, alpha={alpha_loss.detach().item():.6f}, "
                f"scale={scale_loss.detach().item():.6f}, splats={len(splats['means'])}, "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
            progress.update_absolute(completed)

    interrupt()
    with torch.no_grad():
        finite = torch.ones(len(splats["means"]), dtype=torch.bool, device=device)
        for value in splats.values():
            valid = torch.isfinite(value)
            if value.ndim > 1:
                valid = valid.flatten(start_dim=1).all(dim=-1)
            finite &= valid
        finite &= torch.isfinite(torch.exp(splats["scales"])).all(dim=-1)
        if not finite.any():
            raise RuntimeError("HYWorld2 training produced no finite Gaussian parameters")
        if not finite.all():
            remove(params=splats, optimizers=optimizers, state=state, mask=~finite)
        splats["quats"].copy_(F.normalize(splats["quats"], dim=-1))
        with atomic_output_path(output_path) as temporary:
            export_splats(
                means=splats["means"], scales=splats["scales"], quats=splats["quats"],
                opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
                format="ply", save_to=temporary,
            )
        result = _comfy_splat(splats)
    print(f"[VNCCS HYWorld2] Saved {len(splats['means'])} splats to {output_path}")
    progress.update_absolute(training_steps)
    return result


class VNCCSGaussianTrainingHYWorld2:
    """Train a cached VNCCS orbit scene with HYWorld2's standard ADC trainer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": (GAUSSIAN_SCENE_TYPE,),
                "training_steps": ("INT", {"default": 8_000, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "max_splats": ("INT", {"default": 2_000_000, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "sh_degree": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
                "ssim_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.01}),
                "use_scale_regularization": ("BOOLEAN", {"default": True}),
                "antialiased": ("BOOLEAN", {"default": True}),
                "optimizer_mode": (["visible_adam", "fused_adam", "adam"], {"default": "visible_adam"}),
                "refine_start_iter": ("INT", {"default": 150, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "refine_stop_iter": ("INT", {"default": 3_500, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "refine_every": ("INT", {"default": 100, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "reset_every": ("INT", {"default": 99_990, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "grow_grad2d": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "prune_opacity": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "SPLAT")
    RETURN_NAMES = ("gaussian_file", "splat")
    FUNCTION = "train"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    def train(
        self, scene, training_steps=8_000, max_splats=2_000_000, sh_degree=1,
        ssim_weight=0.2, alpha_weight=0.1, use_scale_regularization=True,
        antialiased=True, optimizer_mode="visible_adam", refine_start_iter=150,
        refine_stop_iter=3_500, refine_every=100, reset_every=99_990,
        grow_grad2d=0.0001, prune_opacity=0.005, seed=42,
    ):
        if not isinstance(scene, (tuple, list)) or len(scene) != 2:
            raise ValueError("scene must come from VNCCS Gaussian Camera Reconstruction")
        if training_steps < 1 or max_splats < 1:
            raise ValueError("training_steps and max_splats must be positive")
        if not 0 <= sh_degree <= 3:
            raise ValueError("sh_degree must be between zero and three")
        if not 0 <= ssim_weight <= 1 or not 0 <= prune_opacity <= 1:
            raise ValueError("SSIM weight and prune opacity must be between zero and one")
        if not np.isfinite(alpha_weight) or alpha_weight < 0:
            raise ValueError("alpha_weight must be finite and non-negative")
        if not np.isfinite(grow_grad2d) or grow_grad2d < 0:
            raise ValueError("grow_grad2d must be finite and non-negative")
        if optimizer_mode not in {"visible_adam", "fused_adam", "adam"}:
            raise ValueError("Unknown HYWorld2 optimizer mode")
        if refine_every < 1 or reset_every < 1:
            raise ValueError("Refinement and reset intervals must be positive")
        if refine_start_iter < 0 or refine_stop_iter < 0:
            raise ValueError("Refinement steps must be non-negative")
        reconstruction, training_frames = scene
        output_path = _output_path()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        progress, interrupt = _runtime_helpers(training_steps)
        result = _train_hyworld2_gaussians(
            reconstruction, training_frames, output_path, training_steps,
            max_splats, sh_degree, ssim_weight, alpha_weight,
            use_scale_regularization, antialiased, optimizer_mode,
            refine_start_iter, refine_stop_iter, refine_every, reset_every,
            grow_grad2d, prune_opacity, seed, progress, interrupt,
        )
        return output_path, result


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianTrainingHYWorld2": VNCCSGaussianTrainingHYWorld2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianTrainingHYWorld2": "VNCCS Gaussian Training (HYWorld2)",
}
