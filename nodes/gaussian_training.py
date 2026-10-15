"""In-process Gaussian training for camera-calibrated VNCCS orbit scenes.

The dataset supplies RGB, alpha, camera poses and sparse points.
Training uses public gsplat Python APIs without external processes.
"""

from datetime import datetime
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from .gaussian_depth import DEPTH_LOSS_WEIGHT, depth_loss, prepare_depth_targets
from .gaussian_reconstruction import (
    GAUSSIAN_SCENE_TYPE,
    _runtime_helpers,
    _scene_data,
    _top_view_cameras,
)
from .gaussian_2dgs import (
    DISTORTION_WEIGHT, NORMAL_WEIGHT, compatibility_scales, geometry_losses,
    geometry_maps, geometry_schedule, render_2dgs, split_2dgs,
)
from ..utils import atomic_output_path, safe_join_under


SH_C0 = 0.28209479177387814


def _ssim_loss(rendered, target, kernel):
    rendered = rendered.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    channels = rendered.shape[1]
    moments = torch.cat(
        (rendered, target, rendered.square(), target.square(), rendered * target), dim=0
    )
    horizontal = kernel[None, None, None].expand(channels, 1, 1, -1)
    vertical = kernel[None, None, :, None].expand(channels, 1, -1, 1)
    moments = F.conv2d(moments, horizontal, groups=channels)
    moments = F.conv2d(moments, vertical, groups=channels)
    mu_rendered, mu_target, square_rendered, square_target, product = moments.chunk(
        5, dim=0
    )
    mu_rendered_sq = mu_rendered.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_rendered * mu_target
    variance_rendered = square_rendered - mu_rendered_sq
    variance_target = square_target - mu_target_sq
    covariance = product - mu_product
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_rendered_sq + mu_target_sq + c1)
        * (variance_rendered + variance_target + c2)
    )
    return (1.0 - score.mean()).clamp(0.0, 2.0)


def _comfy_splat(splats):
    try:
        from comfy_api.latest import Types
    except ImportError as error:
        raise RuntimeError("ComfyUI SPLAT support is unavailable; update ComfyUI") from error

    return Types.SPLAT(
        splats["means"].detach().cpu()[None],
        compatibility_scales(splats["scales"].detach()).exp().cpu()[None],
        F.normalize(splats["quats"].detach(), dim=-1).cpu()[None],
        torch.sigmoid(splats["opacities"].detach()).cpu()[None, :, None],
        torch.cat((splats["sh0"], splats["shN"]), dim=1).detach().cpu()[None],
    )


def _output_path():
    try:
        import folder_paths
    except ImportError as error:
        raise RuntimeError("ComfyUI output directory is unavailable") from error
    root = safe_join_under(folder_paths.get_output_directory(), "VNCCS", "Gaussians")
    filename = f"gaussian_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.ply"
    return safe_join_under(root, filename)


@torch.no_grad()
def _export_gaussians(splats, optimizers, state, output_path):
    from gsplat import export_splats
    from gsplat.strategy.ops import remove

    finite = torch.ones(len(splats["means"]), dtype=torch.bool, device=splats["means"].device)
    for value in splats.values():
        valid = torch.isfinite(value)
        if value.ndim > 1:
            valid = valid.flatten(start_dim=1).all(dim=-1)
        finite &= valid
    finite &= torch.isfinite(torch.exp(splats["scales"])).all(dim=-1)
    if not finite.any():
        raise RuntimeError("Gaussian training produced no finite Gaussian parameters")
    if not finite.all():
        remove(params=splats, optimizers=optimizers, state=state, mask=~finite)
    splats["quats"].copy_(F.normalize(splats["quats"], dim=-1))
    if splats["scales"].shape[-1] == 2:
        from gsplat.exporter import splat2ply_bytes

        native_path = str(Path(output_path).with_name(Path(output_path).stem + "_2dgs.ply"))
        data = splat2ply_bytes(
            splats["means"], splats["scales"], splats["quats"], splats["opacities"],
            splats["sh0"].squeeze(1), splats["shN"].transpose(1, 2).flatten(start_dim=1),
        )
        with atomic_output_path(native_path) as temporary:
            Path(temporary).write_bytes(data)
        del data
        print(f"[VNCCS Gaussian] Native two-scale 2DGS saved to {native_path}")
    with atomic_output_path(output_path) as temporary:
        export_splats(
            means=splats["means"], scales=compatibility_scales(splats["scales"]), quats=splats["quats"],
            opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
            format="ply", save_to=temporary,
        )


def _scene_scale(frames):
    views = np.stack([frame["view"] for frame in frames]).astype(np.float64)
    centers = np.linalg.inv(views)[:, :3, 3]
    radius = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
    if not np.isfinite(radius) or radius < 1e-6:
        raise ValueError("Gaussian training requires non-degenerate camera positions")
    return float(radius) * 1.1


def _initial_scales(points):
    import cv2

    points = np.ascontiguousarray(points, dtype=np.float32)
    index = cv2.flann_Index(points, {"algorithm": 1, "trees": 4})
    neighbors, _ = index.knnSearch(points, 4, params={"checks": 64})
    offsets = points[neighbors[:, 1:]] - points[:, None]
    mean_squared_distance = np.square(offsets).sum(axis=-1).mean(axis=-1)
    return (0.5 * np.log(np.maximum(mean_squared_distance, 1e-12))).astype(np.float32)


def _strategy(
    training_steps, refine_start_iter, refine_stop_iter, refine_every,
    reset_every, grow_grad2d, prune_opacity, use_2dgs=False,
):
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    class BudgetedDefaultStrategy(DefaultStrategy):
        """Clone small splats, split large ones and respect the cap."""

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
            if use_2dgs:
                clone_mask &= ~split_mask

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
                if use_2dgs:
                    split_2dgs(params, optimizers, state, split_mask)
                else:
                    split(
                        params=params, optimizers=optimizers, state=state,
                        mask=split_mask, revised_opacity=self.revised_opacity,
                    )
            return cloned, divided

        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            if not use_2dgs:
                return super().step_post_backward(params, optimizers, state, step, info, packed)
            # gsplat 1.5.3's reset condition has an operator-precedence bug.
            # Keep its 3D path unchanged and own the 2DGS schedule explicitly.
            if step >= self.refine_stop_iter:
                return
            self._update_state(params, state, info, packed=packed)
            if (step > self.refine_start_iter and step % self.refine_every == 0
                    and step % self.reset_every >= self.pause_refine_after_reset):
                cloned, divided = self._grow_gs(params, optimizers, state, step)
                pruned = self._prune_gs(params, optimizers, state, step)
                if not len(params["means"]):
                    raise RuntimeError("2DGS refinement removed all surfels; lower prune_opacity")
                state["grad2d"].zero_()
                state["count"].zero_()
                state["radii"].zero_()
                print(f"[VNCCS Gaussian] 2DGS refine {step}: cloned={cloned}, "
                      f"split={divided}, pruned={pruned}, surfels={len(params['means'])}")
            if step > 0 and step % self.reset_every == 0:
                from gsplat.strategy.ops import reset_opa

                reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
                print(f"[VNCCS Gaussian] 2DGS opacity reset at step {step}")

    stop = min(refine_stop_iter, training_steps)
    options = dict(
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
    if use_2dgs:
        options["key_for_gradient"] = "gradient_2dgs"
    return BudgetedDefaultStrategy(**options)


def _training_loss(rendered, target, alpha, target_alpha, kernel, alpha_weight,
                   log_scales, scale_regularization):
    l1 = F.l1_loss(rendered, target)
    ssim = _ssim_loss(rendered, target, kernel) if kernel is not None else l1.new_zeros(())
    alpha_loss = F.l1_loss(alpha, target_alpha)
    scale_loss = l1.new_zeros(())
    if scale_regularization:
        # PhysGaussian ratio regularization, evaluated every ten steps.
        log_ratio = log_scales.amax(dim=-1) - log_scales.amin(dim=-1)
        scale_loss = 0.1 * (log_ratio.exp() - 5.0).clamp_min(0.0).mean()
    return l1, ssim, alpha_loss * alpha_weight, scale_loss


@torch.no_grad()
def _render_view_pairs(splats, previews, sh_degree, antialiased, interrupt):
    from gsplat.rendering import rasterization

    device = splats["means"].device
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    images = []
    for index, (frame, view) in enumerate(previews):
        interrupt()
        source = frame["source_rgb"]
        height, width = source.shape[:2]
        corners = source[[0, 0, -1, -1], [0, -1, 0, -1]]
        background = torch.as_tensor(
            np.median(corners, axis=0) / 255.0, device=device, dtype=torch.float32
        )
        if splats["scales"].shape[-1] == 2:
            rendered, alpha, _ = render_2dgs(
                splats, torch.as_tensor(view, device=device)[None],
                torch.as_tensor(frame["K"], device=device)[None], width, height, sh_degree,
            )
        else:
            rendered, alpha, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=scales, opacities=opacities, colors=colors,
                viewmats=torch.as_tensor(view, device=device)[None],
                Ks=torch.as_tensor(frame["K"], device=device)[None],
                width=width, height=height, packed=False, sh_degree=sh_degree,
                near_plane=0.01, far_plane=1e10,
                rasterize_mode="antialiased" if antialiased else "classic",
            )
        images.extend((
            (rendered + background * (1 - alpha)).clamp(0, 1).float().cpu(),
            torch.from_numpy(source).float()[None] / 255.0,
        ))
        print(
            f"[VNCCS Gaussian] Preview {index + 1}/4: azimuth={index * 90}, "
            f"elevation=45; source={frame['name']}"
        )
    return tuple(images)


@torch.inference_mode(False)
@torch.enable_grad()
def _train_gaussians(
    reconstruction, training_frames, output_path, training_steps, max_splats,
    sh_degree, ssim_weight, alpha_weight, use_scale_regularization, antialiased,
    optimizer_mode, refine_start_iter, refine_stop_iter, refine_every,
    reset_every, grow_grad2d, prune_opacity, seed, progress, interrupt,
    use_depth_loss=False,
    use_2dgs=False, distortion_weight=DISTORTION_WEIGHT, normal_weight=NORMAL_WEIGHT,
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Gaussian training requires an NVIDIA CUDA build of PyTorch")
    try:
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy.ops import remove
        if optimizer_mode == "visible_adam":
            from gsplat.optimizers import SelectiveAdam
    except ImportError as error:
        raise RuntimeError("gsplat 1.5.3 is required for Gaussian training") from error

    device = torch.device("cuda", torch.cuda.current_device())
    if use_2dgs and (use_depth_loss or distortion_weight > 0 or normal_weight > 0):
        from .gaussian_2dgs_cuda import prepare_surface_cuda

        print(f"[VNCCS Gaussian] Initializing fused surface kernels for PyTorch CUDA {torch.version.cuda}")
        prepare_surface_cuda(device)
    points, rgb, frames = _scene_data(reconstruction, training_frames)
    previews = _top_view_cameras(frames)
    if any(frame["source_rgb"] is None for frame, _ in previews):
        raise ValueError(
            "The cached scene has no original RGB references; rerun "
            "VNCCS Gaussian Camera Reconstruction before training"
        )
    scene_scale = _scene_scale(frames)
    if use_depth_loss:
        prepare_depth_targets(reconstruction, training_frames, points, frames, interrupt)
    torch.manual_seed(seed)
    initial_scales = _initial_scales(points)
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
            torch.from_numpy(initial_scales)[:, None].repeat(1, 2 if use_2dgs else 3)
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
    strategy = _strategy(
        training_steps, refine_start_iter, refine_stop_iter, refine_every,
        reset_every, grow_grad2d, prune_opacity, use_2dgs,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    state["max_gaussians"] = max_splats
    frames = [
        {
            "rgba": torch.from_numpy(frame["rgba"]).contiguous().pin_memory(),
            "view": torch.from_numpy(frame["view"]).to(device)[None],
            "K": torch.from_numpy(frame["K"]).to(device)[None],
            "depth": (
                torch.from_numpy(frame["depth"]).pin_memory()
                if use_depth_loss else None
            ),
        }
        for frame in frames
    ]
    kernel = None
    if ssim_weight > 0:
        coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
        kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
        kernel /= kernel.sum()

    # Shuffled epochs give every registered view equal coverage.
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
            depth = (
                frames[index]["depth"].to(device=device, non_blocking=True)[None, ..., None]
                if use_depth_loss else None
            )
        return index, rgba, depth

    prefetched = prefetch_frame()
    started = time.perf_counter()
    last_log = started
    last_log_step = 0
    print(
        f"[VNCCS Gaussian] Training: frames={len(frames)}, initial_splats={count}, "
        f"steps={training_steps}, cap={max_splats}, scene_scale={scene_scale:.6g}, "
        f"means_lr={learning_rates['means']:.6g}, optimizer={optimizer_mode}"
    )
    print(
        f"[VNCCS Gaussian] ADC: start={strategy.refine_start_iter}, "
        f"stop={strategy.refine_stop_iter}, every={strategy.refine_every}, "
        f"grow_grad2d={strategy.grow_grad2d:g}, reset_every={strategy.reset_every}; "
        f"SSIM={ssim_weight:g}, alpha={alpha_weight:g}, "
        f"scale_regularization={use_scale_regularization}, antialiased={antialiased}, "
        f"depth_supervision={use_depth_loss}, depth_weight={DEPTH_LOSS_WEIGHT:g}; "
        f"SH={sh_degree} ({'progressive' if use_2dgs else 'from step 1'}); full-resolution shuffled frames"
    )
    print(f"[VNCCS Gaussian] Output: {output_path}")
    if use_2dgs:
        print(
            f"[VNCCS Gaussian] 2DGS: two tangential scales; planar splitting; "
            f"gsplat projection/tiles with built-in low-pass filter; "
            f"fused CUDA RGB/geometry forward and backward; no feature images or hit lists; "
            f"ray-plane median depth (no stock center-depth interpolation); "
            f"distortion_l2_ndc={distortion_weight:g} after {int(training_steps * 0.1)} steps, "
            f"normal={normal_weight:g} after {int(training_steps * 7 / 30)} steps; "
            "SH increases every 1000 steps; geometry uses all pixels and contributors every step. "
            "Optional MoGe supervision uses bounded-scene median depth. "
            "PLY/SPLAT outputs are thin-3DGS compatibility approximations; "
            "native two-scale PLY is saved alongside; IMAGE outputs use native 2DGS."
        )
    for step in range(training_steps):
        interrupt()
        training_stream.wait_stream(transfer_stream)
        frame_index, rgba, target_depth = prefetched
        rgba.record_stream(training_stream)
        if target_depth is not None:
            target_depth.record_stream(training_stream)
        frame = frames[frame_index]
        if step + 1 < training_steps:
            prefetched = prefetch_frame()
        background = torch.rand((1, 1, 1, 3), device=device)
        target_alpha = rgba[None, ..., 3:4]
        target = rgba[None, ..., :3] * target_alpha + background * (1 - target_alpha)
        height, width = rgba.shape[:2]
        distortion_term = normal_term = rgba.new_zeros(())
        if use_2dgs:
            distortion_scale, normal_scale = geometry_schedule(
                step, training_steps, distortion_weight, normal_weight,
            )
            need_geometry = use_depth_loss or distortion_scale > 0 or normal_scale > 0
            rendered, alpha, info = render_2dgs(
                splats, frame["view"], frame["K"], width, height,
                min(step // 1_000, sh_degree), strategy.absgrad,
                with_geometry=need_geometry,
            )
            rendered_depth = None
            if need_geometry:
                maps = geometry_maps(info, frame["K"])
                distortion, normal, rendered_depth = geometry_losses(
                    maps, frame["view"], frame["K"], use_normal=normal_scale > 0,
                )
                del maps
                distortion_term = distortion_scale * distortion
                normal_term = normal_scale * normal
        else:
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
                render_mode="RGB+ED" if use_depth_loss else "RGB",
            )
            rendered_depth = rendered[..., 3:4] if use_depth_loss else None
        rendered = rendered[..., :3]
        rendered = rendered + background * (1 - alpha)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1, ssim, alpha_loss, scale_loss = _training_loss(
            rendered, target, alpha, target_alpha, kernel, alpha_weight,
            splats["scales"], use_scale_regularization and step % 10 == 0,
        )
        loss = (1 - ssim_weight) * l1 + ssim_weight * ssim + alpha_loss + scale_loss
        depth_term = (
            DEPTH_LOSS_WEIGHT * depth_loss(rendered_depth, target_depth, alpha)
            if use_depth_loss else l1.new_zeros(())
        )
        loss = loss + depth_term + distortion_term + normal_term
        if use_2dgs and not torch.isfinite(loss.detach()).item():
            raise RuntimeError(f"2DGS produced a non-finite loss at step {step + 1}; training stopped before the optimizer update")
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
        if (completed == 1 or completed % 25 == 0 or completed == training_steps
                or (use_2dgs and time.perf_counter() - last_log >= 5.0)):
            phase = "densify" if step < strategy.refine_stop_iter else "finetune"
            now = time.perf_counter()
            speed = (
                f", ms_per_step={(now - last_log) * 1000 / (completed - last_log_step):.2f}"
                if use_2dgs else ""
            )
            print(
                f"[VNCCS Gaussian] {completed}/{training_steps} ({phase}): "
                f"loss={loss.detach().item():.6f}, l1={l1.detach().item():.6f}, "
                f"ssim={ssim.detach().item():.6f}, alpha={alpha_loss.detach().item():.6f}, "
                f"scale={scale_loss.detach().item():.6f}, splats={len(splats['means'])}, "
                f"depth={depth_term.detach().item():.6f}, "
                f"distortion={distortion_term.detach().item():.6f}, "
                f"normal={normal_term.detach().item():.6f}, "
                f"elapsed={now - started:.1f}s{speed}"
            )
            progress.update_absolute(completed)
            last_log = time.perf_counter()
            last_log_step = completed

    interrupt()
    _export_gaussians(splats, optimizers, state, output_path)
    result = _comfy_splat(splats)
    print(f"[VNCCS Gaussian] Saved {len(splats['means'])} splats to {output_path}")
    preview_degree = min((training_steps - 1) // 1_000, sh_degree) if use_2dgs else sh_degree
    preview_images = _render_view_pairs(splats, previews, preview_degree, antialiased, interrupt)
    progress.update_absolute(training_steps)
    return (result, *preview_images)


class VNCCSGaussianTrainer:
    """Train Gaussian splats from a cached VNCCS orbit scene."""

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
                "use_depth_loss": (
                    "BOOLEAN", {
                        "default": False,
                        "tooltip": "Compute and cache native ComfyUI MoGe-2 depth, align it to COLMAP and supervise foreground geometry.",
                    },
                ),
                "use_2dgs": (
                    "BOOLEAN", {
                        "default": False,
                        "tooltip": "Train planar 2D Gaussians with exact ray-plane median depth, squared NDC depth distortion and normal consistency. Fused rendering uses the current PyTorch CUDA release; install.py supplies matching NVRTC only if missing. No MoGe required. PLY/SPLAT outputs approximate surfels as thin 3D Gaussians; native 2DGS PLY is saved alongside.",
                    },
                ),
                "distortion_weight": (
                    "FLOAT", {
                        "default": DISTORTION_WEIGHT, "min": 0.0, "step": 10.0,
                        "tooltip": "2DGS squared NDC-depth distortion weight, enabled after 10% of training. Zero disables this loss.",
                    },
                ),
                "normal_weight": (
                    "FLOAT", {
                        "default": NORMAL_WEIGHT, "min": 0.0, "step": 0.01,
                        "tooltip": "2DGS normal-consistency weight, enabled after 7/30 of training. Zero disables this loss.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "SPLAT") + ("IMAGE",) * 8
    RETURN_NAMES = (
        "gaussian_file", "splat",
        "top_0", "source_0",
        "top_90", "source_90",
        "top_180", "source_180",
        "top_270", "source_270",
    )
    FUNCTION = "train"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    def train(
        self, scene, training_steps=8_000, max_splats=2_000_000, sh_degree=1,
        ssim_weight=0.2, alpha_weight=0.1, use_scale_regularization=True,
        antialiased=True, optimizer_mode="visible_adam", refine_start_iter=150,
        refine_stop_iter=3_500, refine_every=100, reset_every=99_990,
        grow_grad2d=0.0001, prune_opacity=0.005, seed=42, use_depth_loss=False,
        use_2dgs=False, distortion_weight=DISTORTION_WEIGHT, normal_weight=NORMAL_WEIGHT,
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
            raise ValueError("Unknown Gaussian optimizer mode")
        if refine_every < 1 or reset_every < 1:
            raise ValueError("Refinement and reset intervals must be positive")
        if refine_start_iter < 0 or refine_stop_iter < 0:
            raise ValueError("Refinement steps must be non-negative")
        if use_2dgs and (not np.isfinite(distortion_weight) or distortion_weight < 0):
            raise ValueError("distortion_weight must be finite and non-negative")
        if use_2dgs and (not np.isfinite(normal_weight) or normal_weight < 0):
            raise ValueError("normal_weight must be finite and non-negative")
        reconstruction, training_frames = scene
        output_path = _output_path()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        progress, interrupt = _runtime_helpers(training_steps)
        result = _train_gaussians(
            reconstruction, training_frames, output_path, training_steps,
            max_splats, sh_degree, ssim_weight, alpha_weight,
            use_scale_regularization, antialiased, optimizer_mode,
            refine_start_iter, refine_stop_iter, refine_every, reset_every,
            grow_grad2d, prune_opacity, seed, progress, interrupt, use_depth_loss,
            use_2dgs, distortion_weight, normal_weight,
        )
        return (output_path, *result)


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianTrainer": VNCCSGaussianTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianTrainer": "VNCCS Gaussian Trainer",
}
