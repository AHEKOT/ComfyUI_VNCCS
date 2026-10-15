"""Native image-batch to 3D Gaussian reconstruction."""

from datetime import datetime
import math
from pathlib import Path
import tempfile
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .chroma_screen_matte import screen_matte
from ..utils import atomic_output_path, safe_join_under


DEFAULT_MAX_FRAMES = 160
DEFAULT_MAX_SIDE = 1600
DEFAULT_MATCH_WINDOW = 20
DEFAULT_TRAINING_STEPS = 7_000
DEFAULT_MAX_SPLATS = 2_000_000
DEFAULT_SH_DEGREE = 1
DEFAULT_MATTE_TOLERANCE = 0.15
SH_C0 = 0.28209479177387814
INITIAL_OPACITY = 0.5
GROWTH_SELECT_FRACTION = 0.25
FINAL_PRUNE_OPACITY = 0.01
GAUSSIAN_SCENE_TYPE = "VNCCS_GAUSSIAN_SCENE"


def _sharpest_indices(scores, target):
    """Pick the sharpest item from each consecutive window."""
    total = len(scores)
    target = min(max(1, int(target)), total)
    selected = []
    for window in range(target):
        start = window * total // target
        stop = max(start + 1, (window + 1) * total // target)
        selected.append(max(range(start, stop), key=lambda index: (scores[index], index)))
    return selected


@torch.inference_mode()
def _select_frames(images, target=DEFAULT_MAX_FRAMES):
    if len(images) <= target:
        return list(range(len(images)))
    scores = []
    for frame in images:
        rgb = frame[..., :3].float().clamp(0, 1).permute(2, 0, 1)[None]
        height, width = rgb.shape[-2:]
        scale = min(1.0, 640.0 / max(height, width))
        if scale < 1.0:
            rgb = F.interpolate(
                rgb,
                size=(max(3, round(height * scale)), max(3, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            )
        gray = rgb[:, :1] * 0.2126 + rgb[:, 1:2] * 0.7152 + rgb[:, 2:3] * 0.0722
        laplacian = (
            gray[:, :, 1:-1, :-2]
            + gray[:, :, 1:-1, 2:]
            + gray[:, :, :-2, 1:-1]
            + gray[:, :, 2:, 1:-1]
            - 4.0 * gray[:, :, 1:-1, 1:-1]
        )
        scores.append(float(laplacian.var(unbiased=False).cpu()))
    return _sharpest_indices(scores, target)


def _circular_pairs(names, window=DEFAULT_MATCH_WINDOW):
    """Return local and long-range pairs for a closed orbit."""
    count = len(names)
    limit = count // 2
    offsets = list(range(1, window + 1))
    offsets.extend(offset for offset in (16, 32, 64) if offset > window)
    offsets = [offset for offset in offsets if offset <= limit]
    pairs = set()
    for left in range(count):
        for offset in offsets:
            right = (left + offset) % count
            if left != right:
                pairs.add((min(left, right), max(left, right)))
    return [(names[left], names[right]) for left, right in sorted(pairs)]


def _pycolmap_cuda_available(pycolmap):
    device_count = getattr(pycolmap, "get_num_cuda_devices", None)
    if not callable(device_count):
        return False
    try:
        return device_count() > 0
    except RuntimeError:
        return False


def _best_reconstruction(reconstructions):
    models = list((reconstructions or {}).values())
    return max(models, key=lambda model: model.num_reg_images(), default=None)


def _resize_frame(frame, max_side=DEFAULT_MAX_SIDE):
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return frame.float().clamp(0, 1)
    resized = F.interpolate(
        frame.permute(2, 0, 1)[None].float(),
        size=(round(height * scale), round(width * scale)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized[0].permute(1, 2, 0).clamp(0, 1)


def _erode_training_alpha(alpha):
    """Match Gausify's one-pixel training-mask erosion."""
    return -F.max_pool2d(-alpha[None, None], kernel_size=3, stride=1, padding=1)[0, 0]


def _save_frames(
    images, indices, image_dir, mask_dir, max_side, matte_tolerance, progress, interrupt
):
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    training_frames = {}
    foreground_coverage = []
    names = []
    for position, index in enumerate(indices):
        interrupt()
        frame = _resize_frame(images[index], max_side)
        name = f"{position:05d}.png"
        names.append(name)
        rgb = (frame[..., :3] * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(rgb).save(image_dir / name)

        rgba, alpha, _ = screen_matte(
            frame,
            tolerance=matte_tolerance,
            output_mode="straight_rgba",
        )
        foreground_coverage.append(float(alpha.float().mean().cpu()))
        mask = (alpha * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(mask).save(mask_dir / f"{name}.png")
        training_alpha = _erode_training_alpha(alpha)
        training_rgba = torch.cat((rgba[..., :3], training_alpha[..., None]), dim=-1)
        training_frames[name] = (
            (training_rgba * 255).round().to(torch.uint8).cpu().numpy()
        )
        progress.update_absolute(position + 1)
    if float(np.mean(foreground_coverage)) < 0.01:
        raise ValueError("The input batch contains no detectable foreground subject")
    return names, training_frames


def _solve_cameras(
    pycolmap,
    image_dir,
    mask_dir,
    work_dir,
    names,
    match_window,
    use_cuda,
    progress,
    interrupt,
):
    database_path = work_dir / "database.db"
    pairs_path = work_dir / "pairs.txt"
    pairs_path.write_text(
        "".join(
            f"{left} {right}\n"
            for left, right in _circular_pairs(names, match_window)
        ),
        encoding="utf-8",
    )

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = 0.002
    extraction.sift.domain_size_pooling = True
    extraction.sift.estimate_affine_shape = True
    reader = pycolmap.ImageReaderOptions(
        camera_model="SIMPLE_PINHOLE",
        mask_path=mask_dir,
    )
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cuda if use_cuda else pycolmap.Device.cpu,
    )
    progress.update_absolute(len(names) + 1)
    interrupt()

    pycolmap.match_image_pairs(
        database_path,
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs_path),
        device=pycolmap.Device.cuda if use_cuda else pycolmap.Device.cpu,
    )
    progress.update_absolute(len(names) + 2)
    interrupt()

    try:
        pycolmap.calibrate_view_graph(database_path)
    except RuntimeError as error:
        print(f"[VNCCS Gaussian] View graph calibration skipped: {error}")

    global_dir = work_dir / "sparse_global"
    global_dir.mkdir()
    global_options = pycolmap.GlobalPipelineOptions(
        mapper={
            "global_positioning": {"use_gpu": use_cuda, "gpu_index": "0"},
            "bundle_adjustment": {"ceres": {"use_gpu": use_cuda, "gpu_index": "0"}},
            "ba_gpu_index": "0",
        }
    )
    try:
        best = _best_reconstruction(
            pycolmap.global_mapping(database_path, image_dir, global_dir, options=global_options)
        )
    except RuntimeError as error:
        print(f"[VNCCS Gaussian] Global mapping failed: {error}")
        best = None
    progress.update_absolute(len(names) + 3)
    interrupt()

    if best is None or best.num_reg_images() / len(names) < 0.6:
        incremental_dir = work_dir / "sparse_incremental"
        incremental_dir.mkdir()
        incremental_options = pycolmap.IncrementalPipelineOptions(
            mapper={"ba_use_gpu": use_cuda, "ba_gpu_index": "0"}
        )
        candidate = _best_reconstruction(
            pycolmap.incremental_mapping(
                database_path,
                image_dir,
                incremental_dir,
                options=incremental_options,
            )
        )
        if candidate is not None and (
            best is None or candidate.num_reg_images() > best.num_reg_images()
        ):
            best = candidate
    progress.update_absolute(len(names) + 4)

    minimum = min(len(names), max(20, math.ceil(len(names) * 0.5)))
    if best is None or best.num_reg_images() < minimum or best.num_points3D() < 4:
        registered = 0 if best is None else best.num_reg_images()
        raise RuntimeError(
            f"Camera reconstruction registered only {registered}/{len(names)} frames; "
            "use a steadier orbit with more visible character detail"
        )
    print(
        f"[VNCCS Gaussian] Cameras: {best.num_reg_images()}/{len(names)}, "
        f"points: {best.num_points3D()}, error: {best.compute_mean_reprojection_error():.3f}px"
    )
    return best


def _scene_data(reconstruction, training_frames):
    registered = []
    images = dict(reconstruction.images)
    cameras = dict(reconstruction.cameras)
    for image_id in reconstruction.reg_image_ids():
        image = images[int(image_id)]
        if image.name not in training_frames:
            continue
        pose = image.cam_from_world()
        view = np.eye(4, dtype=np.float64)
        view[:3, :4] = np.asarray(pose.matrix(), dtype=np.float64)
        camera = cameras[int(image.camera_id)]
        registered.append(
            (
                image.name,
                view,
                np.asarray(camera.calibration_matrix(), dtype=np.float64),
                training_frames[image.name],
            )
        )
    registered.sort(key=lambda item: item[0])

    points3d = dict(reconstruction.points3D)
    point_ids = sorted(points3d)
    points = np.asarray([points3d[key].xyz for key in point_ids], dtype=np.float32)
    colors = np.asarray([points3d[key].color for key in point_ids], dtype=np.float32) / 255.0
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) < 4:
        raise RuntimeError("Camera reconstruction produced too few finite 3D points")

    views = np.stack([item[1] for item in registered])
    camera_to_world = np.linalg.inv(views)
    center = np.median(camera_to_world[:, :3, 3], axis=0)
    radius = np.median(np.linalg.norm(camera_to_world[:, :3, 3] - center, axis=1))
    if not np.isfinite(radius) or radius < 1e-6:
        raise RuntimeError("Camera reconstruction has a degenerate orbit")
    points = (points - center) / radius
    camera_to_world[:, :3, 3] = (camera_to_world[:, :3, 3] - center) / radius
    views = np.linalg.inv(camera_to_world).astype(np.float32)

    frames = [
        {
            "view": views[index],
            "K": item[2].astype(np.float32),
            "rgba": item[3],
        }
        for index, item in enumerate(registered)
    ]
    return points.astype(np.float32), colors.astype(np.float32), frames


def _point_scene_scale(points):
    sorted_points = np.sort(points, axis=0)
    count = len(sorted_points)
    lower = sorted_points[int(0.125 * count)]
    upper = sorted_points[min(count - 1, int(0.875 * count))]
    return max(float(np.median(upper - lower)), 0.01)


def _initial_scales(points):
    import cv2

    index = cv2.flann_Index(np.ascontiguousarray(points), {"algorithm": 1, "trees": 4})
    neighbors, _ = index.knnSearch(points, 3, params={"checks": 32})
    offsets = points[neighbors[:, 1:]] - points[:, None]
    distances = np.linalg.norm(offsets, axis=2).sum(axis=1) / 4.0
    distances = np.clip(distances, 1e-3, _point_scene_scale(points) * 0.1)
    return np.log(distances).astype(np.float32)


def _training_learning_rates(scene_scale):
    return {
        "means": 2e-5 * scene_scale,
        "scales": 5e-3,
        "quats": 2e-3,
        "opacities": 1.2e-2,
        "sh0": 2e-3,
        "shN": 2e-4,
    }


def _growth_mask(grads, candidates, available, fraction):
    """Select strongest refinement candidates within the splat budget."""
    if available <= 0:
        return torch.zeros_like(candidates)
    candidate_ids = torch.where(candidates)[0]
    if not len(candidate_ids):
        return candidates

    selected_count = min(
        available,
        max(1, int(len(candidate_ids) * fraction + 0.5)),
    )
    if selected_count == len(candidate_ids):
        return candidates

    selected = torch.topk(grads[candidate_ids], k=selected_count, sorted=False).indices
    limited = torch.zeros_like(candidates)
    limited[candidate_ids[selected]] = True
    return limited


def _strategy_options(training_steps, stable_refinement):
    options = {
        "grow_grad2d": 0.0002,
        "grow_scale2d": 0.05,
        "refine_every": 100,
        "refine_start_iter": 0,
        "refine_stop_iter": min(training_steps, 15_000),
        "refine_scale2d_stop_iter": min(training_steps, 15_000),
    }
    if stable_refinement:
        refine_stop = max(1, training_steps // 2)
        options.update(
            grow_grad2d=0.0001,
            refine_stop_iter=refine_stop,
            refine_scale2d_stop_iter=refine_stop,
        )
    return options


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


def _training_loss(rendered, target, alpha, target_alpha, ssim_kernel=None):
    color_loss = F.l1_loss(rendered, target)
    if ssim_kernel is not None:
        color_loss = 0.8 * color_loss + 0.2 * _ssim_loss(
            rendered, target, ssim_kernel
        )
    return color_loss + 0.1 * F.l1_loss(alpha, target_alpha)


def _decayed_opacity_logits(opacity_logits, step, training_steps):
    progress = min(max(step / max(training_steps, 1), 0.0), 1.0)
    opacity = torch.sigmoid(opacity_logits) - 0.004 * (1.0 - progress)
    epsilon = torch.finfo(opacity.dtype).eps
    return torch.logit(opacity.clamp(epsilon, 1.0 - epsilon))


def _brush_split_parameters(
    means, log_scales, quats, opacity_logits, screen_sizes=None
):
    scales = torch.exp(log_scales)
    ratio = scales.square() / scales.square().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    shrink_limit = torch.full_like(ratio[:, :1], 1.0 / math.sqrt(2.0))
    if screen_sizes is not None:
        shrink_limit = torch.minimum(
            shrink_limit,
            0.5 / screen_sizes[:, None].clamp_min(1e-6),
        )
    shrink = 1.0 - ratio * (1.0 - shrink_limit)
    offset = torch.sqrt((1.0 - shrink.square()).clamp_min(0.0)) * scales

    rotation = F.normalize(quats, dim=-1)
    vector, scalar = rotation[:, 1:], rotation[:, :1]
    twice_cross = 2.0 * torch.cross(vector, offset, dim=-1)
    offset = offset + scalar * twice_cross + torch.cross(vector, twice_cross, dim=-1)

    child_means = torch.cat((means - offset, means + offset))
    child_log_scales = (log_scales + torch.log(shrink)).repeat(2, 1)
    opacity = 1.0 - (1.0 - torch.sigmoid(opacity_logits)).pow(1.0 / math.sqrt(2.0))
    child_opacity_logits = torch.logit(
        opacity.clamp(1.0 / 255.0, 254.0 / 255.0)
    ).repeat(2)
    return child_means, child_log_scales, child_opacity_logits


def _splat_prune_masks(splats, scene_scale, opacity_threshold, scale_ratio=None):
    count = len(splats["means"])
    device = splats["means"].device
    scales = torch.exp(splats["scales"])
    finite = torch.ones(count, dtype=torch.bool, device=device)
    for name in ("means", "quats", "opacities", "sh0", "shN"):
        finite &= torch.isfinite(splats[name]).reshape(count, -1).all(dim=-1)
    finite &= torch.isfinite(scales).all(dim=-1)

    nonfinite = ~finite
    opacity = finite & (torch.sigmoid(splats["opacities"]) < opacity_threshold)
    scale = torch.zeros_like(nonfinite)
    if scale_ratio is not None:
        scale = finite & ~opacity & (scales.amax(dim=-1) > scene_scale * scale_ratio)
    return {"nonfinite": nonfinite, "opacity": opacity, "scale3d": scale}


def _splat_ranges(splats):
    opacity = torch.sigmoid(splats["opacities"])
    scale = torch.exp(splats["scales"]).amax(dim=-1)

    def finite_range(values):
        values = values[torch.isfinite(values)]
        if not len(values):
            return float("nan"), float("nan")
        return float(values.min().item()), float(values.max().item())

    return finite_range(opacity), finite_range(scale)


def _comfy_splat(splats):
    try:
        from comfy_api.latest import Types
    except ImportError as error:
        raise RuntimeError("ComfyUI SPLAT support is unavailable; update ComfyUI") from error

    return Types.SPLAT(
        splats["means"].detach().cpu()[None],
        torch.exp(splats["scales"].detach()).cpu()[None],
        F.normalize(splats["quats"].detach(), dim=-1).cpu()[None],
        torch.sigmoid(splats["opacities"].detach()).cpu()[None, :, None],
        torch.cat((splats["sh0"], splats["shN"]), dim=1).detach().cpu()[None],
    )


@torch.inference_mode(False)
@torch.enable_grad()
def _train_gaussians(
    reconstruction,
    training_frames,
    output_path,
    training_steps,
    max_splats,
    sh_degree,
    budgeted_densification,
    stable_refinement,
    antialiased_rasterization,
    ssim_alpha_loss,
    progress,
    interrupt,
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Gaussian training requires an NVIDIA CUDA build of PyTorch")
    try:
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
        from gsplat.strategy.ops import remove, split
    except ModuleNotFoundError as error:
        raise RuntimeError("gsplat 1.5.x is required for Gaussian training") from error

    class BrushDefaultStrategy(DefaultStrategy):
        @torch.no_grad()
        def _update_state(self, params, state, info, packed=False):
            grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
            count = len(params["means"])
            if state["grad2d"] is None:
                state["grad2d"] = torch.zeros(count, device=grads.device)
                state["count"] = torch.zeros(count, device=grads.device)
                state["radii"] = torch.zeros(count, device=grads.device)

            if packed:
                gaussian_ids = info["gaussian_ids"]
                radii = info["radii"].max(dim=-1).values
            else:
                visible = (info["radii"] > 0.0).all(dim=-1)
                gaussian_ids = torch.where(visible)[1]
                grads = grads[visible]
                radii = info["radii"][visible].max(dim=-1).values
            state["grad2d"].scatter_reduce_(
                0,
                gaussian_ids,
                torch.nan_to_num(
                    grads.norm(dim=-1), nan=0.0, posinf=1e32, neginf=0.0
                ),
                reduce="amax",
                include_self=True,
            )
            state["count"].index_add_(
                0,
                gaussian_ids,
                torch.ones_like(gaussian_ids, dtype=torch.float32),
            )
            state["radii"].scatter_reduce_(
                0,
                gaussian_ids,
                radii / float(max(info["width"], info["height"])),
                reduce="amax",
                include_self=True,
            )

        @torch.no_grad()
        def _split_gs(self, params, optimizers, state, mask):
            selected = torch.where(mask)[0]
            selected_count = len(selected)
            if not selected_count:
                return 0
            children = _brush_split_parameters(
                params["means"][selected],
                params["scales"][selected],
                params["quats"][selected],
                params["opacities"][selected],
                state["radii"][selected],
            )
            rest_count = len(params["means"]) - selected_count
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=mask,
                revised_opacity=False,
            )
            params["means"][rest_count:].copy_(children[0])
            params["scales"][rest_count:].copy_(children[1])
            params["opacities"][rest_count:].copy_(children[2])
            return selected_count

        @torch.no_grad()
        def _replacement_mask(self, params, state, count):
            available = min(count, max_splats - len(params["means"]))
            if available <= 0:
                return torch.zeros_like(params["opacities"], dtype=torch.bool)
            weights = torch.sigmoid(params["opacities"]) * (state["count"] > 0)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            available = min(available, int((weights > 0).sum().item()))
            if available <= 0:
                return torch.zeros_like(params["opacities"], dtype=torch.bool)
            selected = torch.multinomial(weights, available, replacement=False)
            mask = torch.zeros_like(weights, dtype=torch.bool)
            mask[selected] = True
            return mask

        @torch.no_grad()
        def _growth_mask(self, params, state, step, reserved):
            grads = state["grad2d"]
            available = max_splats - len(params["means"]) - int(reserved.sum().item())
            oversized = torch.zeros_like(reserved)
            if step < self.refine_scale2d_stop_iter:
                candidates = (state["radii"] > self.grow_scale2d) & ~reserved
                oversized = _growth_mask(
                    state["radii"], candidates, available, fraction=1.0
                )
                available -= int(oversized.sum().item())
            candidates = (grads > self.grow_grad2d) & ~reserved & ~oversized
            fraction = GROWTH_SELECT_FRACTION if budgeted_densification else 1.0
            growth = _growth_mask(grads, candidates, available, fraction)
            return growth, oversized

        @torch.no_grad()
        def step_post_backward(
            self, params, optimizers, state, step, info, packed=False
        ):
            self._update_state(params, state, info, packed=packed)
            if step <= self.refine_start_iter or step % self.refine_every:
                return

            opacity_range, scale_range = _splat_ranges(params)
            prune_reasons = _splat_prune_masks(
                params,
                scene_scale,
                self.prune_opa,
                self.prune_scale3d if step >= self.reset_every else None,
            )
            prune_reasons["scale2d"] = (
                (state["radii"] > self.prune_scale2d)
                & ~prune_reasons["nonfinite"]
                & ~prune_reasons["opacity"]
                & ~prune_reasons["scale3d"]
                if self.reset_every <= step < self.refine_scale2d_stop_iter
                else torch.zeros_like(prune_reasons["nonfinite"])
            )
            prune_mask = torch.stack(tuple(prune_reasons.values())).any(dim=0)
            prune_count = int(prune_mask.sum().item())
            if prune_count == len(params["opacities"]):
                finite = ~prune_reasons["nonfinite"]
                if not finite.any():
                    raise RuntimeError("All Gaussian parameters became non-finite")
                protected = torch.argmax(
                    params["opacities"].masked_fill(~finite, -torch.inf)
                )
                for reason in prune_reasons.values():
                    reason[protected] = False
                prune_mask[protected] = False
                prune_count = len(params["opacities"]) - 1
            prune_counts = {
                reason: int(mask.sum().item())
                for reason, mask in prune_reasons.items()
            }
            if prune_count:
                remove(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=prune_mask,
                )

            replacements = self._replacement_mask(params, state, prune_count)
            high_grad_count = int((state["grad2d"] > self.grow_grad2d).sum().item())
            max_grad = float(state["grad2d"].max().item())
            growth, oversized = (
                self._growth_mask(params, state, step, replacements)
                if step < self.refine_stop_iter
                else (torch.zeros_like(replacements), torch.zeros_like(replacements))
            )
            replacement_count = int(replacements.sum().item())
            growth_count = int(growth.sum().item())
            oversized_count = int(oversized.sum().item())
            self._split_gs(
                params, optimizers, state, replacements | oversized | growth
            )

            params["opacities"].copy_(
                _decayed_opacity_logits(params["opacities"], step, training_steps)
            )
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            print(
                f"[VNCCS Gaussian] Refine {step}/{training_steps}: "
                f"eligible={high_grad_count}, max_grad={max_grad:.6g}, "
                f"growth={growth_count}, oversized={oversized_count}, "
                f"replaced={replacement_count}, pruned={prune_count} "
                f"(opacity={prune_counts['opacity']}, "
                f"nonfinite={prune_counts['nonfinite']}, "
                f"scale3d={prune_counts['scale3d']}, "
                f"scale2d={prune_counts['scale2d']}), "
                f"input_opacity_range=[{opacity_range[0]:.6g}, {opacity_range[1]:.6g}], "
                f"input_scale_range=[{scale_range[0]:.6g}, {scale_range[1]:.6g}], "
                f"splats={len(params['means'])}"
            )

    device = torch.device("cuda", torch.cuda.current_device())
    points, rgb, frames = _scene_data(reconstruction, training_frames)
    frames = [
        {
            "rgba": torch.from_numpy(frame["rgba"]).contiguous().pin_memory(),
            "view": torch.from_numpy(frame["view"]).to(device=device)[None],
            "K": torch.from_numpy(frame["K"]).to(device=device)[None],
        }
        for frame in frames
    ]
    torch.manual_seed(42)
    count = len(points)
    scene_scale = _point_scene_scale(points)
    colors = torch.zeros((count, (sh_degree + 1) ** 2, 3), dtype=torch.float32)
    colors[:, 0] = (torch.from_numpy(rgb) - 0.5) / SH_C0
    quats = torch.zeros((count, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points)),
            "scales": torch.nn.Parameter(
                torch.from_numpy(_initial_scales(points))[:, None].repeat(1, 3)
            ),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((count,), INITIAL_OPACITY))
            ),
            "sh0": torch.nn.Parameter(colors[:, :1]),
            "shN": torch.nn.Parameter(colors[:, 1:]),
        }
    ).to(device)
    learning_rates = _training_learning_rates(scene_scale)
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": rate, "name": name}],
            eps=1e-15,
            fused=True,
        )
        for name, rate in learning_rates.items()
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / training_steps)
    )
    strategy = BrushDefaultStrategy(
        **_strategy_options(training_steps, stable_refinement)
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    generator = torch.Generator().manual_seed(42)
    training_stream = torch.cuda.current_stream(device)
    transfer_stream = torch.cuda.Stream(device=device)

    def prefetch_rgba(index):
        with torch.cuda.stream(transfer_stream):
            return frames[index]["rgba"].to(
                device=device, dtype=torch.float32, non_blocking=True
            ).div_(255.0)

    frame_index = int(torch.randint(len(frames), (1,), generator=generator))
    prefetched_rgba = prefetch_rgba(frame_index)
    log_every = max(1, training_steps // 20)
    ssim_kernel = None
    if ssim_alpha_loss:
        coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
        gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        ssim_kernel = gaussian
    print(
        f"[VNCCS Gaussian] Training {count} initial splats from "
        f"{len(frames)} frames for {training_steps} steps"
    )
    print(
        "[VNCCS Gaussian] Options: "
        f"budgeted densification={'on' if budgeted_densification else 'off'}, "
        f"stable refinement={'on' if stable_refinement else 'off'}, "
        f"antialiasing={'on' if antialiased_rasterization else 'off'}, "
        f"SSIM={'on' if ssim_alpha_loss else 'off'}, alpha loss=on, "
        f"Brush max-gradient refinement=on ({strategy.grow_grad2d:g}, "
        f"stop={strategy.refine_stop_iter}), Brush opacity lifecycle=on, "
        "dense rasterization=on, fused Adam=on, frame prefetch=on"
    )

    for step in range(training_steps):
        if step % 10 == 0:
            interrupt()
            progress.update_absolute(step)
        training_stream.wait_stream(transfer_stream)
        frame = frames[frame_index]
        rgba = prefetched_rgba
        rgba.record_stream(training_stream)
        if step + 1 < training_steps:
            next_frame_index = int(torch.randint(len(frames), (1,), generator=generator))
            next_rgba = prefetch_rgba(next_frame_index)
        target_alpha = rgba[..., 3:4]
        background = torch.rand((1, 1, 1, 3), device=device)
        target = rgba[None, ..., :3] * target_alpha[None] + background * (1.0 - target_alpha[None])
        view = frame["view"]
        intrinsics = frame["K"]
        height, width = rgba.shape[:2]
        rendered, alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat((splats["sh0"], splats["shN"]), dim=1),
            viewmats=view,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
            sh_degree=min(step // 1000, sh_degree),
            rasterize_mode="antialiased" if antialiased_rasterization else "classic",
        )
        rendered = rendered + background * (1.0 - alpha)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss = _training_loss(
            rendered,
            target,
            alpha,
            target_alpha[None],
            ssim_kernel if ssim_alpha_loss else None,
        )
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if step + 1 < training_steps:
            frame_index, prefetched_rgba = next_frame_index, next_rgba
        completed = step + 1
        if completed == 1 or completed % log_every == 0 or completed == training_steps:
            print(
                f"[VNCCS Gaussian] Training {completed}/{training_steps} "
                f"({completed / training_steps:.0%}): loss={loss.detach().item():.6f}, "
                f"splats={len(splats['means'])}"
            )

    progress.update_absolute(training_steps)
    opacity_range, scale_range = _splat_ranges(splats)
    final_reasons = _splat_prune_masks(
        splats,
        scene_scale,
        FINAL_PRUNE_OPACITY,
        strategy.prune_scale3d,
    )
    final_mask = torch.stack(tuple(final_reasons.values())).any(dim=0)
    final_count = int(final_mask.sum().item())
    if final_count == len(splats["means"]):
        finite = ~final_reasons["nonfinite"]
        if not finite.any():
            raise RuntimeError("All Gaussian parameters became non-finite")
        protected = torch.argmax(splats["opacities"].masked_fill(~finite, -torch.inf))
        for reason in final_reasons.values():
            reason[protected] = False
        final_mask[protected] = False
        final_count = len(splats["means"]) - 1
    final_counts = {
        reason: int(mask.sum().item()) for reason, mask in final_reasons.items()
    }
    if final_count:
        remove(
            params=splats,
            optimizers=optimizers,
            state=state,
            mask=final_mask,
        )
    print(
        f"[VNCCS Gaussian] Final cleanup: removed={final_count} "
        f"(opacity={final_counts['opacity']}, "
        f"nonfinite={final_counts['nonfinite']}, "
        f"scale3d={final_counts['scale3d']}), "
        f"input_opacity_range=[{opacity_range[0]:.6g}, {opacity_range[1]:.6g}], "
        f"input_scale_range=[{scale_range[0]:.6g}, {scale_range[1]:.6g}], "
        f"kept={len(splats['means'])}"
    )
    with atomic_output_path(output_path) as temporary:
        export_splats(
            means=splats["means"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            format="ply",
            save_to=temporary,
        )
    print(f"[VNCCS Gaussian] Saved {len(splats['means'])} splats to {output_path}")
    return _comfy_splat(splats)


class _NullProgress:
    def update_absolute(self, _value):
        return None


def _runtime_helpers(total):
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar

        return ProgressBar(total), throw_exception_if_processing_interrupted
    except (ImportError, AttributeError):
        return _NullProgress(), lambda: None


def _output_path():
    try:
        import folder_paths
    except ImportError as error:
        raise RuntimeError("ComfyUI output directory is unavailable") from error
    root = safe_join_under(folder_paths.get_output_directory(), "VNCCS", "Gaussians")
    filename = f"gaussian_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.ply"
    return safe_join_under(root, filename)


class VNCCSGaussianCameraReconstruction:
    """Prepare a camera-calibrated Gaussian scene from a solid-background orbit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_frames": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_FRAMES,
                        "min": 20,
                        "max": 220,
                        "step": 10,
                        "tooltip": "Maximum ordered frames selected from the input batch.",
                    },
                ),
                "max_resolution": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_SIDE,
                        "min": 512,
                        "max": 1_920,
                        "step": 64,
                        "tooltip": "Maximum frame width or height used for cameras and training.",
                    },
                ),
                "match_window": (
                    "INT",
                    {
                        "default": DEFAULT_MATCH_WINDOW,
                        "min": 4,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Neighbor range used for closed-orbit camera matching.",
                    },
                ),
                "matte_tolerance": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MATTE_TOLERANCE,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Solid-background removal tolerance. Higher values remove a wider color range.",
                    },
                ),
            }
        }

    RETURN_TYPES = (GAUSSIAN_SCENE_TYPE,)
    RETURN_NAMES = ("scene",)
    FUNCTION = "reconstruct"
    CATEGORY = "VNCCS/3D"

    def reconstruct(
        self,
        images,
        max_frames=DEFAULT_MAX_FRAMES,
        max_resolution=DEFAULT_MAX_SIDE,
        match_window=DEFAULT_MATCH_WINDOW,
        matte_tolerance=DEFAULT_MATTE_TOLERANCE,
    ):
        if images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("images must be a BHWC RGB or RGBA tensor")
        if len(images) < 20:
            raise ValueError("At least 20 ordered orbit frames are required")
        if images.shape[1] < 16 or images.shape[2] < 16:
            raise ValueError("Orbit frames are too small for camera reconstruction")
        try:
            import pycolmap
        except ModuleNotFoundError as error:
            raise RuntimeError("PyCOLMAP 4.2.x is required for camera reconstruction") from error

        indices = _select_frames(images, max_frames)
        progress, interrupt = _runtime_helpers(len(indices) + 4)
        use_cuda = _pycolmap_cuda_available(pycolmap)
        print("[VNCCS Gaussian] Camera reconstruction started")
        print(
            "[VNCCS Gaussian] PyCOLMAP camera device: "
            + ("CUDA" if use_cuda else "CPU")
        )
        with tempfile.TemporaryDirectory(prefix="vnccs_gaussian_") as temporary:
            work_dir = Path(temporary)
            image_dir = work_dir / "images"
            mask_dir = work_dir / "colmap_masks"
            names, training_frames = _save_frames(
                images,
                indices,
                image_dir,
                mask_dir,
                max_resolution,
                matte_tolerance,
                progress,
                interrupt,
            )
            reconstruction = _solve_cameras(
                pycolmap,
                image_dir,
                mask_dir,
                work_dir,
                names,
                match_window,
                use_cuda,
                progress,
                interrupt,
            )
        print(
            f"[VNCCS Gaussian] Camera stage ready: {len(training_frames)} frames; "
            "ComfyUI will cache this output"
        )
        return ((reconstruction, training_frames),)


class VNCCSGaussianTraining:
    """Train and save Gaussian splats from a cached camera scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": (GAUSSIAN_SCENE_TYPE,),
                "training_steps": (
                    "INT",
                    {
                        "default": DEFAULT_TRAINING_STEPS,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "tooltip": "Training iterations. Any positive count is allowed; 7,000 is practical and 15,000 is a slower final pass.",
                    },
                ),
                "max_splats": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_SPLATS,
                        "min": 100_000,
                        "max": 3_000_000,
                        "step": 100_000,
                        "tooltip": "Growth ceiling for Gaussian splats.",
                    },
                ),
                "sh_degree": (
                    "INT",
                    {
                        "default": DEFAULT_SH_DEGREE,
                        "min": 0,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Spherical harmonics degree. Higher values model more view-dependent color.",
                    },
                ),
                "budgeted_densification": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Grow the strongest candidates gradually without exceeding max_splats.",
                    },
                ),
                "stable_refinement": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use a lower refinement threshold and stop densification halfway through training.",
                    },
                ),
                "antialiased_rasterization": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use gsplat antialiased rasterization for smoother edges and small details.",
                    },
                ),
                "ssim_alpha_loss": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Add composited PyTorch SSIM. Direct alpha supervision stays enabled.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "SPLAT")
    RETURN_NAMES = ("gaussian_file", "splat")
    FUNCTION = "train"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    def train(
        self,
        scene,
        training_steps=DEFAULT_TRAINING_STEPS,
        max_splats=DEFAULT_MAX_SPLATS,
        sh_degree=DEFAULT_SH_DEGREE,
        budgeted_densification=True,
        stable_refinement=True,
        antialiased_rasterization=True,
        ssim_alpha_loss=True,
    ):
        if not isinstance(scene, (tuple, list)) or len(scene) != 2:
            raise ValueError("scene must come from VNCCS Gaussian Camera Reconstruction")
        reconstruction, training_frames = scene
        output_path = _output_path()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        progress, interrupt = _runtime_helpers(training_steps)
        splat = _train_gaussians(
            reconstruction,
            training_frames,
            output_path,
            training_steps,
            max_splats,
            sh_degree,
            budgeted_densification,
            stable_refinement,
            antialiased_rasterization,
            ssim_alpha_loss,
            progress,
            interrupt,
        )
        return output_path, splat


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": VNCCSGaussianCameraReconstruction,
    "VNCCS_GaussianTraining": VNCCSGaussianTraining,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": "VNCCS Gaussian Camera Reconstruction",
    "VNCCS_GaussianTraining": "VNCCS Gaussian Training",
}
