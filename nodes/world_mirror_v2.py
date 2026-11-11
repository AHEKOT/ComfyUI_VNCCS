"""VNCCS WorldMirror V2 loader and image reconstruction.

Adapted from AHEKOT/ComfyUI_HYWorld2 and AHEKOT/ComfyUI_VNCCS_World.
Copyright (c) 2025 MiuProject. Distributed under the MIT license in ../LICENSE.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ._world_mirror_v2_model import depth_to_world_coords_points

_MODEL_TYPE = "VNCCS_WORLDMIRROR_V2_MODEL"
_MODEL_REPO = "tencent/HY-World-2.0"
_MODEL_REVISION = "c662dc2e9dea89f57d12c94eb61b0c8a572c6654"
_PRECISIONS = ("bf16", "convrot8", "convrot4")
_PATCH_SIZE = 14


class VNCCS_WorldMirrorV2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "precision": (list(_PRECISIONS), {
                "default": "bf16",
                "tooltip": "ConvRot8 uses rotated INT8 and ConvRot4 uses rotated INT4 via Comfy Kitchen.",
            }),
        }}

    RETURN_TYPES = (_MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"

    def load_model(self, device="cuda", precision="bf16"):
        from comfy import model_management
        from comfy.model_patcher import ModelPatcher
        from comfy.utils import load_torch_file

        from ._world_mirror_v2_model import (
            _WorldMirror, _load_mirror_state, _native_mirror_ops, _prepare_bf16, _quantize_mirror,
        )

        if precision not in _PRECISIONS:
            raise ValueError(f"Unsupported WorldMirror precision: {precision}.")
        if device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported WorldMirror device: {device}.")
        cpu = torch.device("cpu")
        load_device = model_management.get_torch_device() if device == "cuda" else cpu
        if device == "cuda" and load_device.type != "cuda":
            raise RuntimeError("WorldMirror CUDA mode requires a CUDA device in ComfyUI.")

        config_path, weights_path = _model_files()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with torch.device("meta"):
            model = _WorldMirror(config)
        state = load_torch_file(str(weights_path), safe_load=True, device=cpu)
        _load_mirror_state(model, state)
        del state
        if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
            raise RuntimeError("WorldMirror checkpoint did not initialize every inference tensor.")

        model.eval().requires_grad_(False)
        _prepare_bf16(model)
        if precision != "bf16":
            _quantize_mirror(model, precision, load_device)
        _native_mirror_ops(model)
        return (ModelPatcher(model, load_device=load_device, offload_device=cpu),)


def _model_files():
    import folder_paths

    root = Path(folder_paths.models_dir).resolve()
    directory = (root / "WorldMirror-V2").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("WorldMirror model files must stay inside ComfyUI's model directory.")
    result = []
    for name in ("config.json", "model.safetensors"):
        relative = f"HY-WorldMirror-2.0/{name}"
        path = (directory / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("WorldMirror model files must stay inside ComfyUI's model directory.")
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            print(f"[VNCCS WorldMirror V2] Downloading {relative}", flush=True)
            downloaded = hf_hub_download(repo_id=_MODEL_REPO, revision=_MODEL_REVISION,
                                         filename=relative, local_dir=str(directory), token=False)
            if Path(downloaded).resolve() != path or not path.is_file():
                raise RuntimeError("WorldMirror download returned an unexpected model file.")
        result.append(path)
    return result


def _resize_to_tensor(pil_img, target_size):
    """Resize image so its longest side == target_size (multiple of 14). Returns CHW tensor."""
    orig_w, orig_h = pil_img.size
    if orig_w >= orig_h:
        new_w = target_size
        new_h = max(_PATCH_SIZE, round(orig_h * (new_w / orig_w) / _PATCH_SIZE) * _PATCH_SIZE)
    else:
        new_h = target_size
        new_w = max(_PATCH_SIZE, round(orig_w * (new_h / orig_h) / _PATCH_SIZE) * _PATCH_SIZE)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255
    return tensor, orig_w, orig_h, new_w, new_h


def _adaptive_target_size_from_images(images, max_target_size):
    """Match the official adaptive-size idea for in-memory ComfyUI images."""
    if images is None or images.shape[0] == 0:
        return max_target_size
    heights = images.shape[1]
    widths = images.shape[2]
    longest = max(int(widths), int(heights))
    target = min(max_target_size, longest)
    target = max(_PATCH_SIZE, (target // _PATCH_SIZE) * _PATCH_SIZE)
    return target


def _map_to_comfy_image(value, fallback_shape, normalize=True):
    """Convert [1,S,H,W], [1,S,H,W,1], [S,H,W], or [S,H,W,1] maps to [S,H,W,3]."""
    S, H, W = fallback_shape
    if value is None:
        return torch.zeros(S, H, W, 3, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        t = torch.from_numpy(value)
    else:
        t = value.detach().cpu()
    t = t.float()
    if t.dim() == 5:
        t = t[0]
    if t.dim() == 4 and t.shape[-1] == 1:
        t = t[..., 0]
    elif t.dim() == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        return torch.zeros(S, H, W, 3, dtype=torch.float32)
    if normalize:
        t_min = t.amin(dim=(1, 2), keepdim=True)
        t_max = t.amax(dim=(1, 2), keepdim=True)
        t = (t - t_min) / (t_max - t_min + 1e-8)
    else:
        t = t.clamp(0, 1)
    return t.unsqueeze(-1).repeat(1, 1, 1, 3).float()


def _resize_depth_prior(depth_prior, target_size, expected_count):
    """Resize optional ComfyUI depth IMAGE to [1,S,H,W] for WorldMirror depth conditioning."""
    if depth_prior is None:
        return None
    if depth_prior.shape[0] not in (1, expected_count):
        raise ValueError(
            f"depth_prior must have 1 frame or match images ({expected_count}); got {depth_prior.shape[0]}"
        )
    if depth_prior.shape[0] == 1 and expected_count > 1:
        depth_prior = depth_prior.repeat(expected_count, 1, 1, 1)

    depth_list = []
    for i in range(expected_count):
        depth_np = depth_prior[i].detach().cpu().numpy()
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        depth_np = np.nan_to_num(depth_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if depth_np.max() <= 0:
            depth_np = np.ones_like(depth_np, dtype=np.float32)
        depth_min = depth_np[depth_np > 0].min() if np.any(depth_np > 0) else depth_np.min()
        depth_np = depth_np - depth_min
        if depth_np.max() > 0:
            depth_np = depth_np / depth_np.max()
        orig_h, orig_w = depth_np.shape[:2]
        if orig_w >= orig_h:
            new_w = target_size
            new_h = max(_PATCH_SIZE, round(orig_h * (new_w / orig_w) / _PATCH_SIZE) * _PATCH_SIZE)
        else:
            new_h = target_size
            new_w = max(_PATCH_SIZE, round(orig_w * (new_h / orig_h) / _PATCH_SIZE) * _PATCH_SIZE)
        d = torch.from_numpy(depth_np).float().view(1, 1, orig_h, orig_w)
        d = torch.nn.functional.interpolate(d, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0]
        if new_h > target_size:
            crop = (new_h - target_size) // 2
            d = d[crop:crop + target_size, :]
        depth_list.append(d)
    return torch.stack(depth_list).unsqueeze(0)


def _normalize_camera_poses_to_first(camera_poses):
    """Match official pipeline prior-camera normalization: inv(first_pose) @ pose."""
    if not isinstance(camera_poses, torch.Tensor):
        return camera_poses

    poses = camera_poses
    had_batch = poses.dim() == 4 and poses.shape[0] == 1
    if had_batch:
        work = poses[0]
    elif poses.dim() == 3:
        work = poses
    else:
        return camera_poses

    if work.shape[-2:] == (4, 4):
        work4 = work
        return_3x4 = False
    elif work.shape[-2:] == (3, 4):
        bottom = torch.zeros(
            work.shape[0], 1, 4,
            dtype=work.dtype,
            device=work.device,
        )
        bottom[:, 0, 3] = 1.0
        work4 = torch.cat([work, bottom], dim=1)
        return_3x4 = True
    else:
        return camera_poses

    if work4.shape[0] == 0:
        return camera_poses

    try:
        first = work4[0]
        inv_first = torch.linalg.inv(first.float()).to(dtype=work4.dtype, device=work4.device)
        normalized = inv_first.unsqueeze(0) @ work4
    except Exception as exc:
        print(f"[V2] Official camera-pose normalization skipped: {type(exc).__name__}: {exc}")
        return camera_poses

    if return_3x4:
        normalized = normalized[:, :3, :]
    if had_batch:
        normalized = normalized.unsqueeze(0)
    return normalized


def _extract_pose_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    poses = value.detach().cpu().float()
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 3 and poses.shape[-2:] == (4, 4):
        return poses
    return None


def _apply_mask_to_splats(splats, mask_np):
    if splats is None or mask_np is None:
        return splats
    flat_mask = torch.from_numpy(mask_np.reshape(-1)).bool()

    def _filter_splat_tensor(t):
        mask = flat_mask.to(t.device)
        if t.dim() >= 2 and t.shape[1] == mask.shape[0]:
            return t[:, mask, ...]
        if t.dim() >= 1 and t.shape[0] == mask.shape[0]:
            return t[mask, ...]
        return t

    filtered = {}
    for k, v in splats.items():
        if isinstance(v, list):
            filtered[k] = [
                _filter_splat_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in v
            ]
        elif isinstance(v, torch.Tensor):
            filtered[k] = _filter_splat_tensor(v)
        else:
            filtered[k] = v
    return filtered


def _voxel_prune_splats_dict(splats, voxel_size):
    if splats is None or voxel_size <= 0:
        return splats
    required = ("means", "scales", "quats", "opacities")
    if not isinstance(splats, dict) or any(k not in splats for k in required):
        return splats
    out = dict(splats)
    batches = len(splats["means"]) if isinstance(splats["means"], list) else splats["means"].shape[0]
    pruned = {k: [] for k in ("means", "scales", "quats", "opacities")}
    color_key = "sh" if "sh" in splats else "colors" if "colors" in splats else None
    if color_key is None:
        return splats
    pruned[color_key] = []

    for i in range(batches):
        def get_item(key):
            v = splats[key]
            return v[i] if isinstance(v, list) else v[i]

        means = get_item("means").detach().cpu()
        scales = get_item("scales").detach().cpu()
        quats = get_item("quats").detach().cpu()
        opacities = get_item("opacities").detach().cpu().reshape(-1)
        opacity_max = opacities.max().detach().clone() if opacities.numel() else torch.tensor(1.0)
        if not opacities.numel():
            for key in pruned:
                pruned[key].append(get_item(key).detach().cpu())
            continue
        colors = get_item(color_key).detach().cpu()
        if colors.dim() == 3 and colors.shape[1] == 1:
            colors_for_prune = colors[:, 0, :]
        else:
            colors_for_prune = colors.reshape(colors.shape[0], -1)[:, :3]
        weights = get_item("weights").detach().cpu().reshape(-1) if "weights" in splats else torch.ones_like(opacities)
        if weights.shape[0] != opacities.shape[0]:
            print(
                f"[V2] Ignoring stale splat weights during voxel prune: "
                f"weights={weights.shape[0]}, opacities={opacities.shape[0]}"
            )
            weights = torch.ones_like(opacities)
        means, scales, quats, colors_new, opacities = _voxel_prune_gaussians(
            means, scales, quats, colors_for_prune, opacities, weights, voxel_size=voxel_size
        )
        opacities = opacities.clamp_max(opacity_max)
        pruned["means"].append(means)
        pruned["scales"].append(scales)
        pruned["quats"].append(quats)
        pruned["opacities"].append(opacities)
        if color_key == "sh":
            pruned[color_key].append(colors_new[:, None, :])
        else:
            pruned[color_key].append(colors_new)
    out.update(pruned)
    # Weights are only used as a merge/prune helper. After voxel merging they no
    # longer map 1:1 to the pruned gaussians, so keeping them causes later prune
    # passes to mix tensors with incompatible lengths.
    out.pop("weights", None)
    return out


def _replace_splat_colors_from_images(splats, imgs_tensor):
    if not isinstance(splats, dict) or imgs_tensor is None:
        return splats
    means = splats.get("means")
    if means is None:
        return splats

    B, S, _, H, W = imgs_tensor.shape
    rgb = imgs_tensor.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3)
    sh_dc = (rgb - 0.5) / 0.28209479177387814

    out = dict(splats)
    if isinstance(means, list):
        out["sh"] = [sh_dc[i, :m.shape[0]].to(m.device, dtype=m.dtype)[:, None, :] for i, m in enumerate(means)]
    elif isinstance(means, torch.Tensor):
        out["sh"] = sh_dc[:, :means.shape[1]].to(means.device, dtype=means.dtype)[:, :, None, :]
    else:
        return splats
    return out


def _tune_splats(splats, scale_multiplier=1.0, opacity_floor=0.0):
    if not isinstance(splats, dict):
        return splats
    out = dict(splats)

    def map_value(value, fn):
        if isinstance(value, list):
            return [fn(v) if isinstance(v, torch.Tensor) else v for v in value]
        if isinstance(value, torch.Tensor):
            return fn(value)
        return value

    if scale_multiplier != 1.0 and "scales" in out:
        out["scales"] = map_value(out["scales"], lambda t: t * float(scale_multiplier))
    if opacity_floor > 0.0 and "opacities" in out:
        floor = float(opacity_floor)
        out["opacities"] = map_value(out["opacities"], lambda t: t.clamp_min(floor))
    return out


def _estimate_similarity_transform(src_points, dst_points, eps=1e-8):
    """Estimate dst ~= scale * R * src + t for matched 3D point sets."""
    if src_points.shape != dst_points.shape or src_points.shape[0] < 3:
        return None
    X = src_points.detach().cpu().float()
    Y = dst_points.detach().cpu().float()
    finite = torch.isfinite(X).all(dim=1) & torch.isfinite(Y).all(dim=1)
    X, Y = X[finite], Y[finite]
    if X.shape[0] < 3:
        return None

    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    var_x = (Xc.square().sum(dim=1)).mean()
    if var_x <= eps:
        return None

    cov = (Yc.T @ Xc) / X.shape[0]
    try:
        U, D, Vh = torch.linalg.svd(cov)
    except RuntimeError:
        return None
    S = torch.eye(3, dtype=torch.float32)
    if torch.linalg.det(U @ Vh) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vh
    scale = torch.trace(torch.diag(D) @ S) / var_x
    if not torch.isfinite(scale) or scale <= eps:
        return None
    t = mu_y - scale * (R @ mu_x)
    return {"scale": scale.float(), "rotation": R.float(), "translation": t.float()}


def _compose_similarity(a, b):
    """Compose transforms a(b(x)); each transform is scale, rotation, translation."""
    s = a["scale"] * b["scale"]
    R = a["rotation"] @ b["rotation"]
    t = a["scale"] * (a["rotation"] @ b["translation"]) + a["translation"]
    return {"scale": s.float(), "rotation": R.float(), "translation": t.float()}


def _transform_points_similarity(points, transform):
    R = transform["rotation"].to(points.device, dtype=points.dtype)
    t = transform["translation"].to(points.device, dtype=points.dtype)
    s = transform["scale"].to(points.device, dtype=points.dtype)
    return (points @ R.T) * s + t


def _camera_pose_icp_cloud(poses, intrinsics=None, image_hw=None):
    """Build an ICP point set from camera positions and pose-derived ray samples."""
    p = _extract_pose_tensor(poses)
    if p is None or p.shape[0] < 3:
        return None

    intr = None
    if isinstance(intrinsics, torch.Tensor):
        intr = intrinsics.detach().cpu().float()
        if intr.dim() == 4 and intr.shape[0] == 1:
            intr = intr[0]
        if intr.dim() != 3 or intr.shape[0] != p.shape[0] or intr.shape[-2:] != (3, 3):
            intr = None

    positions = p[:, :3, 3]
    finite = torch.isfinite(positions).all(dim=1)
    positions = positions[finite]
    rotations = p[finite, :3, :3]
    if positions.shape[0] < 3:
        return None

    span = positions.max(dim=0).values - positions.min(dim=0).values
    ray_len = max(float(torch.linalg.norm(span).item()) * 0.05, 1e-3)
    points = [positions]
    for axis in range(3):
        points.append(positions + rotations[:, :, axis] * ray_len)

    if intr is not None:
        intr = intr[finite]
        if image_hw is None:
            h = w = 1.0
        else:
            h, w = float(image_hw[0]), float(image_hw[1])
        uv = torch.tensor(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0], [w * 0.5, h * 0.5]],
            dtype=torch.float32,
        )
        rays = []
        for i in range(intr.shape[0]):
            fx = intr[i, 0, 0].clamp_min(1e-6)
            fy = intr[i, 1, 1].clamp_min(1e-6)
            cx = intr[i, 0, 2]
            cy = intr[i, 1, 2]
            cam = torch.stack([
                (uv[:, 0] - cx) / fx,
                (uv[:, 1] - cy) / fy,
                torch.ones(uv.shape[0], dtype=torch.float32),
            ], dim=1)
            cam = cam / cam.norm(dim=1, keepdim=True).clamp_min(1e-8)
            rays.append(positions[i][None, :] + (cam @ rotations[i].T) * ray_len)
        points.append(torch.cat(rays, dim=0))

    cloud = torch.cat(points, dim=0)
    if cloud.shape[0] < 3 or (cloud.max(dim=0).values - cloud.min(dim=0).values).norm() <= 1e-8:
        return None
    return cloud


def _camera_positions_degenerate(poses, eps=1e-6):
    p = _extract_pose_tensor(poses)
    if p is None or p.shape[0] < 2:
        return True
    positions = p[:, :3, 3]
    finite = torch.isfinite(positions).all(dim=1)
    positions = positions[finite]
    if positions.shape[0] < 2:
        return True
    span = positions.max(dim=0).values - positions.min(dim=0).values
    return float(torch.linalg.norm(span).item()) <= float(eps)


def _icp_similarity(source_points, target_points, max_iterations=40, tolerance=1e-7):
    if source_points is None or target_points is None or source_points.shape[0] < 3 or target_points.shape[0] < 3:
        return None

    source = source_points.detach().cpu().float()
    target = target_points.detach().cpu().float()
    transform = {
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "rotation": torch.eye(3, dtype=torch.float32),
        "translation": torch.zeros(3, dtype=torch.float32),
    }
    initial = None
    if source.shape[0] == target.shape[0]:
        initial = _estimate_similarity_transform(source, target)
    if initial is not None:
        transform = initial
        transformed = _transform_points_similarity(source, transform)
    else:
        transformed = source
    prev_rmse = None
    best_transform = transform
    best_rmse = None

    for _ in range(int(max_iterations)):
        dists = torch.cdist(transformed, target)
        nn_dist, nn_idx = torch.min(dists, dim=1)
        matched = target[nn_idx]
        if nn_dist.numel() >= 8:
            cutoff = torch.quantile(nn_dist, 0.90)
            keep = nn_dist <= cutoff
            if int(keep.sum().item()) >= 3:
                src_fit = transformed[keep]
                dst_fit = matched[keep]
                rmse = torch.sqrt((nn_dist[keep].square()).mean())
            else:
                src_fit, dst_fit = transformed, matched
                rmse = torch.sqrt((nn_dist.square()).mean())
        else:
            src_fit, dst_fit = transformed, matched
            rmse = torch.sqrt((nn_dist.square()).mean())

        delta = _estimate_similarity_transform(src_fit, dst_fit)
        if delta is None:
            return None
        final_rmse = float(rmse.item())
        if best_rmse is None or final_rmse < best_rmse:
            best_rmse = final_rmse
            best_transform = transform
        elif final_rmse > best_rmse + tolerance:
            break
        next_transformed = _transform_points_similarity(transformed, delta)
        next_transform = _compose_similarity(delta, transform)
        next_nn = torch.min(torch.cdist(next_transformed, target), dim=1).values
        next_rmse = float(torch.sqrt(next_nn.square().mean()).item())
        if next_rmse > final_rmse + tolerance:
            break
        transformed = next_transformed
        transform = next_transform
        if prev_rmse is not None and abs(prev_rmse - final_rmse) <= tolerance:
            break
        prev_rmse = final_rmse

    best_transform["rmse"] = float(best_rmse if best_rmse is not None else 0.0)
    return best_transform


def _depth_maps_s_h_w(depth):
    if not isinstance(depth, torch.Tensor):
        return None
    if depth.dim() == 5 and depth.shape[0] == 1 and depth.shape[-1] == 1:
        return depth[0, ..., 0]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.dim() == 4 and depth.shape[0] == 1:
        return depth[0]
    if depth.dim() == 3:
        return depth
    return None


def _backproject_depth_from_input_cameras(depth_maps, input_camera_poses, input_camera_intrinsics):
    poses = _extract_pose_tensor(input_camera_poses)
    if poses is None or not isinstance(input_camera_intrinsics, torch.Tensor):
        return None
    intrs = input_camera_intrinsics.detach().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() != 3 or intrs.shape[-2:] != (3, 3):
        return None
    if poses.shape[0] != depth_maps.shape[0] or intrs.shape[0] != depth_maps.shape[0]:
        return None

    device = depth_maps.device
    depth_f = depth_maps.float()
    poses = poses.to(device=device, dtype=torch.float32)
    intrs = intrs.to(device=device, dtype=torch.float32)
    pts, _, _ = depth_to_world_coords_points(depth_f, poses, intrs)
    return pts


def _map_splat_value(value, fn):
    if isinstance(value, list):
        return [fn(v) if isinstance(v, torch.Tensor) else v for v in value]
    if isinstance(value, torch.Tensor):
        return fn(value)
    return value


def _replace_splat_means_from_points(splats, points):
    if not isinstance(splats, dict) or not isinstance(points, torch.Tensor):
        return splats
    out = dict(splats)
    flat = points.reshape(-1, 3)
    means = out.get("means")
    if isinstance(means, list):
        out["means"] = [
            flat.to(device=m.device, dtype=m.dtype) if isinstance(m, torch.Tensor) else m
            for m in means
        ]
    elif isinstance(means, torch.Tensor):
        repl = flat.to(device=means.device, dtype=means.dtype)
        out["means"] = repl.unsqueeze(0) if means.dim() >= 3 and means.shape[0] == 1 else repl
    else:
        out["means"] = flat.unsqueeze(0)
    return out


def _splat_means_count(splats):
    if not isinstance(splats, dict):
        return None
    means = splats.get("means")
    if isinstance(means, list):
        means = means[0] if means and isinstance(means[0], torch.Tensor) else None
    if not isinstance(means, torch.Tensor):
        return None
    if means.dim() >= 3 and means.shape[0] == 1:
        return int(means.shape[1])
    if means.dim() >= 2:
        return int(means.shape[0])
    return None


def _scale_splat_scales(splats, scale):
    if not isinstance(splats, dict) or "scales" not in splats:
        return splats
    out = dict(splats)
    scale_value = float(abs(scale))
    out["scales"] = _map_splat_value(out["scales"], lambda x: x * scale_value)
    return out


def _stabilize_camera_intrinsics(intrinsics, mode, image_hw=None):
    if not isinstance(intrinsics, torch.Tensor):
        return intrinsics
    intrs = intrinsics.detach().clone().float()
    had_batch = intrs.dim() == 4 and intrs.shape[0] == 1
    if had_batch:
        work = intrs[0]
    elif intrs.dim() == 3:
        work = intrs
    else:
        return intrinsics
    if work.shape[-2:] != (3, 3) or work.shape[0] == 0:
        return intrinsics

    stable = work.clone()
    finite = torch.isfinite(stable).all(dim=(1, 2))
    if not bool(finite.any()):
        return intrinsics
    valid = stable[finite]
    med = valid.median(dim=0).values

    stable[:, 0, 0] = med[0, 0]
    stable[:, 1, 1] = med[1, 1]
    if mode in ("median_focal_center", "median_all"):
        if image_hw is not None:
            h, w = image_hw
            stable[:, 0, 2] = float(w) * 0.5
            stable[:, 1, 2] = float(h) * 0.5
        else:
            stable[:, 0, 2] = med[0, 2]
            stable[:, 1, 2] = med[1, 2]
    elif mode == "median_all":
        stable[:, 0, 2] = med[0, 2]
        stable[:, 1, 2] = med[1, 2]

    return stable.unsqueeze(0) if had_batch else stable


def _auto_stabilize_missing_input_cameras(predictions, mode, image_hw):
    if mode == "off":
        return None, None, None
    pred_poses = predictions.get("camera_poses")
    pred_intrs = predictions.get("camera_intrs")
    poses = _extract_pose_tensor(pred_poses)
    if poses is None or not isinstance(pred_intrs, torch.Tensor):
        print("[V2] Missing-camera stabilization skipped: predicted cameras are unavailable.")
        return None, None, None

    intrs = pred_intrs.detach().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() != 3 or intrs.shape[-2:] != (3, 3) or intrs.shape[0] != poses.shape[0]:
        print("[V2] Missing-camera stabilization skipped: predicted intrinsics are invalid.")
        return None, None, None

    stable_intrs = _stabilize_camera_intrinsics(intrs, "median_focal_center", image_hw=image_hw)
    stable_poses = poses.detach().clone().float()
    info = {
        "source": "predicted_cameras",
        "mode": mode,
        "fx_min": float(intrs[:, 0, 0].min().item()),
        "fx_max": float(intrs[:, 0, 0].max().item()),
        "fx_stable": float(stable_intrs[:, 0, 0].median().item()),
        "fy_min": float(intrs[:, 1, 1].min().item()),
        "fy_max": float(intrs[:, 1, 1].max().item()),
        "fy_stable": float(stable_intrs[:, 1, 1].median().item()),
    }
    print(
        "[V2] Missing-camera stabilization: "
        f"mode={mode}, fx={info['fx_min']:.3f}..{info['fx_max']:.3f}->{info['fx_stable']:.3f}, "
        f"fy={info['fy_min']:.3f}..{info['fy_max']:.3f}->{info['fy_stable']:.3f}"
    )
    return stable_poses, stable_intrs, info


def _project_predictions_from_input_cameras(predictions, input_camera_poses, input_camera_intrinsics, scale_value):
    projected = dict(predictions)
    depth_device = projected["depth"].device if isinstance(projected.get("depth"), torch.Tensor) else torch.device("cpu")
    depth_points = None

    if isinstance(projected.get("depth"), torch.Tensor):
        projected["depth"] = projected["depth"] * float(scale_value)
        depth_maps = _depth_maps_s_h_w(projected["depth"])
        if depth_maps is not None:
            pts = _backproject_depth_from_input_cameras(depth_maps, input_camera_poses, input_camera_intrinsics)
            if pts is not None:
                depth_points = pts
                projected["pts3d"] = pts.unsqueeze(0).to(
                    device=projected["depth"].device,
                    dtype=projected["depth"].dtype,
                )

    if isinstance(projected.get("gs_depth"), torch.Tensor):
        projected["gs_depth"] = projected["gs_depth"] * float(scale_value)
        gs_depth_maps = _depth_maps_s_h_w(projected["gs_depth"])
        if gs_depth_maps is not None and isinstance(projected.get("splats"), dict):
            gs_pts = _backproject_depth_from_input_cameras(gs_depth_maps, input_camera_poses, input_camera_intrinsics)
            if gs_pts is not None:
                projected["splats"] = _replace_splat_means_from_points(projected["splats"], gs_pts)
                projected["splats"] = _scale_splat_scales(projected["splats"], scale_value)
                print(
                    "[V2] Input-camera projection: replaced splat means from gs_depth "
                    f"({int(gs_pts.reshape(-1, 3).shape[0])} points)."
                )
    elif isinstance(projected.get("splats"), dict) and isinstance(depth_points, torch.Tensor):
        splat_count = _splat_means_count(projected["splats"])
        depth_count = int(depth_points.reshape(-1, 3).shape[0])
        if splat_count == depth_count:
            projected["splats"] = _replace_splat_means_from_points(projected["splats"], depth_points)
            projected["splats"] = _scale_splat_scales(projected["splats"], scale_value)
            print(
                "[V2] Input-camera projection: gs_depth missing; replaced splat means "
                f"from regular depth grid ({depth_count} points)."
            )
        else:
            print(
                "[V2] Input-camera projection: gs_depth missing; kept model splat means "
                f"because splat_count={splat_count} and depth_count={depth_count} do not match."
            )

    if isinstance(input_camera_poses, torch.Tensor):
        projected["camera_poses"] = input_camera_poses.unsqueeze(0).to(
            device=depth_device,
            dtype=torch.float32,
        )
    if isinstance(input_camera_intrinsics, torch.Tensor):
        projected["camera_intrs"] = input_camera_intrinsics.unsqueeze(0).to(
            device=depth_device,
            dtype=torch.float32,
        )
    return projected


def _reproject_predictions_from_input_cameras_with_icp_scale(
    predictions,
    input_camera_poses,
    input_camera_intrinsics,
    image_hw,
):
    if _camera_positions_degenerate(input_camera_poses):
        print(
            "[V2] Input-camera projection: input camera positions are co-located, "
            "so ICP scale is not observable. Using scale=1.0 and backprojecting depths "
            "through input rotations/intrinsics."
        )
        projected = _project_predictions_from_input_cameras(
            predictions,
            input_camera_poses,
            input_camera_intrinsics,
            scale_value=1.0,
        )
        return projected, {
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "rotation": torch.eye(3, dtype=torch.float32),
            "translation": torch.zeros(3, dtype=torch.float32),
            "rmse": None,
            "degenerate_input_camera_positions": True,
        }

    pred_cloud = _camera_pose_icp_cloud(
        predictions.get("camera_poses"),
        predictions.get("camera_intrs"),
        image_hw=image_hw,
    )
    input_cloud = _camera_pose_icp_cloud(
        input_camera_poses,
        input_camera_intrinsics,
        image_hw=image_hw,
    )
    transform = _icp_similarity(pred_cloud, input_cloud)
    if transform is None:
        print("[V2] Input-camera ICP depth-scale projection skipped: insufficient or degenerate camera pose point sets.")
        return predictions, None

    scale_value = float(abs(transform["scale"].item()))
    projected = _project_predictions_from_input_cameras(
        predictions,
        input_camera_poses,
        input_camera_intrinsics,
        scale_value=scale_value,
    )

    print(
        "[V2] ICP-scaled depths and reprojected from input cameras: "
        f"scale={scale_value:.6f}, rmse={transform['rmse']:.6f}, "
        f"source_points={pred_cloud.shape[0]}, target_points={input_cloud.shape[0]}"
    )
    return projected, transform


def _log_splat_stats(splats):
    if not isinstance(splats, dict):
        print("[V2 DEBUG] splats stats: none")
        return

    def first_tensor(key):
        value = splats.get(key)
        if isinstance(value, list):
            return value[0] if value and isinstance(value[0], torch.Tensor) else None
        return value if isinstance(value, torch.Tensor) else None

    for key in ("scales", "opacities", "weights"):
        value = first_tensor(key)
        if value is not None:
            print("[V2 DEBUG] splats." + _tensor_stats_line(key, value))


def _set_worldmirror_head_frame_chunk_size(model, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for name in ("depth_head", "pts_head", "norm_head", "gs_head"):
        head = getattr(model, name, None)
        if head is not None:
            setattr(head, "frames_chunk_size", chunk_size)


def _set_worldmirror_gs_param_chunk_size(model, chunk_size):
    chunk_size = max(1, int(chunk_size))
    gs_renderer = getattr(model, "gs_renderer", None)
    if gs_renderer is not None:
        setattr(gs_renderer, "gs_param_chunk_size", chunk_size)


def _set_worldmirror_transformer_mlp_chunk_size(model, chunk_size):
    chunk_size = max(0, int(chunk_size))
    updated = 0
    for module in model.modules():
        if hasattr(module, "inference_chunk_size") and hasattr(module, "fc1") and hasattr(module, "fc2"):
            setattr(module, "inference_chunk_size", chunk_size)
            updated += 1
    if chunk_size > 0:
        print(f"[V2] Transformer MLP chunking enabled: {updated} MLPs, chunk_size={chunk_size}")


def _apply_worldmirror_head_compute_mode(model, mode, use_gsplat):
    original = {
        "enable_pts": getattr(model, "enable_pts", None),
        "enable_norm": getattr(model, "enable_norm", None),
        "enable_gs": getattr(model, "enable_gs", None),
    }
    if mode == "depth+gs":
        model.enable_pts = False
        model.enable_norm = False
        model.enable_gs = bool(use_gsplat and original["enable_gs"])
    elif mode == "depth_only":
        model.enable_pts = False
        model.enable_norm = False
        model.enable_gs = False
    else:
        model.enable_gs = bool(use_gsplat and original["enable_gs"])
    return original


def _restore_worldmirror_head_compute_mode(model, original):
    for key, value in original.items():
        if value is not None:
            setattr(model, key, value)


def _camera_debug_angles(c2w):
    """Return approximate yaw/pitch/roll in degrees from a c2w matrix."""
    R = c2w[:3, :3].float()
    forward = R @ torch.tensor([0.0, 0.0, 1.0], device=R.device)
    up = R @ torch.tensor([0.0, -1.0, 0.0], device=R.device)
    yaw = torch.atan2(forward[0], forward[2]) * 180.0 / torch.pi
    pitch = torch.asin(torch.clamp(-forward[1] / forward.norm().clamp_min(1e-8), -1.0, 1.0)) * 180.0 / torch.pi
    roll = torch.atan2(up[0], -up[1]) * 180.0 / torch.pi
    return yaw.item(), pitch.item(), roll.item()


def _angle_delta_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def _tensor_stats_line(name, value):
    if value is None:
        return f"{name}=none"
    t = value.detach().cpu().float()
    if t.numel() == 0:
        return f"{name}=empty"
    flat = t.flatten()
    sample_note = ""
    max_stats_values = 1_000_000
    if flat.numel() > max_stats_values:
        step = max(1, flat.numel() // max_stats_values)
        flat = flat[::step][:max_stats_values]
        sample_note = f", sampled={flat.numel()}/{t.numel()}"
    return (
        f"{name}: min={flat.min().item():.4f}, p10={torch.quantile(flat, 0.10).item():.4f}, "
        f"median={flat.median().item():.4f}, p90={torch.quantile(flat, 0.90).item():.4f}, "
        f"max={flat.max().item():.4f}{sample_note}"
    )


def _log_camera_table(title, poses, intrs, H, W):
    if poses is None and intrs is None:
        print(f"[V2 DEBUG] {title}: none")
        return
    poses_cpu = poses.detach().cpu().float() if poses is not None else None
    intrs_cpu = intrs.detach().cpu().float() if intrs is not None else None
    if poses_cpu is not None and poses_cpu.dim() == 4:
        poses_cpu = poses_cpu[0]
    if intrs_cpu is not None and intrs_cpu.dim() == 4:
        intrs_cpu = intrs_cpu[0]
    count = poses_cpu.shape[0] if poses_cpu is not None else intrs_cpu.shape[0]
    print(f"[V2 DEBUG] {title}: {count} cameras")
    for i in range(count):
        if poses_cpu is not None:
            yaw, pitch, roll = _camera_debug_angles(poses_cpu[i])
            t = poses_cpu[i, :3, 3]
            pose_part = (
                f"yaw={yaw:8.3f} pitch={pitch:8.3f} roll={roll:8.3f} "
                f"t=({t[0].item(): .4f},{t[1].item(): .4f},{t[2].item(): .4f})"
            )
        else:
            pose_part = "pose=none"
        if intrs_cpu is not None:
            fx = intrs_cpu[i, 0, 0].item()
            fy = intrs_cpu[i, 1, 1].item()
            cx = intrs_cpu[i, 0, 2].item()
            cy = intrs_cpu[i, 1, 2].item()
            fov_x = 2.0 * np.degrees(np.arctan(W * 0.5 / max(fx, 1e-8)))
            fov_y = 2.0 * np.degrees(np.arctan(H * 0.5 / max(fy, 1e-8)))
            intr_part = (
                f"fx={fx:8.3f} fy={fy:8.3f} cx={cx:7.2f} cy={cy:7.2f} "
                f"fov=({fov_x:6.2f},{fov_y:6.2f})"
            )
        else:
            intr_part = "intr=none"
        print(f"[V2 DEBUG]   {i:02d}: {pose_part} | {intr_part}")


def _log_per_view_points(name, points, S, H, W):
    if points is None:
        print(f"[V2 DEBUG] {name}: none")
        return
    pts = points.detach().cpu().float()
    if pts.dim() == 5:
        pts = pts[0].reshape(S, H * W, 3)
    elif pts.dim() == 3 and pts.shape[1] == S * H * W:
        pts = pts[0].reshape(S, H * W, 3)
    elif pts.dim() == 2 and pts.shape[0] == S * H * W:
        pts = pts.reshape(S, H * W, 3)
    else:
        print(f"[V2 DEBUG] {name}: unsupported shape={tuple(pts.shape)}")
        return
    print(f"[V2 DEBUG] {name}: per-view world bounds")
    for i in range(S):
        p = pts[i]
        center = p.mean(dim=0)
        pmin = p.amin(dim=0)
        pmax = p.amax(dim=0)
        print(
            f"[V2 DEBUG]   {i:02d}: center=({center[0]: .4f},{center[1]: .4f},{center[2]: .4f}) "
            f"min=({pmin[0]: .4f},{pmin[1]: .4f},{pmin[2]: .4f}) "
            f"max=({pmax[0]: .4f},{pmax[1]: .4f},{pmax[2]: .4f})"
        )


def _log_mask_coverage(name, mask):
    if mask is None:
        print(f"[V2 DEBUG] {name}: none")
        return
    m = np.asarray(mask).astype(bool)
    if m.ndim == 3:
        total_kept = int(m.sum())
        total = int(m.size)
        print(f"[V2 DEBUG] {name}: kept {total_kept}/{total} ({100.0 * total_kept / max(total, 1):.2f}%)")
        for i in range(m.shape[0]):
            kept = int(m[i].sum())
            count = int(m[i].size)
            print(f"[V2 DEBUG]   {i:02d}: kept {kept}/{count} ({100.0 * kept / max(count, 1):.2f}%)")
    else:
        kept = int(m.sum())
        count = int(m.size)
        print(f"[V2 DEBUG] {name}: kept {kept}/{count} ({100.0 * kept / max(count, 1):.2f}%)")


def _log_worldmirror_debug(predictions, views, camera_poses, camera_intrinsics, imgs_tensor):
    S, _, H, W = imgs_tensor.shape[1:]
    print("[V2 DEBUG] ================= WorldMirror V2 Debug =================")
    print(f"[V2 DEBUG] images: S={S}, H={H}, W={W}")
    _log_camera_table("input camera priors", camera_poses, camera_intrinsics, H, W)
    _log_camera_table("model predicted cameras", predictions.get("camera_poses"), predictions.get("camera_intrs"), H, W)
    if camera_poses is not None and predictions.get("camera_poses") is not None:
        pred = predictions["camera_poses"].detach().cpu().float()[0]
        inp = camera_poses.detach().cpu().float()
        print("[V2 DEBUG] predicted - input angular deltas")
        yaw_deltas = []
        roll_deltas = []
        for i in range(min(inp.shape[0], pred.shape[0])):
            iy, ip, ir = _camera_debug_angles(inp[i])
            py, pp, pr = _camera_debug_angles(pred[i])
            dyaw = _angle_delta_deg(py, iy)
            droll = _angle_delta_deg(pr, ir)
            yaw_deltas.append(dyaw)
            roll_deltas.append(droll)
            print(f"[V2 DEBUG]   {i:02d}: dyaw={dyaw: .3f}, dpitch={pp - ip: .3f}, droll={droll: .3f}")
        if roll_deltas:
            mean_abs_roll = sum(abs(v) for v in roll_deltas) / len(roll_deltas)
            mean_abs_yaw = sum(abs(v) for v in yaw_deltas) / len(yaw_deltas)
            if mean_abs_roll > 120.0 and mean_abs_yaw > 120.0:
                print(
                    "[V2 DEBUG] warning: input camera priors look axis-flipped relative to model predictions; "
                    "try pose_convention=opencv_c2w before tuning offsets."
                )
    print("[V2 DEBUG] " + _tensor_stats_line("depth", predictions.get("depth")))
    print("[V2 DEBUG] " + _tensor_stats_line("gs_depth", predictions.get("gs_depth")))
    print("[V2 DEBUG] " + _tensor_stats_line("depth_conf", predictions.get("depth_conf")))
    print("[V2 DEBUG] " + _tensor_stats_line("pts3d_conf", predictions.get("pts3d_conf")))
    print("[V2 DEBUG] " + _tensor_stats_line("gs_depth_conf", predictions.get("gs_depth_conf")))
    _log_per_view_points("pts3d", predictions.get("pts3d"), S, H, W)
    splats = predictions.get("splats")
    if isinstance(splats, dict):
        _log_per_view_points("splats.means", splats.get("means"), S, H, W)
    print("[V2 DEBUG] =========================================================")


def _compute_sky_mask_from_model(predictions, H, W, S, threshold=0.5):
    """Build sky mask from model predictions. Returns [S,H,W] bool or None."""
    for key in ("gs_depth_mask_logits", "gs_depth_mask", "depth_mask_logits", "depth_mask"):
        if key in predictions:
            prob = predictions[key].sigmoid() if "logits" in key else predictions[key]
            dm = prob[0].detach().cpu()
            if dm.dim() == 4 and dm.shape[-1] == 1:
                dm = dm.squeeze(-1)
            if dm.dim() != 3 or dm.shape[0] != S:
                return None
            mask = (dm > threshold).numpy().astype(bool)
            if mask.shape[1] != H or mask.shape[2] != W:
                mask = np.stack([cv2.resize(mask[i].astype(np.uint8), (W, H),
                                            interpolation=cv2.INTER_NEAREST) > 0
                                 for i in range(S)], axis=0)
            return mask
    return None


def _compute_filter_masks(predictions, sky_mask, confidence_filter, edge_filter,
                          percentile, normal_threshold, depth_threshold):
    depth = predictions["depth"][0, ..., 0].detach().float().cpu()
    valid = torch.ones_like(depth, dtype=torch.bool)
    if confidence_filter:
        confidence = predictions.get("pts3d_conf", predictions["depth_conf"])[0].detach().float().cpu()
        thresholds = torch.quantile(confidence.flatten(1), percentile / 100, dim=1)
        valid &= confidence >= thresholds[:, None, None]
    gs_depth = predictions.get("gs_depth")
    gs_valid = valid.clone() if gs_depth is not None else None

    if edge_filter and "normals" in predictions:
        normals = F.normalize(predictions["normals"][0].detach().float().cpu(), dim=-1)
        for i, normal in enumerate(normals):
            neighbors = F.unfold(F.pad(normal.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate"), 3)
            neighbors = neighbors.reshape(3, 9, *depth.shape[-2:])
            angles = (normal.permute(2, 0, 1)[:, None] * neighbors).sum(0).clamp(-1, 1).acos()
            if confidence_filter:
                keep = F.pad(valid[i][None, None].float(), (1, 1, 1, 1), mode="replicate")
                angles = angles.masked_fill(~F.unfold(keep, 3).reshape_as(angles).bool(), 0)
            normal_edge = F.max_pool2d(angles.amax(0)[None], 3, stride=1, padding=1)[0]
            normal_edge = normal_edge > np.deg2rad(normal_threshold)
            if gs_valid is not None:
                gs_valid[i] &= ~(normal_edge & _depth_edges(gs_depth[0, i, ..., 0].detach().float().cpu(),
                                                             valid[i], depth_threshold))
            valid[i] &= ~(normal_edge & _depth_edges(depth[i], valid[i], depth_threshold))

    if sky_mask is not None:
        valid &= torch.from_numpy(sky_mask)
        if gs_valid is not None:
            gs_valid &= torch.from_numpy(sky_mask)
    return valid.numpy(), gs_valid.numpy() if gs_valid is not None else None


def _depth_edges(depth, valid, threshold):
    maximum = F.max_pool2d(depth.masked_fill(~valid, -torch.inf)[None], 3, stride=1, padding=1)[0]
    minimum = -F.max_pool2d((-depth).masked_fill(~valid, -torch.inf)[None], 3, stride=1, padding=1)[0]
    return (maximum - minimum) / depth > threshold


def _camera_prior(tensor, count, matrix_shapes, name):
    if tensor is None:
        return None
    if torch.is_tensor(tensor) and tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if (not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[0] != count
            or tensor.shape[-2:] not in matrix_shapes or not torch.isfinite(tensor).all()):
        raise ValueError(f"WorldMirror requires one finite camera {name} matrix per image ({count}).")
    tensor = tensor.detach().float()
    if tensor.shape[-2:] == (3, 4):
        bottom = tensor.new_zeros((count, 1, 4))
        bottom[..., 3] = 1
        tensor = torch.cat((tensor, bottom), dim=1)
    return tensor


def _voxel_prune_gaussians(means, scales, quats, colors, opacities, weights, voxel_size=0.002):
    """Voxel-based merging of Gaussian splats via weighted average."""
    N = means.shape[0]
    if N == 0:
        return means, scales, quats, colors, opacities

    voxel_idx = (means / voxel_size).floor().long()
    voxel_idx = voxel_idx - voxel_idx.min(dim=0)[0]
    unique, inv = torch.unique(voxel_idx, dim=0, return_inverse=True)
    K = len(unique)
    if K == N:
        return means, scales, quats, colors, opacities

    w = weights
    wsum = torch.zeros(K, dtype=w.dtype).scatter_add_(0, inv, w).clamp(min=1e-8)

    def _wavg(vals):
        out = torch.zeros(K, *vals.shape[1:], dtype=vals.dtype)
        for d in range(vals.shape[1]):
            out[:, d].scatter_add_(0, inv, vals[:, d] * w)
        return out / wsum.unsqueeze(-1)

    m_opa = torch.zeros(K, dtype=opacities.dtype).scatter_add_(0, inv, w * w) / wsum
    m_quats = torch.zeros(K, 4, dtype=quats.dtype)
    for d in range(4):
        m_quats[:, d].scatter_add_(0, inv, quats[:, d] * w)
    m_quats = m_quats / m_quats.norm(dim=1, keepdim=True).clamp(min=1e-8)

    print(f"[Save] Voxel prune: {N} -> {K} gaussians")
    return _wavg(means), _wavg(scales), m_quats, _wavg(colors), m_opa


def _rescale_input_pose_translation_to_prediction(input_poses, predicted_poses, eps=1e-6):
    poses = _extract_pose_tensor(input_poses)
    pred = _extract_pose_tensor(predicted_poses)
    if poses is None or pred is None or poses.shape[0] != pred.shape[0]:
        return poses if poses is not None else input_poses

    out = poses.clone()
    input_t = poses[:, :3, 3]
    pred_t = pred[:, :3, 3]
    input_norm = input_t.norm(dim=1)

    starts = [0]
    for i in range(1, poses.shape[0]):
        if input_norm[i] <= eps:
            starts.append(i)
    starts = sorted(set(starts))

    changed = 0
    ratios = []
    for start_idx, end_idx in zip(starts, starts[1:] + [poses.shape[0]]):
        if end_idx - start_idx <= 1:
            continue
        base_input = input_t[start_idx]
        base_pred = pred_t[start_idx]
        rel_input = input_t[start_idx:end_idx] - base_input
        rel_pred = pred_t[start_idx:end_idx] - base_pred
        rel_input_norm = rel_input.norm(dim=1)
        rel_pred_norm = rel_pred.norm(dim=1)
        valid = rel_input_norm > eps
        if not bool(valid.any()):
            continue
        scale = torch.ones_like(rel_input_norm)
        scale[valid] = rel_pred_norm[valid] / rel_input_norm[valid].clamp_min(eps)
        out[start_idx:end_idx, :3, 3] = base_input + rel_input * scale[:, None]
        changed += int(valid.sum().item())
        ratios.extend(scale[valid].tolist())

    if changed > 0 and ratios:
        ratios_t = torch.tensor(ratios, dtype=torch.float32)
        print(
            "[WorldMirror V2] Calibrated input camera translation scale from predicted motion: "
            f"frames={changed}, median={ratios_t.median().item():.4f}, "
            f"min={ratios_t.min().item():.4f}, max={ratios_t.max().item():.4f}"
        )
    return out


def _suppress_generated_surfaces_seen_by_anchors(
    pts,
    depth_hi,
    poses,
    intrs,
    anchor_frames,
    abs_tol=0.035,
    rel_tol=0.035,
):
    if pts.dim() != 4 or depth_hi.dim() != 3 or poses.shape[0] != pts.shape[0]:
        return torch.ones(depth_hi.shape, dtype=torch.bool)

    S, H, W, _ = pts.shape
    if anchor_frames.shape[0] != S or not bool(anchor_frames.any()) or not bool((~anchor_frames).any()):
        return torch.ones((S, H, W), dtype=torch.bool)

    keep = torch.ones((S, H, W), dtype=torch.bool)
    anchor_ids = torch.where(anchor_frames)[0]
    generated_ids = torch.where(~anchor_frames)[0]
    total_generated = int(generated_ids.numel() * H * W)
    suppressed = 0

    for gen_id_t in generated_ids:
        gen_id = int(gen_id_t.item())
        points = pts[gen_id].reshape(-1, 3).float()
        covered = torch.zeros(points.shape[0], dtype=torch.bool)

        for anchor_id_t in anchor_ids:
            anchor_id = int(anchor_id_t.item())
            R = poses[anchor_id, :3, :3].float()
            t = poses[anchor_id, :3, 3].float()
            cam = (points - t) @ R
            z = cam[:, 2]
            valid = z > 1e-6
            if not bool(valid.any()):
                continue

            K = intrs[anchor_id].float()
            u = torch.round(cam[:, 0] * K[0, 0] / z.clamp_min(1e-6) + K[0, 2]).long()
            v = torch.round(cam[:, 1] * K[1, 1] / z.clamp_min(1e-6) + K[1, 2]).long()
            valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not bool(valid.any()):
                continue

            valid_ids = torch.where(valid)[0]
            anchor_depth = depth_hi[anchor_id, v[valid_ids], u[valid_ids]].float()
            tol = float(abs_tol) + float(rel_tol) * anchor_depth.abs()
            same_or_in_front = z[valid_ids] <= anchor_depth + tol
            covered[valid_ids[same_or_in_front]] = True

        frame_keep = ~covered.reshape(H, W)
        keep[gen_id] = frame_keep
        suppressed += int(covered.sum().item())

    if suppressed > 0:
        print(
            "[WorldMirror V2] Suppressed low-res stereo splats already covered by high-res anchors: "
            f"{suppressed}/{total_generated} ({100.0 * suppressed / max(total_generated, 1):.2f}%)"
        )
    return keep


class _WorldMirrorV2Inference:
    def run_inference(
        self,
        model,
        images,
        use_gsplat          = True,
        target_size         = 952,
        offload_scheme      = "none",
        low_vram_mode = False,
        head_frame_chunk_size = 2,
        head_compute_mode = "all",
        gs_param_chunk_size = 1,
        transformer_mlp_chunk_size = 0,
        confidence_percentile = 10.0,
        apply_sky_mask      = False,
        filter_edges        = True,
        filter_splats       = False,
        edge_normal_threshold = 1.0,
        edge_depth_threshold  = 0.03,
        apply_confidence_mask = False,
        camera_conditioning = "pose+intrinsics",
        normalize_camera_poses_to_first = False,
        missing_camera_strategy = "off",
        splat_camera_source = "camera_inputs",
        splat_color_source = "input_image",
        adaptive_target_size = False,
        apply_model_masks = False,
        model_mask_threshold = 0.5,
        voxel_prune_splats = True,
        voxel_size = 0.002,
        splat_scale_multiplier = 1.0,
        splat_opacity_floor = 0.0,
        debug_log = False,
        camera_intrinsics   = None,
        camera_poses        = None,
        depth_prior         = None,
    ):

        from comfy import model_management

        if (not torch.is_tensor(images) or images.ndim != 4 or min(images.shape[:3]) == 0
                or images.shape[-1] not in (3, 4) or not images.is_floating_point()
                or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
            raise ValueError("WorldMirror expects a finite RGB or RGBA IMAGE batch between zero and one.")
        if type(target_size) is not int or not 252 <= target_size <= 4096:
            raise ValueError("WorldMirror target_size must be between 252 and 4096.")
        if offload_scheme not in ("none", "model_cpu_offload"):
            raise ValueError(f"Unsupported WorldMirror offload scheme: {offload_scheme}.")
        if low_vram_mode:
            head_frame_chunk_size = 1
            if head_compute_mode == "all":
                head_compute_mode = "depth_only"
            gs_param_chunk_size = 1
            transformer_mlp_chunk_size = 8192
        target_size    = (target_size // _PATCH_SIZE) * _PATCH_SIZE
        if adaptive_target_size:
            target_size = _adaptive_target_size_from_images(images, target_size)
        worldmirror = model.model
        exec_dev = model.load_device
        B = images.shape[0]
        camera_poses = _camera_prior(camera_poses, B, ((3, 4), (4, 4)), "poses")
        camera_intrinsics = _camera_prior(camera_intrinsics, B, ((3, 3),), "intrinsics")
        if normalize_camera_poses_to_first:
            camera_poses = _normalize_camera_poses_to_first(camera_poses)
        tensor_list = []
        adjusted_intrinsics = camera_intrinsics.clone().float() if isinstance(camera_intrinsics, torch.Tensor) else None

        for i in range(B):
            img_np  = (images[i].detach().float().cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, target_size)

            if adjusted_intrinsics is not None:
                sx, sy = new_w / orig_w, new_h / orig_h
                adjusted_intrinsics[i, 0, 0] *= sx
                adjusted_intrinsics[i, 1, 1] *= sy
                adjusted_intrinsics[i, 0, 2] *= sx
                adjusted_intrinsics[i, 1, 2] *= sy

            # centre-crop height if it exceeds target_size
            if new_h > target_size:
                crop = (new_h - target_size) // 2
                t = t[:, crop:crop + target_size, :]
                if adjusted_intrinsics is not None:
                    adjusted_intrinsics[i, 1, 2] -= crop

            # centre-crop width if a future resize strategy ever exceeds target_size
            if new_w > target_size:
                crop = (new_w - target_size) // 2
                t = t[:, :, crop:crop + target_size]
                if adjusted_intrinsics is not None:
                    adjusted_intrinsics[i, 0, 2] -= crop

            tensor_list.append(t)

        imgs_tensor = torch.stack(tensor_list).unsqueeze(0).to(exec_dev)  # [1,S,3,H,W]
        camera_intrinsics = adjusted_intrinsics

        # 2. Build views dict + cond_flags
        views      = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]  # [pose, depth, intrinsics]

        use_pose_prior = camera_conditioning in ("pose+intrinsics", "pose_only")
        use_intrinsics_prior = camera_conditioning in ("pose+intrinsics", "intrinsics_only")
        has_input_cameras = camera_poses is not None and camera_intrinsics is not None
        use_input_camera_source = splat_camera_source == "camera_inputs" and has_input_cameras

        if (use_pose_prior or use_input_camera_source) and camera_poses is not None:
            views["camera_poses"] = camera_poses.unsqueeze(0).to(exec_dev)
        if use_pose_prior and camera_poses is not None:
            cond_flags[0] = 1
        if (use_intrinsics_prior or use_input_camera_source) and camera_intrinsics is not None:
            views["camera_intrs"] = camera_intrinsics.unsqueeze(0).to(exec_dev)
        if use_intrinsics_prior and camera_intrinsics is not None:
            cond_flags[2] = 1
        if depth_prior is not None:
            depth_tensor = _resize_depth_prior(depth_prior, target_size, B).to(exec_dev)
            views["depthmap"] = depth_tensor
            cond_flags[1] = 1

        _set_worldmirror_head_frame_chunk_size(worldmirror, head_frame_chunk_size)
        _set_worldmirror_gs_param_chunk_size(worldmirror, gs_param_chunk_size)
        _set_worldmirror_transformer_mlp_chunk_size(worldmirror, transformer_mlp_chunk_size)
        original_heads = _apply_worldmirror_head_compute_mode(worldmirror, head_compute_mode, use_gsplat)
        cpu_offload = offload_scheme == "model_cpu_offload" and exec_dev.type == "cuda"
        stream_to_cpu = cpu_offload and low_vram_mode and head_compute_mode == "depth+gs"
        if stream_to_cpu:
            views["stream_to_cpu"] = True
        try:
            model_management.throw_exception_if_processing_interrupted()
            model_management.load_models_gpu(
                [model], memory_required=imgs_tensor.numel() * 96, force_full_load=not cpu_offload,
            )
            print(f"[VNCCS WorldMirror V2] Reconstructing {B} images at {target_size}px.")
            with torch.no_grad():
                predictions = worldmirror(views, cond_flags=cond_flags)
        finally:
            _restore_worldmirror_head_compute_mode(worldmirror, original_heads)
            if cpu_offload:
                model_management.unload_model_and_clones(model, unload_additional_models=False)

        if stream_to_cpu:
            del views
            imgs_tensor = imgs_tensor.cpu()
        S, H, W = predictions["depth"].shape[1:4]

        model_camera_poses = predictions.get("camera_poses")
        model_camera_intrs = predictions.get("camera_intrs")
        model_pts3d = predictions.get("pts3d")
        model_pts3d_filtered = None
        if isinstance(model_camera_poses, torch.Tensor):
            model_camera_poses = model_camera_poses.detach().clone()
        if isinstance(model_camera_intrs, torch.Tensor):
            model_camera_intrs = model_camera_intrs.detach().clone()
        if isinstance(model_pts3d, torch.Tensor):
            model_pts3d = model_pts3d.detach().clone()

        input_cameras_were_missing = camera_poses is None and camera_intrinsics is None
        missing_camera_info = None
        if input_cameras_were_missing and missing_camera_strategy != "off":
            auto_poses, auto_intrs, missing_camera_info = _auto_stabilize_missing_input_cameras(
                predictions,
                missing_camera_strategy,
                image_hw=(H, W),
            )
            if auto_poses is not None and auto_intrs is not None:
                camera_poses = auto_poses
                camera_intrinsics = auto_intrs
                if missing_camera_strategy == "stabilize_predicted_intrinsics":
                    predictions = dict(predictions)
                    predictions["camera_poses"] = auto_poses.unsqueeze(0).to(
                        device=predictions["depth"].device,
                        dtype=torch.float32,
                    )
                    predictions["camera_intrs"] = auto_intrs.unsqueeze(0).to(
                        device=predictions["depth"].device,
                        dtype=torch.float32,
                    )
                elif missing_camera_strategy == "reproject_stabilized_predicted":
                    predictions = _project_predictions_from_input_cameras(
                        predictions,
                        auto_poses,
                        auto_intrs,
                        scale_value=1.0,
                    )

        if debug_log:
            _log_worldmirror_debug(predictions, {"img": imgs_tensor} if stream_to_cpu else views,
                                   camera_poses, camera_intrinsics, imgs_tensor)

        input_camera_icp = None
        if use_input_camera_source:
            predictions, input_camera_icp = _reproject_predictions_from_input_cameras_with_icp_scale(
                predictions,
                camera_poses,
                camera_intrinsics,
                image_hw=(H, W),
            )

        sky_mask = _compute_sky_mask_from_model(predictions, H, W, S) if apply_sky_mask else None
        pts_mask, gs_mask = _compute_filter_masks(
            predictions, sky_mask, apply_confidence_mask, filter_edges,
            confidence_percentile, edge_normal_threshold, edge_depth_threshold,
        )

        if apply_model_masks:
            def _native_mask(key):
                value = predictions.get(key)
                if value is None:
                    return None
                t = value[0].detach().cpu().float()
                if t.dim() == 4 and t.shape[-1] == 1:
                    t = t[..., 0]
                return (t.numpy() >= model_mask_threshold)

            depth_model_mask = _native_mask("depth_mask")
            gs_depth_model_mask = _native_mask("gs_depth_mask")
            if depth_model_mask is not None:
                pts_mask = depth_model_mask if pts_mask is None else (pts_mask & depth_model_mask)
            if gs_depth_model_mask is not None:
                gs_mask = gs_depth_model_mask if gs_mask is None else (gs_mask & gs_depth_model_mask)
            elif depth_model_mask is not None and gs_mask is not None:
                gs_mask = gs_mask & depth_model_mask

        if debug_log:
            _log_mask_coverage("pts_filter_mask", pts_mask)
            _log_mask_coverage("gs_filter_mask", gs_mask)
            if not filter_splats:
                print("[V2 DEBUG] gs_filter_mask is diagnostic only; filter_splats=false, so splats are not mask-filtered.")

        # 7. Filter pts3d
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            if pts_mask is not None:
                flat = torch.from_numpy(pts_mask.reshape(-1)).to(pts.device)
                filtered_pts = pts[flat]
            else:
                filtered_pts = pts
        if isinstance(model_pts3d, torch.Tensor):
            model_pts3d_flat = model_pts3d[0].reshape(-1, 3)
            if pts_mask is not None and int(np.prod(pts_mask.shape)) == int(model_pts3d_flat.shape[0]):
                flat = torch.from_numpy(pts_mask.reshape(-1)).to(model_pts3d_flat.device)
                model_pts3d_filtered = model_pts3d_flat[flat]
            else:
                model_pts3d_filtered = model_pts3d_flat

        # 8. Filter splats with GS-specific mask
        splats = predictions.get("splats")
        if splat_color_source == "input_image":
            splats = _replace_splat_colors_from_images(splats, imgs_tensor)
        splats = _tune_splats(splats, splat_scale_multiplier, splat_opacity_floor)
        if debug_log:
            _log_splat_stats(splats)
            if input_camera_icp is not None:
                print("[V2 DEBUG] splats stats after input-camera ICP processing")
                _log_splat_stats(splats)
                if isinstance(splats, dict):
                    _log_per_view_points("splats.means after input-camera projection", splats.get("means"), S, H, W)
        splat_mask = gs_mask if gs_mask is not None else pts_mask
        splats = _apply_mask_to_splats(splats, splat_mask if filter_splats else None)
        if voxel_prune_splats:
            splats = _voxel_prune_splats_dict(splats, voxel_size)

        # 9. Assemble PLY_DATA
        ply_data = {
            "pts3d":          predictions.get("pts3d"),
            "pts3d_filtered": filtered_pts,
            "depth":          predictions.get("depth"),
            "gs_depth":       predictions.get("gs_depth"),
            "normals":        predictions.get("normals"),
            "pts3d_conf":     predictions.get("pts3d_conf"),
            "depth_conf":     predictions.get("depth_conf"),
            "depth_mask":     predictions.get("depth_mask"),
            "gs_depth_conf":  predictions.get("gs_depth_conf"),
            "gs_depth_mask":  predictions.get("gs_depth_mask"),
            "splats":         splats,
            "images":         imgs_tensor,
            "filter_mask": (
                torch.from_numpy(pts_mask.reshape(-1)).to(imgs_tensor.device)
                if pts_mask is not None else None
            ),
            "gs_filter_mask": (
                torch.from_numpy(gs_mask.reshape(-1)).to(imgs_tensor.device)
                if gs_mask is not None else None
            ),
            "camera_poses":   predictions.get("camera_poses"),
            "camera_intrs":   predictions.get("camera_intrs"),
            "model_camera_poses": model_camera_poses,
            "model_camera_intrs": model_camera_intrs,
            "model_pts3d": model_pts3d,
            "model_pts3d_filtered": model_pts3d_filtered,
            "missing_camera_info": missing_camera_info,
        }

        # 10. Depth / normals → ComfyUI IMAGE [S,H,W,3]
        depth_t  = predictions.get("depth")
        normal_t = predictions.get("normals")

        if depth_t is not None:
            d = depth_t[0]                                          # [S,H,W,1]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depth_out = d.repeat(1, 1, 1, 3).cpu().float()         # [S,H,W,3]
        else:
            depth_out = torch.zeros(S, H, W, 3)

        if normal_t is not None:
            normals_out = ((normal_t[0] + 1) / 2).cpu().float()    # [S,H,W,3]
        else:
            normals_out = torch.zeros(S, H, W, 3)

        # 11. Camera outputs + raw_splats for downstream nodes
        cam_poses  = predictions.get("camera_poses")
        cam_intrs  = predictions.get("camera_intrs")
        if cam_poses  is not None: cam_poses  = cam_poses.cpu().float()
        if cam_intrs  is not None: cam_intrs  = cam_intrs.cpu().float()

        predictions["images"] = imgs_tensor   # needed by SplatRefiner

        fallback_shape = (S, H, W)
        filter_mask_out = _map_to_comfy_image(pts_mask, fallback_shape, normalize=False)
        gs_filter_mask_out = _map_to_comfy_image(gs_mask, fallback_shape, normalize=False)

        return (
            ply_data, depth_out, normals_out, cam_poses, cam_intrs, predictions,
            filter_mask_out, gs_filter_mask_out,
        )


class VNCCS_WorldMirrorV2Reconstruction(_WorldMirrorV2Inference):
    """Compact WorldMirror V2 reconstruction with optional depth-based upsampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_MODEL_TYPE,),
                "images": ("IMAGE",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 3430, "min": 252, "max": 4096, "step": 14,
                    "tooltip": "Model inference resolution. Experimental high values are VRAM-heavy; dense splats can be upsampled separately."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {
                    "default": "none",
                    "tooltip": "Move model weights to CPU between GPU use to reduce VRAM at the cost of speed."
                }),
                "low_vram_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply the low-VRAM profile: depth_only, frame chunks=1, GS param chunks=1, transformer MLP chunks=8192."
                }),
                "apply_sky_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove sky-like regions before saving. Useful for outdoor panoramas where sky can create far/infinite splats."
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print camera/depth/splat alignment diagnostics to the ComfyUI console."
                }),
                "enable_splat_upsample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Build dense splats from depth with linked camera priors. Off uses native WorldMirror splats."
                }),
                "splat_upsample_scale": ("FLOAT", {
                    "default": 0.003, "min": 0.0001, "max": 0.05, "step": 0.0001,
                    "tooltip": "Gaussian size for dense backprojected splats."
                }),
                "splat_upsample_scale_mode": (["constant", "depth_adaptive", "footprint_adaptive", "hybrid_adaptive"], {
                    "default": "depth_adaptive",
                    "tooltip": "constant uses one Gaussian size; depth_adaptive grows with depth; footprint_adaptive follows local 3D spacing; hybrid uses both."
                }),
                "splat_upsample_depth_scale_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Depth growth strength, or local footprint multiplier in footprint/hybrid modes."
                }),
                "splat_upsample_depth_scale_max": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 12.0, "step": 0.1,
                    "tooltip": "Maximum multiplier over splat_upsample_scale for adaptive modes."
                }),
                "splat_upsample_voxel_prune": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Voxel-merge dense splats to reduce file size."
                }),
                "splat_upsample_voxel_size": ("FLOAT", {
                    "default": 0.0015, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Voxel size for dense splat compression."
                }),
                "splat_upsample_max_points": ("INT", {
                    "default": 9_000_000, "min": 0, "max": 50_000_000, "step": 100_000,
                    "tooltip": "Depth/distance-aware cap for dense splats. Lower values reduce file size; higher values preserve distant surfaces."
                }),
                "splat_upsample_cap_far_bias": ("FLOAT", {
                    "default": 1.75, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Preserve proportionally more far splats when applying the point cap."
                }),
                "splat_camera_source": (["camera_inputs", "predicted"], {
                    "default": "predicted",
                    "tooltip": "camera_inputs backprojects depths from input cameras using ICP scale. predicted uses the model predicted cameras."
                }),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: intrinsics from Equirect360ToViews or WorldStereo."
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: camera poses from Equirect360ToViews or WorldStereo."
                }),
                "depth_prior": ("IMAGE", {
                    "tooltip": "Optional depth prior matching the input views."
                }),
            },
        }

    RETURN_TYPES  = (
        "PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR", "VNCCS_SPLAT",
        "IMAGE", "IMAGE",
    )
    RETURN_NAMES  = (
        "ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics", "raw_splats",
        "filter_mask", "gs_filter_mask",
    )
    FUNCTION      = "run_inference"
    CATEGORY      = "VNCCS/3D"

    def run_inference(
        self, model, images, target_size=3430, offload_scheme="none", low_vram_mode=False,
        apply_sky_mask=True, debug_log=False, enable_splat_upsample=False,
        splat_upsample_scale=0.003, splat_upsample_scale_mode="depth_adaptive",
        splat_upsample_depth_scale_strength=1.0, splat_upsample_depth_scale_max=3.0,
        splat_upsample_voxel_prune=True, splat_upsample_voxel_size=0.0015,
        splat_upsample_max_points=9_000_000, splat_upsample_cap_far_bias=1.75,
        splat_camera_source="predicted", camera_intrinsics=None, camera_poses=None, depth_prior=None,
    ):
        output = super().run_inference(
            model, images, target_size=target_size, offload_scheme=offload_scheme, low_vram_mode=low_vram_mode,
            head_compute_mode="depth+gs", transformer_mlp_chunk_size=32768,
            apply_sky_mask=apply_sky_mask, filter_splats=apply_sky_mask, debug_log=debug_log,
            splat_camera_source=splat_camera_source, camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics, depth_prior=depth_prior,
        )
        if not enable_splat_upsample:
            return output
        count = images.shape[0]
        return self._build_upsampled_splats(
            output, images, _camera_prior(camera_poses, count, ((3, 4), (4, 4)), "poses"),
            _camera_prior(camera_intrinsics, count, ((3, 3),), "intrinsics"),
            max(images.shape[1:3]), "gs_depth", splat_upsample_scale, splat_upsample_scale_mode,
            splat_upsample_depth_scale_strength, splat_upsample_depth_scale_max, 0.9,
            splat_upsample_voxel_prune, splat_upsample_voxel_size,
            splat_upsample_max_points, splat_upsample_cap_far_bias,
        )

    def _depth_aware_indices(self, depth_values, max_points, far_bias=1.75, bins=8):
        depth = depth_values.detach().cpu().float().flatten()
        total = int(depth.numel())
        max_points = int(max_points)
        if max_points <= 0 or total <= max_points:
            return None

        finite = torch.isfinite(depth)
        if not bool(finite.any()):
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        depth_valid = depth[finite]
        d_min = depth_valid.min()
        d_max = depth_valid.max()
        if (d_max - d_min).abs().item() < 1e-8:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        normalized = ((depth - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
        normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        bin_ids = torch.clamp((normalized * bins).long(), max=bins - 1)
        quotas = []
        remaining = max_points
        for bin_id in range(bins):
            count = int((bin_ids == bin_id).sum().item())
            weight = 1.0 + float(far_bias) * (bin_id / max(1, bins - 1))
            quotas.append([bin_id, count, weight, 0])

        weighted_total = sum(count * weight for _, count, weight, _ in quotas)
        if weighted_total <= 0:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        for item in quotas:
            _, count, weight, _ = item
            quota = min(count, int(round(max_points * (count * weight) / weighted_total)))
            item[3] = quota
            remaining -= quota

        if remaining != 0:
            order = sorted(
                range(len(quotas)),
                key=lambda i: quotas[i][2],
                reverse=remaining > 0,
            )
            while remaining != 0:
                changed = False
                for i in order:
                    if remaining == 0:
                        break
                    count = quotas[i][1]
                    quota = quotas[i][3]
                    if remaining > 0 and quota < count:
                        quotas[i][3] += 1
                        remaining -= 1
                        changed = True
                    elif remaining < 0 and quota > 0:
                        quotas[i][3] -= 1
                        remaining += 1
                        changed = True
                if not changed:
                    break

        selected = []
        for bin_id, count, _, quota in quotas:
            if count <= 0 or quota <= 0:
                continue
            ids = torch.where(bin_ids == bin_id)[0]
            if ids.numel() <= quota:
                selected.append(ids)
            else:
                take = torch.linspace(0, ids.numel() - 1, quota, dtype=torch.long)
                selected.append(ids[take])

        if not selected:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)
        idx = torch.cat(selected).sort().values
        if idx.numel() > max_points:
            take = torch.linspace(0, idx.numel() - 1, max_points, dtype=torch.long)
            idx = idx[take]
        return idx

    def _prepare_highres_views(self, images, camera_intrinsics, upsample_size):
        image_tensors = []
        intrinsics = camera_intrinsics.detach().cpu().float().clone() if isinstance(camera_intrinsics, torch.Tensor) else None
        total = int(images.shape[0])
        for i in range(total):
            img_np = (images[i].detach().float().cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, upsample_size)

            if new_h > upsample_size:
                crop = (new_h - upsample_size) // 2
                t = t[:, crop:crop + upsample_size, :]
                if intrinsics is not None:
                    intrinsics[i, 1, 2] -= crop
            if new_w > upsample_size:
                crop = (new_w - upsample_size) // 2
                t = t[:, :, crop:crop + upsample_size]
                if intrinsics is not None:
                    intrinsics[i, 0, 2] -= crop

            if intrinsics is not None:
                sx, sy = new_w / orig_w, new_h / orig_h
                intrinsics[i, 0, 0] *= sx
                intrinsics[i, 1, 1] *= sy
                intrinsics[i, 0, 2] *= sx
                intrinsics[i, 1, 2] *= sy

            image_tensors.append(t)

        return torch.stack(image_tensors).unsqueeze(0), intrinsics

    def _build_upsampled_splats(
        self,
        output,
        original_images,
        camera_poses,
        camera_intrinsics,
        upsample_size,
        depth_source,
        splat_scale,
        scale_mode,
        depth_scale_strength,
        depth_scale_max,
        splat_opacity,
        voxel_prune,
        voxel_size,
        max_points,
        cap_far_bias,
    ):
        if camera_poses is None or camera_intrinsics is None:
            print("[WorldMirror V2] splat upsample skipped: camera_poses/camera_intrinsics required.")
            return output

        raw = output[5]
        if not isinstance(raw, dict):
            return output
        requested_depth_source = depth_source
        actual_depth_source = depth_source
        depth = raw.get(depth_source)
        if depth is None and depth_source != "depth":
            depth = raw.get("depth")
            actual_depth_source = "depth"
        if not isinstance(depth, torch.Tensor):
            print(f"[WorldMirror V2] splat upsample skipped: missing depth source '{requested_depth_source}'.")
            return output

        if depth.dim() == 5:
            depth_low = depth[0, ..., 0]
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth_low = depth[..., 0]
        elif depth.dim() == 4:
            depth_low = depth[0]
        else:
            print(f"[WorldMirror V2] splat upsample skipped: unsupported depth shape {tuple(depth.shape)}.")
            return output

        highres_imgs, highres_intrs = self._prepare_highres_views(original_images, camera_intrinsics, upsample_size)
        S, _, H, W = highres_imgs.shape[1:]
        depth_hi = torch.nn.functional.interpolate(
            depth_low.detach().cpu().float().unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        predicted_poses = raw.get("camera_poses")
        if predicted_poses is None and len(output) > 3:
            predicted_poses = output[3]
        poses = _rescale_input_pose_translation_to_prediction(camera_poses, predicted_poses)
        intrs = highres_intrs.detach().cpu().float()
        if poses is None:
            print("[WorldMirror V2] splat upsample skipped: invalid input camera_poses.")
            return output
        pts, _, _ = depth_to_world_coords_points(depth_hi, poses, intrs)
        pose_t = poses[:, :3, 3]
        anchor_frames = pose_t.norm(dim=1) <= 1e-5
        keep_mask = _suppress_generated_surfaces_seen_by_anchors(
            pts,
            depth_hi,
            poses,
            intrs,
            anchor_frames,
        )
        keep_mask &= torch.isfinite(depth_hi) & (depth_hi > 0)
        keep_flat = keep_mask.reshape(-1)
        if not keep_flat.any():
            return output
        pts_flat = pts.reshape(S * H * W, 3)[keep_flat]
        rgb_flat = highres_imgs.permute(0, 1, 3, 4, 2).reshape(S * H * W, 3)[keep_flat]
        depth_flat_values = depth_hi.reshape(S * H * W)[keep_flat]
        means = pts_flat.unsqueeze(0)
        sh = ((rgb_flat.unsqueeze(0) - 0.5) / 0.28209479177387814).float()
        depth_flat = depth_flat_values.unsqueeze(0)

        def depth_adaptive_scales():
            d_min = depth_flat.min()
            d_max = depth_flat.max()
            if (d_max - d_min).abs().item() > 1e-8:
                depth_norm = ((depth_flat - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
                multiplier = (1.0 + float(depth_scale_strength) * depth_norm).clamp(
                    1.0,
                    float(depth_scale_max),
                )
            else:
                multiplier = torch.ones_like(depth_flat)
            return torch.full_like(means, float(splat_scale)) * multiplier[..., None], multiplier

        def footprint_adaptive_scales(multiplier=None):
            dx = torch.linalg.norm(pts[:, :, 1:, :] - pts[:, :, :-1, :], dim=-1)
            dy = torch.linalg.norm(pts[:, 1:, :, :] - pts[:, :-1, :, :], dim=-1)
            dx = torch.cat([dx, dx[:, :, -1:]], dim=2)
            dy = torch.cat([dy, dy[:, -1:, :]], dim=1)
            footprint = torch.maximum(dx, dy).clamp_min(float(splat_scale))
            multiplier = max(1.0, float(depth_scale_strength)) if multiplier is None else float(multiplier)
            scales_hw = (footprint * multiplier).clamp(
                float(splat_scale),
                float(splat_scale) * float(depth_scale_max),
            )
            return scales_hw.reshape(S * H * W, 1)[keep_flat].unsqueeze(0).expand_as(means).float()

        if scale_mode == "hybrid_adaptive":
            scales_depth, multiplier = depth_adaptive_scales()
            scales_footprint = footprint_adaptive_scales(multiplier=1.0)
            scales = torch.maximum(scales_depth, scales_footprint)
            print(
                "[WorldMirror V2] Upsample hybrid-adaptive scale: "
                f"base={float(splat_scale):.5f}, max_multiplier={float(multiplier.max().item()):.2f}, "
                f"max_scale={float(scales.max().item()):.5f}, mean_scale={float(scales.mean().item()):.5f}"
            )
        elif scale_mode == "footprint_adaptive":
            scales = footprint_adaptive_scales()
            print(
                "[WorldMirror V2] Upsample footprint-adaptive scale: "
                f"base={float(splat_scale):.5f}, multiplier={max(1.0, float(depth_scale_strength)):.2f}, "
                f"max_scale={float(scales.max().item()):.5f}, mean_scale={float(scales.mean().item()):.5f}"
            )
        elif scale_mode == "depth_adaptive":
            scales, multiplier = depth_adaptive_scales()
            print(
                "[WorldMirror V2] Upsample depth-adaptive scale: "
                f"base={float(splat_scale):.5f}, max_multiplier={float(multiplier.max().item()):.2f}"
            )
        else:
            scales = torch.full_like(means, float(splat_scale))
        opacities = torch.full((1, means.shape[1]), float(splat_opacity), dtype=torch.float32)
        quats = torch.zeros((1, means.shape[1], 4), dtype=torch.float32)
        quats[..., 0] = 1.0
        weights = torch.ones((1, means.shape[1]), dtype=torch.float32)
        if bool(anchor_frames.any()) and bool((~anchor_frames).any()):
            anchor_weight = 64.0
            frame_weights = torch.where(
                anchor_frames,
                torch.full_like(pose_t[:, 0], anchor_weight),
                torch.ones_like(pose_t[:, 0]),
            )
            weights = frame_weights[:, None].expand(S, H * W).reshape(S * H * W)[keep_flat].unsqueeze(0).float()
            print(
                "[WorldMirror V2] High-res anchor splat priority enabled: "
                f"anchors={int(anchor_frames.sum().item())}/{S}, weight={anchor_weight:g}"
            )

        splats = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh": sh[:, :, None, :],
            "weights": weights,
        }
        n_before = means.shape[1]
        if voxel_prune:
            splats = _voxel_prune_splats_dict(splats, voxel_size)

        if max_points > 0:
            distances = (splats["means"][0].norm(dim=-1) if voxel_prune else depth_flat_values)
            indices = self._depth_aware_indices(distances, max_points, far_bias=cap_far_bias)
            if indices is not None:
                splats = {key: [value[0][indices]] for key, value in splats.items()}

        ply_data = dict(output[0])
        ply_data.update({
            "splats": splats,
            "skip_scale_filter": True,
            "images": highres_imgs,
            "camera_poses": poses.unsqueeze(0),
            "camera_intrs": intrs.unsqueeze(0),
            "pts3d": None,
            "pts3d_filtered": None,
        })
        raw_new = dict(raw)
        raw_new.update({
            "splats": splats,
            "images": highres_imgs,
            "camera_poses": poses.unsqueeze(0),
            "camera_intrs": intrs.unsqueeze(0),
        })

        print(
            f"[WorldMirror V2] Upsampled splats: {S} views @ {H}x{W}, "
            f"{n_before} raw gaussians from {actual_depth_source}"
        )
        return (
            ply_data,
            output[1],
            output[2],
            poses.unsqueeze(0),
            intrs.unsqueeze(0),
            raw_new,
            output[6],
            output[7],
        )


NODE_CLASS_MAPPINGS = {
    "VNCCS_WorldMirrorV2Loader": VNCCS_WorldMirrorV2Loader,
    "VNCCS_WorldMirrorV2Reconstruction": VNCCS_WorldMirrorV2Reconstruction,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_WorldMirrorV2Loader": "VNCCS WorldMirror V2 Model Loader",
    "VNCCS_WorldMirrorV2Reconstruction": "VNCCS WorldMirror V2 3D Reconstruction",
}
