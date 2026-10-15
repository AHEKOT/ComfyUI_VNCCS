"""Cached Gaussian depth supervision using ComfyUI's native MoGe nodes."""

from io import BytesIO

import numpy as np
from PIL import Image
import torch


MOGE_MODEL_NAME = "moge_2_vitl_normal_fp16.safetensors"
DEPTH_LOSS_WEIGHT = 0.05


def _align_depth(depth, valid, samples):
    """Recover MoGe's scene scale from observed COLMAP camera-space depths."""
    height, width = depth.shape
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    inside = (
        np.isfinite(samples).all(axis=1) & (samples[:, 2] > 0)
        & (samples[:, 0] >= 0) & (samples[:, 0] < width)
        & (samples[:, 1] >= 0) & (samples[:, 1] < height)
    )
    samples = samples[inside]
    xy = np.floor(samples[:, :2]).astype(np.int64)
    observed = valid[xy[:, 1], xy[:, 0]]
    values = depth[xy[observed, 1], xy[observed, 0]]
    if not len(values):
        return None
    scale = float(np.median(samples[observed, 2] / values))
    if not np.isfinite(scale) or scale <= 0:
        return None
    aligned = np.where(valid, depth * scale, 0).astype(np.float32)
    return np.where(np.isfinite(aligned) & (aligned > 0), aligned, 0).astype(np.float32)


def _depth_samples(reconstruction, points, frames):
    points3d = dict(reconstruction.points3D)
    point_ids = [
        key for key in sorted(points3d)
        if np.isfinite(points3d[key].xyz).all() and np.isfinite(points3d[key].color).all()
    ]
    normalized_points = dict(zip(point_ids, points))
    images = {image.name: image for image in dict(reconstruction.images).values()}
    observations = {}
    for frame in frames:
        samples = []
        for observation in images[frame["name"]].points2D:
            if not observation.has_point3D():
                continue
            point = normalized_points.get(int(observation.point3D_id))
            if point is not None:
                depth = frame["view"][2, :3] @ point + frame["view"][2, 3]
                samples.append((*observation.xy, depth))
        observations[frame["name"]] = samples
    return observations


@torch.inference_mode()
def prepare_depth_targets(reconstruction, training_frames, points, frames, interrupt):
    pending = []
    for frame in frames:
        cached = training_frames[frame["name"]]
        depth = cached.get("moge_depth") if isinstance(cached, dict) else None
        if depth is None:
            if not isinstance(cached, dict) or "source_png" not in cached:
                raise ValueError(
                    "The cached scene has no original RGB frames for MoGe; rerun "
                    "VNCCS Gaussian Camera Reconstruction once"
                )
            pending.append(frame)
        else:
            frame["depth"] = depth
    if not pending:
        print(f"[VNCCS Gaussian] Reusing cached MoGe depth for {len(frames)} frames")
        return

    try:
        import comfy.model_management as management
        from comfy_extras.nodes_moge import LoadMoGeModel, MoGeInference
    except ImportError as error:
        raise RuntimeError("Depth supervision requires ComfyUI's native MoGe nodes; update ComfyUI") from error

    samples = _depth_samples(reconstruction, points, pending)
    print(f"[VNCCS Gaussian] Computing MoGe depth: {len(pending)} frames; {MOGE_MODEL_NAME}")
    try:
        model = LoadMoGeModel.execute(MOGE_MODEL_NAME)[0]
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Place {MOGE_MODEL_NAME} in ComfyUI models/geometry_estimation to enable depth supervision"
        ) from error

    try:
        for start in range(0, len(pending), 4):
            interrupt()
            batch = pending[start:start + 4]
            images = []
            for frame in batch:
                with Image.open(BytesIO(training_frames[frame["name"]]["source_png"])) as source:
                    images.append(np.array(source.convert("RGB"), dtype=np.uint8))
            image_batch = torch.from_numpy(np.stack(images)).float() / 255.0
            width = image_batch.shape[2]
            # The camera stage uses one shared pinhole camera for the entire orbit.
            fov_x = float(np.rad2deg(2 * np.arctan(width / (2 * batch[0]["K"][0, 0]))))
            geometry = MoGeInference.execute(
                moge_model=model, image=image_batch, resolution_level=9,
                fov_x_degrees=fov_x, batch_size=4, force_projection=True,
                apply_mask=True, refine_steps=3,
            )[0]
            depths = geometry["depth"].float().cpu().numpy()
            masks = geometry["mask"].cpu().numpy().astype(bool)
            del geometry
            for index, frame in enumerate(batch):
                depth = depths[index]
                valid = (
                    masks[index] & np.isfinite(depth) & (depth > 0)
                    & (frame["rgba"][..., 3] >= 230)
                )
                aligned = _align_depth(depth, valid, samples[frame["name"]])
                if aligned is None:
                    raise ValueError(
                        f"Cannot align MoGe depth for {frame['name']}: no valid foreground "
                        "COLMAP depth observations; disable depth supervision "
                        "or improve camera reconstruction"
                    )
                training_frames[frame["name"]]["moge_depth"] = aligned
                frame["depth"] = aligned
            print(
                f"[VNCCS Gaussian] MoGe depth: {min(start + 4, len(pending))}/{len(pending)}; "
                "aligned to COLMAP and cached"
            )
    finally:
        management.unload_model_and_clones(model.patcher)


def depth_loss(rendered_depth, target_depth, alpha):
    valid_prediction = torch.isfinite(rendered_depth) & (rendered_depth > 0)
    valid = (
        torch.isfinite(target_depth) & (target_depth > 0)
        & valid_prediction & (alpha.detach() > 0.01)
    )
    predicted = torch.where(valid, rendered_depth, torch.ones_like(rendered_depth))
    target = torch.where(valid, target_depth, torch.ones_like(target_depth))
    error = (predicted.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()).abs()
    return error.sum() / valid.sum().clamp_min(1)
