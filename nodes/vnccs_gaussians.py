"""Camera reconstruction, Gaussian training and textured meshes for VNCCS orbit scenes.

Includes MoGe depth supervision and planar 2DGS with squared NDC-depth distortion.
Tile sorting uses gsplat; fused CUDA kernels composite RGB and geometry.
"""

import ctypes as C
from datetime import datetime
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .chroma_screen_matte import screen_matte
from .helpers.gaussian_cuda_runtime import load_surface_kernels
from ..utils import atomic_output_path, file_fingerprint, safe_join_under


# Camera reconstruction and cached scene

DEFAULT_MAX_FRAMES = 160
DEFAULT_MAX_SIDE = 2048
DEFAULT_MATCH_WINDOW = 20
DEFAULT_MATTE_TOLERANCE = 0.15
DEFAULT_RANSAC_MAX_ERROR = 3.0
DEFAULT_MIN_NUM_INLIERS = 20
DEFAULT_SIFT_MAX_RATIO = 0.75
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
        training_frames[name] = {
            "rgba": (training_rgba * 255).round().to(torch.uint8).cpu().numpy(),
            "alpha_uneroded": mask,
            "source_png": (image_dir / name).read_bytes(),
        }
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
    ransac_max_error=DEFAULT_RANSAC_MAX_ERROR,
    min_num_inliers=DEFAULT_MIN_NUM_INLIERS,
    sift_max_ratio=DEFAULT_SIFT_MAX_RATIO,
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

    matching = pycolmap.FeatureMatchingOptions()
    matching.sift.max_ratio = sift_max_ratio
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.max_error = ransac_max_error
    verification.min_num_inliers = min_num_inliers
    print(
        f"[VNCCS Gaussian] Match thresholds: ransac_error={ransac_max_error:g}px, "
        f"min_inliers={min_num_inliers}, sift_ratio={sift_max_ratio:g}"
    )
    pycolmap.match_image_pairs(
        database_path,
        matching_options=matching,
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs_path),
        verification_options=verification,
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
        cached = training_frames[image.name]
        registered.append(
            (
                image.name,
                view,
                np.asarray(camera.calibration_matrix(), dtype=np.float64),
                cached["rgba"] if isinstance(cached, dict) else cached,
                cached.get("source_rgb") if isinstance(cached, dict) else None,
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
            "name": item[0],
            "view": views[index],
            "K": item[2].astype(np.float32),
            "rgba": item[3],
            "source_rgb": item[4],
            "depth_unit": float(1.0 / radius),
        }
        for index, item in enumerate(registered)
    ]
    return points.astype(np.float32), colors.astype(np.float32), frames


def _top_view_cameras(frames):
    """Return four fixed 45-degree views paired with the nearest source azimuths."""
    poses = np.linalg.inv(np.stack([frame["view"] for frame in frames]).astype(np.float64))
    centers = poses[:, :3, 3]
    forwards = poses[:, :3, 2]
    projectors = np.eye(3)[None] - forwards[:, :, None] * forwards[:, None, :]
    target = np.linalg.lstsq(
        projectors.sum(axis=0),
        np.einsum("nij,nj->i", projectors, centers),
        rcond=None,
    )[0]
    up = -poses[:, :3, 1].mean(axis=0)
    up_length = np.linalg.norm(up)
    if not np.isfinite(target).all() or up_length < 1e-6:
        raise ValueError("Orbit cameras do not define a stable upright preview frame")
    up /= up_length
    offsets = centers - target
    horizontal = offsets - (offsets @ up)[:, None] * up
    lengths = np.linalg.norm(horizontal, axis=1)
    if not np.isfinite(lengths).all() or lengths.min() < 1e-6:
        raise ValueError("Orbit cameras do not define usable preview azimuths")
    directions = horizontal / lengths[:, None]
    front = directions[0]
    side = np.cross(up, front)
    radius = np.median(np.linalg.norm(offsets, axis=1))
    previews = []
    for angle in np.arange(4) * (np.pi / 2):
        direction = np.cos(angle) * front + np.sin(angle) * side
        frame = frames[int(np.argmax(directions @ direction))]
        position = target + radius * (direction + up) / np.sqrt(2.0)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(forward, right), forward))
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ position
        previews.append((frame, view))
    return previews


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
                "ransac_max_error": (
                    "FLOAT",
                    {
                        "default": DEFAULT_RANSAC_MAX_ERROR,
                        "min": 0.01,
                        "step": 0.1,
                        "tooltip": "Maximum geometric verification error in pixels. Lower values are stricter.",
                    },
                ),
                "min_num_inliers": (
                    "INT",
                    {
                        "default": DEFAULT_MIN_NUM_INLIERS,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Minimum geometrically verified feature matches per image pair. Higher values are stricter.",
                    },
                ),
                "sift_max_ratio": (
                    "FLOAT",
                    {
                        "default": DEFAULT_SIFT_MAX_RATIO,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum descriptor distance ratio. Lower values reject more ambiguous feature matches.",
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
        ransac_max_error=DEFAULT_RANSAC_MAX_ERROR,
        min_num_inliers=DEFAULT_MIN_NUM_INLIERS,
        sift_max_ratio=DEFAULT_SIFT_MAX_RATIO,
    ):
        if images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("images must be a BHWC RGB or RGBA tensor")
        if len(images) < 20:
            raise ValueError("At least 20 ordered orbit frames are required")
        if images.shape[1] < 16 or images.shape[2] < 16:
            raise ValueError("Orbit frames are too small for camera reconstruction")
        if not math.isfinite(ransac_max_error) or ransac_max_error <= 0:
            raise ValueError("ransac_max_error must be finite and positive")
        if min_num_inliers < 1:
            raise ValueError("min_num_inliers must be positive")
        if not 0 < sift_max_ratio <= 1:
            raise ValueError("sift_max_ratio must be greater than zero and at most one")
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
                ransac_max_error,
                min_num_inliers,
                sift_max_ratio,
            )
            _, _, frames = _scene_data(reconstruction, training_frames)
            for frame, _ in _top_view_cameras(frames):
                interrupt()
                with Image.open(image_dir / frame["name"]) as source:
                    training_frames[frame["name"]]["source_rgb"] = np.array(
                        source.convert("RGB"), dtype=np.uint8
                    )
        print(
            f"[VNCCS Gaussian] Camera stage ready: {len(training_frames)} frames; "
            "ComfyUI will cache this output"
        )
        return ((reconstruction, training_frames),)


# MoGe depth supervision

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


# Planar Gaussian rendering and refinement

DISTORTION_WEIGHT = 1_000.0
NORMAL_WEIGHT = 0.05
NDC_NEAR = 0.2
NDC_FAR = 100.0
_NEAR_PLANE = NDC_NEAR


def compatibility_scales(log_scales):
    """Approximate surfels by thin ellipsoids for standard 3DGS consumers."""
    if log_scales.shape[-1] != 2:
        return log_scales.expand(-1, 3)
    thickness = log_scales.amin(dim=-1, keepdim=True) + math.log(1e-3)
    return torch.cat((log_scales, thickness), dim=-1)


@torch.no_grad()
def _tie_planar_scale_gradients(log_scales):
    # Equal initial radii and Adam states stay tied; the shared-radius derivative is the sum.
    if log_scales.grad is not None:
        log_scales.grad.copy_(log_scales.grad.sum(dim=-1, keepdim=True))


def _project_surfels(means, quats, scales, view, intrinsics, width, height, near_plane):
    """Perspective surfel transforms and the original three-sigma conic bounds.

    This is per-surfel work; pixel compositing and its backward remain fused CUDA.
    Keep the bounding-center derivative in the same graph as the transform.
    """
    from gsplat.utils import normalized_quat_to_rotmat

    rotation = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))
    camera_rotation = view[0, :3, :3]
    centers = means @ camera_rotation.T + view[0, :3, 3]
    axes = camera_rotation @ (rotation[..., :2] * scales[:, None, :2])
    local_to_camera = torch.cat((axes, centers[..., None]), dim=-1)
    transforms = (intrinsics[0] @ local_to_camera)[None]
    u, v, w = transforms.unbind(-2)
    cutoff = transforms.new_tensor((9.0, 9.0, -1.0))
    denominator = (cutoff * w.square()).sum(-1, keepdim=True)
    valid = torch.isfinite(denominator) & (denominator.abs() > torch.finfo(means.dtype).tiny)
    f = cutoff / torch.where(valid, denominator, torch.ones_like(denominator))
    xy = torch.stack(((f * u * w).sum(-1), (f * v * w).sum(-1)), -1)
    extent2 = xy.square() - torch.stack(((f * u.square()).sum(-1), (f * v.square()).sum(-1)), -1)
    with torch.no_grad():
        radius = torch.ceil(extent2.clamp_min(1e-4).sqrt().amax(-1).clamp_min(3 / math.sqrt(2)))
        visible = (valid[..., 0] & torch.isfinite(xy).all(-1) & torch.isfinite(radius)
                   & (centers[None, :, 2] > near_plane)
                   & (xy[..., 0] + radius > 0) & (xy[..., 0] - radius < width)
                   & (xy[..., 1] + radius > 0) & (xy[..., 1] - radius < height))
        radii = torch.where(visible, radius, 0).to(torch.int32)[..., None].expand(-1, -1, 2).contiguous()
    camera_normal = rotation[..., 2] @ camera_rotation.T
    facing = torch.where((camera_normal * centers).sum(-1, keepdim=True) < 0, 1.0, -1.0)
    normals = (camera_normal * facing)[None]
    return radii, xy, centers[None, :, 2], transforms, normals


def render_2dgs(splats, view, intrinsics, width, height, sh_degree, absgrad=False,
                with_geometry=False, depth_unit=1.0):
    coefficients = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    scales = splats["scales"].exp()
    return _render_surface(splats, coefficients, scales, view, intrinsics,
                           width, height, sh_degree, absgrad, with_geometry, depth_unit)


def _render_surface(splats, coefficients, scales, view, intrinsics, width, height,
                    sh_degree, absgrad, with_geometry=True, depth_unit=1.0):
    from gsplat import spherical_harmonics
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    near_plane = _NEAR_PLANE * depth_unit
    ndc = -NDC_FAR * NDC_NEAR / (NDC_FAR - NDC_NEAR) * depth_unit
    radii, xy, depths, transforms, normals = _project_surfels(
        splats["means"], splats["quats"], scales, view, intrinsics, width, height,
        near_plane,
    )
    tile_width, tile_height = (width + 15) // 16, (height + 15) // 16
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        xy, radii, depths, 16, tile_width, tile_height, packed=False,
    )
    offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    camera_center = -(view[0, :3, :3].T @ view[0, :3, 3])
    rgb = (spherical_harmonics(
        sh_degree, splats["means"][None] - camera_center, coefficients[None],
        masks=(radii > 0).all(dim=-1),
    ) + 0.5).clamp_min(0)
    reference = ndc / depths.detach().mean().clamp_min(near_plane)
    opacities = splats["opacities"].sigmoid()[None]
    densify = torch.zeros_like(xy, requires_grad=True)
    # Transform per surfel, not per pixel; this commutes with alpha compositing.
    world_normals = normals @ view[0, :3, :3]
    image, alpha, maps = rasterize_surface(
        xy, transforms, rgb, opacities, world_normals, reference,
        depths, densify, offsets, flatten_ids, width, height, absgrad,
        with_geometry, near_plane, ndc,
    )
    return image, alpha, {
        "camera_ids": None, "gaussian_ids": None, "radii": radii,
        "means2d": xy, "depths": depths, "ray_transforms": transforms,
        "opacities": opacities, "normals": normals,
        "tile_width": tile_width, "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids,
        "flatten_ids": flatten_ids, "isect_offsets": offsets,
        "width": width, "height": height, "tile_size": 16, "n_cameras": 1,
        "gradient_2dgs": densify, "surface_maps": maps,
    }


def geometry_maps(info, intrinsics):
    """Geometry is already evaluated by the fused CUDA rasterization."""
    if info["n_cameras"] != 1:
        raise ValueError("Surface geometry expects one training camera at a time")
    return info.pop("surface_maps")


def geometry_losses(maps, view, intrinsics, use_normal=True):
    median, alpha, distortion = maps[..., :3].split(1, dim=-1)
    if not use_normal:
        return distortion.mean(), distortion.new_zeros(()), median
    if maps.is_cuda:
        return distortion.mean(), normal_consistency(maps, view, intrinsics), median

    from gsplat.utils import depth_to_normal

    # gsplat returns composited normals in world space, including camera-facing signs.
    surface_normal = depth_to_normal(median, torch.linalg.inv(view), intrinsics)
    target_normal = surface_normal * alpha.detach()
    normal_error = 1.0 - (maps[..., 3:6] * target_normal).sum(-1, keepdim=True)
    normal_loss = normal_error.mean()
    return distortion.mean(), normal_loss, median


def geometry_schedule(step, training_steps, distortion_weight, normal_weight):
    """Scale the original 3k/7k warm-ups to the requested training duration."""
    return (
        distortion_weight if step >= int(training_steps * 3_000 / 30_000) else 0.0,
        normal_weight if step >= int(training_steps * 7_000 / 30_000) else 0.0,
    )


@torch.no_grad()
def split_2dgs(params, optimizers, state, mask):
    from gsplat.utils import normalized_quat_to_rotmat

    selected, rest = torch.where(mask)[0], torch.where(~mask)[0]
    scales = params["scales"][selected].exp()
    rotations = normalized_quat_to_rotmat(F.normalize(params["quats"][selected], dim=-1))
    samples = torch.randn((2, len(selected), 2), device=scales.device) * scales
    offsets = torch.einsum("nij,bnj->bni", rotations[..., :2], samples)
    for name, parameter in list(params.items()):
        repeats = [2] + [1] * (parameter.ndim - 1)
        if name == "means":
            children = (parameter[selected] + offsets).reshape(-1, 3)
        elif name == "scales":
            children = (params["scales"][selected] - math.log(1.6)).repeat(2, 1)
        else:
            children = parameter[selected].repeat(repeats)
        replacement = torch.nn.Parameter(torch.cat((parameter[rest], children)),
                                         requires_grad=parameter.requires_grad)
        params[name] = replacement
        if name not in optimizers:
            if parameter.requires_grad:
                raise RuntimeError(f"Missing optimizer for trainable surfel parameter {name}")
            continue
        optimizer = optimizers[name]
        old_state = optimizer.state.pop(parameter, {})
        for key, value in old_state.items():
            if key != "step":
                old_state[key] = torch.cat((value[rest], value.new_zeros((2 * len(selected), *value.shape[1:]))))
        for group in optimizer.param_groups:
            group["params"] = [replacement]
        optimizer.state[replacement] = old_state
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            repeats = [2] + [1] * (value.ndim - 1)
            children = value[selected].repeat(repeats)
            if name == "radii":
                children.zero_()  # Parent screen bounds are not observations of its children.
            state[name] = torch.cat((value[rest], children))


# Fused CUDA RGB and geometry compositing

_SOURCE = r'''
struct V3 { float x, y, z; };
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
struct Surfel {
    float xy[2], opacity, u[3], v[3], w[3], rgb[3], normal[3];
    int id;
};
__device__ __forceinline__ Surfel load_surfel(
    int g, const float* xy, const float* transforms, const float* rgb,
    const float* opacity, const float* normals) {
    Surfel s;
    s.id=g; s.opacity=opacity[g];
    s.xy[0]=xy[2*g]; s.xy[1]=xy[2*g+1];
    #pragma unroll
    for(int k=0;k<3;++k) {
        s.u[k]=transforms[9*g+k]; s.v[k]=transforms[9*g+3+k];
        s.w[k]=transforms[9*g+6+k]; s.rgb[k]=rgb[3*g+k];
        s.normal[k]=normals[3*g+k];
    }
    return s;
}
struct Hit { V3 hu,hv,c; float local_u,sv,depth,dx,dy,rho3,rho2,visibility,alpha; };
__device__ __forceinline__ bool intersect(const Surfel& s,float px,float py,float near_plane,Hit& h) {
    h.hu={px*s.w[0]-s.u[0],px*s.w[1]-s.u[1],px*s.w[2]-s.u[2]};
    h.hv={py*s.w[0]-s.v[0],py*s.w[1]-s.v[1],py*s.w[2]-s.v[2]};
    h.c=cross3(h.hu,h.hv);
    if(h.c.z==0.f) return false;
    h.local_u=h.c.x/h.c.z; h.sv=h.c.y/h.c.z;
    h.depth=h.local_u*s.w[0]+h.sv*s.w[1]+s.w[2];
    // The same accepted intersections feed RGB, alpha, depth, normals and distortion.
    if(!isfinite(h.local_u)||!isfinite(h.sv)||!isfinite(h.depth)||h.depth<near_plane) return false;
    h.dx=s.xy[0]-px; h.dy=s.xy[1]-py;
    h.rho3=h.local_u*h.local_u+h.sv*h.sv;
    h.rho2=2.f*(h.dx*h.dx+h.dy*h.dy);
    h.visibility=__expf(-0.5f*fminf(h.rho3,h.rho2));
    h.alpha=fminf(0.99f,s.opacity*h.visibility);
    return h.alpha>=1.f/255.f;
}
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for(int offset=16;offset;offset>>=1)
        value+=__shfl_down_sync(0xffffffff,value,offset);
    return value;
}

extern "C" __global__ void surface_forward(
    const float* xy,const float* transforms,const float* rgb,const float* opacity,
    const float* normals,const float* reference,const int* offsets,const int* ids,
    int width,int height,int intersections,float ndc,float near_plane,int geometry,
    float* image,float* alphas,float* maps,float* moments,int* last,int* median) {
    __shared__ Surfel batch[256];
    const int lane=threadIdx.y*16+threadIdx.x;
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    const int tiles_x=(width+15)/16,tiles_y=(height+15)/16;
    const int tile=blockIdx.y*tiles_x+blockIdx.x;
    const int start=offsets[tile];
    const int end=tile+1<tiles_x*tiles_y?offsets[tile+1]:intersections;
    const bool inside=x<width&&y<height;
    const float px=x+0.5f,py=y+0.5f;
    float T=1.f,A=0.f,B=0.f,C=0.f,color[3]={},normal[3]={};
    float median_depth=0.f;
    int last_id=start-1,median_id=-1;
    bool done=!inside;
    for(int base=start;base<end;base+=256) {
        if(__syncthreads_count(done)==256) break;
        const int index=base+lane;
        if(index<end) batch[lane]=load_surfel(ids[index],xy,transforms,rgb,opacity,normals);
        __syncthreads();
        const int size=min(256,end-base);
        for(int j=0;j<size&&!done;++j) {
            const Surfel& s=batch[j];
            Hit h;
            if(!intersect(s,px,py,near_plane,h)) continue;
            const float next=T*(1.f-h.alpha);
            if(next<1e-4f) { done=true; break; }
            const float weight=T*h.alpha;
            #pragma unroll
            for(int k=0;k<3;++k) {
                color[k]+=weight*s.rgb[k];
                if(geometry) normal[k]+=weight*s.normal[k];
            }
            if(geometry) {
                if(T>0.5f) { median_id=s.id; median_depth=h.depth; }
                const float m=ndc/h.depth-reference[0];
                A+=weight; B+=weight*m; C+=weight*m*m;
            }
            T=next; last_id=base+j;
        }
        __syncthreads();
    }
    if(!inside) return;
    const int pixel=y*width+x;
    #pragma unroll
    for(int k=0;k<3;++k) { image[3*pixel+k]=color[k]; maps[6*pixel+3+k]=normal[k]; }
    alphas[pixel]=1.f-T;
    maps[6*pixel]=median_depth;
    maps[6*pixel+1]=1.f-T; maps[6*pixel+2]=fmaxf(A*C-B*B,0.f);
    moments[4*pixel]=A; moments[4*pixel+1]=B;
    moments[4*pixel+2]=C; moments[4*pixel+3]=T;
    last[pixel]=last_id; median[pixel]=median_id;
}

extern "C" __global__ void surface_backward(
    const float* xy,const float* transforms,const float* rgb,const float* opacity,
    const float* normals,const float* reference,const int* offsets,const int* ids,
    int width,int height,int intersections,float ndc,float near_plane,int geometry,
    const float* maps,const float* moments,const int* last,const int* median,
    const float* v_image,const float* v_alpha,const float* v_maps,
    float* v_xy,float* v_transform,float* v_rgb,float* v_opacity,
    float* v_normals,float* v_xy_abs) {
    __shared__ Surfel batch[256];
    const int lane=threadIdx.y*16+threadIdx.x;
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    const int tiles_x=(width+15)/16,tiles_y=(height+15)/16;
    const int tile=blockIdx.y*tiles_x+blockIdx.x;
    const int start=offsets[tile];
    const int end=tile+1<tiles_x*tiles_y?offsets[tile+1]:intersections;
    const bool inside=x<width&&y<height;
    const int pixel=inside?y*width+x:0;
    const int final=inside?last[pixel]:start-1;
    int warp_final=final;
    #pragma unroll
    for(int offset=16;offset;offset>>=1)
        warp_final=max(warp_final,__shfl_xor_sync(0xffffffff,warp_final,offset));
    const float px=x+0.5f,py=y+0.5f;
    const float A=moments[4*pixel],B=moments[4*pixel+1],C=moments[4*pixel+2];
    const float T_final=moments[4*pixel+3];
    float T=T_final,tail_rgb[3]={},tail_normal[3]={},tail_distortion=0.f;
    float color_grad[3]={},normal_grad[3]={};
    #pragma unroll
    for(int k=0;k<3;++k) {
        color_grad[k]=inside&&v_image?v_image[3*pixel+k]:0.f;
        normal_grad[k]=inside&&v_maps&&geometry?v_maps[6*pixel+3+k]:0.f;
    }
    const float ga=inside&&v_alpha?v_alpha[pixel]:0.f;
    const float gm=inside&&v_maps?v_maps[6*pixel+1]:0.f;
    const float gd=inside&&v_maps&&geometry&&A*C-B*B>=0.f?v_maps[6*pixel+2]:0.f;
    const float gz=inside&&v_maps&&geometry?v_maps[6*pixel]:0.f;
    for(int base=end-1;base>=start;base-=256) {
        const int index=base-lane;
        if(index>=start) batch[lane]=load_surfel(ids[index],xy,transforms,rgb,opacity,normals);
        __syncthreads();
        const int size=min(256,base-start+1);
        for(int j=max(0,base-warp_final);j<size;++j) {
            const Surfel& s=batch[j];
            // xy(2), transform(9), RGB(3), opacity(1), normal(3).
            float gradient[18]={};
            if(inside&&base-j<=final) {
                Hit h;
                if(intersect(s,px,py,near_plane,h)) {
                    const float reciprocal=1.f/(1.f-h.alpha);
                    T*=reciprocal;
                    const float weight=h.alpha*T;
                    float va=T_final*reciprocal*(ga+gm);
                    #pragma unroll
                    for(int k=0;k<3;++k) {
                        gradient[11+k]=weight*color_grad[k];
                        gradient[15+k]=weight*normal_grad[k];
                        va+=(s.rgb[k]*T-tail_rgb[k]*reciprocal)*color_grad[k];
                        va+=(s.normal[k]*T-tail_normal[k]*reciprocal)*normal_grad[k];
                        tail_rgb[k]+=weight*s.rgb[k]; tail_normal[k]+=weight*s.normal[k];
                    }
                    float dw=0.f,vz=0.f;
                    if(geometry) {
                        const float m=ndc/h.depth-reference[0];
                        dw=gd*(C+A*m*m-2.f*B*m);
                        vz=-2.f*weight*(A*m-B)*gd*ndc/(h.depth*h.depth);
                    }
                    va+=T*dw-tail_distortion*reciprocal;
                    tail_distortion+=weight*dw;
                    if(s.id==median[pixel]&&gz!=0.f) {
                        vz+=gz;
                    }
                    // Match the reference alpha-cap backward; do not freeze opaque surfels.
                    const float vG=s.opacity*va;
                    gradient[14]=h.visibility*va;
                    const float vu=(h.rho3<=h.rho2?-h.visibility*vG*h.local_u:0.f)+vz*s.w[0];
                    const float vv=(h.rho3<=h.rho2?-h.visibility*vG*h.sv:0.f)+vz*s.w[1];
                    const V3 vc={vu/h.c.z,vv/h.c.z,-(vu*h.local_u+vv*h.sv)/h.c.z};
                    const V3 vhu=cross3(h.hv,vc),vhv=cross3(vc,h.hu);
                    gradient[2]=-vhu.x; gradient[3]=-vhu.y; gradient[4]=-vhu.z;
                    gradient[5]=-vhv.x; gradient[6]=-vhv.y; gradient[7]=-vhv.z;
                    gradient[8]=px*vhu.x+py*vhv.x+vz*h.local_u;
                    gradient[9]=px*vhu.y+py*vhv.y+vz*h.sv;
                    gradient[10]=px*vhu.z+py*vhv.z+vz;
                    if(h.rho3>h.rho2) {
                        gradient[0]=-2.f*h.visibility*vG*h.dx;
                        gradient[1]=-2.f*h.visibility*vG*h.dy;
                    }
                }
            }
            float ax=0.f,ay=0.f;
            if(v_xy_abs) { ax=warp_sum(fabsf(gradient[4])); ay=warp_sum(fabsf(gradient[7])); }
            #pragma unroll
            for(int k=0;k<18;++k) gradient[k]=warp_sum(gradient[k]);
            if((lane&31)==0) {
                const int g=s.id;
                #pragma unroll
                for(int k=0;k<2;++k) if(gradient[k]!=0.f) atomicAdd(v_xy+2*g+k,gradient[k]);
                #pragma unroll
                for(int k=0;k<9;++k) if(gradient[2+k]!=0.f) atomicAdd(v_transform+9*g+k,gradient[2+k]);
                #pragma unroll
                for(int k=0;k<3;++k) {
                    if(gradient[11+k]!=0.f) atomicAdd(v_rgb+3*g+k,gradient[11+k]);
                    if(gradient[15+k]!=0.f) atomicAdd(v_normals+3*g+k,gradient[15+k]);
                }
                if(gradient[14]!=0.f) atomicAdd(v_opacity+g,gradient[14]);
                if(v_xy_abs) { if(ax!=0.f) atomicAdd(v_xy_abs+2*g,ax); if(ay!=0.f) atomicAdd(v_xy_abs+2*g+1,ay); }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ V3 world_ray(int x,int y,const float* R,const float* K) {
    const float rx=(x+0.5f-K[2])/K[0],ry=(y+0.5f-K[5])/K[4];
    return {R[0]*rx+R[3]*ry+R[6],R[1]*rx+R[4]*ry+R[7],R[2]*rx+R[5]*ry+R[8]};
}
__device__ __forceinline__ bool normal_data(
    const float* maps,const float* R,const float* K,int width,int height,int x,int y,
    V3& dx,V3& dy,V3& c,float& length) {
    if(x<1||y<1||x>=width-1||y>=height-1) return false;
    const int p=y*width+x;
    const V3 down=world_ray(x,y+1,R,K),up=world_ray(x,y-1,R,K);
    const V3 right=world_ray(x+1,y,R,K),left=world_ray(x-1,y,R,K);
    const float zd=maps[6*(p+width)],zu=maps[6*(p-width)];
    const float zr=maps[6*(p+1)],zl=maps[6*(p-1)];
    dx={zd*down.x-zu*up.x,zd*down.y-zu*up.y,zd*down.z-zu*up.z};
    dy={zr*right.x-zl*left.x,zr*right.y-zl*left.y,zr*right.z-zl*left.z};
    c=cross3(dx,dy); length=sqrtf(c.x*c.x+c.y*c.y+c.z*c.z);
    return true;
}
extern "C" __global__ void normal_forward(
    const float* maps,const float* R,const float* K,int width,int height,float* loss) {
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=width||y>=height) return;
    const int p=y*width+x;
    V3 dx,dy,c; float length;
    loss[p]=1.f;
    if(!normal_data(maps,R,K,width,height,x,y,dx,dy,c,length)) return;
    const float reciprocal=1.f/fmaxf(length,1e-12f);
    loss[p]=1.f-maps[6*p+1]*reciprocal*(maps[6*p+3]*c.x+maps[6*p+4]*c.y+maps[6*p+5]*c.z);
}
extern "C" __global__ void normal_backward(
    const float* maps,const float* R,const float* K,int width,int height,
    const float* v_loss,float* v_maps) {
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=width||y>=height) return;
    const int p=y*width+x;
    if(v_loss[p]==0.f) return;
    V3 dx,dy,c; float length;
    if(!normal_data(maps,R,K,width,height,x,y,dx,dy,c,length)) return;
    const float reciprocal=1.f/fmaxf(length,1e-12f);
    const float factor=-v_loss[p]*maps[6*p+1];
    const V3 n={c.x*reciprocal,c.y*reciprocal,c.z*reciprocal};
    V3 vc={factor*maps[6*p+3],factor*maps[6*p+4],factor*maps[6*p+5]};
    if(length>=1e-12f) {
        const float projection=vc.x*n.x+vc.y*n.y+vc.z*n.z;
        vc={vc.x-projection*n.x,vc.y-projection*n.y,vc.z-projection*n.z};
    }
    vc={vc.x*reciprocal,vc.y*reciprocal,vc.z*reciprocal};
    const V3 vx=cross3(dy,vc),vy=cross3(vc,dx);
    const V3 down=world_ray(x,y+1,R,K),up=world_ray(x,y-1,R,K);
    const V3 right=world_ray(x+1,y,R,K),left=world_ray(x-1,y,R,K);
    atomicAdd(v_maps+6*(p+width),vx.x*down.x+vx.y*down.y+vx.z*down.z);
    atomicAdd(v_maps+6*(p-width),-(vx.x*up.x+vx.y*up.y+vx.z*up.z));
    atomicAdd(v_maps+6*(p+1),vy.x*right.x+vy.y*right.y+vy.z*right.z);
    atomicAdd(v_maps+6*(p-1),-(vy.x*left.x+vy.y*left.y+vy.z*left.z));
    v_maps[6*p+3]=factor*n.x; v_maps[6*p+4]=factor*n.y; v_maps[6*p+5]=factor*n.z;
}
'''


@lru_cache(maxsize=None)
def _kernels(device_index):
    with torch.cuda.device(device_index):
        # A default stream handle alone may not initialize the primary context.
        _context_storage = torch.empty(1, device=torch.device("cuda", device_index))
        torch.cuda.current_stream(device_index)
        return load_surface_kernels(
            _SOURCE, ("surface_forward", "surface_backward", "normal_forward", "normal_backward"),
            torch.__file__, torch.version.cuda, torch.cuda.get_device_capability(device_index),
        )


def prepare_surface_cuda(device):
    """Initialize before the timed training loop, never at the first loss warm-up."""
    with torch.cuda.device(device):
        _kernels(device.index)


def _launch(kernel, tensors, scalars, outputs, width, height, device):
    # Pass PyTorch storage directly on its own stream; no copies or stream handoff.
    arguments = tuple(C.c_void_p(t.data_ptr() if t is not None else 0) for t in tensors)
    arguments += scalars
    arguments += tuple(C.c_void_p(t.data_ptr() if t is not None else 0) for t in outputs)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        kernel(((width + 15) // 16, (height + 15) // 16), (16, 16), arguments, stream=stream)


class _SurfaceRasterization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, transforms, rgb, opacity, normals,
                reference, depths, densify, offsets, ids, width, height, absgrad,
                with_geometry, near_plane, ndc):
        forward = _kernels(xy.device.index)[0]
        inputs = tuple(t.contiguous() for t in (xy, transforms, rgb, opacity, normals,
                                               reference, depths, densify, offsets, ids))
        xy, transforms, rgb, opacity, normals, reference, depths, densify, offsets, ids = inputs
        image = xy.new_empty((1, height, width, 3))
        alpha = xy.new_empty((1, height, width, 1))
        maps = xy.new_empty((1, height, width, 6))
        moments = xy.new_empty((height, width, 4))
        last = torch.empty((height, width), device=xy.device, dtype=torch.int32)
        median = torch.empty_like(last)
        scalars = (C.c_int(width), C.c_int(height), C.c_int(ids.numel()),
                   C.c_float(ndc), C.c_float(near_plane), C.c_int(with_geometry))
        _launch(forward, inputs[:6] + (offsets, ids), scalars,
                (image, alpha, maps, moments, last, median), width, height, xy.device)
        ctx.save_for_backward(*inputs, maps, moments, last, median)
        ctx.width, ctx.height, ctx.absgrad, ctx.scalars = width, height, absgrad, scalars
        ctx.set_materialize_grads(False)
        return image, alpha, maps

    @staticmethod
    def backward(ctx, v_image, v_alpha, v_maps):
        backward = _kernels(ctx.saved_tensors[0].device.index)[1]
        inputs = ctx.saved_tensors[:-4]
        maps, moments, last, median = ctx.saved_tensors[-4:]
        xy, transforms, rgb, opacity, normals, reference, depths, densify, offsets, ids = inputs
        gradients = tuple(torch.zeros_like(t) for t in inputs[:5])
        v_xy, v_transform, v_rgb, v_opacity, v_normals = gradients
        v_abs = torch.zeros_like(xy) if ctx.absgrad else None
        upstream = tuple(t.contiguous() if t is not None else None for t in (v_image, v_alpha, v_maps))
        _launch(backward, inputs[:6] + (offsets, ids), ctx.scalars,
                (maps, moments, last, median, *upstream, *gradients, v_abs), ctx.width, ctx.height, xy.device)
        if ctx.absgrad:
            densify.absgrad = v_abs * depths[..., None]
        v_densify = torch.stack((v_transform[..., 0, 2], v_transform[..., 1, 2]), dim=-1) * depths[..., None]
        return (*gradients, None, None, v_densify, None, None, None, None, None, None, None, None)


def rasterize_surface(xy, transforms, rgb, opacity, normals,
                      reference, depths, densify, offsets, ids, width, height, absgrad=False,
                      with_geometry=True, near_plane=_NEAR_PLANE,
                      ndc=-NDC_FAR * NDC_NEAR / (NDC_FAR - NDC_NEAR)):
    """Single-camera, unpacked float32 interface to the fused forward/backward."""
    if width < 1 or height < 1 or xy.ndim != 3 or xy.shape[0] != 1 or xy.shape[1] < 1:
        raise ValueError("Surface rasterization requires one nonempty camera and positive image dimensions")
    count = xy.shape[1]
    shapes = ((1, count, 2), (1, count, 3, 3), (1, count, 3), (1, count),
              (1, count, 3), (), (1, count),
              (1, count, 2), (1, (height + 15) // 16, (width + 15) // 16))
    tensors = (xy, transforms, rgb, opacity, normals, reference, depths, densify, offsets)
    for index, (tensor, shape) in enumerate(zip(tensors, shapes)):
        expected_dtype = torch.int32 if index == 8 else torch.float32
        if tuple(tensor.shape) != shape or tensor.dtype != expected_dtype or tensor.device != xy.device or not tensor.is_cuda:
            raise ValueError(f"Invalid surface rasterization input {index}: expected CUDA {expected_dtype} {shape}")
    if ids.ndim != 1 or ids.dtype != torch.int32 or ids.device != xy.device:
        raise ValueError("Surface tile indices must be an int32 vector on the rendering device")
    if not math.isfinite(near_plane) or near_plane <= 0 or not math.isfinite(ndc) or ndc >= 0:
        raise ValueError("Surface depth units must define a positive near plane and negative NDC coefficient")
    return _SurfaceRasterization.apply(*tensors, ids, width, height, absgrad, with_geometry, near_plane, ndc)


class _NormalConsistency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, maps, rotation, K):
        maps, rotation, K = maps.contiguous(), rotation.contiguous(), K.contiguous()
        height, width = maps.shape[1:3]
        loss = maps.new_empty((1, height, width, 1))
        scalars = (C.c_int(width), C.c_int(height))
        _launch(_kernels(maps.device.index)[2], (maps, rotation, K), scalars,
                (loss,), width, height, maps.device)
        ctx.save_for_backward(maps, rotation, K)
        ctx.width, ctx.height, ctx.scalars = width, height, scalars
        return loss

    @staticmethod
    def backward(ctx, v_loss):
        maps, rotation, K = ctx.saved_tensors
        gradient = torch.zeros_like(maps)
        upstream = v_loss.contiguous()
        _launch(_kernels(maps.device.index)[3], (maps, rotation, K), ctx.scalars,
                (upstream, gradient), ctx.width, ctx.height, maps.device)
        return gradient, None, None


def normal_consistency(maps, view, intrinsics):
    """Exact central-difference normal objective, including the detached alpha mask."""
    if (not maps.is_cuda or maps.dtype != torch.float32 or maps.ndim != 4
            or maps.shape[0] != 1 or maps.shape[-1] != 6):
        raise ValueError("Normal consistency requires a single CUDA float32 six-channel geometry image")
    for tensor, shape in ((view, (1, 4, 4)), (intrinsics, (1, 3, 3))):
        if tensor.device != maps.device or tensor.dtype != maps.dtype or tuple(tensor.shape) != shape:
            raise ValueError("Normal consistency requires camera matrices on the rendering device")
    return _NormalConsistency.apply(maps, view[0, :3, :3], intrinsics).mean()


# Gaussian training and exports

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
    moments = F.conv2d(moments, horizontal, padding=(0, 5), groups=channels)
    moments = F.conv2d(moments, vertical, padding=(5, 0), groups=channels)
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
def _prune_surfels(splats, optimizers, state, opacity_threshold, allow_size, label,
                    screen_radii=None):
    from gsplat.strategy.ops import remove

    finite = torch.ones(len(splats["means"]), dtype=torch.bool, device=splats["means"].device)
    for value in splats.values():
        finite &= torch.isfinite(value).reshape(len(finite), -1).all(-1)
    scales = splats["scales"].exp().amax(-1)
    finite &= torch.isfinite(scales)
    opacity = splats["opacities"].sigmoid().flatten()
    faint = finite & (opacity < opacity_threshold)
    large = finite & (scales > 0.1 * state["scene_scale"]) if allow_size else torch.zeros_like(finite)
    screen = (finite & (screen_radii > 0.15)
              if allow_size and screen_radii is not None else torch.zeros_like(finite))
    discard = ~finite | faint | large | screen
    if discard.all():
        raise RuntimeError("2DGS cleanup would remove all surfels; inspect the scene scale and prune_opacity")
    counts = torch.stack(((~finite).sum(), faint.sum(), large.sum(), screen.sum(), discard.sum())).tolist()
    if counts[-1]:
        remove(params=splats, optimizers=optimizers, state=state, mask=discard)
    kept = ~discard
    opacity_range = opacity[kept].aminmax()
    scale_range = scales[kept].aminmax()
    print(f"[VNCCS Gaussian] {label}: removed={counts[-1]}, nonfinite={counts[0]}, "
          f"opacity={counts[1]}, world_size={counts[2]}, screen_size={counts[3]} (reasons may overlap); "
          f"remaining={len(splats['means'])}, opacity_range={opacity_range.min.item():.5g}..{opacity_range.max.item():.5g}, "
          f"max_axis_range={scale_range.min.item():.5g}..{scale_range.max.item():.5g}")
    return counts[-1]


@torch.no_grad()
def _export_gaussians(splats, optimizers, state, output_path, prune_opacity=None, allow_size=False):
    from gsplat import export_splats
    from gsplat.strategy.ops import remove

    if prune_opacity is not None:
        _prune_surfels(splats, optimizers, state, prune_opacity, allow_size, "Final cleanup")
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


def _initial_scales(points, depth_unit=1.0):
    import cv2

    points = np.ascontiguousarray(points, dtype=np.float32)
    index = cv2.flann_Index(points, {"algorithm": 1, "trees": 1})
    neighbors, _ = index.knnSearch(points, 4, params={"checks": -1})
    offsets = points[neighbors[:, 1:]] - points[:, None]
    mean_squared_distance = np.square(offsets).sum(axis=-1).mean(axis=-1)
    return (0.5 * np.log(np.maximum(mean_squared_distance, 1e-7 * depth_unit**2))).astype(np.float32)


def _refinement_schedule(training_steps, start, stop, reset):
    """Minus one selects the original 500/15000/3000 proportions; explicit steps stay explicit."""
    return (
        int(training_steps / 60) if start == -1 else start,
        min(training_steps, int(training_steps / 2) if stop == -1 else stop),
        max(1, int(training_steps / 10)) if reset == -1 else reset,
    )


def _strategy(
    training_steps, refine_start_iter, refine_stop_iter, refine_every,
    reset_every, grow_grad2d, prune_opacity, use_2dgs=False,
):
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    refine_start_iter, refine_stop_iter, reset_every = _refinement_schedule(
        training_steps, refine_start_iter, refine_stop_iter, reset_every,
    )

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
                return False
            self._update_state(params, state, info, packed=packed)
            changed = False
            if (step > self.refine_start_iter and step % self.refine_every == 0
                    and step % self.reset_every >= self.pause_refine_after_reset):
                cloned, divided = self._grow_gs(params, optimizers, state, step)
                pruned = _prune_surfels(
                    params, optimizers, state, self.prune_opa,
                    step > self.size_prune_start, f"Cleanup {step}", state["radii"],
                )
                state["grad2d"].zero_()
                state["count"].zero_()
                state["radii"].zero_()
                print(f"[VNCCS Gaussian] 2DGS refine {step}: cloned={cloned}, "
                      f"split={divided}, pruned={pruned}, surfels={len(params['means'])}")
                changed = True
            if step > 0 and step % self.reset_every == 0:
                from gsplat.strategy.ops import reset_opa

                reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
                print(f"[VNCCS Gaussian] 2DGS opacity reset at step {step}")
                changed = True
            return changed

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
    strategy = BudgetedDefaultStrategy(**options)
    strategy.size_prune_start = max(refine_start_iter, int(training_steps * 0.1))
    return strategy


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
                depth_unit=frame.get("depth_unit", 1.0),
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
    point_splats=False,
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Gaussian training requires an NVIDIA CUDA build of PyTorch")
    if use_2dgs and optimizer_mode == "visible_adam":
        print("[VNCCS Gaussian] Legacy visible_adam selected: using fused_adam for exact Adam updates in 2DGS")
        optimizer_mode = "fused_adam"
    try:
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy.ops import remove
        if optimizer_mode == "visible_adam":
            from gsplat.optimizers import SelectiveAdam
    except ImportError as error:
        raise RuntimeError("gsplat 1.5.3 is required for Gaussian training") from error

    device = torch.device("cuda", torch.cuda.current_device())
    if use_2dgs:
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
    depth_unit = frames[0].get("depth_unit", 1.0)
    if use_2dgs:
        legacy_masks = 0
        for frame in frames:
            cached = training_frames[frame["name"]]
            original_alpha = cached.get("alpha_uneroded") if isinstance(cached, dict) else None
            if original_alpha is not None:
                frame["rgba"] = frame["rgba"].copy()
                frame["rgba"][..., 3] = original_alpha
            else:
                legacy_masks += 1
        if legacy_masks:
            print(f"[VNCCS Gaussian] {legacy_masks} cached frames have legacy eroded alpha; "
                  "keeping the cached cameras and masks. Newly reconstructed scenes preserve the original 2DGS silhouette.")
    if use_depth_loss:
        prepare_depth_targets(reconstruction, training_frames, points, frames, interrupt)
    torch.manual_seed(seed)
    initial_scales = _initial_scales(points, depth_unit)
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
    sh_interval = max(1, training_steps // 30) if use_2dgs else 1000
    fixed_background = torch.zeros((1, 1, 1, 3), device=device) if use_2dgs else None
    print(
        f"[VNCCS Gaussian] Training: frames={len(frames)}, initial_splats={count}, "
        f"steps={training_steps}, cap={max_splats}, scene_scale={scene_scale:.6g}, "
        f"means_lr={learning_rates['means']:.6g}, optimizer={optimizer_mode}"
    )
    print(
        f"[VNCCS Gaussian] ADC: start={strategy.refine_start_iter}, "
        f"stop={strategy.refine_stop_iter}, every={strategy.refine_every}, "
        f"grow_grad2d={strategy.grow_grad2d:g}, reset_every={strategy.reset_every}; "
        f"size_prune_after={strategy.size_prune_start}; "
        f"SSIM={ssim_weight:g}, alpha={alpha_weight:g}, "
        f"scale_regularization={use_scale_regularization}, antialiased={antialiased}, "
        f"depth_supervision={use_depth_loss}, depth_weight={DEPTH_LOSS_WEIGHT:g}; "
        f"SH={sh_degree} ({'progressive' if use_2dgs else 'from step 1'}); full-resolution shuffled frames"
    )
    print(f"[VNCCS Gaussian] Output: {output_path}")
    if use_2dgs:
        print(
            f"[VNCCS Gaussian] 2DGS: {'circular splats with shared radius' if point_splats else 'elliptical splats with two tangential scales'}; planar splitting; "
            f"perspective three-sigma projection/bounds; gsplat tile sorting; "
            f"fused CUDA RGB/geometry forward and backward; no feature images or hit lists; "
            f"ray-plane median depth (no stock center-depth interpolation); "
            f"distortion_l2_ndc={distortion_weight:g} after {int(training_steps * 0.1)} steps, "
            f"normal={normal_weight:g} after {int(training_steps * 7 / 30)} steps; "
            f"SH increases every {sh_interval} steps; geometry uses all pixels and contributors every step. "
            f"COLMAP depth unit={depth_unit:.6g}; NDC and clipping preserve pre-normalization depth units. "
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
        background = fixed_background if use_2dgs else torch.rand((1, 1, 1, 3), device=device)
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
                min(step // sh_interval, sh_degree), strategy.absgrad,
                with_geometry=need_geometry, depth_unit=depth_unit,
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
        if use_2dgs and point_splats:
            _tie_planar_scale_gradients(splats["scales"])
        refined = (strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
                   if use_2dgs else False)
        visibility = (info["radii"] > 0).all(dim=-1).any(dim=0)
        for optimizer in optimizers.values():
            if not refined:
                if optimizer_mode == "visible_adam":
                    optimizer.step(visibility)
                else:
                    optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if not use_2dgs:
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
    _export_gaussians(splats, optimizers, state, output_path,
                      prune_opacity if use_2dgs else None,
                      use_2dgs and training_steps > strategy.size_prune_start)
    result = _comfy_splat(splats)
    print(f"[VNCCS Gaussian] Saved {len(splats['means'])} splats to {output_path}")
    preview_degree = min((training_steps - 1) // sh_interval, sh_degree) if use_2dgs else sh_degree
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
                "sh_degree": ("INT", {"default": 3, "min": 0, "max": 3, "step": 1}),
                "ssim_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.01}),
                "use_scale_regularization": ("BOOLEAN", {"default": False}),
                "antialiased": ("BOOLEAN", {"default": True, "tooltip": "3DGS only. 2DGS always uses its native low-pass filter."}),
                "optimizer_mode": (["fused_adam", "adam", "visible_adam"], {"default": "fused_adam", "tooltip": "2DGS uses exact Adam; legacy visible_adam maps to fused_adam in this mode."}),
                "refine_start_iter": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "-1: automatic, 1/60 of training. Non-negative values are explicit steps."}),
                "refine_stop_iter": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "-1: automatic, half of training. Non-negative values are explicit steps."}),
                "refine_every": ("INT", {"default": 100, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "reset_every": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "-1: automatic, every 10% of training. Positive values are explicit intervals; 0 is invalid. Size pruning is independent."}),
                "grow_grad2d": ("FLOAT", {"default": 0.0002, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "prune_opacity": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
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
                "point_splats": (
                    "BOOLEAN", {
                        "default": False,
                        "tooltip": "2DGS only: train circular surface splats with one shared adaptive tangential radius instead of elongated ellipses. Splitting and export preserve this shape. Tilted discs still look narrow edge-on. Ignored when use_2dgs is disabled.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "SPLAT") + ("IMAGE",) * 8 + ("STRING",)
    RETURN_NAMES = (
        "gaussian_file", "splat",
        "top_0", "source_0",
        "top_90", "source_90",
        "top_180", "source_180",
        "top_270", "source_270",
        "native_2dgs_file",
    )
    FUNCTION = "train"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    def train(
        self, scene, training_steps=8_000, max_splats=2_000_000, sh_degree=3,
        ssim_weight=0.2, alpha_weight=0.1, use_scale_regularization=False,
        antialiased=True, optimizer_mode="fused_adam", refine_start_iter=-1,
        refine_stop_iter=-1, refine_every=100, reset_every=-1,
        grow_grad2d=0.0002, prune_opacity=0.05, seed=42, use_depth_loss=False,
        use_2dgs=False, distortion_weight=DISTORTION_WEIGHT, normal_weight=NORMAL_WEIGHT,
        point_splats=False,
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
        if refine_every < 1 or (reset_every != -1 and reset_every < 1):
            raise ValueError("Refinement interval must be positive; reset interval must be positive or -1 (automatic)")
        if refine_start_iter < -1 or refine_stop_iter < -1:
            raise ValueError("Refinement steps must be non-negative or -1 (automatic)")
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
            use_2dgs, distortion_weight, normal_weight, point_splats,
        )
        native_path = str(Path(output_path).with_name(Path(output_path).stem + "_2dgs.ply")) if use_2dgs else ""
        return (output_path, *result, native_path)


# Textured mesh extraction from native planar splats

_MESH_EXTRACT_MAX_BYTES = 1 << 30


def _mesh_tsdf_volume(voxel_size, block_count=1000):
    import open3d as o3d

    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32,) * 3, attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_size, block_resolution=16, block_count=block_count,
        device=o3d.core.Device("CPU:0"),
    )


def _mesh_extract_surface(volume, voxel_size, interrupt):
    """Extract every TSDF cell with bounded scratch buffers and welded chunk seams."""
    import open3d as o3d

    table = volume.hashmap()
    active = table.active_buf_indices().numpy().astype(np.int64)
    keys = table.key_tensor().numpy()[active]
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, active = keys[order], active[order]
    source_indices = {tuple(key): int(index) for key, index in zip(keys, active)}
    attributes = {name: volume.attribute(name).numpy() for name in ("tsdf", "weight", "color")}
    resolution = attributes["tsdf"].shape[1]
    # Open3D's marching-cubes ArrayIndexer uses int32 BYTE offsets, not size_t.
    # The largest native buffer reserves 3 vertices/cell, 3 triangles/vertex, 3 indices/triangle.
    bytes_per_block = resolution**3 * 3 * 3 * 3 * np.dtype(np.int32).itemsize
    batch_size = _MESH_EXTRACT_MAX_BYTES // (27 * bytes_per_block)
    if batch_size < 1:
        raise ValueError("A TSDF block exceeds the safe marching-cubes scratch-buffer budget")
    neighbors = np.array([(x, y, z) for x in (-1, 0, 1)
                          for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=np.int32)
    row_type = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    block_width = float(np.float32(voxel_size)) * resolution
    mesh = o3d.geometry.TriangleMesh()
    total = math.ceil(len(keys) / batch_size)
    print(f"[VNCCS PLY2MESH] Extracting {len(keys)} blocks in {total} batches; "
          f"scratch budget={_MESH_EXTRACT_MAX_BYTES // (1 << 20)} MiB", flush=True)
    for batch, start in enumerate(range(0, len(keys), batch_size), 1):
        interrupt()
        core = keys[start:start + batch_size]
        halo = np.unique((core[:, None, :] + neighbors[None]).reshape(-1, 3), axis=0)
        indices = np.fromiter((source_indices.get(tuple(key), -1) for key in halo), dtype=np.int64)
        present = indices >= 0
        halo, indices = halo[present], indices[present]
        section = _mesh_tsdf_volume(voxel_size, len(halo))
        target_indices, _ = section.hashmap().activate(o3d.core.Tensor(halo))
        target_indices = target_indices.numpy().astype(np.int64)
        for name, values in attributes.items():
            section.attribute(name).numpy()[target_indices] = values[indices]
        piece = section.extract_triangle_mesh(weight_threshold=0.5).to_legacy()
        del section
        if len(piece.triangles):
            # Each triangle has one owning block; halo triangles belong to other batches.
            points = np.asarray(piece.vertices)[np.asarray(piece.triangles)].mean(axis=1)
            owners = np.ascontiguousarray(np.floor(points / block_width), dtype=np.int64)
            core_rows = np.ascontiguousarray(core, dtype=np.int64).view(row_type).reshape(-1)
            owned = np.isin(owners.view(row_type).reshape(-1), core_rows)
            piece.remove_triangles_by_mask(~owned)
            piece.remove_unreferenced_vertices()
            mesh += piece
        print(f"[VNCCS PLY2MESH] Mesh extraction: {batch}/{total}; "
              f"blocks={len(core)}, halo_blocks={len(halo)}, triangles={len(mesh.triangles)}", flush=True)
    # All batches interpolate the same voxel values; identical seam positions weld exactly.
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_ply_path(value):
    import folder_paths

    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("Connect native_2dgs_file from VNCCS Gaussian Trainer")
    # Trainer outputs are absolute; ComfyUI's annotated resolver accepts relative names only.
    path = Path(value)
    if not path.is_absolute():
        path = Path(folder_paths.get_annotated_filepath(value))
    path = path.absolute()
    for root in (folder_paths.get_output_directory(), folder_paths.get_input_directory(),
                 folder_paths.get_temp_directory()):
        try:
            relative = path.relative_to(Path(root).absolute())
        except ValueError:
            continue
        checked = safe_join_under(root, str(relative))
        if Path(checked).suffix.lower() != ".ply" or not Path(checked).is_file():
            raise ValueError("ply_path must point to an existing PLY file")
        return checked
    raise ValueError("PLY files must be inside ComfyUI input, output or temp directories")


def _mesh_load_surfels(path):
    from plyfile import PlyData

    ply = PlyData.read(path)
    if "vertex" not in ply or not len(ply["vertex"]):
        raise ValueError("The PLY has no vertices")
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1"}
    required.update(f"rot_{i}" for i in range(4))
    required.update(f"f_dc_{i}" for i in range(3))
    if not required.issubset(names) or "scale_2" in names:
        raise ValueError("PLY2MESH requires native 2DGS PLY: connect native_2dgs_file, not gaussian_file")
    rest = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name[7:]))
    degree = math.isqrt(1 + len(rest) // 3) - 1
    if (not 0 <= degree <= 3 or len(rest) != 3 * ((degree + 1)**2 - 1)
            or rest != [f"f_rest_{i}" for i in range(len(rest))]):
        raise ValueError("The PLY contains unsupported or incomplete spherical harmonics")

    def columns(fields):
        return np.column_stack([vertices[field] for field in fields]).astype(np.float32)

    count = len(vertices)
    splats = {
        "means": columns(("x", "y", "z")),
        "scales": columns(("scale_0", "scale_1")),
        "quats": columns([f"rot_{i}" for i in range(4)]),
        "opacities": np.asarray(vertices["opacity"], dtype=np.float32).copy(),
        "sh0": columns([f"f_dc_{i}" for i in range(3)])[:, None, :],
        "shN": (columns(rest).reshape(count, 3, -1).transpose(0, 2, 1)
                if rest else np.empty((count, 0, 3), dtype=np.float32)),
    }
    if not all(np.isfinite(value).all() for value in splats.values()):
        raise ValueError("The native PLY contains non-finite splat parameters")
    lengths = np.linalg.norm(splats["quats"].astype(np.float64), axis=1, keepdims=True)
    if (lengths <= 1e-12).any() or (np.abs(splats["scales"]) > 80).any():
        raise ValueError("The native PLY contains invalid rotations or scales")
    splats["quats"] = (splats["quats"] / lengths).astype(np.float32)
    return splats, degree


def _mesh_source(frame, cache):
    entry = cache[frame["name"]]
    source = BytesIO(entry["source_png"]) if "source_png" in entry else entry["source_path"]
    with Image.open(source) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _mesh_project(points, frame):
    camera = points @ frame["view"][:3, :3].T + frame["view"][:3, 3]
    projected = camera @ frame["K"].T
    # Our rasterizer samples (x + .5, y + .5); arrays and Open3D RGBD use integer centers.
    pixels = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-12) - 0.5
    return pixels, camera[:, 2]


def _mesh_unproject(x, y, depth, frame):
    pixels = np.column_stack((x + 0.5, y + 0.5, np.ones_like(x)))
    camera = (pixels @ np.linalg.inv(frame["K"]).T) * depth[:, None]
    return (camera - frame["view"][:3, 3]) @ frame["view"][:3, :3]


def _mesh_consistent_depth(depth, frame, neighbors, voxel_size, tolerance, interrupt):
    """Require one other view to confirm each retained surface sample."""
    y, x = np.nonzero(depth > 0)
    result = np.zeros_like(depth)
    for start in range(0, len(x), 65_536):
        interrupt()
        xx, yy = x[start:start + 65_536], y[start:start + 65_536]
        reference = depth[yy, xx]
        points = _mesh_unproject(xx, yy, reference, frame)
        supported = np.zeros(len(xx), dtype=bool)
        for other, other_depth in neighbors:
            pixels, z = _mesh_project(points, other)
            height, width = other_depth.shape
            inside = ((z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1))
            indices = np.flatnonzero(inside & ~supported)
            if not len(indices):
                continue
            sampled_xy = np.rint(pixels[indices]).astype(np.int64)
            sampled = other_depth[sampled_xy[:, 1], sampled_xy[:, 0]]
            agrees = (sampled > 0) & (np.abs(sampled - z[indices]) <= np.maximum(2 * voxel_size, tolerance * z[indices]))
            back = _mesh_unproject(sampled_xy[:, 0], sampled_xy[:, 1], sampled, other)
            back_pixels, _ = _mesh_project(back, frame)
            agrees &= np.linalg.norm(back_pixels - np.column_stack((xx[indices], yy[indices])), axis=1) <= 1.5
            supported[indices[agrees]] = True
        result[yy[supported], xx[supported]] = reference[supported]
    return result


def _mesh_integrate_frame(volume, frame, depth, rgb, voxel_size):
    import open3d as o3d

    y, x = np.nonzero(depth > 0)
    if not len(x):
        return 0
    # Allocate blocks from all pixel-center rays, avoiding legacy TSDF's implicit /1000.
    points = _mesh_unproject(x, y, depth[y, x], frame).astype(np.float32)
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points))
    blocks = volume.compute_unique_block_coordinates(cloud, trunc_voxel_multiplier=4.0)
    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth, dtype=np.float32)))
    # Tensor TSDF requires a float depth/float RGB pair; RGB floats are in [0, 1].
    color_image = o3d.t.geometry.Image(o3d.core.Tensor(rgb.astype(np.float32) / 255.0))
    # Tensor integration floors projected coordinates; our K already addresses pixel cells.
    intrinsic = o3d.core.Tensor(frame["K"].astype(np.float64))
    extrinsic = o3d.core.Tensor(frame["view"].astype(np.float64))
    volume.integrate(
        blocks, depth_image, color_image, intrinsic, extrinsic,
        depth_scale=1.0, depth_max=float(depth.max()) + 4 * voxel_size,
        trunc_voxel_multiplier=4.0,
    )
    return len(blocks)


def _mesh_pack_charts(extents, size, padding=4):
    """Pack chart rectangles, never the high-poly surface, on the CPU."""
    extents = np.asarray(extents, dtype=np.float64)
    if extents.ndim != 2 or extents.shape[1] != 2 or not np.isfinite(extents).all() or (extents < 0).any():
        raise ValueError("UV chart bounds must be finite and non-negative")
    rotate = extents[:, 1] > extents[:, 0]
    spans = extents.copy()
    spans[rotate] = spans[rotate, ::-1]

    def pack(scale):
        dimensions = np.ceil(spans * scale).astype(np.int64) + 1 + 2 * padding
        order = np.lexsort((-dimensions[:, 0], -dimensions[:, 1]))
        offsets = np.empty(dimensions.shape, dtype=np.float64)
        x = y = row_height = 0
        for index in order:
            width, height = dimensions[index]
            if width > size or height > size:
                return None
            if x + width > size:
                x, y, row_height = 0, y + row_height, 0
            if y + height > size:
                return None
            offsets[index] = (x + padding + .5, y + padding + .5)
            x += width
            row_height = max(row_height, height)
        return offsets

    if pack(0) is None:
        raise ValueError(f"{size}x{size} cannot hold {len(extents)} UV islands with {padding}-pixel gutters; increase texture_size")
    low, high = 0., size / max(float(spans.max()), 1.e-12)
    for _ in range(24):
        scale = (low + high) / 2
        if pack(scale) is None:
            high = scale
        else:
            low = scale
    return pack(low).astype(np.float32), rotate, low


def _mesh_chart_bounds(values, charts, count):
    lower = values.new_full((count, values.shape[1]), float("inf"))
    upper = values.new_full((count, values.shape[1]), -float("inf"))
    indices = charts.long()[:, None].expand_as(values)
    lower.scatter_reduce_(0, indices, values, reduce="amin", include_self=True)
    upper.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
    return lower, upper


def _mesh_save_array(path, values):
    with atomic_output_path(str(path)) as temporary:
        with open(temporary, "wb") as handle:
            np.save(handle, values, allow_pickle=False)


def _mesh_gpu_frame(vertices, faces, frame, cache, directory, index, voxel_size, interrupt):
    from .helpers.gaussian_mesh_cuda import depth_map

    device = vertices.device
    matrix = torch.as_tensor(frame["K"] @ frame["view"][:3], device=device).float().contiguous()
    center = torch.as_tensor(-(frame["view"][:3, :3].T @ frame["view"][:3, 3]), device=device).float().contiguous()
    reference = np.load(Path(directory) / f"filtered_{index}.npy", allow_pickle=False)
    alpha = cache[frame["name"]]["alpha_uneroded"]
    if (reference.ndim != 2 or alpha.shape != reference.shape or alpha.dtype != np.uint8
            or not np.isfinite(reference).all() or (reference < 0).any()):
        raise ValueError("Saved mesh depth or foreground mask is invalid")
    height, width = reference.shape
    z_path = Path(directory) / f"mesh_z_{index}.npy"
    if z_path.is_file():
        z = np.load(z_path, allow_pickle=False)
        if z.shape != reference.shape or z.dtype != np.float32 or np.isnan(z).any() or (z < 0).any():
            raise ValueError("Saved mesh visibility buffer is invalid; original checkpoint was preserved")
        z = torch.as_tensor(z, device=device).contiguous()
    else:
        z = depth_map(vertices, faces, matrix, width, height, interrupt)
        _mesh_save_array(z_path, z.cpu().numpy())
    return (matrix, center, z, torch.as_tensor(reference, device=device).float().contiguous(),
            torch.as_tensor(alpha, device=device, dtype=torch.uint8).contiguous(), width, height)


def _mesh_build_uv(vertices, faces, frames, cache, directory, size, voxel_size, tolerance, interrupt):
    from .helpers.gaussian_mesh_cuda import color_overlaps, launch, uv_map

    device = vertices.device
    selected = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    camera_count = len(selected)
    matrices = [frames[i]["K"] @ frames[i]["view"][:3] for i in selected]
    for axis in range(3):
        for sign in (1, -1):
            matrix = np.zeros((3, 4), dtype=np.float32)
            matrix[0, (axis + 1) % 3], matrix[1, (axis + 2) % 3] = 1, sign
            matrix[2, 3] = 1
            matrices.append(matrix)
    matrices = torch.as_tensor(np.stack(matrices), device=device).float()
    modes = torch.empty(len(faces), device=device, dtype=torch.int32)
    areas = torch.empty(len(faces), device=device)
    centers = torch.empty((len(faces), 3), device=device)
    for start in range(0, len(faces), 262_144):
        interrupt()
        triangles = vertices[faces[start:start + 262_144].long()]
        normal = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        axis = normal.abs().argmax(dim=1)
        modes[start:start + len(triangles)] = camera_count + axis * 2 + (normal.gather(1, axis[:, None])[:, 0] < 0)
        areas[start:start + len(triangles)] = normal.norm(dim=1) * .5
        centers[start:start + len(triangles)] = triangles.mean(dim=1)
    scores = torch.zeros(len(faces), device=device)
    for mode, index in enumerate(selected):
        frame = frames[index]
        matrix, center, depth, reference, alpha, width, height = _mesh_gpu_frame(
            vertices, faces, frame, cache, directory, index, voxel_size, interrupt)
        launch("choose_view", (vertices, faces, matrix, center, depth, reference, alpha, scores, modes), len(faces),
               (C.c_int(width), C.c_int(height), C.c_int(mode), C.c_float(voxel_size), C.c_float(tolerance),
                C.c_float(frame["K"][0, 0] * frame["K"][1, 1])), interrupt)
        print(f"[VNCCS PLY2MESH] GPU chart visibility: {mode + 1}/{camera_count}", flush=True)
    del scores, matrix, center, depth, reference, alpha
    raw_uv = vertices.new_empty((len(faces), 3, 2))
    for start in range(0, len(faces), 262_144):
        interrupt()
        ids = modes[start:start + 262_144].long()
        points = vertices[faces[start:start + len(ids)].long()]
        m = matrices[ids]
        # Explicit FP32 products avoid TF32 rounding and keep shared UV vertices identical.
        projected = (points[:, :, None, :] * m[:, None, :, :3]).sum(dim=-1) + m[:, None, :, 3]
        raw_uv[start:start + len(ids)] = projected[:, :, :2] / projected[:, :, 2:].clamp_min(1.e-12)
    extent = (vertices.amax(dim=0) - vertices.amin(dim=0)).max().clamp_min(1.e-12)
    cells = ((centers - vertices.amin(dim=0)) / (extent / 8)).floor().clamp(0, 7).int()
    groups = (((cells[:, 0] * 8 + cells[:, 1]) * 8 + cells[:, 2]) * len(matrices) + modes).int()
    signed_uv_area = ((raw_uv[:, 1, 0] - raw_uv[:, 0, 0]) * (raw_uv[:, 2, 1] - raw_uv[:, 0, 1])
                      - (raw_uv[:, 1, 1] - raw_uv[:, 0, 1]) * (raw_uv[:, 2, 0] - raw_uv[:, 0, 0]))
    # A camera projection has two sheets at a fold; opposite windings must never share a patch.
    groups = groups * 2 + (signed_uv_area < 0).int()
    area_uv = signed_uv_area.abs() * .5
    del signed_uv_area
    # Disconnected fragments may share one projection patch when their UVs do not overlap.
    charts = torch.unique(groups, sorted=True, return_inverse=True)[1].to(torch.int32)
    print(f"[VNCCS PLY2MESH] GPU projection patches: {int(charts.max()) + 1} for {len(faces)} triangles", flush=True)
    del cells, centers, groups, matrices, modes
    iteration = 0
    while True:
        interrupt()
        iteration += 1
        count = int(charts.max()) + 1
        lower, _ = _mesh_chart_bounds(raw_uv.amin(dim=1), charts, count)
        _, upper = _mesh_chart_bounds(raw_uv.amax(dim=1), charts, count)
        chart_area = torch.zeros(count, device=device).scatter_add_(0, charts.long(), areas)
        chart_uv_area = torch.zeros_like(chart_area).scatter_add_(0, charts.long(), area_uv)
        density = (chart_area / chart_uv_area.clamp_min(1.e-30)).sqrt()
        offsets, rotated, scale = _mesh_pack_charts(((upper - lower) * density[:, None]).cpu().numpy(), size)
        offsets = torch.as_tensor(offsets, device=device)
        rotated = torch.as_tensor(rotated, device=device)
        uv = torch.empty_like(raw_uv)
        for start in range(0, len(faces), 262_144):
            interrupt()
            labels = charts[start:start + 262_144].long()
            value = (raw_uv[start:start + len(labels)] - lower[labels, None]) * density[labels, None, None] * scale
            value = torch.where(rotated[labels, None, None], value.flip(-1), value)
            uv[start:start + len(labels)] = (value + offsets[labels, None]) / size
        owner, conflicts = uv_map(uv, charts, size, interrupt)
        bad = int(torch.count_nonzero(conflicts))
        print(f"[VNCCS PLY2MESH] GPU UV pass {iteration}: islands={count}, overlapping_islands={bad}", flush=True)
        if not bad:
            return uv, charts, owner
        layers = color_overlaps(uv, charts, size, interrupt)
        groups = charts.long() * (int(layers.max()) + 1) + layers.long()
        del owner, uv, charts
        charts = torch.unique(groups, sorted=True, return_inverse=True)[1].to(torch.int32)


def _mesh_bake_texture(mesh, frames, cache, depth_dir, texture_size, voxel_size, tolerance,
                       progress, interrupt, offset):
    import comfy.model_management as management
    import open3d as o3d
    from .helpers.gaussian_mesh_cuda import dilate_texture, launch, prepare_mesh_cuda, uv_map

    device = management.get_torch_device()
    if device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("PLY2MESH UV generation and texture baking require ComfyUI's NVIDIA CUDA device")
    prepare_mesh_cuda(device)
    # Release idle ComfyUI models before allocating mesh hash tables and texel maps.
    requested = len(mesh.triangle.indices) * 240 + texture_size**2 * 80
    management.free_memory(requested, device)
    vertices = torch.as_tensor(mesh.vertex.positions.numpy(), device=device).float().contiguous()
    faces = torch.as_tensor(mesh.triangle.indices.numpy(), device=device, dtype=torch.int32).contiguous()
    vertex_normals = torch.as_tensor(mesh.vertex.normals.numpy(), device=device).float().contiguous()
    vertex_colors = torch.as_tensor(mesh.vertex.colors.numpy(), device=device).float().contiguous()
    started = time.perf_counter()
    atlas_path = Path(depth_dir) / f"atlas_gpu_v4_{texture_size}.npz"
    print(f"[VNCCS PLY2MESH] GPU UV charts: {len(faces)} triangles, atlas={texture_size}x{texture_size}", flush=True)
    if atlas_path.is_file():
        with np.load(atlas_path, allow_pickle=False) as saved:
            uv = torch.as_tensor(saved["uv"], device=device).float().contiguous()
            charts = torch.as_tensor(saved["charts"], device=device, dtype=torch.int32).contiguous()
        if (uv.shape != (len(faces), 3, 2) or charts.shape != (len(faces),) or not torch.isfinite(uv).all()
                or (uv < 0).any() or (uv > 1).any() or charts.min() < 0 or charts.max() >= len(faces)):
            raise ValueError("Saved UV checkpoint is invalid; the original was preserved")
        owner, conflicts = uv_map(uv, charts, texture_size, interrupt)
        if torch.count_nonzero(conflicts):
            raise ValueError("Saved UV checkpoint contains overlapping islands; the original was preserved")
        print("[VNCCS PLY2MESH] Reusing saved GPU UV charts", flush=True)
    else:
        uv, charts, owner = _mesh_build_uv(vertices, faces, frames, cache, depth_dir, texture_size,
                                          voxel_size, tolerance, interrupt)
        with atomic_output_path(str(atlas_path)) as temporary:
            with open(temporary, "wb") as handle:
                np.savez(handle, uv=uv.cpu().numpy(), charts=charts.cpu().numpy())
    print(f"[VNCCS PLY2MESH] GPU UV charts ready: elapsed={time.perf_counter() - started:.1f}s", flush=True)
    uv_cpu = uv.cpu().numpy().copy()
    uv_cpu[..., 1] = 1 - uv_cpu[..., 1]
    mesh.triangle.texture_uvs = o3d.core.Tensor(uv_cpu)
    mesh.triangle.vnccs_chart = o3d.core.Tensor(charts.cpu().numpy()[:, None])
    positions = torch.zeros((texture_size**2, 3), device=device)
    normals = torch.zeros_like(positions)
    colors = torch.zeros_like(positions)
    scores = torch.zeros(texture_size**2, device=device)
    launch("uv_attributes", (vertices, faces, vertex_normals, vertex_colors, uv, owner, positions, normals, colors),
           texture_size**2, (C.c_int(texture_size), C.c_int(texture_size)), interrupt)
    del uv, charts, vertex_normals, vertex_colors
    print(f"[VNCCS PLY2MESH] GPU texture baking: {len(frames)} source views", flush=True)
    for index, frame in enumerate(frames):
        interrupt()
        frame_started = time.perf_counter()
        matrix, center, depth, reference, alpha, width, height = _mesh_gpu_frame(
            vertices, faces, frame, cache, depth_dir, index, voxel_size, interrupt)
        source = _mesh_source(frame, cache)
        if source.shape != (height, width, 3) or source.dtype != np.uint8:
            raise ValueError("Saved original RGB and camera depth have different dimensions or types")
        rgb = torch.as_tensor(source, device=device).contiguous()
        launch("texture_view", (positions, normals, owner, matrix, center, reference, depth,
                                 alpha, rgb, scores, colors), texture_size**2,
               (C.c_int(width), C.c_int(height), C.c_float(voxel_size), C.c_float(tolerance),
                C.c_float(frame["K"][0, 0] * frame["K"][1, 1])), interrupt)
        torch.cuda.current_stream(device).synchronize()
        progress.update_absolute(offset + index + 1)
        print(f"[VNCCS PLY2MESH] GPU texture views: {index + 1}/{len(frames)}; "
              f"view_time={time.perf_counter() - frame_started:.2f}s", flush=True)
    covered = int(torch.count_nonzero(scores))
    occupied = int(torch.count_nonzero(owner != 0x7FFFFFFF))
    print(f"[VNCCS PLY2MESH] Source texture coverage: {covered}/{occupied} texels; uncovered areas retain TSDF colors", flush=True)
    if not covered:
        raise RuntimeError("No source frame covers the mesh UVs; the PLY and scene may not match")
    colors = dilate_texture(colors, owner, texture_size, interrupt)
    result = (colors.reshape(texture_size, texture_size, 3).clamp(0, 1) * 255).round().byte().cpu().numpy()
    print(f"[VNCCS PLY2MESH] GPU texture ready: elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return result


def _mesh_uv_vertices(mesh, device=None):
    """Split only UV seams, retaining smooth normals at duplicated vertices."""
    if hasattr(mesh.triangle, "vnccs_chart"):
        if device is None:
            import comfy.model_management as management

            device = management.get_torch_device()
        faces = torch.as_tensor(mesh.triangle.indices.numpy(), device=device, dtype=torch.int64)
        charts = torch.as_tensor(mesh.triangle.vnccs_chart.numpy().reshape(-1), device=device, dtype=torch.int64)
        count = int(charts.max()) + 1
        keys = faces * count + charts[:, None]
        unique, inverse = torch.unique(keys.reshape(-1), sorted=True, return_inverse=True)
        del keys, faces, charts
        uv = torch.as_tensor(mesh.triangle.texture_uvs.numpy(), device=device).float().reshape(-1, 2)
        uv = torch.stack((uv[:, 0], 1 - uv[:, 1]), dim=1)
        # UVs are identical for a shared (vertex, chart); reduce without float-key sorting.
        packed_uv = torch.zeros((len(unique), 2), device=device)
        packed_uv.scatter_reduce_(0, inverse[:, None].expand(-1, 2), uv, reduce="amax", include_self=True)
        indices = (unique // count).cpu().numpy()
        return (
            torch.from_numpy(mesh.vertex.positions.numpy()[indices].astype(np.float32)),
            inverse.reshape(-1, 3).cpu(), packed_uv.cpu(),
            torch.from_numpy(mesh.vertex.normals.numpy()[indices].astype(np.float32)),
        )
    faces = mesh.triangle.indices.numpy()
    uv = mesh.triangle.texture_uvs.numpy().reshape(-1, 2).copy()
    # Open3D's atlas has bottom-left UV origin; glTF images have top-left origin.
    uv[:, 1] = 1 - uv[:, 1]
    corners = np.column_stack((faces.reshape(-1), uv))
    unique, inverse = np.unique(corners, axis=0, return_inverse=True)
    indices = unique[:, 0].astype(np.int64)
    return (
        torch.from_numpy(mesh.vertex.positions.numpy()[indices].astype(np.float32)),
        torch.from_numpy(inverse.reshape(-1, 3).astype(np.int64)),
        torch.from_numpy(unique[:, 1:].astype(np.float32)),
        torch.from_numpy(mesh.vertex.normals.numpy()[indices].astype(np.float32)),
    )


def _mesh_checkpoint(path, scene, settings, voxel_size):
    """Keep calibrated inputs and completed stages outside ComfyUI's cleared temp folder."""
    import folder_paths

    fingerprint = file_fingerprint(path)
    key = hashlib.sha256(json.dumps([fingerprint, settings], sort_keys=True).encode("utf-8")).hexdigest()
    root = safe_join_under(folder_paths.get_output_directory(), "VNCCS", "Meshes", "cache", key)
    latest = safe_join_under(root, "latest.json")
    if scene is None:
        if not Path(latest).is_file():
            raise ValueError("No saved camera scene exists for this PLY/settings. Connect the matching scene once; training is not required")
        with open(latest, encoding="utf-8") as handle:
            scene_key = json.load(handle)
        if not isinstance(scene_key, str) or len(scene_key) != 64 or any(c not in "0123456789abcdef" for c in scene_key):
            raise ValueError("Invalid mesh checkpoint index; existing data was preserved")
        directory = safe_join_under(root, scene_key)
        with open(safe_join_under(directory, "scene.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("version") != 1 or metadata.get("source") != fingerprint or metadata.get("settings") != settings:
            raise ValueError("Mesh checkpoint version or source does not match; existing data was preserved")
        frames, cache = [], {}
        for index, record in enumerate(metadata["frames"]):
            view = np.asarray(record["view"], dtype=np.float32)
            intrinsic = np.asarray(record["K"], dtype=np.float32)
            shape = tuple(record["size"])
            if (view.shape != (4, 4) or intrinsic.shape != (3, 3) or len(shape) != 2
                    or min(shape) < 1 or not np.isfinite(view).all() or not np.isfinite(intrinsic).all()
                    or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
                raise ValueError("Saved camera matrices or image dimensions are invalid")
            alpha = np.load(safe_join_under(directory, f"alpha_{index}.npy"), allow_pickle=False)
            if alpha.shape != shape or alpha.dtype != np.uint8:
                raise ValueError("Saved foreground mask is invalid")
            frames.append({"name": record["name"], "view": view, "K": intrinsic,
                           "size": shape, "depth_unit": record["depth_unit"]})
            cache[record["name"]] = {"source_path": safe_join_under(directory, f"source_{index}.png"),
                                     "alpha_uneroded": alpha}
        if len(frames) < 2 or len(cache) != len(frames):
            raise ValueError("Saved camera scene has missing or duplicate frames")
        print(f"[VNCCS PLY2MESH] Restored saved cameras and RGB: {len(frames)} views", flush=True)
        return directory, frames, cache

    _, _, frames = _scene_data(*scene)
    cache = scene[1]
    if len(frames) < 2:
        raise ValueError("PLY2MESH requires at least two registered views")
    digest = hashlib.sha256()
    for frame in frames:
        entry = cache[frame["name"]]
        if not isinstance(entry, dict) or "source_png" not in entry or "alpha_uneroded" not in entry:
            raise ValueError("The scene has no original RGB/masks; a matching Camera Reconstruction scene is required")
        if (not np.isfinite(frame["view"]).all() or not np.isfinite(frame["K"]).all()
                or frame["K"][0, 0] <= 0 or frame["K"][1, 1] <= 0):
            raise ValueError("The scene contains invalid camera matrices")
        frame["size"] = tuple(frame["rgba"].shape[:2])
        if entry["alpha_uneroded"].shape != frame["size"]:
            raise ValueError("The scene's masks and frames have different resolutions")
        digest.update(frame["name"].encode("utf-8"))
        digest.update(np.ascontiguousarray(frame["view"]).tobytes())
        digest.update(np.ascontiguousarray(frame["K"]).tobytes())
        digest.update(entry["source_png"])
        digest.update(entry["alpha_uneroded"].tobytes())
    scene_key = digest.hexdigest()
    directory = safe_join_under(root, scene_key)
    Path(directory).mkdir(parents=True, exist_ok=True)
    metadata_path = safe_join_under(directory, "scene.json")
    if Path(metadata_path).is_file():
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("version") != 1 or metadata.get("source") != fingerprint or metadata.get("settings") != settings:
            raise ValueError("Mesh checkpoint metadata is invalid; existing data was preserved")
    else:
        records = []
        for index, frame in enumerate(frames):
            entry = cache[frame["name"]]
            with atomic_output_path(safe_join_under(directory, f"source_{index}.png")) as temporary:
                with open(temporary, "wb") as handle:
                    handle.write(entry["source_png"])
            _mesh_save_array(safe_join_under(directory, f"alpha_{index}.npy"), entry["alpha_uneroded"])
            records.append({"name": frame["name"], "view": frame["view"].tolist(), "K": frame["K"].tolist(),
                            "size": list(frame["size"]), "depth_unit": float(frame["depth_unit"])})
        metadata = {"version": 1, "source": fingerprint, "settings": settings,
                    "voxel_size": voxel_size, "frames": records}
        with atomic_output_path(metadata_path) as temporary:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
    with atomic_output_path(latest) as temporary:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(scene_key, handle)
    return directory, frames, cache


def _mesh_save_geometry(mesh, directory):
    with atomic_output_path(safe_join_under(directory, "geometry.npz")) as temporary:
        with open(temporary, "wb") as handle:
            np.savez(handle, positions=np.asarray(mesh.vertices, dtype=np.float32),
                     faces=np.asarray(mesh.triangles, dtype=np.int32),
                     normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                     colors=np.asarray(mesh.vertex_colors, dtype=np.float32))
    print(f"[VNCCS PLY2MESH] Saved mesh checkpoint: {directory}", flush=True)


def _mesh_load_geometry(directory):
    import open3d as o3d

    with np.load(safe_join_under(directory, "geometry.npz"), allow_pickle=False) as saved:
        positions, faces, normals, colors = (saved[name] for name in ("positions", "faces", "normals", "colors"))
    if (positions.ndim != 2 or positions.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.dtype != np.int32 or normals.shape != positions.shape
            or colors.shape != positions.shape or not np.isfinite(positions).all()
            or not np.isfinite(normals).all() or not np.isfinite(colors).all()
            or faces.min() < 0 or faces.max() >= len(positions)):
        raise ValueError("Saved mesh geometry is invalid; existing data was preserved")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(positions), o3d.utility.Vector3iVector(faces))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    print(f"[VNCCS PLY2MESH] Reusing mesh checkpoint: {len(faces)} triangles", flush=True)
    return mesh


@torch.inference_mode(False)
@torch.no_grad()
def _ply_to_mesh(path, scene, mesh_resolution, texture_size, alpha_threshold,
                 depth_tolerance, min_component_faces, target_faces):
    import folder_paths
    import comfy.model_management as management
    try:
        import open3d as o3d
        from comfy_api.latest import Types
        from comfy_extras.nodes_save_3d import save_glb
        arrays, degree = _mesh_load_surfels(path)
    except ImportError as error:
        raise RuntimeError(f"PLY2MESH dependency import failed: {error}") from error

    device = management.get_torch_device()
    if device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("PLY2MESH depth rendering requires the existing ComfyUI NVIDIA CUDA device")
    bounds = np.quantile(arrays["means"], (0.01, 0.99), axis=0)
    voxel_size = float(np.max(bounds[1] - bounds[0])) / mesh_resolution
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("The PLY has a degenerate spatial extent")
    settings = {"version": 1, "mesh_resolution": mesh_resolution, "alpha_threshold": alpha_threshold,
                "depth_tolerance": depth_tolerance, "min_component_faces": min_component_faces,
                "target_faces": target_faces}
    directory, frames, cache = _mesh_checkpoint(path, scene, settings, voxel_size)
    progress, interrupt = _runtime_helpers(3 * len(frames) + 2)
    started = time.perf_counter()
    print(f"[VNCCS PLY2MESH] Checkpoint: {directory}", flush=True)
    if Path(safe_join_under(directory, "geometry.npz")).is_file():
        mesh = _mesh_load_geometry(directory)
        del arrays
    else:
        prepare_surface_cuda(device)
        splats = {name: torch.from_numpy(np.ascontiguousarray(value)).to(device) for name, value in arrays.items()}
        del arrays
        print(f"[VNCCS PLY2MESH] Native 2DGS: {len(splats['means'])} surfels, {len(frames)} views, voxel={voxel_size:.7f}", flush=True)
        for index, frame in enumerate(frames):
            interrupt()
            destination = Path(directory) / f"depth_{index}.npy"
            if destination.is_file():
                print(f"[VNCCS PLY2MESH] Reusing depth: {index + 1}/{len(frames)}", flush=True)
                continue
            height, width = frame["size"]
            rendered, alpha, info = render_2dgs(
                splats, torch.as_tensor(frame["view"], device=device)[None],
                torch.as_tensor(frame["K"], device=device)[None], width, height, degree,
                with_geometry=True, depth_unit=frame["depth_unit"],
            )
            depth = info["surface_maps"][0, ..., 0].cpu().numpy().copy()
            valid = (np.isfinite(depth) & (depth > 0)
                     & (alpha[0, ..., 0].cpu().numpy() >= alpha_threshold)
                     & (cache[frame["name"]]["alpha_uneroded"] >= 230))
            depth[~valid] = 0
            _mesh_save_array(destination, depth)
            del rendered, info, alpha
            progress.update_absolute(index + 1)
            print(f"[VNCCS PLY2MESH] Depth: {index + 1}/{len(frames)}, valid={np.count_nonzero(valid)}", flush=True)
        del splats
        volume = _mesh_tsdf_volume(voxel_size)
        centers = np.stack([-(f["view"][:3, :3].T @ f["view"][:3, 3]) for f in frames])
        for index, frame in enumerate(frames):
            interrupt()
            depth = np.load(Path(directory) / f"depth_{index}.npy", allow_pickle=False)
            destination = Path(directory) / f"filtered_{index}.npy"
            if destination.is_file():
                filtered = np.load(destination, allow_pickle=False)
            else:
                order = np.argsort(np.linalg.norm(centers - centers[index], axis=1))
                nearest = [int(j) for j in order if j != index][:4]
                neighbors = [(frames[j], np.load(Path(directory) / f"depth_{j}.npy", allow_pickle=False))
                             for j in nearest]
                filtered = _mesh_consistent_depth(depth, frame, neighbors, voxel_size, depth_tolerance, interrupt)
                _mesh_save_array(destination, filtered)
                del neighbors
            rgb = _mesh_source(frame, cache)
            if rgb.shape[:2] != filtered.shape:
                raise ValueError("The scene's original RGB and rendered depth have different resolutions")
            blocks = _mesh_integrate_frame(volume, frame, filtered, rgb, voxel_size)
            progress.update_absolute(len(frames) + index + 1)
            print(f"[VNCCS PLY2MESH] TSDF: {index + 1}/{len(frames)}, "
                  f"consistent={np.count_nonzero(filtered)}/{np.count_nonzero(depth)}, "
                  f"frame_blocks={blocks}, total_blocks={volume.hashmap().size()}", flush=True)
        interrupt()
        print("[VNCCS PLY2MESH] Extracting and cleaning triangle mesh on CPU")
        if not volume.hashmap().size():
            raise RuntimeError("TSDF has no allocated surface blocks: no valid depth samples reached integration")
        mesh = _mesh_extract_surface(volume, voxel_size, interrupt)
        del volume
        raw_faces = len(mesh.triangles)
        print(f"[VNCCS PLY2MESH] Raw mesh: {len(mesh.vertices)} vertices, {raw_faces} triangles")
        if not raw_faces:
            raise RuntimeError("TSDF has surface blocks but no zero-crossing triangles before cleanup; inspect depth/camera consistency")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        if not len(mesh.triangles):
            raise RuntimeError(f"TSDF produced {raw_faces} triangles, but all were degenerate")
        largest_component = 0
        if len(mesh.triangles) and min_component_faces:
            labels, counts, _ = mesh.cluster_connected_triangles()
            largest_component = max(counts, default=0)
            mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < min_component_faces)
        mesh.remove_unreferenced_vertices()
        if not len(mesh.triangles):
            raise RuntimeError(
                f"Mesh cleanup removed all {raw_faces} triangles; largest component={largest_component}, "
                f"min_component_faces={min_component_faces}. Set min_component_faces=0 to retain disconnected surfaces"
            )
        if target_faces and len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        mesh.compute_vertex_normals()
        print(f"[VNCCS PLY2MESH] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        _mesh_save_geometry(mesh, directory)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    progress.update_absolute(2 * len(frames) + 1)
    texture = _mesh_bake_texture(mesh, frames, cache, directory, texture_size, voxel_size,
                                 depth_tolerance, progress, interrupt, 2 * len(frames) + 1)
    interrupt()
    vertices, faces, uvs, normals = _mesh_uv_vertices(mesh)
    texture_tensor = torch.from_numpy(texture).float()[None] / 255.0
    result = Types.MESH(vertices[None], faces[None], uvs=uvs[None], normals=normals[None], texture=texture_tensor)
    output = safe_join_under(folder_paths.get_output_directory(), "VNCCS", "Meshes",
                             f"mesh_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.glb")
    with atomic_output_path(output) as target:
        save_glb(vertices, faces, filepath=target, uvs=uvs, normals=normals, texture_image=Image.fromarray(texture))
    progress.update_absolute(3 * len(frames) + 2)
    print(f"[VNCCS PLY2MESH] Saved textured mesh to {output}; elapsed={time.perf_counter() - started:.1f}s")
    return result, output, texture_tensor


class VNCCSPLY2MESH:
    """Fuse native surfel depth into a mesh and bake its original orbit images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ply_path": ("STRING", {"default": "", "tooltip": "Connect native_2dgs_file from a use_2dgs training run. Standard 3DGS/compatibility PLY is not accepted."}),
            "mesh_resolution": ("INT", {"default": 512, "min": 1, "max": 0x7FFFFFFF, "step": 64, "tooltip": "Voxels across the subject's robust longest extent. Higher values retain finer geometry and need more CPU time/RAM."}),
            "texture_size": ("INT", {"default": 4096, "min": 1, "max": 0x7FFFFFFF, "step": 1024, "tooltip": "Square texture size. UV projection and source-image baking use ComfyUI's CUDA device; rectangle packing and file export use CPU."}),
            "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum rendered coverage for surface extraction. Higher values remove uncertain surfaces but may create holes."}),
            "depth_tolerance": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Relative depth disagreement allowed between neighboring cameras; also allows two voxels of discretization error."}),
            "min_component_faces": ("INT", {"default": 100, "min": 0, "max": 0x7FFFFFFF, "step": 1, "tooltip": "Remove disconnected components smaller than this triangle count. Zero preserves every component."}),
            "target_faces": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1000, "tooltip": "Optional mesh simplification target. Zero keeps full extracted detail."}),
        }, "optional": {
            "scene": (GAUSSIAN_SCENE_TYPE, {"tooltip": "Connect the exact camera scene used for this PLY once. Later exports can restore cameras, RGB, depth and mesh from the persistent checkpoint without this input."}),
        }}

    RETURN_TYPES = ("MESH", "STRING", "IMAGE")
    RETURN_NAMES = ("mesh", "glb_file", "texture")
    FUNCTION = "convert"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, texture_size):
        from importlib.metadata import PackageNotFoundError, version

        if texture_size < 1:
            return "texture_size must be positive"
        # Linked PLY/scene do not exist yet; report missing packages before node execution.
        for package in ("open3d", "plyfile"):
            try:
                version(package)
            except PackageNotFoundError:
                return (
                    f"PLY2MESH: missing Python package '{package}'. "
                    "Install the VNCCS dependencies in the Python environment that runs ComfyUI."
                )
        return True

    @classmethod
    def IS_CHANGED(cls, ply_path, **_kwargs):
        return file_fingerprint(_mesh_ply_path(ply_path))

    def convert(self, ply_path, scene=None, mesh_resolution=512, texture_size=4096,
                alpha_threshold=0.5, depth_tolerance=0.01, min_component_faces=100, target_faces=0):
        if scene is not None and (not isinstance(scene, (tuple, list)) or len(scene) != 2):
            raise ValueError("scene must come from VNCCS Gaussian Camera Reconstruction")
        if mesh_resolution < 1 or texture_size < 1 or min_component_faces < 0 or target_faces < 0:
            raise ValueError("Resolutions must be positive; component and simplification counts must be non-negative")
        if not 0 <= alpha_threshold <= 1 or not 0 <= depth_tolerance <= 1:
            raise ValueError("Alpha threshold and depth tolerance must be finite and between zero and one")
        return _ply_to_mesh(_mesh_ply_path(ply_path), scene, mesh_resolution, texture_size,
                            alpha_threshold, depth_tolerance, min_component_faces, target_faces)


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": VNCCSGaussianCameraReconstruction,
    "VNCCS_GaussianTrainer": VNCCSGaussianTrainer,
    "VNCCS_PLY2MESH": VNCCSPLY2MESH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": "VNCCS Gaussian Camera Reconstruction",
    "VNCCS_GaussianTrainer": "VNCCS Gaussian Trainer",
    "VNCCS_PLY2MESH": "VNCCS PLY2MESH",
}
