"""Native image-batch to 3D Gaussian reconstruction."""

import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .chroma_screen_matte import screen_matte


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


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": VNCCSGaussianCameraReconstruction,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": "VNCCS Gaussian Camera Reconstruction",
}
