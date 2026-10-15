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
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized[0].permute(1, 2, 0).clamp(0, 1)


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
        training_frames[name] = (rgba * 255).round().to(torch.uint8).cpu().numpy()
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


def _initial_scales(points):
    import cv2

    index = cv2.flann_Index(np.ascontiguousarray(points), {"algorithm": 1, "trees": 4})
    neighbors, _ = index.knnSearch(points, 4, params={"checks": 32})
    offsets = points[neighbors[:, 1:]] - points[:, None]
    distances = np.sqrt(np.square(offsets).sum(axis=2).mean(axis=1))
    return np.log(np.maximum(distances, 1e-6)).astype(np.float32)


@torch.inference_mode(False)
@torch.enable_grad()
def _train_gaussians(
    reconstruction,
    training_frames,
    output_path,
    training_steps,
    max_splats,
    sh_degree,
    progress,
    interrupt,
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Gaussian training requires an NVIDIA CUDA build of PyTorch")
    try:
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
    except ModuleNotFoundError as error:
        raise RuntimeError("gsplat 1.5.x is required for Gaussian training") from error

    device = torch.device("cuda", torch.cuda.current_device())
    points, rgb, frames = _scene_data(reconstruction, training_frames)
    torch.manual_seed(42)
    count = len(points)
    colors = torch.zeros((count, (sh_degree + 1) ** 2, 3), dtype=torch.float32)
    colors[:, 0] = (torch.from_numpy(rgb) - 0.5) / SH_C0
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points)),
            "scales": torch.nn.Parameter(
                torch.from_numpy(_initial_scales(points))[:, None].repeat(1, 3)
            ),
            "quats": torch.nn.Parameter(torch.rand((count, 4))),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((count,), 0.1))),
            "sh0": torch.nn.Parameter(colors[:, :1]),
            "shN": torch.nn.Parameter(colors[:, 1:]),
        }
    ).to(device)
    learning_rates = {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": rate, "name": name}], eps=1e-15)
        for name, rate in learning_rates.items()
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / training_steps)
    )
    strategy = DefaultStrategy(refine_stop_iter=training_steps)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=1.1)
    generator = torch.Generator().manual_seed(42)
    log_every = max(1, training_steps // 20)
    print(
        f"[VNCCS Gaussian] Training {count} initial splats from "
        f"{len(frames)} frames for {training_steps} steps"
    )

    for step in range(training_steps):
        if step % 10 == 0:
            interrupt()
            progress.update_absolute(step)
        frame = frames[int(torch.randint(len(frames), (1,), generator=generator))]
        rgba = torch.from_numpy(frame["rgba"]).to(device=device, dtype=torch.float32) / 255.0
        target_alpha = rgba[..., 3:4]
        background = torch.rand((1, 1, 1, 3), device=device)
        target = rgba[None, ..., :3] * target_alpha[None] + background * (1.0 - target_alpha[None])
        view = torch.from_numpy(frame["view"]).to(device)[None]
        intrinsics = torch.from_numpy(frame["K"]).to(device)[None]
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
            packed=True,
            sh_degree=min(step // 1000, sh_degree),
        )
        rendered = rendered + background * (1.0 - alpha)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss = F.l1_loss(rendered, target)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        strategy.grow_grad2d = (
            0.0002 if len(splats["means"]) < max_splats // 2 else math.inf
        )
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=True)
        completed = step + 1
        if completed == 1 or completed % log_every == 0 or completed == training_steps:
            print(
                f"[VNCCS Gaussian] Training {completed}/{training_steps} "
                f"({completed / training_steps:.0%}): loss={loss.detach().item():.6f}, "
                f"splats={len(splats['means'])}"
            )

    progress.update_absolute(training_steps)
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gaussian_file",)
    FUNCTION = "train"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True

    def train(
        self,
        scene,
        training_steps=DEFAULT_TRAINING_STEPS,
        max_splats=DEFAULT_MAX_SPLATS,
        sh_degree=DEFAULT_SH_DEGREE,
    ):
        if not isinstance(scene, (tuple, list)) or len(scene) != 2:
            raise ValueError("scene must come from VNCCS Gaussian Camera Reconstruction")
        reconstruction, training_frames = scene
        output_path = _output_path()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        progress, interrupt = _runtime_helpers(training_steps)
        _train_gaussians(
            reconstruction,
            training_frames,
            output_path,
            training_steps,
            max_splats,
            sh_degree,
            progress,
            interrupt,
        )
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": VNCCSGaussianCameraReconstruction,
    "VNCCS_GaussianTraining": VNCCSGaussianTraining,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_GaussianCameraReconstruction": "VNCCS Gaussian Camera Reconstruction",
    "VNCCS_GaussianTraining": "VNCCS Gaussian Training",
}
