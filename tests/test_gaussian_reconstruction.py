"""Contract tests for native Gaussian camera reconstruction."""

import sys
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

import nodes.gaussian_reconstruction as reconstruction
from nodes.gaussian_reconstruction import (
    DEFAULT_MAX_SIDE,
    GAUSSIAN_SCENE_TYPE,
    VNCCSGaussianCameraReconstruction,
    _circular_pairs,
    _erode_training_alpha,
    _pycolmap_cuda_available,
    _resize_frame,
    _sharpest_indices,
    _solve_cameras,
    _top_view_cameras,
)


def _orbit_frames(angles):
    target = np.array([1.0, 2.0, 3.0])
    up = np.array([0.0, 1.0, 0.0])
    frames = []
    for index, angle in enumerate(np.deg2rad(angles)):
        position = target + 2.0 * np.array([np.sin(angle), 0.0, np.cos(angle)])
        forward = (target - position) / 2.0
        right = np.cross(forward, up)
        rotation = np.stack((right, np.cross(forward, right), forward))
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ position
        frames.append({
            "name": f"{index:05d}.png", "view": view,
            "K": np.array([[12, 0, 8], [0, 12, 8], [0, 0, 1]], dtype=np.float32),
        })
    return frames


def test_top_cameras_use_fixed_azimuths_and_nearest_original_views():
    frames = _orbit_frames([0, 22, 83, 100, 173, 226, 274, 330])

    previews = _top_view_cameras(frames)

    assert [frame["name"] for frame, _ in previews] == [
        "00000.png", "00002.png", "00004.png", "00006.png",
    ]
    target = np.array([1.0, 2.0, 3.0])
    for index, (_, view) in enumerate(previews):
        pose = np.linalg.inv(view)
        angle = index * np.pi / 2
        expected_offset = np.sqrt(2) * np.array([np.sin(angle), 1.0, np.cos(angle)])
        np.testing.assert_allclose(pose[:3, 3] - target, expected_offset, atol=1e-5)
        np.testing.assert_allclose(view[:3, :3] @ target + view[:3, 3], [0, 0, 2], atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(view[:3, :3]), 1.0, atol=1e-6)


def test_top_cameras_follow_scene_orientation_instead_of_world_up():
    frames = _orbit_frames([0, 83, 173, 274])
    transform = np.array([
        [1, 0, 0, 4], [0, 0.8, -0.6, -3], [0, 0.6, 0.8, 2], [0, 0, 0, 1],
    ], dtype=np.float32)
    rotated = [
        {**frame, "view": frame["view"] @ np.linalg.inv(transform)}
        for frame in frames
    ]

    previews = _top_view_cameras(frames)
    rotated_previews = _top_view_cameras(rotated)

    for (frame, view), (rotated_frame, rotated_view) in zip(previews, rotated_previews):
        assert frame["name"] == rotated_frame["name"]
        np.testing.assert_allclose(rotated_view @ transform, view, atol=1e-5)


def test_camera_cache_keeps_only_four_original_rgb_references(monkeypatch):
    frames = _orbit_frames(np.arange(20) * 18)
    cameras = {1: SimpleNamespace(calibration_matrix=lambda: frames[0]["K"])}
    images_by_id = {
        index: SimpleNamespace(
            name=frame["name"], camera_id=1,
            cam_from_world=lambda view=frame["view"]: SimpleNamespace(matrix=lambda: view[:3]),
        )
        for index, frame in enumerate(frames)
    }
    model = SimpleNamespace(
        images=images_by_id, cameras=cameras,
        reg_image_ids=lambda: list(reversed(range(20))),
        points3D={
            index: SimpleNamespace(xyz=np.array(point), color=np.array([128, 128, 128]))
            for index, point in enumerate([[0, 2, 3], [2, 2, 3], [1, 1, 3], [1, 3, 3]])
        },
    )

    def matte(frame, **_kwargs):
        alpha = torch.ones(frame.shape[:2])
        return torch.cat((torch.zeros_like(frame), alpha[..., None]), dim=-1), alpha, None

    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(get_num_cuda_devices=lambda: 0))
    monkeypatch.setattr(reconstruction, "screen_matte", matte)
    camera_arguments = []

    def solve_cameras(*args):
        camera_arguments.extend(args)
        return model

    monkeypatch.setattr(reconstruction, "_solve_cameras", solve_cameras)
    monkeypatch.setattr(
        reconstruction, "_runtime_helpers",
        lambda _total: (SimpleNamespace(update_absolute=lambda _value: None), lambda: None),
    )
    images = torch.zeros((20, 16, 16, 3))
    images[..., 0] = torch.arange(20)[:, None, None] / 20
    images[..., 1:] = 0.4

    ((cached_model, cached_frames),) = VNCCSGaussianCameraReconstruction().reconstruct(images)
    assert camera_arguments[-3:] == [3.0, 20, 0.75]
    _, _, scene_frames = reconstruction._scene_data(cached_model, cached_frames)

    references = {name: data for name, data in cached_frames.items() if "source_rgb" in data}
    assert set(references) == {"00000.png", "00005.png", "00010.png", "00015.png"}
    assert [frame["name"] for frame in scene_frames] == [frame["name"] for frame in frames]
    for name, data in references.items():
        expected = (images[int(name[:5])] * 255).round().to(torch.uint8).numpy()
        np.testing.assert_array_equal(data["source_rgb"], expected)
        assert not data["rgba"][..., :3].any()
    assert sum(frame["source_rgb"] is not None for frame in scene_frames) == 4
    for name, data in cached_frames.items():
        with Image.open(BytesIO(data["source_png"])) as source:
            expected = (images[int(name[:5])] * 255).round().to(torch.uint8).numpy()
            np.testing.assert_array_equal(np.array(source), expected)


def test_closed_orbit_pairs_wrap_around_without_duplicates():
    names = [f"{index:05d}.png" for index in range(10)]
    pairs = _circular_pairs(names, window=2)

    assert len(pairs) == 20
    assert (names[0], names[9]) in pairs
    assert len(pairs) == len(set(pairs))


def test_sharpest_frame_is_selected_per_ordered_window():
    assert _sharpest_indices([1, 4, 3, 2, 8, 7], 3) == [1, 2, 4]


def test_resize_preserves_input_alpha_channel():
    frame = torch.zeros((20, 10, 4))
    frame[..., 3] = 0.25

    resized = _resize_frame(frame, max_side=16)

    assert resized.shape == (16, 8, 4)
    assert torch.allclose(resized[..., 3], torch.full((16, 8), 0.25))


def test_training_alpha_is_eroded_by_one_pixel():
    alpha = torch.zeros((5, 5))
    alpha[1:4, 1:4] = 1.0

    eroded = _erode_training_alpha(alpha)

    assert eroded.sum().item() == pytest.approx(1.0)
    assert eroded[2, 2].item() == pytest.approx(1.0)


def test_pycolmap_cuda_requires_a_cuda_capable_build_and_device():
    class CudaBuild:
        @staticmethod
        def get_num_cuda_devices():
            return 1

    class CpuBuild:
        pass

    assert _pycolmap_cuda_available(CudaBuild()) is True
    assert _pycolmap_cuda_available(CpuBuild()) is False


def test_camera_stage_has_no_training_settings_or_resolution_ceiling():
    inputs = VNCCSGaussianCameraReconstruction.INPUT_TYPES()["required"]

    assert list(inputs) == [
        "images", "max_frames", "max_resolution", "match_window", "matte_tolerance",
        "ransac_max_error", "min_num_inliers", "sift_max_ratio",
    ]
    assert VNCCSGaussianCameraReconstruction.RETURN_TYPES == (GAUSSIAN_SCENE_TYPE,)
    assert inputs["max_resolution"][1]["default"] == DEFAULT_MAX_SIDE == 2048
    assert "max" not in inputs["max_resolution"][1]
    assert inputs["ransac_max_error"][1]["default"] == 3.0
    assert inputs["min_num_inliers"][1]["default"] == 20
    assert inputs["sift_max_ratio"][1]["default"] == 0.75


@pytest.mark.parametrize("error,inliers,ratio,use_cuda", [
    (3.0, 20, 0.75, False), (2.5, 24, 0.72, True),
])
def test_matching_thresholds_reach_pycolmap(tmp_path, error, inliers, ratio, use_cuda):
    captured = {}

    class MatchingCaptured(Exception):
        pass

    def match_image_pairs(_database, **kwargs):
        captured.update(kwargs)
        raise MatchingCaptured

    pycolmap = SimpleNamespace(
        FeatureExtractionOptions=lambda: SimpleNamespace(sift=SimpleNamespace()),
        FeatureMatchingOptions=lambda: SimpleNamespace(sift=SimpleNamespace()),
        TwoViewGeometryOptions=lambda: SimpleNamespace(ransac=SimpleNamespace()),
        ImageReaderOptions=SimpleNamespace, ImportedPairingOptions=SimpleNamespace,
        CameraMode=SimpleNamespace(SINGLE="single"), Device=SimpleNamespace(cuda="cuda", cpu="cpu"),
        extract_features=lambda *_args, **_kwargs: None,
        match_image_pairs=match_image_pairs,
    )
    with pytest.raises(MatchingCaptured):
        _solve_cameras(
            pycolmap, tmp_path, tmp_path, tmp_path, ["00000.png", "00001.png"], 20,
            use_cuda, SimpleNamespace(update_absolute=lambda _value: None), lambda: None,
            error, inliers, ratio,
        )

    assert captured["matching_options"].sift.max_ratio == ratio
    assert captured["verification_options"].ransac.max_error == error
    assert captured["verification_options"].min_num_inliers == inliers
    assert captured["device"] == ("cuda" if use_cuda else "cpu")


@pytest.mark.parametrize("options", [
    {"ransac_max_error": 0}, {"ransac_max_error": float("nan")},
    {"ransac_max_error": float("inf")}, {"min_num_inliers": 0},
    {"sift_max_ratio": 0}, {"sift_max_ratio": 1.1}, {"sift_max_ratio": float("nan")},
])
def test_invalid_matching_thresholds_fail_before_camera_work(options):
    with pytest.raises(ValueError):
        VNCCSGaussianCameraReconstruction().reconstruct(torch.zeros((20, 16, 16, 3)), **options)
