"""Contract tests for native Gaussian camera reconstruction."""

import pytest

torch = pytest.importorskip("torch")

from nodes.gaussian_reconstruction import (
    DEFAULT_MAX_SIDE,
    GAUSSIAN_SCENE_TYPE,
    VNCCSGaussianCameraReconstruction,
    _circular_pairs,
    _erode_training_alpha,
    _pycolmap_cuda_available,
    _resize_frame,
    _sharpest_indices,
)


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
        "images", "max_frames", "max_resolution", "match_window", "matte_tolerance"
    ]
    assert VNCCSGaussianCameraReconstruction.RETURN_TYPES == (GAUSSIAN_SCENE_TYPE,)
    assert inputs["max_resolution"][1]["default"] == DEFAULT_MAX_SIDE == 2048
    assert "max" not in inputs["max_resolution"][1]
