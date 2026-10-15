"""Contract tests for native Gaussian reconstruction."""

import pytest

torch = pytest.importorskip("torch")

from nodes.gaussian_reconstruction import (
    DEFAULT_TRAINING_STEPS,
    GAUSSIAN_SCENE_TYPE,
    VNCCSGaussianCameraReconstruction,
    VNCCSGaussianTraining,
    _circular_pairs,
    _pycolmap_cuda_available,
    _resize_frame,
    _sharpest_indices,
    _train_gaussians,
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


def test_pycolmap_cuda_requires_a_cuda_capable_build_and_device():
    class CudaBuild:
        @staticmethod
        def get_num_cuda_devices():
            return 1

    class CpuBuild:
        pass

    assert _pycolmap_cuda_available(CudaBuild()) is True
    assert _pycolmap_cuda_available(CpuBuild()) is False


def test_training_reenables_autograd_inside_comfy_inference_mode(monkeypatch):
    modes = {}

    def unavailable_cuda():
        modes["inference"] = torch.is_inference_mode_enabled()
        modes["grad"] = torch.is_grad_enabled()
        return False

    monkeypatch.setattr(torch.cuda, "is_available", unavailable_cuda)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA build"):
            _train_gaussians(None, None, None, 2_000, 100_000, 1, None, None)

    assert modes == {"inference": False, "grad": True}


def test_camera_and_training_are_separate_comfy_cache_stages():
    camera_inputs = VNCCSGaussianCameraReconstruction.INPUT_TYPES()["required"]
    training_inputs = VNCCSGaussianTraining.INPUT_TYPES()["required"]

    assert list(camera_inputs) == [
        "images", "max_frames", "max_resolution", "match_window", "matte_tolerance"
    ]
    assert VNCCSGaussianCameraReconstruction.RETURN_TYPES == (GAUSSIAN_SCENE_TYPE,)
    assert training_inputs["scene"] == (GAUSSIAN_SCENE_TYPE,)
    assert training_inputs["training_steps"][1]["default"] == DEFAULT_TRAINING_STEPS == 7_000
    assert training_inputs["training_steps"][1]["min"] == 1
    assert training_inputs["training_steps"][1]["max"] == 0xFFFFFFFFFFFFFFFF
    assert training_inputs["training_steps"][1]["step"] == 1
    assert training_inputs["max_splats"][1]["default"] == 2_000_000
    assert training_inputs["sh_degree"][1]["default"] == 1
    assert VNCCSGaussianTraining.RETURN_TYPES == ("STRING",)
    assert VNCCSGaussianTraining.RETURN_NAMES == ("gaussian_file",)
    assert VNCCSGaussianTraining.CATEGORY == "VNCCS/3D"
