"""Contract tests for native Gaussian reconstruction."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import nodes.gaussian_reconstruction as gaussian_reconstruction
from nodes.gaussian_reconstruction import (
    DEFAULT_TRAINING_STEPS,
    GAUSSIAN_SCENE_TYPE,
    INITIAL_OPACITY,
    VNCCSGaussianCameraReconstruction,
    VNCCSGaussianTraining,
    _brush_split_parameters,
    _circular_pairs,
    _comfy_splat,
    _decayed_opacity_logits,
    _erode_training_alpha,
    _growth_mask,
    _initial_scales,
    _point_scene_scale,
    _pycolmap_cuda_available,
    _resize_frame,
    _sharpest_indices,
    _splat_prune_masks,
    _ssim_loss,
    _strategy_options,
    _training_learning_rates,
    _training_loss,
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


def test_training_reenables_autograd_inside_comfy_inference_mode(monkeypatch):
    modes = {}

    def unavailable_cuda():
        modes["inference"] = torch.is_inference_mode_enabled()
        modes["grad"] = torch.is_grad_enabled()
        return False

    monkeypatch.setattr(torch.cuda, "is_available", unavailable_cuda)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA build"):
            _train_gaussians(
                None, None, None, 2_000, 100_000, 1, True, True, True, True, None, None
            )

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
    assert training_inputs["budgeted_densification"][1]["default"] is True
    assert training_inputs["stable_refinement"][1]["default"] is True
    assert training_inputs["antialiased_rasterization"][1]["default"] is True
    assert training_inputs["ssim_alpha_loss"][1]["default"] is True
    assert VNCCSGaussianTraining.RETURN_TYPES == ("STRING", "SPLAT")
    assert VNCCSGaussianTraining.RETURN_NAMES == ("gaussian_file", "splat")
    assert VNCCSGaussianTraining.CATEGORY == "VNCCS/3D"


def test_brush_initial_scale_uses_half_the_two_neighbor_average(monkeypatch):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    class FakeIndex:
        def knnSearch(self, query, count, params):
            neighbors = np.column_stack(
                (
                    np.arange(len(query)),
                    np.ones(len(query), dtype=np.int64),
                    np.full(len(query), 2, dtype=np.int64),
                )
            )
            return neighbors, np.zeros_like(neighbors, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(flann_Index=lambda *_args, **_kwargs: FakeIndex()),
    )

    assert _point_scene_scale(points) == pytest.approx(2.0)
    assert np.exp(_initial_scales(points)[0]) == pytest.approx(0.015, abs=1e-6)


def test_training_uses_brush_initial_opacity_and_learning_rates():
    assert INITIAL_OPACITY == 0.5
    assert _training_learning_rates(2.0) == {
        "means": 4e-5,
        "scales": 5e-3,
        "quats": 2e-3,
        "opacities": 1.2e-2,
        "sh0": 2e-3,
        "shN": 2e-4,
    }


def test_budgeted_growth_keeps_the_strongest_quarter_of_candidates():
    grads = torch.tensor([0.1, 0.9, 0.4, 0.8])
    candidates = torch.ones(4, dtype=torch.bool)

    limited = _growth_mask(grads, candidates, available=2, fraction=0.25)

    assert limited.tolist() == [False, True, False, False]


def test_brush_split_is_symmetric_and_mass_aware():
    means = torch.zeros((1, 3))
    log_scales = torch.zeros((1, 3))
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    opacity_logits = torch.zeros(1)

    child_means, child_log_scales, child_opacity_logits = _brush_split_parameters(
        means, log_scales, quats, opacity_logits
    )

    offset = 1.0 / np.sqrt(2.0)
    assert torch.allclose(child_means[0], torch.full((3,), -offset))
    assert torch.allclose(child_means[1], torch.full((3,), offset))
    assert torch.allclose(torch.exp(child_log_scales), torch.full((2, 3), offset))
    expected_opacity = 1.0 - 0.5 ** (1.0 / np.sqrt(2.0))
    assert torch.allclose(
        torch.sigmoid(child_opacity_logits),
        torch.full((2,), expected_opacity),
    )


def test_comfy_splat_uses_activated_batched_cpu_tensors(monkeypatch):
    class FakeSplat:
        def __init__(self, positions, scales, rotations, opacities, sh):
            self.values = positions, scales, rotations, opacities, sh

    monkeypatch.setitem(
        sys.modules,
        "comfy_api.latest",
        SimpleNamespace(Types=SimpleNamespace(SPLAT=FakeSplat)),
    )
    params = {
        "means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]])),
        "scales": torch.nn.Parameter(torch.log(torch.tensor([[2.0, 3.0, 4.0]]))),
        "quats": torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
        "opacities": torch.nn.Parameter(torch.tensor([0.0])),
        "sh0": torch.nn.Parameter(torch.ones((1, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((1, 3, 3))),
    }

    positions, scales, rotations, opacities, sh = _comfy_splat(params).values

    assert positions.shape == (1, 1, 3)
    assert torch.allclose(scales, torch.tensor([[[2.0, 3.0, 4.0]]]))
    assert torch.allclose(rotations, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    assert opacities.shape == (1, 1, 1)
    assert opacities.item() == pytest.approx(0.5)
    assert sh.shape == (1, 1, 4, 3)
    assert all(
        value.device.type == "cpu"
        for value in (positions, scales, rotations, opacities, sh)
    )


def test_training_returns_file_path_and_comfy_splat(monkeypatch, tmp_path):
    output_path = str(tmp_path / "trained.ply")
    splat = object()
    monkeypatch.setattr(gaussian_reconstruction, "_output_path", lambda: output_path)
    monkeypatch.setattr(
        gaussian_reconstruction,
        "_runtime_helpers",
        lambda _total: (object(), object()),
    )
    monkeypatch.setattr(
        gaussian_reconstruction,
        "_train_gaussians",
        lambda *_args: splat,
    )

    result = VNCCSGaussianTraining().train((object(), {}), training_steps=1)

    assert result == (output_path, splat)


def test_stable_refinement_uses_first_half():
    assert _strategy_options(7_000, False) == {
        "grow_grad2d": 0.0002,
        "grow_scale2d": 0.05,
        "refine_every": 100,
        "refine_start_iter": 0,
        "refine_stop_iter": 7_000,
        "refine_scale2d_stop_iter": 7_000,
    }
    assert _strategy_options(17_000, False) == {
        "grow_grad2d": 0.0002,
        "grow_scale2d": 0.05,
        "refine_every": 100,
        "refine_start_iter": 0,
        "refine_stop_iter": 15_000,
        "refine_scale2d_stop_iter": 15_000,
    }
    assert _strategy_options(7_000, True) == {
        "grow_grad2d": 0.0001,
        "grow_scale2d": 0.05,
        "refine_every": 100,
        "refine_start_iter": 0,
        "refine_stop_iter": 3_500,
        "refine_scale2d_stop_iter": 3_500,
    }


def test_prune_masks_cover_invalid_low_opacity_and_oversized_splats():
    splats = {
        "means": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [float("nan"), 0.0, 0.0],
            ]
        ),
        "scales": torch.log(
            torch.tensor([[0.01] * 3, [0.01] * 3, [0.2] * 3, [0.01] * 3])
        ),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
        "opacities": torch.logit(torch.tensor([0.5, 0.005, 0.5, 0.5])),
        "sh0": torch.zeros((4, 1, 3)),
        "shN": torch.zeros((4, 3, 3)),
    }

    masks = _splat_prune_masks(
        splats, scene_scale=1.0, opacity_threshold=0.01, scale_ratio=0.1
    )

    assert masks["nonfinite"].tolist() == [False, False, False, True]
    assert masks["opacity"].tolist() == [False, True, False, False]
    assert masks["scale3d"].tolist() == [False, False, True, False]


def test_pytorch_ssim_loss_is_zero_for_identical_images():
    image = torch.rand((1, 16, 16, 3))
    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()

    assert _ssim_loss(image, image, gaussian).item() == pytest.approx(0.0, abs=1e-6)
    assert _ssim_loss(image, 1.0 - image, gaussian).item() > 0.1


def test_alpha_supervision_stays_enabled_without_ssim():
    image = torch.zeros((1, 16, 16, 3))
    alpha = torch.ones((1, 16, 16, 1))
    target_alpha = torch.zeros_like(alpha)
    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()

    without_ssim = _training_loss(image, image, alpha, target_alpha)
    with_ssim = _training_loss(image, image, alpha, target_alpha, gaussian)

    assert without_ssim.item() == pytest.approx(0.1)
    assert with_ssim.item() == pytest.approx(0.1)


def test_brush_opacity_decay_fades_out_over_training():
    logits = torch.zeros(1)

    early = torch.sigmoid(_decayed_opacity_logits(logits, 0, 1_000))
    final = torch.sigmoid(_decayed_opacity_logits(logits, 1_000, 1_000))

    assert early.item() == pytest.approx(0.496)
    assert final.item() == pytest.approx(0.5)


def test_separable_ssim_matches_the_full_gaussian_window():
    rendered = torch.rand((1, 16, 16, 3))
    target = torch.rand((1, 16, 16, 3))
    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    full_kernel = torch.outer(gaussian, gaussian)[None, None].repeat(3, 1, 1, 1)
    rendered_nchw = rendered.permute(0, 3, 1, 2)
    target_nchw = target.permute(0, 3, 1, 2)
    mu_rendered = F.conv2d(rendered_nchw, full_kernel, groups=3)
    mu_target = F.conv2d(target_nchw, full_kernel, groups=3)
    variance_rendered = F.conv2d(
        rendered_nchw.square(), full_kernel, groups=3
    ) - mu_rendered.square()
    variance_target = F.conv2d(
        target_nchw.square(), full_kernel, groups=3
    ) - mu_target.square()
    covariance = F.conv2d(
        rendered_nchw * target_nchw, full_kernel, groups=3
    ) - mu_rendered * mu_target
    reference_score = (
        (2 * mu_rendered * mu_target + 0.01**2) * (2 * covariance + 0.03**2)
    ) / (
        (mu_rendered.square() + mu_target.square() + 0.01**2)
        * (variance_rendered + variance_target + 0.03**2)
    )
    reference_loss = (1.0 - reference_score.mean()).clamp(0.0, 2.0)

    assert _ssim_loss(rendered, target, gaussian).item() == pytest.approx(
        reference_loss.item(), abs=1e-5
    )
