"""Contracts for the standalone Gaussian orbit trainer."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import nodes.gaussian_training as training


def test_scene_scale_uses_camera_extent_instead_of_subject_thickness():
    frames = []
    for position in ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]):
        view = np.eye(4)
        view[:3, 3] = -np.asarray(position)
        frames.append({"view": view})

    assert training._scene_scale(frames) == pytest.approx(2.2)


def test_initial_scales_use_rms_of_three_neighbors_without_brush_floor(monkeypatch):
    points = np.array([[0, 0, 0], [0.0001, 0, 0], [0.0002, 0, 0], [0.0003, 0, 0]], dtype=np.float32)

    class Index:
        def knnSearch(self, query, count, params):
            assert count == 4
            neighbors = np.tile(np.arange(4), (4, 1))
            return neighbors, np.zeros_like(neighbors)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(flann_Index=lambda *_args: Index()))
    scales = np.exp(training._initial_scales(points))

    assert scales[0] == pytest.approx(np.sqrt((1e-8 + 4e-8 + 9e-8) / 3))
    assert scales[0] < 1e-3


def test_alpha_loss_and_physgaussian_penalty_are_independent():
    rgb = torch.zeros((1, 16, 16, 3))
    predicted_alpha = torch.zeros((1, 16, 16, 1))
    target_alpha = torch.ones_like(predicted_alpha)
    log_scales = torch.log(torch.tensor([[1.0, 1.0, 10.0]]))

    l1, ssim, alpha, scale = training._training_loss(
        rgb, rgb, predicted_alpha, target_alpha, None, 0.1, log_scales, True,
    )

    assert l1.item() == 0
    assert ssim.item() == 0
    assert alpha.item() == pytest.approx(0.1)
    assert scale.item() == pytest.approx(0.5)


def test_training_node_passes_scene_and_parameters_to_its_own_trainer(monkeypatch, tmp_path):
    output = str(tmp_path / "trained.ply")
    splat = object()
    captured = []
    monkeypatch.setattr(training, "_output_path", lambda: output)
    monkeypatch.setattr(training, "_runtime_helpers", lambda _steps: (None, None))

    def trainer(*args):
        captured.extend(args)
        return splat

    monkeypatch.setattr(training, "_train_gaussians", trainer)
    scene = (object(), {})
    result = training.VNCCSGaussianTrainer().train(
        scene, training_steps=1, max_splats=1, optimizer_mode="adam",
    )

    assert result == (output, splat)
    assert captured[:5] == [scene[0], scene[1], output, 1, 1]
    assert captured[10] == "adam"


def test_strategy_uses_standard_clone_split_with_hard_cap(monkeypatch):
    class DefaultStrategy:
        grow_scale3d = 0.01
        grow_scale2d = 0.05
        revised_opacity = False

        def __init__(self, **options):
            self.__dict__.update(options)

    calls = []

    def duplicate(**kwargs):
        calls.append(("clone", kwargs["mask"].clone()))

    def split(**kwargs):
        calls.append(("split", kwargs["mask"].clone()))

    monkeypatch.setitem(sys.modules, "gsplat.strategy", SimpleNamespace(DefaultStrategy=DefaultStrategy))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(duplicate=duplicate, split=split))
    strategy = training._strategy(8_000, 150, 3_500, 100, 99_990, 0.0001, 0.005)
    params = {
        "means": torch.zeros((3, 3)),
        "scales": torch.log(torch.tensor([[0.001] * 3, [0.1] * 3, [0.1] * 3])),
    }
    state = {
        "grad2d": torch.tensor([0.002, 0.003, 0.1]),
        "count": torch.tensor([1.0, 1.0, 1000.0]),
        "radii": torch.zeros(3),
        "scene_scale": 1.0,
        "max_gaussians": 4,
    }

    cloned, divided = strategy._grow_gs(params, {}, state, 200)

    assert (cloned, divided) == (0, 1)
    assert calls[0][0] == "split"
    assert calls[0][1].tolist() == [False, True, False]
    assert strategy.refine_stop_iter == 3_500


def test_training_reenables_autograd_inside_comfy_inference_mode(monkeypatch):
    modes = {}

    def unavailable_cuda():
        modes["inference"] = torch.is_inference_mode_enabled()
        modes["grad"] = torch.is_grad_enabled()
        return False

    monkeypatch.setattr(torch.cuda, "is_available", unavailable_cuda)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA build"):
            training._train_gaussians(
                None, None, None, 1, 1, 1, 0.2, 0.1, True, True,
                "adam", 150, 3_500, 100, 99_990, 0.0001, 0.005, 42, None, None,
            )

    assert modes == {"inference": False, "grad": True}


def test_training_stage_keeps_scene_input_and_file_and_splat_outputs():
    inputs = training.VNCCSGaussianTrainer.INPUT_TYPES()["required"]

    assert inputs["scene"] == (training.GAUSSIAN_SCENE_TYPE,)
    assert "images" not in inputs
    assert "max_resolution" not in inputs
    assert training.VNCCSGaussianTrainer.RETURN_TYPES == ("STRING", "SPLAT")
    assert training.VNCCSGaussianTrainer.RETURN_NAMES == ("gaussian_file", "splat")
    assert training.NODE_CLASS_MAPPINGS == {
        "VNCCS_GaussianTrainer": training.VNCCSGaussianTrainer,
    }
    assert training.NODE_DISPLAY_NAME_MAPPINGS == {
        "VNCCS_GaussianTrainer": "VNCCS Gaussian Trainer",
    }


@pytest.mark.parametrize("radii", [[2.0, 3.0, 4.0], [2.0]])
def test_comfy_splat_uses_activated_batched_cpu_tensors(monkeypatch, radii):
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
        "scales": torch.nn.Parameter(torch.log(torch.tensor([radii]))),
        "quats": torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
        "opacities": torch.nn.Parameter(torch.tensor([0.0])),
        "sh0": torch.nn.Parameter(torch.ones((1, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((1, 3, 3))),
    }

    positions, scales, rotations, opacities, sh = training._comfy_splat(params).values

    assert positions.shape == (1, 1, 3)
    expected_scales = radii if len(radii) == 3 else radii * 3
    assert torch.allclose(scales, torch.tensor([[expected_scales]]))
    assert torch.allclose(rotations, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    assert opacities.shape == (1, 1, 1)
    assert opacities.item() == pytest.approx(0.5)
    assert sh.shape == (1, 1, 4, 3)
    assert all(
        value.device.type == "cpu"
        for value in (positions, scales, rotations, opacities, sh)
    )


def test_pytorch_ssim_loss_is_zero_for_identical_images():
    image = torch.rand((1, 16, 16, 3))
    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()

    assert training._ssim_loss(image, image, gaussian).item() == pytest.approx(0.0, abs=1e-6)
    assert training._ssim_loss(image, 1.0 - image, gaussian).item() > 0.1


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

    assert training._ssim_loss(rendered, target, gaussian).item() == pytest.approx(
        reference_loss.item(), abs=1e-5
    )
