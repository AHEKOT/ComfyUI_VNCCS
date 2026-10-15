"""Contracts for the standalone HYWorld2 orbit trainer."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nodes.gaussian_training_hyworld2 as training


def test_scene_scale_uses_camera_extent_instead_of_subject_thickness():
    frames = []
    for position in ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]):
        view = np.eye(4)
        view[:3, 3] = -np.asarray(position)
        frames.append({"view": view})

    assert training._hyworld2_scene_scale(frames) == pytest.approx(2.2)


def test_initial_scales_use_rms_of_three_neighbors_without_brush_floor(monkeypatch):
    points = np.array([[0, 0, 0], [0.0001, 0, 0], [0.0002, 0, 0], [0.0003, 0, 0]], dtype=np.float32)

    class Index:
        def knnSearch(self, query, count, params):
            assert count == 4
            neighbors = np.tile(np.arange(4), (4, 1))
            return neighbors, np.zeros_like(neighbors)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(flann_Index=lambda *_args: Index()))
    scales = np.exp(training._hyworld2_initial_scales(points))

    assert scales[0] == pytest.approx(np.sqrt((1e-8 + 4e-8 + 9e-8) / 3))
    assert scales[0] < 1e-3


def test_alpha_loss_and_physgaussian_penalty_are_independent():
    rgb = torch.zeros((1, 16, 16, 3))
    predicted_alpha = torch.zeros((1, 16, 16, 1))
    target_alpha = torch.ones_like(predicted_alpha)
    log_scales = torch.log(torch.tensor([[1.0, 1.0, 10.0]]))

    l1, ssim, alpha, scale = training._hyworld2_loss(
        rgb, rgb, predicted_alpha, target_alpha, None, 0.1, log_scales, True,
    )

    assert l1.item() == 0
    assert ssim.item() == 0
    assert alpha.item() == pytest.approx(0.1)
    assert scale.item() == pytest.approx(0.5)


def test_training_node_passes_scene_and_parameters_to_its_own_trainer(monkeypatch, tmp_path):
    output = str(tmp_path / "hyworld2.ply")
    splat = object()
    captured = []
    monkeypatch.setattr(training, "_output_path", lambda: output)
    monkeypatch.setattr(training, "_runtime_helpers", lambda _steps: (None, None))

    def trainer(*args):
        captured.extend(args)
        return splat

    monkeypatch.setattr(training, "_train_hyworld2_gaussians", trainer)
    scene = (object(), {})
    result = training.VNCCSGaussianTrainingHYWorld2().train(
        scene, training_steps=1, max_splats=1, optimizer_mode="adam",
    )

    assert result == (output, splat)
    assert captured[:5] == [scene[0], scene[1], output, 1, 1]
    assert captured[10] == "adam"


def test_hyworld2_strategy_uses_standard_clone_split_with_hard_cap(monkeypatch):
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
    strategy = training._hyworld2_strategy(8_000, 150, 3_500, 100, 99_990, 0.0001, 0.005)
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
