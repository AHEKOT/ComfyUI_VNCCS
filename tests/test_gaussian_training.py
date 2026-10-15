"""Contracts for the standalone Gaussian orbit trainer."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import nodes.vnccs_gaussians as training


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


@pytest.mark.parametrize("use_depth_loss", [False, True])
@pytest.mark.parametrize("use_2dgs", [False, True])
def test_training_node_passes_scene_and_parameters_to_its_own_trainer(
    monkeypatch, tmp_path, use_depth_loss, use_2dgs,
):
    output = str(tmp_path / "trained.ply")
    splat = object()
    images = tuple(object() for _ in range(8))
    captured = []
    monkeypatch.setattr(training, "_output_path", lambda: output)
    monkeypatch.setattr(training, "_runtime_helpers", lambda _steps: (None, None))

    def trainer(*args):
        captured.extend(args)
        return (splat, *images)

    monkeypatch.setattr(training, "_train_gaussians", trainer)
    scene = (object(), {})
    result = training.VNCCSGaussianTrainer().train(
        scene, training_steps=1, max_splats=1, optimizer_mode="adam", use_depth_loss=use_depth_loss,
        use_2dgs=use_2dgs, distortion_weight=777.0, normal_weight=0.03,
    )

    assert result == (output, splat, *images)
    assert captured[:5] == [scene[0], scene[1], output, 1, 1]
    assert captured[10] == "adam"
    assert captured[-4:] == [use_depth_loss, use_2dgs, 777.0, 0.03]


@pytest.mark.parametrize("options, message", [
    ({"distortion_weight": -0.1}, "distortion_weight must be finite and non-negative"),
    ({"distortion_weight": float("nan")}, "distortion_weight must be finite and non-negative"),
    ({"normal_weight": -0.1}, "normal_weight must be finite and non-negative"),
    ({"normal_weight": float("inf")}, "normal_weight must be finite and non-negative"),
])
def test_surface_options_fail_before_training_when_enabled(options, message):
    with pytest.raises(ValueError, match=message):
        training.VNCCSGaussianTrainer().train((object(), {}), use_2dgs=True, **options)


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


def test_2dgs_strategy_uses_surfel_gradient_and_does_not_clone_screen_split_candidates(monkeypatch):
    class DefaultStrategy:
        grow_scale3d = 0.01
        grow_scale2d = 0.05
        revised_opacity = False

        def __init__(self, **options):
            self.__dict__.update(options)

    calls = []
    monkeypatch.setitem(sys.modules, "gsplat.strategy", SimpleNamespace(DefaultStrategy=DefaultStrategy))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(
        duplicate=lambda **kwargs: calls.append("unexpected clone"),
        split=lambda **kwargs: calls.append("unexpected 3D split"),
    ))
    monkeypatch.setattr(training, "split_2dgs", lambda params, optimizers, state, mask: calls.append(mask.tolist()))
    strategy = training._strategy(8000, 150, 3500, 100, 99990, 0.0001, 0.005, use_2dgs=True)
    params = {"means": torch.zeros((2, 3)), "scales": torch.full((2, 2), -8.0)}
    state = {"grad2d": torch.tensor([0.1, 0.2]), "count": torch.ones(2),
             "radii": torch.tensor([0.2, 0.0]), "scene_scale": 1.0, "max_gaussians": 3}
    assert strategy.key_for_gradient == "gradient_2dgs"
    # The cap admits only the highest-gradient candidate, a clone of the second surfel.
    # Raising the first candidate's gradient must make it a split, never clone+split.
    state["grad2d"][0] = 0.3
    assert strategy._grow_gs(params, {}, state, 200) == (0, 1)
    assert calls == [[True, False]]


def test_2dgs_resets_opacity_and_stops_refining_at_the_requested_step(monkeypatch):
    class DefaultStrategy:
        pause_refine_after_reset = 0

        def __init__(self, **options):
            self.__dict__.update(options)

        def _update_state(self, params, state, info, packed):
            state["updated"] = state.get("updated", 0) + 1

    calls = []
    monkeypatch.setitem(sys.modules, "gsplat.strategy", SimpleNamespace(DefaultStrategy=DefaultStrategy))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(
        duplicate=lambda **_kwargs: None, split=lambda **_kwargs: None,
        reset_opa=lambda **kwargs: calls.append(kwargs["value"]),
    ))
    strategy = training._strategy(20, 150, 15, 100, 5, 0.0001, 0.005, use_2dgs=True)
    state = {}
    strategy.step_post_backward({}, {}, state, 0, {})
    strategy.step_post_backward({}, {}, state, 5, {})
    strategy.step_post_backward({}, {}, state, 15, {})
    assert state["updated"] == 2
    assert calls == [0.01]


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


def test_old_scene_without_source_rgb_fails_before_training(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(export_splats=lambda: None))
    monkeypatch.setitem(sys.modules, "gsplat.rendering", SimpleNamespace(rasterization=lambda: None))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(remove=lambda: None))
    monkeypatch.setattr(training, "_scene_data", lambda *_args: (None, None, []))
    monkeypatch.setattr(training, "_top_view_cameras", lambda _frames: [({"source_rgb": None}, None)])

    with pytest.raises(ValueError, match="rerun VNCCS Gaussian Camera Reconstruction"):
        training._train_gaussians(
            None, None, None, 1, 1, 1, 0.2, 0.1, True, True,
            "adam", 150, 3_500, 100, 99_990, 0.0001, 0.005, 42, None, None,
        )


def test_training_stage_keeps_scene_input_and_file_and_splat_outputs():
    inputs = training.VNCCSGaussianTrainer.INPUT_TYPES()["required"]

    assert inputs["scene"] == (training.GAUSSIAN_SCENE_TYPE,)
    assert "images" not in inputs
    assert "max_resolution" not in inputs
    assert inputs["use_depth_loss"][0] == "BOOLEAN"
    assert inputs["use_depth_loss"][1]["default"] is False
    assert list(inputs)[-5:] == [
        "seed", "use_depth_loss", "use_2dgs", "distortion_weight", "normal_weight",
    ]
    assert inputs["use_2dgs"][1]["default"] is False
    assert inputs["distortion_weight"][1]["default"] == 1_000
    assert inputs["normal_weight"][1]["default"] == 0.05
    assert "max" not in inputs["distortion_weight"][1]
    assert "max" not in inputs["normal_weight"][1]
    assert "surface_refinement" not in inputs
    assert training.VNCCSGaussianTrainer.RETURN_TYPES == ("STRING", "SPLAT") + ("IMAGE",) * 8
    assert training.VNCCSGaussianTrainer.RETURN_NAMES == (
        "gaussian_file", "splat", "top_0", "source_0", "top_90", "source_90",
        "top_180", "source_180", "top_270", "source_270",
    )
    assert training.NODE_CLASS_MAPPINGS == {
        "VNCCS_GaussianCameraReconstruction": training.VNCCSGaussianCameraReconstruction,
        "VNCCS_GaussianTrainer": training.VNCCSGaussianTrainer,
    }
    assert training.NODE_DISPLAY_NAME_MAPPINGS == {
        "VNCCS_GaussianCameraReconstruction": "VNCCS Gaussian Camera Reconstruction",
        "VNCCS_GaussianTrainer": "VNCCS Gaussian Trainer",
    }


@pytest.mark.parametrize("fail", [False, True])
def test_export_cleans_finite_parameters_normalizes_rotations_and_publishes_atomically(monkeypatch, tmp_path, fail):
    params = {
        "means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0]])),
        "scales": torch.nn.Parameter(torch.zeros((2, 3))),
        "quats": torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 2)),
        "opacities": torch.nn.Parameter(torch.zeros(2)),
        "sh0": torch.nn.Parameter(torch.zeros((2, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((2, 0, 3)), requires_grad=False),
    }
    output = tmp_path / "trained.ply"
    output.write_bytes(b"previous PLY")
    removed = []

    def remove(*, params, optimizers, state, mask):
        assert not torch.is_grad_enabled()
        removed.append(mask.tolist())
        for name, value in list(params.items()):
            params[name] = torch.nn.Parameter(value[~mask], requires_grad=value.requires_grad)

    def export_splats(**kwargs):
        assert not torch.is_grad_enabled()
        assert output.read_bytes() == b"previous PLY"
        assert Path(kwargs["save_to"]) != output
        assert kwargs["format"] == "ply"
        assert kwargs["means"].shape == (1, 3)
        assert torch.equal(kwargs["quats"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        Path(kwargs["save_to"]).write_bytes(b"complete PLY")
        if fail:
            raise RuntimeError("export failed")

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(export_splats=export_splats))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(remove=remove))

    if fail:
        with pytest.raises(RuntimeError, match="export failed"):
            training._export_gaussians(params, {}, {}, str(output))
    else:
        training._export_gaussians(params, {}, {}, str(output))

    assert removed == [[False, True]]
    assert output.read_bytes() == (b"previous PLY" if fail else b"complete PLY")
    assert list(tmp_path.iterdir()) == [output]


def test_2dgs_export_keeps_native_two_scales_and_publishes_thin_3dgs_compatibility(monkeypatch, tmp_path):
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]])),
        "scales": torch.nn.Parameter(torch.log(torch.tensor([[0.2, 0.3]]))),
        "quats": torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
        "opacities": torch.nn.Parameter(torch.zeros(1)),
        "sh0": torch.nn.Parameter(torch.zeros((1, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((1, 0, 3)), requires_grad=False),
    })
    captured = []

    def native_ply(means, scales, quats, opacities, sh0, shN):
        assert scales.shape == (1, 2)
        assert sh0.shape == (1, 3) and shN.shape == (1, 0)
        assert torch.equal(quats, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        return b"native 2DGS"

    def export_splats(**kwargs):
        captured.append(kwargs["scales"].detach().clone())
        Path(kwargs["save_to"]).write_bytes(b"compatibility 3DGS")

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(export_splats=export_splats))
    monkeypatch.setitem(sys.modules, "gsplat.exporter", SimpleNamespace(splat2ply_bytes=native_ply))
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(remove=lambda **_kwargs: None))
    output = tmp_path / "trained.ply"
    training._export_gaussians(params, {}, {}, str(output))
    assert output.read_bytes() == b"compatibility 3DGS"
    assert (tmp_path / "trained_2dgs.ply").read_bytes() == b"native 2DGS"
    assert torch.allclose(captured[0].exp(), torch.tensor([[0.2, 0.3, 0.0002]]))
    assert params["scales"].shape == (1, 2), "Export must not change the trained representation"


def test_2dgs_previews_use_native_surfel_renderer_not_the_compatibility_ellipsoids(monkeypatch):
    def forbidden(**_kwargs):
        pytest.fail("2DGS previews must not use the 3DGS renderer")

    calls = []

    def render(splats, view, intrinsics, width, height, degree):
        calls.append(degree)
        assert splats["scales"].shape == (1, 2)
        return torch.full((1, height, width, 3), 0.25), torch.ones((1, height, width, 1)), {}

    monkeypatch.setitem(sys.modules, "gsplat.rendering", SimpleNamespace(rasterization=forbidden))
    monkeypatch.setattr(training, "render_2dgs", render)
    params = {"means": torch.zeros((1, 3)), "scales": torch.zeros((1, 2)),
              "opacities": torch.zeros(1), "sh0": torch.zeros((1, 1, 3)), "shN": torch.zeros((1, 3, 3))}
    frame = {"name": "frame.png", "source_rgb": np.full((8, 6, 3), 255, dtype=np.uint8),
             "K": np.eye(3, dtype=np.float32)}
    images = training._render_view_pairs(params, [(frame, np.eye(4, dtype=np.float32))] * 4, 1, True, lambda: None)
    assert calls == [1] * 4 and len(images) == 8
    assert torch.equal(images[0], torch.full((1, 8, 6, 3), 0.25))


@pytest.mark.parametrize("antialiased", [False, True])
def test_preview_pairs_use_fixed_views_and_unmodified_source_rgb(monkeypatch, antialiased):
    calls = []
    interrupts = []

    def rasterization(**kwargs):
        assert not torch.is_grad_enabled()
        calls.append(kwargs)
        shape = (1, kwargs["height"], kwargs["width"])
        return torch.full((*shape, 3), 0.25), torch.full((*shape, 1), 0.5), {}

    monkeypatch.setitem(sys.modules, "gsplat.rendering", SimpleNamespace(rasterization=rasterization))
    splats = {
        "means": torch.zeros((1, 3)), "scales": torch.zeros((1, 3)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "opacities": torch.zeros(1), "sh0": torch.zeros((1, 1, 3)),
        "shN": torch.zeros((1, 3, 3)),
    }
    previews = []
    for index in range(4):
        view = np.eye(4, dtype=np.float32)
        view[0, 3] = index
        frame = {
            "name": f"{index:05d}.png", "K": np.eye(3, dtype=np.float32),
            "source_rgb": np.full((8, 6, 3), 32 * (index + 1), dtype=np.uint8),
        }
        previews.append((frame, view))

    images = training._render_view_pairs(
        splats, previews, 1, antialiased, lambda: interrupts.append(True),
    )

    assert len(images) == 8
    assert len(interrupts) == 4
    for index, (frame, view) in enumerate(previews):
        rendered, source = images[index * 2:index * 2 + 2]
        expected = torch.from_numpy(frame["source_rgb"]).float()[None] / 255.0
        assert torch.equal(source, expected)
        assert torch.allclose(rendered, 0.25 + expected * 0.5)
        assert source.shape == rendered.shape == (1, 8, 6, 3)
        assert source.dtype == rendered.dtype == torch.float32
        assert source.device.type == rendered.device.type == "cpu"
        assert torch.equal(calls[index]["viewmats"], torch.from_numpy(view)[None])
        assert calls[index]["rasterize_mode"] == ("antialiased" if antialiased else "classic")
        assert calls[index]["sh_degree"] == 1


@pytest.mark.parametrize("radii", [[2.0, 3.0, 4.0], [2.0], [2.0, 3.0]])
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
    expected_scales = radii if len(radii) == 3 else (radii + [0.002] if len(radii) == 2 else radii * 3)
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
