"""Native MoGe integration and Gaussian depth-supervision contracts."""

from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

import nodes.vnccs_gaussians as depth


def test_depth_scale_uses_observed_foreground_points_and_ignores_invalid_pixels():
    predicted = np.array([[2.0, 3.0], [np.inf, 5.0]], dtype=np.float32)
    valid = np.array([[True, True], [False, False]])
    samples = [
        [0.5, 0.5, 4.0], [1.5, 0.5, 6.0], [0.5, 1.5, 99.0],
        [-1, 0, 1], [2, 0, 1], [0, 0, -1], [np.nan, 0, 1],
    ]

    aligned = depth._align_depth(predicted, valid, samples)

    np.testing.assert_array_equal(aligned, [[4, 6], [0, 0]])
    assert aligned.dtype == np.float32
    assert depth._align_depth(predicted, valid, []) is None
    assert depth._align_depth(predicted, valid, [[0.5, 1.5, 1]]) is None


def test_sparse_depth_samples_keep_only_real_observations_of_finite_points():
    def observation(point_id, xy, registered=True):
        return SimpleNamespace(
            point3D_id=point_id, xy=xy, has_point3D=lambda: registered,
        )

    model = SimpleNamespace(
        points3D={
            10: SimpleNamespace(xyz=np.array([0, 0, 6]), color=np.ones(3)),
            20: SimpleNamespace(xyz=np.array([0, 0, 8]), color=np.full(3, np.nan)),
            30: SimpleNamespace(xyz=np.array([2, 2, 10]), color=np.ones(3)),
        },
        images={7: SimpleNamespace(name="frame.png", points2D=[
            observation(30, [1.5, 0.5]), observation(10, [0.5, 0.5]),
            observation(20, [0.5, 1.5]), observation(99, [0, 0]),
            observation(10, [0, 0], registered=False),
        ])},
    )
    points = np.array([[0, 0, 3], [1, 1, 5]], dtype=np.float32)
    frames = [{"name": "frame.png", "view": np.eye(4)}]

    samples = depth._depth_samples(model, points, frames)

    np.testing.assert_array_equal(samples["frame.png"], [[1.5, 0.5, 5], [0.5, 0.5, 3]])


def _frames_and_cache(count):
    frames, cached = [], {}
    for index in range(count):
        rgb = np.full((2, 2, 3), index * 20, dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        name = f"{index:05d}.png"
        frames.append({
            "name": name,
            "K": np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32),
            "rgba": np.full((2, 2, 4), 255, dtype=np.uint8),
        })
        cached[name] = {"source_png": buffer.getvalue()}
    return frames, cached


def _native_moge(monkeypatch, inference):
    calls = []
    model = SimpleNamespace(patcher=object())

    class Loader:
        @classmethod
        def execute(cls, model_name):
            assert not torch.is_grad_enabled()
            calls.append(("load", model_name))
            return (model,)

    management = SimpleNamespace(
        unload_model_and_clones=lambda patcher: calls.append(("unload", patcher)),
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", management, raising=False)
    package = ModuleType("comfy_extras")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_moge", SimpleNamespace(
        LoadMoGeModel=Loader, MoGeInference=SimpleNamespace(execute=inference),
    ))
    return model, calls


@pytest.mark.parametrize("pre_cached", [0, 1])
def test_native_moge_uses_raw_rgb_camera_fov_and_reuses_depth_cache(monkeypatch, pre_cached):
    frames, cached = _frames_and_cache(6)
    if pre_cached:
        cached[frames[0]["name"]]["moge_depth"] = np.ones((2, 2), dtype=np.float32)
    inferences = []

    def infer(**kwargs):
        assert not torch.is_grad_enabled()
        inferences.append(kwargs)
        count = len(kwargs["image"])
        prediction = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).expand(count, -1, -1)
        return ({"depth": prediction, "mask": torch.ones_like(prediction, dtype=torch.bool)},)

    model, calls = _native_moge(monkeypatch, infer)
    monkeypatch.setattr(depth, "_depth_samples", lambda _model, _points, pending: {
        frame["name"]: [[0.5, 0.5, 4], [1.5, 0.5, 6]] for frame in pending
    })

    depth.prepare_depth_targets(None, cached, None, frames, lambda: None)

    assert calls == [("load", depth.MOGE_MODEL_NAME), ("unload", model.patcher)]
    assert [len(call["image"]) for call in inferences] == [4, 2 - pre_cached]
    for call in inferences:
        assert call["moge_model"] is model
        assert call["resolution_level"] == 9
        assert call["fov_x_degrees"] == pytest.approx(90)
        assert call["force_projection"] is True
        assert call["apply_mask"] is True
        assert call["refine_steps"] == 3
    first_rgb = inferences[0]["image"][0]
    assert torch.allclose(first_rgb, torch.full_like(first_rgb, pre_cached * 20 / 255))
    for frame in frames[pre_cached:]:
        np.testing.assert_array_equal(frame["depth"], [[4, 6], [8, 10]])
        assert frame["depth"] is cached[frame["name"]]["moge_depth"]

    fresh_frames = [{key: value for key, value in frame.items() if key != "depth"} for frame in frames]
    depth.prepare_depth_targets(None, cached, None, fresh_frames, lambda: None)
    assert len(inferences) == 2
    assert len(calls) == 2
    for frame in fresh_frames:
        assert frame["depth"] is cached[frame["name"]]["moge_depth"]


def test_native_moge_unloads_its_model_when_inference_fails(monkeypatch):
    frames, cached = _frames_and_cache(1)

    def fail(**_kwargs):
        raise RuntimeError("inference failed")

    model, calls = _native_moge(monkeypatch, fail)
    monkeypatch.setattr(depth, "_depth_samples", lambda *_args: {})

    with pytest.raises(RuntimeError, match="inference failed"):
        depth.prepare_depth_targets(None, cached, None, frames, lambda: None)

    assert calls[-1] == ("unload", model.patcher)
    assert "moge_depth" not in cached[frames[0]["name"]]


def test_depth_requires_original_rgb_instead_of_visualization_or_matted_frames():
    frames, cached = _frames_and_cache(1)
    cached[frames[0]["name"]].pop("source_png")

    with pytest.raises(ValueError, match="rerun VNCCS Gaussian Camera Reconstruction"):
        depth.prepare_depth_targets(None, cached, None, frames, lambda: None)


def test_depth_cache_does_not_import_native_moge_again(monkeypatch):
    frames, cached = _frames_and_cache(1)
    expected = np.ones((2, 2), dtype=np.float32)
    cached[frames[0]["name"]]["moge_depth"] = expected
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_moge", None)

    depth.prepare_depth_targets(None, cached, None, frames, lambda: None)

    assert frames[0]["depth"] is expected


def test_depth_loss_masks_invalid_background_and_has_finite_foreground_gradients():
    predicted = torch.tensor([4.0, 100.0, float("nan"), 0.0, 8.0], requires_grad=True)
    target = torch.tensor([2.0, 0.0, 3.0, 5.0, float("inf")])
    alpha = torch.ones_like(predicted)

    loss = depth.depth_loss(predicted, target, alpha)
    loss.backward()

    assert loss.item() == pytest.approx(np.log(2))
    assert torch.allclose(predicted.grad, torch.tensor([0.25, 0, 0, 0, 0]))
    assert torch.isfinite(predicted.grad).all()


def test_depth_loss_is_zero_when_no_supervised_splat_is_visible():
    predicted = torch.ones(4, requires_grad=True)

    loss = depth.depth_loss(predicted, torch.ones(4), torch.zeros(4))
    loss.backward()

    assert loss.item() == 0
    assert torch.equal(predicted.grad, torch.zeros(4))
