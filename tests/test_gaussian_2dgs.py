"""Contracts for fused 2DGS rendering; model execution is never required."""

import math
import inspect
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import nodes.vnccs_gaussians as surface


def _splats(dtype=torch.float64):
    return {
        "means": torch.tensor([[0.0, 0.0, 2.0], [0.15, 0.05, 3.0]], dtype=dtype, requires_grad=True),
        "quats": torch.tensor([[math.cos(0.25), 0.0, math.sin(0.25), 0.0],
                               [math.cos(0.1), -math.sin(0.1), 0.0, 0.0]], dtype=dtype, requires_grad=True),
        "scales": torch.zeros((2, 2), dtype=dtype, requires_grad=True),
        "opacities": torch.zeros(2, dtype=dtype, requires_grad=True),
        "sh0": torch.zeros((2, 1, 3), dtype=dtype, requires_grad=True),
        "shN": torch.zeros((2, 3, 3), dtype=dtype, requires_grad=True),
    }


def _rays(K, width, height):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pixels = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), -1).to(K)
    return pixels @ torch.linalg.inv(K[0]).T


def test_inverse_depth_plane_matches_independent_ray_plane_intersections_and_pose():
    params = _splats()
    view = torch.eye(4, dtype=torch.float64)[None]
    c, s = math.cos(0.4), math.sin(0.4)
    view[0, :3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
    view[0, 0, 3] = 0.3
    planes, valid, _ = surface._inverse_depth_planes(params["means"], params["quats"], view)
    rays = torch.tensor([[-0.2, 0.0, 1.0], [0.0, 0.0, 1.0], [0.3, 0.1, 1.0]], dtype=torch.float64)
    normal = view[0, :3, :3] @ torch.tensor([math.sin(0.5), 0.0, math.cos(0.5)], dtype=torch.float64)
    center = view[0, :3, :3] @ params["means"][0] + view[0, :3, 3]
    expected = (normal @ center) / (rays @ normal)
    actual = 1 / (rays @ planes[0])
    assert valid.all() and torch.allclose(actual, expected, atol=1e-12)
    assert not torch.allclose(actual, actual[0].expand_as(actual))
    actual.sum().backward()
    for value in (params["means"], params["quats"]):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_camera_plane_singularity_has_finite_zero_geometry():
    params = _splats()
    means = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    planes, valid, reference = surface._inverse_depth_planes(
        means, params["quats"][:1], torch.eye(4, dtype=torch.float64)[None],
    )
    assert not valid.any() and not planes.any() and torch.isfinite(reference)
    planes.sum().backward()
    assert torch.isfinite(means.grad).all()


def test_geometry_dispatch_reuses_gsplat_projection_and_tiles_without_feature_images(monkeypatch):
    params = _splats(torch.float32)
    calls = []
    maps = torch.randn((1, 8, 6, 6), requires_grad=True)
    view, K = torch.eye(4)[None], torch.eye(3)[None]

    def projection(means, quats, scales, *_args, **options):
        calls.append("projection")
        assert scales.shape == (2, 3) and torch.equal(scales[:, 2], torch.ones(2))
        assert options["packed"] is False
        return (torch.ones((1, 2, 2), dtype=torch.int32), torch.zeros((1, 2, 2)),
                torch.ones((1, 2)), torch.zeros((1, 2, 3, 3)), torch.ones((1, 2, 3)))

    def tiles(xy, radii, depths, tile_size, tile_width, tile_height, **_kwargs):
        calls.append("tiles")
        assert (tile_size, tile_width, tile_height) == (16, 1, 1)
        return torch.ones((1, 2)), torch.arange(2), torch.arange(2, dtype=torch.int32)

    def render(*args):
        calls.append("render")
        assert args[2].shape == (1, 2, 3), "Only RGB, not sixteen feature channels"
        assert args[5].shape == (2, 3) and args[6].shape == (2, 1)
        assert args[10].requires_grad, "Densification must retain its gradient input"
        return torch.zeros((1, 8, 6, 3)), torch.zeros((1, 8, 6, 1)), maps

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(
        spherical_harmonics=lambda degree, dirs, coefficients, **_kwargs: coefficients[..., 0, :] * 0.28209479177387814))
    monkeypatch.setitem(sys.modules, "gsplat.cuda._wrapper", SimpleNamespace(
        fully_fused_projection_2dgs=projection, isect_tiles=tiles,
        isect_offset_encode=lambda *_args: torch.zeros((1, 1, 1), dtype=torch.int32)))
    monkeypatch.setattr(surface, "rasterize_surface", render)
    image, alpha, info = surface.render_2dgs(params, view, K, 6, 8, 1, with_geometry=True)
    assert calls == ["projection", "tiles", "render"]
    assert image.shape == (1, 8, 6, 3) and alpha.shape == (1, 8, 6, 1)
    assert surface.geometry_maps(info, K) is maps


def test_native_rgb_only_renderer_remains_three_channel_and_does_not_build_geometry(monkeypatch):
    params = _splats(torch.float32)
    captured = []

    def render(**kwargs):
        captured.append(kwargs)
        rgb = torch.zeros((1, 8, 6, 3))
        alpha = torch.zeros((1, 8, 6, 1))
        return rgb, alpha, None, None, None, None, {"gradient_2dgs": "gradient"}

    monkeypatch.setitem(sys.modules, "gsplat.rendering", SimpleNamespace(rasterization_2dgs=render))
    rgb, alpha, info = surface.render_2dgs(params, torch.eye(4)[None], torch.eye(3)[None], 6, 8, 1)
    assert rgb.shape == (1, 8, 6, 3) and alpha.shape == (1, 8, 6, 1)
    assert info == {"gradient_2dgs": "gradient"}
    assert captured[0]["sh_degree"] == 1 and captured[0]["colors"].shape == (2, 4, 3)
    assert captured[0]["render_mode"] == "RGB" and captured[0]["distloss"] is False


def test_normal_consistency_and_moge_use_exact_median_depth_in_world_coordinates(monkeypatch):
    captured = []

    def depth_to_normal(depth, camera_to_world, intrinsics):
        captured.append(camera_to_world)
        points = depth * _rays(intrinsics, 5, 5)[None]
        points = points @ camera_to_world[0, :3, :3].T + camera_to_world[0, :3, 3]
        dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
        dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
        return F.pad(F.normalize(torch.cross(dx, dy, dim=-1), dim=-1), (0, 0, 1, 1, 1, 1))

    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(depth_to_normal=depth_to_normal))
    median = torch.full((1, 5, 5, 1), 2.0, requires_grad=True)
    normal = torch.tensor([0.1, 0.0, -0.99]).expand(1, 5, 5, 3).clone().requires_grad_()
    maps = torch.cat((median, torch.ones_like(median), torch.zeros_like(median), normal), -1)
    view = torch.eye(4)[None]
    view[0, 0, 3] = 3
    distortion, consistency, supervised_depth = surface.geometry_losses(maps, view, torch.eye(3)[None])
    consistency.backward()
    assert torch.equal(captured[0], torch.linalg.inv(view))
    assert torch.equal(supervised_depth, median) and distortion.item() == 0
    assert consistency.item() > 0
    assert median.grad.abs().sum() > 0 and torch.isfinite(median.grad).all()
    assert normal.grad.abs().sum() > 0
    _, disabled_normal, supervised_depth = surface.geometry_losses(maps.detach(), view, torch.eye(3)[None], use_normal=False)
    assert disabled_normal.item() == 0 and torch.equal(supervised_depth, median.detach())


def test_geometry_path_has_no_contributor_enumeration_checkpoint_or_host_transfers():
    source = "\n".join(inspect.getsource(function) for function in (
        surface.compatibility_scales, surface._inverse_depth_planes,
        surface.render_2dgs, surface._render_surface, surface.geometry_maps,
        surface.geometry_losses, surface.geometry_schedule, surface.split_2dgs,
    ))
    for forbidden in ("rasterize_to_indices_in_range", "torch.utils.checkpoint", "_geometry_stripe",
                      "torch.cumsum", "torch.cummax", ".cpu()", ".tolist()"):
        assert forbidden not in source


def test_planar_split_preserves_plane_optimizer_state_and_frozen_sh(monkeypatch):
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(
        normalized_quat_to_rotmat=lambda quats: rotation[None].expand(len(quats), -1, -1),
    ))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        "scales": torch.nn.Parameter(torch.log(torch.tensor([[0.2, 0.4], [0.1, 0.3]]))),
        "quats": torch.nn.Parameter(torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]] * 2)),
        "opacities": torch.nn.Parameter(torch.zeros(2)),
        "sh0": torch.nn.Parameter(torch.zeros((2, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((2, 0, 3)), requires_grad=False),
    })
    optimizers = {name: torch.optim.Adam([value], lr=0.01) for name, value in params.items() if value.requires_grad}
    sum(value.square().sum() for value in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    original = {name: value.detach().clone() for name, value in params.items()}
    moments = {name: optimizer.state[params[name]]["exp_avg"].clone() for name, optimizer in optimizers.items()}
    state = {"grad2d": torch.tensor([0.1, 0.2]), "count": torch.tensor([5.0, 7.0]), "scene_scale": 1.0}
    surface.split_2dgs(params, optimizers, state, torch.tensor([True, False]))
    assert params["scales"].shape == (3, 2)
    assert torch.equal(params["means"][0], original["means"][1])
    offset = params["means"][1:] - original["means"][0]
    assert torch.allclose(offset @ rotation[:, 2], torch.zeros(2), atol=1e-7)
    assert torch.allclose(params["scales"][1:].exp(), original["scales"][:1].exp().expand(2, -1) / 1.6)
    assert not params["shN"].requires_grad and state["count"].tolist() == [7.0, 5.0, 5.0]
    for name, optimizer in optimizers.items():
        assert optimizer.param_groups[0]["params"][0] is params[name]
        moment = optimizer.state[params[name]]["exp_avg"]
        assert torch.equal(moment[0], moments[name][1]) and not moment[1:].any()


def test_short_training_has_proportional_warmups_and_no_extra_training_steps():
    assert surface.geometry_schedule(0, 1, 1000, 0.05) == (1000, 0.05)
    assert surface.geometry_schedule(799, 8000, 1000, 0.05) == (0, 0)
    assert surface.geometry_schedule(800, 8000, 1000, 0.05) == (1000, 0)
    assert surface.geometry_schedule(1866, 8000, 1000, 0.05) == (1000, 0.05)
    assert surface.geometry_schedule(9999, 8000, 0, 0) == (0, 0)
