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


def _quaternion_rotations(quats):
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def test_projection_matches_independent_ray_plane_intersections_and_pose(monkeypatch):
    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(normalized_quat_to_rotmat=_quaternion_rotations))
    params = _splats()
    view = torch.eye(4, dtype=torch.float64)[None]
    c, s = math.cos(0.4), math.sin(0.4)
    view[0, :3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
    view[0, 0, 3] = 0.3
    _, _, _, transforms, _ = surface._project_surfels(
        params["means"], params["quats"], params["scales"].exp(), view,
        torch.eye(3, dtype=torch.float64)[None], 64, 64, 0.2,
    )
    rays = torch.tensor([[-0.2, 0.0, 1.0], [0.0, 0.0, 1.0], [0.3, 0.1, 1.0]], dtype=torch.float64)
    normal = view[0, :3, :3] @ torch.tensor([math.sin(0.5), 0.0, math.cos(0.5)], dtype=torch.float64)
    center = view[0, :3, :3] @ params["means"][0] + view[0, :3, 3]
    expected = (normal @ center) / (rays @ normal)
    homogeneous = torch.linalg.solve(transforms[0, 0], rays.T).T
    actual = 1 / homogeneous[:, 2]
    assert torch.allclose(actual, expected, atol=1e-12)
    assert not torch.allclose(actual, actual[0].expand_as(actual))
    actual.sum().backward()
    for value in (params["means"], params["quats"]):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_camera_plane_singularity_is_culled_without_nonfinite_transforms(monkeypatch):
    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(normalized_quat_to_rotmat=_quaternion_rotations))
    params = _splats()
    means = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    radii, xy, _, transforms, _ = surface._project_surfels(
        means, params["quats"][:1], params["scales"][:1].exp(), torch.eye(4, dtype=torch.float64)[None],
        torch.eye(3, dtype=torch.float64)[None], 64, 64, 0.2,
    )
    assert not radii.any() and torch.isfinite(transforms).all() and torch.isfinite(xy).all()
    transforms.sum().backward()
    assert torch.isfinite(means.grad).all()


def test_projection_has_three_sigma_bounds_and_preserves_scene_units(monkeypatch):
    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(normalized_quat_to_rotmat=_quaternion_rotations))
    means = torch.tensor([[0., 0., 2.]], dtype=torch.float64)
    quats = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
    scales = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    view = torch.eye(4, dtype=torch.float64)[None]
    K = torch.tensor([[[100., 0., 32.], [0., 80., 24.], [0., 0., 1.]]], dtype=torch.float64)
    base = surface._project_surfels(means, quats, scales, view, K, 64, 48, 0.2)
    scaled = surface._project_surfels(means * 7, quats, scales * 7, view, K, 64, 48, 1.4)
    assert torch.equal(base[0], torch.full((1, 1, 2), 24, dtype=torch.int32))
    assert torch.equal(base[1], torch.tensor([[[32., 24.]]], dtype=torch.float64))
    for index in (0, 1, 4):
        assert torch.allclose(base[index], scaled[index])
    assert torch.allclose(base[3] * 7, scaled[3])


@pytest.mark.parametrize("with_geometry", [False, True])
def test_geometry_dispatch_reuses_native_projection_and_gsplat_tiles(monkeypatch, with_geometry):
    params = _splats(torch.float32)
    calls = []
    maps = torch.randn((1, 8, 6, 6), requires_grad=True)
    view, K = torch.eye(4)[None], torch.eye(3)[None]

    def projection(means, quats, scales, *_args, **options):
        calls.append("projection")
        assert scales.shape == (2, 2)
        return (torch.ones((1, 2, 2), dtype=torch.int32), torch.zeros((1, 2, 2)),
                torch.ones((1, 2)), torch.zeros((1, 2, 3, 3)), torch.ones((1, 2, 3)))

    def tiles(xy, radii, depths, tile_size, tile_width, tile_height, **_kwargs):
        calls.append("tiles")
        assert (tile_size, tile_width, tile_height) == (16, 1, 1)
        return torch.ones((1, 2)), torch.arange(2), torch.arange(2, dtype=torch.int32)

    def render(*args):
        calls.append("render")
        assert args[2].shape == (1, 2, 3), "Only RGB, not sixteen feature channels"
        assert args[5].shape == () and args[6].shape == (1, 2)
        assert args[7].requires_grad, "Densification must retain its gradient input"
        assert args[13] is with_geometry
        return torch.zeros((1, 8, 6, 3)), torch.zeros((1, 8, 6, 1)), maps

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(
        spherical_harmonics=lambda degree, dirs, coefficients, **_kwargs: coefficients[..., 0, :] * 0.28209479177387814))
    monkeypatch.setitem(sys.modules, "gsplat.cuda._wrapper", SimpleNamespace(
        isect_tiles=tiles,
        isect_offset_encode=lambda *_args: torch.zeros((1, 1, 1), dtype=torch.int32)))
    monkeypatch.setattr(surface, "_project_surfels", projection)
    monkeypatch.setattr(surface, "rasterize_surface", render)
    image, alpha, info = surface.render_2dgs(params, view, K, 6, 8, 1, with_geometry=with_geometry)
    assert calls == ["projection", "tiles", "render"]
    assert image.shape == (1, 8, 6, 3) and alpha.shape == (1, 8, 6, 1)
    assert surface.geometry_maps(info, K) is maps


def test_native_rgb_only_renderer_remains_three_channel_and_does_not_build_geometry(monkeypatch):
    params = _splats(torch.float32)
    captured = []

    def render(*args):
        captured.append(args)
        rgb = torch.zeros((1, 8, 6, 3))
        alpha = torch.zeros((1, 8, 6, 1))
        return rgb, alpha, {"gradient_2dgs": "gradient"}

    monkeypatch.setattr(surface, "_render_surface", render)
    rgb, alpha, info = surface.render_2dgs(params, torch.eye(4)[None], torch.eye(3)[None], 6, 8, 1)
    assert rgb.shape == (1, 8, 6, 3) and alpha.shape == (1, 8, 6, 1)
    assert info == {"gradient_2dgs": "gradient"}
    assert captured[0][7] == 1 and captured[0][1].shape == (2, 4, 3)
    assert captured[0][9] is False


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
        surface.compatibility_scales, surface._project_surfels,
        surface.render_2dgs, surface._render_surface, surface.geometry_maps,
        surface.geometry_losses, surface.geometry_schedule, surface.split_2dgs,
    ))
    for forbidden in ("rasterize_to_indices_in_range", "torch.utils.checkpoint", "_geometry_stripe",
                      "torch.cumsum", "torch.cummax", ".cpu()", ".tolist()"):
        assert forbidden not in source


def test_point_splats_optimize_one_shared_radius_with_both_axis_gradients():
    radius = torch.nn.Parameter(torch.tensor([[0.2], [0.1]], dtype=torch.float64).log())
    scales = torch.nn.Parameter(radius.detach().repeat(1, 2))
    radius_optimizer = torch.optim.Adam([radius], lr=0.005, eps=1e-15)
    scales_optimizer = torch.optim.Adam([scales], lr=0.005, eps=1e-15)
    weights = torch.tensor([[3.0, -1.0], [-2.0, 4.0]], dtype=torch.float64)

    surface._tie_planar_scale_gradients(scales)
    assert scales.grad is None
    for step in range(4):
        axis_weights = weights.roll(step % 2, dims=1) * (-1) ** step
        (radius.expand(-1, 2).exp() * axis_weights).sum().backward()
        (scales.exp() * axis_weights).sum().backward()
        surface._tie_planar_scale_gradients(scales)
        assert torch.allclose(scales.grad, radius.grad.expand(-1, 2))
        radius_optimizer.step()
        scales_optimizer.step()
        assert torch.equal(scales[:, 0], scales[:, 1])
        assert torch.allclose(scales, radius.expand(-1, 2))
        radius_optimizer.zero_grad(set_to_none=True)
        scales_optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("point_splats", [False, True])
def test_planar_split_preserves_plane_optimizer_state_and_frozen_sh(monkeypatch, point_splats):
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    monkeypatch.setitem(sys.modules, "gsplat.utils", SimpleNamespace(
        normalized_quat_to_rotmat=lambda quats: rotation[None].expand(len(quats), -1, -1),
    ))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(
            [[0.2, 0.2], [0.1, 0.1]] if point_splats else [[0.2, 0.4], [0.1, 0.3]],
        ))),
        "quats": torch.nn.Parameter(torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]] * 2)),
        "opacities": torch.nn.Parameter(torch.zeros(2)),
        "sh0": torch.nn.Parameter(torch.zeros((2, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((2, 0, 3)), requires_grad=False),
    })
    optimizers = {name: torch.optim.Adam([value], lr=0.01) for name, value in params.items() if value.requires_grad}
    sum(value.square().sum() for value in params.values()).backward()
    if point_splats:
        surface._tie_planar_scale_gradients(params["scales"])
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    original = {name: value.detach().clone() for name, value in params.items()}
    moments = {name: optimizer.state[params[name]]["exp_avg"].clone() for name, optimizer in optimizers.items()}
    state = {"grad2d": torch.tensor([0.1, 0.2]), "count": torch.tensor([5.0, 7.0]),
             "radii": torch.tensor([0.2, 0.01]), "scene_scale": 1.0}
    surface.split_2dgs(params, optimizers, state, torch.tensor([True, False]))
    assert params["scales"].shape == (3, 2)
    assert torch.equal(params["means"][0], original["means"][1])
    offset = params["means"][1:] - original["means"][0]
    assert torch.allclose(offset @ rotation[:, 2], torch.zeros(2), atol=1e-7)
    assert torch.allclose(params["scales"][1:].exp(), original["scales"][:1].exp().expand(2, -1) / 1.6)
    assert not params["shN"].requires_grad and state["count"].tolist() == [7.0, 5.0, 5.0]
    assert torch.equal(state["radii"], torch.tensor([0.01, 0., 0.]))
    for name, optimizer in optimizers.items():
        assert optimizer.param_groups[0]["params"][0] is params[name]
        moment = optimizer.state[params[name]]["exp_avg"]
        assert torch.equal(moment[0], moments[name][1]) and not moment[1:].any()
    if point_splats:
        assert torch.equal(params["scales"][:, 0], params["scales"][:, 1])
        for key in ("exp_avg", "exp_avg_sq"):
            moment = optimizers["scales"].state[params["scales"]][key]
            assert torch.equal(moment[:, 0], moment[:, 1])


def test_short_training_has_proportional_warmups_and_no_extra_training_steps():
    assert surface.geometry_schedule(0, 1, 1000, 0.05) == (1000, 0.05)
    assert surface.geometry_schedule(799, 8000, 1000, 0.05) == (0, 0)
    assert surface.geometry_schedule(800, 8000, 1000, 0.05) == (1000, 0)
    assert surface.geometry_schedule(1866, 8000, 1000, 0.05) == (1000, 0.05)
    assert surface.geometry_schedule(9999, 8000, 0, 0) == (0, 0)
