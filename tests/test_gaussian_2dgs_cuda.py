"""Fused rasterizer contracts and optional tiny CUDA gradient comparisons."""

from contextlib import nullcontext
import ctypes as C
import inspect
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import nodes.vnccs_gaussians as backend


def _inputs(device, width=5, height=3, count=3):
    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)

    xy = tensor([[[2.4, 1.4], [2.6, 1.6], [2.2, 1.5]]])
    transforms = tensor([[[[3.0, 0.0, 2.4], [0.0, 3.0, 1.4], [0.1, -0.05, 1.0]],
                          [[2.4, 0.1, 2.6], [0.0, 2.8, 1.6], [-0.1, 0.05, 1.0]],
                          [[0.4, 0.0, 2.2], [0.0, 0.4, 1.5], [0.0, 0.0, 1.0]]]])
    rgb = tensor([[[0.8, 0.2, 0.1], [0.2, 0.6, 0.3], [0.1, 0.3, 0.9]]])
    opacity = tensor([[0.32, 0.46, 0.98]])
    normals = tensor([[[0.0, 0.0, -1.0], [0.2, 0.1, -0.98], [-0.1, 0.0, -0.99]]])
    values = [xy, transforms, rgb, opacity, normals]
    if count != 3:
        values = [value.repeat(1, (count + 2) // 3, *([1] * (value.ndim - 2)))[:, :count]
                  .detach().requires_grad_() for value in values]
    reference = torch.tensor(-0.08, device=device)
    depths = torch.arange(2, count + 2, device=device, dtype=torch.float32)[None]
    densify = torch.zeros((1, count, 2), device=device, requires_grad=True)
    tiles_x, tiles_y = (width + 15) // 16, (height + 15) // 16
    offsets = (torch.arange(tiles_x * tiles_y, device=device, dtype=torch.int32) * count).reshape(1, tiles_y, tiles_x)
    ids = torch.arange(count, device=device, dtype=torch.int32).repeat(tiles_x * tiles_y)
    return (*values, reference, depths, densify, offsets, ids, width, height)


def _oracle(inputs):
    """Reference 2DGS compositing, with an independent inverse-homography intersection.

    Equations: hbb1/diff-surfel-rasterization e0ed0207, forward.cu 349-419.
    Use the reference straight-through alpha cap, not the derivative of clamp.
    """
    xy, transforms, rgb, opacity, normals, reference, *_rest = inputs
    width, height = inputs[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=xy.device), torch.arange(width, device=xy.device), indexing="ij")
    px, py = xx + 0.5, yy + 0.5
    T = torch.ones_like(px)
    M1, M2, distortion = torch.zeros_like(T), torch.zeros_like(T), torch.zeros_like(T)
    image, normal = torch.zeros((*T.shape, 3), device=xy.device), torch.zeros((*T.shape, 3), device=xy.device)
    active = torch.ones_like(T, dtype=torch.bool)
    median = torch.zeros_like(T)
    pixels = torch.stack((px, py, torch.ones_like(px)), -1)
    for g in range(transforms.shape[1]):
        matrix = transforms[0, g]
        invertible = torch.linalg.det(matrix) != 0
        safe_matrix = torch.where(invertible, matrix, torch.eye(3, device=matrix.device))
        local = torch.linalg.solve(safe_matrix, pixels.reshape(-1, 3).T).T.reshape(height, width, 3)
        finite_depth = local[..., 2] != 0
        z = 1 / torch.where(finite_depth, local[..., 2], torch.ones_like(T))
        uv = local[..., :2] * z[..., None]
        rho3 = uv.square().sum(-1)
        rho2 = 2 * ((xy[0, g, 0] - px).square() + (xy[0, g, 1] - py).square())
        raw_alpha = opacity[0, g] * torch.exp(-0.5 * torch.minimum(rho3, rho2))
        a = raw_alpha + (raw_alpha.clamp_max(0.99) - raw_alpha).detach()
        candidate = active & invertible & finite_depth & (z >= backend._NEAR_PLANE) & (a >= 1 / 255)
        next_T = T * (1 - a)
        active = active & ~(candidate & (next_T < 1e-4))
        contributes = candidate & (next_T >= 1e-4)
        weight = torch.where(contributes, T * a, torch.zeros_like(T))
        image = image + weight[..., None] * rgb[0, g]
        normal = normal + weight[..., None] * normals[0, g]
        select = contributes & (T > 0.5)
        median = torch.where(select, z, median)
        m = -backend.NDC_FAR * backend.NDC_NEAR / (backend.NDC_FAR - backend.NDC_NEAR) / z - reference
        distortion = distortion + (m.square() * (1 - T) + M2 - 2 * m * M1) * weight
        M1, M2 = M1 + weight * m, M2 + weight * m.square()
        T = torch.where(contributes, next_T, T)
    maps = torch.cat((median[..., None], (1 - T)[..., None], distortion[..., None], normal), -1)
    return image[None], (1 - T)[None, ..., None], maps[None]


def test_cuda_wrapper_rejects_cpu_inputs_before_loading_any_gpu_library(monkeypatch):
    monkeypatch.setattr(backend, "_kernels", lambda *_args: pytest.fail("Must validate before CUDA initialization"))
    with pytest.raises(ValueError, match="expected CUDA"):
        backend.rasterize_surface(*_inputs("cpu"))


def test_kernel_launch_uses_torch_stream_and_raw_pointers_without_tensor_copies(monkeypatch):
    launches = []
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace(cuda_stream=1234))
    value = SimpleNamespace(data_ptr=lambda: 9876)
    backend._launch(lambda *args, **kwargs: launches.append((args, kwargs)), (value,),
                    (C.c_int(17),), (None,), 17, 18, torch.device("cuda:0"))
    args, kwargs = launches[0]
    assert args[:2] == ((2, 2), (16, 16))
    assert tuple(value.value for value in args[2]) == (9876, 17, None)
    assert kwargs == {"stream": 1234}


def test_surface_backward_preserves_tuple_arguments_without_loading_cuda(monkeypatch):
    inputs = _inputs("cpu")
    saved = inputs[:-2]
    ctx = SimpleNamespace(saved_tensors=(*saved, None, None, None, None),
                          width=inputs[-2], height=inputs[-1], absgrad=False, scalars=())
    launches = []
    monkeypatch.setattr(backend, "_kernels", lambda _device: (None, "backward"))
    monkeypatch.setattr(backend, "_launch", lambda *args: launches.append(args))

    gradients = backend._SurfaceRasterization.backward(ctx, None, None, None)

    assert len(launches) == 1 and launches[0][0] == "backward"
    arguments = launches[0][1]
    assert isinstance(arguments, tuple) and len(arguments) == 8
    assert all(actual is expected for actual, expected in zip(arguments, saved[:6] + saved[-2:]))
    assert len(gradients) == 16
    assert gradients[7].shape == saved[7].shape


def test_fused_backend_does_not_materialize_feature_images_or_host_hit_lists():
    source = backend._SOURCE + "\n".join(inspect.getsource(function) for function in (
        backend._kernels, backend.prepare_surface_cuda, backend._launch,
        backend._SurfaceRasterization, backend.rasterize_surface,
        backend._NormalConsistency, backend.normal_consistency,
    ))
    for forbidden in ("rasterize_to_indices_in_range", "torch.utils.checkpoint", ".cpu()", ".tolist()", "torch.cuda.synchronize"):
        assert forbidden not in source
    assert "surface_forward" in source and "surface_backward" in source
    assert "load_surface_kernels" in source
    assert "+vz*h.local_u" in source and "+vz*h.sv" in source
    assert "h.depth<near_plane" in source


@pytest.mark.parametrize("variant", ["regular", "background", "invalid_plane", "near_plane", "clipped", "early_stop", "batch", "partial_tiles"])
def test_cuda_forward_and_all_parameter_gradients_match_independent_equations(variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the rasterizer comparison")
    inputs = list(_inputs("cuda", 17 if variant == "partial_tiles" else 5,
                          18 if variant == "partial_tiles" else 3,
                          300 if variant in ("batch", "early_stop") else 3))
    with torch.no_grad():
        if variant == "background":
            inputs[3].zero_()
        elif variant == "invalid_plane":
            inputs[1][0, 1].zero_()
        elif variant == "near_plane":
            inputs[1][0, 1, 2] = torch.tensor([0.0, 0.0, 0.1], device="cuda")
        elif variant == "clipped":
            inputs[3].fill_(2.0)
        elif variant == "batch":
            inputs[3].fill_(0.006)
        elif variant == "early_stop":
            inputs[3].fill_(0.95)
    image, alpha, maps = backend.rasterize_surface(*inputs)
    reference = _oracle(inputs)
    for actual, expected in zip((image, alpha, maps), reference):
        assert torch.allclose(actual, expected, atol=4e-5, rtol=2e-4)
    generator = torch.Generator(device="cuda").manual_seed(12)
    weights = [torch.randn(output.shape, device="cuda", generator=generator) for output in reference]
    expected_loss = sum((output * weight).sum() for output, weight in zip(reference, weights))
    actual_loss = sum((output * weight).sum() for output, weight in zip((image, alpha, maps), weights))
    expected_gradients = torch.autograd.grad(expected_loss, inputs[:5])
    actual_gradients = torch.autograd.grad(actual_loss, (*inputs[:5], inputs[7]))
    for actual, expected in zip(actual_gradients[:5], expected_gradients):
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected, atol=3e-4, rtol=2e-3)
    expected_densify = torch.stack((actual_gradients[1][..., 0, 2], actual_gradients[1][..., 1, 2]), -1) * inputs[6][..., None]
    assert torch.allclose(actual_gradients[5], expected_densify)


def test_cuda_distortion_matches_pairwise_ray_depths_and_is_centering_invariant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the distortion comparison")
    inputs = list(_inputs("cuda"))
    expected = _oracle(inputs)[2][..., 2]
    maps = backend.rasterize_surface(*inputs)[2]
    assert torch.allclose(maps[..., 2], expected, atol=2e-7, rtol=1e-4)
    # One contributor has no pairwise spread, irrespective of surfel tilt.
    inputs[3] = torch.tensor([[0.5, 0.0, 0.0]], device="cuda", requires_grad=True)
    maps = backend.rasterize_surface(*inputs)[2]
    assert maps[..., 2].abs().max() < 2e-8
    # The original multi-contributor objective does not depend on the NDC origin.
    inputs[3] = torch.tensor([[0.32, 0.46, 0.98]], device="cuda", requires_grad=True)
    for shift in (-0.2, 0.2):
        inputs[5] = torch.tensor(shift, device="cuda")
        shifted = backend.rasterize_surface(*inputs)[2][..., 2]
        assert torch.allclose(shifted, expected, atol=2e-7, rtol=1e-4)


def test_cuda_geometry_does_not_change_rgb_and_depth_alone_reaches_densification():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the rasterizer comparison")
    inputs = _inputs("cuda")
    rgb_only = backend.rasterize_surface(*inputs, with_geometry=False)
    full = backend.rasterize_surface(*inputs, with_geometry=True)
    for actual, expected in zip(rgb_only[:2], full[:2]):
        assert torch.equal(actual, expected)
    assert not rgb_only[2][..., (0, 2, 3, 4, 5)].any()
    transform_grad, densify_grad = torch.autograd.grad(full[2][..., 0].sum(), (inputs[1], inputs[7]))
    assert torch.isfinite(densify_grad).all() and densify_grad.abs().sum() > 0
    expected = torch.stack((transform_grad[..., 0, 2], transform_grad[..., 1, 2]), -1) * inputs[6][..., None]
    assert torch.allclose(densify_grad, expected)


@pytest.mark.parametrize("variant", ["regular", "rotated", "hole", "background"])
def test_cuda_normal_stencil_matches_world_space_objective_and_depth_gradients(variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the normal comparison")
    height, width = 7, 9
    yy, xx = torch.meshgrid(torch.arange(height, device="cuda"),
                            torch.arange(width, device="cuda"), indexing="ij")
    depth = (2 + 0.03 * xx + 0.02 * yy + 0.001 * xx * yy)[None, ..., None]
    mass = torch.full_like(depth, 0.8)
    if variant == "hole":
        mass[:, 3, 4] = 0
    elif variant == "background":
        mass.zero_()
    normals = torch.tensor([0.15, -0.05, -0.77], device="cuda").expand(1, height, width, 3)
    maps = torch.cat((depth, mass, torch.zeros_like(depth), normals), -1).detach().requires_grad_()
    view = torch.eye(4, device="cuda")[None]
    if variant == "rotated":
        c, s = math.cos(0.4), math.sin(0.4)
        view[0, :3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device="cuda")
        view[0, :3, 3] = torch.tensor([0.3, -0.2, 0.1], device="cuda")
    K = torch.tensor([[[8.0, 0, 4.3], [0, 7.0, 3.2], [0, 0, 1]]], device="cuda")
    pixels = torch.stack((xx + 0.5, yy + 0.5, torch.ones_like(xx)), -1).float()
    rays = pixels @ torch.linalg.inv(K[0]).T
    camera_to_world = torch.linalg.inv(view)
    points = (maps[..., :1] * rays) @ camera_to_world[0, :3, :3].T + camera_to_world[0, :3, 3]
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    surface_normal = F.pad(F.normalize(torch.cross(dx, dy, dim=-1), dim=-1), (0, 0, 1, 1, 1, 1))
    alpha = maps[..., 1:2].detach()
    error = 1 - (maps[..., 3:] * surface_normal * alpha).sum(-1, keepdim=True)
    expected = error.mean()
    actual = backend.normal_consistency(maps, view, K)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
    expected_gradient, = torch.autograd.grad(expected, maps)
    actual_gradient, = torch.autograd.grad(actual, maps)
    assert torch.isfinite(actual_gradient).all()
    assert torch.allclose(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-4)
    assert not actual_gradient[..., 1:3].any()
