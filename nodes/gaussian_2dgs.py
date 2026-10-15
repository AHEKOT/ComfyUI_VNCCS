"""Surface Gaussians with fused CUDA RGB and geometry compositing.

Implements the planar representation and geometry objectives of 2DGS
(arXiv:2403.17888, including Appendix A's squared NDC-depth distortion).
Stock gsplat 1.5.3 depth interpolation is deliberately unused. Projection and
tile sorting stay in gsplat; a fused CUDA kernel evaluates ray-plane depth and
its moments directly, without full-resolution feature images or hit lists.
"""

import math

import torch
import torch.nn.functional as F


DISTORTION_WEIGHT = 1_000.0
NORMAL_WEIGHT = 0.05
NDC_NEAR = 0.2
NDC_FAR = 1_000.0
_NEAR_PLANE = 0.01


def compatibility_scales(log_scales):
    """Approximate surfels by thin ellipsoids for standard 3DGS consumers."""
    if log_scales.shape[-1] != 2:
        return log_scales.expand(-1, 3)
    thickness = log_scales.amin(dim=-1, keepdim=True) + math.log(1e-3)
    return torch.cat((log_scales, thickness), dim=-1)


def _inverse_depth_planes(means, quats, view):
    """For camera ray r=(x,y,1), intersection z is 1 / dot(plane, r)."""
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    normals = torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x),
                           1 - 2 * (x.square() + y.square())), dim=-1)
    rotation = view[0, :3, :3]
    camera_normals = normals @ rotation.T
    camera_means = means @ rotation.T + view[0, :3, 3]
    distance = (camera_normals * camera_means).sum(-1, keepdim=True)
    # A plane through the camera has no numerically defined intersection depth.
    tolerance = torch.finfo(means.dtype).eps * camera_means.norm(dim=-1, keepdim=True)
    valid = torch.isfinite(distance) & (distance.abs() > tolerance)
    planes = camera_normals / torch.where(valid, distance, torch.ones_like(distance))
    planes = torch.where(valid, planes, torch.zeros_like(planes))
    # Constant translation of NDC depth leaves pairwise distortion unchanged.
    center = camera_means[:, 2].detach().mean().clamp_min(_NEAR_PLANE)
    reference = -NDC_FAR * NDC_NEAR / (NDC_FAR - NDC_NEAR) / center
    return planes, valid, reference


def render_2dgs(splats, view, intrinsics, width, height, sh_degree, absgrad=False,
                with_geometry=False):
    coefficients = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    scales = splats["scales"].exp()
    # The third scale is not a parameter and is never used as surfel thickness.
    scales = torch.cat((scales, torch.ones_like(scales[:, :1])), dim=-1)
    if with_geometry:
        return _render_surface(splats, coefficients, scales, view, intrinsics,
                               width, height, sh_degree, absgrad)

    from gsplat.rendering import rasterization_2dgs

    rendered, alpha, _, _, _, _, info = rasterization_2dgs(
        means=splats["means"], quats=splats["quats"], scales=scales,
        opacities=splats["opacities"].sigmoid(),
        colors=coefficients,
        viewmats=view, Ks=intrinsics, width=width, height=height,
        packed=False, absgrad=absgrad, sh_degree=sh_degree,
        near_plane=_NEAR_PLANE, far_plane=1e10, render_mode="RGB", distloss=False,
    )
    return rendered[..., :3], alpha, info


def _render_surface(splats, coefficients, scales, view, intrinsics, width, height, sh_degree, absgrad):
    from gsplat import spherical_harmonics
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs, isect_offset_encode, isect_tiles
    from .gaussian_2dgs_cuda import rasterize_surface

    radii, xy, depths, transforms, normals = fully_fused_projection_2dgs(
        splats["means"], splats["quats"], scales, view, intrinsics, width, height,
        near_plane=_NEAR_PLANE, far_plane=1e10, packed=False,
    )
    tile_width, tile_height = (width + 15) // 16, (height + 15) // 16
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        xy, radii, depths, 16, tile_width, tile_height, packed=False,
    )
    offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    camera_center = -(view[0, :3, :3].T @ view[0, :3, 3])
    rgb = (spherical_harmonics(
        sh_degree, splats["means"][None] - camera_center, coefficients[None],
        masks=(radii > 0).all(dim=-1),
    ) + 0.5).clamp_min(0)
    planes, valid, reference = _inverse_depth_planes(splats["means"], splats["quats"], view)
    opacities = splats["opacities"].sigmoid()[None]
    densify = torch.zeros_like(xy, requires_grad=True)
    # Transform per surfel, not per pixel; this commutes with alpha compositing.
    world_normals = normals @ view[0, :3, :3]
    image, alpha, maps = rasterize_surface(
        xy, transforms, rgb, opacities, world_normals, planes, valid, reference,
        intrinsics, depths, densify, offsets, flatten_ids, width, height, absgrad,
    )
    return image, alpha, {
        "camera_ids": None, "gaussian_ids": None, "radii": radii,
        "means2d": xy, "depths": depths, "ray_transforms": transforms,
        "opacities": opacities, "normals": normals,
        "tile_width": tile_width, "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids,
        "flatten_ids": flatten_ids, "isect_offsets": offsets,
        "width": width, "height": height, "tile_size": 16, "n_cameras": 1,
        "gradient_2dgs": densify, "surface_maps": maps,
    }


def geometry_maps(info, intrinsics):
    """Geometry is already evaluated by the fused CUDA rasterization."""
    if info["n_cameras"] != 1:
        raise ValueError("Surface geometry expects one training camera at a time")
    return info.pop("surface_maps")


def geometry_losses(maps, view, intrinsics, use_normal=True):
    median, alpha, distortion = maps[..., :3].split(1, dim=-1)
    if not use_normal:
        return distortion.mean(), distortion.new_zeros(()), median
    if maps.is_cuda:
        from .gaussian_2dgs_cuda import normal_consistency

        return distortion.mean(), normal_consistency(maps, view, intrinsics), median

    from gsplat.utils import depth_to_normal

    # gsplat returns composited normals in world space, including camera-facing signs.
    surface_normal = depth_to_normal(median, torch.linalg.inv(view), intrinsics)
    valid = (alpha.detach() > 0.01) & torch.isfinite(median) & (median > 0)
    interior = F.avg_pool2d(valid.permute(0, 3, 1, 2).float(), 3, 1, 1) > 0.999
    interior = interior.permute(0, 2, 3, 1)
    target_normal = surface_normal * alpha.detach()
    normal_error = 1.0 - (maps[..., 3:6] * target_normal).sum(-1, keepdim=True)
    normal_loss = torch.where(interior, normal_error, torch.zeros_like(normal_error)).mean()
    return distortion.mean(), normal_loss, median


def geometry_schedule(step, training_steps, distortion_weight, normal_weight):
    """Scale the original 3k/7k warm-ups to the requested training duration."""
    return (
        distortion_weight if step >= int(training_steps * 3_000 / 30_000) else 0.0,
        normal_weight if step >= int(training_steps * 7_000 / 30_000) else 0.0,
    )


@torch.no_grad()
def split_2dgs(params, optimizers, state, mask):
    from gsplat.utils import normalized_quat_to_rotmat

    selected, rest = torch.where(mask)[0], torch.where(~mask)[0]
    scales = params["scales"][selected].exp()
    rotations = normalized_quat_to_rotmat(F.normalize(params["quats"][selected], dim=-1))
    samples = torch.randn((2, len(selected), 2), device=scales.device) * scales
    offsets = torch.einsum("nij,bnj->bni", rotations[..., :2], samples)
    for name, parameter in list(params.items()):
        repeats = [2] + [1] * (parameter.ndim - 1)
        if name == "means":
            children = (parameter[selected] + offsets).reshape(-1, 3)
        elif name == "scales":
            children = (params["scales"][selected] - math.log(1.6)).repeat(2, 1)
        else:
            children = parameter[selected].repeat(repeats)
        replacement = torch.nn.Parameter(torch.cat((parameter[rest], children)),
                                         requires_grad=parameter.requires_grad)
        params[name] = replacement
        if name not in optimizers:
            if parameter.requires_grad:
                raise RuntimeError(f"Missing optimizer for trainable surfel parameter {name}")
            continue
        optimizer = optimizers[name]
        old_state = optimizer.state.pop(parameter, {})
        for key, value in old_state.items():
            if key != "step":
                old_state[key] = torch.cat((value[rest], value.new_zeros((2 * len(selected), *value.shape[1:]))))
        for group in optimizer.param_groups:
            group["params"] = [replacement]
        optimizer.state[replacement] = old_state
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            repeats = [2] + [1] * (value.ndim - 1)
            state[name] = torch.cat((value[rest], value[selected].repeat(repeats)))
