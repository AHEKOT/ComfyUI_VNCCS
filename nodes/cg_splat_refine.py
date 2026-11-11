"""Refine a WorldMirror reconstruction in its calibrated camera frame."""

import math
from itertools import product

import numpy as np


def _view_evidence(points, depth, pose, intrinsic):
    rotation = pose[:3, :3]
    camera = (points - pose[:3, 3]) @ rotation
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    ids = np.flatnonzero(np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0))
    x, y, z = camera[ids].T
    u, v = fx * x / z + cx, fy * y / z + cy
    height, width = depth.shape
    inside = (u >= 1) & (u < width - 2) & (v >= 1) & (v < height - 2)
    u, v, z, ids = u[inside], v[inside], z[inside], ids[inside]
    x, y = np.rint(u).astype(np.intp), np.rint(v).astype(np.intp)
    neighbors = np.stack((depth[y, x], depth[y, x - 1], depth[y, x + 1], depth[y - 1, x], depth[y + 1, x]))
    center, left, right, top, bottom = neighbors
    smooth = (np.isfinite(neighbors).all(axis=0) & (center > 0)
              & (neighbors.max(axis=0) - neighbors.min(axis=0) < 0.045 * center))
    center, left, right, top, bottom = neighbors[:, smooth]
    ids, u, v, z, x, y = ids[smooth], u[smooth], v[smooth], z[smooth], x[smooth], y[smooth]
    dx, dy = (right - left) * 0.5, (bottom - top) * 0.5
    measured = center + dx * (u - x) + dy * (v - y)
    error = (measured - z) / measured
    normal = np.column_stack((-fx * dx, -fy * dy, measured + (u - cx) * dx + (v - cy) * dy))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    ray = np.column_stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)))
    incidence = (ray * normal).sum(axis=1)
    stable = incidence > 0.25
    match = stable & (np.abs(error) <= 0.035)
    correction = normal[match] @ rotation.T
    correction *= ((measured[match] - z[match]) * incidence[match])[:, None]
    return (ids[match], correction, ids[stable & (error > 0.065)],
            ids[stable & (error < -0.055) & (error > -0.18)])


def _components(points, voxel_size, check_cancel):
    cells = np.clip(np.floor(points / voxel_size), -1_000_000, 1_000_000).astype(np.int64)
    cells -= cells.min(axis=0)
    dims = cells.max(axis=0) + 3
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys, inverse = np.unique(cells @ strides, return_inverse=True)
    roots = np.arange(len(keys), dtype=np.intp)
    edges = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        check_cancel()
        other = np.searchsorted(keys, keys + np.dot(offset, strides))
        match = (other < len(keys)) & (keys[np.minimum(other, len(keys) - 1)] == keys + np.dot(offset, strides))
        if match.any():
            edges.append((roots[match].copy(), other[match]))
    while edges:
        check_cancel()
        old = roots.copy()
        for left, right in edges:
            a, b = old[left], old[right]
            different = a != b
            a, b = a[different], b[different]
            np.minimum.at(roots, np.maximum(a, b), np.minimum(a, b))
        while not np.array_equal(roots, roots[roots]):
            roots = roots[roots]
        if np.array_equal(roots, old):
            break
    return roots[inverse]


def refine_splats(splats, depth, poses, intrinsics, check_cancel):
    import torch

    if not splats or len(splats["means"][0]) == 0:
        raise ValueError("WorldMirror returned no splats for refinement")
    means = splats["means"][0].detach().cpu().float().numpy()
    depth = depth.detach().cpu().float().numpy()
    poses, intrinsics = (tensor.detach().cpu().float().numpy() for tensor in (poses, intrinsics))
    if depth.ndim != 3 or poses.shape != (len(depth), 4, 4) or intrinsics.shape != (len(depth), 3, 3):
        raise ValueError("WorldMirror refinement requires one depth map and camera per frame")
    finite = np.isfinite(means).all(axis=1)
    points = means[finite].copy()
    sampled = depth[:, ::16, ::16]
    valid = sampled[np.isfinite(sampled) & (sampled > 0)]
    if not len(valid):
        valid = depth[np.isfinite(depth) & (depth > 0)]
    if not len(points) or not len(valid):
        raise ValueError("WorldMirror refinement has no valid geometry or depth samples")
    unit = float(np.median(valid))
    support = np.zeros(len(points), dtype=np.int32)
    free = np.zeros_like(support)
    behind = np.zeros_like(support)
    # Two passes re-evaluate visibility after moving surface centers.
    for _ in range(2):
        support.fill(0)
        free.fill(0)
        behind.fill(0)
        displacement = np.zeros_like(points)
        centers = []
        for view, pose, intrinsic in zip(depth, poses, intrinsics, strict=True):
            if any(np.linalg.norm(pose[:3, 3] - center) < unit * 0.002 for center in centers):
                continue
            centers.append(pose[:3, 3])
            for start in range(0, len(points), 262_144):
                check_cancel()
                match, correction, empty, occluded = _view_evidence(points[start:start + 262_144], view, pose, intrinsic)
                support[match + start] += 1
                displacement[match + start] += correction
                free[empty + start] += 1
                behind[occluded + start] += 1
        agreed = (support >= 2) & (support > free)
        step = displacement[agreed] / support[agreed, None]
        length = np.linalg.norm(step, axis=1)
        step *= np.minimum(1., unit * 0.02 / np.maximum(length, 1e-8))[:, None]
        points[agreed] += step

    # Occlusion alone is expected. Suppress a nearby rear layer only after repeated votes and weak direct support.
    unsupported = (support <= 1) & ((free >= 2) | (behind >= 3))
    components = _components(points, unit * 0.012, check_cancel)
    counts = np.bincount(components)
    backed = np.bincount(components, weights=(support >= 1), minlength=len(counts))
    verified = np.bincount(components, weights=(support >= 2), minlength=len(counts))
    contradicted = np.bincount(components, weights=unsupported, minlength=len(counts))
    small = counts <= max(12, min(256, math.ceil(len(points) * 0.00003)))
    detached = ((backed == 0) & small) | ((verified <= counts * 0.1) & (contradicted > counts * 0.5))
    keep = ~unsupported & ~detached[components]
    if not keep.any():
        raise ValueError("WorldMirror refinement found no supported splats")
    check_cancel()
    indices = torch.from_numpy(np.flatnonzero(finite)[keep])
    output = {key: [splats[key][0][indices.to(splats[key][0].device)]] for key in ("sh", "scales", "quats", "opacities")}
    output["means"] = [torch.from_numpy(points[keep])]
    moved = np.count_nonzero(np.linalg.norm(points[keep] - means[indices.numpy()], axis=1) > unit * 1e-4)
    return output, {"input_splats": len(means), "output_splats": len(indices),
                    "moved_splats": int(moved), "removed_depth_conflicts": int(unsupported.sum()),
                    "removed_detached_splats": int((~unsupported & detached[components]).sum())}
