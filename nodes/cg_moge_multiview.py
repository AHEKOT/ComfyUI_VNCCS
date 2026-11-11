"""Render, calibrate and merge camera references for MoGe reconstruction."""

import json
import math
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


REFINED_MOGE_VIEWS = (("up", (0, -2, 0)), ("forward", (0, 0, 2)), ("right", (1, 0, 0)), ("left", (-1, 0, 0)))
EXPANDED_VIEWS = (("forward-right", 1.5, 1), ("forward-left", 2.0, -1))
VIEW_STEP = 0.12


def require_colmap():
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError('MoGe requires PyCOLMAP. Install "pycolmap>=4.1,<5" in the ComfyUI Python environment.') from error
    if getattr(pycolmap, "FeatureExtractionOptions", None) is None:
        raise RuntimeError("MoGe requires PyCOLMAP 4.1 or newer")
    return pycolmap


def reference_depth(depth):
    import torch

    depth = depth.flatten().float()
    sample = depth[::max(1, len(depth) // 4096)]
    sample = sample[torch.isfinite(sample) & (sample > 0)]
    if not len(sample):
        raise ValueError("The source camera has no visible depth for shifted views")
    return float(torch.quantile(sample, 0.25))


def _camera_travel(bounds, start, direction, distance, clearance):
    lo, hi = bounds
    end = start + distance * direction
    nearby = ((lo <= np.maximum(start, end) + clearance) & (hi >= np.minimum(start, end) - clearance)).all(axis=1)
    lo, hi = lo[nearby] - clearance, hi[nearby] + clearance
    entry, exit = np.zeros(len(lo)), np.full(len(lo), distance)
    for axis in np.flatnonzero(np.abs(direction) > 1e-7):
        a = (lo[:, axis] - start[axis]) / direction[axis]
        b = (hi[:, axis] - start[axis]) / direction[axis]
        entry = np.maximum(entry, np.minimum(a, b))
        exit = np.minimum(exit, np.maximum(a, b))
    return max(0.0, float(entry[entry <= exit].min(initial=distance)) - 1e-5 * clearance)


def expanded_camera_poses(source_pose, depth_unit, points, bounds, focal, size, check_cancel):
    height, width = size
    fx, fy = focal
    pose = source_pose.detach().cpu().float().numpy()
    if not np.isfinite(pose).all():
        raise ValueError("Invalid source camera for expanded views")
    margin = 0.03 * depth_unit
    sample = points[::max(1, len(points) // 65000)].detach().cpu().float().numpy()
    x, y, z = ((sample - pose[:3, 3]) @ pose[:3, :3] / depth_unit).T
    forward = np.isfinite(z) & (z > 0.05)
    ahead = z[forward & (np.abs(x) < 0.15 * z) & (np.abs(y) < 0.2 * z)]
    clearance = 0.8 * np.quantile(ahead, 0.1) if len(ahead) >= 32 else 0.0
    clearance = min(clearance, _camera_travel(bounds, pose[:3, 3], pose[:3, 2],
                                               2 * VIEW_STEP * max(factor for _, factor, _ in EXPANDED_VIEWS) * depth_unit,
                                               margin) / depth_unit)
    poses = []
    for _, factor, turn in EXPANDED_VIEWS:
        side = z[forward & (turn * x > 0.15 * z) & (np.abs(y) < 0.6 * z)]
        limit = min(2 * VIEW_STEP * factor, clearance, 0.8 * np.quantile(side, 0.25)) if len(side) >= 32 else 0.0
        candidates = []
        for step in dict.fromkeys(limit * fraction for fraction in (1, 0.75, 0.5, 0.25, 0)):
            check_cancel()
            for angle in (90, 75, 60, 45, 35, 25, 15):
                c, s = math.cos(math.radians(angle)), turn * math.sin(math.radians(angle))
                depth = s * x + c * (z - step)
                front = np.isfinite(depth) & (depth > 0.06)
                u = 0.5 + fx / width * (c * x[front] - s * (z[front] - step)) / depth[front]
                v = 0.5 + fy / height * y[front] / depth[front]
                inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
                cells = np.zeros((16, 24), dtype=bool)
                cells[(v[inside] * 16).astype(int), (u[inside] * 24).astype(int)] = True
                if has_expansion_coverage(cells):
                    candidate = source_pose @ source_pose.new_tensor(
                        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, step * depth_unit], [0, 0, 0, 1]])
                    candidates.append((angle + 45 * step, candidate))
        options = []
        for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True)[:6]:
            check_cancel()
            start, backward = candidate[:3, 3].cpu().numpy(), -candidate[:3, 2].cpu().numpy()
            travel = _camera_travel(bounds, start, backward, 3 * VIEW_STEP * depth_unit, margin)
            candidate[:3, 3] += candidate.new_tensor(backward * travel)
            options.append(candidate)
        poses.append(options)
    return poses


def has_expansion_coverage(visible):
    height, width = visible.shape
    center = visible[height // 4:3 * height // 4, width // 4:3 * width // 4]
    return visible.mean() >= 0.45 and center.mean() >= 0.50


def expansion_matches_capture(frame, result, source):
    import cv2

    height, width = frame.shape[1:3]
    scale = min(1, 1024 / max(height, width))
    size = (round(width * scale), round(height * scale))
    sift = cv2.SIFT_create(nfeatures=3000)

    def features(image, *, capture=False):
        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
        mask = cv2.erode((gray > 4).astype(np.uint8), np.ones((7, 7), np.uint8)) if capture else None
        return sift.detectAndCompute(gray, mask)

    generated, target = features(result)
    if target is None or len(target) < 2:
        return False
    matcher = cv2.BFMatcher()

    def aligned(image, *, capture=False):
        points, descriptors = features(image, capture=capture)
        if descriptors is None:
            return 0
        return sum(best.distance < 0.78 * other.distance
                   and math.dist(points[best.queryIdx].pt, generated[best.trainIdx].pt) <= 0.018 * max(size)
                   for best, other in matcher.knnMatch(descriptors, target, k=2))

    side = aligned(frame, capture=True)
    return side >= 12 and aligned(source) <= max(50, 2 * side)


def _render_mesh_views(mesh, intrinsics, poses, size, depth_unit, check_cancel, *, align_corners):
    import torch
    from torch.nn.functional import grid_sample
    from comfy.model_management import get_torch_device
    try:
        from comfy_extras.sam3d_body.rasterizer import _rasterize_person
    except ImportError as error:
        raise RuntimeError("Update ComfyUI to use MoGe Refine views with the native mesh rasterizer") from error

    device = get_torch_device()
    texture = mesh.texture[0].to(device=device, dtype=torch.float32)
    height, width = size
    texture = texture.permute(2, 0, 1).unsqueeze(0).contiguous()
    vertices = mesh.vertices[0].to(device=device, dtype=torch.float32)
    vertices = vertices * vertices.new_tensor([1, -1, -1]) / depth_unit
    faces = mesh.faces[0].to(device=device, dtype=torch.long)
    uvs = mesh.uvs[0].to(device=device, dtype=torch.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("MoGe returned invalid camera intrinsics for the shifted views")
    for pose in poses:
        check_cancel()
        pose = pose.to(device=device, dtype=torch.float32)
        view = (vertices - pose[:3, 3] / depth_unit) @ pose[:3, :3]
        z = view[:, 2]
        view[:, 0] += z * ((cx - width * 0.5) / fx)
        view[:, 1] = view[:, 1] * (fy / fx) + z * ((cy - height * 0.5) / fx)
        z_buffer = vertices.new_full((height * width,), float("inf"))
        colors = vertices.new_zeros((height * width, 3))
        visible = torch.zeros(height * width, device=device, dtype=torch.bool)

        def shade(face_ids, barycentric):
            if not len(face_ids):
                return colors[:0]
            corners = faces[face_ids]
            weights = barycentric / z[corners]
            uv = (uvs[corners] * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)
            grid = (uv * 2 - 1).reshape(1, 1, -1, 2)
            return grid_sample(texture, grid, align_corners=align_corners, padding_mode="border")[0, :, 0].T

        _rasterize_person(view, faces, fx, width, height, z_buffer, colors, visible, shade)
        yield colors.reshape(1, height, width, 3).cpu(), (~visible.reshape(height, width)).cpu().numpy(), z_buffer.reshape(height, width)


def render_shifted_views(mesh, intrinsics, check_cancel):
    import torch

    height, width = mesh.texture.shape[1:3]
    depth_unit = reference_depth(-mesh.vertices[0, :, 2])
    poses = torch.eye(4).repeat(len(REFINED_MOGE_VIEWS), 1, 1)
    poses[:, :3, 3] = VIEW_STEP * depth_unit * torch.tensor([shift for _, shift in REFINED_MOGE_VIEWS])
    calibration = intrinsics.detach().cpu().numpy() * np.array([[width], [height], [1]])
    views = []
    for frame, holes, _ in _render_mesh_views(mesh, calibration, poses, (height, width), depth_unit, check_cancel,
                                              align_corners=True):
        if holes.all():
            raise ValueError("MoGe produced an empty shifted camera view")
        views.append((frame, holes))
    return views


def render_expanded_mesh_views(mesh, source_depth, calibration, check_cancel):
    import torch

    depth_unit = reference_depth(source_depth)
    vertices = mesh.vertices[0] * mesh.vertices.new_tensor([1, -1, -1])
    corners = vertices[mesh.faces[0].long()]
    bounds = corners.amin(dim=1).cpu().numpy(), corners.amax(dim=1).cpu().numpy()
    del corners
    candidates = expanded_camera_poses(torch.eye(4), depth_unit, vertices, bounds, (calibration[0, 0], calibration[1, 1]),
                                       source_depth.shape, check_cancel)
    views, poses, tracks = [None] * 2, [None] * 2, [None] * 2
    for index, options in enumerate(candidates):
        renders = _render_mesh_views(mesh, calibration, options, source_depth.shape, depth_unit, check_cancel,
                                     align_corners=False)
        for pose, (frame, holes, z_buffer) in zip(options, renders, strict=True):
            if not has_expansion_coverage(~holes):
                continue
            y, x = np.nonzero(~holes)
            stride = max(1, len(x) // 4096)
            y, x = y[::stride], x[::stride]
            z = z_buffer[y, x].cpu().numpy() * depth_unit
            tracks[index] = np.column_stack((x + 0.5, y + 0.5, z))
            views[index], poses[index] = (frame, holes), pose.numpy()
            break
    return views, poses, tracks


def _colmap_matches(colmap, database):
    with colmap.Database.open(database) as db:
        images = sorted(db.read_all_images(), key=lambda image: image.name)
        keypoints = {image.name: db.num_keypoints_for_image(image.image_id) for image in images}
        pairs = []
        for first, second in combinations(images, 2):
            ids = first.image_id, second.image_id
            geometry = db.read_two_view_geometry(*ids) if db.exists_two_view_geometry(*ids) else None
            pairs.append({"views": [first.name, second.name],
                          "inliers": len(geometry.inlier_matches) if geometry is not None else 0,
                          "geometry": colmap.TwoViewGeometryConfiguration(geometry.config).name if geometry is not None else None})
    return {"keypoints": keypoints, "pairs": pairs}


def register_views(images, check_cancel, *, diagnostics=None, focal_length=None, capture_holes=None):
    colmap = require_colmap()
    if diagnostics is None:
        names = [f"{index:02}.png" for index in range(len(images))]
    else:
        labels = ["original", *(name for name, _ in REFINED_MOGE_VIEWS)]
        if len(images) != len(labels):
            raise ValueError("MoGe diagnostics require the original and four refined views")
        names = [f"{index:02}-{label}.png" for index, label in enumerate(labels)]
    with TemporaryDirectory(prefix="vnccs-colmap-") as temporary:
        root = Path(temporary)
        image_dir = root / "views" if diagnostics is None else diagnostics / "colmap"
        image_dir.mkdir()
        for name, image in zip(names, images):
            check_cancel()
            rgb = (np.asarray(image) * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(image_dir / name)
        # The adapter translates one virtual camera without changing its lens.
        reader = colmap.ImageReaderOptions(camera_model="SIMPLE_PINHOLE")
        if focal_length is not None:
            height, width = images.shape[1:3]
            reader.camera_params = f"{focal_length},{width / 2},{height / 2}"
        extraction = colmap.FeatureExtractionOptions(num_threads=len(names), max_image_size=2048)
        matching = colmap.FeatureMatchingOptions(num_threads=len(names), guided_matching=True)
        # QI2 can redraw details and shift keypoints slightly between camera views.
        matching.sift.max_ratio = 0.88
        matching.sift.max_distance = 0.8
        verification = colmap.TwoViewGeometryOptions()
        verification.ransac.max_error = 6.0
        options = colmap.IncrementalPipelineOptions(min_model_size=len(names), multiple_models=False)
        if focal_length is not None:
            options.ba_refine_focal_length = False
            options.mapper.abs_pose_refine_focal_length = False
        options.mapper.init_min_num_inliers = 60
        options.mapper.init_max_error = verification.ransac.max_error
        options.mapper.init_min_tri_angle = 3.0
        options.mapper.abs_pose_min_num_inliers = 20
        options.mapper.abs_pose_min_inlier_ratio = 0.20
        options.mapper.filter_max_reproj_error = verification.ransac.max_error
        options.triangulation.ignore_two_view_tracks = False
        report_path = diagnostics / "colmap_report.json" if diagnostics is not None else None
        attempts = ("full", "masked") if capture_holes else ("full",)
        for attempt in attempts:
            if attempt == "masked":
                # Generated pixels in capture holes can dominate matching with no usable parallax.
                mask_dir = root / "masks"
                mask_dir.mkdir()
                for index, name in enumerate(names):
                    holes = capture_holes.get(index)
                    mask = np.full(images.shape[1:3], 255, dtype=np.uint8) if holes is None else (~holes).astype(np.uint8) * 255
                    Image.fromarray(mask).save(mask_dir / (name + ".png"))
                reader.mask_path = mask_dir
            database = root / f"{attempt}.db"
            check_cancel()
            colmap.extract_features(database, image_dir, camera_mode=colmap.CameraMode.SINGLE,
                                    reader_options=reader, extraction_options=extraction, device=colmap.Device.cpu)
            check_cancel()
            colmap.match_exhaustive(database, matching_options=matching, verification_options=verification, device=colmap.Device.cpu)
            check_cancel()
            if report_path is not None:
                attempt_report = _colmap_matches(colmap, database)
                if attempt == "full":
                    report = attempt_report
                else:
                    report["masked_retry"] = attempt_report
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            reconstructions = colmap.incremental_mapping(database, image_dir, root / attempt, options=options)
            check_cancel()
            if report_path is not None:
                attempt_report["registered_models"] = [
                    sorted(model.image(image_id).name for image_id in model.reg_image_ids())
                    for model in reconstructions.values()
                ]
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            reconstruction = next((model for model in reconstructions.values() if model.num_reg_images() == len(names)), None)
            if reconstruction is not None:
                registered = {image.name: image for image in map(reconstruction.image, reconstruction.reg_image_ids())}
                break
        else:
            failure = f"COLMAP could not register all {len(names)} views in one scene."
            detail = f"See {report_path}." if report_path is not None else "The generated views must share consistent scene details."
            raise ValueError(f"{failure} {detail}")

        calibration = registered[names[0]].camera.calibration_matrix()
        if not np.isfinite(calibration).all() or min(calibration[0, 0], calibration[1, 1]) <= 0:
            raise ValueError("COLMAP returned invalid camera intrinsics")
        poses, tracks = [], []
        for name in names:
            check_cancel()
            image = registered[name]
            pose = np.eye(4)
            pose[:3] = image.cam_from_world().matrix()
            if not np.isfinite(pose).all():
                raise ValueError("COLMAP returned an invalid camera pose")
            poses.append(pose)
            tracks.append(np.asarray([
                (*point.xy, pose[2, :3] @ reconstruction.point3D(point.point3D_id).xyz + pose[2, 3])
                for point in image.points2D if point.has_point3D()
            ], dtype=np.float64).reshape(-1, 3))
    return calibration, np.stack(poses), tracks


def _depth_scale(depth, tracks):
    height, width = depth.shape
    x, y, observed = tracks.T
    valid = np.isfinite(tracks).all(axis=1) & (x >= 0) & (x < width) & (y >= 0) & (y < height) & (observed > 0)
    x, y, observed = x[valid].astype(np.intp), y[valid].astype(np.intp), observed[valid]
    predicted = depth[y, x]
    valid = np.isfinite(predicted) & (predicted > 0)
    ratios = observed[valid] / predicted[valid]
    if ratios.size >= 8:
        median = np.median(ratios)
        inliers = ratios[np.abs(ratios - median) <= 0.25 * median]
        if inliers.size >= max(8, math.ceil(ratios.size / 2)):
            return float(np.median(inliers))
    raise ValueError("MoGe depths could not be aligned with the COLMAP tracks")


def _project_pixels(vertices, transform, shape, calibration):
    fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    height, width = shape
    projected = vertices if transform is None else vertices @ transform[:3, :3].T + transform[:3, 3]
    indices = np.flatnonzero(projected[:, 2] > 1e-6)
    x, y, z = projected[indices].T
    u, v = fx * x / z + cx, fy * y / z + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return indices[inside], v[inside].astype(np.intp), u[inside].astype(np.intp), z[inside]


def _additional_surface_vertices(vertices, index, depth, poses, camera_to_world, calibration, accepted, stride,
                                 capture_holes=None):
    import cv2

    shape = depth.shape[1:]
    coverage = np.zeros(shape, dtype=np.uint8)
    from_source = poses[index] @ camera_to_world[0]
    for surface in accepted:
        _, y, x, _ = _project_pixels(surface, from_source, shape, calibration)
        coverage[y, x] = 1
    # Fill pixel gaps between projected vertices, including decimated meshes.
    coverage = cv2.dilate(coverage, np.ones((2 * stride + 1, 2 * stride + 1), dtype=np.uint8))
    indices, y, x, _ = _project_pixels(vertices, None, shape, calibration)
    uncovered = coverage[y, x] == 0
    holes = capture_holes[y[uncovered], x[uncovered]] if capture_holes is not None else None
    indices = indices[uncovered]
    candidates = vertices[indices]

    source_idx, y, x, z = _project_pixels(candidates, poses[0] @ camera_to_world[index], shape, calibration)
    source_depth = depth[0, y, x]
    known = np.isfinite(source_depth)
    tolerance = 0.08
    # A seed render can prove a disocclusion even when no other shifted view sees it.
    valid = np.ones(len(candidates), dtype=bool)
    valid[source_idx[known]] = np.abs(z[known] - source_depth[known]) <= tolerance * source_depth[known]
    hidden = source_idx[known & (z > (1 + tolerance) * source_depth)]
    if holes is not None:
        valid[hidden[holes[hidden]]] = True
        hidden = hidden[~valid[hidden]]
    for other in range(1, len(depth)):
        if other == index or not len(hidden):
            continue
        seen, y, x, z = _project_pixels(candidates[hidden], poses[other] @ camera_to_world[index], shape, calibration)
        expected = depth[other, y, x]
        agrees = np.isfinite(expected) & (np.abs(z - expected) <= tolerance * expected)
        valid[hidden[seen[agrees]]] = True
        hidden = hidden[~valid[hidden]]
    keep = np.zeros(len(vertices), dtype=bool)
    keep[indices[valid]] = True
    return keep


def merge_moge_views(geometry, images, calibration, poses, tracks, call_node, check_cancel, *, capture_holes=None):
    import torch

    depth = geometry.get("depth")
    if not torch.is_tensor(depth) or tuple(depth.shape) != tuple(images.shape[:3]):
        raise ValueError("MoGe did not return a depth map for each camera view")
    depth = depth.detach().cpu().float()
    scales = [_depth_scale(np.asarray(view), track) for view, track in zip(depth, tracks)]
    # COLMAP's world scale is arbitrary. Use the original MoGe frame as the scale reference.
    depth = depth * torch.tensor([scale / scales[0] for scale in scales])[:, None, None]
    depth = depth.masked_fill(~torch.isfinite(depth) | (depth <= 0), torch.inf)
    depth_maps = np.asarray(depth)
    poses = poses.copy()
    poses[:, :3, 3] /= scales[0]
    camera_to_world = np.linalg.inv(poses)
    fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    count, height, width, _ = images.shape
    x = (torch.arange(width, dtype=torch.float32) + 0.5 - cx) / fx
    y = (torch.arange(height, dtype=torch.float32) + 0.5 - cy) / fy
    flip_yz = torch.tensor([1., -1., -1.])
    rows = math.ceil(count / 2)
    tile_height = min(height, 8192 // rows)
    if tile_height < height:
        images = torch.nn.functional.interpolate(images.movedim(-1, 1), size=(tile_height, width),
                                                 mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    source_pixels = torch.tensor([width - 1, tile_height - 1])
    texture_size = torch.tensor([2 * width, rows * tile_height])
    texture = images.new_zeros((rows * tile_height, 2 * width, 3))
    vertices, faces, uvs = [], [], []
    offset = 0
    stride = max(1, math.ceil(max(width, height) / 2048))
    for index, view_depth in enumerate(depth):
        check_cancel()
        points = torch.stack((view_depth * x, view_depth * y[:, None], view_depth), dim=-1)
        view = {"points": points.unsqueeze(0), "depth": view_depth.unsqueeze(0), "intrinsics": calibration}
        mesh, = call_node("MoGePointMapToMesh", moge_geometry=view, batch_index=0, texture=False,
                          decimation=stride, discontinuity_threshold=0.04)
        verts, triangles = mesh.vertices[0] * flip_yz, mesh.faces[0].long()
        if index:
            keep = _additional_surface_vertices(np.asarray(verts), index, depth_maps, poses, camera_to_world,
                                                calibration, [np.asarray(surface) for surface in vertices], stride,
                                                capture_holes.get(index) if capture_holes else None)
            triangles = triangles[torch.from_numpy(keep)[triangles].all(dim=1)]
        if not len(triangles):
            continue
        used = torch.zeros(len(verts), dtype=torch.bool)
        used[triangles.flatten()] = True
        transform = torch.from_numpy(poses[0] @ camera_to_world[index]).float()
        vertices.append(verts[used] @ transform[:3, :3].T + transform[:3, 3])
        faces.append((used.cumsum(0) - 1)[triangles] + offset)
        offset += len(vertices[-1])
        top, left = index // 2 * tile_height, index % 2 * width
        texture[top:top + tile_height, left:left + width] = images[index]
        tile = torch.tensor([left + 0.5, top + 0.5])
        uvs.append((mesh.uvs[0, used] * source_pixels + tile) / texture_size)

    return type(mesh)(vertices=(torch.cat(vertices) * flip_yz)[None], faces=torch.cat(faces)[None],
                      uvs=torch.cat(uvs)[None], texture=texture[None], unlit=True)
