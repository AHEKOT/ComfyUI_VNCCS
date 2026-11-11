"""Render, calibrate and merge camera references for MoGe reconstruction."""

import json
import math
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


REFINED_MOGE_VIEWS = (("up", (0, -2, 0)), ("forward", (0, 0, 2)), ("right", (1, 0, 0)), ("left", (-1, 0, 0)))


def require_colmap():
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError('MoGe requires PyCOLMAP. Install "pycolmap>=4.1,<5" in the ComfyUI Python environment.') from error
    if getattr(pycolmap, "FeatureExtractionOptions", None) is None:
        raise RuntimeError("MoGe requires PyCOLMAP 4.1 or newer")
    return pycolmap


def render_shifted_views(mesh, intrinsics, check_cancel, *, turn_camera=True):
    import torch
    from torch.nn.functional import grid_sample
    from comfy.model_management import get_torch_device
    try:
        from comfy_extras.sam3d_body.rasterizer import _rasterize_person
    except ImportError as error:
        raise RuntimeError("Update ComfyUI to use MoGe Refine views with the native mesh rasterizer") from error

    device = get_torch_device()
    texture = mesh.texture[0].to(device=device, dtype=torch.float32)
    height, width, _ = texture.shape
    texture = texture.permute(2, 0, 1).unsqueeze(0).contiguous()
    vertices = mesh.vertices[0].to(device=device, dtype=torch.float32)
    vertices = vertices * vertices.new_tensor([1, -1, -1])
    faces = mesh.faces[0].to(device=device, dtype=torch.long)
    uvs = mesh.uvs[0].to(device=device, dtype=torch.float32)
    sample = vertices[::max(1, len(vertices) // 4096), 2]
    sample = sample[torch.isfinite(sample) & (sample > 0)]
    if not len(sample):
        raise ValueError("MoGe returned no visible geometry for the shifted views")
    depth_unit = float(torch.quantile(sample, 0.25))
    vertices = vertices / depth_unit
    fx, fy = float(intrinsics[0, 0]) * width, float(intrinsics[1, 1]) * height
    cx, cy = float(intrinsics[0, 2]) * width, float(intrinsics[1, 2]) * height
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("MoGe returned invalid camera intrinsics for the shifted views")
    # Each view starts at the source pose in OpenCV coordinates (+X right, +Y down).
    calibration = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    poses = np.repeat(np.eye(4)[None], 1 + len(REFINED_MOGE_VIEWS), axis=0)
    views = []
    sine, cosine = math.sin(math.radians(20)), math.cos(math.radians(20))
    for index, (name, direction) in enumerate(REFINED_MOGE_VIEWS, 1):
        check_cancel()
        center = 0.12 * np.asarray(direction)
        rotation = poses[index, :3, :3]
        if turn_camera and name == "up":
            rotation[:] = ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine))
        elif turn_camera and name in ("right", "left"):
            yaw = sine if name == "right" else -sine
            rotation[:] = ((cosine, 0, yaw), (0, 1, 0), (-yaw, 0, cosine))
        view = vertices - vertices.new_tensor(center)
        if turn_camera and name != "forward":
            view = view @ vertices.new_tensor(rotation.T)
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
            return grid_sample(texture, grid, align_corners=True, padding_mode="border")[0, :, 0].T

        _rasterize_person(view, faces, fx, width, height, z_buffer, colors, visible, shade)
        if not visible.any():
            raise ValueError("MoGe produced an empty shifted camera view")
        poses[index, :3, 3] = -depth_unit * (rotation @ center)
        views.append((colors.reshape(1, height, width, 3).cpu(), (~visible.reshape(height, width)).cpu().numpy(),
                      z_buffer.reshape(height, width).cpu().numpy() * depth_unit))
    return views, calibration, poses


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


def capture_depth_tracks(raw_rgb, generated, existing, anchors):
    import cv2

    height, width = existing.shape
    sift = cv2.SIFT_create(nfeatures=4000)
    mask = cv2.erode(anchors.astype(np.uint8), np.ones((5, 5), dtype=np.uint8))
    first, desc = sift.detectAndCompute(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY), mask)
    second, candidate = sift.detectAndCompute(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY), mask)
    tracks = []
    if desc is not None and candidate is not None and len(candidate) >= 2:
        for match, alternative in cv2.BFMatcher().knnMatch(desc, candidate, k=2):
            x, y = first[match.queryIdx].pt
            point = second[match.trainIdx].pt
            if match.distance < 0.78 * alternative.distance and math.dist((x, y), point) <= 0.01 * max(width, height):
                z = existing[int(y), int(x)]
                if np.isfinite(z):
                    tracks.append((*point, z))
    return np.asarray(tracks, dtype=np.float64).reshape(-1, 3)


def align_expansion_depth(path, camera, raw, refined, depth, focal, check_cancel):
    import struct

    import torch
    from comfy.model_management import get_torch_device
    from comfy_extras.sam3d_body.rasterizer import _rasterize_person

    # Read the single textured primitive emitted by MeshToFile3D for this model.
    with path.open("rb") as stream:
        magic, version, _, length, kind = struct.unpack("<4sIII4s", stream.read(20))
        if magic != b"glTF" or version != 2 or kind != b"JSON" or length > 4 * 1024 * 1024:
            raise ValueError("Unsupported saved MoGe mesh")
        document = json.loads(stream.read(length))
        size, kind = struct.unpack("<I4s", stream.read(8))
        if kind != b"BIN\x00" or size > path.stat().st_size - stream.tell():
            raise ValueError("Invalid saved MoGe mesh buffer")
        data = stream.read(size)

    def accessor(index, channels):
        item = document["accessors"][index]
        view = document["bufferViews"][item["bufferView"]]
        dtype = {5123: "<u2", 5125: "<u4", 5126: "<f4"}[item["componentType"]]
        offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        return torch.from_numpy(np.frombuffer(data, dtype=dtype, count=item["count"] * channels, offset=offset)
                                .reshape(-1, channels).copy())

    primitive, = document["meshes"][0]["primitives"]
    vertices = accessor(primitive["attributes"]["POSITION"], 3)
    faces = accessor(primitive["indices"], 1).long().reshape(-1, 3)
    del data

    position = vertices.new_tensor(camera["position"])
    forward = torch.nn.functional.normalize(vertices.new_tensor(camera["target"]) - position, dim=0)
    right = torch.linalg.cross(forward, vertices.new_tensor(camera["up"]))
    if right.norm() < 1e-6:
        raise ValueError("The MoGe capture camera has an invalid up direction")
    right = torch.nn.functional.normalize(right, dim=0)
    basis = torch.stack((right, torch.linalg.cross(forward, right), forward), dim=1)
    view = (vertices - position) @ basis
    height, width = depth.shape
    device = get_torch_device()
    check_cancel()
    z_buffer = torch.full((height * width,), float("inf"), device=device)
    visible = torch.zeros(height * width, dtype=torch.bool, device=device)
    dummy = torch.zeros((height * width, 3), device=device)
    # Normalize only for the rasterizer's fixed 0.05 near plane. Keep anchors in mesh units.
    positive = view[::max(1, len(view) // 4096), 2]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("The capture does not see the saved MoGe scene")
    unit = float(positive.median()) * 0.1
    _rasterize_person((view / unit).to(device), faces.to(device), focal, width, height, z_buffer, dummy, visible,
                      lambda ids, _weights: dummy.new_zeros((len(ids), 3)))
    existing = z_buffer.reshape(height, width).cpu().numpy() * unit
    raw_rgb = np.asarray(raw.resize((width, height), Image.Resampling.LANCZOS))
    generated = (refined.numpy() * 255).round().astype(np.uint8)
    raw_visible = raw_rgb.any(axis=-1)
    anchors = np.isfinite(existing) & raw_visible

    tracks = capture_depth_tracks(raw_rgb, generated, existing, anchors)
    if len(tracks) < 12:
        raise ValueError("QI2 changed too much of the captured geometry to align the expanded MoGe view")
    try:
        scale = _depth_scale(depth.numpy(), tracks)
    except ValueError as error:
        raise ValueError("The expanded view's MoGe depth disagrees with the saved scene") from error
    pose = np.eye(4)
    flip = vertices.new_tensor([1, -1, -1])
    pose[:3, :3] = (basis.T * flip).numpy()
    pose[:3, 3] = (-basis.T @ position).numpy()
    return pose, scale


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


def _surface_quality(depth, calibration):
    fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0)
    dy, dx = np.gradient(z)
    # The perspective surface normal is proportional to (-fx*dx, -fy*dy, z + (u-cx)*dx + (v-cy)*dy).
    u = np.arange(width, dtype=np.float32) + 0.5 - cx
    v = np.arange(height, dtype=np.float32)[:, None] + 0.5 - cy
    normal = np.sqrt((fx * dx) ** 2 + (fy * dy) ** 2 + (z + u * dx + v * dy) ** 2)
    ray = np.sqrt(1 + (u / fx) ** 2 + (v / fy) ** 2)
    quality = fx * fy / np.maximum(normal * ray * z, 1e-10)
    return np.where(valid, quality, 0)


def _best_surface_vertices(vertices, index, depths, calibrations, poses, camera_to_world, qualities):
    shape = depths.shape[1:]
    indices, y, x, _ = _project_pixels(vertices, None, shape, calibrations[index])
    own_quality = np.zeros(len(vertices), dtype=np.float32)
    own_quality[indices] = qualities[index][y, x]
    keep = own_quality > 0
    for other, calibration in enumerate(calibrations):
        if other == index:
            continue
        ids, y, x, z = _project_pixels(vertices, poses[other] @ camera_to_world[index], shape, calibration)
        observed = depths[other, y, x]
        competing = qualities[other][y, x]
        own = own_quality[ids]
        # Prefer a closer or more frontal observation where both cameras see the same surface.
        comparable = np.isfinite(observed) & (np.abs(z - observed) <= 0.18 * z)
        wins = (competing > 1.2 * own) | ((other < index) & (competing >= own / 1.2))
        # A nearer surface that a stronger view should see but does not is likely a depth outlier.
        unsupported = (competing > 1.2 * own) & (z < 0.82 * observed)
        keep[ids[(comparable & wins) | unsupported]] = False
    return keep


def merge_moge_views(geometry, images, calibration, poses, tracks, call_node, check_cancel, *, capture_holes=None):
    import torch

    depth = geometry.get("depth")
    if not torch.is_tensor(depth) or tuple(depth.shape) != tuple(images.shape[:3]):
        raise ValueError("MoGe did not return a depth map for each camera view")
    depth = depth.detach().cpu().float()
    scales = [_depth_scale(np.asarray(view), track) for view, track in zip(depth, tracks)]
    # COLMAP's world scale is arbitrary. Use the original MoGe frame as the scale reference.
    poses = poses.copy()
    poses[:, :3, 3] /= scales[0]
    return _mesh_moge_surfaces(depth, images, [calibration] * len(images), poses,
                               [scale / scales[0] for scale in scales], call_node, check_cancel,
                               capture_holes=capture_holes)


def fuse_moge_views(depth, images, cameras, call_node, check_cancel):
    calibration = np.asarray(cameras["intrinsics"], dtype=np.float64)
    poses = np.asarray(cameras["world_to_camera"], dtype=np.float64)
    scales = np.asarray(cameras["depth_scales"], dtype=np.float32)
    count = len(images)
    if (tuple(depth.shape) != tuple(images.shape[:3]) or calibration.shape != (count, 3, 3)
            or poses.shape != (count, 4, 4) or scales.shape != (count,)
            or not np.isfinite(calibration).all() or not np.isfinite(poses).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0) or np.any(calibration[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("Saved camera calibration does not match the reconstruction frames")
    return _mesh_moge_surfaces(depth, images, calibration, poses, scales, call_node, check_cancel, ranked=True)


def _mesh_moge_surfaces(depth, images, calibrations, poses, scales, call_node, check_cancel, *, capture_holes=None, ranked=False):
    import torch

    depth = depth * torch.as_tensor(scales)[:, None, None]
    depth = depth.masked_fill(~torch.isfinite(depth) | (depth <= 0), torch.inf)
    depth_maps = np.asarray(depth)
    qualities = [_surface_quality(view, calibration) for view, calibration in zip(depth_maps, calibrations)] if ranked else None
    camera_to_world = np.linalg.inv(poses)
    count, height, width, _ = images.shape
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
    for index, (view_depth, calibration) in enumerate(zip(depth, calibrations, strict=True)):
        check_cancel()
        fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
        x = (torch.arange(width, dtype=torch.float32) + 0.5 - cx) / fx
        y = (torch.arange(height, dtype=torch.float32) + 0.5 - cy) / fy
        points = torch.stack((view_depth * x, view_depth * y[:, None], view_depth), dim=-1)
        view = {"points": points.unsqueeze(0), "depth": view_depth.unsqueeze(0), "intrinsics": calibration}
        mesh, = call_node("MoGePointMapToMesh", moge_geometry=view, batch_index=0, texture=False,
                          decimation=stride, discontinuity_threshold=0.04)
        verts, triangles = mesh.vertices[0] * flip_yz, mesh.faces[0].long()
        if ranked:
            keep = _best_surface_vertices(np.asarray(verts), index, depth_maps, calibrations, poses, camera_to_world, qualities)
            triangles = triangles[torch.from_numpy(keep)[triangles].all(dim=1)]
        elif index:
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

    if not faces:
        raise ValueError("MoGe found no consistent surfaces across the reconstruction views")
    return type(mesh)(vertices=(torch.cat(vertices) * flip_yz)[None], faces=torch.cat(faces)[None],
                      uvs=torch.cat(uvs)[None], texture=texture[None], unlit=True)
