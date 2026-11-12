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
    if diagnostics is not None and len(images) == len(REFINED_MOGE_VIEWS) + 1 and focal_length is not None:
        labels = ["original", *(name for name, _ in REFINED_MOGE_VIEWS)]
        names = [f"{index:02}-{label}.png" for index, label in enumerate(labels)]
    else:
        names = [f"{index:02}.png" for index in range(len(images))]
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


def depth_transform(depth, tracks):
    height, width = depth.shape
    x, y, observed = tracks.T
    valid = np.isfinite(tracks).all(axis=1) & (x >= 0) & (x < width) & (y >= 0) & (y < height) & (observed > 0)
    x, y, observed = x[valid].astype(np.intp), y[valid].astype(np.intp), observed[valid]
    predicted = depth[y, x]
    valid = np.isfinite(predicted) & (predicted > 0)
    predicted, observed = predicted[valid], observed[valid]
    ratios = observed / predicted
    if ratios.size < 8:
        raise ValueError("MoGe depths could not be aligned with the reference observations")
    median = np.median(ratios)
    robust = np.abs(ratios - median) <= 0.25 * median
    inliers = ratios[robust]
    if inliers.size < max(8, math.ceil(ratios.size / 2)):
        raise ValueError("MoGe depths could not be aligned with the reference observations")
    scale = float(np.median(inliers))
    # A shift is unstable when the shared observations are all at similar depths.
    if inliers.size < 12 or np.ptp(np.percentile(predicted[robust], [10, 90])) < 0.25 * np.median(predicted[robust]):
        return scale, 0.
    samples, target = predicted[robust], observed[robust]
    affine_scale, offset = np.linalg.lstsq(np.column_stack((samples, np.ones_like(samples))) / target[:, None],
                                          np.ones_like(target), rcond=None)[0]
    error = np.abs(scale * samples - target) / target
    affine_error = np.abs(affine_scale * samples + offset - target) / target
    if (0.5 * scale <= affine_scale <= 1.5 * scale and abs(offset) <= 0.35 * np.median(target)
            and np.percentile(affine_error, 75) < 0.8 * np.percentile(error, 75)):
        return float(affine_scale), float(offset)
    return scale, 0.


def capture_depth_tracks(raw_rgb, generated, existing, anchors):
    import cv2

    height, width = existing.shape
    sift = cv2.SIFT_create(nfeatures=4000)
    mask = cv2.erode(anchors.astype(np.uint8), np.ones((5, 5), dtype=np.uint8))
    first, desc = sift.detectAndCompute(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY), mask)
    second, candidate = sift.detectAndCompute(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY), mask)
    # The composite preserves pixels in reliable overlap, including areas without SIFT features.
    step = max(1, math.ceil(math.sqrt(height * width / 4096)))
    matching = np.abs(raw_rgb[::step, ::step].astype(np.int16) - generated[::step, ::step]).max(axis=-1) <= 8
    ys, xs = np.nonzero(anchors[::step, ::step] & matching)
    ys, xs = ys * step, xs * step
    tracks = np.column_stack((xs + 0.5, ys + 0.5, existing[ys, xs]))
    feature_tracks = []
    if desc is not None and candidate is not None and len(candidate) >= 2:
        for match, alternative in cv2.BFMatcher().knnMatch(desc, candidate, k=2):
            x, y = first[match.queryIdx].pt
            point = second[match.trainIdx].pt
            if match.distance < 0.78 * alternative.distance and math.dist((x, y), point) <= 0.01 * max(width, height):
                z = existing[int(y), int(x)]
                if np.isfinite(z):
                    feature_tracks.append((*point, z))
    return np.concatenate((tracks, np.asarray(feature_tracks, dtype=np.float64).reshape(-1, 3)))


def rasterize_expansion_depth(path, camera, shape, focal, check_cancel):
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
    height, width = shape
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
    pose = np.eye(4)
    flip = vertices.new_tensor([1, -1, -1])
    pose[:3, :3] = (basis.T * flip).numpy()
    pose[:3, 3] = (-basis.T @ position).numpy()
    return existing, pose


def align_expansion_depth(path, camera, raw, refined, depth, focal, check_cancel):
    height, width = depth.shape
    existing, pose = rasterize_expansion_depth(path, camera, depth.shape, focal, check_cancel)
    raw_rgb = np.asarray(raw.resize((width, height), Image.Resampling.LANCZOS))
    generated = (refined.numpy() * 255).round().astype(np.uint8)
    tracks = capture_depth_tracks(raw_rgb, generated, existing, depth_interior(existing))
    if len(tracks) < 12:
        raise ValueError("QI2 changed too much of the captured geometry to align the expanded MoGe view")
    try:
        scale, offset = depth_transform(depth.numpy(), tracks)
    except ValueError as error:
        raise ValueError("The expanded view's MoGe depth disagrees with the saved scene") from error
    return pose, scale, offset


def depth_interior(depth):
    import cv2

    valid = np.isfinite(depth) & (depth > 0)
    finite = np.where(valid, depth, 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    near, far = cv2.erode(finite, kernel), cv2.dilate(finite, kernel)
    return valid & (near > 0) & (far - near <= 0.04 * finite)


def composite_capture(raw, generated, depth):
    import cv2

    raw_rgb = np.asarray(raw.resize(generated.size, Image.Resampling.LANCZOS), dtype=np.float32)
    generated_rgb = np.asarray(generated, dtype=np.float32)
    agreement = cv2.blur(np.abs(raw_rgb - generated_rgb).mean(axis=-1), (5, 5)) <= 32
    reliable = depth_interior(depth) & agreement
    if reliable.all():
        return Image.fromarray(raw_rgb.astype(np.uint8))
    distance = cv2.distanceTransform(reliable.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3)
    weight = (distance / 3).clip(0, 1)[..., None]
    pixels = generated_rgb + (raw_rgb - generated_rgb) * weight
    return Image.fromarray(pixels.round().astype(np.uint8))


def _project_pixels(vertices, transform, shape, calibration):
    fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    height, width = shape
    projected = vertices if transform is None else vertices @ transform[:3, :3].T + transform[:3, 3]
    indices = np.flatnonzero(projected[:, 2] > 1e-6)
    x, y, z = projected[indices].T
    u, v = fx * x / z + cx, fy * y / z + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return indices[inside], v[inside].astype(np.intp), u[inside].astype(np.intp), z[inside]


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


def _consensus_depths(depths, calibrations, poses, check_cancel):
    import cv2

    height, width = depths.shape[1:]
    step = max(1, math.ceil(math.sqrt(height * width / 300_000)))
    y, x = np.mgrid[:height:step, :width:step]
    interiors = [depth_interior(depth) for depth in depths]
    camera_to_world = np.linalg.inv(poses)
    aligned = []
    for index, (depth, calibration) in enumerate(zip(depths, calibrations, strict=True)):
        check_cancel()
        grid = interiors[index][::step, ::step]
        ys, xs = y[grid], x[grid]
        values = depth[ys, xs]
        rays = np.column_stack(((xs + 0.5 - calibration[0, 2]) / calibration[0, 0],
                                (ys + 0.5 - calibration[1, 2]) / calibration[1, 1], np.ones_like(values)))
        points = rays * values[:, None]
        correction = np.zeros(len(points), dtype=np.float32)
        support = np.ones_like(correction)
        for other, target in enumerate(calibrations):
            if other == index:
                continue
            check_cancel()
            transform = poses[other] @ camera_to_world[index]
            ids, py, px, projected = _project_pixels(points, transform, (height, width), target)
            visible = interiors[other][py, px]
            ids, py, px, projected = ids[visible], py[visible], px[visible], projected[visible]
            slope = rays[ids] @ transform[2, :3]
            # Use only nearby surfaces that agree in the overlap. Never pull a point across an occlusion.
            error = depths[other, py, px] - projected
            delta = np.divide(error, slope, out=np.zeros_like(error), where=slope > 0.25)
            agrees = (slope > 0.25) & (np.abs(delta) <= 0.08 * np.minimum(values[ids], values[ids] + delta))
            ids, delta, slope = ids[agrees], delta[agrees], slope[agrees]
            refined = rays[ids] * (values[ids] + delta)[:, None]
            checked, py, px, projected = _project_pixels(refined, transform, (height, width), target)
            stable = interiors[other][py, px] & (np.abs(depths[other, py, px] - projected) <= 0.02 * projected)
            checked = checked[stable]
            # A grazing observation gives a weaker constraint along the current ray.
            weight = slope[checked].clip(max=1)
            correction[ids[checked]] += weight * delta[checked]
            support[ids[checked]] += weight
        field = np.zeros(grid.shape, dtype=np.float32)
        field[grid] = correction / support
        if step > 1:
            field = cv2.warpAffine(field, np.array([[step, 0, 0], [0, step, 0]], dtype=np.float32),
                                   (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result = depth.copy()
        valid = interiors[index]
        bound = 0.08 * depth[valid]
        result[valid] += field[valid].clip(-bound, bound)
        aligned.append(result)
    return np.stack(aligned)


def _overlap_faces(vertices, triangles, index, depths, calibrations, poses, camera_to_world, qualities, interiors):
    shape = depths.shape[1:]
    indices, y, x, _ = _project_pixels(vertices, None, shape, calibrations[index])
    own_quality = np.zeros(len(vertices), dtype=np.float32)
    own_quality[indices] = qualities[index][y, x]
    covered = np.zeros(len(triangles), dtype=bool)
    corroborated = np.zeros(len(vertices), dtype=bool)
    contradicted = np.zeros(len(vertices), dtype=bool)
    for other, calibration in enumerate(calibrations):
        if other == index:
            continue
        ids, y, x, z = _project_pixels(vertices, poses[other] @ camera_to_world[index], shape, calibration)
        interior = interiors[other][y, x]
        ids, y, x, z = ids[interior], y[interior], x[interior], z[interior]
        observed = depths[other, y, x]
        competing, own = qualities[other][y, x], own_quality[ids]
        comparable = np.abs(z - observed) <= 0.06 * z
        wins = (competing > 1.2 * own) | ((other < index) & (competing >= own / 1.2))
        corroborated[ids[comparable]] = True
        contradicted[ids[(competing > 1.2 * own) & (z < 0.82 * observed)]] = True
        # Drop only faces fully covered by one observation. Different winners at the
        # three corners do not establish coverage and used to cut cracks in the mesh.
        view_coverage = np.zeros(len(vertices), dtype=bool)
        view_coverage[ids[comparable & wins]] = True
        covered |= view_coverage[triangles].all(axis=1)
    return covered | (contradicted & ~corroborated)[triangles].all(axis=1)


def fuse_moge_views(depth, images, cameras, call_node, check_cancel):
    calibration = np.asarray(cameras["intrinsics"], dtype=np.float64)
    poses = np.asarray(cameras["world_to_camera"], dtype=np.float64)
    scales = np.asarray(cameras["depth_scales"], dtype=np.float32)
    count = len(images)
    offsets = np.asarray(cameras.get("depth_offsets", np.zeros(count)), dtype=np.float32)
    if (tuple(depth.shape) != tuple(images.shape[:3]) or calibration.shape != (count, 3, 3)
            or poses.shape != (count, 4, 4) or scales.shape != (count,) or offsets.shape != (count,)
            or not np.isfinite(calibration).all() or not np.isfinite(poses).all() or not np.isfinite(scales).all()
            or not np.isfinite(offsets).all()
            or np.any(scales <= 0) or np.any(calibration[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("Saved camera calibration does not match the reconstruction frames")
    import torch

    valid = torch.isfinite(depth) & (depth > 0)
    aligned = depth * torch.as_tensor(scales)[:, None, None] + torch.as_tensor(offsets)[:, None, None]
    return _mesh_moge_surfaces(aligned.masked_fill(~valid, torch.inf), images, calibration, poses, call_node, check_cancel)


def _mesh_moge_surfaces(depth, images, calibrations, poses, call_node, check_cancel):
    import cv2
    import torch

    depth = depth.masked_fill(~torch.isfinite(depth) | (depth <= 0), torch.inf)
    depth_maps = _consensus_depths(depth.numpy(), calibrations, poses, check_cancel)
    depth = torch.from_numpy(depth_maps)
    qualities = [_surface_quality(view, calibration) for view, calibration in zip(depth_maps, calibrations, strict=True)]
    # Keep both surfaces across a small band at image and occlusion edges.
    interiors = [cv2.erode(depth_interior(view).astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for view in depth_maps]
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
        covered = _overlap_faces(np.asarray(verts), triangles.numpy(), index, depth_maps, calibrations, poses,
                                 camera_to_world, qualities, interiors)
        triangles = triangles[~torch.from_numpy(covered)]
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
