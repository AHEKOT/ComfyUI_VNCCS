"""CG jobs and HTTP routes preserve the main extension's scene ownership."""

import asyncio
import copy
import io
import json
import sys
import types
from unittest import mock

import pytest
from test_background_creator import web
from PIL import Image

from helpers.backend_package import service_package
from helpers.cg_factory_fixture import factory_fixture


load = service_package("vnccs_cg_service_test")
service = load("nodes.cg_creator")
library = load("nodes.cg_creator")
generation = load("nodes.cg_creator")
routes_module = load("nodes.cg_creator")
runtime, factory = factory_fixture()
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 42}
MAIN_FIELDS = ("editor", "images", "selected_image_id", "generator_settings", "generated_sets", "future_owner_data")


def png(size=(128, 64), color="red"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def write_document(document):
    path = library.root() / document["name"] / f"{document['name']}_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2).encode() + b"\n"
    path.write_bytes(encoded)
    path.with_suffix(".json.backup").write_bytes(encoded)
    return path


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    monkeypatch.setattr(library.background, "root", library.root)
    monkeypatch.setattr(service, "provider", lambda: {"runtime": runtime, "storage": factory, "web_root": tmp_path / "web"})
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {}, raising=False)
    monkeypatch.setattr(factory, "_factory_root", lambda: tmp_path / "factory")
    monkeypatch.setattr(service, "_JOBS", {})
    factory_jobs = {}
    monkeypatch.setattr(runtime, "_JOBS", factory_jobs)
    monkeypatch.setattr(factory, "_JOBS", factory_jobs)
    worker = mock.Mock()
    monkeypatch.setattr(service, "_WORKER", worker)
    monkeypatch.setattr(runtime, "_emit", mock.Mock())
    monkeypatch.setattr(generation, "progress", lambda _job_id: {"stage": "unknown"})
    monkeypatch.setattr(generation, "interrupt", mock.Mock(return_value=False))
    generated = mock.Mock(return_value=(Image.new("RGB", (128, 64), "blue"),
        {"settings": generation.normalize_settings({}), "prompt": "Finished camera frame", "model": "qi2"}))
    monkeypatch.setattr(generation, "generate_image", generated)
    document = {"schema_version": 2, "background_id": "a" * 32, "name": "Quiet room", "revision": 7,
        "archived": False, "editor": {"description": "Keep the main generator scene", "future": {"x": 3}},
        "images": [{"image_id": "b" * 32, "file": "source.png", "width": 128, "height": 64,
            "settings": {"generator": "main", "seed": 3}, "preview_revision": 4}],
        "selected_image_id": "b" * 32, "models": [], "cgs": [],
        "generator_settings": {"model": "qi2", "sampling": {"steps": 25}, "future": [1, 2]},
        "generated_sets": [],
        "future_owner_data": {"revision": "unrelated", "values": ["keep", "all"]}}
    write_document(document)
    for index, image in enumerate(document["images"]):
        target = library.root() / document["name"] / image["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png(color="red" if index == 0 else "green"))

    def run_pending():
        function, job, callback = worker.submit.call_args.args
        assert function is runtime._run_job
        function(job, callback)
        return job

    return types.SimpleNamespace(document=document, worker=worker, generated=generated,
                                 run_pending=run_pending, tmp_path=tmp_path)


def with_generated_set(workspace, document=None):
    document = copy.deepcopy(document or workspace.document)
    set_id = "7" * 32
    group = {"set_id": set_id, "version": "", "source_image_id": document["images"][0]["image_id"],
        "variants": [{"id": "morning", "name": "Morning"}, {"id": "night", "name": "Night"}],
        "settings": {"future": "Keep main set settings"}, "images": []}
    for index, (name, color) in enumerate((("Original", "red"), ("Morning", "yellow"), ("Night", "navy"))):
        relative = f"Images/{index:02d}_{name}.png"
        target = library.root() / document["name"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png(color=color))
        group["images"].append({"name": name, "file": relative, "width": 128, "height": 64,
                                "revision": document["revision"], "settings": {"seed": index}})
    document["generated_sets"].append(group)
    write_document(document)
    return document


def source_variant(document, index=1):
    group = document["generated_sets"][-1]
    variant = group["variants"][index - 1]
    return {"set_id": group["set_id"], "index": index, "variant_id": variant["id"], "name": variant["name"]}


def with_model(workspace, document=None, image_id=None, source_variant=None):
    document = document or workspace.document
    source = workspace.tmp_path / "source-scene"
    object_id = "d" * 32
    assets = source / "objects" / object_id
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "model.ply").write_bytes(b"immutable Gaussian fixture")
    scene = {"schema_version": 11, "scene_id": "e" * 32, "camera": CAMERA,
        "objects": [{"object_id": object_id, "source_camera": CAMERA,
                     "files": {"ply": f"objects/{object_id}/model.ply"}}]}
    return library.commit_model(document["background_id"], image_id or document["images"][0]["image_id"],
                                source, scene, {"seed": 1}, document["revision"], source_variant=source_variant)


def job_payload(document, action="cg", **changes):
    model = document["models"][-1] if document.get("models") else {}
    return {"action": action, "revision": document["revision"],
        "image_id": model.get("source_image_id", document["images"][0]["image_id"]),
        "source_variant": copy.deepcopy(model.get("source_variant")),
        "model_id": model.get("model_id", ""), "name": "Window", "camera": copy.deepcopy(CAMERA),
        "settings": copy.deepcopy(generation.DEFAULT_SETTINGS), **changes}


def assert_main_fields_preserved(before, after):
    assert {key: after[key] for key in MAIN_FIELDS} == {key: before[key] for key in MAIN_FIELDS}


@pytest.mark.parametrize("count", [4, 5])
def test_multiview_meshing_aligns_depth_scale_and_textures_to_original_camera(count):
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    calibration = np.array([[4., 0, 2], [0, 4, 2], [0, 0, 1]])
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [4, 0, 0], [-4, 0, 0], [0, 4, 0], [0, -4, 0]][:count]
    tracks = [np.array([[x + 0.5, y + 0.5, 4.] for x in range(4) for y in range(4)])] * count
    depth = torch.stack([torch.full((4, 4), value) for value in (2., 1., 4., 2., 8.)[:count]])
    colors = (0., 0.25, 0.5, 0.75, 1.)[:count]
    images = torch.stack([torch.full((4, 4, 3), value) for value in colors])

    def triangulate(name, *, moge_geometry, **_options):
        assert name == "MoGePointMapToMesh"
        points = moge_geometry["points"][:, :2, :2].reshape(1, 4, 3)
        return (types.SimpleNamespace(vertices=points * torch.tensor([1, -1, -1]),
            faces=torch.tensor([[[0, 2, 1], [1, 2, 3]]]), uvs=torch.tensor([[[0., 0], [1, 0], [0, 1], [1, 1]]])),)

    scales, offsets = np.array([multiview.depth_transform(d.numpy(), t) for d, t in zip(depth, tracks)]).T
    poses[:, :3, 3] /= scales[0]
    cameras = {"intrinsics": [calibration] * count, "world_to_camera": poses,
               "depth_scales": scales / scales[0], "depth_offsets": offsets / scales[0]}
    mesh = multiview.fuse_moge_views(depth, images, cameras, triangulate, lambda: None)
    assert torch.all(mesh.vertices[0, :, 2] == -2)
    assert mesh.vertices[0, ::4, :2].tolist() == [[-0.75, 0.75], [-2.75, 0.75], [1.25, 0.75], [-0.75, 2.75], [-0.75, -1.25]][:count]
    assert mesh.faces.shape == (1, 2 * count, 3)
    height, width = mesh.texture.shape[1:3]
    assert (height, width) == (4 * ((count + 1) // 2), 8)
    tile_origins = [(4 * (index // 2), 4 * (index % 2)) for index in range(count)]
    np.testing.assert_allclose(mesh.uvs[0, ::4], [[(x + .5) / width, (y + .5) / height] for y, x in tile_origins])
    assert [mesh.texture[0, y, x, 0].item() for y, x in tile_origins] == list(colors)


def test_moge_overlap_preserves_faces_at_the_boundary():
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    calibration = np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    depths = np.full((3, 8, 8), 2., dtype=np.float32)
    qualities = np.ones_like(depths)
    interiors = np.ones(depths.shape, dtype=bool)
    # The same wall continues across the boundary between views 0 and 1.
    interiors[0, :, 4:] = False
    interiors[1, :, :4] = False
    pixels = np.array([[2.5, 2.5], [3.5, 2.5], [2.5, 3.5], [4.5, 2.5]])
    vertices = np.column_stack(((pixels - 4.) / 4, np.full(len(pixels), 2.)))
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    dropped = multiview._overlap_faces(vertices, triangles, 2, depths, [calibration] * 3,
                                        poses, poses, qualities, interiors)
    assert dropped.tolist() == [True, False]


def test_moge_overlap_aligns_a_wall_without_crossing_an_occlusion():
    pytest.importorskip("cv2")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    calibration = np.array([[24., 0., 12.], [0., 24., 12.], [0., 0., 1.]])
    depths = np.stack([np.full((24, 24), value, dtype=np.float32) for value in (2., 2.1)])
    depths[1, :, 18:] = 1.
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 0.2
    fused = multiview._consensus_depths(depths, [calibration] * 2, poses, lambda: None)
    assert fused[0, 12, 8] == pytest.approx(2.05)
    assert fused[1, 12, 10] == pytest.approx(2.05)
    assert fused[0, 12, 18] == 2.
    assert fused[1, 12, 20] == 1.


def test_moge_depth_registration_fits_a_shift_from_reliable_tracks():
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    depth = np.linspace(1, 4, 32)[None]
    tracks = np.column_stack((np.arange(32) + .5, np.full(32, .5), 1.3 * depth[0] + .4))
    tracks[-1, 2] = 100
    assert multiview.depth_transform(depth, tracks) == pytest.approx((1.3, .4))


def test_moge_composite_keeps_observed_pixels_and_leaves_new_geometry_visible():
    pytest.importorskip("cv2")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    raw = Image.new("RGB", (32, 16), (0, 0, 0))
    generated = Image.new("RGB", raw.size, (18, 18, 18))
    depth = np.ones((16, 32))
    depth[:, 16:] = np.inf
    result = multiview.composite_capture(raw, generated, depth)
    assert result.getpixel((4, 8)) == (0, 0, 0)
    assert result.getpixel((20, 8)) == (18, 18, 18)
    assert 0 < result.getpixel((13, 8))[0] < 18


def test_moge_fusion_uses_closer_view_for_shared_surface(monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    monkeypatch.setattr(multiview, "_restore_reference_coverage", lambda *_args: None)
    calibration = [[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]]
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 2, 3] = -0.5
    cameras = {"intrinsics": [calibration] * 2, "world_to_camera": poses, "depth_scales": [1., 1.]}
    depths = torch.stack((torch.full((8, 8), 2.), torch.full((8, 8), 1.55)))
    images = torch.stack((torch.zeros((8, 8, 3)), torch.ones((8, 8, 3))))

    def triangulate(_name, *, moge_geometry, **_options):
        points = moge_geometry["points"][:, 3:5, 3:5].reshape(1, 4, 3)
        return (types.SimpleNamespace(vertices=points * torch.tensor([1, -1, -1]),
            faces=torch.tensor([[[0, 2, 1], [1, 2, 3]]]), uvs=torch.tensor([[[0., 0], [1, 0], [0, 1], [1, 1]]])),)

    mesh = multiview.fuse_moge_views(depths, images, cameras, triangulate, lambda: None)

    assert mesh.faces.shape == (1, 2, 3)
    torch.testing.assert_close(mesh.vertices[0, :, 2], torch.full((4,), -2.025))
    assert torch.all(mesh.uvs[0, :, 0] > 0.5)


@pytest.mark.parametrize("upper_view", [False, True])
def test_moge_fusion_restores_only_faces_that_cover_missing_reference_pixels(monkeypatch, upper_view):
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")

    def rasterize(view, faces, _focal, width, height, depth, owners, visible, shade):
        assert (height, width) == (2, 4)
        if upper_view:
            torch.testing.assert_close(view[0, 0], view.new_tensor(2_000.))
        pixels = (faces[:, 0] == 4).long()
        wins = depth[pixels] > 1
        ids = torch.where(wins)[0]
        depth[pixels[ids]] = 1
        owners[pixels[ids]] = shade(ids, None)
        visible[pixels[ids]] = True

    monkeypatch.setitem(sys.modules, "comfy_extras.sam3d_body.rasterizer",
                        types.SimpleNamespace(_rasterize_person=rasterize))
    vertices = torch.ones((8, 3))
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [4, 5, 6], [5, 6, 7]])
    dropped = torch.tensor([False, True, True, True])
    surfaces = [(vertices, faces, dropped, None)]
    if upper_view:
        other = torch.ones(4, dtype=torch.bool)
        surfaces.insert(0, (vertices, faces, other, None))
        transform = torch.eye(4)
        transform[0, 3] = 1.
        multiview._restore_reference_coverage(surfaces, (1, 2), np.diag([1., 1., 1.]), lambda: None,
                                              reference_to_camera=transform, source_index=1)
        assert other.all()
    else:
        multiview._restore_reference_coverage(surfaces, (1, 2), np.eye(3), lambda: None)
    assert dropped.tolist() == [False, True, False, False]


def test_moge_mesh_preserves_a_surface_next_to_a_depth_jump():
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    depth = torch.tensor([[[2., 2.], [2., 8.]]])
    image = torch.zeros(1, 2, 2, 3)

    def triangulate(_name, *, moge_geometry, discontinuity_threshold, **_options):
        assert discontinuity_threshold == 0.
        points = moge_geometry["points"].reshape(1, 4, 3)
        return (types.SimpleNamespace(vertices=points * torch.tensor([1, -1, -1]),
            faces=torch.tensor([[[0, 2, 1], [1, 2, 3]]]), uvs=torch.zeros(1, 4, 2)),)

    cameras = {"intrinsics": [np.eye(3)], "world_to_camera": [np.eye(4)], "depth_scales": [1.]}
    mesh = multiview.fuse_moge_views(depth, image, cameras, triangulate, lambda: None)
    assert mesh.faces.tolist() == [[[0, 2, 1]]]
    assert mesh.vertices[0, :, 2].tolist() == [-2.] * 3


@pytest.mark.parametrize(("turn_camera", "additional_angles"), [(True, False), (True, True), (False, False)])
def test_shifted_views_render_and_save_the_same_camera_pose(monkeypatch, turn_camera, additional_angles):
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu")))
    rendered = []

    def rasterize(view, _faces, _focal, _width, _height, depth, _colors, visible, _shade):
        rendered.append(view.clone())
        depth.fill_(float(view[0, 2]))
        visible.fill_(True)

    monkeypatch.setitem(sys.modules, "comfy_extras.sam3d_body.rasterizer",
                        types.SimpleNamespace(_rasterize_person=rasterize))
    points = torch.tensor([[-1., 0., 2.], [1., 0., 2.], [0., 1., 2.]])
    mesh = types.SimpleNamespace(texture=torch.ones(1, 8, 8, 3),
                                vertices=(points * torch.tensor([1., -1., -1.]))[None],
                                faces=torch.tensor([[[0, 1, 2]]]), uvs=torch.zeros(1, 3, 2))
    intrinsics = torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    views, _, poses = multiview.render_shifted_views(
        mesh, intrinsics, lambda: None, turn_camera=turn_camera, additional_angles=additional_angles)

    sine, cosine = np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20))
    if turn_camera:
        expected_centers = [[0, -.48, 0], [.24, 0, 0], [-.24, 0, 0], [.48, -.72, .24], [-.48, -.72, .24]]
        down, forward = np.sin(np.deg2rad(24)), np.cos(np.deg2rad(24))
        expected_look = [[0, sine, cosine], [-sine, 0, cosine], [sine, 0, cosine],
                         [-forward * sine, down, forward * cosine], [forward * sine, down, forward * cosine]]
    else:
        expected_centers = [[0, -.48, 0], [0, 0, .48], [.24, 0, 0], [-.24, 0, 0]]
        expected_look = [[0, 0, 1]] * 4
    if turn_camera and not additional_angles:
        expected_centers = expected_centers[:3]
        expected_look = expected_look[:3]
    np.testing.assert_allclose(np.linalg.inv(poses[1:])[:, :3, 3], expected_centers, atol=1e-8)
    np.testing.assert_allclose(poses[1:, 2, :3], expected_look)
    for view, actual, pose in zip(views, rendered, poses[1:], strict=True):
        expected = points.numpy() @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(actual.numpy() * 2, expected, atol=1e-7)
        assert view[2][0, 0] == pytest.approx(expected[0, 2])


@pytest.mark.parametrize("sign", [1, -1])
def test_side_camera_reaches_the_source_edge_and_preserves_overlap(sign):
    pytest.importorskip("cv2")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    height, width, focal = 108, 192, 115.
    calibration = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    y, x = np.mgrid[:height, :width]
    points = np.column_stack(((x.ravel() + .5 - width / 2) / focal,
                              (y.ravel() + .5 - height / 2) / focal, np.ones(height * width))).astype(np.float32)
    sine, cosine = np.sin(sign * np.deg2rad(20)), np.cos(np.deg2rad(20))
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    original = np.array([sign * .12, 0., 0.])
    fitted = multiview._fit_shifted_camera(points, original, rotation, (height, width), calibration, lambda: None)

    def coverage(center):
        ids, rows, cols, _ = multiview._project_pixels((points - center) @ rotation.T, None, (height, width), calibration)
        gaps = np.full(height, width)
        np.minimum.at(gaps, rows, cols if sign > 0 else width - 1 - cols)
        return np.median(gaps[height // 4:3 * height // 4]), len(ids)

    old_gap, old_overlap = coverage(original)
    gap, overlap = coverage(fitted)
    assert old_gap > width * .15
    assert gap <= .02 * width and overlap >= .95 * old_overlap
    assert sign * fitted[0] >= abs(original[0]) and fitted[2] > 0


@pytest.mark.parametrize(("reject_first_view", "additional_angles"), [(False, False), (False, True), (True, True)])
def test_moge_refine_views_keeps_diagnostics_and_publishes_only_the_final_model(
        workspace, monkeypatch, reject_first_view, additional_angles):
    torch = pytest.importorskip("torch")
    import numpy as np

    monkeypatch.setattr(runtime, "_model_operation_lock", lambda: service.threading.RLock(), raising=False)
    multiview = load("nodes.cg_moge_multiview")
    monkeypatch.setattr(multiview, "require_colmap", mock.Mock())
    calibration = torch.diag(torch.tensor([1., 1.5, 1.]))
    infer = mock.Mock(side_effect=lambda batch, *_args: {
        "intrinsics": [calibration], "depth": torch.full(batch.shape[:3], 2.),
    })
    monkeypatch.setattr(service, "_infer_moge", infer)
    mesh = object()
    export = mock.Mock(return_value=(types.SimpleNamespace(get_bytes=lambda: b"glTF fixture"),))
    monkeypatch.setattr(service, "_call_comfy_node", lambda name, **kwargs: (
        (mesh,) if name == "MoGePointMapToMesh" else export(**kwargs)))
    view_count = 5 if additional_angles else 3
    poses = np.repeat(np.eye(4)[None], view_count + 1, axis=0)
    poses[1:, :3, 3] = [[0, .48, 0], [-.24, 0, 0], [.24, 0, 0], [-.48, .72, -.24], [.48, .72, -.24]][:view_count]

    def capture(*_args, additional_angles):
        assert additional_angles == (view_count == 5)
        image = infer.call_args.args[0]
        height, width = image.shape[1:3]
        frames = [(torch.full_like(image, value), np.zeros((height, width), dtype=bool),
                   np.full((height, width), 6.)) for value in (0.2, 0.4, 0.6, 0.8, 1.)[:view_count]]
        focal = 1.5 * height
        return frames, np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]), poses

    rasterize = mock.Mock(side_effect=capture)
    monkeypatch.setattr(multiview, "render_shifted_views", rasterize)
    seen = []

    def refine(settings, _job_id, _cancel, frame, background, *, save_dataset, progress_range):
        assert settings["resolution_mp"] == 0.05 and settings["generation"]["cache_device"] == "off"
        assert not save_dataset and 0 < progress_range[0] < progress_range[1] < 0.6
        assert background.getpixel((0, 0)) == (255, 0, 0)
        seen.append(frame.getpixel((0, 0))[0])
        return frame, {}

    monkeypatch.setattr(service, "generate_image", refine)
    observed = np.array([[i + .5, i + .5, 6.] for i in range(8)])
    matcher = mock.Mock(side_effect=[np.empty((0, 3)) if reject_first_view else observed, *[observed] * (view_count - 1)])
    monkeypatch.setattr(multiview, "capture_depth_tracks", matcher)
    require = mock.Mock(side_effect=AssertionError("Refine views must not require COLMAP"))
    register = mock.Mock(side_effect=AssertionError("Refine views must use the known cameras"))
    monkeypatch.setattr(multiview, "require_colmap", require)
    monkeypatch.setattr(multiview, "register_views", register)
    merge = mock.Mock(return_value=mesh)
    monkeypatch.setattr(multiview, "fuse_moge_views", merge)
    document = workspace.document
    settings = service.normalize_settings({"resolution_mp": 3, "generation": {"cache_device": "cpu"}})
    options = {"provider": "moge", "moge_resolution_mp": 0.05}
    if additional_angles:
        options["moge_additional_angles"] = True
    service.start_job(document["background_id"], job_payload(document, "model", settings=settings, model_settings=options))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert [len(call.args[0]) for call in infer.call_args_list] == [1, view_count]
    assert rasterize.call_args.args[:2] == (mesh, calibration)
    assert seen == [51, 102, 153, 204, 255][:view_count]
    require.assert_not_called()
    register.assert_not_called()
    merge.assert_called_once()
    depths, images, cameras, *_ = merge.call_args.args
    retained = [index for index in range(view_count + 1) if index != 1 or not reject_first_view]
    assert len(depths) == len(images) == len(retained)
    assert (images[:, 0, 0, 0] * 255).round().tolist() == [[255, *seen][i] for i in retained]
    np.testing.assert_array_equal(cameras["world_to_camera"], poses[retained])
    assert cameras["depth_scales"] == [1.] + [3.] * (len(retained) - 1)
    assert cameras["depth_offsets"] == [0.] * len(retained)
    export.assert_called_once_with(mesh=mesh)
    saved = library.load_background(document["background_id"])
    assert len(saved["models"]) == 1 and not saved["cgs"]
    directory = library._model_directory(library.root() / saved["name"], saved["models"][0]["model_id"], saved)
    assert len(list((directory / "reconstruction").glob("[0-9]*.png"))) == len(retained)
    diagnostics = library.root().parent / "CG_MoGe_Diagnostics" / job["job_id"]
    with Image.open(diagnostics / "image_1.png") as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    views = ("01-up", "02-right", "03-left", "04-up-right", "05-up-left")[:view_count]
    assert {entry.name for entry in diagnostics.iterdir() if entry.is_dir()} == set(views)
    for view, pixel in zip(views, seen, strict=True):
        for file in ("image_2.png", "qi2_result.png", "reconstruction_frame.png"):
            with Image.open(diagnostics / view / file) as image:
                assert image.getpixel((0, 0)) == (pixel,) * 3
    assert not list(factory._factory_root().iterdir())


@pytest.mark.parametrize(("provider_name", "camera_source"), [
    ("moge", "camera_inputs"), ("moge", "colmap"),
    ("worldmirror", "camera_inputs"), ("worldmirror", "predicted"),
])
@pytest.mark.parametrize("save", [True, False, None])
def test_expansion_keeps_references_and_geometry_private_until_save(workspace, monkeypatch, routes, provider_name, camera_source, save):
    torch = pytest.importorskip("torch")
    import numpy as np

    multiview = load("nodes.cg_moge_multiview")
    monkeypatch.setattr(multiview, "require_colmap", mock.Mock())
    batch = torch.stack([torch.full((64, 128, 3), value) for value in (0., .25, .5, .75, 1.)])
    calls = []
    pose = np.eye(4)
    cameras = {"intrinsics": [[[100., 0., 64.], [0., 100., 32.], [0., 0., 1.]]] * len(batch),
               "world_to_camera": [pose.tolist()] * len(batch), "depth_scales": [1.] * len(batch)}
    known_views = (np.array(cameras["intrinsics"][0]), np.repeat(pose[None], len(batch), axis=0), None, None)
    monkeypatch.setattr(service, "_generate_refined_moge_views", mock.Mock(return_value=(batch, 100., {}, known_views)))
    inferred = []

    def reconstruct(_job, images, scene, folder, *_args, **_kwargs):
        calls.append(images)
        name = "model.glb" if provider_name == "moge" else "model.ply"
        (folder / name).write_bytes(f"geometry-{len(images)}".encode())
        service._set_reconstruction_object(scene, f"{len(images):032x}", CAMERA,
                                           "model" if provider_name == "moge" else "ply", name)

    worldmirror_batches = []

    def reconstruct_worldmirror(job, images, scene, folder, _settings, saved_cameras):
        reconstruct(job, images, scene, folder)
        worldmirror_batches.append(saved_cameras)
        return saved_cameras if saved_cameras is not None else {
            "intrinsics": [cameras["intrinsics"][0]] * len(images), "world_to_camera": [pose.tolist()] * len(images),
        }

    monkeypatch.setattr(service, "_generate_worldmirror_scene", reconstruct_worldmirror)
    if provider_name == "moge":
        monkeypatch.setattr(runtime, "_model_operation_lock", lambda: service.threading.RLock(), raising=False)

        def register(images, _check_cancel, **options):
            assert options["diagnostics"].is_dir()
            tracks = np.column_stack((np.arange(8) + .5, np.arange(8) + .5, np.ones(8)))
            return np.array(cameras["intrinsics"][0]), np.repeat(pose[None], len(images), axis=0), [tracks] * len(images)

        register = mock.Mock(side_effect=register)
        monkeypatch.setattr(multiview, "register_views", register)
        if camera_source == "camera_inputs":
            def generate(job, images, scene, folder, **_options):
                reconstruct(job, images, scene, folder)
                return images, {}, cameras
            monkeypatch.setattr(service, "_generate_moge_scene", generate)

        def infer(images, _fov):
            inferred.append(images)
            return {"depth": torch.ones(images.shape[:3])}

        def fuse(depth, images, calibration, _call_node, _check_cancel):
            assert depth.shape == images.shape[:3]
            expected = [1.] * len(images) if camera_source == "colmap" else [1.] * 5 + [2.]
            assert calibration["depth_scales"] == expected
            return images

        monkeypatch.setattr(service, "_infer_moge", infer)
        align = mock.Mock(return_value=(pose, 2., .1))
        monkeypatch.setattr(multiview, "rasterize_expansion_depth",
                            lambda _path, _camera, shape, _focal, _cancel: (np.ones(shape), pose))
        monkeypatch.setattr(multiview, "align_expansion_depth", align)
        monkeypatch.setattr(multiview, "fuse_moge_views", fuse)
        monkeypatch.setattr(service, "_save_moge_mesh",
                            lambda job, scene, folder, mesh, _camera: reconstruct(job, mesh, scene, folder))
    document = workspace.document
    background_id = document["background_id"]
    service.start_job(background_id, job_payload(document, "model",
        model_settings={"provider": provider_name, f"{provider_name}_camera_source": camera_source}))
    first = workspace.run_pending()
    assert first["status"] == "completed", first["error"]
    document = library.load_background(background_id)
    model_id = document["models"][0]["model_id"]
    directory = library.root() / document["name"]
    base = library._model_directory(directory, model_id, document)
    original = {path.name: path.read_bytes() for path in (base / "reconstruction").glob("*.png")}
    assert len(original) == 5
    assert service.public_model(background_id, model_id)["reconstruction_size"] == [128, 64]
    if provider_name == "moge" and camera_source == "camera_inputs":
        multiview.require_colmap.side_effect = AssertionError("MoGe expansion must reuse the captured camera")
    service.start_job(background_id, job_payload(document, "expand", model_camera=CAMERA), png(color="yellow"))
    frame = workspace.run_pending()
    assert frame["status"] == "completed", frame["error"]
    frame_id = frame["result"]["expansion_frame_id"]
    saved_model = service.public_model(background_id, model_id)
    assert saved_model["expansion_frame_id"] == frame_id
    assert "expansion_id" not in saved_model
    assert [len(images) for images in calls] == [5]
    assert library.load_background(background_id) == document
    url = "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}"
    frame_info = {"background_id": background_id, "model_id": model_id, "expansion_id": frame_id}
    for kind, pixel in (("before", (255, 255, 0)), ("after", (0, 0, 255))):
        response = call(routes, "GET", url + "/frame/{kind}", Request(info={**frame_info, "kind": kind}))
        assert response.status == 200
        with Image.open(response.path) as image:
            assert image.getpixel((0, 0)) == pixel
    if save is None:
        response = call(routes, "DELETE", url + "/frame", Request(info=frame_info))
        assert response.status == 200, response.text
        assert "expansion_frame_id" not in service.public_model(background_id, model_id)
        assert [len(images) for images in calls] == [5]
        assert not (base / ".expansion-frame").exists()
        assert all((base / "reconstruction" / name).read_bytes() == data for name, data in original.items())
        return
    service.start_job(background_id, job_payload(document, "expand", expansion_frame_id=frame_id))
    expansion = workspace.run_pending()
    assert expansion["status"] == "completed", expansion["error"]
    expansion_id = expansion["job_id"]
    assert [len(images) for images in calls] == [5, 6]
    assert calls[-1][-1, 0, 0].tolist() == [0., 0., 1.]
    if provider_name == "moge":
        if camera_source == "colmap":
            assert [len(call.args[0]) for call in register.call_args_list] == [5, 6]
            assert torch.equal(inferred[-1], calls[-1])
            align.assert_not_called()
        else:
            register.assert_not_called()
            assert torch.equal(torch.cat(inferred), calls[-1])
            assert align.call_args.args[1] == CAMERA
        diagnostics = library.root().parent / "CG_Expansion_Diagnostics" / frame_id
        assert (diagnostics / "capture.png").is_file() and (diagnostics / "qi2_result.png").is_file()
    elif camera_source == "camera_inputs":
        captured = worldmirror_batches[-1]
        assert captured["world_to_camera"][:5] == cameras["world_to_camera"]
        capture_pose = np.linalg.inv(captured["world_to_camera"][-1])
        np.testing.assert_allclose(capture_pose[:3, 3], CAMERA["position"])
        look = np.subtract(CAMERA["target"], CAMERA["position"])
        np.testing.assert_allclose(capture_pose[:3, 2], look / np.linalg.norm(look))
        assert captured["intrinsics"][-1][0][2] == 64
        assert captured["intrinsics"][-1][1][2] == 32
        focal_y = captured["intrinsics"][-1][1][1]
        assert np.rad2deg(2 * np.arctan(32 / focal_y)) == pytest.approx(CAMERA["fov"])
    else:
        assert worldmirror_batches == [None, None]
    workspace.generated.assert_called_once()
    assert workspace.generated.call_args.args[3].getpixel((0, 0)) == (255, 255, 0)
    assert library.load_background(background_id) == document
    assert service.public_model(background_id, model_id)["expansion_id"] == expansion_id
    assert not (base / ".expansion-frame").exists()
    info = {"background_id": background_id, "model_id": model_id, "expansion_id": expansion_id}
    response = call(routes, "GET", url, Request(info=info))
    assert response.status == 200
    preview = json.loads(response.text)
    assert f"/expansion/{expansion_id}/objects/" in next(iter(preview["objects"][0]["urls"].values()))
    response = call(routes, "POST", url, Request({"save": save}, info=info))
    assert response.status == 200, response.text
    updated = json.loads(response.text)
    updated_model = updated["models"][0]["model_id"]
    assert (updated_model != model_id) == save
    assert not (base / ".expansion").exists()
    assert len(list((base / "reconstruction").glob("*.png"))) == (6 if save else 5)
    assert all((base / "reconstruction" / name).read_bytes() == data for name, data in original.items())
    manifest = json.loads((base / "reconstruction" / "settings.json").read_text())
    assert manifest["camera_source"] == camera_source
    if provider_name == "moge":
        expected_scales = [1.] * 5 + ([1. if camera_source == "colmap" else 2.] if save else [])
        assert manifest["cameras"]["depth_scales"] == expected_scales
        if save:
            assert manifest["cameras"]["depth_offsets"] == [0.] * 5 + [0. if camera_source == "colmap" else .1]


@pytest.mark.parametrize("name", [None, "", "   "])
def test_cg_creation_requires_an_explicit_scene_name(workspace, name):
    document = with_model(workspace)
    payload = job_payload(document, name=name)
    if name is None:
        payload.pop("name")
    with pytest.raises(ValueError, match="name"):
        service.start_job(document["background_id"], payload, png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("payload", [None, [], {"action": "background"}, {"action": "wizard"},
    {"action": "upload"}, {"action": "create"}, {"action": "export"}, {"action": "unknown"}])
def test_only_model_and_cg_actions_are_accepted(workspace, payload):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(workspace.document["background_id"]) == workspace.document


@pytest.mark.parametrize("revision", [None, True, 0, 999])
def test_job_uses_current_disk_revision_instead_of_saved_widget_metadata(workspace, revision):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document, revision=revision), png())
    newer = copy.deepcopy(document)
    newer["revision"] += 1
    newer["editor"]["saved_after_queue"] = "Keep the authored edit"
    write_document(newer)
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    saved = library.load_background(newer["background_id"])
    assert saved["revision"] == newer["revision"] + 1
    assert saved["editor"] == newer["editor"] and len(saved["cgs"]) == 1


@pytest.mark.parametrize("condition", ["archived", "recovered"])
def test_read_only_scene_fails_before_inference(workspace, condition):
    document = workspace.document
    if condition == "archived":
        write_document({**document, "archived": True})
    else:
        (library.root() / document["name"] / f"{document['name']}_config.json").write_bytes(b"{damaged")
    with pytest.raises(ValueError, match="read-only"):
        service.start_job(document["background_id"], job_payload(document, "model"))
    workspace.worker.submit.assert_not_called()


def test_main_document_without_model_or_cg_arrays_is_read_without_rewriting(workspace):
    document = copy.deepcopy(workspace.document)
    document.pop("models")
    document.pop("cgs")
    path = write_document(document)
    original = path.read_bytes()
    public = service.public_document(library.load_background(document["background_id"]))
    assert public["models"] == [] and public["cgs"] == []
    assert "models" not in document and "cgs" not in document
    assert path.read_bytes() == original
    status = service.start_job(document["background_id"], job_payload(document, "model"))
    assert status["status"] == "queued"


def test_public_generated_set_urls_use_local_routes_and_preserve_main_metadata(workspace):
    document = with_generated_set(workspace)
    before = copy.deepcopy(document)
    public = service.public_document(document)
    group = public["generated_sets"][-1]
    base = f"{service.API_BASE}/backgrounds/{document['background_id']}/sets/{group['set_id']}"
    assert [image["url"] for image in group["images"]] == [f"{base}/{index}?v={document['revision']}" for index in range(3)]
    assert group["settings"] == document["generated_sets"][-1]["settings"]
    assert document == before == library.load_background(document["background_id"])
    assert public["images"][0]["url"].endswith("?v=4")


def test_regenerated_variant_revision_updates_the_preview_url_and_loaded_pixels(workspace):
    document = with_generated_set(workspace)
    group = document["generated_sets"][-1]
    previous_url = service.public_document(document)["generated_sets"][-1]["images"][1]["url"]
    entry = group["images"][1]
    entry["revision"] += 1
    document["revision"] += 1
    path = library.root() / document["name"] / entry["file"]
    path.write_bytes(png(color="blue"))
    write_document(document)
    current = service.public_document(library.load_background(document["background_id"]))
    current_url = current["generated_sets"][-1]["images"][1]["url"]
    assert current_url != previous_url and current_url.endswith(f"?v={entry['revision']}")
    with Image.open(library.image_path(document["background_id"], group["source_image_id"], source_variant(document))) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_cg_job_captures_settings_camera_and_frame_without_changing_main_variants(workspace):
    document = with_model(workspace)
    payload = job_payload(document)
    payload["settings"]["generation"]["steps"] = 12
    service.start_job(document["background_id"], payload, png())
    payload["camera"]["position"][0] = 99
    payload["settings"]["generation"]["steps"] = 99
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    settings, _job_id, _event, reference, background = workspace.generated.call_args.args
    assert settings["generation"]["steps"] == 12
    assert reference.size == (128, 64) and reference.getpixel((0, 0)) == (255, 0, 0)
    assert background.size == (128, 64) and background.getpixel((0, 0)) == (255, 0, 0)
    saved = library.load_background(document["background_id"])
    assert_main_fields_preserved(document, saved)
    assert saved["models"] == document["models"]
    cg = saved["cgs"][-1]
    assert "settings" not in cg and "cg_settings" not in saved and "model_settings" not in saved
    assert cg["camera"] == CAMERA and cg["model_id"] == document["models"][-1]["model_id"]
    assert cg["source_image_id"] == document["models"][-1]["source_image_id"]
    assert cg["source_image_id"] == document["selected_image_id"]
    with Image.open(library.cg_image_path(document["background_id"], cg["cg_id"])) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    assert job["result"]["document"]["cgs"][-1]["url"].endswith(f"/{cg['cg_id']}/image")


def test_character_job_captures_selected_costume_sprite_and_emotion(workspace, monkeypatch):
    from _vnccs import utils
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(workspace.tmp_path))
    folder = workspace.tmp_path / "Maya" / "Sprites" / "Casual"
    for emotion in ("Neutral", "Happy"):
        (folder / emotion).mkdir(parents=True)
    (folder / "Neutral" / "sprite_pose_1.png").write_bytes(png(color="white"))
    selected = folder / "Happy" / "sprite_Happy_0001.png"
    Image.new("RGBA", (64, 128), (0, 0, 255, 128)).save(selected)
    document = with_model(workspace)
    prompt = "Keep the red scarf.\nHave her look toward the window."
    character = {"id": "1" * 32, "position": [0, 0, 0], "scale": 1,
        "pose": {"bones": {}, "bonePositions": {}, "modelRotation": [0, 0, 0]},
        "source": {"character": "Maya", "costume": "Casual", "sprite": "sprite_pose_1.png", "emotion": "Happy"},
        "prompt": prompt}
    service.start_job(document["background_id"], job_payload(document, characters=[character]), png())
    character["source"]["emotion"] = "Neutral"
    character["prompt"] = "Later edit"
    selected.unlink()
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    sprite = workspace.generated.call_args.kwargs["character"]
    assert sprite.size == (64, 128) and sprite.getpixel((0, 0)) == (0, 0, 255, 128)
    assert workspace.generated.call_args.kwargs["character_prompt"] == prompt
    saved = library.load_background(document["background_id"])
    assert saved["cgs"][-1]["characters"][0]["source"]["emotion"] == "Happy"
    assert saved["cgs"][-1]["characters"][0]["prompt"] == prompt


def test_cg_rejects_model_from_another_source_variant(workspace):
    document = with_model(workspace)
    with pytest.raises(FileNotFoundError, match="replaced"):
        service.start_job(document["background_id"], job_payload(document, image_id="f" * 32), png())
    workspace.worker.submit.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("requested", ["original", "night"])
def test_cg_rejects_model_from_another_generated_variant_of_the_same_image(workspace, requested):
    document = with_generated_set(workspace)
    morning = source_variant(document)
    document = with_model(workspace, document, source_variant=morning)
    different = source_variant(document, 2) if requested == "night" else None
    with pytest.raises(ValueError, match="another background variant"):
        service.start_job(document["background_id"], job_payload(document, source_variant=different), png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


def test_cg_job_binds_saved_frame_to_the_exact_generated_variant(workspace):
    document = with_generated_set(workspace)
    morning = source_variant(document)
    document = with_model(workspace, document, source_variant=morning)
    payload = job_payload(document)
    service.start_job(document["background_id"], payload, png())
    payload["source_variant"].update(source_variant(document, 2))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    reference, background = workspace.generated.call_args.args[-2:]
    assert background.getpixel((0, 0)) == (255, 255, 0)
    assert reference.getpixel((0, 0)) == (255, 0, 0)
    saved = library.load_background(document["background_id"])
    assert saved["cgs"][-1]["source_variant"] == morning
    assert saved["cgs"][-1]["source_image_id"] == document["images"][0]["image_id"]
    assert_main_fields_preserved(document, saved)


def test_model_job_rejects_a_generated_set_owned_by_another_base_image(workspace):
    document = with_generated_set(workspace)
    with pytest.raises(ValueError, match="another background source"):
        service.start_job(document["background_id"], job_payload(document, "model",
            image_id="f" * 32, source_variant=source_variant(document)))
    workspace.worker.submit.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("variant", [[], {"set_id": "../outside", "index": 1},
    {"set_id": "7" * 32, "index": True}, {"set_id": "7" * 32, "index": 65}])
def test_invalid_variant_metadata_is_rejected_before_queuing(workspace, variant):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model", source_variant=variant))
    workspace.worker.submit.assert_not_called()


def test_cg_rejects_model_owned_by_another_background(workspace):
    document = with_model(workspace)
    other = {**copy.deepcopy(workspace.document), "background_id": "9" * 32, "name": "Other room"}
    write_document(other)
    (library.root() / other["name"] / "source.png").write_bytes(png())
    other = with_model(workspace, other)
    with pytest.raises(FileNotFoundError):
        service.start_job(document["background_id"], job_payload(document, model_id=other["models"][-1]["model_id"]), png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()


@pytest.mark.parametrize("camera,frame,name", [(None, png(), "View"), ({}, png(), "View"),
    (CAMERA, None, "View"), (CAMERA, png((127, 64)), "View"), (CAMERA, png((32, 16)), "View"),
    (CAMERA, png(), "../bad")])
def test_invalid_cg_inputs_never_queue_inference(workspace, camera, frame, name):
    document = with_model(workspace)
    with pytest.raises((ValueError, TypeError)):
        service.start_job(document["background_id"], job_payload(document, camera=camera, name=name), frame)
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()


def test_busy_workspace_rejects_another_job(workspace):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document), png())
    with pytest.raises(RuntimeError, match="busy"):
        service.start_job(document["background_id"], job_payload(document), png())
    assert workspace.worker.submit.call_count == 1


def test_full_active_queue_rejects_work_without_dropping_jobs(workspace):
    service._JOBS.update({str(index): {"background_id": "other", "status": "queued"} for index in range(64)})
    with pytest.raises(RuntimeError, match="queue is full"):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model"))
    assert len(service._JOBS) == 64
    workspace.worker.submit.assert_not_called()


def test_cancelled_queued_cg_job_never_calls_models_or_writes(workspace):
    document = with_model(workspace)
    status = service.start_job(document["background_id"], job_payload(document), png())
    service.cancel_job(status["job_id"])
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    generation.interrupt.assert_called_once_with(status["job_id"])
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


def test_sampling_cancellation_is_translated_to_factory_job_cancellation(workspace):
    document = with_model(workspace)
    workspace.generated.side_effect = generation.GenerationCancelled("CG generation cancelled")
    service.start_job(document["background_id"], job_payload(document), png())
    job = workspace.run_pending()
    assert job["status"] == "cancelled" and job["error"] == "CG generation cancelled"
    assert library.load_background(document["background_id"]) == document


def test_stop_after_sampling_prevents_committing_a_finished_frame(workspace):
    document = with_model(workspace)
    def generate(_settings, _job_id, event, _reference, _background, character=None, *, character_prompt=""):
        event.set()
        return Image.new("RGB", (128, 64)), {}
    workspace.generated.side_effect = generate
    service.start_job(document["background_id"], job_payload(document), png())
    assert workspace.run_pending()["status"] == "cancelled"
    assert library.load_background(document["background_id"]) == document


def test_late_cg_preserves_unrelated_main_settings_saved_while_sampling(workspace):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document), png())
    newer = copy.deepcopy(document)
    newer["revision"] += 1
    newer["editor"]["new_main_edit"] = "Keep the newest main scene"
    write_document(newer)
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    saved = library.load_background(document["background_id"])
    assert saved["editor"] == newer["editor"]
    assert saved["revision"] == newer["revision"] + 1 and len(saved["cgs"]) == 1


def test_late_cg_cannot_publish_after_the_source_pixels_change(workspace):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document), png())
    source = library.root() / document["name"] / document["images"][0]["file"]
    source.write_bytes(png(color="green"))
    metadata = (source.parent / f"{document['name']}_config.json").read_bytes()
    job = workspace.run_pending()
    assert job["status"] == "failed" and "changed" in job["error"]
    assert (source.parent / f"{document['name']}_config.json").read_bytes() == metadata
    assert library.load_background(document["background_id"])["cgs"] == []
    assert not (source.parent / "CG").exists()


@pytest.mark.parametrize(("size", "expected_size", "sky", "precision", "max_splats", "camera_source", "additional_angles"), [
    (2048, 2044, True, "convrot4", 2_000_000, "camera_inputs", False),
    (1022, 1022, False, "convrot8", 5_000_000, "predicted", True),
])
def test_worldmirror_model_job_saves_native_splats_and_cameras(workspace, monkeypatch, size, expected_size, sky, precision, max_splats,
                                                             camera_source, additional_angles):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    import numpy as np

    mirror = load("nodes.world_mirror_v2")
    monkeypatch.setattr(runtime, "_model_operation_lock", lambda: service.threading.RLock(), raising=False)
    batch = torch.zeros(6 if additional_angles else 4, 64, 128, 3)
    model = object()
    loader = mock.Mock(return_value=(model,))
    monkeypatch.setattr(mirror.VNCCS_WorldMirrorV2Loader, "load_model", loader)
    pose = torch.tensor([[0., 0., 1., 1.], [0., 1., 0., 2.], [-1., 0., 0., 3.], [0., 0., 0., 1.]])
    intrinsics = torch.diag(torch.tensor([16., 16., 1.]))
    poses = pose.repeat(len(batch), 1, 1)
    poses[:, 0, 3] += torch.arange(len(batch))
    world_to_camera = torch.linalg.inv(poses).numpy()
    calibration = np.diag([32., 32., 1.])
    refine = mock.Mock(return_value=(batch, 100., {}, (calibration, world_to_camera, None, None)))
    monkeypatch.setattr(service, "_generate_refined_moge_views", refine)
    splats = {"means": torch.tensor([[[2., 2., 3.]]]), "sh": torch.tensor([[[[.2, .3, .4]]]]),
              "opacities": torch.tensor([[.75]]), "scales": torch.tensor([[[.125, .25, .5]]]),
              "quats": torch.tensor([[[1., 0., 0., 0.]]])}
    predicted = mock.Mock(return_value=(
        {"splats": splats}, torch.zeros(len(batch), 32, 64, 3), None, poses[None], intrinsics.repeat(len(batch), 1, 1)[None],
    ))
    monkeypatch.setattr(mirror.VNCCS_WorldMirrorV2Reconstruction, "run_inference", predicted)
    document = with_generated_set(workspace)
    variant = source_variant(document)
    settings = {"provider": "worldmirror", "worldmirror_target_size": size, "worldmirror_apply_sky_mask": sky,
                "worldmirror_precision": precision, "worldmirror_max_splats": max_splats, "worldmirror_resolution_mp": 1.5,
                "worldmirror_camera_source": camera_source}
    if additional_angles:
        settings["worldmirror_additional_angles"] = True
    service.start_job(document["background_id"], job_payload(document, "model", model_settings=settings, source_variant=variant))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert job["scene_id"] == "" and not list(factory._factory_root().iterdir())
    loader.assert_called_once_with(precision=precision)
    (loaded, image), options = predicted.call_args
    assert predicted.call_count == 1 and loaded is model and image is batch
    refine.assert_called_once()
    assert refine.call_args.kwargs["additional_angles"] is additional_angles
    assert refine.call_args.args[1].getpixel((0, 0)) == (255, 255, 0)
    assert refine.call_args.args[2]["resolution_mp"] == 1.5
    assert refine.call_args.args[2]["generation"]["cache_device"] == "off"
    assert refine.call_args.args[-1].name == job["job_id"]
    if camera_source == "camera_inputs":
        torch.testing.assert_close(options.pop("camera_poses"), poses)
        torch.testing.assert_close(options.pop("camera_intrinsics"),
                                   torch.tensor(calibration, dtype=torch.float32).repeat(len(batch), 1, 1))
    assert options == {"target_size": expected_size, "apply_sky_mask": sky, "enable_splat_upsample": False,
                       "splat_camera_source": camera_source, "low_vram_mode": True, "offload_scheme": "model_cpu_offload",
                       "debug_log": False, "splat_max_points": max_splats}
    saved = library.load_background(document["background_id"])
    model_id = saved["models"][0]["model_id"]
    assert saved["models"][0]["settings"] == {"provider": "worldmirror"}
    directory = library._model_directory(library.root() / saved["name"], model_id, saved)
    cameras = json.loads((directory / "reconstruction" / "settings.json").read_text())["cameras"]
    np.testing.assert_allclose(cameras["world_to_camera"], world_to_camera)
    assert cameras["intrinsics"] == [calibration.tolist()] * len(batch)
    scene = service.public_model(document["background_id"], model_id)
    item = scene["objects"][0]
    assert scene["source_variant"] == variant
    assert scene["camera"] == item["source_camera"] == {
        "position": [1, 2, 3], "target": [2, 2, 3], "up": [0, -1, 0], "fov": 90,
    }
    assert item["transform"]["rotation"] == [180, 90, 0]
    assert library.model_scene(document["background_id"], model_id)["objects"][0]["transform"]["rotation"] == [0, 0, 0]
    assert item["urls"]["splat"].endswith("/splat")
    path = library.model_asset_path(document["background_id"], model_id, item["object_id"], "ply")
    header, vertices = path.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0\nelement vertex 1\n" in header
    np.testing.assert_allclose(np.frombuffer(vertices, "<f4"), [
        2, 2, 3, 0, 0, 0, .2, .3, .4, np.log(3), *np.log([.125, .25, .5]), 1, 0, 0, 0,
    ])


def test_moge_model_job_uses_native_nodes_and_saves_source_camera(workspace, monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np

    monkeypatch.setattr(runtime, "_model_operation_lock", lambda: service.threading.RLock(), raising=False)
    utils = sys.modules["comfy.utils"]
    prompt_server = type("PromptServer", (), {})()
    workflow_progress = lambda *_args, **_kwargs: prompt_server.last_prompt_id
    monkeypatch.setattr(utils, "PROGRESS_BAR_HOOK", workflow_progress, raising=False)
    monkeypatch.setattr(utils, "set_progress_bar_global_hook",
                        lambda hook: setattr(utils, "PROGRESS_BAR_HOOK", hook), raising=False)
    check_interrupt = mock.Mock()
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        types.SimpleNamespace(throw_exception_if_processing_interrupted=check_interrupt))
    checkpoint = "moge_2_vitl_normal_fp16.safetensors"
    monkeypatch.setattr(service, "_installed", lambda category: {
        "geometry_estimation": [checkpoint],
    }.get(category, []))
    batch = torch.zeros(4, 64, 128, 3)
    multiview = load("nodes.cg_moge_multiview")
    focal = 32 / np.tan(np.deg2rad(47 / 2))
    poses = np.repeat(np.eye(4)[None], len(batch), axis=0)
    source_depth = torch.ones(batch.shape[1:3])
    tracks = np.array([[x + 0.5, 0.5, 1.] for x in range(8)])
    holes = np.zeros(batch.shape[1:3], dtype=bool)
    frames = [(batch[:1], holes, source_depth.numpy())] * (len(batch) - 1)
    refine = mock.Mock(return_value=(batch, focal, dict.fromkeys(range(1, len(batch)), holes),
        (np.diag([focal, focal, 1]), poses, source_depth, frames)))
    monkeypatch.setattr(service, "_generate_refined_moge_views", refine)
    monkeypatch.setattr(multiview, "capture_depth_tracks", lambda *_args: tracks)
    model, geometry, mesh = object(), {"depth": torch.ones(batch[1:].shape[:3])}, types.SimpleNamespace(unlit=True)
    merge = mock.Mock(return_value=mesh)
    monkeypatch.setattr(multiview, "fuse_moge_views", merge)
    asset = b"glTF" + b"fixture payload"
    calls = []

    def native(name, **inputs):
        if name == "MoGeInference":
            assert torch.equal(inputs.pop("image"), batch[1:])
            utils.PROGRESS_BAR_HOOK(1, 1, None, node_id=None)
        calls.append((name, inputs))
        if name == "MeshToFile3D":
            assert mesh.unlit
        return {"LoadMoGeModel": (model,), "MoGeInference": (geometry,),
                "MeshToFile3D": (types.SimpleNamespace(get_bytes=lambda: asset),)}[name]

    monkeypatch.setattr(service, "_call_comfy_node", native)
    document = with_generated_set(workspace)
    variant = source_variant(document)
    service.start_job(document["background_id"], job_payload(document, "model",
        model_settings={"provider": "moge"}, source_variant=variant))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert utils.PROGRESS_BAR_HOOK is workflow_progress
    check_interrupt.assert_called_once_with()
    assert job["scene_id"] == "" and not list(factory._factory_root().iterdir())
    assert torch.equal(merge.call_args.args[1], batch)
    assert refine.call_args.args[1].getpixel((0, 0)) == (255, 255, 0)
    assert calls[0] == ("LoadMoGeModel", {"model_name": checkpoint})
    assert calls[1][0] == "MoGeInference" and calls[1][1]["batch_size"] == 3
    assert calls[1][1]["fov_x_degrees"] == pytest.approx(np.rad2deg(2 * np.arctan(64 / focal)))
    assert calls[2] == ("MeshToFile3D", {"mesh": mesh})
    saved = library.load_background(document["background_id"])
    model_id = saved["models"][0]["model_id"]
    assert saved["models"][0]["settings"] == {"provider": "moge"}
    directory = library._model_directory(library.root() / saved["name"], model_id, saved)
    saved_cameras = json.loads((directory / "reconstruction" / "settings.json").read_text())["cameras"]
    assert saved_cameras["world_to_camera"] == poses.tolist()
    assert saved_cameras["depth_scales"] == [1.] * len(batch)
    scene = service.public_model(document["background_id"], model_id)
    item = scene["objects"][0]
    assert scene["source_variant"] == variant
    assert item["asset_kind"] == "mesh" and set(item["urls"]) == {"model"}
    assert scene["camera"] == item["source_camera"] == {
        "position": [0, 0, 0], "target": [0, 0, -1], "up": [0, 1, 0], "fov": pytest.approx(47.0),
    }
    assert library.model_asset_path(document["background_id"], model_id, item["object_id"], "model").read_bytes() == asset


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_or_cancelled_reconstruction_removes_its_temporary_scene(workspace, monkeypatch, cancelled):
    document = workspace.document
    created = factory.create_scene("Temporary reconstruction")
    source = factory.resolve_scene_dir(created["scene_id"])
    monkeypatch.setattr(factory, "create_scene", lambda _name: created)
    failure = runtime.JobCancelled("Cancelled") if cancelled else RuntimeError("Reconstruction failed")
    monkeypatch.setattr(service, "_generate_refined_moge_views", mock.Mock(side_effect=failure))
    before = library.load_background(document["background_id"])
    service.start_job(document["background_id"], job_payload(document, "model", model_settings={"provider": "triposplat"}))
    job = workspace.run_pending()
    assert job["status"] == ("cancelled" if cancelled else "failed")
    assert job["scene_id"] == "" and not source.exists()
    assert library.load_background(document["background_id"]) == before


def test_cancel_after_publication_keeps_the_completed_result(workspace, monkeypatch):
    document = with_model(workspace)
    publish = service.public_document
    def acknowledge(saved):
        job = next(iter(service._JOBS.values()))
        assert job["published"]
        service.cancel_job(job["job_id"])
        assert not job["cancel_event"].is_set()
        return publish(saved)
    monkeypatch.setattr(service, "public_document", acknowledge)
    service.start_job(document["background_id"], job_payload(document), png())
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert len(job["result"]["document"]["cgs"]) == 1
    generation.interrupt.assert_not_called()


def test_shared_background_job_guard_is_respected(workspace, monkeypatch):
    monkeypatch.setattr(library.background, "_JOBS", {"main": {
        "background_id": workspace.document["background_id"], "status": "running"}})
    with pytest.raises(RuntimeError, match="busy"):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model"))
    workspace.worker.submit.assert_not_called()


def test_main_inference_slot_is_held_during_refinement_and_released(workspace, monkeypatch):
    document = with_model(workspace)
    slot = mock.Mock()
    slot.acquire.return_value = True
    monkeypatch.setattr(library, "background_owner", lambda: types.SimpleNamespace(inference_lock=slot))
    generated = workspace.generated.return_value
    def infer(*_args, **_kwargs):
        slot.acquire.assert_called_once_with(timeout=0.25)
        slot.release.assert_not_called()
        return generated
    workspace.generated.side_effect = infer
    service.start_job(document["background_id"], job_payload(document), png())
    assert workspace.run_pending()["status"] == "completed"
    slot.release.assert_called_once_with()


def test_waiting_for_main_inference_slot_can_be_cancelled_without_releasing_its_lock(workspace, monkeypatch):
    document = with_model(workspace)
    status = service.start_job(document["background_id"], job_payload(document), png())
    job = service._JOBS[status["job_id"]]
    slot = mock.Mock()
    def acquire(*, timeout):
        job["cancel_event"].set()
        return False
    slot.acquire.side_effect = acquire
    monkeypatch.setattr(library, "background_owner", lambda: types.SimpleNamespace(inference_lock=slot))
    assert workspace.run_pending()["status"] == "cancelled"
    slot.release.assert_not_called()
    workspace.generated.assert_not_called()
    generation.interrupt.assert_not_called()


class Request:
    def __init__(self, payload=None, *, info=None, raw=None, multipart=None, headers=None):
        self.raw = json.dumps({} if payload is None else payload).encode() if raw is None else raw
        self.match_info = info or {}
        self.headers = {"Content-Length": str(len(self.raw)), "X-VNCCS-CSRF": "1", **(headers or {})}
        self.content_length = len(self.raw)
        self.content_type = "multipart/form-data" if multipart is not None else "application/json"
        self.multipart = multipart

    async def read(self):
        return self.raw

    async def post(self):
        return self.multipart


@pytest.fixture
def routes(workspace, monkeypatch):
    monkeypatch.setattr(routes_module, "_REGISTERED", False)
    monkeypatch.setattr(sys.modules["aiohttp"], "web", web)
    table = web.RouteTableDef()
    routes_module.register_routes(table)
    return {(route.method, route.path): route.handler for route in table}


def call(routes, method, suffix, request):
    return asyncio.run(routes[(method, service.API_BASE + suffix)](request))


def test_http_cg_deletion_hides_only_the_target_and_keeps_workflow_assets(routes, workspace):
    document = with_model(workspace); image = Image.new("RGB", (128, 64))
    for name in ("Failed", "Keep"):
        document = library.commit_cg(document["background_id"], name, image, image, CAMERA,
                                     document["models"][0]["model_id"], document["revision"])
    cg = document["cgs"][0]; info = {"background_id": document["background_id"], "cg_id": cg["cg_id"]}
    response = call(routes, "DELETE", "/backgrounds/{background_id}/cg/{cg_id}", Request({"revision": document["revision"]}, info=info))
    assert response.status == 200
    public = json.loads(response.text)
    assert [item["cg_id"] for item in public["cgs"]] == [document["cgs"][1]["cg_id"]]
    assert public["revision"] == document["revision"] + 1
    assert_main_fields_preserved(service.public_document(document), public)
    assert public["models"] == document["models"]
    stored = library.load_background(document["background_id"])
    assert_main_fields_preserved(document, stored)
    assert stored["cgs"][0] == {**cg, "archived": True}
    assert library.cg_image_path(document["background_id"], cg["cg_id"]).is_file()
    workspace.worker.submit.assert_not_called()


@pytest.mark.parametrize("condition,status", [("missing", 404), ("busy", 409), ("read-only", 400)])
def test_http_cg_deletion_rejects_unsafe_or_stale_requests(routes, workspace, condition, status):
    document = with_model(workspace); image = Image.new("RGB", (128, 64))
    document = library.commit_cg(document["background_id"], "Keep", image, image, CAMERA,
                                 document["models"][0]["model_id"], document["revision"])
    if condition == "read-only":
        document["archived"] = True; write_document(document)
    if condition == "busy":
        service._JOBS["busy"] = {"background_id": document["background_id"], "status": "running"}
    info = {"background_id": document["background_id"], "cg_id": "0" * 32 if condition == "missing" else document["cgs"][0]["cg_id"]}
    response = call(routes, "DELETE", "/backgrounds/{background_id}/cg/{cg_id}",
                    Request({"revision": document["revision"] - (condition == "stale")}, info=info))
    assert response.status == status
    assert library.load_background(document["background_id"]) == document


def test_http_inventory_has_no_background_authoring_actions(routes):
    expected = {("GET", "/config"), ("GET", "/backgrounds"), ("GET", "/backgrounds/{background_id}"),
        ("GET", "/characters"), ("GET", "/characters/{character}"), ("GET", "/characters/{character}/sprite"),
        ("POST", "/backgrounds/{background_id}/jobs"),
        ("GET", "/backgrounds/{background_id}/jobs/{request_id}"),
        ("GET", "/jobs/{job_id}"), ("POST", "/jobs/{job_id}/cancel"),
        ("DELETE", "/backgrounds/{background_id}/cg/{cg_id}"),
        ("GET", "/backgrounds/{background_id}/images/{image_id}"),
        ("GET", "/backgrounds/{background_id}/sets/{set_id}/{index}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}/objects/{object_id}/{kind}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}"),
        ("POST", "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}/objects/{object_id}/{kind}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}/frame/{kind}"),
        ("DELETE", "/backgrounds/{background_id}/models/{model_id}/expansion/{expansion_id}/frame"),
        ("GET", "/backgrounds/{background_id}/cg/{cg_id}/{kind}")}
    assert set(routes) == {(method, service.API_BASE + path) for method, path in expected}


@pytest.mark.parametrize("request_id", ["9" * 32, "plain-http-fallback_request-123"])
def test_generation_can_reconnect_before_the_submission_response_arrives(routes, workspace, request_id):
    document = with_model(workspace)
    payload = job_payload(document, request_id=request_id)
    submitted = service.start_job(document["background_id"], payload, png())
    info = {"background_id": document["background_id"], "request_id": request_id}
    response = call(routes, "GET", "/backgrounds/{background_id}/jobs/{request_id}", Request(info=info))
    assert response.status == 200 and json.loads(response.text)["job_id"] == submitted["job_id"]
    workspace.worker.submit.assert_called_once()
    for changes in ({"background_id": "0" * 32}, {"request_id": "0" * 32}):
        missing = call(routes, "GET", "/backgrounds/{background_id}/jobs/{request_id}", Request(info={**info, **changes}))
        assert missing.status == 404
    workspace.run_pending()
    completed = call(routes, "GET", "/backgrounds/{background_id}/jobs/{request_id}", Request(info=info))
    result = json.loads(completed.text)
    assert result["status"] == "completed", result["error"]


@pytest.mark.parametrize("request_id", [None, "", "../job", [], 7])
def test_generation_request_reference_is_validated_before_work_is_started(workspace, request_id):
    with pytest.raises(ValueError, match="request reference"):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model", request_id=request_id))
    workspace.worker.submit.assert_not_called()


def test_http_config_and_library_read_without_settings_writes(routes, workspace, monkeypatch):
    monkeypatch.setattr(generation, "configuration", lambda: {"settings": generation.DEFAULT_SETTINGS, "model": "QI2"})
    response = call(routes, "GET", "/config", Request())
    assert response.status == 200 and json.loads(response.text)["model"] == "QI2"
    document = {**workspace.document, "cg_settings": {"generation": {"steps": 99}}, "model_settings": {"steps": 12}}
    path = write_document(document)
    original = path.read_bytes()
    response = call(routes, "GET", "/backgrounds", Request())
    public = json.loads(response.text)["backgrounds"][0]
    assert public["images"][0]["url"].endswith(f"/{document['images'][0]['image_id']}?v=4")
    assert "cg_settings" not in public and "model_settings" not in public
    info = {"background_id": document["background_id"]}
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
    assert json.loads(response.text)["revision"] == document["revision"]
    assert ("PATCH", service.API_BASE + "/backgrounds/{background_id}") not in routes
    assert path.read_bytes() == original


def test_http_generated_set_image_uses_the_local_saved_set_resolver(routes, workspace):
    document = with_generated_set(workspace)
    info = {"background_id": document["background_id"], "set_id": document["generated_sets"][-1]["set_id"], "index": "1"}
    with mock.patch.object(library, "background_set_image_path", wraps=library.background_set_image_path) as resolver:
        response = call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info))
    resolver.assert_called_once_with(info["background_id"], info["set_id"], 1)
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"] == "no-store"
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("action", ["background", "wizard", "upload", "create", "delete"])
def test_http_jobs_cannot_author_backgrounds(routes, workspace, action):
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs",
        Request(job_payload(workspace.document, action), info={"background_id": workspace.document["background_id"]}))
    assert response.status == 400
    workspace.worker.submit.assert_not_called()


@pytest.mark.parametrize("raw", [b"[]", b"null", b"false", b"{invalid"])
def test_http_invalid_json_is_rejected_before_work(routes, workspace, raw):
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs",
        Request(raw=raw, info={"background_id": workspace.document["background_id"]}))
    assert response.status == 400 and json.loads(response.text)["error"]
    workspace.worker.submit.assert_not_called()
    assert library.load_background(workspace.document["background_id"]) == workspace.document


def test_http_request_size_limits_apply_before_work(routes, workspace):
    info = {"background_id": workspace.document["background_id"]}
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        call(routes, "POST", "/backgrounds/{background_id}/jobs",
            Request(info=info, headers={"Content-Length": str(service.MAX_REQUEST_BYTES + 1)}))
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs", Request(info=info, raw=b" " * (256 * 1024 + 1)))
    assert response.status == 400
    workspace.worker.submit.assert_not_called()


def test_http_capture_job_status_and_cancel_use_cg_only_actions(routes, workspace):
    document = with_model(workspace)
    form = {"data": json.dumps(job_payload(document)), "image": types.SimpleNamespace(file=io.BytesIO(png()))}
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs",
        Request(info={"background_id": document["background_id"]}, multipart=form))
    assert response.status == 200
    status = json.loads(response.text)
    job_info = {"job_id": status["job_id"]}
    response = call(routes, "GET", "/jobs/{job_id}", Request(info=job_info))
    assert json.loads(response.text)["status"] == "queued"
    response = call(routes, "POST", "/jobs/{job_id}/cancel", Request(info=job_info))
    assert response.status == 200 and service._JOBS[status["job_id"]]["cancel_event"].is_set()
    assert workspace.run_pending()["status"] == "cancelled"
    response = call(routes, "GET", "/jobs/{job_id}", Request(info={"job_id": "missing"}))
    assert response.status == 404


def test_http_owned_assets_are_immutable_and_do_not_expose_model_paths(routes, workspace):
    document = with_model(workspace)
    model = document["models"][-1]
    raw = Image.new("RGB", (128, 64), "red")
    document = library.commit_cg(document["background_id"], "Window", raw, raw, CAMERA, model["model_id"], document["revision"])
    info = {"background_id": document["background_id"], "image_id": document["images"][0]["image_id"],
            "model_id": model["model_id"], "object_id": "d" * 32, "kind": "ply", "cg_id": document["cgs"][-1]["cg_id"]}
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}", Request(info=info))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"] == "no-store"
    response = call(routes, "GET", "/backgrounds/{background_id}/models/{model_id}", Request(info=info))
    item = json.loads(response.text)["objects"][0]
    assert item["locked"] is True and "files" not in item
    assert item["urls"]["splat"].startswith(service.API_BASE)
    response = call(routes, "GET", "/backgrounds/{background_id}/models/{model_id}/objects/{object_id}/{kind}", Request(info=info))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"].endswith("immutable")
    response = call(routes, "GET", "/backgrounds/{background_id}/cg/{cg_id}/{kind}", Request(info={**info, "kind": "image"}))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"].endswith("immutable")
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}", Request(info={**info, "image_id": "../outside"}))
    assert response.status == 404


def test_http_cross_origin_mutations_fail_before_parsing_or_work(routes, workspace):
    request = Request({"revision": workspace.document["revision"]}, info={"background_id": workspace.document["background_id"]},
                      headers={"Sec-Fetch-Site": "cross-site"})
    request.read = mock.AsyncMock(side_effect=AssertionError("Unsafe body must not be parsed"))
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs", request)
    assert response.status == 403
    request.read.assert_not_called()
    workspace.worker.submit.assert_not_called()
