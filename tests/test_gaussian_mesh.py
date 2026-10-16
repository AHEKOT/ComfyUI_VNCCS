"""Native PLY, camera alignment and textured mesh contracts; no GPU execution."""

import sys
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

import nodes.vnccs_gaussians as gaussian


def _frame(tx=0.0):
    view = np.eye(4, dtype=np.float32)
    view[0, 3] = tx
    return {"view": view, "K": np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]], dtype=np.float32)}


def test_mesh_camera_roundtrip_keeps_half_pixel_centers_and_world_transform():
    frame = _frame(0.25)
    x, y = np.array([0, 1, 3]), np.array([0, 2, 3])
    depth = np.array([1, 2, 4], dtype=np.float32)
    points = gaussian._mesh_unproject(x, y, depth, frame)
    pixels, projected_depth = gaussian._mesh_project(points, frame)
    np.testing.assert_allclose(pixels, np.column_stack((x, y)), atol=1e-6)
    np.testing.assert_allclose(projected_depth, depth)
    np.testing.assert_allclose(points[0], [-0.625, -0.375, 1])




def test_tensor_tsdf_preserves_float_depth_units_camera_pose_and_rgb():
    o3d = pytest.importorskip("open3d")
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3d.core.float32,) * 3,
        attr_channels=((1,), (1,), (3,)), voxel_size=.01, block_resolution=16,
        block_count=16, device=o3d.core.Device("CPU:0"),
    )
    frame = _frame(.1)
    frame["view"][2, 3] = .3
    frame["K"] = np.array([[40, 0, 24], [0, 40, 24], [0, 0, 1]], dtype=np.float32)
    depth = np.full((48, 48), .7, dtype=np.float32)
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    assert gaussian._mesh_integrate_frame(volume, frame, depth, rgb, .01) > 0
    mesh = volume.extract_triangle_mesh(weight_threshold=.5).to_legacy()
    vertices = np.asarray(mesh.vertices)
    assert len(mesh.triangles) > 0
    np.testing.assert_allclose(vertices[:, 2], .4, atol=.011)
    np.testing.assert_allclose(np.asarray(mesh.vertex_colors), np.tile([1., 0., 0.], (len(vertices), 1)), atol=1e-5)


def test_chunked_tsdf_extraction_preserves_triangles_and_welds_block_seams(monkeypatch):
    o3d = pytest.importorskip("open3d")
    volume = gaussian._mesh_tsdf_volume(.01, block_count=16)
    frame = _frame(.1)
    frame["view"][2, 3] = .3
    frame["K"] = np.array([[40, 0, 24], [0, 40, 24], [0, 0, 1]], dtype=np.float32)
    depth = np.full((48, 48), .7, dtype=np.float32)
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    gaussian._mesh_integrate_frame(volume, frame, depth, rgb, .01)
    reference = volume.extract_triangle_mesh(weight_threshold=.5).to_legacy()
    # Force one owning block per batch so the plane crosses many chunk boundaries.
    monkeypatch.setattr(gaussian, "_MESH_EXTRACT_MAX_BYTES", 27 * 16**3 * 3 * 3 * 3 * 4)
    result = gaussian._mesh_extract_surface(volume, .01, lambda: None)

    def triangles(mesh):
        xyz = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        xyz = np.rint(xyz * 1e6).astype(np.int64)
        order = np.lexsort((xyz[:, :, 2], xyz[:, :, 1], xyz[:, :, 0]), axis=1)
        xyz = np.take_along_axis(xyz, order[:, :, None], axis=1)
        return np.unique(xyz.reshape(-1, 9), axis=0)

    np.testing.assert_array_equal(triangles(result), triangles(reference))
    assert len(result.vertices) == len(np.unique(np.asarray(result.vertices), axis=0))
    np.testing.assert_allclose(np.asarray(result.vertex_colors), np.tile([1., 0., 0.], (len(result.vertices), 1)), atol=1e-5)


def test_tensor_tsdf_skips_empty_depth_without_allocating_or_integrating():
    pytest.importorskip("open3d")
    # No volume methods exist: an empty masked frame must be skipped entirely.
    assert gaussian._mesh_integrate_frame(
        object(), _frame(), np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4, 3), dtype=np.uint8), .01,
    ) == 0


@pytest.fixture
def native_ply(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    fields = ["x", "y", "z", "opacity", "scale_0", "scale_1"]
    fields += [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)]
    fields += [f"f_rest_{i}" for i in range(9)]
    values = np.zeros(2, dtype=[(name, "f4") for name in fields])
    values["rot_0"] = 2
    values["scale_0"], values["scale_1"] = -3, -4
    values["opacity"] = -1
    for i in range(9):
        values[f"f_rest_{i}"] = i
    path = tmp_path / "native_2dgs.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(values, "vertex")], byte_order="<").write(path)
    return path, values


def test_mesh_ply_loader_keeps_log_parameters_and_decodes_channel_major_sh(native_ply):
    splats, degree = gaussian._mesh_load_surfels(str(native_ply[0]))
    assert degree == 1
    np.testing.assert_allclose(splats["scales"], [[-3, -4], [-3, -4]])
    np.testing.assert_allclose(splats["opacities"], [-1, -1])
    np.testing.assert_allclose(splats["quats"], [[1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(splats["shN"][0], [[0, 3, 6], [1, 4, 7], [2, 5, 8]])


@pytest.mark.parametrize("damage", ["scale_2", "nan", "zero_rotation", "incomplete_sh"])
def test_mesh_ply_loader_rejects_wrong_format_and_invalid_parameters(native_ply, damage):
    import plyfile

    path, source = native_ply
    fields = list(source.dtype.names)
    if damage == "scale_2":
        fields.append("scale_2")
    if damage == "incomplete_sh":
        fields.remove("f_rest_8")
    values = np.zeros(len(source), dtype=[(name, "f4") for name in fields])
    for field in source.dtype.names:
        if field in fields:
            values[field] = source[field]
    if damage == "nan":
        values["x"][0] = np.nan
    if damage == "zero_rotation":
        values["rot_0"][0] = 0
    plyfile.PlyData([plyfile.PlyElement.describe(values, "vertex")]).write(path)
    with pytest.raises(ValueError):
        gaussian._mesh_load_surfels(str(path))


def test_mesh_paths_restrict_reads_and_fingerprint_file_changes(tmp_path, monkeypatch):
    folders = sys.modules["folder_paths"]
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(folders, "get_output_directory", lambda: str(root))
    monkeypatch.setattr(folders, "get_input_directory", lambda: str(tmp_path / "input"), raising=False)
    monkeypatch.setattr(folders, "get_temp_directory", lambda: str(tmp_path / "temp"), raising=False)

    def reject_absolute_annotation(value):
        raise ValueError(f"Invalid file path: {value!r}")

    monkeypatch.setattr(folders, "get_annotated_filepath", reject_absolute_annotation, raising=False)
    path = root / "native.ply"
    path.write_bytes(b"first")
    before = gaussian.VNCCSPLY2MESH.IS_CHANGED(str(path))
    path.write_bytes(b"second")
    assert gaussian.VNCCSPLY2MESH.IS_CHANGED(str(path)) != before
    with pytest.raises(ValueError, match="inside ComfyUI"):
        gaussian._mesh_ply_path(str(tmp_path / "private.ply"))
    with pytest.raises(ValueError, match="escapes"):
        gaussian._mesh_ply_path(str(root / ".." / "private.ply"))


@pytest.mark.parametrize("name", ["native.ply", "native.ply [output]"])
def test_mesh_relative_names_still_use_comfy_annotation_resolver(tmp_path, monkeypatch, name):
    folders = sys.modules["folder_paths"]
    root = tmp_path / "output"
    root.mkdir()
    path = root / "native.ply"
    path.write_bytes(b"PLY")
    calls = []

    def resolve(value):
        calls.append(value)
        return str(path)

    monkeypatch.setattr(folders, "get_output_directory", lambda: str(root))
    monkeypatch.setattr(folders, "get_input_directory", lambda: str(tmp_path / "input"), raising=False)
    monkeypatch.setattr(folders, "get_temp_directory", lambda: str(tmp_path / "temp"), raising=False)
    monkeypatch.setattr(folders, "get_annotated_filepath", resolve, raising=False)
    assert gaussian._mesh_ply_path(name) == str(path)
    assert calls == [name]


def test_mesh_path_rejects_a_symlink_outside_managed_directories(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    private = tmp_path / "private.ply"
    private.write_bytes(b"not a managed input")
    link = root / "native.ply"
    try:
        link.symlink_to(private)
    except OSError:
        pytest.skip("This platform does not permit creating symlinks")
    folders = sys.modules["folder_paths"]
    monkeypatch.setattr(folders, "get_output_directory", lambda: str(root))
    monkeypatch.setattr(folders, "get_input_directory", lambda: str(tmp_path / "input"), raising=False)
    monkeypatch.setattr(folders, "get_temp_directory", lambda: str(tmp_path / "temp"), raising=False)
    monkeypatch.setattr(folders, "get_annotated_filepath", lambda value: value, raising=False)
    with pytest.raises(ValueError, match="outside"):
        gaussian._mesh_ply_path(str(link))


def test_uv_seams_split_vertices_but_keep_normals_and_gltf_texture_orientation():
    def tensor(array):
        return SimpleNamespace(numpy=lambda: np.asarray(array))

    mesh = SimpleNamespace(
        vertex=SimpleNamespace(
            positions=tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
            normals=tensor([[0, 0, 1]] * 4),
        ),
        triangle=SimpleNamespace(
            indices=tensor([[0, 1, 2], [1, 3, 2]]),
            texture_uvs=tensor([[[0, 0], [.4, 0], [0, .4]], [[.6, .6], [1, 1], [.6, 1]]]),
        ),
    )
    vertices, faces, uv, normals = gaussian._mesh_uv_vertices(mesh)
    assert len(vertices) == 6
    np.testing.assert_allclose(vertices[faces].numpy(), np.asarray(mesh.vertex.positions.numpy())[mesh.triangle.indices.numpy()])
    expected = mesh.triangle.texture_uvs.numpy().copy()
    expected[..., 1] = 1 - expected[..., 1]
    np.testing.assert_allclose(uv[faces].numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(normals.numpy(), [[0, 0, 1]] * 6)


def test_texture_baking_uses_original_rgb_and_rejects_masked_background(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Native mesh texture kernels require NVIDIA CUDA")
    o3d = pytest.importorskip("open3d")
    management = sys.modules["comfy.model_management"]
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cuda"), raising=False)
    monkeypatch.setattr(management, "free_memory", lambda *_: None, raising=False)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-.5, -.5, 1], [.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1]]),
        o3d.utility.Vector3iVector([[0, 2, 1], [1, 2, 3]]),
    )
    legacy.vertex_colors = o3d.utility.Vector3dVector([[0, 0, 1]] * 4)
    legacy.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    frames, cache = [], {}
    for i, color in enumerate(((255, 0, 0), (0, 255, 0))):
        frame = _frame()
        frame["name"] = str(i)
        frames.append(frame)
        buffer = BytesIO()
        Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
        cache[str(i)] = {"source_png": buffer.getvalue(), "alpha_uneroded": np.full((4, 4), 255 if i == 0 else 0, dtype=np.uint8)}
        np.save(tmp_path / f"filtered_{i}.npy", np.ones((4, 4), dtype=np.float32), allow_pickle=False)
    progress = SimpleNamespace(update_absolute=lambda _: None)
    texture = gaussian._mesh_bake_texture(mesh, frames, cache, tmp_path, 32, .01, .01, progress, lambda: None, 0)
    assert texture.shape == (32, 32, 3) and texture.dtype == np.uint8
    assert texture[..., 0].max() == 255  # Original red RGB overrides coarse blue vertex colors.
    assert texture[..., 1].max() == 0  # The green masked frame never contaminates the texture.


def _native_mesh_stubs(monkeypatch, unwrap=None):
    module = SimpleNamespace(_uv_unwrap=unwrap or (lambda *_: None))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_mesh_postprocess", module)
    monkeypatch.setitem(sys.modules, "comfy_extras.mesh3d.postprocess.qem_decimate",
                        SimpleNamespace(QEMConfig=object, qem_decimate_simplify=lambda *_: None))


def test_native_uv_unwrap_preserves_seams_normals_and_reuses_atlas(tmp_path, monkeypatch):
    o3d = pytest.importorskip("open3d")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    mesh.compute_vertex_normals()
    mesh.vertex.colors = o3d.core.Tensor(np.ones((len(mesh.vertex.positions), 3), dtype=np.float32))
    positions, normals = mesh.vertex.positions.numpy().copy(), mesh.vertex.normals.numpy().copy()
    calls = []

    def unwrap(vertices, faces, segmenter, resolution, padding, weld):
        calls.append((vertices.device.type, segmenter, resolution, padding, weld))
        return np.array([0, 1, 2, 0]), np.array([[0, 1, 2], [3, 2, 1]]), np.array([[.1, .2], [.8, .2], [.1, .9], [.9, .9]])

    _native_mesh_stubs(monkeypatch, unwrap)
    monkeypatch.setattr(gaussian, "_mesh_validate_uv", lambda mapping, faces, uv, *_: (
        mapping, faces, uv, np.zeros(len(faces), dtype=np.int32), torch.zeros((128, 128), dtype=torch.int32)))
    result, owner = gaussian._mesh_unwrap(mesh, 128, torch.device("cpu"), lambda: None, tmp_path)
    assert calls == [("cpu", "pec", 128, 4, 0.0)]
    np.testing.assert_array_equal(result.vertex.positions.numpy(), positions[[0, 1, 2, 0]])
    np.testing.assert_array_equal(result.vertex.normals.numpy(), normals[[0, 1, 2, 0]])
    np.testing.assert_allclose(result.triangle.texture_uvs.numpy()[0], [[.1, .2], [.8, .2], [.1, .9]])
    fresh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    fresh.compute_vertex_normals()
    fresh.vertex.colors = o3d.core.Tensor(np.ones((len(fresh.vertex.positions), 3), dtype=np.float32))
    gaussian._mesh_unwrap(fresh, 128, torch.device("cpu"), lambda: None, tmp_path)
    assert len(calls) == 1


def _uv_overlap(a, b):
    """Independent float64 separating-axis check for fixture triangle interiors."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    for triangle in (a, b):
        for p, q in zip(triangle, np.roll(triangle, -1, axis=0)):
            direction = q - p
            axis = np.array([-direction[1], direction[0]])
            left, right = (a - p) @ axis, (b - p) @ axis
            if min(left.max(), right.max()) - max(left.min(), right.min()) <= 1.e-14:
                return False
    return True


def _failing_uv_fixture():
    # Actual intersecting faces 405698 and 407823 from the reported native atlas.
    uv = np.array([[.02227882109582424, .001735609257593751],
                   [.022739138454198837, .0008943542488850653],
                   [.02398335374891758, .0015520605957135558],
                   [.024714870378375053, .0002812928578350693],
                   [.024071408435702324, .001896207220852375]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int64)
    positions = np.column_stack((uv, np.arange(5))).astype(np.float32)
    return np.arange(5), faces, uv, positions


def _uv_repair_stubs(monkeypatch, bitmap_overlap=False):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    checks, pack_calls = [], []

    def connected(faces, *_):
        shared = np.intersect1d(faces[0].numpy(), faces[1].numpy()).size
        return torch.tensor([0, 0 if shared else 1], dtype=torch.int32)

    def raster(uv, charts, size, interrupt):
        assert not charts.any(), "All charts must participate in the global overlap check"
        overlap = _uv_overlap(*uv.numpy())
        checks.append(overlap)
        return torch.zeros((size, size), dtype=torch.int32), torch.tensor([int(overlap)], dtype=torch.int32)

    def pack(uv, offsets, faces, face_offsets, area_3d, uv_area, **kwargs):
        pack_calls.append((offsets.copy(), faces.copy(), face_offsets.copy()))
        assert np.all(area_3d > 0) and np.all(uv_area > 0)
        kwargs["progress_callback"](2, 4)
        return ()

    def place(uv, offsets, *_):
        if bitmap_overlap:
            return uv.astype(np.float32)  # Exercise the strict conservative-packing path.
        output = uv.copy()
        for chart, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            local = uv[start:stop] - uv[start:stop].min(0)
            output[start:stop] = local * 100 + np.array([chart * .5 + .05, .05])
        return output.astype(np.float32)

    _native_mesh_stubs(monkeypatch)
    sys.modules["comfy_extras.nodes_mesh_postprocess"]._uv_pack = SimpleNamespace(
        pack_bitmap_concat=pack, apply_placements_concat=place)
    monkeypatch.setattr(kernels, "connected_charts", connected)
    monkeypatch.setattr(kernels, "uv_map", raster)
    monkeypatch.setattr(kernels, "color_overlaps", lambda *_: torch.tensor([0, 1], dtype=torch.int32))
    return checks, pack_calls


@pytest.mark.parametrize("bitmap_overlap", [False, True])
def test_reported_uv_intersections_are_repaired_without_dropping_or_moving_geometry(monkeypatch, bitmap_overlap):
    mapping, faces, uv, positions = _failing_uv_fixture()
    assert _uv_overlap(*uv[faces])
    checks, calls = _uv_repair_stubs(monkeypatch, bitmap_overlap)
    original = positions[mapping[faces]].copy()
    new_mapping, new_faces, new_uv, charts, owner = gaussian._mesh_validate_uv(
        mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)
    assert checks == ([True, True, False] if bitmap_overlap else [True, False])
    assert len(calls) == 1 and len(new_faces) == len(faces)
    assert not _uv_overlap(*new_uv[new_faces])
    np.testing.assert_array_equal(positions[new_mapping[new_faces]], original)
    assert len(new_mapping) == 6  # Only the shared UV seam gets duplicated.
    assert not np.intersect1d(new_faces[0], new_faces[1]).size
    assert owner.shape == (128, 128) and set(charts) == {0, 1}


def test_legacy_failed_uv_cache_is_repaired_and_published_only_after_validation(tmp_path, monkeypatch):
    mapping, faces, uv, positions = _failing_uv_fixture()
    old = tmp_path / "atlas_native_v1_128.npz"
    np.savez(old, mapping=mapping, faces=faces, uv=uv)
    original_bytes = old.read_bytes()
    checks, calls = _uv_repair_stubs(monkeypatch)

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def numpy(self):
            return self.array

        def __len__(self):
            return len(self.array)

    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(core=SimpleNamespace(Tensor=Tensor)))

    def forbidden(*_):
        raise AssertionError("A retained native UV cache must not rerun unwrapping")

    sys.modules["comfy_extras.nodes_mesh_postprocess"]._uv_unwrap = forbidden

    def mesh():
        return SimpleNamespace(
            vertex=SimpleNamespace(positions=Tensor(positions), normals=Tensor(np.ones_like(positions)),
                                   colors=Tensor(np.full_like(positions, .5))),
            triangle=SimpleNamespace(indices=Tensor(faces)))

    result, _ = gaussian._mesh_unwrap(mesh(), 128, torch.device("cpu"), lambda: None, tmp_path)
    new = tmp_path / "atlas_native_v2_128.npz"
    assert new.is_file() and old.read_bytes() == original_bytes
    np.testing.assert_array_equal(result.vertex.positions.numpy()[result.triangle.indices.numpy()], positions[faces])
    first_checks = len(checks)
    gaussian._mesh_unwrap(mesh(), 128, torch.device("cpu"), lambda: None, tmp_path)
    assert len(checks) == first_checks + 1 and not checks[-1] and len(calls) == 1


def test_conservative_uv_packing_keeps_gutters_and_rejects_impossible_atlas():
    uv = np.array([[0., 0.], [2., 0.], [0., 1.], [0., 0.], [1., 0.], [0., 2.]])
    packed = gaussian._mesh_pack_uv_rectangles(uv, np.array([0, 3, 6]), 128, lambda: None)
    assert np.isfinite(packed).all() and packed.min() >= 0 and packed.max() <= 1
    first, second = packed[:3], packed[3:]
    separation = np.maximum(second.min(0) - first.max(0), first.min(0) - second.max(0))
    assert separation.max() * 128 >= 8  # Four-pixel gutters on each island.
    with pytest.raises(ValueError, match="increase texture_size"):
        gaussian._mesh_pack_uv_rectangles(uv, np.array([0, 3, 6]), 8, lambda: None)


def test_rectified_stereo_keeps_vertical_projection_and_recovers_metric_depth():
    frame = _frame(.25)
    points = np.array([[.1, .2, 2.], [.5, -.4, 3.]])
    baseline = .14
    other = dict(frame, view=gaussian._mesh_stereo_view(frame, baseline))
    left_pixels, z = gaussian._mesh_project(points, frame)
    right_pixels, _ = gaussian._mesh_project(points, other)
    np.testing.assert_allclose(left_pixels[:, 1], right_pixels[:, 1])
    disparity = left_pixels[:, 0] - right_pixels[:, 0]
    np.testing.assert_allclose(frame["K"][0, 0] * baseline / disparity, z)
    np.testing.assert_array_equal(other["view"][:3, :3], frame["view"][:3, :3])


def test_stereo_depth_rejects_occlusion_nan_negative_and_background():
    left = torch.full((4, 8), 1.)
    right = torch.ones_like(left)
    alpha = torch.ones_like(left)
    right[:, 2] = 4.
    left[0, 4] = float("nan")
    left[1, 4] = -1
    alpha[2, 4] = 0
    result = gaussian._mesh_stereo_depth(left, right, alpha, torch.ones_like(alpha), 10., .1, .5, .5)
    assert not result[:, 0].any()  # No right image correspondence exists.
    assert not result[:, 3].any()  # The right-view disparity does not agree.
    assert result[0, 4] == result[1, 4] == result[2, 4] == 0
    assert result[3, 6] == 1.
    assert torch.isfinite(result).all()


def test_reverse_stereo_flips_before_unpadding_odd_width_without_pixel_drift():
    rgb = torch.arange(65).float()[None, :, None].expand(33, -1, 3)
    calls = []

    def model(first, second, **kwargs):
        calls.append(kwargs)
        return None, -first[:, :1]

    left, right = gaussian._mesh_stereo_disparities(model, rgb, rgb, 10, lambda: None)
    torch.testing.assert_close(left, rgb[..., 0])
    torch.testing.assert_close(right, rgb[..., 0])
    assert calls == [{"iters": 10, "test_mode": True}] * 2


def test_mesh_upright_is_a_proper_y_up_rotation_for_vertices_and_normals():
    vertices = torch.tensor([[1., -2., 3.], [2., 1., 1.], [0., 0., 0.]])
    normals = torch.nn.functional.normalize(vertices + .5, dim=-1)
    result, transformed_normals = gaussian._mesh_upright(vertices, normals, [_frame()])
    torch.testing.assert_close(result, vertices * torch.tensor([1., -1., -1.]))
    torch.testing.assert_close(transformed_normals, normals * torch.tensor([1., -1., -1.]))
    before = torch.linalg.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    after = torch.linalg.cross(result[1] - result[0], result[2] - result[0])
    torch.testing.assert_close(after, before * torch.tensor([1., -1., -1.]))
    # A world coordinate rotation must not change the final orientation.
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]], dtype=np.float32)
    frame = _frame()
    frame["view"][:3, :3] = rotation.T
    spun, spun_normals = gaussian._mesh_upright(vertices @ torch.from_numpy(rotation).T,
                                               normals @ torch.from_numpy(rotation).T, [frame])
    torch.testing.assert_close(spun, result)
    torch.testing.assert_close(spun_normals, transformed_normals)


def test_checkpoint_restores_cameras_masks_and_rgb_without_reconstruction(tmp_path, monkeypatch):
    folders = sys.modules["folder_paths"]
    monkeypatch.setattr(folders, "get_output_directory", lambda: str(tmp_path))
    source = tmp_path / "native.ply"
    source.write_bytes(b"existing trained splat")
    frames, cache = [], {}
    for index in range(2):
        frame = _frame(index * .1)
        frame.update(name=f"{index}.png", rgba=np.zeros((4, 4, 4), dtype=np.uint8), depth_unit=1.)
        frames.append(frame)
        buffer = BytesIO()
        Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
        cache[frame["name"]] = {"source_png": buffer.getvalue(), "alpha_uneroded": np.full((4, 4), 255, dtype=np.uint8)}
    monkeypatch.setattr(gaussian, "_scene_data", lambda *_: (None, None, frames))
    settings = {"version": 1, "mesh_resolution": 512}
    directory, _, _ = gaussian._mesh_checkpoint(str(source), (object(), cache), settings, .01)

    def forbidden(*_):
        raise AssertionError("Restoring an export must not reconstruct cameras")

    monkeypatch.setattr(gaussian, "_scene_data", forbidden)
    restored_dir, restored, restored_cache = gaussian._mesh_checkpoint(str(source), None, settings, .01)
    assert restored_dir == directory
    assert len(restored) == 2
    for expected, frame in zip(frames, restored):
        np.testing.assert_array_equal(frame["view"], expected["view"])
        np.testing.assert_array_equal(frame["K"], expected["K"])
        np.testing.assert_array_equal(restored_cache[frame["name"]]["alpha_uneroded"], cache[frame["name"]]["alpha_uneroded"])
        np.testing.assert_array_equal(gaussian._mesh_source(frame, restored_cache)[0, 0], [255, 0, 0])
    assert "scene" in gaussian.VNCCSPLY2MESH.INPUT_TYPES()["optional"]
    # Changing the algorithm or export settings must retain calibrated inputs, not old geometry/depth.
    migrated_dir, migrated, migrated_cache = gaussian._mesh_checkpoint(
        str(source), None, {"version": 2, "method": "gs2mesh_dlnr_v1"}, .02)
    assert migrated_dir != directory
    assert len(migrated) == len(frames)
    np.testing.assert_array_equal(gaussian._mesh_source(migrated[0], migrated_cache)[0, 0], [255, 0, 0])
    assert not (gaussian.Path(migrated_dir) / "geometry.npz").exists()


def test_chart_vertex_export_preserves_geometry_and_splits_only_uv_seams():
    def tensor(array):
        return SimpleNamespace(numpy=lambda: np.asarray(array))

    mesh = SimpleNamespace(
        vertex=SimpleNamespace(positions=tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                               normals=tensor([[0, 0, 1]] * 4)),
        triangle=SimpleNamespace(indices=tensor([[0, 1, 2], [1, 3, 2]]),
                                 texture_uvs=tensor([[[0, 1], [.4, 1], [0, .6]], [[.6, .4], [1, 0], [.6, 0]]]),
                                 vnccs_chart=tensor([[0], [1]])),
    )
    vertices, faces, uv, normals = gaussian._mesh_uv_vertices(mesh, torch.device("cpu"))
    assert len(vertices) == 6
    np.testing.assert_array_equal(vertices[faces], np.asarray(mesh.vertex.positions.numpy())[mesh.triangle.indices.numpy()])
    expected = mesh.triangle.texture_uvs.numpy().copy()
    expected[..., 1] = 1 - expected[..., 1]
    np.testing.assert_allclose(uv[faces], expected, atol=1e-6)
    np.testing.assert_array_equal(normals, [[0, 0, 1]] * 6)


def test_cuda_mesh_depth_and_connectivity_keep_occlusion_and_partition_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("Native mesh rasterization requires NVIDIA CUDA")
    from nodes.helpers.gaussian_mesh_cuda import connected_charts, depth_map

    vertices = torch.tensor([[-1., -1., 1.], [1., -1., 1.], [0., 1., 1.],
                             [-2., -2., 2.], [2., -2., 2.], [0., 2., 2.]], device="cuda")
    faces = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]], device="cuda", dtype=torch.int32)
    matrix = torch.tensor([[4., 0., 4., 0.], [0., 4., 4., 0.], [0., 0., 1., 0.]], device="cuda")
    depth = depth_map(vertices, faces, matrix, 8, 8, lambda: None)
    assert depth[3, 3].item() == pytest.approx(1.)
    groups = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
    labels = connected_charts(faces, groups, lambda: None)
    assert len(torch.unique(labels)) == 3  # Disconnected layer and different chart stay separate.
    labels = connected_charts(faces, torch.zeros_like(groups), lambda: None)
    assert labels[0] == labels[2] and labels[0] != labels[1]


def test_cuda_uv_overlap_detection_and_attribute_interpolation():
    if not torch.cuda.is_available():
        pytest.skip("Native UV rasterization requires NVIDIA CUDA")
    from nodes.helpers.gaussian_mesh_cuda import launch, uv_map
    import ctypes as C

    uv = torch.tensor([[[.1, .1], [.9, .1], [.1, .9]]] * 2, device="cuda")
    charts = torch.zeros(2, device="cuda", dtype=torch.int32)
    owner, conflicts = uv_map(uv, charts, 32, lambda: None)
    assert conflicts[0].item() == 1
    vertices = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], device="cuda")
    faces = torch.tensor([[0, 1, 2]] * 2, device="cuda", dtype=torch.int32)
    normal = torch.tensor([[0., 0., 1.]] * 3, device="cuda")
    color = torch.tensor([[1., 0., 0.]] * 3, device="cuda")
    positions = torch.zeros((32**2, 3), device="cuda")
    normals, colors = torch.zeros_like(positions), torch.zeros_like(positions)
    launch("uv_attributes", (vertices, faces, normal, color, uv, owner, positions, normals, colors),
           32**2, (C.c_int(32), C.c_int(32)), lambda: None)
    valid = owner.flatten() != 0x7FFFFFFF
    torch.testing.assert_close(positions[valid, 2], torch.ones_like(positions[valid, 2]))
    torch.testing.assert_close(colors[valid], color[:1].expand(int(valid.sum()), -1))


def test_cuda_uv_overlap_detects_subpixel_intersections_but_not_shared_edges():
    if not torch.cuda.is_available():
        pytest.skip("Native UV overlap testing requires NVIDIA CUDA")
    from nodes.helpers.gaussian_mesh_cuda import uv_map

    # Neither tiny triangle covers a texel center; checking rendered pixels is insufficient.
    tiny = torch.tensor([[[.1, .1], [.1001, .1], [.1, .1001]]] * 2, device="cuda")
    charts = torch.zeros(2, device="cuda", dtype=torch.int32)
    owner, conflicts = uv_map(tiny, charts, 32, lambda: None)
    assert torch.all(owner == 0x7FFFFFFF)
    assert conflicts[0].item() == 1
    adjacent = torch.tensor([[[.1, .1], [.9, .1], [.1, .9]],
                             [[.9, .1], [.9, .9], [.1, .9]]], device="cuda")
    _, conflicts = uv_map(adjacent, charts, 32, lambda: None)
    assert conflicts[0].item() == 0
    collapsed = torch.tensor([[[.5, .5], [.5, .5], [.5, .5]]] * 2, device="cuda")
    owner, conflicts = uv_map(collapsed, charts, 32, lambda: None)
    assert torch.all(owner == 0x7FFFFFFF)
    assert conflicts[0].item() == 0  # Coincident points have no overlapping interiors.


def test_cuda_overlap_graph_coloring_does_not_turn_a_chain_into_one_layer_per_face():
    if not torch.cuda.is_available():
        pytest.skip("Native UV graph coloring requires NVIDIA CUDA")
    from nodes.helpers.gaussian_mesh_cuda import _color_graph

    count = 1024
    left = torch.arange(count - 1, device="cuda", dtype=torch.int64)
    right = left + 1
    colors = _color_graph((left << 32) | right, count, lambda: None)
    assert torch.all(colors[left] != colors[right])
    assert int(colors.max()) <= 2  # Greedy coloring needs at most degree + 1, not 1024 slices.
    pairs = [(a << 32) | b for a in range(5) for b in range(a + 1, 5)]
    colors = _color_graph(torch.tensor(pairs, device="cuda", dtype=torch.int64), 5, lambda: None)
    assert len(torch.unique(colors)) == 5


def test_mesh_node_forwards_settings_without_changing_trainer_contract(monkeypatch):
    received = []
    scene = (object(), {})
    output = (object(), "mesh.glb", object())
    monkeypatch.setattr(gaussian, "_mesh_ply_path", lambda value: value)

    def convert(*args):
        received.extend(args)
        return output

    monkeypatch.setattr(gaussian, "_ply_to_mesh", convert)
    assert gaussian.VNCCSPLY2MESH().convert("native.ply", scene, 1024, 8192, .7, .02, 0, 25000) is output
    assert received == ["native.ply", scene, 1024, 8192, .7, .02, 0, 25000, .07, 10, 3.0]
    assert gaussian.VNCCSPLY2MESH.RETURN_TYPES == ("MESH", "STRING", "IMAGE")
    assert gaussian.VNCCSPLY2MESH.RETURN_NAMES == ("mesh", "glb_file", "texture")


@pytest.mark.parametrize("missing", ["open3d", "plyfile", None])
def test_mesh_validation_reports_missing_package_without_loading_it(monkeypatch, missing):
    from importlib import metadata

    requested = []

    def version(package):
        requested.append(package)
        if package == missing:
            raise metadata.PackageNotFoundError(package)
        return "installed"

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setitem(sys.modules, "open3d", None)
    monkeypatch.setitem(sys.modules, "plyfile", None)
    _native_mesh_stubs(monkeypatch)
    result = gaussian.VNCCSPLY2MESH.VALIDATE_INPUTS(texture_size=4096)
    if missing:
        assert f"missing Python package '{missing}'" in result
    else:
        assert result is True
    assert requested[0] == "open3d"


def test_mesh_dependency_error_preserves_the_actual_failed_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace())
    with pytest.raises(RuntimeError, match="PLY2MESH dependency import failed:.*open3d"):
        gaussian._ply_to_mesh("unused.ply", None, 512, 4096, .5, .01, 100, 0)


@pytest.mark.parametrize("settings", [
    {"mesh_resolution": 0}, {"texture_size": 0}, {"alpha_threshold": float("nan")},
    {"depth_tolerance": float("inf")}, {"min_component_faces": -1}, {"target_faces": -1},
    {"stereo_baseline": 0}, {"stereo_baseline": float("nan")}, {"stereo_iterations": 0},
    {"stereo_lr_threshold": float("inf")}, {"stereo_lr_threshold": -1},
])
def test_mesh_settings_fail_before_rendering(settings):
    with pytest.raises(ValueError):
        gaussian.VNCCSPLY2MESH().convert("unused.ply", (object(), {}), **settings)


def test_stereo_weight_loading_is_tensor_only_and_removes_dataparallel_prefix(tmp_path):
    path = tmp_path / "model.pth"
    torch.save({"module.extractor.weight": torch.ones(2, 3)}, path)
    weights = gaussian._mesh_read_stereo_weights(path)
    assert list(weights) == ["extractor.weight"]
    torch.testing.assert_close(weights["extractor.weight"], torch.ones(2, 3))
    torch.save({"not_weights": "arbitrary metadata"}, path)
    with pytest.raises(ValueError, match="tensor state dictionary"):
        gaussian._mesh_read_stereo_weights(path)


def test_partial_model_download_never_publishes_a_broken_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules["folder_paths"], "models_dir", str(tmp_path))

    def download(url, path, progress):
        assert url == gaussian._MESH_STEREO_URL and progress
        gaussian.Path(path).write_bytes(b"partial download")

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(ValueError, match="unexpected size"):
        gaussian._mesh_stereo_weights()
    assert not (tmp_path / "gs2mesh" / gaussian._MESH_STEREO_MODEL).exists()


def test_completed_stereo_views_do_not_load_model_render_or_retrain(tmp_path, monkeypatch):
    frames, cache = [], {}
    for i in range(2):
        frame = dict(_frame(i * .2), name=str(i), size=(4, 4))
        frames.append(frame)
        cache[str(i)] = {"alpha_uneroded": np.full((4, 4), 255, dtype=np.uint8)}
        np.save(tmp_path / f"depth_{i}.npy", np.ones((4, 4), dtype=np.float32))
        Image.new("RGB", (4, 4), "red").save(tmp_path / f"render_{i}.png")

    def forbidden(*_):
        raise AssertionError("Completed stereo data must not load DLNR or render again")

    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(free_memory=forbidden))
    monkeypatch.setattr(gaussian, "_mesh_stereo_model", forbidden)
    monkeypatch.setattr(gaussian, "_mesh_stereo_render", forbidden)
    progress = SimpleNamespace(update_absolute=lambda *_: None)
    gaussian._mesh_stereo_views({}, 1, frames, cache, tmp_path, torch.device("cpu"), "unused",
                                .07, 10, 3., .5, progress, lambda: None)
    np.testing.assert_array_equal(gaussian._mesh_reference_rgb(frames[0], cache)[0, 0], [255, 0, 0])


def test_new_stereo_stage_runs_both_directions_masks_depth_and_caches_rgb(tmp_path, monkeypatch):
    frames = [dict(_frame(i * .2), name=str(i), size=(33, 65), depth_unit=1.) for i in range(2)]
    mask = np.full((33, 65), 255, dtype=np.uint8)
    mask[1, 4] = 0
    cache = {str(i): {"alpha_uneroded": mask} for i in range(2)}
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(free_memory=lambda *_: None))
    calls = []

    def model(first, second, **kwargs):
        calls.append(kwargs)
        return None, torch.full_like(first[:, :1], -.28)

    monkeypatch.setattr(gaussian, "_mesh_stereo_model", lambda *_: model)
    monkeypatch.setattr(gaussian, "prepare_surface_cuda", lambda *_: None)
    monkeypatch.setattr(gaussian, "_mesh_stereo_render", lambda *_: (
        torch.full((33, 65, 3), 64, dtype=torch.uint8), torch.ones(33, 65)))
    gaussian._mesh_stereo_views({"means": np.array([[0., 0., .1]], dtype=np.float32)}, 1, frames, cache,
                                tmp_path, torch.device("cpu"), "unused", .07, 10, .5, .5,
                                SimpleNamespace(update_absolute=lambda *_: None), lambda: None)
    assert calls == [{"iters": 10, "test_mode": True}] * 4
    depth = np.load(tmp_path / "depth_0.npy")
    assert depth.dtype == np.float32 and depth.shape == mask.shape
    assert depth[1, 4] == 0 and depth[2, 4] == pytest.approx(.1)
    np.testing.assert_array_equal(gaussian._mesh_reference_rgb(frames[0], cache)[0, 0], [64, 64, 64])


def test_bundled_dlnr_checkpoint_layers_exist_without_allocating_model_weights():
    from nodes.helpers.gaussian_stereo import DLNR

    args = SimpleNamespace(corr_implementation="reg", corr_levels=4, corr_radius=4,
                           hidden_dims=[128] * 3, mixed_precision=False, n_downsample=2, n_gru_layers=3)
    with torch.device("meta"):
        model = DLNR(args)
    state = model.state_dict()
    assert state["extractor.outputs16.0.weight"].shape == (128, 256, 1, 1)
    assert state["update_block.encoder.convc1.weight"].shape == (64, 36, 1, 1)
    assert state["normalizationRefinement.final_conv.weight"].shape == (1, 32, 3, 3)
    assert all(value.device.type == "meta" for value in state.values())


def test_native_dlnr_reshaping_preserves_channel_and_pixel_order():
    from nodes.helpers.gaussian_stereo import to_3d, to_4d

    source = torch.arange(2 * 12 * 3 * 5).reshape(2, 12, 3, 5).transpose(2, 3)
    flat = to_3d(source)
    expected = source.permute(0, 2, 3, 1).reshape(2, 15, 12)
    torch.testing.assert_close(flat, expected)
    torch.testing.assert_close(to_4d(flat, 5, 3), source)
    heads = source.reshape(2, 3, 4, 15)
    for channel in range(12):
        torch.testing.assert_close(heads[:, channel // 4, channel % 4], source[:, channel].flatten(1))


def test_export_pipeline_returns_and_saves_identical_upright_unlit_mesh(tmp_path, monkeypatch):
    _native_mesh_stubs(monkeypatch)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        SimpleNamespace(get_torch_device=lambda: SimpleNamespace(type="cuda")))
    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(
        t=SimpleNamespace(geometry=SimpleNamespace(TriangleMesh=SimpleNamespace(from_legacy=lambda mesh: mesh)))))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", SimpleNamespace(
        Types=SimpleNamespace(MESH=lambda vertices, faces, **kwargs: SimpleNamespace(vertices=vertices, faces=faces, **kwargs))))
    saved = {}

    def save(vertices, faces, filepath, **kwargs):
        saved.update(vertices=vertices, faces=faces, **kwargs)
        gaussian.Path(filepath).write_bytes(b"test export; no GPU or model execution")

    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_save_3d", SimpleNamespace(save_glb=save))
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"metadata only")
    monkeypatch.setattr(gaussian, "_mesh_stereo_weights", lambda: str(weights))
    monkeypatch.setattr(gaussian, "_mesh_load_surfels", lambda _: ({"means": np.array([[0., 0., 0.], [1., 1., 1.]])}, 0))
    frames = [dict(_frame(), size=(64, 64))] * 2
    monkeypatch.setattr(gaussian, "_mesh_checkpoint", lambda *_: (str(tmp_path), frames, {}))
    monkeypatch.setattr(gaussian, "_mesh_stereo_views", lambda *_: None)
    key = gaussian.hashlib.sha256(gaussian.json.dumps([2, 512, 100, 500000]).encode()).hexdigest()
    directory = tmp_path / key
    directory.mkdir()
    (directory / "geometry.npz").write_bytes(b"checkpoint")
    monkeypatch.setattr(gaussian, "_mesh_load_geometry", lambda *_: object())
    monkeypatch.setattr(gaussian, "_runtime_helpers", lambda *_: (SimpleNamespace(update_absolute=lambda *_: None), lambda: None))
    texture = np.full((4, 4, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(gaussian, "_mesh_bake_texture", lambda *_: texture)
    vertices = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    normals = torch.tensor([[0., -1., 0.]] * 3)
    monkeypatch.setattr(gaussian, "_mesh_uv_vertices", lambda *_: (
        vertices, torch.tensor([[0, 1, 2]]), torch.tensor([[0., 0.], [1., 0.], [0., 1.]]), normals))
    result, output, image = gaussian._ply_to_mesh("unused.ply", None, 512, 4, .5, .01, 100, 500000)
    assert gaussian.Path(output).is_file() and result.unlit and saved["unlit"]
    torch.testing.assert_close(result.vertices[0], vertices * torch.tensor([1., -1., -1.]))
    torch.testing.assert_close(result.vertices[0], saved["vertices"])
    torch.testing.assert_close(result.normals[0], saved["normals"])
    assert image.shape == (1, 4, 4, 3)
