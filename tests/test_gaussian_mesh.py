"""Native PLY, camera alignment and textured mesh contracts; no GPU execution."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_mesh_visibility_stays_on_device_without_disk_reads_or_writes(monkeypatch):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    frame = dict(_frame(.25), name="frame.png")
    reference = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.float32)
    alpha = (reference > 0).astype(np.uint8) * 255
    cache = {frame["name"]: {"mesh_depth": reference, "alpha_uneroded": alpha}}
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    visibility = torch.full(reference.shape, float("inf"))
    visibility[0, 0] = 1
    interrupt = lambda: None
    calls = []

    def render_depth(v, f, matrix, width, height, callback):
        assert v is vertices and f is faces and callback is interrupt
        assert (width, height) == (3, 2)
        torch.testing.assert_close(matrix, torch.from_numpy(frame["K"] @ frame["view"][:3]))
        calls.append(True)
        return visibility

    def forbidden(*_, **__):
        pytest.fail("Mesh visibility must not access disk caches")

    monkeypatch.setattr(kernels, "depth_map", render_depth)
    monkeypatch.setattr(gaussian.np, "load", forbidden)
    monkeypatch.setattr(gaussian, "atomic_output_path", forbidden)
    _, center, depth, loaded_reference, loaded_alpha, width, height = gaussian._mesh_gpu_frame(
        vertices, faces, frame, cache, interrupt)

    assert calls == [True]
    assert depth is visibility and depth.device == vertices.device
    assert (width, height) == (3, 2)
    torch.testing.assert_close(center, torch.tensor([-.25, 0, 0]))
    np.testing.assert_array_equal(loaded_reference.numpy(), reference)
    np.testing.assert_array_equal(loaded_alpha.numpy(), alpha)



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


def test_texture_baking_uses_splat_rgb_and_rejects_masked_background(monkeypatch):
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
        cache[str(i)] = {
            "mesh_rgb": np.full((4, 4, 3), color, dtype=np.uint8),
            "mesh_depth": np.ones((4, 4), dtype=np.float32),
            "alpha_uneroded": np.full((4, 4), 255 if i == 0 else 0, dtype=np.uint8),
        }
    progress = SimpleNamespace(update_absolute=lambda _: None)
    texture = gaussian._mesh_bake_texture(mesh, frames, cache, 32, .01, .01, progress, lambda: None, 0)
    assert texture.shape == (32, 32, 3) and texture.dtype == np.uint8
    assert texture[..., 0].max() == 255  # Splat red RGB overrides coarse blue vertex colors.
    assert texture[..., 1].max() == 0  # The green masked frame never contaminates the texture.


def _native_mesh_stubs(monkeypatch, unwrap=None):
    module = SimpleNamespace(_uv_unwrap=unwrap or (lambda *_: None))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_mesh_postprocess", module)
    monkeypatch.setitem(sys.modules, "comfy_extras.mesh3d.postprocess.qem_decimate",
                        SimpleNamespace(QEMConfig=object, qem_decimate_simplify=lambda *_: None))


def test_native_uv_unwrap_preserves_seams_normals_without_persisting_atlas(tmp_path, monkeypatch):
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
    result, owner = gaussian._mesh_unwrap(mesh, 128, torch.device("cpu"), lambda: None)
    assert calls == [("cpu", "pec", 128, 4, 0.0)]
    np.testing.assert_array_equal(result.vertex.positions.numpy(), positions[[0, 1, 2, 0]])
    np.testing.assert_array_equal(result.vertex.normals.numpy(), normals[[0, 1, 2, 0]])
    np.testing.assert_allclose(result.triangle.texture_uvs.numpy()[0], [[.1, .2], [.8, .2], [.1, .9]])
    fresh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    fresh.compute_vertex_normals()
    fresh.vertex.colors = o3d.core.Tensor(np.ones((len(fresh.vertex.positions), 3), dtype=np.float32))
    gaussian._mesh_unwrap(fresh, 128, torch.device("cpu"), lambda: None)
    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


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


def _uv_owner_cpu(uv, size):
    """Independent float64 pixel-center rasterization; never executes CUDA kernels."""
    y, x = np.mgrid[:size, :size]
    x, y = (x + .5) / size, (y + .5) / size
    owner = np.full((size, size), 0x7FFFFFFF, dtype=np.int32)
    for face, (a, b, c) in enumerate(np.asarray(uv, dtype=np.float64)):
        ab, ac = b - a, c - a
        area = ab[0] * ac[1] - ab[1] * ac[0]
        if area == 0:
            continue
        u = ((x - a[0]) * ac[1] - (y - a[1]) * ac[0]) / area
        v = (ab[0] * (y - a[1]) - ab[1] * (x - a[0])) / area
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        owner[inside & (owner == 0x7FFFFFFF)] = face
    return torch.from_numpy(owner)


def _uv_repair_stubs(monkeypatch):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    checks = []

    def connected(faces, *_):
        shared = np.intersect1d(faces[0].numpy(), faces[1].numpy()).size
        return torch.tensor([0, 0 if shared else 1], dtype=torch.int32)

    def raster(uv, charts, size, interrupt):
        assert not charts.any(), "All charts must participate in the global overlap check"
        overlap = _uv_overlap(*uv.numpy())
        checks.append(overlap)
        return _uv_owner_cpu(uv.numpy(), size), torch.tensor([int(overlap)], dtype=torch.int32)

    _native_mesh_stubs(monkeypatch)
    monkeypatch.setattr(kernels, "connected_charts", connected)
    monkeypatch.setattr(kernels, "uv_map", raster)
    monkeypatch.setattr(kernels, "color_overlaps", lambda *_: torch.tensor([0, 1], dtype=torch.int32))
    return checks


@pytest.mark.parametrize("reverse_winding", [False, True])
def test_reported_uv_intersections_are_repaired_without_dropping_or_moving_geometry(monkeypatch, reverse_winding):
    mapping, faces, uv, positions = _failing_uv_fixture()
    if reverse_winding:
        faces = faces[:, ::-1].copy()
    assert _uv_overlap(*uv[faces])
    checks = _uv_repair_stubs(monkeypatch)
    original = positions[mapping[faces]].copy()
    new_mapping, new_faces, new_uv, charts, owner = gaussian._mesh_validate_uv(
        mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)
    assert checks == [True, False] and len(new_faces) == len(faces)
    assert not _uv_overlap(*new_uv[new_faces])
    np.testing.assert_array_equal(positions[new_mapping[new_faces]], original)
    assert len(new_mapping) == 6  # Only the shared UV seam gets duplicated.
    assert not np.intersect1d(new_faces[0], new_faces[1]).size
    assert owner.shape == (128, 128) and set(charts) == {0, 1}
    assert all(torch.count_nonzero(owner == face) for face in range(len(faces)))


@pytest.mark.parametrize("layout", ["overlap", "collapsed", "repair_error"])
def test_native_uv_unwrap_calls_only_pec_and_never_resegments_or_writes_cache(tmp_path, monkeypatch, layout):
    mapping, faces, uv, positions = _failing_uv_fixture()
    if layout == "collapsed":
        uv = np.zeros_like(uv)
    checks = _uv_repair_stubs(monkeypatch)

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def numpy(self):
            return self.array

        def __len__(self):
            return len(self.array)

    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(core=SimpleNamespace(Tensor=Tensor)))
    segmenters = []

    def unwrap(*args):
        segmenters.append(args[2])
        assert args[2] == "pec", "No slow segmentation path is permitted"
        return mapping, faces, uv

    sys.modules["comfy_extras.nodes_mesh_postprocess"]._uv_unwrap = unwrap

    def forbidden(*_, **__):
        pytest.fail("UV unwrapping must not access disk caches")

    monkeypatch.setattr(gaussian, "atomic_output_path", forbidden)
    monkeypatch.setattr(gaussian.np, "load", forbidden)
    monkeypatch.setattr(gaussian.np, "savez", forbidden)
    mesh = SimpleNamespace(
        vertex=SimpleNamespace(positions=Tensor(positions), normals=Tensor(np.ones_like(positions)),
                               colors=Tensor(np.full_like(positions, .5))),
        triangle=SimpleNamespace(indices=Tensor(faces)))
    if layout == "repair_error":
        def fail(*_):
            raise ValueError("Atlas cannot fit at a positive scale")

        monkeypatch.setattr(gaussian, "_mesh_validate_uv", fail)
        with pytest.raises(ValueError, match="Atlas cannot fit"):
            gaussian._mesh_unwrap(mesh, 128, torch.device("cpu"), lambda: None)
        np.testing.assert_array_equal(mesh.vertex.positions.numpy(), positions)
    else:
        result, owner = gaussian._mesh_unwrap(mesh, 128, torch.device("cpu"), lambda: None)
        np.testing.assert_array_equal(result.vertex.positions.numpy()[result.triangle.indices.numpy()], positions[faces])
        assert checks == ([False, False] if layout == "collapsed" else [True, False])
        assert all(torch.count_nonzero(owner == face) for face in range(len(faces)))
    assert segmenters == ["pec"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("layout,expected", [
    ("point", True), ("line", True), ("near_line", True), ("uniform_tiny", False),
    ("world_sliver", False), ("zero_world_area", False),
])
def test_uv_repair_distinguishes_collapsed_parameterization_from_thin_geometry(layout, expected):
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
    if layout in {"point", "zero_world_area"}:
        uv[:] = .25
    elif layout == "line":
        uv[2] = [.5, 0.]
    elif layout in {"near_line", "world_sliver"}:
        uv[2] = [.5, 1.e-10]
    else:
        uv *= 1.e-12
    if layout == "world_sliver":
        positions[:, :2] = uv
    elif layout == "zero_world_area":
        positions[2] = [.5, 0., 0.]
    assert gaussian._mesh_bad_uv_faces(np.arange(3), np.array([[0, 1, 2]]), uv, positions).tolist() == [expected]


@pytest.mark.parametrize("layout", ["point", "line", "near_line"])
def test_repacking_restores_bad_uvs_without_poisoning_healthy_chart_density(layout):
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
    faces = np.arange(6).reshape(2, 3)
    uv = np.array([[.2, .3], [.2, .3], [.2, .3], [0., 0.], [1.e-12, 0.], [0., 1.e-12]])
    if layout != "point":
        uv[:3] = [[0., 0.], [1., 0.], [.5, 1.e-10 if layout == "near_line" else 0.]]
    original_uv, original_positions = uv.copy(), positions.copy()
    mapping, packed_faces, packed_uv = gaussian._mesh_repack_uv(
        np.arange(6), faces, uv, positions, np.array([0, 1]), 128, lambda: None)
    np.testing.assert_array_equal(positions[mapping[packed_faces]], positions[faces])
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(uv, original_uv)
    assert not gaussian._mesh_bad_uv_faces(mapping, packed_faces, packed_uv, positions).any()
    assert not _uv_overlap(*packed_uv[packed_faces])
    world_triangles, uv_triangles = positions[mapping[packed_faces]], packed_uv[packed_faces]
    world_edges = np.linalg.norm(np.roll(world_triangles, -1, axis=1) - world_triangles, axis=2)
    uv_edges = np.linalg.norm(np.roll(uv_triangles, -1, axis=1) - uv_triangles, axis=2)
    ratio = uv_edges / world_edges
    np.testing.assert_allclose(ratio, np.broadcast_to(ratio[:, :1], ratio.shape), rtol=1.e-5)
    owner = _uv_owner_cpu(uv_triangles, 128)
    assert all(torch.count_nonzero(owner == face) for face in range(len(faces)))


def test_repacking_keeps_a_thin_world_face_representable_without_changing_geometry():
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [.5, 1.e-12, 0.],
                          [2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
    faces = np.arange(6).reshape(2, 3)
    uv = np.array([[.2, .3]] * 3 + [[0., 0.], [1., 0.], [0., 1.]])
    mapping, result_faces, result_uv = gaussian._mesh_repack_uv(
        np.arange(6), faces, uv, positions, np.array([0, 1]), 4097, lambda: None)
    np.testing.assert_array_equal(positions[mapping[result_faces]], positions[faces])
    assert not gaussian._mesh_bad_uv_faces(mapping, result_faces, result_uv, positions).any()
    triangles = result_uv[result_faces].astype(np.float64)
    ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    assert np.all(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) > 0)


def test_repacking_preserves_internal_small_faces_at_real_atlas_precision():
    chart = np.array([[.5, .5], [.5 + 2**-18, .5], [.5 + 2**-19, .5 + 2**-24],
                      [0., 0.], [1., 0.]], dtype=np.float32)
    local_faces = np.array([[0, 1, 2], [1, 0, 3], [1, 3, 4]])
    count = 1024
    uv = np.tile(chart, (count, 1))
    positions = np.column_stack((uv, np.zeros(len(uv))))
    faces = (local_faces[None] + np.arange(count)[:, None, None] * len(chart)).reshape(-1, 3)
    assert not gaussian._mesh_bad_uv_faces(np.arange(len(uv)), faces, uv, positions).any()
    mapping, result_faces, result_uv = gaussian._mesh_repack_uv(
        np.arange(len(uv)), faces, uv, positions, np.repeat(np.arange(count), len(local_faces)),
        4097, lambda: None)
    np.testing.assert_array_equal(positions[mapping[result_faces]], positions[faces])
    assert not gaussian._mesh_bad_uv_faces(mapping, result_faces, result_uv, positions).any()
    flipped = result_uv.copy()
    flipped[:, 1] = 1 - flipped[:, 1]
    assert not gaussian._mesh_bad_uv_faces(mapping, result_faces, flipped, positions).any()
    assert np.isfinite(result_uv).all() and result_uv.min() >= 0 and result_uv.max() <= 1
    # Splitting the UV seams must not remove or reorient even a tiny world triangle.
    assert len(result_faces) == len(faces)


def test_uv_validation_checks_the_fp32_coordinates_used_by_the_baker(monkeypatch):
    mapping, faces, uv, positions = _failing_uv_fixture()
    uv[:3] = [[.125, 0.], [.1875, 0.], [.125, 2**-26]]
    uv[3:] = [[.5, .5], [.75, .75]]
    assert not gaussian._mesh_bad_uv_faces(mapping, faces, uv, positions).any()
    flipped = uv.copy()
    flipped[:, 1] = 1 - flipped[:, 1]
    assert gaussian._mesh_bad_uv_faces(mapping, faces, flipped, positions).any()
    checks = _uv_repair_stubs(monkeypatch)
    result_mapping, result_faces, result_uv, _, owner = gaussian._mesh_validate_uv(
        mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)
    np.testing.assert_array_equal(positions[result_mapping[result_faces]], positions[faces])
    assert len(checks) == 2 and not checks[-1]
    assert torch.count_nonzero(owner != 0x7FFFFFFF)


def test_uv_validation_stops_with_actual_fault_counts_when_repair_cannot_progress(monkeypatch):
    from importlib import import_module

    mapping, faces, uv, positions = _failing_uv_fixture()
    _uv_repair_stubs(monkeypatch)
    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    calls = []

    def broken_raster(uv, charts, size, interrupt):
        calls.append(True)
        return torch.zeros((size, size), dtype=torch.int32), torch.ones(1, dtype=torch.int32)

    monkeypatch.setattr(kernels, "uv_map", broken_raster)
    monkeypatch.setattr(gaussian, "_mesh_repack_uv", lambda mapping, faces, uv, *_: (mapping, faces, uv))
    with pytest.raises(ValueError, match="no progress: overlaps=True, collapsed_faces=0, covered_texels=16384"):
        gaussian._mesh_validate_uv(mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)
    assert len(calls) == 2


def test_uv_validation_continues_only_while_repairs_split_additional_seams(monkeypatch):
    from importlib import import_module

    mapping, faces, uv, positions = _failing_uv_fixture()
    faces = np.tile(faces, (2, 1))
    _uv_repair_stubs(monkeypatch)
    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    checks = []

    def connected(faces, *_):
        return torch.tensor(([0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 2])[len(checks)], dtype=torch.int32)

    def raster(uv, charts, size, interrupt):
        checks.append(True)
        return torch.zeros((size, size), dtype=torch.int32), torch.tensor([len(checks) < 3], dtype=torch.int32)

    # Isolate the progress gate from the numerical repair, covered by the geometry tests above.
    monkeypatch.setattr(kernels, "connected_charts", connected)
    monkeypatch.setattr(kernels, "uv_map", raster)
    monkeypatch.setattr(kernels, "color_overlaps", lambda *_: torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    monkeypatch.setattr(gaussian, "_mesh_repack_uv", lambda mapping, faces, uv, *_: (mapping, faces, uv))
    gaussian._mesh_validate_uv(mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)
    assert len(checks) == 3


def test_empty_uv_atlas_is_not_accepted_as_an_overlap_free_layout(monkeypatch):
    from importlib import import_module

    mapping, faces, uv, positions = _failing_uv_fixture()
    _uv_repair_stubs(monkeypatch)
    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    monkeypatch.setattr(kernels, "uv_map", lambda uv, charts, size, interrupt: (
        torch.full((size, size), 0x7FFFFFFF, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)))
    with pytest.raises(ValueError, match="UV atlas has no covered texels"):
        gaussian._mesh_validate_uv(mapping, faces, uv, positions, 128, torch.device("cpu"), lambda: None)


def test_empty_uv_atlas_fails_before_uploading_mesh_or_processing_source_views(monkeypatch):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    monkeypatch.setattr(kernels, "prepare_mesh_cuda", lambda *_: None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(
        get_torch_device=lambda: torch.device("cuda"), free_memory=lambda *_: None))
    mesh = SimpleNamespace(triangle=SimpleNamespace(indices=np.array([[0, 1, 2]])))
    empty = torch.full((32, 32), 0x7FFFFFFF, dtype=torch.int32)
    monkeypatch.setattr(gaussian, "_mesh_unwrap", lambda *_: (mesh, empty))
    monkeypatch.setattr(gaussian, "_mesh_gpu_frame", lambda *_: pytest.fail("Empty UVs must not process source views"))
    with pytest.raises(RuntimeError, match="UV unwrapping failed before source-view baking"):
        gaussian._mesh_bake_texture(mesh, [{}], {}, 32, .01, .01, None, lambda: None, 0)



def test_conservative_uv_packing_keeps_gutters_and_rejects_impossible_atlas():
    uv = np.array([[0., 0.], [2., 0.], [0., 1.], [0., 0.], [1., 0.], [0., 2.]])
    packed = gaussian._mesh_pack_uv_rectangles(uv, np.array([0, 3, 6]), 128, lambda: None)
    assert np.isfinite(packed).all() and packed.min() >= 0 and packed.max() <= 1
    first, second = packed[:3], packed[3:]
    separation = np.maximum(second.min(0) - first.max(0), first.min(0) - second.max(0))
    assert separation.max() * 128 >= 8  # Four-pixel gutters on each island.
    with pytest.raises(ValueError, match="increase texture_size"):
        gaussian._mesh_pack_uv_rectangles(uv, np.array([0, 3, 6]), 8, lambda: None)


def test_conservative_uv_packing_does_not_shrink_every_chart_to_a_point():
    # 40 point-sized 9x9 boxes fit in 64x64; 40 nonzero 10x10 boxes do not.
    uv = np.tile([[0., 0.], [1., 0.], [0., 1.]], (40, 1))
    with pytest.raises(ValueError, match="increase texture_size"):
        gaussian._mesh_pack_uv_rectangles(uv, np.arange(0, len(uv) + 1, 3), 64, lambda: None)


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


def test_mesh_scene_uses_node_data_without_copying_arrays_or_mutating_input(tmp_path, monkeypatch):
    frames, source = [], {}
    for index in range(2):
        frame = dict(_frame(index * .1), name=str(index),
                     rgba=np.zeros((4, 4, 4), dtype=np.uint8), depth_unit=1.)
        frames.append(frame)
        source[frame["name"]] = {"alpha_uneroded": np.full((4, 4), 255, dtype=np.uint8)}
    monkeypatch.setattr(gaussian, "_scene_data", lambda *_: (None, None, frames))
    result, data = gaussian._mesh_scene((object(), source))
    assert len(result) == 2 and result[0]["size"] == (4, 4)
    for expected, frame in zip(frames, result):
        assert frame["view"] is expected["view"] and frame["K"] is expected["K"]
        assert data[frame["name"]]["alpha_uneroded"] is source[frame["name"]]["alpha_uneroded"]
    data["0"]["mesh_depth"] = np.ones((4, 4))
    assert "mesh_depth" not in source["0"] and "size" not in frames[0]
    assert not list(tmp_path.iterdir())
    assert "scene" in gaussian.VNCCSPLY2MESH.INPUT_TYPES()["required"]


@pytest.mark.parametrize("scene", [None, (), (object(),), "not a scene"])
def test_mesh_rejects_missing_scene_before_imports_downloads_or_file_access(scene):
    with pytest.raises(ValueError, match="matching scene"):
        gaussian._ply_to_mesh("unused.ply", scene, 512, 4096, .5, .01, 100, 0)


@pytest.mark.parametrize("damage", ["view", "intrinsic", "mask_shape", "mask_type", "missing_mask"])
def test_memory_scene_preserves_camera_and_mask_validation(monkeypatch, damage):
    frames = [dict(_frame(), name=str(i), rgba=np.zeros((4, 4, 4), dtype=np.uint8)) for i in range(2)]
    data = {str(i): {"alpha_uneroded": np.full((4, 4), 255, dtype=np.uint8)} for i in range(2)}
    if damage == "view":
        frames[0]["view"][0, 0] = np.nan
    elif damage == "intrinsic":
        frames[0]["K"][0, 0] = 0
    elif damage == "mask_shape":
        data["0"]["alpha_uneroded"] = np.zeros((3, 4), dtype=np.uint8)
    elif damage == "mask_type":
        data["0"]["alpha_uneroded"] = np.zeros((4, 4), dtype=np.float32)
    else:
        data["0"].clear()
    monkeypatch.setattr(gaussian, "_scene_data", lambda *_: (None, None, frames))
    with pytest.raises(ValueError):
        gaussian._mesh_scene((object(), data))



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


def test_uv_intersection_predicate_matches_geometry_without_a_cuda_device(tmp_path):
    import ctypes as C
    import shutil
    import subprocess
    from nodes.helpers.gaussian_mesh_cuda import _SOURCE

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required to execute the shared CUDA geometry predicate")
    predicate = _SOURCE.split('__device__ bool uv_interiors_overlap', 1)[1].split('extern "C" __global__', 1)[0]
    source = '#include <cmath>\nbool uv_interiors_overlap' + predicate
    source += '\nextern "C" bool overlaps(const float* a,const float* b) { return uv_interiors_overlap(a,b); }'
    library_path = tmp_path / "uv_predicate.so"
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(library_path)],
                   input=source, text=True, capture_output=True, check=True, timeout=30)
    library = C.CDLL(str(library_path))
    library.overlaps.argtypes = [C.POINTER(C.c_float)] * 2
    library.overlaps.restype = C.c_bool
    cases = [tuple(_failing_uv_fixture()[2][_failing_uv_fixture()[1]])]
    tiny = np.array([[.1, .1], [.1001, .1], [.1, .1001]], dtype=np.float32)
    cases.extend([(tiny, tiny), (tiny, tiny + .25)])
    generator = np.random.default_rng(75351)
    for _ in range(500):
        a = generator.uniform(.1, .9, 2)
        length, height = 10**generator.uniform(-4, -1), 10**generator.uniform(-6, -3)
        b = a + [length, .3 * length]
        c, d = a + .47 * (b - a) + [0, height], a + .53 * (b - a) - [0, height]
        first = np.array([a, b, c], dtype=np.float32)
        second = np.array([b, a, d], dtype=np.float32)
        cases.extend([(first, second), (first, first)])
    for first, second in cases:
        first, second = np.ascontiguousarray(first), np.ascontiguousarray(second)
        actual = library.overlaps(first.ctypes.data_as(C.POINTER(C.c_float)),
                                  second.ctypes.data_as(C.POINTER(C.c_float)))
        assert actual == _uv_overlap(first, second), (first, second)


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
    output = (object(), object())
    monkeypatch.setattr(gaussian, "_mesh_ply_path", lambda value: value)

    def convert(*args):
        received.extend(args)
        return output

    monkeypatch.setattr(gaussian, "_ply_to_mesh", convert)
    assert gaussian.VNCCSPLY2MESH().convert("native.ply", scene, 1024, 8192, .7, .02, 0, 25000) is output
    assert received == ["native.ply", scene, 1024, 8192, .7, .02, 0, 25000, .07, 10, 3.0]
    assert gaussian.VNCCSPLY2MESH.RETURN_TYPES == ("MESH", "IMAGE")
    assert gaussian.VNCCSPLY2MESH.RETURN_NAMES == ("mesh", "texture")


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
    monkeypatch.setattr(gaussian, "_mesh_scene", lambda *_: ([], {}))
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



def test_stereo_stage_runs_both_directions_and_keeps_rgb_and_depth_in_memory(tmp_path, monkeypatch):
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
                                torch.device("cpu"), "unused", .07, 10, .5, .5,
                                SimpleNamespace(update_absolute=lambda *_: None), lambda: None)
    assert calls == [{"iters": 10, "test_mode": True}] * 4
    depth = cache["0"]["mesh_depth"]
    assert depth.dtype == np.float32 and depth.shape == mask.shape
    assert depth[1, 4] == 0 and depth[2, 4] == pytest.approx(.1)
    np.testing.assert_array_equal(gaussian._mesh_reference_rgb(frames[0], cache)[0, 0], [64, 64, 64])
    assert not list(tmp_path.iterdir())


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


def test_mesh_pipeline_returns_upright_unlit_mesh_without_any_disk_writes(tmp_path, monkeypatch):
    _native_mesh_stubs(monkeypatch)
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        SimpleNamespace(get_torch_device=lambda: SimpleNamespace(type="cuda")))
    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(
        t=SimpleNamespace(geometry=SimpleNamespace(TriangleMesh=SimpleNamespace(from_legacy=lambda mesh: mesh)))))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", SimpleNamespace(
        Types=SimpleNamespace(MESH=lambda vertices, faces, **kwargs: SimpleNamespace(vertices=vertices, faces=faces, **kwargs))))
    # Saving is exclusively the downstream node's responsibility.
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_save_3d", None)
    monkeypatch.setattr(gaussian, "_mesh_stereo_weights", lambda: "existing_model.pth")
    monkeypatch.setattr(gaussian, "_mesh_load_surfels", lambda _: ({"means": np.array([[0., 0., 0.], [1., 1., 1.]])}, 0))
    frames = [dict(_frame(), name=str(i), size=(64, 64)) for i in range(2)]
    data = {str(i): {"mesh_depth": np.ones((64, 64), dtype=np.float32),
                     "mesh_rgb": np.full((64, 64, 3), 128, dtype=np.uint8)} for i in range(2)}
    monkeypatch.setattr(gaussian, "_mesh_scene", lambda *_: (frames, data))
    monkeypatch.setattr(gaussian, "_mesh_stereo_views", lambda *_: None)
    volume = SimpleNamespace(hashmap=lambda: SimpleNamespace(size=lambda: 1))
    monkeypatch.setattr(gaussian, "_mesh_tsdf_volume", lambda *_: volume)
    integrated = []

    def integrate(actual_volume, frame, depth, rgb, voxel_size):
        assert actual_volume is volume
        assert depth is data[frame["name"]]["mesh_depth"] and rgb is data[frame["name"]]["mesh_rgb"]
        integrated.append(frame["name"])
        return 1

    monkeypatch.setattr(gaussian, "_mesh_integrate_frame", integrate)
    mesh = SimpleNamespace(vertices=[0, 1, 2], triangles=[[0, 1, 2]])
    for method in ("remove_degenerate_triangles", "remove_duplicated_triangles",
                   "remove_duplicated_vertices", "remove_unreferenced_vertices", "compute_vertex_normals"):
        setattr(mesh, method, lambda: None)
    monkeypatch.setattr(gaussian, "_mesh_extract_surface", lambda *_: mesh)
    monkeypatch.setattr(gaussian, "_runtime_helpers", lambda *_: (SimpleNamespace(update_absolute=lambda *_: None), lambda: None))
    texture = np.full((4, 4, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(gaussian, "_mesh_bake_texture", lambda *_: texture)
    vertices = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    normals = torch.tensor([[0., -1., 0.]] * 3)
    monkeypatch.setattr(gaussian, "_mesh_uv_vertices", lambda *_: (
        vertices, torch.tensor([[0, 1, 2]]), torch.tensor([[0., 0.], [1., 0.], [0., 1.]]), normals))

    def forbidden(*_, **__):
        pytest.fail("PLY2MESH must not read or write disk caches or save an output file")

    with monkeypatch.context() as io:
        io.setattr(gaussian, "atomic_output_path", forbidden)
        io.setattr(gaussian.Path, "mkdir", forbidden)
        for method in ("load", "save", "savez"):
            io.setattr(gaussian.np, method, forbidden)
        result, image = gaussian._ply_to_mesh("unused.ply", (object(), {}), 512, 4, .5, .01, 0, 0)
    assert integrated == ["0", "1"] and result.unlit
    torch.testing.assert_close(result.vertices[0], vertices * torch.tensor([1., -1., -1.]))
    torch.testing.assert_close(result.normals[0], normals * torch.tensor([1., -1., -1.]))
    assert image.shape == (1, 4, 4, 3) and image is result.texture
    assert not list(tmp_path.iterdir())
