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


def test_depth_consistency_rejects_isolated_surface_without_moving_valid_depth():
    depth = np.ones((4, 4), dtype=np.float32)
    shifted = _frame(-0.25)
    other = np.ones_like(depth)
    other[1, 1] = 2
    result = gaussian._mesh_consistent_depth(depth, _frame(), [(shifted, other)], 0.001, 0.01, lambda: None)
    assert not result[:, 0].any()  # Outside the second camera.
    assert result[1, 2] == 0  # This point disagrees in camera depth.
    assert result[1, 3] == 1
    assert set(np.unique(result)) == {0, 1}


def test_depth_consistency_requires_another_view_and_honors_cancellation():
    depth = np.ones((4, 4), dtype=np.float32)
    assert not gaussian._mesh_consistent_depth(depth, _frame(), [], 0.01, 0.01, lambda: None).any()

    def interrupt():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        gaussian._mesh_consistent_depth(depth, _frame(), [], 0.01, 0.01, interrupt)


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


def test_uv_rectangle_packing_preserves_gutters_and_does_not_overlap():
    extents = np.array([[10., 30.], [40., 5.], [12., 12.], [1., 2.]])
    offsets, rotate, scale = gaussian._mesh_pack_charts(extents, 128)
    assert scale > 0
    spans = extents.copy()
    spans[rotate] = spans[rotate, ::-1]
    lower = offsets - 4.5
    upper = lower + np.ceil(spans * scale) + 9
    assert (lower >= 0).all() and (upper <= 128).all()
    for left in range(len(extents)):
        for right in range(left):
            assert ((upper[left] <= lower[right]) | (upper[right] <= lower[left])).any()


def test_uv_packing_reports_real_texture_capacity_instead_of_dropping_charts():
    with pytest.raises(ValueError, match="increase texture_size"):
        gaussian._mesh_pack_charts(np.ones((2, 2)), 8)
    with pytest.raises(ValueError, match="finite"):
        gaussian._mesh_pack_charts([[float("nan"), 1]], 128)


def test_disconnected_projection_fragments_share_patches_and_only_overlap_splits_them(tmp_path, monkeypatch):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)

    # Every triangle has private vertices, but disjoint UV fragments need no private gutter.
    vertices = torch.tensor([point for y in range(20) for x in range(20)
                             for point in ((x, y, 0.), (x + .6, y, 0.), (x, y + .6, 0.))])
    faces = torch.arange(len(vertices), dtype=torch.int32).reshape(-1, 3)
    frames = [_frame(), _frame()]

    def gpu_frame(*_):
        return (torch.zeros((3, 4)), torch.zeros(3), torch.ones((4, 4)), torch.ones((4, 4)),
                torch.ones((4, 4), dtype=torch.uint8), 4, 4)

    def forbidden(*_):
        raise AssertionError("Disconnected fragments must not allocate separate projection islands")

    patch_counts = []

    def raster(uv, charts, size, interrupt):
        count = int(charts.max()) + 1
        patch_counts.append(count)
        # Exercise the refinement path too: splitting layers must not reintroduce connectivity fragmentation.
        conflicts = torch.ones(count, dtype=torch.int32) if len(patch_counts) == 1 else torch.zeros(count, dtype=torch.int32)
        return torch.zeros((size, size), dtype=torch.int32), conflicts

    monkeypatch.setattr(gaussian, "_mesh_gpu_frame", gpu_frame)
    monkeypatch.setattr(kernels, "launch", lambda *_: None)
    monkeypatch.setattr(kernels, "connected_charts", forbidden)
    monkeypatch.setattr(kernels, "uv_map", raster)
    monkeypatch.setattr(kernels, "color_overlaps", lambda uv, *_: torch.arange(len(uv), dtype=torch.int32) % 2)
    uv, charts, _ = gaussian._mesh_build_uv(vertices, faces, frames, {}, tmp_path, 128, .01, .01, lambda: None)
    assert patch_counts == [64, 128]
    assert uv.shape == (400, 3, 2) and len(charts) == 400
    assert torch.isfinite(uv).all() and (uv >= 0).all() and (uv <= 1).all()


def test_camera_projection_keeps_front_and_back_windings_in_different_patches(tmp_path, monkeypatch):
    from importlib import import_module

    kernels = import_module(".helpers.gaussian_mesh_cuda", gaussian.__package__)
    vertices = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
                             [0., 0., 1.01], [0., 1., 1.01], [1., 0., 1.01]])
    faces = torch.arange(6, dtype=torch.int32).reshape(-1, 3)

    def gpu_frame(*_):
        return (torch.zeros((3, 4)), torch.zeros(3), torch.ones((4, 4)), torch.ones((4, 4)),
                torch.ones((4, 4), dtype=torch.uint8), 4, 4)

    def choose_view(name, tensors, *_):
        tensors[-1].zero_()  # Both triangles use camera zero, not the axis fallback.

    def raster(uv, charts, size, interrupt):
        assert charts[0] != charts[1]
        return torch.zeros((size, size), dtype=torch.int32), torch.zeros(2, dtype=torch.int32)

    monkeypatch.setattr(gaussian, "_mesh_gpu_frame", gpu_frame)
    monkeypatch.setattr(kernels, "launch", choose_view)
    monkeypatch.setattr(kernels, "uv_map", raster)
    gaussian._mesh_build_uv(vertices, faces, [_frame(), _frame()], {}, tmp_path, 64, .01, .01, lambda: None)


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
    assert received == ["native.ply", scene, 1024, 8192, .7, .02, 0, 25000]
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
])
def test_mesh_settings_fail_before_rendering(settings):
    with pytest.raises(ValueError):
        gaussian.VNCCSPLY2MESH().convert("unused.ply", (object(), {}), **settings)
