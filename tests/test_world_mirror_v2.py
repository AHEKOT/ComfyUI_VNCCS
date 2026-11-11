import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

import comfy
import folder_paths
import huggingface_hub
from nodes import world_mirror_v2 as mirror
from nodes._world_mirror_v2_model import _GaussianParameters, _WorldMirror


def test_streamed_dense_heads_keep_frame_order_and_gaussian_parameters(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None))
    class Depth:
        projects = [None]

        def __call__(self, tokens, *, images, patch_start_idx):
            depth = tokens[0][..., :1].reshape(1, images.shape[1], 1, 1, 1).expand(1, -1, 2, 2, 1)
            return depth, depth[..., 0], depth[..., 0]

    class GaussianDepth(Depth):
        def __call__(self, tokens, *, images, patch_start_idx):
            depth, conf, mask = super().__call__(tokens, images=images, patch_start_idx=patch_start_idx)
            return images[:, :, :2], depth, conf, mask

    renderer = _GaussianParameters(feature_dim=4).eval()
    model = SimpleNamespace(depth_head=Depth(), gs_head=GaussianDepth(), gs_renderer=renderer, enable_gs=True)
    images = torch.rand(1, 3, 3, 2, 2)
    tokens = [torch.tensor([2., 4., 6.]).reshape(1, 3, 1, 1)]
    cameras = {"camera_poses": torch.eye(4)[None, None].repeat(1, 3, 1, 1),
               "camera_intrs": torch.eye(3)[None, None].repeat(1, 3, 1, 1)}
    features, depth, _, _ = model.gs_head(tokens, images=images, patch_start_idx=0)
    with torch.inference_mode():
        expected = renderer(features, images, {**cameras, "gs_depth": depth})
        streamed = _WorldMirror._stream_dense_heads(model, tokens, images, 0, cameras)
    torch.testing.assert_close(streamed["gs_depth"], depth)
    for key in expected:
        torch.testing.assert_close(streamed["splats"][key], expected[key])


def test_model_files_download_only_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    directory = tmp_path / "WorldMirror-V2"
    model_dir = directory / "HY-WorldMirror-2.0"
    model_dir.mkdir(parents=True)
    config = model_dir / "config.json"
    config.write_text("{}")
    weights = model_dir / "model.safetensors"
    calls = []

    def download(*, repo_id, revision, filename, local_dir, token):
        assert repo_id == "tencent/HY-World-2.0"
        assert len(revision) == 40 and token is False
        path = Path(local_dir) / filename
        assert path == weights
        path.touch()
        calls.append(path)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert mirror._model_files() == [config, weights]
    assert mirror._model_files() == [config, weights]
    assert calls == [weights]


def test_reconstruction_keeps_batch_and_native_outputs(monkeypatch):
    monkeypatch.setattr(comfy, "model_management", SimpleNamespace(
        load_models_gpu=lambda models, **options: None,
        throw_exception_if_processing_interrupted=lambda: None,
    ), raising=False)
    loader = mirror.VNCCS_WorldMirrorV2Loader
    reconstruct = mirror.VNCCS_WorldMirrorV2Reconstruction
    assert loader.RETURN_TYPES == reconstruct.INPUT_TYPES()["required"]["model"]
    assert set(loader.INPUT_TYPES()["optional"]["precision"][0]) == {"bf16", "convrot8", "convrot4"}

    class Network(torch.nn.Module):
        enable_gs = enable_pts = enable_norm = True

        def __init__(self):
            super().__init__()
            self.gs_renderer = _GaussianParameters(feature_dim=4)

        def forward(self, views, cond_flags):
            images = views["img"]
            assert cond_flags == [1, 0, 1]
            assert views["camera_poses"].shape == (1, 2, 4, 4)
            depth = torch.ones(1, 2, 14, 28, 1)
            predictions = {
                "depth": depth, "gs_depth": depth, "normals": depth.expand(-1, -1, -1, -1, 3),
                "camera_poses": views["camera_poses"], "camera_intrs": views["camera_intrs"],
            }
            predictions["splats"] = self.gs_renderer(torch.zeros(1, 2, 2, 14, 28), images, predictions)
            return predictions

    network = Network()
    patcher = SimpleNamespace(model=network, load_device=torch.device("cpu"))
    poses = torch.eye(4).repeat(2, 1, 1)
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    outputs = mirror._WorldMirrorV2Inference().run_inference(
        patcher, torch.ones(2, 14, 28, 3), camera_poses=poses, camera_intrinsics=intrinsics,
        adaptive_target_size=True, voxel_prune_splats=False,
    )
    ply, depth, normals, camera_poses, camera_intrinsics, raw, points, splats = outputs
    assert len(outputs) == len(reconstruct.RETURN_TYPES)
    assert ply["splats"]["means"].shape == (1, 2 * 14 * 28, 3)
    assert raw["images"].shape == (1, 2, 3, 14, 28)
    assert depth.shape == normals.shape == points.shape == splats.shape == (2, 14, 28, 3)
    assert torch.equal(camera_poses, poses[None])
    assert torch.equal(camera_intrinsics, intrinsics[None])
    assert network.enable_gs and network.enable_pts and network.enable_norm


def test_compact_reconstruction_defaults_and_optional_upsampling(monkeypatch):
    node = mirror.VNCCS_WorldMirrorV2Reconstruction()
    optional = node.INPUT_TYPES()["optional"]
    assert "head_compute_mode" not in optional
    assert {key: optional[key][1]["default"] for key in (
        "target_size", "low_vram_mode", "apply_sky_mask", "enable_splat_upsample", "splat_camera_source",
    )} == {"target_size": 3430, "low_vram_mode": False, "apply_sky_mask": True,
          "enable_splat_upsample": False, "splat_camera_source": "predicted"}
    images = torch.ones(1, 14, 28, 3)
    depth = torch.ones(1, 1, 14, 28, 1)
    pose = torch.eye(4)[None]
    intrinsics = torch.eye(3)[None]
    native = ({"splats": "native"}, None, None, pose[None], intrinsics[None], {"gs_depth": depth}, None, None)

    def run_native(self, model, images, **kwargs):
        assert kwargs["head_compute_mode"] == "depth+gs"
        assert kwargs["splat_camera_source"] == "predicted"
        return native

    monkeypatch.setattr(mirror._WorldMirrorV2Inference, "run_inference", run_native)
    assert node.run_inference(None, images) is native
    result = node.run_inference(None, images, enable_splat_upsample=True,
        camera_poses=pose, camera_intrinsics=intrinsics,
        splat_upsample_voxel_prune=False, splat_upsample_max_points=40)
    splats = result[0]["splats"]
    assert splats is result[5]["splats"]
    assert all(value[0].shape[0] == 40 for value in splats.values())
