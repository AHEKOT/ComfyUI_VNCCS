"""
WorldMirror V2 ComfyUI nodes — uses HY-World-2.0 (tencent/HY-World-2.0).

Nodes:
  - VNCCS_LoadWorldMirrorV2Model   — download + load V2 model
  - VNCCS_WorldMirrorV2_3D         — V2 inference, PLY_DATA output
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms

# ── expose background-data/hy-world-2.0/ for `import hyworld2.*` ──────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY_WORLD_V2_DIR = os.path.join(PROJECT_ROOT, "background-data", "hy-world-2.0")
if HY_WORLD_V2_DIR not in sys.path:
    sys.path.insert(0, HY_WORLD_V2_DIR)

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from gsplat.rendering import rasterization as _r
    GSPLAT_AVAILABLE = True
    del _r
except ImportError:
    GSPLAT_AVAILABLE = False

# ── V2 utilities ───────────────────────────────────────────────────────────────
try:
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
        compute_filter_mask,           # high-level: pts_mask + gs_mask
        _compute_sky_mask_from_model,  # model-native sky mask (no ONNX needed)
    )
    from hyworld2.worldrecon.hyworldmirror.utils.visual_util import (
        segment_sky,
        download_file_from_url,
    )
    V2_UTILS_AVAILABLE = True
except Exception as _e:
    print(f"⚠️ [VNCCS V2] Could not import V2 utilities: {_e}")
    V2_UTILS_AVAILABLE = False

_PATCH_SIZE = 14


# ── image preprocessing (V2 logic: scale longest side) ────────────────────────
def _resize_to_tensor(pil_img, target_size):
    """Resize image so its longest side == target_size (multiple of 14). Returns CHW tensor."""
    orig_w, orig_h = pil_img.size
    if orig_w >= orig_h:
        new_w = target_size
        new_h = round(orig_h * (new_w / orig_w) / _PATCH_SIZE) * _PATCH_SIZE
    else:
        new_h = target_size
        new_w = round(orig_w * (new_h / orig_h) / _PATCH_SIZE) * _PATCH_SIZE
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return transforms.ToTensor()(pil_img), orig_w, orig_h, new_w, new_h


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_LoadWorldMirrorV2Model
# ─────────────────────────────────────────────────────────────────────────────
class VNCCS_LoadWorldMirrorV2Model:
    """Download and load WorldMirror 2.0 (tencent/HY-World-2.0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp8", "float32"], {
                    "default": "bf16",
                    "tooltip": (
                        "bf16: recommended, ~2× VRAM vs float32. "
                        "fp8: weight-only quantization via torchao, ~2× vs bf16 (requires torchao, Ampere+). "
                        "float32: full precision."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("WORLDMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"

    def load_model(self, device="cuda", precision="bf16"):
        from huggingface_hub import snapshot_download
        from hyworld2.worldrecon.hyworldmirror.models.models.worldmirror import WorldMirror

        # ── resolve local cache path ──────────────────────────────────────────
        models_base = (
            folder_paths.models_dir if FOLDER_PATHS_AVAILABLE
            else os.path.join(PROJECT_ROOT, "models")
        )
        local_dir = os.path.join(models_base, "WorldMirror-V2")
        model_dir = os.path.join(local_dir, "HY-WorldMirror-2.0")
        weights   = os.path.join(model_dir, "model.safetensors")
        config    = os.path.join(model_dir, "config.json")

        # ── download if not cached ────────────────────────────────────────────
        if os.path.exists(weights) and os.path.exists(config):
            print(f"✅ [V2] Cached model: {model_dir}")
        else:
            print(f"⬇️ [V2] Downloading → {model_dir}  (~5 GB)")
            snapshot_download(
                repo_id="tencent/HY-World-2.0",
                allow_patterns=["HY-WorldMirror-2.0/**"],
                local_dir=local_dir,
            )
            print("✅ [V2] Download complete")

        # ── load in float32 ───────────────────────────────────────────────────
        # Always load without enable_bf16 so weights arrive as float32 and
        # .to() is the standard nn.Module version. We apply precision below.
        print(f"🔄 [V2] Loading model (device={device}, precision={precision})")
        model = WorldMirror.from_pretrained(model_dir)
        model = model.to(device)

        # ── bf16 ──────────────────────────────────────────────────────────────
        if precision == "bf16":
            from hyworld2.worldrecon.pipeline import _collect_fp32_critical_modules
            crit = _collect_fp32_critical_modules(model)
            model.to(torch.bfloat16)
            for mod in crit:
                mod.to(torch.float32)

            def _input_cast_hook(module, args):
                if not args:
                    return args
                dtype = next((p.dtype for p in module.parameters(recurse=False)), None)
                if dtype is None:
                    return args
                return tuple(
                    a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
                    for a in args
                )
            for _, module in model.named_modules():
                if not any(True for _ in module.children()):
                    own = list(module.parameters(recurse=False))
                    if own and all(p.dtype == torch.bfloat16 for p in own):
                        module.register_forward_pre_hook(_input_cast_hook)

            model.enable_bf16 = True
            model.to = model._bf16_to

        # ── fp8 weight-only via torchao ───────────────────────────────────────
        elif precision == "fp8":
            try:
                from torchao.quantization import quantize_, float8_weight_only
            except ImportError:
                raise ImportError(
                    "torchao is required for fp8. Install with: pip install torchao"
                )
            # fp8 weight-only: weights stored as e4m3fn, dequantized to bf16 for matmul.
            # Uses bf16 activations in the forward pass.
            from hyworld2.worldrecon.pipeline import _collect_fp32_critical_modules
            crit = _collect_fp32_critical_modules(model)
            model.to(torch.bfloat16)
            for mod in crit:
                mod.to(torch.float32)

            quantize_(model, float8_weight_only())
            print(f"✅ [V2] fp8 weight quantization applied")

            # Still need the bf16 forward path for activations
            def _input_cast_hook(module, args):
                if not args:
                    return args
                dtype = next((p.dtype for p in module.parameters(recurse=False)), None)
                if dtype is None:
                    return args
                return tuple(
                    a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
                    for a in args
                )
            for _, module in model.named_modules():
                if not any(True for _ in module.children()):
                    own = list(module.parameters(recurse=False))
                    if own and all(p.dtype == torch.bfloat16 for p in own):
                        module.register_forward_pre_hook(_input_cast_hook)

            model.enable_bf16 = True
            model.to = model._bf16_to

        model.eval()
        print("✅ [V2] Model ready")

        return ({"model": model, "device": device},)


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_WorldMirrorV2_3D
# ─────────────────────────────────────────────────────────────────────────────
class VNCCS_WorldMirrorV2_3D:
    """WorldMirror V2 — 3D reconstruction from images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("WORLDMIRROR_MODEL",),
                "images":    ("IMAGE",),
                "use_gsplat": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Gaussian Splatting output. Requires gsplat>=1.5.3."
                }),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 952, "min": 252, "max": 1400, "step": 14,
                    "tooltip": "Longest side in pixels. V2 natively supports high resolutions."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {"default": "none"}),
                "confidence_percentile": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Discard bottom N% lowest-confidence points."
                }),
                "apply_sky_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove sky. V2 uses its own depth_mask prediction — no ONNX required."
                }),
                "filter_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove points at depth discontinuities."
                }),
                "edge_normal_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 90.0, "step": 0.5}),
                "edge_depth_threshold":  ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: intrinsics from Equirect360ToViews node."
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: extrinsics from Equirect360ToViews node."
                }),
            }
        }

    RETURN_TYPES  = ("PLY_DATA", "IMAGE",       "IMAGE",       "TENSOR",         "TENSOR",             "VNCCS_SPLAT")
    RETURN_NAMES  = ("ply_data", "depth_maps",  "normal_maps", "camera_poses",   "camera_intrinsics",  "raw_splats")
    FUNCTION      = "run_inference"
    CATEGORY      = "VNCCS/3D"

    def run_inference(
        self,
        model,
        images,
        use_gsplat          = True,
        target_size         = 952,
        offload_scheme      = "none",
        confidence_percentile = 10.0,
        apply_sky_mask      = False,
        filter_edges        = True,
        edge_normal_threshold = 1.0,
        edge_depth_threshold  = 0.03,
        camera_intrinsics   = None,
        camera_poses        = None,
    ):
        target_size    = (target_size // _PATCH_SIZE) * _PATCH_SIZE
        worldmirror    = model["model"]
        exec_dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_dev   = next(worldmirror.parameters()).device

        # ── 1. Preprocess: ComfyUI IMAGE [B,H,W,C] → tensor [1,S,3,H,W] ─────
        B = images.shape[0]
        tensor_list = []

        for i in range(B):
            img_np  = (images[i].cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, target_size)

            # centre-crop height if it exceeds target_size
            if new_h > target_size:
                crop = (new_h - target_size) // 2
                t = t[:, crop:crop + target_size, :]
                if camera_intrinsics is not None:
                    camera_intrinsics = camera_intrinsics.clone()
                    camera_intrinsics[i, 1, 2] -= crop

            # scale intrinsics to match resized resolution
            if camera_intrinsics is not None:
                camera_intrinsics = camera_intrinsics.clone()
                sx, sy = new_w / orig_w, new_h / orig_h
                camera_intrinsics[i, 0, 0] *= sx
                camera_intrinsics[i, 1, 1] *= sy
                camera_intrinsics[i, 0, 2] *= sx
                camera_intrinsics[i, 1, 2] *= sy

            tensor_list.append(t)

        imgs_tensor = torch.stack(tensor_list).unsqueeze(0).to(exec_dev)  # [1,S,3,H,W]

        # ── 2. Build views dict + cond_flags ──────────────────────────────────
        views      = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]  # [pose, depth, intrinsics]

        if camera_poses is not None:
            views["camera_poses"] = camera_poses.unsqueeze(0).to(exec_dev)
            cond_flags[0] = 1
        if camera_intrinsics is not None:
            views["camera_intrs"] = camera_intrinsics.unsqueeze(0).to(exec_dev)
            cond_flags[2] = 1

        # ── 3. Offload ────────────────────────────────────────────────────────
        if offload_scheme == "model_cpu_offload" and exec_dev.type == "cuda":
            try:
                from accelerate import cpu_offload
                cpu_offload(worldmirror, execution_device=exec_dev)
            except Exception as e:
                print(f"⚠️ [V2] model_cpu_offload failed ({e}), moving to GPU.")
                worldmirror.to(exec_dev)
        else:
            if original_dev != exec_dev:
                worldmirror.to(exec_dev)

        # ── 4. Inference ──────────────────────────────────────────────────────
        original_gs = worldmirror.enable_gs
        worldmirror.enable_gs = use_gsplat and GSPLAT_AVAILABLE

        try:
            print(f"🚀 [V2] Inference: {B} images @ {target_size}px, gs={worldmirror.enable_gs}")
            with torch.no_grad():
                predictions = worldmirror(
                    views      = views,
                    cond_flags = cond_flags,
                    is_inference = True,
                )
            print("✅ [V2] Inference complete")
        finally:
            worldmirror.enable_gs = original_gs
            if offload_scheme == "none" and original_dev.type == "cpu":
                worldmirror.to("cpu")
                torch.cuda.empty_cache()

        # ── 5. Sky mask (model-native first, ONNX fallback) ──────────────────
        S, H, W = predictions["depth"].shape[1:4]
        sky_mask_np = None

        if apply_sky_mask and V2_UTILS_AVAILABLE:
            sky_mask_np = _compute_sky_mask_from_model(predictions, H, W, S)
            if sky_mask_np is not None:
                print(f"[V2] Sky mask: model-native ({S} frames)")
            elif ONNX_AVAILABLE:
                sky_model_path = _get_skyseg_path()
                if sky_model_path:
                    try:
                        sess   = onnxruntime.InferenceSession(sky_model_path)
                        frames = []
                        for i in range(S):
                            np_img = (imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            m = segment_sky(np_img, sess)
                            if m.shape[:2] != (H, W) and cv2 is not None:
                                m = cv2.resize(m, (W, H))
                            frames.append(m)
                        sky_mask_np = np.stack(frames) > 0
                        print(f"[V2] Sky mask: ONNX ({S} frames)")
                    except Exception as e:
                        print(f"⚠️ [V2] Sky segmentation failed: {e}")

        # ── 6. Geometric filter mask (V2 native) ─────────────────────────────
        pts_mask = gs_mask = None
        if V2_UTILS_AVAILABLE and "depth" in predictions and "normals" in predictions:
            pts_mask, gs_mask = compute_filter_mask(
                predictions          = predictions,
                imgs                 = imgs_tensor,
                img_paths            = [],            # not needed: sky_mask provided directly
                H                    = H, W = W, S = S,
                apply_confidence_mask= True,
                apply_edge_mask      = filter_edges,
                apply_sky_mask       = apply_sky_mask,
                confidence_percentile= confidence_percentile,
                edge_normal_threshold= edge_normal_threshold,
                edge_depth_threshold = edge_depth_threshold,
                sky_mask             = sky_mask_np,
                use_gs_depth         = "gs_depth" in predictions,
            )

        # ── 7. Filter pts3d ───────────────────────────────────────────────────
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            if pts_mask is not None:
                flat = torch.from_numpy(pts_mask.reshape(-1)).to(pts.device)
                filtered_pts = pts[flat]
            else:
                filtered_pts = pts

        # ── 8. Filter splats with GS-specific mask ────────────────────────────
        splats = predictions.get("splats")
        if splats is not None and gs_mask is not None:
            flat_gs = torch.from_numpy(gs_mask.reshape(-1))
            filtered = {}
            for k, v in splats.items():
                if isinstance(v, list):
                    filtered[k] = [
                        t[flat_gs.to(t.device)]
                        if (isinstance(t, torch.Tensor) and t.shape[0] == flat_gs.shape[0])
                        else t
                        for t in v
                    ]
                else:
                    filtered[k] = v
            splats = filtered

        # ── 9. Assemble PLY_DATA ──────────────────────────────────────────────
        ply_data = {
            "pts3d":          predictions.get("pts3d"),
            "pts3d_filtered": filtered_pts,
            "pts3d_conf":     predictions.get("pts3d_conf"),
            "splats":         splats,
            "images":         imgs_tensor,
            "filter_mask": (
                torch.from_numpy(pts_mask.reshape(-1)).to(exec_dev)
                if pts_mask is not None else None
            ),
            "camera_poses":   predictions.get("camera_poses"),
            "camera_intrs":   predictions.get("camera_intrs"),
        }

        # ── 10. Depth / normals → ComfyUI IMAGE [S,H,W,3] ────────────────────
        depth_t  = predictions.get("depth")
        normal_t = predictions.get("normals")

        if depth_t is not None:
            d = depth_t[0]                                          # [S,H,W,1]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depth_out = d.repeat(1, 1, 1, 3).cpu().float()         # [S,H,W,3]
        else:
            depth_out = torch.zeros(B, target_size, target_size, 3)

        if normal_t is not None:
            normals_out = ((normal_t[0] + 1) / 2).cpu().float()    # [S,H,W,3]
        else:
            normals_out = torch.zeros(B, target_size, target_size, 3)

        # ── 11. Camera outputs + raw_splats for downstream nodes ──────────────
        cam_poses  = predictions.get("camera_poses")
        cam_intrs  = predictions.get("camera_intrs")
        if cam_poses  is not None: cam_poses  = cam_poses.cpu().float()
        if cam_intrs  is not None: cam_intrs  = cam_intrs.cpu().float()

        predictions["images"] = imgs_tensor   # needed by SplatRefiner

        return ply_data, depth_out, normals_out, cam_poses, cam_intrs, predictions


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _get_skyseg_path():
    """Return local path to skyseg.onnx, downloading if absent."""
    base = folder_paths.models_dir if FOLDER_PATHS_AVAILABLE else os.path.join(PROJECT_ROOT, "models")
    path = os.path.join(base, "skyseg.onnx")
    if not os.path.exists(path) and V2_UTILS_AVAILABLE:
        try:
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                path,
            )
        except Exception as e:
            print(f"❌ [V2] skyseg.onnx download failed: {e}")
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldMirrorV2Model": VNCCS_LoadWorldMirrorV2Model,
    "VNCCS_WorldMirrorV2_3D":       VNCCS_WorldMirrorV2_3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldMirrorV2Model": "🌍 Load WorldMirror V2 Model",
    "VNCCS_WorldMirrorV2_3D":       "🌍 WorldMirror V2 3D Reconstruction",
}
