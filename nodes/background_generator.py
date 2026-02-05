"""
VNCCS 3D Background Generation Nodes

Uses WorldMirror for 3D reconstruction from images.
Includes equirectangular panorama to perspective views conversion.
"""

import os
import sys
import math
import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths


# Ensure worldmirror is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDMIRROR_DIR = os.path.join(PROJECT_ROOT, "background-data", "worldmirror")
if WORLDMIRROR_DIR not in sys.path:
    sys.path.insert(0, WORLDMIRROR_DIR)

# Import FastPLYRenderer after setting up sys.path
# Import FastPLYRenderer after setting up sys.path
try:
    from src.utils.fast_ply_render import FastPLYRenderer
    # Import ported utils for advanced filtering
    from src.utils.visual_util import segment_sky, download_file_from_url
    from src.utils.geometry import depth_edge, normals_edge
except ImportError:
    # If using direct import (e.g. dev environment)
    from background_data.worldmirror.src.utils.fast_ply_render import FastPLYRenderer
    from background_data.worldmirror.src.utils.visual_util import segment_sky, download_file_from_url
    from background_data.worldmirror.src.utils.geometry import depth_edge, normals_edge

try:
    import onnxruntime
    SKYSEG_AVAILABLE = True
except ImportError:
    SKYSEG_AVAILABLE = False
    print("⚠️ [VNCCS] onnxruntime not found. Sky segmentation will be disabled.")


# ----------------------------------------------------------------------------
# GSPLAT DIAGNOSTICS
# ----------------------------------------------------------------------------
try:
    import gsplat
    print(f"✅ [VNCCS] gsplat library detected: Version {gsplat.__version__}")
    GSPLAT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [VNCCS] gsplat library NOT found: {e}")
    GSPLAT_AVAILABLE = False
except Exception as e:
    print(f"❌ [VNCCS] Error loading gsplat: {e}")
    GSPLAT_AVAILABLE = False

if torch.cuda.is_available():
    print(f"✅ [VNCCS] CUDA is available: {torch.version.cuda}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ [VNCCS] CUDA is NOT available. gsplat requires CUDA.")
# ----------------------------------------------------------------------------


# ============================================================================
# Utility Functions
# ============================================================================

def equirect_to_perspective(equirect_img, fov_deg, yaw_deg, pitch_deg, output_size=(512, 512)):
    """
    Extract a perspective view from an equirectangular panorama using PyTorch grid_sample.
    
    Args:
        equirect_img: PIL Image of equirectangular panorama (2:1 aspect ratio)
        fov_deg: Field of view in degrees (horizontal and vertical)
        yaw_deg: Horizontal rotation (0=front, 90=right, 180=back, 270=left)
        pitch_deg: Vertical rotation (positive=up, negative=down)
        output_size: Output image size (width, height)
    
    Returns:
        PIL Image of the perspective view
    """
    # Convert image to tensor [1, C, H, W]
    if isinstance(equirect_img, Image.Image):
        img_np = np.array(equirect_img.convert("RGB"))
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        # Assume tensor input if needed, but for now stick to PIL interop
        img_np = np.array(equirect_img)
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    B, C, H, W = img_tensor.shape
    out_w, out_h = output_size
    
    # Create ray grid
    # We essentially want to verify "rays_z = f" logic in a vectorized way
    
    fov_rad = math.radians(fov_deg)
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    
    f = 1.0 / math.tan(fov_rad / 2)
    
    # 1. Create meshgrid for output pixels (normalized -1 to 1)
    # y is down in image coords but we want y up for 3D logic usually. 
    # original code: yv is linspace(-1, 1), rays_y = -yv.
    # Let's match typical conventions: x right, y down (image) -> y up (world)? 
    # Let's stick to the previous code's coordinate system to minimize regression risk.
    # Previous: rays_x = xv, rays_y = -yv, rays_z = -f
    
    # Use torch meshgrid
    device = torch.device('cpu') # or 'cuda' if we want speed, but stay safe on CPU for image ops
    
    xv = torch.linspace(-1, 1, out_w, device=device)
    yv = torch.linspace(-1, 1, out_h, device=device)
    
    grid_y, grid_x = torch.meshgrid(yv, xv, indexing='ij')
    
    rays_x = grid_x
    rays_y = -grid_y
    rays_z = torch.full_like(grid_x, -f)
    
    # Normalize rays
    rays_norm = torch.sqrt(rays_x**2 + rays_y**2 + rays_z**2)
    rays_x = rays_x / rays_norm
    rays_y = rays_y / rays_norm
    rays_z = rays_z / rays_norm
    
    # Apply Rotations
    # Pitch (X axis)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    rays_y_p = rays_y * cos_p - rays_z * sin_p
    rays_z_p = rays_y * sin_p + rays_z * cos_p
    rays_x_p = rays_x # Unchanged
    
    # Yaw (Y axis)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rays_x_r = rays_x_p * cos_y + rays_z_p * sin_y
    rays_z_r = -rays_x_p * sin_y + rays_z_p * cos_y
    rays_y_r = rays_y_p # Unchanged
    
    # XYZ -> Spherical (theta, phi)
    # theta = atan2(x, -z) -> [-pi, pi]
    # phi = asin(y) -> [-pi/2, pi/2]
    
    theta = torch.atan2(rays_x_r, -rays_z_r)
    phi = torch.asin(torch.clamp(rays_y_r, -1.0, 1.0))
    
    # Spherical -> UV Grid [-1, 1] for grid_sample
    # u = theta / pi
    # v = - (2 * phi / pi) ? 
    # Recall v=0 is top (-1), v=1 is bottom (1)?
    # Previous code: v = (0.5 - phi/pi) * H -> 
    # If phi = pi/2 (up), v = 0.
    # If phi = -pi/2 (down), v = H.
    # In grid_sample: -1 is top, 1 is bottom.
    # So we want map phi (pi/2 -> -pi/2) to (-1 -> 1)
    
    u = theta / math.pi # [-1, 1]
    v = -2.0 * phi / math.pi # [-1 (up), 1 (down)]
    
    grid = torch.stack((u, v), dim=-1).unsqueeze(0) # [1, H, W, 2]
    
    # Sampling
    # Bicubic is smoother but requires align_corners=True/False checks
    # align_corners=True matches typical geometric implementations better
    
    out_tensor = torch.nn.functional.grid_sample(
        img_tensor, 
        grid, 
        mode='bicubic', 
        padding_mode='border', 
        align_corners=True
    )
    
    out_np = (out_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(out_np)


def create_filter_mask(
    pts3d_conf: np.ndarray,
    depth_preds: np.ndarray, 
    normal_preds: np.ndarray,
    sky_mask: np.ndarray,
    confidence_percentile: float = 10.0,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    apply_sky_mask: bool = False,
) -> np.ndarray:
    """
    Create comprehensive filter mask based on confidence, edges, and sky segmentation.
    Ported from HunyuanWorld-Mirror infer.py.
    """
    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []
    
    for i in range(S):
        final_mask = None
        
        if apply_confidence_mask:
            # Compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :]  # [H, W]
            percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask
        
        if apply_edge_mask:
            # Compute edge mask based on the normalmap
            normal_pred = normal_preds[i]  # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # Compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0]  # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask
        
        if apply_sky_mask and sky_mask is not None:
            # Apply sky mask filtering (sky_mask is already inverted: True = non-sky)
            sky_mask_frame = sky_mask[i]  # [H, W]
            if final_mask is None:
                final_mask = sky_mask_frame
            else:
                final_mask = final_mask & sky_mask_frame
        
        final_mask_list.append(final_mask)
    
    # Stack all frame masks
    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0)  # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool)  # [S, H, W]
    
    return final_mask


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class VNCCS_LoadWorldMirrorModel:
    """Load WorldMirror model for 3D reconstruction."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cpu"}),
                "sampling_strategy": (["conservative", "uniform"], {"default": "uniform"}),
                "enable_conf_filter": ("BOOLEAN", {"default": False}), 
                "conf_threshold_percent": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0}),
            }
        }
    
    RETURN_TYPES = ("WORLDMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"
    
    def load_model(self, device="cuda", sampling_strategy="uniform", enable_conf_filter=False, conf_threshold_percent=30.0):
        from src.models.models.worldmirror import WorldMirror
        
        print(f"🔄 Loading WorldMirror model (Strategy: {sampling_strategy}, Conf Filter: {enable_conf_filter}, Thresh: {conf_threshold_percent}%)")
        
        gs_params = {
            "enable_conf_filter": enable_conf_filter,
            "conf_threshold_percent": conf_threshold_percent
        }
        
        model = WorldMirror.from_pretrained(
            "tencent/HunyuanWorld-Mirror", 
            sampling_strategy=sampling_strategy,
            gs_params=gs_params
        )
        model = model.to(device)
        model.eval()
        print("✅ WorldMirror model loaded")
        
        return ({"model": model, "device": device},)


class VNCCS_WorldMirror3D:
    """Run 3D reconstruction on input images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WORLDMIRROR_MODEL",),
                "images": ("IMAGE",),
                "use_gsplat": ("BOOLEAN", {"default": True, "tooltip": "Enable Gaussian Splatting renderer (High Quality). If disabled, falls back to Point Cloud."}),
            },
            "optional": {
                "target_size": ("INT", {"default": 518, "min": 252, "max": 1024, "step": 14}),
                "offload_scheme": (["none", "model_cpu_offload", "sequential_cpu_offload"], {"default": "none"}),
                "stabilization": (["none", "panorama_lock"], {"default": "none"}),
                "confidence_percentile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "apply_sky_mask": ("BOOLEAN", {"default": False, "tooltip": "Remove sky regions (requires onnxruntime and skyseg.onnx)"}),
                "filter_edges": ("BOOLEAN", {"default": True, "tooltip": "Remove artifact points at object boundaries"}),
                "edge_normal_threshold": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 90.0, "step": 0.5}),
                "edge_depth_threshold": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics")
    FUNCTION = "run_inference"
    CATEGORY = "VNCCS/3D"
    
    def run_inference(self, model, images, use_gsplat=True, target_size=518, offload_scheme="none", stabilization="none", 
                      confidence_percentile=10.0, apply_sky_mask=False, filter_edges=True,
                      edge_normal_threshold=5.0, edge_depth_threshold=0.03):
        from torchvision import transforms
        
        if apply_sky_mask and not SKYSEG_AVAILABLE:
            print("⚠️ Sky segmentation requested but onnxruntime is missing. Ignoring.")
            apply_sky_mask = False
        
        # Ensure target_size is divisible by 14
        target_size = (target_size // 14) * 14
        
        worldmirror = model["model"]
        device = model["device"]
        
        # Convert ComfyUI images to tensor
        B, H, W, C = images.shape
        
        tensor_list = []
        converter = transforms.ToTensor()
        patch_size = 14
        
        for i in range(B):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            if pil_img.mode == "RGBA":
                white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
                pil_img = Image.alpha_composite(white_bg, pil_img)
            pil_img = pil_img.convert("RGB")
            
            orig_w, orig_h = pil_img.size
            new_w = target_size
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
            
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            tensor_img = converter(pil_img)
            
            if new_h > target_size:
                crop_start = (new_h - target_size) // 2
                tensor_img = tensor_img[:, crop_start:crop_start + target_size, :]
            
            tensor_list.append(tensor_img)
        
        # device management
        execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_device = next(worldmirror.parameters()).device
        
        imgs_tensor = torch.stack(tensor_list)
        # Use execution_device (GPU) for inputs, as accelerate expects inputs on the compute device
        imgs_tensor = imgs_tensor.unsqueeze(0).to(execution_device)

        print(f"🚀 Running WorldMirror inference on {B} images (offload: {offload_scheme})...")
        
        if offload_scheme != "none" and execution_device.type == "cuda":
            if offload_scheme == "sequential_cpu_offload":
                 # Manual Block Offloading (OOM fix for <16GB/24GB cards)
                 # We keep the main model on CPU, but move heads to GPU.
                 # The transformer handles its own internal block moving.
                 
                 # 0. STRIP ACCELERATE HOOKS if present (conflicting with manual offload)
                 def recursive_remove_hooks(module):
                     if hasattr(module, "_hf_hook"):
                         del module._hf_hook
                     if hasattr(module, "_old_forward"):
                         module.forward = module._old_forward
                         del module._old_forward
                     for child in module.children():
                         recursive_remove_hooks(child)

                 try:
                     recursive_remove_hooks(worldmirror)
                     # Also try accelerate's official method just in case
                     from accelerate.hooks import remove_hook_from_module
                     remove_hook_from_module(worldmirror, recurse=True)
                 except Exception as e:
                     print(f"⚠️ Failed to remove hooks: {e}")

                 # 1. Enable manual offload in transformer
                 if hasattr(worldmirror.visual_geometry_transformer, "manual_offload"):
                      worldmirror.visual_geometry_transformer.manual_offload = True
                 
                 # 2. Move heads to GPU manually (they are small-ish)
                 heads = [
                     getattr(worldmirror, "cam_head", None),
                     getattr(worldmirror, "pts_head", None),
                     getattr(worldmirror, "depth_head", None),
                     getattr(worldmirror, "norm_head", None),
                     getattr(worldmirror, "gs_head", None),
                 ]
                 for head in heads:
                     if head is not None:
                         # Check for corrupted/meta state from previous runs
                         param = next(head.parameters(), None)
                         if param is not None and param.device.type == 'meta':
                             raise RuntimeError(
                                 "CRITICAL: WorldMirror model is in a corrupted state (weights are on 'meta' device). "
                                 "This happens if a previous run crashed or used a different offload scheme. "
                                 "Please RESTART ComfyUI or invalidating the 'Load WorldMirror Model' node to reload fresh weights."
                             )
                         
                         # If head is on meta device, this might fail unless we materialize it.
                         head.to(execution_device)
            
            elif offload_scheme == "model_cpu_offload":
                try:
                    from accelerate import cpu_offload
                    cpu_offload(worldmirror, execution_device=execution_device)
                except ImportError:
                    print("⚠️ Accelerate not installed, ignoring model_cpu_offload")
                except Exception as e:
                    print(f"⚠️ Offload failed: {e}")
                    # Do NOT fallback to full GPU load if offload requested, as it likely causes OOM
        else:
            # Standard behavior: move everything to GPU
            param = next(worldmirror.parameters())
            # Check if likely meta device
            if param.device.type == 'meta':
                 print("⚠️ Model is on meta device, attempting to materialize empty weights...")
                 worldmirror.to_empty(device=execution_device)
                 # This is risky as weights are lost. But better than crash.
                 # Ideally user should reload model.
            elif param.device != execution_device:
                 worldmirror.to(execution_device)

        views = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]
        
        try:
            # Override GS State based on toggle
            original_gs_state = getattr(worldmirror, "enable_gs", True)
            
            # Force disable if gsplat lib is missing
            if use_gsplat and not GSPLAT_AVAILABLE:
                print("⚠️ gsplat requested but library not found. Falling back to Point Cloud.")
                worldmirror.enable_gs = False
            else:
                worldmirror.enable_gs = use_gsplat

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    predictions = worldmirror(views=views, cond_flags=cond_flags, stabilization=stabilization, confidence_percentile=confidence_percentile)
        finally:
            # Restore original state to avoid side effects
            if hasattr(worldmirror, "enable_gs"):
                worldmirror.enable_gs = original_gs_state

            # Cleanup for Manual Sequential Offload
            if offload_scheme == "sequential_cpu_offload":
                 # Move heads back to CPU
                 heads = [
                        getattr(worldmirror, "cam_head", None),
                        getattr(worldmirror, "pts_head", None),
                        getattr(worldmirror, "depth_head", None),
                        getattr(worldmirror, "norm_head", None),
                        getattr(worldmirror, "gs_head", None),
                    ]
                 for head in heads:
                        if head is not None:
                            head.to("cpu")
                 
                 # Disable manual flag (reset state)
                 if hasattr(worldmirror.visual_geometry_transformer, "manual_offload"):
                         worldmirror.visual_geometry_transformer.manual_offload = False
                 
                 torch.cuda.empty_cache()

            if offload_scheme == "none" and original_device.type == "cpu":
                 # If we moved it manually to GPU, and it came from CPU (which is now default),
                 # maybe we should move it back to save memory for other nodes?
                 # Actually, usually ComfyUI models stay where they are put.
                 # But if user selected "cpu" in loader, they expect it in CPU.
                 # Let's move it back if it wasn't offloaded by accelerate (accelerate handles it).
                 worldmirror.to("cpu")
                 torch.cuda.empty_cache()

        print("✅ Inference complete")
        
        # ============================================================================
        # Post-Processing: Filtering & Sky Masking (Ported)
        # ============================================================================
        
        S, H, W = predictions["depth"].shape[1:4] # Get dimensions from depth map
        B_batch = predictions["depth"].shape[0]   # Although we usually have B=1 in this node structure

        # 1. Compute Sky Mask
        sky_mask_np = None
        if apply_sky_mask:
            print("🌤️ Computing sky masks...")
            sky_model_path = os.path.join(folder_paths.models_dir, "skyseg.onnx")
            
            # Download if missing
            if not os.path.exists(sky_model_path):
                print(f"⬇️ Downloading skyseg.onnx to {sky_model_path}...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                    sky_model_path
                )
            
            if os.path.exists(sky_model_path):
                try:
                    skyseg_session = onnxruntime.InferenceSession(sky_model_path)
                    sky_mask_list = []
                    
                    # We need original images for sky seg ideally, but using tensor inputs converted back is fine
                    # images is [B, H_orig, W_orig, 3] or similar? No, images is [B, H, W, 3] from Comfy
                    # But note: we need the images that match the INFERENCE resolution (H,W) for mask consistency
                    
                    # Convert input tensor images to numpy for skyseg
                    # imgs_tensor is [1, S, 3, H, W]
                    for i in range(S):
                        img_np = imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy() # [H, W, 3]
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        sky_mask_frame = segment_sky(img_np, skyseg_session)
                        # Resize mask to match H×W if needed (segment_sky handles it but double check)
                        if sky_mask_frame.shape[0] != H or sky_mask_frame.shape[1] != W:
                            sky_mask_frame = cv2.resize(sky_mask_frame, (W, H))
                        sky_mask_list.append(sky_mask_frame)
                    
                    sky_mask_np = np.stack(sky_mask_list, axis=0) # [S, H, W]
                    sky_mask_np = sky_mask_np > 0 # Binary: True = non-sky
                    print(f"✅ Sky masks computed for {S} frames")
                except Exception as e:
                    print(f"❌ Sky segmentation failed: {e}")
                    sky_mask_np = None
            else:
                print("❌ Failed to download skyseg.onnx")

        # 2. Compute Filter Mask
        print("🔍 Computing geometric filter mask...")
        pts3d_conf_np = predictions["pts3d_conf"][0].detach().cpu().numpy()
        depth_preds_np = predictions["depth"][0].detach().cpu().numpy()
        normal_preds_np = predictions["normals"][0].detach().cpu().numpy()
        
        final_mask = create_filter_mask(
            pts3d_conf=pts3d_conf_np,
            depth_preds=depth_preds_np,
            normal_preds=normal_preds_np,
            sky_mask=sky_mask_np,
            confidence_percentile=confidence_percentile,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=True,
            apply_edge_mask=filter_edges,
            apply_sky_mask=apply_sky_mask
        ) # [S, H, W] bool array
        
        # 3. Apply Limit to outputs
        # We need to filter 'pts3d' (point cloud) AND 'splats' (gaussians)
        
        # Filter Point Cloud [1, S, H, W, 3] -> Flat List of valid points
        # Actually, ComfyUI style output keeps structure usually, but for PLY saving we want list
        # We will RETURN the mask or filtered points. 
        # Existing code accesses predictions["pts3d"] which is [1, S, H, W, 3]
        
        # Let's flatten and filter pts3d for the output dictionary
        # BUT: predictions["pts3d"] maps to pixels. If we remove pixels, we lose grid structure.
        # The PLY saver handles flattening. We should zero out invalid points or mark them.
        # Or better: Provide the filtered list directly in ply_data.
        
        all_pts_list = []
        all_conf_list = []
        
        if "splats" in predictions:
            # Note: We DO NOT filter splats here with final_mask!
            # Why: If enable_prune=True (default), splats are a sparse point cloud and do not map 1:1 to pixels.
            # Using a pixel-grid mask (final_mask) on sparse splats is mathematically wrong and causes size mismatches.
            # We pass splats raw, trusting the Model's internal GaussianSplatRenderer to have pruned/filtered them.
            print(f"ℹ️ Passing native Gaussian Splats (Pruned/Internal Filter).")
            # Ensure consistency of list vs tensor
            pass
        
        # Prepare mask for Point Cloud filtering
        final_mask_flat = torch.from_numpy(final_mask.reshape(-1)).to(execution_device)

        # Filter Points 3D (for PLY_DATA)
        # pts3d is [1, S, H, W, 3]. Flatten to [S*H*W, 3] and filter
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            # pts_conf = predictions["pts3d_conf"][0].reshape(-1)
            
            filtered_pts = pts[final_mask_flat.to(pts.device)]
            # filtered_conf = pts_conf[final_mask_flat]

        ply_data = {
            "pts3d": predictions.get("pts3d"), # Raw structured points (useful for depth maps)
            "pts3d_filtered": filtered_pts, # Filtered flat points
            "pts3d_conf": predictions.get("pts3d_conf"),
            "splats": predictions.get("splats"), # Now contains FILTERED splats
            "images": imgs_tensor,
            "filter_mask": final_mask_flat, # Pass mask for color filtering in fallback
            "camera_poses": predictions.get("camera_poses"),
            "camera_intrs": predictions.get("camera_intrs"),
        } 
        

        
        depth_tensor = predictions.get("depth")
        if depth_tensor is not None:
            depth = depth_tensor[0]
            depth_min = depth.min()
            depth_max = depth.max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            depth_rgb = depth_norm.repeat(1, 1, 1, 3)
            depth_out = depth_rgb.cpu().float()
        else:
            depth_out = torch.zeros(B, target_size, target_size, 3)
        
        normal_tensor = predictions.get("normals")
        if normal_tensor is not None:
            normals = normal_tensor[0]
            normals_out = ((normals + 1) / 2).cpu().float()
        else:
            normals_out = torch.zeros(B, target_size, target_size, 3)
        
        # Extract camera data for separate outputs
        camera_poses_out = predictions.get("camera_poses")
        camera_intrs_out = predictions.get("camera_intrs")
        
        # Move to CPU for ComfyUI compatibility
        if camera_poses_out is not None:
            camera_poses_out = camera_poses_out.cpu().float()
        if camera_intrs_out is not None:
            camera_intrs_out = camera_intrs_out.cpu().float()
        
        return (ply_data, depth_out, normals_out, camera_poses_out, camera_intrs_out)


class VNCCS_Equirect360ToViews:
    """Convert 360° equirectangular panorama to perspective views."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
            },
            "optional": {
                # FOV 60 minimizes edge distortion ("stretching") which is critical for SfM/Gaussian consistency
                "fov": ("INT", {"default": 60, "min": 30, "max": 120}),
                # Smaller step (30) ensures overlap despite narrower FOV
                "yaw_step": ("INT", {"default": 30, "min": 15, "max": 90}),
                "pitches": ("STRING", {"default": "0,-30,30"}),
                "output_size": ("INT", {"default": 518, "min": 252, "max": 1022, "step": 14}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("views",)
    FUNCTION = "extract_views"
    CATEGORY = "VNCCS/3D"
    
    def extract_views(self, panorama, fov=90, yaw_step=45, pitches="0,-30,30", output_size=518):
        pitch_list = [int(p.strip()) for p in pitches.split(",")]
        
        img_np = (panorama[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        yaw_angles = list(range(0, 360, yaw_step))
        
        views = []
        total = len(yaw_angles) * len(pitch_list)
        
        print(f"🔄 Extracting {total} views from 360° panorama...")
        
        for yaw in yaw_angles:
            for pitch in pitch_list:
                view = equirect_to_perspective(
                    pil_img,
                    fov_deg=fov,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    output_size=(output_size, output_size)
                )
                view_np = np.array(view).astype(np.float32) / 255.0
                views.append(view_np)
        
        views_tensor = torch.from_numpy(np.stack(views, axis=0))
        
        print(f"✅ Extracted {total} views")
        
        return (views_tensor,)


def extract_splat_params(data):
    """
    Robustly extract Gaussian Splatting parameters from WorldMirror output.
    Handles batches, lists, and point cloud fallbacks.
    """
    if data is None: 
        print("❌ [extract_splat_params] ERROR: Input data is None")
        return None
    
    print(f"🔍 [extract_splat_params] Input keys: {list(data.keys())}")
    splats = data.get("splats")
    pts3d = data.get("pts3d")
    images = data.get("images")
    
    # Debug: print raw value of splats
    print(f"🔍 [extract_splat_params] splats raw value: {type(splats)}, is None: {splats is None}")
    if splats is not None:
        if isinstance(splats, dict):
            print(f"🔍 [extract_splat_params] splats dict keys: {list(splats.keys())}")
        elif isinstance(splats, (list, tuple)):
            print(f"🔍 [extract_splat_params] splats is list/tuple with {len(splats)} elements")
            if len(splats) > 0:
                print(f"🔍 [extract_splat_params] splats[0] type: {type(splats[0])}")
        else:
            print(f"🔍 [extract_splat_params] splats is {type(splats)}")
    
    device = torch.device("cpu") # Extract to CPU for saving
    
    # Try to extract from splats
    if splats is not None:
        print(f"🔍 [extract_splat_params] Splats found, type: {type(splats)}")
        
        # Handle list-wrapped splats from some ComfyUI custom types
        if isinstance(splats, list) and len(splats) > 0:
            print(f"🔍 [extract_splat_params] splats is list with {len(splats)} elements, taking first")
            splats = splats[0]
            
        if isinstance(splats, dict) and len(splats) > 0:
            print(f"✅ [extract_splat_params] Processing splats dict with keys: {list(splats.keys())}")
            
            # Extract parameters from splats dict
            keys = ["means", "scales", "quats", "sh", "colors", "opacities"]
            params = {}
            for k in keys:
                v = splats.get(k)
                if v is None: continue
                # Handle potential list wrapper from some versions
                if isinstance(v, list): v = v[0]
                # Handle batch dimension [B, N, ...] -> [N, ...]
                if v.dim() >= 3: v = v[0]
                params[k] = v.detach().cpu().float()
            
            if "means" not in params:
                print("❌ [extract_splat_params] No 'means' in splats, falling back to pts3d")
            else:
                means = params["means"].reshape(-1, 3)
                
                scales = params.get("scales")
                if scales is not None:
                    scales = scales.reshape(-1, 3)
                    # Heuristic: if scales are mostly negative, they are in log space
                    if scales.to(torch.float32).mean() < -0.5:
                        print(f"🔍 [extract_splat_params] Detecting log-scales, applying exp()")
                        scales = torch.exp(scales)
                else:
                    print("⚠️ [extract_splat_params] No scales in splats, using default")
                    scales = torch.ones(means.shape[0], 3) * 0.01
                
                quats = params.get("quats")
                if quats is not None:
                    quats = quats.reshape(-1, 4)
                    if quats.shape[1] == 4 and quats.abs().sum() < 1e-6:
                        quats[:, 0] = 1.0  # Default to identity if all zeros
                else:
                    quats = torch.zeros(means.shape[0], 4)
                    quats[:, 0] = 1.0
                
                opacities = params.get("opacities")
                if opacities is not None:
                    opacities = opacities.reshape(-1)
                    # Heuristic: if opacities have values far outside [0, 1], they are logits
                    if opacities.min() < -2.0 or opacities.max() > 2.0:
                        print(f"🔍 [extract_splat_params] Detecting logit-opacities, applying sigmoid()")
                        opacities = torch.sigmoid(opacities)
                else:
                    opacities = torch.ones(means.shape[0]) * 0.9
                
                # Colors: convert to RGB [0, 1] if they look like SH coefficients
                colors_data = params.get("sh", params.get("colors"))
                if colors_data is not None:
                    if colors_data.dim() == 3:  # [N, SH, 3] -> DC is first SH
                        colors_data = colors_data[:, 0, :]
                    colors_data = colors_data.reshape(-1, 3)
                    
                    # Consistent with save_utils heuristic
                    c_np = colors_data.numpy()
                    if c_np.min() < -0.1 or c_np.max() > 1.1:
                        # Definitely SH
                        colors_data = colors_data * 0.28209479177387814 + 0.5
                    colors_data = torch.clamp(colors_data, 0.0, 1.0)
                else:
                    colors_data = torch.ones(means.shape[0], 3) * 0.5
                
                print(f"📊 [extract_splat_params] Gaussian Stats:")
                print(f"   - Means:  {means.shape} min={means.min(dim=0)[0].tolist()}, max={means.max(dim=0)[0].tolist()}")
                if scales is not None: print(f"   - Scales: {scales.shape}")
                if quats is not None: print(f"   - Quats:  {quats.shape}")
                if colors_data is not None: print(f"   - Colors: {colors_data.shape}")
                if opacities is not None: print(f"   - Opacity:{opacities.shape} min={opacities.min().item():.3f}")
                
                return means, scales, quats, colors_data, opacities
        else:
            print(f"⚠️ [extract_splat_params] splats is {type(splats)}, not a valid dict")
    
    # Fallback: Convert point cloud to dummy Gaussians
    if pts3d is not None or data.get("pts3d_filtered") is not None:
        print("🔍 [extract_splat_params] Using pts3d fallback (point cloud mode)")
        
        if data.get("pts3d_filtered") is not None:
             print("✨ Using FILTERED point cloud")
             means = data["pts3d_filtered"].detach().cpu().float()
        else:
             means = pts3d[0].view(-1, 3).detach().cpu().float()
             
        N = means.shape[0]
        
        # Cap at 2M to avoid crashing viewers
        if N > 2000000:
            idx = torch.randperm(N)[:2000000]
            means = means[idx]
            N = 2000000
        else:
            idx = None
            
        scales = torch.ones(N, 3) * 0.005  # Small visible splats
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0  # Identity rotation
        opacities = torch.ones(N) * 10.0  # High value (will be sigmoid'ed to ~1.0)
        
        if images is not None:
            S = pts3d.shape[1]
            colors_data = images[0, :S].permute(0, 2, 3, 1).reshape(-1, 3)
            
            # Apply filter mask if available (Crucial for mismatch fix)
            mask = data.get("filter_mask")
            if mask is not None:
                # Ensure mask matches raw size
                if mask.shape[0] == colors_data.shape[0]:
                    print(f"✨ [extract_splat_params] Applying filter mask to colors | Mask: {mask.shape} | Colors: {colors_data.shape}")
                    colors_data = colors_data[mask.to(colors_data.device)]
                else:
                    print(f"⚠️ [extract_splat_params] Mask size {mask.shape[0]} != Colors size {colors_data.shape[0]}, ignoring")
            
            if idx is not None:
                colors_data = colors_data[idx]
            colors_data = colors_data.detach().cpu().float()
            
            # DEBUG: Color Stats
            print(f"🎨 [extract_splat_params] Colors Stats: Min={colors_data.min():.3f}, Max={colors_data.max():.3f}, Mean={colors_data.mean():.3f}")
        else:
            colors_data = torch.ones(N, 3) * 0.5
        
        print(f"📊 [extract_splat_params] Point Cloud Stats: {N:,} points | GSPLAT_AVAILABLE={GSPLAT_AVAILABLE}")
            
        return means, scales, quats, colors_data, opacities
        
    print("❌ [extract_splat_params] No valid data found")
    return None



class VNCCS_SavePLY:
    """Save 3D reconstruction as PLY file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "filename": ("STRING", {"default": "output"}),
            },
            "optional": {
                "save_pointcloud": ("BOOLEAN", {"default": False}),
                "save_gaussians": ("BOOLEAN", {"default": True}),
                "rotate_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0}),
                "rotate_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0}),
                "rotate_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_ply"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True
    
    def _rotation_matrix(self, rx, ry, rz):
        """Create rotation matrix from Euler angles (degrees)."""
        rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return Rz @ Ry @ Rx

    def _rotate_quaternions(self, quats, R):
        """
        Rotate quaternions by rotation matrix R.
        quats: [N, 4] (w, x, y, z) or (x, y, z, w)? gsplat usually (w, x, y, z) or (x, y, z, w).
        3DGS usually uses (w, x, y, z).
        """
        # Convert R to quaternion q_R checking typical conversion math
        # Minimal implementation of Matrix to Quat (assuming R is pure rotation)
        # R is [3, 3]
        
        # We need batch multiplication.
        # Use pytorch3d or similar logic if available? No, simple math.
        # q_new = q_R * q_old
        
        # 1. R -> q_R (w, x, y, z)
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        
        q_R = torch.tensor([w, x, y, z], device=quats.device, dtype=quats.dtype)
        
        # 2. Quaternion Multiplication (Hamilton Product)
        # (w1, x1, y1, z1) * (w2, x2, y2, z2)
        # Input quats is [N, 4]
        
        # q_R is [4]
        w1, x1, y1, z1 = q_R[0], q_R[1], q_R[2], q_R[3]
        w2, x2, y2, z2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=1)

    def _get_unique_path(self, directory, filename, suffix, extension):
        counter = 1
        while True:
            # Use 5 digits like ComfyUI SaveImage
            full_filename = f"{filename}_{counter:05}_{suffix}.{extension}"
            path = os.path.join(directory, full_filename)
            if not os.path.exists(path):
                return path
            counter += 1


    def save_ply(self, ply_data, filename="output", save_pointcloud=False, save_gaussians=True,
                 rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
        from src.utils.save_utils import save_scene_ply, save_gs_ply
        
        output_dir = folder_paths.get_output_directory()
        saved_files = []
        
        R = None
        if rotate_x != 0 or rotate_y != 0 or rotate_z != 0:
            R = self._rotation_matrix(rotate_x, rotate_y, rotate_z).cpu()
            print(f"🔄 Applying rotation: X={rotate_x}°, Y={rotate_y}°, Z={rotate_z}°")
        
        if save_gaussians:
            # 1. Try Native Extraction (Primary)
            splats = ply_data.get("splats")
            native_success = False
            
            if splats is not None and isinstance(splats, dict):
                try:
                    # Native extraction matching infer.py
                    def get_tensor(k, dim):
                         v = splats.get(k)
                         if v is None: return None
                         if isinstance(v, list): v = v[0]
                         # if v.dim() > 2: v = v[0]  <-- REMOVED: broken for SH [N, 1, 3]
                         return v.reshape(-1, dim).detach().cpu().float()
                         
                    means = get_tensor("means", 3)
                    scales = get_tensor("scales", 3)
                    quats = get_tensor("quats", 4)
                    
                    print(f"🔍 [SavePLY] Inspecting Native Splats: Means={means.shape if means is not None else 'None'}")
                    
                    # Colors logic from infer.py
                    if "sh" in splats:
                        # Use SH if available (infer.py prefers SH)
                        colors = get_tensor("sh", 3) 
                    else:
                        colors = get_tensor("colors", 3)
                        
                    # FIX: Handle broadcasting if colors are global (1, C) but means are (N, 3)
                    if colors is not None and means is not None:
                        if colors.shape[0] == 1 and means.shape[0] > 1:
                            print(f"🎨 [SavePLY] Broadcasting global color {colors.shape} to {means.shape[0]} points")
                            colors = colors.repeat(means.shape[0], 1)
                         
                    opacities = get_tensor("opacities", 1).reshape(-1)
                    
                    if means is not None and scales is not None and quats is not None:
                         # Debug Stats
                         print(f"📊 [SavePLY Native] Stats:")
                         print(f"   - Means: {means.shape} [Min: {means.min():.3f}, Max: {means.max():.3f}]")
                         print(f"   - Scales: {scales.shape} [Min: {scales.min():.3f}, Max: {scales.max():.3f}, Mean: {scales.mean():.3f}]")
                         print(f"   - Opacity: {opacities.shape} [Min: {opacities.min():.3f}, Max: {opacities.max():.3f}]")

                         # Apply rotation to means AND quaternions
                         if R is not None:
                             means = (means @ R.T)
                             quats = self._rotate_quaternions(quats, R)
                         
                         gs_path = self._get_unique_path(output_dir, filename, "gaussians", "ply")
                         save_gs_ply(gs_path, means, scales, quats, colors, opacities)
                         saved_files.append(gs_path)
                         print(f"💾 [SavePLY] SUCCESS: Saved gaussians (Native): {os.path.basename(gs_path)} ({len(means)} pts)")
                         native_success = True
                except Exception as e:
                    print(f"⚠️ [SavePLY] Native extraction failed, falling back: {e}")
            
            # 2. Fallback to Helper (if splats missing or native failed)
            if not native_success:
                print("⚠️ [SavePLY] Using fallback extraction...")
                params = extract_splat_params(ply_data)
                if params:
                    means, scales, quats, colors, opacities = params
                    if R is not None:
                        means = (means.to(torch.float32) @ R.T).cpu()
                    
                    # FIX: Convert RGB to SH for correct color rendering in splat viewers
                    # Viewer: Color = 0.5 + 0.282 * SH
                    # Inverse: SH = (Color - 0.5) / 0.282
                    print("🎨 [SavePLY] Converting RGB to SH for Fallback Splats...")
                    SH_C0 = 0.28209479177387814
                    colors = (colors - 0.5) / SH_C0

                    gs_path = self._get_unique_path(output_dir, filename, "gaussians", "ply")
                    save_gs_ply(gs_path, means, scales, quats, colors, opacities)
                    saved_files.append(gs_path)
                    print(f"💾 [SavePLY] SUCCESS: Saved gaussians (Fallback): {os.path.basename(gs_path)} ({len(means)} pts)")
                else:
                    print("⚠️ [SavePLY] No splat data available after extraction")

        if save_pointcloud and ply_data.get("pts3d") is not None:
            pts3d = ply_data["pts3d"]
            images = ply_data["images"]
            means = pts3d[0].view(-1, 3).cpu()
            S = pts3d.shape[1]
            colors = images[0, :S].permute(0, 2, 3, 1).reshape(-1, 3).cpu()
            
            if R is not None:
                means = (means.to(torch.float32) @ R.T).cpu()
            
            # Use unique path generation to avoid overwriting
            pc_path = self._get_unique_path(output_dir, filename, "pointcloud", "ply")
            print(f"⏳ [SavePLY] Saving PointCloud PLY to: {pc_path}")
            
            # CRITICAL FIX for Pale Colors:
            # save_gs_ply expects SH coefficients. Viewer does: Color = 0.5 + 0.282 * SH
            # We have RGB [0..1]. So we need to Inverse Transform: SH = (RGB - 0.5) / 0.282
            # Otherwise 0->0.5 (Gray), 1->0.78 (Light Gray).
            SH_C0 = 0.28209479177387814
            colors_sh = (colors - 0.5) / SH_C0
            
            # We use save_gs_ply even for fallback points to get splat visualization
            # Creating dummy scales/quats/opacity for the point cloud
            from src.utils.save_utils import save_gs_ply
            
            N = len(means)
            dummy_scales = torch.ones(N, 3) * -4.6 # exp(-4.6) ~ 0.01
            dummy_quats = torch.zeros(N, 4); dummy_quats[:, 0] = 1.0
            dummy_opacities = torch.ones(N) * 100.0 # Opaque
            
            save_gs_ply(pc_path, means, dummy_scales, dummy_quats, colors_sh, dummy_opacities)
            # save_scene_ply(pc_path, means, colors) # OLD method (simple points)
            
            saved_files.append(pc_path)
            print(f"💾 [SavePLY] SUCCESS: Saved pointcloud: {os.path.basename(pc_path)} ({len(means)} pts)")
        
        if not saved_files:
            return ("",)
            
        # Prioritize returning the Gaussian file for preview
        for f in saved_files:
            if "_gaussians" in f:
                return (f,)
                
        return (saved_files[0],)
class VNCCS_BackgroundPreview:
    """
    Preview Gaussian Splatting PLY files with interactive gsplat.js viewer.
    
    Displays 3D Gaussian Splats in an interactive WebGL viewer with orbit controls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_width": ("STRING", {
                    "default": "512",
                    "tooltip": "Preview window width in pixels (integer)"
                }),
            },
            "optional": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "4x4 camera extrinsics matrix for initial view"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "3x3 camera intrinsics matrix for FOV"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_path", "ply_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "VNCCS/3D"
    OUTPUT_IS_LIST = (False, False)
    
    @classmethod
    def IS_CHANGED(cls, ply_path=None, **kwargs):
        """Force re-execution when a new video is recorded."""
        import glob
        output_dir = folder_paths.get_output_directory()
        try:
            pattern = os.path.join(output_dir, "gaussian-recording-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                return os.path.getmtime(video_files[0])
        except Exception:
            pass
        if ply_path:
            return hash(ply_path)
        return None
    
    def preview(self, ply_path=None, preview_width=512, extrinsics=None, intrinsics=None, **kwargs):
        """Prepare PLY file for gsplat.js preview."""
        import glob
        
        # Ensure preview_width is an int
        try:
            width_val = int(preview_width) if preview_width else 512
        except (ValueError, TypeError):
            width_val = 512
        
        # If no path provided, we can't preview
        if not ply_path:
            return {"ui": {}, "result": ("", "")}
        
        # Validate ply_path
        if not os.path.exists(ply_path):
            print(f"[VNCCS_BackgroundPreview] PLY file not found: {ply_path}")
            return {"ui": {"error": [f"File not found: {ply_path}"]}, "result": ("", "")}
        
        filename = os.path.basename(ply_path)
        # Prepare relative path and type for ComfyUI /view endpoint
        output_dir = folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()
        
        file_type = "output"
        rel_path = ""
        
        # Force forward slashes for Windows compatibility in browser URLs
        ply_path_norm = ply_path.replace("\\", "/")
        output_dir_norm = output_dir.replace("\\", "/")
        temp_dir_norm = temp_dir.replace("\\", "/")

        if ply_path_norm.startswith(output_dir_norm):
            rel_path = os.path.relpath(ply_path, output_dir).replace("\\", "/")
            file_type = "output"
        elif ply_path_norm.startswith(temp_dir_norm):
            rel_path = os.path.relpath(ply_path, temp_dir).replace("\\", "/")
            file_type = "temp"
        else:
            rel_path = os.path.basename(ply_path)
            file_type = "output" # Fallback
            
        subfolder = os.path.dirname(rel_path).replace("\\", "/")
        filename = os.path.basename(rel_path)
            
        file_size = os.path.getsize(ply_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"🔍 [VNCCS_BackgroundPreview] Preparing UI Data:")
        print(f"   - Full Path: {ply_path}")
        print(f"   - Filename: {filename}")
        print(f"   - Subfolder: {subfolder}")
        print(f"   - Type: {file_type}")
        print(f"   - Size: {file_size_mb:.2f} MB")
        
        # Find latest recorded video (optional/legacy)
        video_path = ""
        try:
            pattern = os.path.join(output_dir, "gaussian-recording-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                video_path = os.path.abspath(video_files[0])
                print(f"   - Found video: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"   ⚠️ Error finding video: {e}")
            
        ui_data = {
            "filename": [filename],
            "subfolder": [subfolder],
            "type": [file_type],
            "ply_path": [rel_path],
            "file_size_mb": [round(file_size_mb, 2)],
            "preview_width": [preview_width],
        }
        
        # Add camera parameters if provided
        if extrinsics is not None:
            ui_data["extrinsics"] = [extrinsics]
        if intrinsics is not None:
            ui_data["intrinsics"] = [intrinsics]
        
        print(f"✅ [VNCCS_BackgroundPreview] UI data ready. Returning to frontend.")
        return {"ui": ui_data, "result": (video_path, ply_path)}




class VNCCS_DecomposePLYData:
    """
    Extract individual components from PLY_DATA.
    
    Useful for accessing camera poses, intrinsics, point clouds, and splat parameters
    without going through the preview node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
            },
            "optional": {
                "view_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("camera_pose", "camera_intrinsics", "pts3d", "pts3d_conf")
    FUNCTION = "decompose"
    CATEGORY = "VNCCS/3D"
    
    def decompose(self, ply_data, view_index=0):
        """Extract camera and point cloud data from PLY data."""
        
        camera_pose = None
        camera_intrs = None
        pts3d = None
        pts3d_conf = None
        
        # Extract camera pose for specified view
        if ply_data.get("camera_poses") is not None:
            poses = ply_data["camera_poses"]
            # Shape: [B, S, 4, 4]
            S = poses.shape[1]
            idx = min(view_index, S - 1)
            camera_pose = poses[0, idx].cpu().float()  # [4, 4]
            print(f"[VNCCS_DecomposePLYData] Extracted camera pose for view {idx}")
        
        # Extract camera intrinsics for specified view
        if ply_data.get("camera_intrs") is not None:
            intrs = ply_data["camera_intrs"]
            # Shape: [B, S, 3, 3]
            S = intrs.shape[1]
            idx = min(view_index, S - 1)
            camera_intrs = intrs[0, idx].cpu().float()  # [3, 3]
            print(f"[VNCCS_DecomposePLYData] Extracted camera intrinsics for view {idx}")
        
        # Extract 3D points for specified view
        if ply_data.get("pts3d") is not None:
            pts = ply_data["pts3d"]
            # Shape: [B, S, H, W, 3]
            S = pts.shape[1]
            idx = min(view_index, S - 1)
            pts3d = pts[0, idx].cpu().float()  # [H, W, 3]
            print(f"[VNCCS_DecomposePLYData] Extracted pts3d for view {idx}")
        
        # Extract 3D point confidence for specified view
        if ply_data.get("pts3d_conf") is not None:
            conf = ply_data["pts3d_conf"]
            # Shape: [B, S, H, W]
            S = conf.shape[1]
            idx = min(view_index, S - 1)
            pts3d_conf = conf[0, idx].cpu().float()  # [H, W]
            print(f"[VNCCS_DecomposePLYData] Extracted pts3d_conf for view {idx}")
        
        return (camera_pose, camera_intrs, pts3d, pts3d_conf)


class VNCCS_PLYSceneRenderer:
    """
    Render multiple views from PLY/Gaussian splat for scene restoration.
    
    Loads Gaussian data from a PLY file (same format as the preview widget uses).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"forceInput": True, "tooltip": "Path to Gaussian PLY file from SavePLY node"}),
            },
            "optional": {
                "coverage_mode": (["minimal", "balanced", "ideal", "testing"], {"default": "balanced"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "fov": ("FLOAT", {"default": 90.0, "min": 30.0, "max": 120.0, "step": 5.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("views", "camera_poses", "camera_intrinsics")
    FUNCTION = "render_views"
    CATEGORY = "VNCCS/3D"
    OUTPUT_IS_LIST = (True, False, False)
    
    def _load_gaussian_ply(self, ply_path):
        """Load Gaussian parameters from PLY file (same format as viewer uses)."""
        from plyfile import PlyData
        
        print(f"🔍 [PLYSceneRenderer] Loading PLY: {ply_path}")
        ply = PlyData.read(ply_path)
        vertex = ply['vertex']
        
        N = len(vertex.data)
        print(f"   - Loaded {N:,} vertices")
        
        # Extract positions
        means = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float32)
        
        # Extract scales (stored as log in PLY, need to exp)
        if 'scale_0' in vertex.data.dtype.names:
            scales = np.column_stack([
                np.exp(vertex['scale_0']),
                np.exp(vertex['scale_1']),
                np.exp(vertex['scale_2'])
            ]).astype(np.float32)
            print(f"   - Scales: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
        else:
            # Fallback for simple point clouds
            scales = np.ones((N, 3), dtype=np.float32) * 0.01
            print("   - ⚠️ No scales found, using default 0.01")
        
        # Extract rotations
        if 'rot_0' in vertex.data.dtype.names:
            quats = np.column_stack([
                vertex['rot_0'], vertex['rot_1'], 
                vertex['rot_2'], vertex['rot_3']
            ]).astype(np.float32)
        else:
            quats = np.zeros((N, 4), dtype=np.float32)
            quats[:, 0] = 1.0  # Identity rotation
        
        # Extract colors (SH DC -> RGB)
        if 'f_dc_0' in vertex.data.dtype.names:
            sh_dc = np.column_stack([
                vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
            ]).astype(np.float32)
            # Convert SH DC to RGB: RGB = SH * 0.28209 + 0.5
            colors = sh_dc * 0.28209479177387814 + 0.5
            colors = np.clip(colors, 0.0, 1.0)
            print(f"   - Colors (from SH): min={colors.min():.3f}, max={colors.max():.3f}")
        elif 'red' in vertex.data.dtype.names:
            # Simple RGB point cloud
            colors = np.column_stack([
                vertex['red'], vertex['green'], vertex['blue']
            ]).astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            print(f"   - Colors (RGB): min={colors.min():.3f}, max={colors.max():.3f}")
        else:
            colors = np.ones((N, 3), dtype=np.float32) * 0.5
            print("   - ⚠️ No colors found, using gray")
        
        # Extract opacities (stored as logit in PLY, need sigmoid)
        if 'opacity' in vertex.data.dtype.names:
            opacity_raw = vertex['opacity'].astype(np.float32)
            # If values are outside [0, 1], they are logits
            if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
                opacities = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid
                print(f"   - Opacities (from logit): min={opacities.min():.3f}, max={opacities.max():.3f}")
            else:
                opacities = np.clip(opacity_raw, 0.0, 1.0)
                print(f"   - Opacities (direct): min={opacities.min():.3f}, max={opacities.max():.3f}")
        else:
            opacities = np.ones(N, dtype=np.float32) * 0.9
            print("   - ⚠️ No opacities found, using 0.9")
        
        return means, scales, quats, colors, opacities
    
    def _build_intrinsics(self, width, height, fov_deg, device):
        """Build intrinsic matrix from FOV and image size."""
        fov_rad = math.radians(fov_deg)
        focal = width / (2.0 * math.tan(fov_rad / 2.0))
        
        K = torch.zeros(3, 3, device=device)
        K[0, 0] = focal  # fx
        K[1, 1] = focal  # fy
        K[0, 2] = width / 2.0  # cx
        K[1, 2] = height / 2.0  # cy
        K[2, 2] = 1.0
        return K
    
    def _rotation_matrix_y(self, angle_deg, device):
        """Create rotation matrix around Y axis."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32, device=device)
    
    def _rotation_matrix_x(self, angle_deg, device):
        """Create rotation matrix around X axis."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=torch.float32, device=device)
    
    def _apply_rotation_to_pose(self, pose, rot_matrix):
        """Apply rotation to camera pose (c2w matrix)."""
        new_pose = pose.clone()
        # Rotate the rotation part of the pose
        new_pose[:3, :3] = pose[:3, :3] @ rot_matrix.T
        return new_pose
    
    def _translate_pose(self, pose, translation):
        """Translate camera position in world space."""
        new_pose = pose.clone()
        new_pose[:3, 3] = new_pose[:3, 3] + translation
        return new_pose
    
    def _make_look_at_pose(self, position, target, device):
        """Create a camera pose looking from position toward target."""
        forward = target - position
        forward = forward / (torch.norm(forward) + 1e-8)
        
        # Use Y-down convention (common in vision)
        up = torch.tensor([0.0, -1.0, 0.0], device=device)
        right = torch.linalg.cross(forward, up)
        right_norm = torch.norm(right)
        
        # Handle case when forward is parallel to up
        if right_norm < 1e-6:
            up = torch.tensor([0.0, 0.0, 1.0], device=device)
            right = torch.linalg.cross(forward, up)
        
        right = right / (torch.norm(right) + 1e-8)
        up = torch.linalg.cross(right, forward)
        
        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward # Fix: OpenCV style expects +Z forward
        pose[:3, 3] = position
        return pose
    
    def _generate_corner_positions(self, center, size, base_height, shrink=0.35, device=None):
        """Generate 4 corner positions inside room bounds."""
        half_x = size[0] * shrink
        half_z = size[2] * shrink
        
        corners = [
            center + torch.tensor([half_x, 0, half_z], device=device),   # +X +Z
            center + torch.tensor([-half_x, 0, half_z], device=device),  # -X +Z
            center + torch.tensor([-half_x, 0, -half_z], device=device), # -X -Z
            center + torch.tensor([half_x, 0, -half_z], device=device),  # +X -Z
        ]
        
        # Set Y to base camera height
        for c in corners:
            c[1] = base_height
        
        return corners
    
    def _generate_edge_positions(self, center, size, base_height, shrink=0.35, device=None):
        """Generate 4 edge midpoint positions inside room bounds."""
        half_x = size[0] * shrink
        half_z = size[2] * shrink
        
        edges = [
            center + torch.tensor([half_x, 0, 0], device=device),   # +X edge
            center + torch.tensor([-half_x, 0, 0], device=device),  # -X edge
            center + torch.tensor([0, 0, half_z], device=device),   # +Z edge
            center + torch.tensor([0, 0, -half_z], device=device),  # -Z edge
        ]
        
        for e in edges:
            e[1] = base_height
        
        return edges
    
    def _generate_coverage_views(self, scene_bounds, coverage_mode, seed, device):
        """
        Generate camera views optimized for >90% scene coverage.
        
        Strategy:
        - minimal: 4 corners looking inward + 2 pitch variants = 6 views
        - balanced: corners + edges + pitch variants = 10 views
        - ideal: corners + edges + center + multiple pitches = 14 views
        - testing: ideal + 360 rotation from center (looking outward) = 26 views
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        views = []
        center = (scene_bounds["min"] + scene_bounds["max"]) / 2
        size = scene_bounds["max"] - scene_bounds["min"]
        base_height = center[1].item()  # Use center Y as camera height
        
        # Always include corner positions (best for triangulation)
        corners = self._generate_corner_positions(center, size, base_height, shrink=0.35, device=device)
        
        if coverage_mode == "minimal":
            # 4 corners looking at center (diagonal coverage)
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 2 corner views with pitch down (floor/objects) from opposite corners
            for pos in [corners[0], corners[2]]:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]  # Look at floor level
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
                
        elif coverage_mode == "balanced":
            # 4 corners looking at center
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 4 edge positions
            edges = self._generate_edge_positions(center, size, base_height, shrink=0.35, device=device)
            for pos in edges:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 2 views looking down at floor from corners
            for pos in [corners[0], corners[2]]:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
                
        elif coverage_mode == "ideal" or coverage_mode == "testing":  # ideal logic applies to testing too
            # 4 corners looking at center
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 4 edge positions
            edges = self._generate_edge_positions(center, size, base_height, shrink=0.35, device=device)
            for pos in edges:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # Center view looking around (4 directions: +X, -X, +Z, -Z)
            center_h = center.clone()
            center_h[1] = base_height
            look_directions = [
                center_h + torch.tensor([1, 0, 0], device=device),
                center_h + torch.tensor([-1, 0, 0], device=device),
                center_h + torch.tensor([0, 0, 1], device=device),
                center_h + torch.tensor([0, 0, -1], device=device),
            ]
            for target in look_directions:
                pose = self._make_look_at_pose(center_h, target, device)
                views.append(pose)
            
            # 2 random high-coverage views
            for _ in range(2):
                rand_pos = corners[np.random.randint(0, 4)] + torch.randn(3, device=device) * 0.1
                rand_target = center + torch.randn(3, device=device) * 0.2
                pose = self._make_look_at_pose(rand_pos, rand_target, device)
                views.append(pose)
            
            # 4 views looking DOWN at floor (cover horizontal surfaces)
            for pos in corners:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
            
            # 2 views looking UP at ceiling (cover overhead)
            for pos in [corners[1], corners[3]]:
                ceiling_target = center.clone()
                ceiling_target[1] = scene_bounds["max"][1]
                pose = self._make_look_at_pose(pos, ceiling_target, device)
                views.append(pose)
        
        if coverage_mode == "testing":
             # "testing" mode EXTENDS "ideal" mode by adding a 360 spin
             # Similar to Equirect360ToViews but inside the room bounds
             
             # Calculate spin radius (slightly smaller than room bounds)
             radius = min(size[0], size[2]) * 0.3
             num_frames = 12 # 30 degree steps
             
             for i in range(num_frames):
                 angle = 2 * np.pi * i / num_frames
                 # Camera position is CENTER
                 cam_pos_center = center.clone()
                 cam_pos_center[1] = base_height
                 
                 # Look OUTWARD at a point on circle
                 # Note: in many 3D systems Z is forward/back, X is left/right. 
                 # Let's rotate in XZ plane.
                 target_x = center[0] + radius * np.cos(angle)
                 target_z = center[2] + radius * np.sin(angle)
                 target_pos = torch.tensor([target_x, base_height, target_z], device=device)
                 
                 # Camera at center, looking at target
                 pose = self._make_look_at_pose(cam_pos_center, target_pos, device)
                 views.append(pose)
        
        return views
    
    def _get_scene_bounds(self, means):
        """Get scene bounding box from gaussian means."""
        bounds_min = means.min(axis=0)
        bounds_max = means.max(axis=0)
        return {"min": torch.from_numpy(bounds_min), "max": torch.from_numpy(bounds_max)}
    
    def render_views(self, ply_path, coverage_mode="balanced", width=1024, height=1024, fov=90.0, seed=0):
        """Render multiple views from PLY file using ModernGL renderer."""
        import time
        import os
        
        print(f"🎥 [PLYSceneRenderer] RENDER START")
        print(f"   - PLY File: {os.path.basename(ply_path)}")
        print(f"   - Coverage Mode: {coverage_mode}")
        print(f"   - Resolution: {width}x{height}, FOV: {fov}")
        
        if not os.path.exists(ply_path):
            raise ValueError(f"PLY file not found: {ply_path}")
        
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Device: {device}")
        
        # Load Gaussian data from PLY file
        means_np, scales_np, quats_np, colors_np, opacities_np = self._load_gaussian_ply(ply_path)
        N = len(means_np)
        
        # Get scene bounds for view generation
        scene_bounds = self._get_scene_bounds(means_np)
        scene_bounds["min"] = scene_bounds["min"].to(device)
        scene_bounds["max"] = scene_bounds["max"].to(device)
        print(f"   - Scene bounds: {scene_bounds['min'].tolist()} to {scene_bounds['max'].tolist()}")
        
        # Build intrinsics
        K = self._build_intrinsics(width, height, fov, device)
        print(f"   - Intrinsics K:\n{K.cpu().numpy()}")
        
        # Generate camera poses using corner/edge strategy for optimal triangulation
        all_poses = self._generate_coverage_views(scene_bounds, coverage_mode, seed, device)
        print(f"   - Generated {len(all_poses)} views")
        
        # Stack poses and intrinsics
        poses_tensor = torch.stack(all_poses)  # [V, 4, 4]
        Ks_tensor = K.unsqueeze(0).expand(len(all_poses), -1, -1)  # [V, 3, 3]
        
        print(f"   Rendering {len(all_poses)} views at {width}x{height}...")
        
        
        # Initialize Fast Renderer
        print(f"   - Initializing FastPLYRenderer...")
        renderer = FastPLYRenderer(device)
        
        # Prepare Tensors on GPU
        print(f"   - Moving {N:,} points to GPU...")
        means_t = torch.from_numpy(means_np).to(device)
        colors_t = torch.from_numpy(colors_np).to(device)
        opacities_t = torch.from_numpy(opacities_np).to(device)
        scales_t = torch.from_numpy(scales_np).to(device)
        
        # Render each view
        rendered_images = []
        
        print(f"   📊 Gaussian Stats:")
        print(f"      - Points: {N:,}")
        print(f"      - Means:  avg={means_np.mean(axis=0)}, range=[{means_np.min(axis=0)}, {means_np.max(axis=0)}]")
        
        render_start = time.time()
            
        for i, pose in enumerate(all_poses):
            # Render using Fast PyTorch Splatter
            img_tensor = renderer.render(
                means=means_t,
                colors=colors_t,
                opacities=opacities_t,
                scales=scales_t,
                c2w=pose,
                width=width,
                height=height,
                fov_deg=fov
            )
            
            # DEBUG: Check if image is not black
            mean_val = img_tensor.mean().item()
            if (i + 1) % 5 == 0 or i == 0:
                 print(f"      - View {i}: mean pixel value {mean_val:.2f}")
                 
            # FastPLYRenderer returns float [H, W, 3] on GPU
            # ComfyUI expects [1, H, W, 3] on CPU (usually)
            rendered_images.append(img_tensor.cpu().unsqueeze(0))
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Rendered {i + 1}/{len(all_poses)} views")
        
        # Cleanup large tensors before finishing
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Move camera data to CPU
        poses_out = poses_tensor.cpu().float()
        Ks_out = Ks_tensor.cpu().float()
        
        total_render_time = time.time() - render_start
        total_total_time = time.time() - start_time
        print(f"✅ [PLYSceneRenderer] FINISHED")
        print(f"   - Render time: {total_render_time:.2f}s ({total_render_time/len(rendered_images):.3f}s/view)")
        print(f"   - Total time:  {total_total_time:.2f}s")
        
        return (rendered_images, poses_out, Ks_out)



# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": VNCCS_LoadWorldMirrorModel,
    "VNCCS_WorldMirror3D": VNCCS_WorldMirror3D,
    "VNCCS_Equirect360ToViews": VNCCS_Equirect360ToViews,
    "VNCCS_SavePLY": VNCCS_SavePLY,
    "VNCCS_BackgroundPreview": VNCCS_BackgroundPreview,
    "VNCCS_DecomposePLYData": VNCCS_DecomposePLYData,
    "VNCCS_PLYSceneRenderer": VNCCS_PLYSceneRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "🌐 Load WorldMirror Model",
    "VNCCS_WorldMirror3D": "🏔️ WorldMirror 3D Reconstruction",
    "VNCCS_Equirect360ToViews": "🔄 360° Panorama to Views",
    "VNCCS_SavePLY": "💾 Save PLY File",
    "VNCCS_BackgroundPreview": "👁️ Background Preview",
    "VNCCS_DecomposePLYData": "📦 Decompose PLY Data",
    "VNCCS_PLYSceneRenderer": "🎥 PLY Scene Renderer",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "VNCCS/3D",
    "VNCCS_WorldMirror3D": "VNCCS/3D",
    "VNCCS_Equirect360ToViews": "VNCCS/3D",
    "VNCCS_SavePLY": "VNCCS/3D",
    "VNCCS_BackgroundPreview": "VNCCS/3D",
    "VNCCS_DecomposePLYData": "VNCCS/3D",
    "VNCCS_PLYSceneRenderer": "VNCCS/3D",
}
