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
import folder_paths


# Ensure worldmirror is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDMIRROR_DIR = os.path.join(PROJECT_ROOT, "background-data", "worldmirror")
if WORLDMIRROR_DIR not in sys.path:
    sys.path.insert(0, WORLDMIRROR_DIR)

# Import FastPLYRenderer after setting up sys.path
try:
    from src.utils.fast_ply_render import FastPLYRenderer
except ImportError:
    # If using direct import (e.g. dev environment)
    from background_data.worldmirror.src.utils.fast_ply_render import FastPLYRenderer



# ============================================================================
# Utility Functions
# ============================================================================

def equirect_to_perspective(equirect_img, fov_deg, yaw_deg, pitch_deg, output_size=(512, 512)):
    """
    Extract a perspective view from an equirectangular panorama.
    
    Args:
        equirect_img: PIL Image of equirectangular panorama (2:1 aspect ratio)
        fov_deg: Field of view in degrees (horizontal and vertical)
        yaw_deg: Horizontal rotation (0=front, 90=right, 180=back, 270=left)
        pitch_deg: Vertical rotation (positive=up, negative=down)
        output_size: Output image size (width, height)
    
    Returns:
        PIL Image of the perspective view
    """
    equirect = np.array(equirect_img)
    h_eq, w_eq = equirect.shape[:2]
    
    out_w, out_h = output_size
    
    # Convert FOV to radians
    fov = math.radians(fov_deg)
    
    # Convert yaw and pitch to radians
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    
    # Create output pixel grid
    x = np.linspace(-1, 1, out_w)
    y = np.linspace(-1, 1, out_h)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate 3D ray directions for each output pixel
    f = 1.0 / math.tan(fov / 2)  # focal length
    
    # Ray directions in camera space (looking down -Z axis)
    rays_x = xv
    rays_y = -yv  # flip y
    rays_z = -np.ones_like(xv) * f
    
    # Normalize rays
    rays_len = np.sqrt(rays_x**2 + rays_y**2 + rays_z**2)
    rays_x /= rays_len
    rays_y /= rays_len
    rays_z /= rays_len
    
    # Rotation matrix for pitch (around X axis)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    # Rotation matrix for yaw (around Y axis)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    
    # Apply pitch rotation first
    rays_y_p = rays_y * cos_p - rays_z * sin_p
    rays_z_p = rays_y * sin_p + rays_z * cos_p
    rays_x_p = rays_x
    
    # Apply yaw rotation
    rays_x_r = rays_x_p * cos_y + rays_z_p * sin_y
    rays_z_r = -rays_x_p * sin_y + rays_z_p * cos_y
    rays_y_r = rays_y_p
    
    # Convert 3D rays to spherical coordinates
    theta = np.arctan2(rays_x_r, -rays_z_r)  # [-pi, pi]
    phi = np.arcsin(np.clip(rays_y_r, -1, 1))  # [-pi/2, pi/2]
    
    # Convert spherical to equirectangular pixel coordinates
    u = (theta / math.pi + 1) / 2 * w_eq  # [0, w_eq]
    v = (0.5 - phi / math.pi) * h_eq  # [0, h_eq]
    
    # Clamp coordinates
    u = np.clip(u, 0, w_eq - 1).astype(np.float32)
    v = np.clip(v, 0, h_eq - 1).astype(np.float32)
    
    # Bilinear interpolation
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, w_eq - 1)
    v1 = np.minimum(v0 + 1, h_eq - 1)
    
    du = u - u0
    dv = v - v0
    
    # Get pixel values
    if len(equirect.shape) == 3:
        du = du[:, :, np.newaxis]
        dv = dv[:, :, np.newaxis]
        
    p00 = equirect[v0, u0]
    p01 = equirect[v0, u1]
    p10 = equirect[v1, u0]
    p11 = equirect[v1, u1]
    
    # Bilinear interpolation
    output = (1 - du) * (1 - dv) * p00 + \
             du * (1 - dv) * p01 + \
             (1 - du) * dv * p10 + \
             du * dv * p11
    
    return Image.fromarray(output.astype(np.uint8))


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
            }
        }
    
    RETURN_TYPES = ("WORLDMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"
    
    def load_model(self, device="cuda"):
        from src.models.models.worldmirror import WorldMirror
        
        print("🔄 Loading WorldMirror model...")
        model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror")
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
            },
            "optional": {
                "target_size": ("INT", {"default": 518, "min": 252, "max": 1024, "step": 14}),
                "offload_scheme": (["none", "model_cpu_offload", "sequential_cpu_offload"], {"default": "none"}),
                "stabilization": (["none", "panorama_lock"], {"default": "none"}),
                "confidence_percentile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics")
    FUNCTION = "run_inference"
    CATEGORY = "VNCCS/3D"
    
    def run_inference(self, model, images, target_size=518, offload_scheme="none", stabilization="none", confidence_percentile=10.0):
        from torchvision import transforms
        
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
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    predictions = worldmirror(views=views, cond_flags=cond_flags, stabilization=stabilization, confidence_percentile=confidence_percentile)
        finally:
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
        
        ply_data = {
            "pts3d": predictions.get("pts3d"),
            "pts3d_conf": predictions.get("pts3d_conf"),
            "splats": predictions.get("splats"),
            "images": imgs_tensor,
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
                "fov": ("INT", {"default": 90, "min": 30, "max": 120}),
                "yaw_step": ("INT", {"default": 45, "min": 15, "max": 90}),
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
                print(f"   - Means:  min={means.min(dim=0)[0].tolist()}, max={means.max(dim=0)[0].tolist()}")
                print(f"   - Scales: min={scales.min(dim=0)[0].tolist()}, max={scales.max(dim=0)[0].tolist()}, mean={scales.mean(dim=0).tolist()}")
                print(f"   - Opacity: min={opacities.min().item():.3f}, max={opacities.max().item():.3f}, mean={opacities.mean().item():.3f}")
                
                return means, scales, quats, colors_data, opacities
        else:
            print(f"⚠️ [extract_splat_params] splats is {type(splats)}, not a valid dict")
    
    # Fallback: Convert point cloud to dummy Gaussians
    if pts3d is not None:
        print("🔍 [extract_splat_params] Using pts3d fallback (point cloud mode)")
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
            if idx is not None:
                colors_data = colors_data[idx]
            colors_data = colors_data.detach().cpu().float()
        else:
            colors_data = torch.ones(N, 3) * 0.5
        
        print(f"📊 [extract_splat_params] Point Cloud Stats: {N:,} points")
            
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
            params = extract_splat_params(ply_data)
            if params:
                means, scales, quats, colors, opacities = params
                if R is not None:
                    means = (means.to(torch.float32) @ R.T).cpu()
                
                # Use unique path generation to avoid overwriting
                gs_path = self._get_unique_path(output_dir, filename, "gaussians", "ply")
                print(f"⏳ [SavePLY] Saving Gaussian PLY to: {gs_path}")
                save_gs_ply(gs_path, means, scales, quats, colors, opacities)
                saved_files.append(gs_path)
                print(f"💾 [SavePLY] SUCCESS: Saved gaussians: {os.path.basename(gs_path)} ({len(means)} pts)")
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
            save_scene_ply(pc_path, means, colors)
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
