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
                "device": (["cuda", "cpu"], {"default": "cuda"}),
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
                "target_size": ("INT", {"default": 518, "min": 256, "max": 1024, "step": 14}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ply_data", "depth_maps", "normal_maps")
    FUNCTION = "run_inference"
    CATEGORY = "VNCCS/3D"
    
    def run_inference(self, model, images, target_size=518):
        from torchvision import transforms
        
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
        
        imgs_tensor = torch.stack(tensor_list)
        imgs_tensor = imgs_tensor.unsqueeze(0).to(device)

        print(f"🚀 Running WorldMirror inference on {B} images...")
        views = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                predictions = worldmirror(views=views, cond_flags=cond_flags)
        
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
        
        return (ply_data, depth_out, normals_out)


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
                "output_size": ("INT", {"default": 518, "min": 256, "max": 1024, "step": 14}),
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
                "save_pointcloud": ("BOOLEAN", {"default": True}),
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
    
    def save_ply(self, ply_data, filename="output", save_pointcloud=True, save_gaussians=True,
                 rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
        from src.utils.save_utils import save_scene_ply, save_gs_ply
        
        output_dir = folder_paths.get_output_directory()
        saved_files = []
        
        R = None
        if rotate_x != 0 or rotate_y != 0 or rotate_z != 0:
            R = self._rotation_matrix(rotate_x, rotate_y, rotate_z)
            print(f"🔄 Applying rotation: X={rotate_x}°, Y={rotate_y}°, Z={rotate_z}°")
        
        if save_pointcloud and ply_data.get("pts3d") is not None:
            pts3d = ply_data["pts3d"]
            images = ply_data["images"]
            
            S = pts3d.shape[1]
            pts_list = []
            colors_list = []
            
            for i in range(S):
                pts = pts3d[0, i]
                img_colors = images[0, i].permute(1, 2, 0)
                img_colors = (img_colors * 255).to(torch.uint8)
                
                pts_list.append(pts.reshape(-1, 3))
                colors_list.append(img_colors.reshape(-1, 3))
            
            all_pts = torch.cat(pts_list, dim=0)
            all_colors = torch.cat(colors_list, dim=0)
            
            if R is not None:
                all_pts = all_pts.cpu().float() @ R.T
            
            pc_path = os.path.join(output_dir, f"{filename}_pointcloud.ply")
            save_scene_ply(pc_path, all_pts, all_colors)
            saved_files.append(pc_path)
            print(f"💾 Saved pointcloud: {pc_path}")
        
        if save_gaussians and ply_data.get("splats") is not None:
            splats = ply_data["splats"]
            means = splats["means"][0].reshape(-1, 3)
            scales = splats["scales"][0].reshape(-1, 3)
            quats = splats["quats"][0].reshape(-1, 4)
            colors = splats.get("sh", splats.get("colors"))[0].reshape(-1, 3)
            opacities = splats["opacities"][0].reshape(-1)
            
            if R is not None:
                means = means.cpu().float() @ R.T
            
            gs_path = os.path.join(output_dir, f"{filename}_gaussians.ply")
            save_gs_ply(gs_path, means, scales, quats, colors, opacities)
            saved_files.append(gs_path)
            print(f"💾 Saved gaussians: {gs_path}")
        
        return (", ".join(saved_files),)
class VNCCS_BackgroundPreview:
    """
    Preview Gaussian Splatting PLY files with interactive gsplat.js viewer.
    
    Displays 3D Gaussian Splats in an interactive WebGL viewer with orbit controls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
                "ply_data": ("PLY_DATA", {
                    "forceInput": True,
                    "tooltip": "PLY data from WorldMirror reconstruction"
                }),
                "preview_width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Preview window width in pixels"
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
    def IS_CHANGED(cls, ply_path=None, ply_data=None, **kwargs):
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
    
    def preview(self, ply_path=None, ply_data=None, preview_width=512):
        """Prepare PLY file for gsplat.js preview."""
        import glob
        
        saved_ply_path = None
        
        # Handle ply_data input - save to file first
        if ply_data is not None and ply_data.get("splats") is not None:
            print("[VNCCS_BackgroundPreview] Converting PLY data to file...")
            from src.utils.save_utils import save_gs_ply
            
            output_dir = folder_paths.get_output_directory()
            import time
            timestamp = int(time.time() * 1000)
            output_filename = f"background_preview_{timestamp}.ply"
            saved_ply_path = os.path.join(output_dir, output_filename)
            
            splats = ply_data["splats"]
            means = splats["means"][0].reshape(-1, 3)
            scales = splats["scales"][0].reshape(-1, 3)
            quats = splats["quats"][0].reshape(-1, 4)
            colors = splats.get("sh", splats.get("colors"))[0].reshape(-1, 3)
            opacities = splats["opacities"][0].reshape(-1)
            
            save_gs_ply(saved_ply_path, means, scales, quats, colors, opacities)
            ply_path = saved_ply_path
            print(f"[VNCCS_BackgroundPreview] Saved temporary PLY: {saved_ply_path}")
        
        # Validate ply_path
        if not ply_path:
            print("[VNCCS_BackgroundPreview] No PLY path or data provided")
            return {"ui": {"error": ["No PLY path or data provided"]}, "result": ("", "")}
        
        if not os.path.exists(ply_path):
            print(f"[VNCCS_BackgroundPreview] PLY file not found: {ply_path}")
            return {"ui": {"error": [f"File not found: {ply_path}"]}, "result": ("", "")}
        
        if saved_ply_path is None:
            saved_ply_path = ply_path
        
        filename = os.path.basename(ply_path)
        output_dir = folder_paths.get_output_directory()
        
        if ply_path.startswith(output_dir):
            relative_path = os.path.relpath(ply_path, output_dir)
        else:
            relative_path = filename
        
        file_size = os.path.getsize(ply_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[VNCCS_BackgroundPreview] Loading PLY: {filename} ({file_size_mb:.2f} MB)")
        
        ui_data = {
            "ply_file": [relative_path],
            "filename": [filename],
            "file_size_mb": [round(file_size_mb, 2)],
            "preview_width": [preview_width],
        }
        
        # Find latest recorded video
        video_path = ""
        try:
            pattern = os.path.join(output_dir, "gaussian-recording-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                video_path = os.path.abspath(video_files[0])
                print(f"[VNCCS_BackgroundPreview] Found video: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"[VNCCS_BackgroundPreview] Error finding video: {e}")
        
        return {"ui": ui_data, "result": (video_path, saved_ply_path)}


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": VNCCS_LoadWorldMirrorModel,
    "VNCCS_WorldMirror3D": VNCCS_WorldMirror3D,
    "VNCCS_Equirect360ToViews": VNCCS_Equirect360ToViews,
    "VNCCS_SavePLY": VNCCS_SavePLY,
    "VNCCS_BackgroundPreview": VNCCS_BackgroundPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "🌐 Load WorldMirror Model",
    "VNCCS_WorldMirror3D": "🏔️ WorldMirror 3D Reconstruction",
    "VNCCS_Equirect360ToViews": "🔄 360° Panorama to Views",
    "VNCCS_SavePLY": "💾 Save PLY File",
    "VNCCS_BackgroundPreview": "👁️ Background Preview",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "VNCCS/3D",
    "VNCCS_WorldMirror3D": "VNCCS/3D",
    "VNCCS_Equirect360ToViews": "VNCCS/3D",
    "VNCCS_SavePLY": "VNCCS/3D",
    "VNCCS_BackgroundPreview": "VNCCS/3D",
}
