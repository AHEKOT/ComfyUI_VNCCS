"""
Utilities for saving point clouds and Gaussian splat data.
Minimal version for ComfyUI node.
"""
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def save_scene_ply(path: Path,
                   points_xyz: torch.Tensor,
                   point_colors: torch.Tensor,
                   valid_mask: torch.Tensor = None) -> None:
    """Save point cloud to PLY format (gsplat.js compatible)"""
    pts = points_xyz.detach().cpu().to(torch.float32).numpy().reshape(-1, 3)
    # Convert colors to float32 in 0-1 range for gsplat.js compatibility
    colors = point_colors.detach().cpu().to(torch.float32).numpy().reshape(-1, 3)
    if colors.max() > 1.0:
        colors = colors / 255.0  # Normalize if uint8 values
    
    # Filter out invalid points (NaN, Inf)
    if valid_mask is None:
        valid_mask = np.isfinite(pts).all(axis=1)
    else:
        valid_mask = valid_mask.detach().cpu().numpy().reshape(-1)
    pts = pts[valid_mask]
    colors = colors[valid_mask]
    
    # Handle empty point cloud
    if len(pts) == 0:
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    # Create PLY data with uint8 colors for standard viewer compatibility
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), 
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    
    # Fast numpy record array construction
    vertex_elements = np.empty(len(pts), dtype=vertex_dtype)
    vertex_elements["x"] = pts[:, 0]
    vertex_elements["y"] = pts[:, 1]
    vertex_elements["z"] = pts[:, 2]
    
    # Ensure colors are 0-255 uint8
    if colors.dtype != np.uint8:
        if colors.max() <= 1.01:
            colors = (colors * 255)
        # Vectorized clipping and cast
        colors = np.clip(colors, 0, 255).astype(np.uint8)
            
    vertex_elements["red"] = colors[:, 0]
    vertex_elements["green"] = colors[:, 1]
    vertex_elements["blue"] = colors[:, 2]
    
    # Write PLY file
    PlyData([PlyElement.describe(vertex_elements, "vertex")], text=False).write(str(path))


def save_gs_ply(path: Path,
                means: torch.Tensor,
                scales: torch.Tensor,
                rotations: torch.Tensor,
                rgbs: torch.Tensor,
                opacities: torch.Tensor) -> None:
    """
    Export Gaussian splat data to PLY format.
    
    Args:
        path: Output PLY file path
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
    """
    # Remove NaN/Inf points and tiny scales or opacities that bloat the file
    valid_mask = torch.isfinite(means).all(dim=-1) & \
                 torch.isfinite(scales).all(dim=-1) & \
                 torch.isfinite(rotations).all(dim=-1) & \
                 torch.isfinite(rgbs).all(dim=-1) & \
                 torch.isfinite(opacities)
    
    # Filter points
    means = means[valid_mask].reshape(-1, 3)
    scales = scales[valid_mask].reshape(-1, 3)
    rotations = rotations[valid_mask].reshape(-1, 4)
    rgbs = rgbs[valid_mask].reshape(-1, 3)
    opacities = opacities[valid_mask].reshape(-1)

    # Construct attribute names
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")

    # Prepare PLY data structure
    dtype_full = [(attribute, "f4") for attribute in attributes]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    
    # Convert RGB to SH DC if it looks like raw RGB [f_dc_0 = (RGB - 0.5) / 0.28209]
    rgbs_data = rgbs.detach().cpu().float().numpy()
    if rgbs_data.max() > 1.01:
        rgbs_data = rgbs_data / 255.0
    
    # Simple heuristic: if most values are positive and in 0-1, it's likely RGB
    if rgbs_data.min() >= -0.1 and rgbs_data.max() <= 1.1:
        sh_dc = (rgbs_data - 0.5) / 0.28209479177387814
    else:
        # Likely already SH coefficients
        sh_dc = rgbs_data
    
    # Fast assignment using record array fields
    elements["x"] = means[:, 0].detach().cpu().numpy()
    elements["y"] = means[:, 1].detach().cpu().numpy()
    elements["z"] = means[:, 2].detach().cpu().numpy()
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    for i in range(3):
        elements[f"f_dc_{i}"] = sh_dc[:, i]
    elements["opacity"] = opacities.detach().cpu().numpy()
    
    scales_np = scales.detach().cpu().numpy()
    for i in range(3):
        elements[f"scale_{i}"] = np.log(np.maximum(scales_np[:, i], 1e-8))
        
    rotations_np = rotations.detach().cpu().numpy()
    for i in range(4):
        elements[f"rot_{i}"] = rotations_np[:, i]
    
    # Write to PLY file
    print(f"📁 [save_gs_ply] Writing {len(elements)} vertices to {path}...")
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(str(path))
    print(f"✅ [save_gs_ply] Write complete.")
