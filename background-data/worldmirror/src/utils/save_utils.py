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
    """Save point cloud to PLY format"""
    pts = points_xyz.detach().cpu().to(torch.float32).numpy().reshape(-1, 3)
    colors = point_colors.detach().cpu().to(torch.uint8).numpy().reshape(-1, 3)
    
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
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    # Create PLY data
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), 
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex_elements = np.empty(len(pts), dtype=vertex_dtype)
    vertex_elements["x"] = pts[:, 0]
    vertex_elements["y"] = pts[:, 1]
    vertex_elements["z"] = pts[:, 2]
    vertex_elements["red"] = colors[:, 0]
    vertex_elements["green"] = colors[:, 1]
    vertex_elements["blue"] = colors[:, 2]
    
    # Write PLY file
    PlyData([PlyElement.describe(vertex_elements, "vertex")]).write(str(path))


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
    # Filter out points with scales greater than the 95th percentile
    scale_threshold = torch.quantile(scales.max(dim=-1)[0], 0.95, dim=0)
    filter_mask = scales.max(dim=-1)[0] <= scale_threshold

    # Apply the filter to all tensors
    means = means[filter_mask].reshape(-1, 3)
    scales = scales[filter_mask].reshape(-1, 3)
    rotations = rotations[filter_mask].reshape(-1, 4)
    rgbs = rgbs[filter_mask].reshape(-1, 3)
    opacities = opacities[filter_mask].reshape(-1)

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
    
    # Concatenate all attributes
    attributes_data = (
        means.float().detach().cpu().numpy(),
        torch.zeros_like(means).float().detach().cpu().numpy(),
        rgbs.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes_data = np.concatenate(attributes_data, axis=1)
    elements[:] = list(map(tuple, attributes_data))
    
    # Write to PLY file
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
