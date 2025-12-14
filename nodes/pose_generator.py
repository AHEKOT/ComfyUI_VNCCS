"""VNCCS Pose Generator Node

Generates pose images in 512x1536 format with interactive visual editor.
Outputs both schematic view and OpenPose format.
"""

import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

# Import from pose_utils folder
LEGACY_JOINT_ALIASES = {}

try:
    # Try relative import first
    from ..pose_utils.skeleton_512x1536 import (
        Skeleton,
        DEFAULT_SKELETON,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        LEGACY_JOINT_ALIASES,
    )
    from ..pose_utils.pose_renderer import render_schematic, render_openpose, convert_to_comfyui_format
except (ImportError, ValueError) as e:
    # Fallback to adding to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    utils_dir = os.path.join(parent_dir, "pose_utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    
    try:
        from skeleton_512x1536 import (
            Skeleton,
            DEFAULT_SKELETON,
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
            LEGACY_JOINT_ALIASES,
        )
        from pose_renderer import render_schematic, render_openpose, convert_to_comfyui_format
    except ImportError:
        # Last resort: try as package import
        import importlib.util
        spec_skeleton = importlib.util.spec_from_file_location("skeleton_512x1536", os.path.join(utils_dir, "skeleton_512x1536.py"))
        skeleton_module = importlib.util.module_from_spec(spec_skeleton)
        spec_skeleton.loader.exec_module(skeleton_module)
        
        spec_renderer = importlib.util.spec_from_file_location("pose_renderer", os.path.join(utils_dir, "pose_renderer.py"))
        renderer_module = importlib.util.module_from_spec(spec_renderer)
        spec_renderer.loader.exec_module(renderer_module)
        
        Skeleton = skeleton_module.Skeleton
        DEFAULT_SKELETON = skeleton_module.DEFAULT_SKELETON
        CANVAS_WIDTH = skeleton_module.CANVAS_WIDTH
        CANVAS_HEIGHT = skeleton_module.CANVAS_HEIGHT
        LEGACY_JOINT_ALIASES = getattr(skeleton_module, "LEGACY_JOINT_ALIASES", {})
        render_schematic = renderer_module.render_schematic
        render_openpose = renderer_module.render_openpose
        convert_to_comfyui_format = renderer_module.convert_to_comfyui_format


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _sanitize_joints(joints_data: Dict[str, Tuple]) -> Dict[str, Tuple[int, int]]:
    """Normalize joint payload into clamped integer tuples following OpenPose BODY_25 names."""
    sanitized: Dict[str, Tuple[int, int]] = {}

    for raw_name, coords in joints_data.items():
        joint_name = LEGACY_JOINT_ALIASES.get(raw_name, raw_name)
        if joint_name not in DEFAULT_SKELETON:
            continue

        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue

        try:
            x = float(coords[0])
            y = float(coords[1])
        except (TypeError, ValueError):
            continue

        x_int = _clamp(int(round(x)), 0, CANVAS_WIDTH - 1)
        y_int = _clamp(int(round(y)), 0, CANVAS_HEIGHT - 1)
        sanitized[joint_name] = (x_int, y_int)

    for joint_name, default_coords in DEFAULT_SKELETON.items():
        sanitized.setdefault(joint_name, default_coords)

    return sanitized


class VNCCS_PoseGenerator:
    """Pose Generator with visual editor for creating character poses"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to load default preset
        default_pose_json = None
        preset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "presets", "poses", "vnccs_poseset.json"
        )
        
        if os.path.exists(preset_path):
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    default_pose_json = json.dumps(json.load(f), indent=2)
            except Exception as e:
                print(f"[VNCCS] Warning: Could not load default preset: {e}")
        
        # Fallback to creating default 12 poses if preset doesn't exist
        if default_pose_json is None:
            default_pose_list = []
            for _ in range(12):
                default_pose_list.append({
                    "joints": {name: list(pos) for name, pos in DEFAULT_SKELETON.items()}
                })
            default_pose_json = json.dumps({
                "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
                "poses": default_pose_list
            }, indent=2)
            
        return {
            "required": {
                "pose_data": ("STRING", {
                    "default": default_pose_json,
                    "multiline": True,
                    "dynamicPrompts": False
                }),
                "line_thickness": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("openpose_grid",)
    FUNCTION = "generate"
    CATEGORY = "VNCCS/pose"
    
    def generate(self, pose_data: str, line_thickness: int = 3):
        """Generate OpenPose image grid from pose data (12 poses in 6x2 layout)
        
        Args:
            pose_data: JSON string containing list of 12 poses
            line_thickness: Thickness of lines in OpenPose rendering
        
        Returns:
            Tuple containing (openpose_grid,) in ComfyUI format
        """
        print(f"[VNCCS] Generating pose grid...")
        
        try:
            data = json.loads(pose_data)
        except json.JSONDecodeError as exc:
            print(f"[VNCCS] ERROR: Invalid JSON in pose_data: {exc}")
            data = {}

        # Extract poses list
        poses_data = data.get("poses", [])
        if not isinstance(poses_data, list):
            # Handle legacy single pose format by wrapping it
            joints_payload = data.get("joints", {})
            if joints_payload:
                poses_data = [{"joints": joints_payload}]
            else:
                poses_data = []
        
        # Ensure we have exactly 12 poses, filling with default if needed
        poses = []
        for i in range(12):
            if i < len(poses_data) and isinstance(poses_data[i], dict):
                # Handle both wrapped {"joints": {...}} and flat {...} formats
                if "joints" in poses_data[i]:
                    joints_payload = poses_data[i]["joints"]
                else:
                    joints_payload = poses_data[i]
                
                poses.append(_sanitize_joints(joints_payload))
            else:
                poses.append(DEFAULT_SKELETON.copy())
        
        print(f"[VNCCS] Processing {len(poses)} poses")
        
        # Single pose dimensions
        w, h = CANVAS_WIDTH, CANVAS_HEIGHT
        
        # Grid dimensions (6 columns, 2 rows)
        cols = 6
        rows = 2
        grid_w = w * cols
        grid_h = h * rows
        
        # Create empty grid (RGB)
        # OpenPose: Black background
        openpose_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx, joints in enumerate(poses):
            # Calculate grid position
            row = idx // cols
            col = idx % cols
            x_offset = col * w
            y_offset = row * h
            
            # Render OpenPose
            openpose_img = render_openpose(joints, w, h, line_thickness)
            
            # Place in grid
            openpose_grid[y_offset:y_offset+h, x_offset:x_offset+w] = openpose_img
            
        # Convert to ComfyUI format [B, H, W, C]
        openpose_tensor = convert_to_comfyui_format(openpose_grid)
        
        # Convert to torch tensors
        openpose_tensor = torch.from_numpy(openpose_tensor)
        
        print(f"[VNCCS] Generated grid image:")
        print(f"  OpenPose: {openpose_tensor.shape}")
        
        return (openpose_tensor,)


NODE_CLASS_MAPPINGS = {
    "VNCCS_PoseGenerator": VNCCS_PoseGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PoseGenerator": "VNCCS Pose Generator",
}
