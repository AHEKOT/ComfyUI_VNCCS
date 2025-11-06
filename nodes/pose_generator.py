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
        BONE_CONNECTIONS,
    )
    from ..pose_utils.pose_renderer import render_schematic, render_openpose, convert_to_comfyui_format
    from ..pose_utils.advanced_renderer import render_all_maps
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
            BONE_CONNECTIONS,
        )
        from pose_renderer import render_schematic, render_openpose, convert_to_comfyui_format
        from advanced_renderer import render_all_maps
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
        BONE_CONNECTIONS = getattr(skeleton_module, "BONE_CONNECTIONS", [])
        render_schematic = renderer_module.render_schematic
        render_openpose = renderer_module.render_openpose
        convert_to_comfyui_format = renderer_module.convert_to_comfyui_format
        
        spec_advanced = importlib.util.spec_from_file_location("advanced_renderer", os.path.join(utils_dir, "advanced_renderer.py"))
        advanced_module = importlib.util.module_from_spec(spec_advanced)
        spec_advanced.loader.exec_module(advanced_module)
        render_all_maps = advanced_module.render_all_maps


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _unwrap_openpose_payload(payload):
    """Peel common wrappers around OpenPose payloads until list/dict/string."""
    visited = set()
    current = payload
    while True:
        if current is None:
            return None
        ident = id(current)
        if ident in visited:
            return current
        visited.add(ident)

        if isinstance(current, (dict, list)):
            return current
        if isinstance(current, (bytes, bytearray)):
            try:
                current = current.decode("utf-8")
                continue
            except Exception:
                return None
        if isinstance(current, str):
            try:
                return json.loads(current)
            except json.JSONDecodeError:
                print("[VNCCS] WARNING: Invalid JSON keypoints format")
                return None
        # Objects with helpful accessors
        for attr in ("json", "to_json", "as_dict", "to_dict", "dict", "data"):
            if hasattr(current, attr):
                try:
                    candidate = getattr(current, attr)
                    candidate = candidate() if callable(candidate) else candidate
                    if candidate is not None and candidate is not current:
                        current = candidate
                        break
                except Exception:
                    continue
        else:
            if hasattr(current, "__dict__"):
                current = dict(current.__dict__)
                continue
            try:
                return json.loads(str(current))
            except Exception:
                return None


def _parse_openpose_json(keypoints_source) -> Dict[str, Tuple[int, int]]:
    """Parse OpenPose JSON format keypoints with automatic rescaling.
    
    OpenPose JSON format has keypoints as flat array: [x1, y1, conf1, x2, y2, conf2, ...]
    BODY_25 model has 25 keypoints in this order:
    0: Nose
    1: Neck
    2-3: Right Shoulder, Right Elbow
    4: Right Wrist
    5-6: Left Shoulder, Left Elbow
    7: Left Wrist
    8: Mid Hip (computed, often 0,0,0)
    9: Right Hip
    10: Right Knee
    11: Right Ankle
    12: Left Hip
    13: Left Knee
    14: Left Ankle
    15-18: Right Eye, Left Eye, Right Ear, Left Ear
    19-24: BigToe, SmallToe, Heel (for each foot)
    
    Automatically rescales coordinates to VNCCS canvas (512x1536)
    """
    # OpenPose BODY_25 keypoint names in order (index matches BODY_25 spec)
    OPENPOSE_NAMES = [
        "nose", "neck",
        "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist",
        "mid_hip",  # Index 8 - usually computed as average of hips
        "r_hip", "r_knee", "r_ankle",
        "l_hip", "l_knee", "l_ankle",
        "r_eye", "l_eye", "r_ear", "l_ear",
        # Extended keypoints (19-24)
        "l_bigtoe", "l_smalltoe", "l_heel",
        "r_bigtoe", "r_smalltoe", "r_heel"
    ]
    
    # Alternative 18-point format (OpenPose BODY_18 classic)
    CUSTOM_18_NAMES = [
        "nose",           # 0 (1 in 1-indexed)
        "neck",           # 1 (2 in 1-indexed)
        "r_shoulder",     # 2 (3 in 1-indexed)
        "r_elbow",        # 3 (4 in 1-indexed)
        "r_wrist",        # 4 (5 in 1-indexed)
        "l_shoulder",     # 5 (6 in 1-indexed)
        "l_elbow",        # 6 (7 in 1-indexed)
        "l_wrist",        # 7 (8 in 1-indexed)
        "r_hip",          # 8 (9 in 1-indexed)
        "r_knee",         # 9 (10 in 1-indexed)
        "r_ankle",        # 10 (11 in 1-indexed)
        "l_hip",          # 11 (12 in 1-indexed)
        "l_knee",         # 12 (13 in 1-indexed)
        "l_ankle",        # 13 (14 in 1-indexed)
        "r_eye",          # 14 (15 in 1-indexed)
        "l_eye",          # 15 (16 in 1-indexed)
        "r_ear",          # 16 (17 in 1-indexed)
        "l_ear"           # 17 (18 in 1-indexed)
    ]
    
    data = _unwrap_openpose_payload(keypoints_source)
    if data is None:
        return {}
    
    joints = {}
    
    # Auto-detect source canvas dimensions
    source_width = None
    source_height = None
    
    alias_map = {
        "right_shoulder": "r_shoulder", "right_elbow": "r_elbow", "right_wrist": "r_wrist",
        "left_shoulder": "l_shoulder", "left_elbow": "l_elbow", "left_wrist": "l_wrist",
        "right_hip": "r_hip", "right_knee": "r_knee", "right_ankle": "r_ankle",
        "left_hip": "l_hip", "left_knee": "l_knee", "left_ankle": "l_ankle",
        "right_eye": "r_eye", "left_eye": "l_eye", "right_ear": "r_ear", "left_ear": "l_ear",
    }

    # Handle different OpenPose JSON formats
    if isinstance(data, list):
        # Array format with multiple frames - take first
        if len(data) > 0:
            data = data[0]
    
    direct_positions = None

    if isinstance(data, dict):
        # Extract canvas dimensions if available
        source_width = data.get("canvas_width")
        source_height = data.get("canvas_height")

        # Handle dict of joint names -> coordinates directly
        direct_joints = {}
        for raw_name, raw_value in data.items():
            mapped_name = alias_map.get(raw_name, raw_name)
            if mapped_name in OPENPOSE_NAMES:
                if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                    try:
                        x = float(raw_value[0])
                        y = float(raw_value[1])
                        direct_joints[mapped_name] = (x, y)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(raw_value, dict):
                    try:
                        x = float(raw_value.get("x", 0))
                        y = float(raw_value.get("y", 0))
                        direct_joints[mapped_name] = (x, y)
                    except (ValueError, TypeError):
                        continue
        if direct_joints:
            direct_positions = direct_joints
        
        # Format 1: {keypoints: [x, y, conf, ...]} or {people: [{pose_keypoints_2d: [...]}]}
        if "people" in data and isinstance(data["people"], list) and len(data["people"]) > 0:
            person = data["people"][0]
            if "pose_keypoints_2d" in person:
                keypoints = person["pose_keypoints_2d"]
            else:
                keypoints = []
        elif "json" in data and isinstance(data["json"], (str, bytes)):
            return _parse_openpose_json(data["json"])
        elif "keypoints" in data:
            keypoints = data["keypoints"]
        elif "points" in data:
            keypoints = data["points"]
        elif "pose_keypoints_2d" in data:
            keypoints = data["pose_keypoints_2d"]
        elif "data" in data:
            return _parse_openpose_json(data["data"])
        else:
            # Try to extract from flat structure
            keypoints = data.get("body_keypoints", [])
    elif isinstance(data, list):
        # Format 2: Direct array [x, y, conf, ...]
        keypoints = data
    else:
        print("[VNCCS] WARNING: Unknown keypoints format")
        return {}
    
    # Calculate scale factors if source dimensions differ from target
    scale_x = 1.0
    scale_y = 1.0
    if source_width and source_height:
        if source_width != CANVAS_WIDTH or source_height != CANVAS_HEIGHT:
            scale_x = CANVAS_WIDTH / source_width
            scale_y = CANVAS_HEIGHT / source_height
            print(f"[VNCCS] Auto-rescaling keypoints from {source_width}x{source_height} to {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
            print(f"[VNCCS] Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")

    if direct_positions is not None:
        for joint_name, (x_raw, y_raw) in direct_positions.items():
            x = x_raw * scale_x
            y = y_raw * scale_y
            joints[joint_name] = (
                _clamp(int(round(x)), 0, CANVAS_WIDTH - 1),
                _clamp(int(round(y)), 0, CANVAS_HEIGHT - 1),
            )
        print(f"[VNCCS] Parsed {len(joints)} direct keypoints from POSE payload")
        return joints
    
    # Auto-detect format based on keypoint count
    keypoint_count = len(keypoints) // 3
    name_mapping = CUSTOM_18_NAMES if keypoint_count == 18 else OPENPOSE_NAMES
    print(f"[VNCCS] Detected {keypoint_count} keypoints, using {'CUSTOM_18' if keypoint_count == 18 else 'BODY_25'} format")

    # Parse keypoints array (every 3 elements: x, y, confidence)
    for i in range(min(len(name_mapping), len(keypoints) // 3)):
        try:
            x_raw = keypoints[i * 3]
            y_raw = keypoints[i * 3 + 1]
            conf = keypoints[i * 3 + 2] if i * 3 + 2 < len(keypoints) else 0
        except (IndexError, TypeError):
            break
        
        # Only use keypoints with confidence > 0
        if conf > 0 and x_raw is not None and y_raw is not None:
            try:
                x = float(x_raw) * scale_x
                y = float(y_raw) * scale_y
            except (ValueError, TypeError):
                continue
            joint_name = name_mapping[i]
            x_int = _clamp(int(round(x)), 0, CANVAS_WIDTH - 1)
            y_int = _clamp(int(round(y)), 0, CANVAS_HEIGHT - 1)
            joints[joint_name] = (x_int, y_int)
    
    print(f"[VNCCS] Parsed {len(joints)} keypoints from OpenPose JSON")
    return joints


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
        return {
            "required": {
                "pose_data": ("STRING", {
                    "default": json.dumps({
                        "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
                        "joints": {name: list(pos) for name, pos in DEFAULT_SKELETON.items()}
                    }, indent=2),
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
                "show_joints": ("BOOLEAN", {
                    "default": True
                }),
                "show_body_parts": ("BOOLEAN", {
                    "default": True
                }),
            },
            "optional": {
                "json_keypoints": ("POSE_KEYPOINT",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("schematic", "openpose", "depth", "normal", "canny")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/pose"
    
    def generate(self, pose_data: str, line_thickness: int = 3, 
                show_joints: bool = True, show_body_parts: bool = True,
                json_keypoints = None):
        """Generate schematic, OpenPose, depth, normal, and canny images from pose data
        
        Args:
            pose_data: JSON string containing joint coordinates
            line_thickness: Thickness of lines in OpenPose output
            show_joints: Show joint circles in schematic
            show_body_parts: Show body part ovals in schematic
            json_keypoints: Optional POSE_KEYPOINT data from OpenPose detector
        
        Returns:
            Tuple of (schematic, openpose, depth, normal, canny) in ComfyUI format
        """
        print(f"[VNCCS] Generating pose images...")
        
        # First, load base pose from pose_data widget
        try:
            data = json.loads(pose_data)
        except json.JSONDecodeError as exc:
            print(f"[VNCCS] ERROR: Invalid JSON in pose_data: {exc}")
            joints = DEFAULT_SKELETON.copy()
        else:
            joints_payload = data.get("joints", {}) if isinstance(data, dict) else {}
            if not isinstance(joints_payload, dict):
                print("[VNCCS] WARNING: joints payload is not a dict, reverting to default pose")
                joints = DEFAULT_SKELETON.copy()
            else:
                joints = _sanitize_joints(joints_payload)
                print(f"[VNCCS] Loaded {len(joints)} joints from pose_data")
        
        # If json_keypoints is provided, override specific joints
        if json_keypoints is not None:
            print(f"[VNCCS] Overriding with keypoints from POSE_KEYPOINT input ({type(json_keypoints)})...")
            joints_override = _parse_openpose_json(json_keypoints)
            if joints_override:
                # Override only the joints that came from keypoints
                joints.update(joints_override)
                print(f"[VNCCS] Applied {len(joints_override)} keypoints override")
            else:
                print("[VNCCS] WARNING: Failed to extract keypoints from POSE_KEYPOINT input")
        
        # Get canvas dimensions
        width = CANVAS_WIDTH
        height = CANVAS_HEIGHT
        
        # Render schematic view
        print(f"[VNCCS] Rendering schematic view ({width}x{height})...")
        schematic_img = render_schematic(
            joints, 
            width, 
            height, 
            show_joints=show_joints,
            show_body_parts=show_body_parts
        )
        
        # Convert RGBA to RGB (blend with white background)
        if schematic_img.shape[2] == 4:
            alpha = schematic_img[:, :, 3:4] / 255.0
            rgb = schematic_img[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            schematic_img = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        # Render OpenPose view
        print(f"[VNCCS] Rendering OpenPose view...")
        openpose_img = render_openpose(joints, width, height, line_thickness)
        
        # Render advanced maps (depth, normal, canny)
        print(f"[VNCCS] Rendering depth, normal, and canny maps...")
        advanced_maps = render_all_maps(joints, BONE_CONNECTIONS, width, height, line_thickness)
        depth_img = advanced_maps['depth']
        normal_img = advanced_maps['normal']
        canny_img = advanced_maps['canny']
        
        # Convert to ComfyUI format [B, H, W, C] with values in [0, 1]
        schematic_tensor = torch.from_numpy(convert_to_comfyui_format(schematic_img))
        openpose_tensor = torch.from_numpy(convert_to_comfyui_format(openpose_img))
        depth_tensor = torch.from_numpy(convert_to_comfyui_format(depth_img))
        normal_tensor = torch.from_numpy(convert_to_comfyui_format(normal_img))
        canny_tensor = torch.from_numpy(convert_to_comfyui_format(canny_img))
        
        print(f"[VNCCS] Generated images:")
        print(f"  Schematic: {schematic_tensor.shape}")
        print(f"  OpenPose: {openpose_tensor.shape}")
        print(f"  Depth: {depth_tensor.shape}")
        print(f"  Normal: {normal_tensor.shape}")
        print(f"  Canny: {canny_tensor.shape}")
        
        return (schematic_tensor, openpose_tensor, depth_tensor, normal_tensor, canny_tensor)


NODE_CLASS_MAPPINGS = {
    "VNCCS_PoseGenerator": VNCCS_PoseGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PoseGenerator": "VNCCS Pose Generator",
}
