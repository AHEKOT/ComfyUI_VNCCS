"""Pose renderer for generating schematic and OpenPose images"""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
import sys
from pathlib import Path

# Import from same directory
try:
    from .skeleton_512x1536 import BODY_PARTS, BONE_CONNECTIONS
    from .bone_colors import get_bone_color_bgr, get_bone_color
except (ImportError, ValueError):
    try:
        from skeleton_512x1536 import BODY_PARTS, BONE_CONNECTIONS
        from bone_colors import get_bone_color_bgr, get_bone_color
    except ImportError:
        # If still fails, define minimal constants
        BODY_PARTS = []
        BONE_CONNECTIONS = []
        # Fallback functions if import fails
        def get_bone_color_bgr(j1, j2, idx=None):
            return (255, 255, 255)  # Default white
        def get_bone_color(j1, j2, idx=None):
            return (255, 255, 255)  # Default white


def as_point(value: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    """Convert a point-like value to an integer tuple for OpenCV"""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = int(round(float(value[0])))
        y = int(round(float(value[1])))
    except (TypeError, ValueError):
        return None
    return (x, y)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def draw_ellipse_between_points(img, pt1, pt2, width, color, alpha=255):
    """Draw an ellipse (oval) between two points"""
    p1 = as_point(pt1)
    p2 = as_point(pt2)
    if p1 is None or p2 is None:
        return

    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate center
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Calculate length and angle
    length = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    # Draw ellipse
    if len(color) == 3:
        color = (*color, alpha)
    
    axes = (width // 2, length // 2)
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)


def render_schematic(joints: Dict[str, Tuple[int, int]], 
                     width: int = 512, 
                     height: int = 1536,
                     show_joints: bool = True,
                     show_body_parts: bool = True) -> np.ndarray:
    """Render schematic view with body parts and joints
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates
        width: Canvas width
        height: Canvas height
        show_joints: Whether to draw joint circles
        show_body_parts: Whether to draw body part ovals
    
    Returns:
        RGBA numpy array [H, W, 4]
    """
    # Create transparent canvas
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Draw body parts (ovals)
    if show_body_parts:
        for part in BODY_PARTS:
            joint_names = part["joints"]
            width_val = part["width"]
            color_hex = part["color"]
            color = hex_to_rgb(color_hex)
            
            if len(joint_names) == 2:
                # Simple two-joint oval
                j1, j2 = joint_names
                if j1 in joints and j2 in joints:
                    pt1 = joints[j1]
                    pt2 = joints[j2]
                    draw_ellipse_between_points(img, pt1, pt2, width_val, color, 220)
            
            elif len(joint_names) == 3:
                # Triangle area (for hips)
                j1, j2, j3 = joint_names
                if j1 in joints and j2 in joints and j3 in joints:
                    # Draw from center to each hip
                    center = joints[j1]
                    hip_r = joints[j2]
                    hip_l = joints[j3]
                    
                    draw_ellipse_between_points(img, center, hip_r, width_val, color, 220)
                    draw_ellipse_between_points(img, center, hip_l, width_val, color, 220)
    
    # Draw bones (lines between joints)
    for joint1, joint2 in BONE_CONNECTIONS:
        if joint1 in joints and joint2 in joints:
            pt1 = as_point(joints[joint1])
            pt2 = as_point(joints[joint2])
            if pt1 is not None and pt2 is not None:
                cv2.line(img, pt1, pt2, (60, 60, 60, 255), 2, cv2.LINE_AA)

    # Draw joints (circles)
    if show_joints:
        for joint_name, (x, y) in joints.items():
            point = as_point((x, y))
            if point is None:
                continue
            cv2.circle(img, point, 6, (255, 100, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(img, point, 6, (180, 50, 50, 255), 1, cv2.LINE_AA)
    
    return img


def render_openpose(joints: Dict[str, Tuple[int, int]], 
                    width: int = 512, 
                    height: int = 1536,
                    line_thickness: int = 3) -> np.ndarray:
    """Render OpenPose format (colored ellipses on black background, matching reference implementation)
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates
        width: Canvas width
        height: Canvas height
        line_thickness: Thickness of bone ellipses (pose_marker_size in reference)
    
    Returns:
        RGB numpy array [H, W, 3]
    """
    # Create black canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw bones as filled ellipses (matching reference implementation)
    for bone_index, (joint1, joint2) in enumerate(BONE_CONNECTIONS):
        if joint1 in joints and joint2 in joints:
            pt1 = as_point(joints[joint1])
            pt2 = as_point(joints[joint2])
            if pt1 is not None and pt2 is not None:
                # Get RGB color and convert to BGR
                color_rgb = get_bone_color(joint1, joint2, bone_index)
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                
                # Calculate ellipse parameters (same as reference)
                Y = np.array([pt1[0], pt2[0]], dtype=float)  # X coordinates
                X = np.array([pt1[1], pt2[1]], dtype=float)  # Y coordinates
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                
                # Draw ellipse (cv2.ellipse2Poly returns polygon points for the ellipse)
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)),           # center
                    (int(length / 2), line_thickness),  # axes (half length, thickness)
                    int(angle),                    # rotation angle
                    0, 360,                        # start/end angles (full ellipse)
                    1                              # number of curve segments
                )
                cv2.fillConvexPoly(img, polygon, color_bgr)
    
    # Reduce canvas opacity to 60% (matching reference)
    img = (img * 0.6).astype(np.uint8)
    
    # Draw joints as circles on top (using same color as first connected bone)
    for joint_name, coords in joints.items():
        point = as_point(coords)
        if point is None:
            continue
        # Find first bone connected to this joint to get its color
        color_rgb = (255, 255, 255)  # Default white
        for bone_index, (joint1, joint2) in enumerate(BONE_CONNECTIONS):
            if joint1 == joint_name or joint2 == joint_name:
                color_rgb = get_bone_color(joint1, joint2, bone_index)
                break
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.circle(img, point, line_thickness, color_bgr, thickness=-1)

    # Convert back to RGB for consumers (ComfyUI expects RGB ordering)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def convert_to_comfyui_format(img: np.ndarray) -> np.ndarray:
    """Convert image to ComfyUI format [B, H, W, C] with values in [0, 1]
    
    Args:
        img: Numpy array in [H, W, C] format with values in [0, 255]
    
    Returns:
        Tensor in [1, H, W, C] format with values in [0, 1]
    """
    # Normalize to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_float, axis=0)
    
    return img_batch
