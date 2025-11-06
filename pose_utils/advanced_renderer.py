"""Advanced pose rendering - depth, normal, and canny edge detection"""

import cv2
import numpy as np
from typing import Dict, Tuple

def render_depth_map(joints: Dict[str, Tuple[int, int]], 
                     width: int = 512, 
                     height: int = 1536,
                     near: float = 0.0,
                     far: float = 1000.0) -> np.ndarray:
    """Render depth map based on Z-coordinates (simulated from Y position)
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates
        width: Image width
        height: Image height
        near: Near plane distance
        far: Far plane distance
    
    Returns:
        RGB numpy array [H, W, 3] with depth visualization
    """
    # Create black image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Simulate depth based on Y coordinate (higher = further)
    # This is a simplified approach since we don't have real 3D data
    for joint_name, (x, y) in joints.items():
        if not (0 <= x < width and 0 <= y < height):
            continue
        
        # Normalize Y to depth (0-1 range, inverted so top is far)
        normalized_depth = y / height
        depth_value = int((1.0 - normalized_depth) * 255)
        
        # Draw circle at joint position
        cv2.circle(img, (x, y), 8, (depth_value, depth_value, depth_value), -1, cv2.LINE_AA)
    
    # Optional: Apply Gaussian blur for smoother depth
    img = cv2.GaussianBlur(img, (21, 21), 0)
    
    return img


def render_normal_map(joints: Dict[str, Tuple[int, int]], 
                      width: int = 512, 
                      height: int = 1536) -> np.ndarray:
    """Render normal map (surface orientation as RGB)
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates
        width: Image width
        height: Image height
    
    Returns:
        RGB numpy array [H, W, 3] with normal visualization
    """
    # Create image with neutral normal (facing camera)
    # Neutral normal is (0, 0, 1) which maps to RGB (128, 128, 255)
    img = np.full((height, width, 3), [128, 128, 255], dtype=np.uint8)
    
    # For each joint, simulate normals based on position
    for joint_name, (x, y) in joints.items():
        if not (0 <= x < width and 0 <= y < height):
            continue
        
        # Calculate normal direction based on position relative to center
        center_x = width / 2
        center_y = height / 2
        
        # Normal components
        nx = (x - center_x) / center_x  # -1 to 1
        ny = (y - center_y) / center_y  # -1 to 1
        nz = 0.7  # Mostly facing forward
        
        # Normalize
        length = np.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            nx /= length
            ny /= length
            nz /= length
        
        # Convert to RGB (map -1,1 to 0,255)
        r = int((nx * 0.5 + 0.5) * 255)
        g = int((ny * 0.5 + 0.5) * 255)
        b = int((nz * 0.5 + 0.5) * 255)
        
        # Draw circle at joint position
        cv2.circle(img, (x, y), 10, (r, g, b), -1, cv2.LINE_AA)
    
    # Apply slight blur for smoother normals
    img = cv2.GaussianBlur(img, (11, 11), 0)
    
    return img


def render_canny_edges(joints: Dict[str, Tuple[int, int]], 
                       connections: list,
                       width: int = 512, 
                       height: int = 1536,
                       line_thickness: int = 3,
                       low_threshold: int = 50,
                       high_threshold: int = 150) -> np.ndarray:
    """Render canny edge detection from pose skeleton
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates
        connections: List of (joint1, joint2) bone connections
        width: Image width
        height: Image height
        line_thickness: Thickness of skeleton lines
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
    
    Returns:
        RGB numpy array [H, W, 3] with white edges on black background
    """
    # First render basic skeleton
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Draw bones
    for joint1_name, joint2_name in connections:
        if joint1_name not in joints or joint2_name not in joints:
            continue
        
        pt1 = joints[joint1_name]
        pt2 = joints[joint2_name]
        
        if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
            0 <= pt2[0] < width and 0 <= pt2[1] < height):
            cv2.line(img, pt1, pt2, 255, line_thickness, cv2.LINE_AA)
    
    # Draw joints
    for x, y in joints.values():
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 5, 255, -1, cv2.LINE_AA)
    
    # Apply Gaussian blur before edge detection
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_blur, low_threshold, high_threshold)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return img_rgb


def render_all_maps(joints: Dict[str, Tuple[int, int]],
                    connections: list,
                    width: int = 512,
                    height: int = 1536,
                    line_thickness: int = 3) -> Dict[str, np.ndarray]:
    """Render all map types at once
    
    Args:
        joints: Dictionary of joint names to (x, y) coordinates  
        connections: List of (joint1, joint2) bone connections
        width: Image width
        height: Image height
        line_thickness: Line thickness for rendering
    
    Returns:
        Dictionary with keys 'depth', 'normal', 'canny' mapping to RGB arrays
    """
    return {
        'depth': render_depth_map(joints, width, height),
        'normal': render_normal_map(joints, width, height),
        'canny': render_canny_edges(joints, connections, width, height, line_thickness)
    }
