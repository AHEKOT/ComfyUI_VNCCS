"""Bone color configuration for Python side"""

# Color palette shared between web widget and Python renderer.
# Ordering matches `BONE_CONNECTIONS` defined in skeleton_512x1536.py and
# the JavaScript editor so preview and export use identical colors.

# OpenPose standard color palette (18 colors from reference implementation)
# RGB format for Python/OpenCV (note: OpenCV uses BGR, so colors are in RGB here and converted when drawing)
OPENPOSE_COLORS = [
    (255, 0, 0),      # 0: Red
    (255, 85, 0),     # 1: Orange
    (255, 170, 0),    # 2: Dark orange
    (255, 255, 0),    # 3: Yellow
    (170, 255, 0),    # 4: Yellow-green
    (85, 255, 0),     # 5: Light green
    (0, 255, 0),      # 6: Green
    (0, 255, 85),     # 7: Green-cyan
    (0, 255, 170),    # 8: Cyan-green
    (0, 255, 255),    # 9: Cyan
    (0, 170, 255),    # 10: Cyan-blue
    (0, 85, 255),     # 11: Light blue
    (0, 0, 255),      # 12: Blue
    (85, 0, 255),     # 13: Purple-blue
    (170, 0, 255),    # 14: Purple
    (255, 0, 255),    # 15: Magenta
    (255, 0, 170),    # 16: Pink
    (255, 0, 85),     # 17: Hot pink
]

# Color palette for bones (matching BONE_CONNECTIONS / widget order)
BONE_COLORS = {
    # Head and face (0-4)
    ("nose", "neck"): OPENPOSE_COLORS[12],         # 0: Blue
    ("nose", "r_eye"): OPENPOSE_COLORS[13],        # 1: Purple-blue
    ("r_eye", "r_ear"): OPENPOSE_COLORS[14],       # 2: Purple
    ("nose", "l_eye"): OPENPOSE_COLORS[15],        # 3: Magenta
    ("l_eye", "l_ear"): OPENPOSE_COLORS[16],       # 4: Pink

    # Shoulder line and torso (5-7)
    ("r_shoulder", "l_shoulder"): OPENPOSE_COLORS[0],  # 5: Red
    ("neck", "r_hip"): OPENPOSE_COLORS[6],             # 6: Green
    ("neck", "l_hip"): OPENPOSE_COLORS[9],             # 7: Cyan

    # Right arm (8-9)
    ("r_shoulder", "r_elbow"): OPENPOSE_COLORS[1],     # 8: Orange
    ("r_elbow", "r_wrist"): OPENPOSE_COLORS[2],        # 9: Dark orange

    # Left arm (10-11)
    ("l_shoulder", "l_elbow"): OPENPOSE_COLORS[3],     # 10: Yellow
    ("l_elbow", "l_wrist"): OPENPOSE_COLORS[4],        # 11: Yellow-green

    # Right leg (12-13)
    ("r_hip", "r_knee"): OPENPOSE_COLORS[7],          # 12: Green-cyan
    ("r_knee", "r_ankle"): OPENPOSE_COLORS[8],        # 13: Cyan-green

    # Left leg (14-15)
    ("l_hip", "l_knee"): OPENPOSE_COLORS[10],         # 14: Cyan-blue
    ("l_knee", "l_ankle"): OPENPOSE_COLORS[11],       # 15: Light blue
}

# Fallback color palette - matches BONE_CONNECTIONS order exactly
FALLBACK_PALETTE = [
    OPENPOSE_COLORS[12],  # 0: nose->neck (Blue)
    OPENPOSE_COLORS[13],  # 1: nose->r_eye (Purple-blue)
    OPENPOSE_COLORS[14],  # 2: r_eye->r_ear (Purple)
    OPENPOSE_COLORS[15],  # 3: nose->l_eye (Magenta)
    OPENPOSE_COLORS[16],  # 4: l_eye->l_ear (Pink)
    OPENPOSE_COLORS[0],   # 5: r_shoulder->l_shoulder (Red)
    OPENPOSE_COLORS[6],   # 6: neck->r_hip (Green)
    OPENPOSE_COLORS[9],   # 7: neck->l_hip (Cyan)
    OPENPOSE_COLORS[1],   # 8: r_shoulder->r_elbow (Orange)
    OPENPOSE_COLORS[2],   # 9: r_elbow->r_wrist (Dark orange)
    OPENPOSE_COLORS[3],   # 10: l_shoulder->l_elbow (Yellow)
    OPENPOSE_COLORS[4],   # 11: l_elbow->l_wrist (Yellow-green)
    OPENPOSE_COLORS[7],   # 12: r_hip->r_knee (Green-cyan)
    OPENPOSE_COLORS[8],   # 13: r_knee->r_ankle (Cyan-green)
    OPENPOSE_COLORS[10],  # 14: l_hip->l_knee (Cyan-blue)
    OPENPOSE_COLORS[11],  # 15: l_knee->l_ankle (Light blue)
]


def get_bone_color(joint1, joint2, bone_index=None):
    """Get RGB color for a bone connection.
    
    Args:
        joint1: Name of first joint
        joint2: Name of second joint
        bone_index: Bone index in BONE_CONNECTIONS (primary source)
        
    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    # Primary: use bone_index for consistent palette-based colors
    if bone_index is not None and bone_index < len(FALLBACK_PALETTE):
        return FALLBACK_PALETTE[bone_index]
    
    # Secondary: try direct lookup by joint pair
    key = (joint1, joint2)
    if key in BONE_COLORS:
        return BONE_COLORS[key]
    
    # Tertiary: try reverse lookup
    key_reverse = (joint2, joint1)
    if key_reverse in BONE_COLORS:
        return BONE_COLORS[key_reverse]
    
    # Default white
    return (255, 255, 255)


def get_bone_color_bgr(joint1, joint2, bone_index=None):
    """Get BGR color for a bone connection (for OpenCV).
    
    Args:
        joint1: Name of first joint
        joint2: Name of second joint
        bone_index: Fallback index if direct lookup fails
        
    Returns:
        Tuple of (B, G, R) values (0-255)
    """
    r, g, b = get_bone_color(joint1, joint2, bone_index)
    return (b, g, r)
