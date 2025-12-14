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

# FALLBACK_PALETTE is used when direct name lookup fails.
# It maps bone index (position in BONE_CONNECTIONS) to color index in OPENPOSE_COLORS.
# MUST match JavaScript FALLBACK_PALETTE exactly.
FALLBACK_PALETTE = [
    OPENPOSE_COLORS[12],  # 0: nose->neck (Blue)
    OPENPOSE_COLORS[5],   # 1: neck->r_shoulder (Light green) - SWAPPED
    OPENPOSE_COLORS[6],   # 2: r_shoulder->r_elbow (Green) - SWAPPED
    OPENPOSE_COLORS[7],   # 3: r_elbow->r_wrist (Green-cyan) - SWAPPED
    OPENPOSE_COLORS[1],   # 4: neck->l_shoulder (Orange) - SWAPPED
    OPENPOSE_COLORS[2],   # 5: l_shoulder->l_elbow (Dark orange) - SWAPPED
    OPENPOSE_COLORS[3],   # 6: l_elbow->l_wrist (Yellow) - SWAPPED
    OPENPOSE_COLORS[11],  # 7: neck->r_hip (Light blue) - SWAPPED
    OPENPOSE_COLORS[8],   # 8: neck->l_hip (Cyan-green) - SWAPPED
    OPENPOSE_COLORS[12],  # 9: r_hip->r_knee (Blue) - SWAPPED
    OPENPOSE_COLORS[13],  # 10: r_knee->r_ankle (Purple-blue) - SWAPPED
    OPENPOSE_COLORS[9],   # 11: l_hip->l_knee (Cyan) - SWAPPED
    OPENPOSE_COLORS[10],  # 12: l_knee->l_ankle (Cyan-blue) - SWAPPED
    OPENPOSE_COLORS[14],  # 13: nose->r_eye (Purple)
    OPENPOSE_COLORS[16],  # 14: r_eye->r_ear (Pink)
    OPENPOSE_COLORS[14],  # 15: nose->l_eye (Purple)
    OPENPOSE_COLORS[16],  # 16: l_eye->l_ear (Pink)
]

# Color palette for bones (matching BONE_CONNECTIONS / widget order)
BONE_COLORS = {
    # Upper body
    ("nose", "neck"): OPENPOSE_COLORS[12],
    ("neck", "r_shoulder"): OPENPOSE_COLORS[5],   # Light green - SWAPPED
    ("r_shoulder", "r_elbow"): OPENPOSE_COLORS[6],  # Green - SWAPPED
    ("r_elbow", "r_wrist"): OPENPOSE_COLORS[7],     # Green-cyan - SWAPPED
    ("neck", "l_shoulder"): OPENPOSE_COLORS[1],     # Orange - SWAPPED
    ("l_shoulder", "l_elbow"): OPENPOSE_COLORS[2],  # Dark orange - SWAPPED
    ("l_elbow", "l_wrist"): OPENPOSE_COLORS[3],     # Yellow - SWAPPED
    ("neck", "r_hip"): OPENPOSE_COLORS[11],         # Light blue - SWAPPED
    ("neck", "l_hip"): OPENPOSE_COLORS[8],          # Cyan-green - SWAPPED

    # Right leg
    ("r_hip", "r_knee"): OPENPOSE_COLORS[12],       # Blue - SWAPPED
    ("r_knee", "r_ankle"): OPENPOSE_COLORS[13],     # Purple-blue - SWAPPED

    # Left leg
    ("l_hip", "l_knee"): OPENPOSE_COLORS[9],        # Cyan - SWAPPED
    ("l_knee", "l_ankle"): OPENPOSE_COLORS[10],     # Cyan-blue - SWAPPED
    
    # Face
    ("nose", "r_eye"): OPENPOSE_COLORS[14],
    ("r_eye", "r_ear"): OPENPOSE_COLORS[16],
    ("nose", "l_eye"): OPENPOSE_COLORS[14],
    ("l_eye", "l_ear"): OPENPOSE_COLORS[16],
}

# Joint colors - average of all connected bones or use primary bone color
JOINT_COLORS = {
    # Head
    "nose": OPENPOSE_COLORS[12],       # Blue (nose-neck)
    "neck": OPENPOSE_COLORS[12],       # Blue
    "r_eye": OPENPOSE_COLORS[14],      # Purple
    "l_eye": OPENPOSE_COLORS[14],      # Purple
    "r_ear": OPENPOSE_COLORS[16],      # Pink
    "l_ear": OPENPOSE_COLORS[16],      # Pink
    
    # Right arm (GREEN)
    "r_shoulder": OPENPOSE_COLORS[5],  # Light green
    "r_elbow": OPENPOSE_COLORS[6],     # Green
    "r_wrist": OPENPOSE_COLORS[7],     # Green-cyan
    
    # Left arm (RED-YELLOW)
    "l_shoulder": OPENPOSE_COLORS[1],  # Orange
    "l_elbow": OPENPOSE_COLORS[2],     # Dark orange
    "l_wrist": OPENPOSE_COLORS[3],     # Yellow
    
    # Right leg (BLUE)
    "r_hip": OPENPOSE_COLORS[11],      # Light blue
    "r_knee": OPENPOSE_COLORS[12],     # Blue
    "r_ankle": OPENPOSE_COLORS[13],    # Purple-blue
    
    # Left leg (CYAN)
    "l_hip": OPENPOSE_COLORS[8],       # Cyan-green
    "l_knee": OPENPOSE_COLORS[9],      # Cyan
    "l_ankle": OPENPOSE_COLORS[10],    # Cyan-blue
}

def get_joint_color(joint_name):
    """Get RGB color for a joint.
    
    Args:
        joint_name: Name of the joint
        
    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    return JOINT_COLORS.get(joint_name, (255, 255, 255))

# Fallback palette is unreliable because BONE_CONNECTIONS order varies.
# We will rely on BONE_COLORS lookup.

def get_bone_color(joint1, joint2, bone_index=None):
    """Get RGB color for a bone connection.
    
    Args:
        joint1: Name of first joint
        joint2: Name of second joint
        bone_index: Index in BONE_CONNECTIONS (primary source)
        
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
    
    # Default to white if not found
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
