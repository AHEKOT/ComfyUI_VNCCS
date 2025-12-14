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
    OPENPOSE_COLORS[1],   # 1: neck->r_shoulder (Orange) - RIGHT is orange/warm
    OPENPOSE_COLORS[2],   # 2: r_shoulder->r_elbow (Dark orange) - RIGHT
    OPENPOSE_COLORS[3],   # 3: r_elbow->r_wrist (Yellow) - RIGHT
    OPENPOSE_COLORS[5],   # 4: neck->l_shoulder (Light green) - LEFT is cool
    OPENPOSE_COLORS[6],   # 5: l_shoulder->l_elbow (Green) - LEFT
    OPENPOSE_COLORS[7],   # 6: l_elbow->l_wrist (Green-cyan) - LEFT
    OPENPOSE_COLORS[8],   # 7: neck->r_hip (Cyan-green) - RIGHT hip
    OPENPOSE_COLORS[11],  # 8: neck->l_hip (Light blue) - LEFT hip
    OPENPOSE_COLORS[5],   # 9: r_hip->r_knee (Light green) - RIGHT leg
    OPENPOSE_COLORS[6],   # 10: r_knee->r_ankle (Green) - RIGHT leg
    OPENPOSE_COLORS[9],   # 11: l_hip->l_knee (Cyan) - LEFT leg
    OPENPOSE_COLORS[10],  # 12: l_knee->l_ankle (Cyan-blue) - LEFT leg
    OPENPOSE_COLORS[14],  # 13: nose->r_eye (Purple)
    OPENPOSE_COLORS[16],  # 14: r_eye->r_ear (Pink)
    OPENPOSE_COLORS[14],  # 15: nose->l_eye (Purple)
    OPENPOSE_COLORS[16],  # 16: l_eye->l_ear (Pink)
]

# Color palette for bones (matching BONE_CONNECTIONS / widget order)
BONE_COLORS = {
    # Upper body
    ("nose", "neck"): OPENPOSE_COLORS[12],
    ("neck", "r_shoulder"): OPENPOSE_COLORS[1],   # Orange - RIGHT is orange/warm
    ("r_shoulder", "r_elbow"): OPENPOSE_COLORS[2],  # Dark orange - RIGHT
    ("r_elbow", "r_wrist"): OPENPOSE_COLORS[3],     # Yellow - RIGHT
    ("neck", "l_shoulder"): OPENPOSE_COLORS[5],     # Light green - LEFT is cool
    ("l_shoulder", "l_elbow"): OPENPOSE_COLORS[6],  # Green - LEFT
    ("l_elbow", "l_wrist"): OPENPOSE_COLORS[7],     # Green-cyan - LEFT
    ("neck", "r_hip"): OPENPOSE_COLORS[8],         # Cyan-green - RIGHT hip
    ("neck", "l_hip"): OPENPOSE_COLORS[11],          # Light blue - LEFT hip

    # Right leg (GREEN)
    ("r_hip", "r_knee"): OPENPOSE_COLORS[5],       # Light green - RIGHT leg
    ("r_knee", "r_ankle"): OPENPOSE_COLORS[6],     # Green - RIGHT leg

    # Left leg (CYAN-BLUE)
    ("l_hip", "l_knee"): OPENPOSE_COLORS[9],        # Cyan - LEFT leg
    ("l_knee", "l_ankle"): OPENPOSE_COLORS[10],     # Cyan-blue - LEFT leg
    
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
    
    # Right arm (ORANGE-YELLOW) - warm colors
    "r_shoulder": OPENPOSE_COLORS[1],  # Orange
    "r_elbow": OPENPOSE_COLORS[2],     # Dark orange
    "r_wrist": OPENPOSE_COLORS[3],     # Yellow
    
    # Left arm (GREEN) - cool colors
    "l_shoulder": OPENPOSE_COLORS[5],  # Light green
    "l_elbow": OPENPOSE_COLORS[6],     # Green
    "l_wrist": OPENPOSE_COLORS[7],     # Green-cyan
    
    # Right leg (GREEN)
    "r_hip": OPENPOSE_COLORS[8],       # Cyan-green
    "r_knee": OPENPOSE_COLORS[5],      # Light green
    "r_ankle": OPENPOSE_COLORS[6],     # Green
    
    # Left leg (CYAN-BLUE)
    "l_hip": OPENPOSE_COLORS[11],      # Light blue
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
