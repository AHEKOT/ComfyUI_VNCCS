"""Skeleton model for 512x1536 pose generator

Defines the structure of human skeleton with joints, bones, and body parts.
Optimized for portrait orientation with 85% canvas usage.
"""

import json
from typing import Dict, List, Tuple, Optional

# Canvas configuration
CANVAS_WIDTH = 512
CANVAS_HEIGHT = 1536
PERSON_HEIGHT = int(CANVAS_HEIGHT * 0.85)  # 1305px
PERSON_WIDTH = int(CANVAS_WIDTH * 0.85)    # 435px
OFFSET_X = (CANVAS_WIDTH - PERSON_WIDTH) // 2   # 38px
OFFSET_Y = (CANVAS_HEIGHT - PERSON_HEIGHT) // 2  # 115px

# Default skeleton pose (standing, back view)
DEFAULT_SKELETON = {
    # BODY_18 format (18 keypoints without toe/heel details)
    "nose": (256, 200),          # 0
    "neck": (256, 280),          # 1
    "r_shoulder": (320, 320),    # 2
    "r_elbow": (350, 520),       # 3
    "r_wrist": (360, 720),       # 4
    "l_shoulder": (192, 320),    # 5
    "l_elbow": (162, 520),       # 6
    "l_wrist": (152, 720),       # 7
    "r_hip": (290, 720),         # 8
    "r_knee": (295, 1020),      # 9
    "r_ankle": (300, 1320),     # 10
    "l_hip": (222, 720),        # 11
    "l_knee": (217, 1020),      # 12
    "l_ankle": (212, 1320),     # 13
    "r_eye": (270, 185),        # 14
    "l_eye": (242, 185),        # 15
    "r_ear": (285, 195),        # 16
    "l_ear": (227, 195),        # 17
}

LEGACY_JOINT_ALIASES = {}

# Bone connections (for drawing lines)
BONE_CONNECTIONS = [
    # Match the web widget ordering exactly so preview and exports stay in sync
    # Head and face
    ("nose", "neck"),
    ("nose", "r_eye"),
    ("r_eye", "r_ear"),
    ("nose", "l_eye"),
    ("l_eye", "l_ear"),

    # Shoulder line and torso sides
    ("r_shoulder", "l_shoulder"),
    ("neck", "r_hip"),
    ("neck", "l_hip"),

    # Arms
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),

    # Legs
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
]

# Body parts for oval rendering (schematic view)
BODY_PARTS = [
    {
        "name": "head",
        "joints": ["nose", "neck"],
        "width": 110,
        "color": "#FFE5D9"
    },
    {
        "name": "torso_right",
        "joints": ["neck", "r_hip"],
        "width": 165,
        "color": "#FFE5D9"
    },
    {
        "name": "torso_left",
        "joints": ["neck", "l_hip"],
        "width": 165,
        "color": "#FFE5D9"
    },
    {
        "name": "hip_band",
        "joints": ["r_hip", "l_hip"],
        "width": 190,
        "color": "#FFE5D9"
    },
    {
        "name": "r_upper_arm",
        "joints": ["r_shoulder", "r_elbow"],
        "width": 55,
        "color": "#FFE5D9"
    },
    {
        "name": "r_forearm",
        "joints": ["r_elbow", "r_wrist"],
        "width": 45,
        "color": "#FFE5D9"
    },
    {
        "name": "l_upper_arm",
        "joints": ["l_shoulder", "l_elbow"],
        "width": 55,
        "color": "#FFE5D9"
    },
    {
        "name": "l_forearm",
        "joints": ["l_elbow", "l_wrist"],
        "width": 45,
        "color": "#FFE5D9"
    },
    {
        "name": "r_thigh",
        "joints": ["r_hip", "r_knee"],
        "width": 95,
        "color": "#FFE5D9"
    },
    {
        "name": "r_calf",
        "joints": ["r_knee", "r_ankle"],
        "width": 75,
        "color": "#FFE5D9"
    },
    {
        "name": "l_thigh",
        "joints": ["l_hip", "l_knee"],
        "width": 95,
        "color": "#FFE5D9"
    },
    {
        "name": "l_calf",
        "joints": ["l_knee", "l_ankle"],
        "width": 75,
        "color": "#FFE5D9"
    }
]

# Joint hierarchy for IK (parent -> children)
JOINT_HIERARCHY = {
    "nose": [],
    "neck": ["nose", "r_shoulder", "l_shoulder", "r_hip", "l_hip"],
    "r_shoulder": ["r_elbow"],
    "r_elbow": ["r_wrist"],
    "r_wrist": [],
    "l_shoulder": ["l_elbow"],
    "l_elbow": ["l_wrist"],
    "l_wrist": [],
    "r_hip": ["r_knee"],
    "r_knee": ["r_ankle"],
    "r_ankle": ["r_bigtoe", "r_smalltoe", "r_heel"],
    "l_hip": ["l_knee"],
    "l_knee": ["l_ankle"],
    "l_ankle": ["l_bigtoe", "l_smalltoe", "l_heel"],
}


class Skeleton:
    """Skeleton model with joints and bones"""
    
    def __init__(self, joints: Optional[Dict[str, Tuple[int, int]]] = None):
        """Initialize skeleton with default or custom joints"""
        self.joints = joints if joints else DEFAULT_SKELETON.copy()
        self.selected_joint = None
    
    def get_joint(self, name: str) -> Tuple[int, int]:
        """Get joint coordinates by name"""
        return self.joints.get(name, (0, 0))
    
    def set_joint(self, name: str, x: int, y: int):
        """Set joint coordinates"""
        if name in self.joints:
            self.joints[name] = (x, y)
    
    def move_joint(self, name: str, dx: int, dy: int):
        """Move joint by delta"""
        if name in self.joints:
            x, y = self.joints[name]
            self.joints[name] = (x + dx, y + dy)
    
    def to_json(self) -> str:
        """Export skeleton to JSON string"""
        data = {
            "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
            "joints": {name: list(pos) for name, pos in self.joints.items()}
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Skeleton':
        """Create skeleton from JSON string"""
        data = json.loads(json_str)
        joints = {name: tuple(pos) for name, pos in data.get("joints", {}).items()}
        return cls(joints)
    
    def reset(self):
        """Reset to default pose"""
        self.joints = DEFAULT_SKELETON.copy()
    
    def get_bone_connections(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get list of bone connections as coordinate pairs"""
        connections = []
        for joint1, joint2 in BONE_CONNECTIONS:
            if joint1 in self.joints and joint2 in self.joints:
                connections.append((self.joints[joint1], self.joints[joint2]))
        return connections
    
    def get_body_parts(self) -> List[Dict]:
        """Get body parts with joint coordinates"""
        parts = []
        for part in BODY_PARTS:
            part_data = part.copy()
            part_data["joint_coords"] = [self.joints.get(j, (0, 0)) for j in part["joints"]]
            parts.append(part_data)
        return parts
