"""Pose utilities for VNCCS Pose Generator."""

from .skeleton_512x1536 import (
    Skeleton,
    DEFAULT_SKELETON,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    BONE_CONNECTIONS,
    BODY_PARTS,
    LEGACY_JOINT_ALIASES
)

from .pose_renderer import (
    render_schematic,
    render_openpose,
    convert_to_comfyui_format
)

from .advanced_renderer import (
    render_depth_map,
    render_normal_map,
    render_canny_edges,
    render_all_maps
)

from .bone_colors import (
    OPENPOSE_COLORS,
    BONE_COLORS,
    FALLBACK_PALETTE,
    get_bone_color,
    get_bone_color_bgr
)

__all__ = [
    'Skeleton',
    'DEFAULT_SKELETON',
    'CANVAS_WIDTH',
    'CANVAS_HEIGHT',
    'BONE_CONNECTIONS',
    'BODY_PARTS',
    'LEGACY_JOINT_ALIASES',
    'render_schematic',
    'render_openpose',
    'convert_to_comfyui_format',
    'render_depth_map',
    'render_normal_map',
    'render_canny_edges',
    'render_all_maps',
    'OPENPOSE_COLORS',
    'BONE_COLORS',
    'FALLBACK_PALETTE',
    'get_bone_color',
    'get_bone_color_bgr',
]
