"""VNCCS Character Debugger Node

A clone of Pose Generator used for debugging skeleton mapping between OpenPose and MakeHuman.
"""

import json
import os
import sys
import numpy as np
import torch
from .pose_generator import VNCCS_PoseGenerator, DEFAULT_SKELETON, CANVAS_WIDTH, CANVAS_HEIGHT

class VNCCS_CharacterDebugger(VNCCS_PoseGenerator):
    """Debug version of Pose Generator to align skeletons"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # reuse standard inputs
        types = super().INPUT_TYPES()
        return types
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("openpose_grid",)
    FUNCTION = "generate"
    CATEGORY = "VNCCS/debug"
    
    # We reuse the generate method from parent

NODE_CLASS_MAPPINGS = {
    "VNCCS_CharacterDebugger": VNCCS_CharacterDebugger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_CharacterDebugger": "VNCCS Character Debugger",
}
