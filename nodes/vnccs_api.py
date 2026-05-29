"""
VNCCS Custom API Nodes for ComfyUI

This module implements three custom nodes that serve as injection points for the VNCCS UI:
1. VNCCS_Variable - Universal value injection point with auto-casting
2. VNCCS_ImageList - Converts JSON array of base64 images to ComfyUI IMAGE LIST
3. VNCCS_LoraBypass - LoRA loader with runtime bypass capability

These nodes allow the VNCCS UI to inject values by finding nodes by title
instead of relying on fragile numeric node IDs.
"""

import base64
import io
import json
import re
from typing import Any, Tuple

import torch
import numpy as np
from PIL import Image

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

try:
    from .model_path_utils import get_full_path_agnostic
except Exception:
    from model_path_utils import get_full_path_agnostic


class VNCCS_Variable:
    """
    Universal injection point node. Accepts a string value and auto-casts to int/float/str
    based on content. Output type is ANY (*) to allow connections to any input.

    The VNCCS UI finds this node by _meta.title and overwrites inputs.value before execution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "VNCCS/Variables"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Allow connections to any type."""
        return True

    def execute(self, value: str) -> Tuple[Any]:
        """
        Execute the node: auto-cast value based on content.

        Args:
            value: String input (can be int, float, or string)

        Returns:
            Tuple containing the cast value (int, float, or str)
        """
        # Strip whitespace
        value = value.strip()

        # Try integer cast
        if re.fullmatch(r"[+-]?\d+", value):
            return (int(value),)

        # Try float cast
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            return (float(value),)

        # Return as string
        return (value,)


class VNCCS_ImageList:
    """
    Accepts a JSON array of base64-encoded images and outputs them as a ComfyUI IMAGE LIST.
    Each image is decoded independently and output as a separate tensor in list mode.

    The VNCCS UI injects a JSON array of base64 strings into the 'images' input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)  # Critical: output as LIST, not batch
    INPUT_IS_LIST = False
    FUNCTION = "execute"
    CATEGORY = "VNCCS/Variables"
    OUTPUT_NODE = False

    def execute(self, images: str) -> Tuple[list]:
        """
        Decode JSON array of base64 images to torch tensors.

        Args:
            images: JSON string containing array of base64-encoded images

        Returns:
            Tuple containing list of image tensors (1×H×W×C each)

        Raises:
            ValueError: If JSON is invalid, array is empty, or base64 decode fails
        """
        # Parse JSON
        try:
            image_array = json.loads(images)
        except json.JSONDecodeError as e:
            raise ValueError(f"VNCCS_ImageList: invalid JSON: {e}")

        if not isinstance(image_array, list):
            raise ValueError("VNCCS_ImageList: 'images' must be a JSON array")

        if len(image_array) == 0:
            raise ValueError("VNCCS_ImageList: no images provided")

        # Decode each image
        tensors = []
        for idx, img_data in enumerate(image_array):
            try:
                # Strip data URI prefix if present
                if isinstance(img_data, str):
                    if img_data.startswith("data:"):
                        # Remove "data:image/...;base64," prefix
                        img_data = img_data.split(",", 1)[1] if "," in img_data else img_data

                    # Base64 decode
                    img_bytes = base64.b64decode(img_data)
                else:
                    raise ValueError(f"Image at index {idx} is not a string")

                # Decode to PIL Image
                pil_image = Image.open(io.BytesIO(img_bytes))

                # Convert to RGB if needed (handle RGBA, grayscale, etc.)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # Convert to numpy array (H×W×C, uint8)
                np_image = np.array(pil_image).astype(np.float32) / 255.0

                # Convert to torch tensor and add batch dimension (1×H×W×C)
                tensor = torch.from_numpy(np_image)[None,]

                tensors.append(tensor)

            except Exception as e:
                raise ValueError(f"VNCCS_ImageList: failed to decode image at index {idx}: {e}")

        # Return as list (OUTPUT_IS_LIST=True means each element processes separately)
        return (tensors,)


class VNCCS_LoraBypass:
    """
    LoRA loader with runtime bypass capability. When lora_name is "None" or empty,
    passes MODEL and CLIP through unchanged. Otherwise loads and applies the LoRA.

    This eliminates the need to delete/rewire nodes when toggling LoRA usage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING", {
                    "default": "None",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "execute"
    CATEGORY = "VNCCS/Variables"
    OUTPUT_NODE = False

    def execute(self, model, clip, lora_name: str, strength_model: float, strength_clip: float):
        """
        Load LoRA or bypass based on lora_name.

        Args:
            model: Input MODEL
            clip: Input CLIP
            lora_name: LoRA filename, or "None"/"" to bypass
            strength_model: LoRA strength for model
            strength_clip: LoRA strength for CLIP

        Returns:
            Tuple of (MODEL, CLIP) - either with LoRA applied or passed through
        """
        # Check for bypass
        if lora_name.strip().lower() in ("none", ""):
            return (model, clip)

        # Validate ComfyUI availability
        if not COMFY_AVAILABLE:
            raise RuntimeError("VNCCS_LoraBypass: ComfyUI modules not available")

        try:
            # Get full path to LoRA file
            lora_path = get_full_path_agnostic(folder_paths, "loras", lora_name, require_exists=True)

            if lora_path is None:
                raise FileNotFoundError(f"LoRA not found: {lora_name}")

            # Load LoRA weights
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Apply LoRA to model and CLIP
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )

            return (model_lora, clip_lora)

        except Exception as e:
            raise RuntimeError(f"VNCCS_LoraBypass: failed to load LoRA '{lora_name}': {e}")


class VNCCS_LoraStack:
    """
    Sequential LoRA stack loader. Accepts a JSON array of LoRA configurations and applies
    them sequentially to MODEL and CLIP. Slots with "None" or empty names are skipped.

    This replaces multiple separate LoRA loader nodes with a single configurable stack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "loras": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "execute"
    CATEGORY = "VNCCS/LoRA"
    OUTPUT_NODE = False

    def execute(self, model, clip, loras: str):
        """
        Apply a stack of LoRAs sequentially.

        Args:
            model: Input MODEL
            clip: Input CLIP
            loras: JSON array of LoRA slot objects:
                [
                    {"name": "lora.safetensors", "strength": 0.8},
                    {"name": "lora2.safetensors", "strength_model": 1.0, "strength_clip": 0.5},
                    {"name": "None"},  # Skipped
                    ...
                ]

        Returns:
            Tuple of (MODEL, CLIP) with all active LoRAs applied in order

        Raises:
            ValueError: If JSON is invalid or not an array
            FileNotFoundError: If a LoRA file is not found
        """
        # Parse JSON
        try:
            slots = json.loads(loras)
        except json.JSONDecodeError as e:
            raise ValueError(f"VNCCS_LoraStack: invalid JSON: {e}")

        if not isinstance(slots, list):
            raise ValueError("VNCCS_LoraStack: 'loras' must be a JSON array")

        # Start with input model/clip
        current_model = model
        current_clip = clip

        # Validate ComfyUI availability
        if not COMFY_AVAILABLE:
            raise RuntimeError("VNCCS_LoraStack: ComfyUI modules not available")

        # Apply each LoRA in sequence
        for i, slot in enumerate(slots):
            if not isinstance(slot, dict):
                raise ValueError(f"VNCCS_LoraStack: slot {i} must be an object, got {type(slot).__name__}")

            # Get LoRA name
            name = slot.get("name", "")
            if not name or name.strip().lower() in ("none", ""):
                continue  # Skip empty/None slots

            # Get strength values
            base_strength = slot.get("strength", 1.0)
            strength_model = slot.get("strength_model", base_strength)
            strength_clip = slot.get("strength_clip", base_strength)

            # Skip if both strengths are zero
            if strength_model == 0 and strength_clip == 0:
                continue

            try:
                # Get full path to LoRA file
                lora_path = get_full_path_agnostic(folder_paths, "loras", name, require_exists=True)

                if lora_path is None:
                    raise FileNotFoundError(
                        f"VNCCS_LoraStack: LoRA '{name}' not found (slot {i})"
                    )

                # Load LoRA weights
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

                # Apply LoRA to current model and CLIP
                current_model, current_clip = comfy.sd.load_lora_for_models(
                    current_model, current_clip, lora, strength_model, strength_clip
                )

            except Exception as e:
                raise RuntimeError(
                    f"VNCCS_LoraStack: failed to load LoRA '{name}' at slot {i}: {e}"
                )

        return (current_model, current_clip)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VNCCS_Variable": VNCCS_Variable,
    "VNCCS_ImageList": VNCCS_ImageList,
    "VNCCS_LoraBypass": VNCCS_LoraBypass,
    "VNCCS_LoraStack": VNCCS_LoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Variable": "VNCCS Variable",
    "VNCCS_ImageList": "VNCCS Image List",
    "VNCCS_LoraBypass": "VNCCS LoRA Bypass",
    "VNCCS_LoraStack": "VNCCS LoRA Stack",
}
