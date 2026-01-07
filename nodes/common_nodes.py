"""
Small collection of simple utility nodes for VNCCS: Integer, Float, String, MultilineText.
Each node accepts a single input named `value` and returns it unchanged. These are useful
for wiring constants or simple passthroughs in flows.

File: nodes/common_nodes.py
Category used: VNCCS
"""
from typing import Tuple
import json

# No heavy imports required; keep these nodes minimal and dependency-free.

class VNCCS_Integer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    CATEGORY = "VNCCS"
    FUNCTION = "pass_through"

    def pass_through(self, value: int) -> Tuple[int]:
        # value may be passed as a list by the UI framework
        if isinstance(value, list):
            value = value[0]
        return (int(value),)


class VNCCS_Float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1e12, "max": 1e12, "step": 0.01})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    CATEGORY = "VNCCS"
    FUNCTION = "pass_through"

    def pass_through(self, value: float) -> Tuple[float]:
        if isinstance(value, list):
            value = value[0]
        return (float(value),)


class VNCCS_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    CATEGORY = "VNCCS"
    FUNCTION = "pass_through"

    def pass_through(self, value: str) -> Tuple[str]:
        if isinstance(value, list):
            value = value[0]
        return (str(value),)


class VNCCS_MultilineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    CATEGORY = "VNCCS"
    FUNCTION = "pass_through"

    def pass_through(self, value: str) -> Tuple[str]:
        if isinstance(value, list):
            value = value[0]
        # Keep the string as-is (may contain newlines)
        return (str(value),)


class VNCCS_PromptConcat:
    """Concatenate up to 4 input strings with a chosen separator.

    Empty inputs are omitted to avoid producing repeated separators.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"default": "", "multiline": False}),
                "b": ("STRING", {"default": "", "multiline": False}),
                "c": ("STRING", {"default": "", "multiline": False}),
                "d": ("STRING", {"default": "", "multiline": False}),
                "separator": ("STRING", {"default": ",", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    CATEGORY = "VNCCS"
    FUNCTION = "concat"

    def concat(self, a: str, b: str, c: str, d: str, separator: str = ","):
        # Unwrap list inputs from the UI
        if isinstance(a, list):
            a = a[0]
        if isinstance(b, list):
            b = b[0]
        if isinstance(c, list):
            c = c[0]
        if isinstance(d, list):
            d = d[0]
        if isinstance(separator, list):
            separator = separator[0]

        parts = [str(x).strip() for x in (a, b, c, d)]
        # Omit empty parts
        parts = [p for p in parts if p != ""]

        joined = separator.join(parts)
        return (joined,)


class VNCCS_PositionControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 0-360 degrees, step 45. Using display="slider" forces the UI widget.
                "azimuth": ("INT", {"default": 0, "min": 0, "max": 360, "step": 45, "display": "slider", "tooltip": "Angle of the camera around the subject (0=Front, 90=Right, 180=Back)"}),
                # -30 to 60, step 30. Using display="slider" forces the UI widget.
                "elevation": ("INT", {"default": 0, "min": -30, "max": 60, "step": 30, "display": "slider", "tooltip": "Vertical angle of the camera (-30=Low, 0=Eye Level, 60=High)"}),
                "distance": (["close-up", "medium shot", "wide shot"], {"default": "medium shot"}),
                "include_trigger": ("BOOLEAN", {"default": True, "tooltip": "Include <sks> trigger word"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "VNCCS"
    FUNCTION = "generate_prompt"
    
    def generate_prompt(self, azimuth, elevation, distance, include_trigger):
        # Normalize azimuth to 0-359
        azimuth = int(azimuth) % 360
        
        # Define exact mapping based on Qwen-Image-Edit-2511-Multiple-Angles-LoRA documentation
        azimuth_map = {
             0: "front view",
            45: "front-right quarter view",
            90: "right side view",
           135: "back-right quarter view",
           180: "back view",
           225: "back-left quarter view",
           270: "left side view",
           315: "front-left quarter view"
        }
        
        # Find closest key (handling the step constraint, but robust to typed values)
        # Handle 360/0 wrap-around specialized check
        if azimuth > 337.5:
             closest_azimuth = 0
        else:
             closest_azimuth = min(azimuth_map.keys(), key=lambda x: abs(x - azimuth))
             
        az_str = azimuth_map[closest_azimuth]

        # Elevation Map
        elevation_map = {
            -30: "low-angle shot",
              0: "eye-level shot",
             30: "elevated shot",
             60: "high-angle shot"
        }
        
        closest_elevation = min(elevation_map.keys(), key=lambda x: abs(x - elevation))
        el_str = elevation_map[closest_elevation]
        
        # Build Prompt
        parts = []
        if include_trigger:
            parts.append("<sks>")
            
        parts.append(az_str)
        parts.append(el_str)
        parts.append(distance)
        
        return (" ".join(parts),)


class VNCCS_VisualPositionControl(VNCCS_PositionControl):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The JS widget submits a JSON string here.
                # format: {"azimuth": 0, "elevation": 0, "distance": "medium shot", "include_trigger": true}
                "camera_data": ("STRING", {"default": "{}", "hidden": True}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "VNCCS"
    FUNCTION = "generate_prompt_from_json"

    def generate_prompt_from_json(self, camera_data):
        try:
            data = json.loads(camera_data)
        except json.JSONDecodeError:
            # Fallback defaults
            data = {"azimuth": 0, "elevation": 0, "distance": "medium shot", "include_trigger": True}
        
        return self.generate_prompt(
            data.get("azimuth", 0), 
            data.get("elevation", 0), 
            data.get("distance", "medium shot"), 
            data.get("include_trigger", True)
        )


# Node registration maps (if the loader for this project expects them)
NODE_CLASS_MAPPINGS = {
    "VNCCS_Integer": VNCCS_Integer,
    "VNCCS_Float": VNCCS_Float,
    "VNCCS_String": VNCCS_String,
    "VNCCS_MultilineText": VNCCS_MultilineText,
    "VNCCS_PromptConcat": VNCCS_PromptConcat,
    "VNCCS_PositionControl": VNCCS_PositionControl,
    "VNCCS_VisualPositionControl": VNCCS_VisualPositionControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Integer": "VNCCS Integer",
    "VNCCS_Float": "VNCCS Float",
    "VNCCS_String": "VNCCS String",
    "VNCCS_MultilineText": "VNCCS Multiline Text",
    "VNCCS_PromptConcat": "VNCCS Prompt Concat",
    "VNCCS_PositionControl": "VNCCS Position Control",
    "VNCCS_VisualPositionControl": "VNCCS Visual Camera Control",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_Integer": "VNCCS",
    "VNCCS_Float": "VNCCS",
    "VNCCS_String": "VNCCS",
    "VNCCS_MultilineText": "VNCCS",
    "VNCCS_PromptConcat": "VNCCS",
    "VNCCS_PositionControl": "VNCCS",
    "VNCCS_VisualPositionControl": "VNCCS",
}

