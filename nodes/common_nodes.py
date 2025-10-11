"""
Small collection of simple utility nodes for VNCCS: Integer, Float, String, MultilineText.
Each node accepts a single input named `value` and returns it unchanged. These are useful
for wiring constants or simple passthroughs in flows.

File: nodes/common_nodes.py
Category used: VNCCS
"""
from typing import Tuple

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


# Node registration maps (if the loader for this project expects them)
NODE_CLASS_MAPPINGS = {
    "VNCCS_Integer": VNCCS_Integer,
    "VNCCS_Float": VNCCS_Float,
    "VNCCS_String": VNCCS_String,
    "VNCCS_MultilineText": VNCCS_MultilineText,
    "VNCCS_PromptConcat": VNCCS_PromptConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Integer": "VNCCS Integer",
    "VNCCS_Float": "VNCCS Float",
    "VNCCS_String": "VNCCS String",
    "VNCCS_MultilineText": "VNCCS Multiline Text",
    "VNCCS_PromptConcat": "VNCCS Prompt Concat",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_Integer": "VNCCS",
    "VNCCS_Float": "VNCCS",
    "VNCCS_String": "VNCCS",
    "VNCCS_MultilineText": "VNCCS",
    "VNCCS_PromptConcat": "VNCCS",
}
