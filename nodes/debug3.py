"""VNCCS Debug3 Node - DOM-Based Three.js Widget

Uses addDOMWidget for real DOM canvas with native mouse events.
"""

class VNCCS_Debug3:
    """Debug node with DOM-based Three.js viewer."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VNCCS/Debug"
    OUTPUT_NODE = True

    def run(self):
        return {}

NODE_CLASS_MAPPINGS = {
    "VNCCS_Debug3": VNCCS_Debug3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Debug3": "VNCCS Debug3 (DOM Widget)"
}
