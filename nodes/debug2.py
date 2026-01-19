"""VNCCS Debug2 Node - MakeHuman Skeleton Debug

Simple debug node to verify skeleton-mesh relationship.
NO skinning logic - just displays skeleton and mesh side by side.
"""

class VNCCS_Debug2:
    """Debug node for skeleton visualization."""
    
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
    "VNCCS_Debug2": VNCCS_Debug2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Debug2": "VNCCS MH Skeleton Debug"
}
