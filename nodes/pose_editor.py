"""VNCCS Pose Editor Node - Opens mannequin.js pose editor for intuitive character posing."""

class VNCCS_PoseEditor:
    """
    Opens a mannequin.js-based pose editor in a popup window.
    Users can pose the mannequin by dragging, then apply the pose to MakeHuman skeleton.
    """
    
    CATEGORY = "VNCCS/Posing"
    RETURN_TYPES = ()
    FUNCTION = "open_editor"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {}
        }
    
    def open_editor(self):
        """The actual editor is opened via the JavaScript widget."""
        return ()


NODE_CLASS_MAPPINGS = {
    "VNCCS_PoseEditor": VNCCS_PoseEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PoseEditor": "VNCCS Pose Editor (mannequin.js)"
}
