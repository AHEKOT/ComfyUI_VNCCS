"""Project wrapper around the installed Pose Studio node."""


def studio_node():
    import nodes

    studio = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("VNCCS_PoseStudio")
    if studio is None:
        raise RuntimeError("VNCCS Pose Manager requires VNCCS-Utils Pose Studio. Install or enable VNCCS-Utils.")
    return studio


class VNCCS_PoseManager:
    CATEGORY = "VNCCS/pose"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE,VIDEO", "STRING")
    RETURN_NAMES = ("images", "lighting_prompt")
    OUTPUT_IS_LIST = (True, True)

    @classmethod
    def INPUT_TYPES(cls):
        try:
            inputs = studio_node().INPUT_TYPES()
        except RuntimeError:
            # Keep existing workflows readable while the dependency is unavailable.
            inputs = {
                "required": {"pose_data": ("STRING", {"multiline": True, "dynamicPrompts": False})},
                "optional": {"pose_image": ("IMAGE",),
                             "camera_prompt": ("STRING", {"forceInput": True, "default": ""}),
                             "animation_image_batch": ("BOOLEAN", {"default": False})},
                "hidden": {"unique_id": "UNIQUE_ID"},
            }
        kind, options = inputs["required"]["pose_data"]
        return {**inputs, "required": {**inputs["required"], "pose_data":
            (kind, {**options, "default": '{"export":{"interface_mode":"manager"}}'})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return studio_node().IS_CHANGED(**kwargs)

    def generate(self, **kwargs):
        studio = studio_node()()
        result = studio.generate(**kwargs)
        self.OUTPUT_IS_LIST = studio.OUTPUT_IS_LIST
        return result


NODE_CLASS_MAPPINGS = {"VNCCS_PoseManager": VNCCS_PoseManager}
NODE_DISPLAY_NAME_MAPPINGS = {"VNCCS_PoseManager": "VNCCS Pose Manager"}
