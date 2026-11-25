import sys
import types

import pytest

from conftest import _preload_node

manager = _preload_node("pose_manager")


def test_manager_delegates_schema_execution_and_output_shape(monkeypatch):
    schema = {"required": {"pose_data": ("STRING", {"default": "{}", "dynamicPrompts": False})}, "optional": {"pose_image": ("IMAGE",)}}
    calls = []
    class Studio:
        @classmethod
        def INPUT_TYPES(cls):
            return schema
        @classmethod
        def IS_CHANGED(cls, **kwargs):
            calls.append(kwargs)
            return "changed"
        def generate(self, **kwargs):
            calls.append(kwargs)
            self.OUTPUT_IS_LIST = (False, True)
            return ("native images", ["native lighting"])
    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(NODE_CLASS_MAPPINGS={"VNCCS_PoseStudio": Studio}))
    inputs = manager.VNCCS_PoseManager.INPUT_TYPES()
    assert inputs["optional"] == schema["optional"]
    assert '"manager"' in inputs["required"]["pose_data"][1]["default"]
    assert schema["required"]["pose_data"][1]["default"] == "{}"
    node = manager.VNCCS_PoseManager()
    assert node.IS_CHANGED(pose_data="authored") == "changed"
    assert node.generate(pose_data="authored") == ("native images", ["native lighting"])
    assert node.OUTPUT_IS_LIST == (False, True)
    assert calls == [{"pose_data": "authored"}] * 2


def test_missing_utils_keeps_workflow_schema_but_fails_execution(monkeypatch):
    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    assert manager.VNCCS_PoseManager.INPUT_TYPES()["hidden"] == {"unique_id": "UNIQUE_ID"}
    with pytest.raises(RuntimeError, match="requires VNCCS-Utils"):
        manager.VNCCS_PoseManager().generate()
