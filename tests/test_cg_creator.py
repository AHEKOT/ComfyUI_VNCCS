"""The CG workflow node only returns an explicitly selected finished frame."""

import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from helpers.backend_package import service_package


load = service_package("vnccs_cg_node_test")
module = load("nodes.cg_creator")
storage = load("nodes.cg_creator")
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 42}


def test_node_registration_and_image_output_contract():
    node = module.VNCCS_CGCreator
    assert node.RETURN_TYPES == ("IMAGE",) and node.RETURN_NAMES == ("image",)
    assert node.FUNCTION == "create_cg" and node.OUTPUT_NODE is True
    assert node.CATEGORY == "VNCCS/3D"
    assert list(node.INPUT_TYPES()) == ["required"]
    assert list(node.INPUT_TYPES()["required"]) == ["cg_data"]
    assert node.INPUT_TYPES()["required"]["cg_data"][1]["dynamicPrompts"] is False
    registration = (Path(__file__).resolve().parents[1] / "nodes" / "cg_creator.py").read_text()
    assert '"VNCCS_CGCreator": VNCCS_CGCreator' in registration
    assert '"VNCCS_CGCreator": "VNCCS CG Creator"' in registration
    assert "CG_CREATOR_MAPPINGS" in (Path(__file__).resolve().parents[1] / "nodes" / "__init__.py").read_text()


@pytest.mark.parametrize("raw", [None, [], "{", "[]", "null", '{"schema_version":2}',
    '{"schema_version":true}', '{"schema_version":1,"background_id":"../file"}',
    '{"schema_version":1,"background_id":[]}', '{"schema_version":1,"model_id":42}',
    '{"schema_version":1,"selected_cg_id":"../file"}', '{"schema_version":1,"camera":{}}',
    '{"schema_version":1,"selected_variant":{"set_id":"../bad","index":1}}',
    '{"schema_version":1,"selected_variant":{"set_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","index":0}}',
    " " * (256 * 1024 + 1)])
def test_malformed_workflow_state_is_rejected(raw):
    assert module.VNCCS_CGCreator.VALIDATE_INPUTS(raw) is not True


def test_empty_selection_is_a_valid_draft_state():
    assert module.VNCCS_CGCreator.VALIDATE_INPUTS(module.EMPTY_STATE) is True
    assert module.parse_state(module.EMPTY_STATE)["selected_cg_id"] == ""


@pytest.fixture
def execution(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    monkeypatch.setattr(storage.background, "root", storage.root)
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {}, raising=False)
    background_id, image_id, model_id, cg_id = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    document = {"schema_version": 2, "background_id": background_id, "name": "Saved room", "revision": 7,
        "archived": False, "editor": {"description": "Keep the main generator draft"},
        "images": [{"image_id": image_id, "file": "source.png", "width": 128, "height": 64, "settings": {"seed": 3}}],
        "selected_image_id": image_id,
        "models": [{"model_id": model_id, "source_image_id": image_id, "settings": {"seed": 4}}],
        "cgs": [{"cg_id": cg_id, "name": "Window", "background_id": background_id, "model_id": model_id,
            "source_image_id": image_id, "camera": CAMERA, "settings": {"seed": 5}, "width": 128, "height": 64}],
        "generated_sets": []}
    directory = storage.root() / document["name"]
    directory.mkdir(parents=True)
    encoded = json.dumps(document).encode()
    (directory / "Saved room_config.json").write_bytes(encoded)
    (directory / "Saved room_config.json.backup").write_bytes(encoded)
    frames = directory / "CG" / "Window" / cg_id
    frames.mkdir(parents=True)
    Image.new("RGB", (128, 64), "red").save(frames / "raw.png")
    Image.new("RGB", (128, 64), "blue").save(frames / "image.png")
    factory = storage.background
    def preview(path):
        with Image.open(path) as image:
            return (np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[None]
    factory._preview_tensor = mock.Mock(side_effect=preview)
    monkeypatch.setattr(factory, "_preview_tensor", factory._preview_tensor)
    generation = module
    monkeypatch.setattr(generation, "generate_image", mock.Mock(side_effect=AssertionError("Workflow execution must never run QI2")))
    state = {"schema_version": 1, "background_id": background_id, "selected_image_id": image_id,
             "model_id": model_id, "selected_cg_id": cg_id, "camera": CAMERA}
    return types.SimpleNamespace(node=module.VNCCS_CGCreator(), state=state, preview=factory._preview_tensor,
                                 generation=generation, directory=directory, frames=frames)


def test_selected_finished_cg_is_the_only_image_sent_downstream(execution):
    output, = execution.node.create_cg(json.dumps(execution.state))
    assert output.shape == (1, 64, 128, 3)
    np.testing.assert_array_equal(output[0, 0, 0], [0, 0, 1])
    execution.preview.assert_called_once_with(execution.directory / "CG" / "Window" / "image.png")
    assert not execution.frames.exists()
    execution.generation.generate_image.assert_not_called()


@pytest.mark.parametrize("missing", ["background_id", "selected_cg_id"])
def test_unfinished_selection_stops_before_preview_or_generation(execution, missing):
    state = {**execution.state, missing: ""}
    with pytest.raises(ValueError, match="Generate and select a finished CG"):
        execution.node.create_cg(json.dumps(state))
    execution.preview.assert_not_called()
    execution.generation.generate_image.assert_not_called()


def test_missing_finished_asset_does_not_generate_a_replacement(execution):
    (execution.frames / "image.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        execution.node.create_cg(json.dumps(execution.state))
    execution.preview.assert_not_called()
    execution.generation.generate_image.assert_not_called()


def test_unknown_finished_cg_id_cannot_fall_back_to_another_frame(execution):
    state = {**execution.state, "selected_cg_id": "f" * 32}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        execution.node.create_cg(json.dumps(state))
    execution.preview.assert_not_called()
    execution.generation.generate_image.assert_not_called()


def test_saved_frame_remains_usable_when_its_background_is_archived(execution):
    path = execution.directory / "Saved room_config.json"
    document = json.loads(path.read_bytes())
    document["archived"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    output, = execution.node.create_cg(json.dumps(execution.state))
    np.testing.assert_array_equal(output[0, 0, 0], [0, 0, 1])
    execution.generation.generate_image.assert_not_called()
