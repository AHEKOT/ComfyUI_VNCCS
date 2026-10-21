"""Cross-owner CG contracts without importing Utils, Torch or model weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from helpers.backend_package import service_package

load = service_package("vnccs_cg_migration_test")
bridge = load("nodes.cg_creator")
storage = load("nodes.cg_creator")
owner = storage.background
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 60}


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "library")
    monkeypatch.setattr(owner, "root", storage.root)
    return tmp_path


def test_missing_or_old_utils_is_reported_without_importing_it(monkeypatch):
    import nodes
    for node in (None, type("OldFactory", (), {})):
        monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {"VNCCS_3DFactory": node}, raising=False)
        with pytest.raises(RuntimeError, match="Update ComfyUI_VNCCS_Utils"):
            bridge.provider()
        assert "error" in bridge.frontend_configuration()


def test_provider_discovery_uses_registered_mount_instead_of_checkout_name(monkeypatch, tmp_path):
    import nodes
    value = {"version": 1, "storage": object(), "runtime": object(), "web_root": tmp_path / "web"}
    factory = type("Factory", (), {"cg_provider": classmethod(lambda cls: value)})
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {"VNCCS_3DFactory": factory}, raising=False)
    monkeypatch.setattr(nodes, "EXTENSION_WEB_DIRS", {"Renamed Utils": value["web_root"]}, raising=False)
    assert bridge.provider() is value
    assert bridge.frontend_configuration() == {"version": 1, "module_url": "/extensions/Renamed%20Utils/factory3d/cg_provider.mjs"}
    value["version"] = 2
    with pytest.raises(RuntimeError, match="Unsupported"):
        bridge.provider()


def test_background_edits_and_regeneration_preserve_model_sources_and_cgs(library):
    document = owner.create_background("Room", {"description": "Original background"})
    document = owner.add_image(document["background_id"], Image.new("RGB", (64, 32), "red"), {})
    image_id = document["selected_image_id"]
    old_source = owner.image_path(document["background_id"], image_id).read_bytes()
    source = library / "factory-scene"
    source.mkdir()
    (source / "model.ply").write_bytes(b"immutable model fixture")
    scene = {"camera": CAMERA, "objects": [{"object_id": "d" * 32, "files": {"ply": "model.ply"}}]}
    document = storage.commit_model(document["background_id"], image_id, source, scene, {}, document["revision"])
    model_id = document["models"][0]["model_id"]
    image = Image.new("RGB", (64, 32), "blue")
    document = storage.commit_cg(document["background_id"], "View", image, image, CAMERA, {}, model_id, document["revision"])
    cg_id = document["cgs"][0]["cg_id"]
    document = owner.save_background(document["background_id"], {"description": "New authored settings"})
    document = owner.add_image(document["background_id"], Image.new("RGB", (64, 32), "green"), {})
    assert document["selected_image_id"] != image_id
    assert owner.image_path(document["background_id"], image_id).read_bytes() == old_source
    assert storage.cg_image_path(document["background_id"], cg_id).is_file()
    assert storage.load_background(document["background_id"])["models"][0]["model_id"] == model_id


def test_cg_reads_recover_the_background_publication_journal(library, monkeypatch):
    document = owner.create_background("Room")
    calls = []
    recover = owner._recover_set_publication
    monkeypatch.setattr(owner, "_recover_set_publication", lambda directory: (calls.append(directory), recover(directory)))
    assert storage.load_background(document["background_id"]) == document
    assert calls == [storage.root() / "Room"]


def test_normal_cg_saves_accept_missing_revision_and_preserve_new_background_fields(library):
    document = owner.create_background("Room", {"description": "First"})
    owner.save_background(document["background_id"], {"description": "Current"})
    saved = storage._save_settings(document["background_id"], {}, {"future_profile": [1, 2]}, None)
    assert saved["editor"] == {"description": "Current"}
    assert saved["cg_settings"]["future_profile"] == [1, 2]
    assert saved["revision"] == 3
