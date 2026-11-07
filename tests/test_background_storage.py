"""Current background assets and independent CGs, without workflows or model weights."""

import copy
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from helpers.backend_package import service_package


load = service_package("vnccs_current_background_storage_test")
background = load("nodes.background_creator")
cg = load("nodes.cg_creator")
storage = background.background_storage
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 60}


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "VNCCS" / "Backgrounds"
    monkeypatch.setattr(background, "root", lambda: root)
    monkeypatch.setattr(cg, "root", lambda: root)
    document = background.create_background("Room", background.normalize_editor({}))
    document = background.add_image(document["background_id"], Image.new("RGB", (128, 64), "red"), {"source": "upload"})
    return document, root / "Room", tmp_path


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def build_model(document, tmp_path, color=b"geometry"):
    source = tmp_path / "factory"
    source.mkdir(exist_ok=True)
    (source / "model.ply").write_bytes(color)
    scene = {"scene_id": "f" * 32, "camera": CAMERA,
             "objects": [{"object_id": "d" * 32, "files": {"ply": "model.ply"}, "source_camera": CAMERA}]}
    return cg.commit_model(document["background_id"], document["selected_image_id"], source, scene, {}, None)


def test_disk_configuration_contains_current_state_without_asset_history(library):
    document, directory, _ = library
    identity = document["selected_image_id"]
    for color in ("blue", "green", "yellow"):
        document = background.add_image(document["background_id"], Image.new("RGB", (128, 64), color), {})
        assert document["selected_image_id"] == identity
        assert len(document["images"]) == 1
        assert background.load_background(document["background_id"]) == document
    disk = json.loads((directory / "Room_config.json").read_bytes())
    assert disk["schema_version"] == 3
    assert not {"images", "selected_image_id", "generated_sets", "models", "cgs"} & disk.keys()
    assert disk["source"]["file"] == "source.png"
    assert disk["variant_outputs"] is None
    assert (directory / "Room_config.json.backup").read_bytes() == (directory / "Room_config.json").read_bytes()
    assert sorted(path.name for path in directory.glob("*.png")) == ["source.png"]
    assert not (directory / "Sources").exists()


def test_replacing_source_invalidates_models_and_variants_but_keeps_saved_cg(library):
    document, directory, tmp_path = library
    document = build_model(document, tmp_path)
    model = document["models"][0]
    image = Image.new("RGB", (128, 64), "blue")
    document = cg.commit_cg(document["background_id"], "View", image, image, CAMERA, {}, model["model_id"], None)
    saved = copy.deepcopy(document["cgs"][0])
    config = directory / "CG" / "View" / "View_config.json"
    original = config.read_bytes()
    document = background.add_image(document["background_id"], Image.new("RGB", (128, 64), "green"), {})
    loaded = cg.load_background(document["background_id"])
    assert loaded["models"] == [] and loaded["generated_sets"] == []
    assert len(loaded["cgs"]) == 1 and loaded["cgs"][0]["model_id"] == model["model_id"]
    assert config.read_bytes() == original
    assert cg.cg_image_path(document["background_id"], saved["cg_id"]).is_file()
    rebuilt = build_model(document, tmp_path, b"replacement")
    assert len(rebuilt["models"]) == 1
    assert rebuilt["models"][0]["file"] == model["file"] == "model/Original/scene.json"
    assert rebuilt["models"][0]["model_id"] != model["model_id"]
    assert cg.load_background(document["background_id"])["cgs"][0]["model_id"] == model["model_id"]
    assert config.read_bytes() == original


def test_ready_cg_is_readable_after_its_model_directory_is_lost(library):
    document, directory, tmp_path = library
    document = build_model(document, tmp_path)
    image = Image.new("RGB", (128, 64))
    document = cg.commit_cg(document["background_id"], "Keep", image, image, CAMERA, {}, document["models"][0]["model_id"], None)
    identity = document["cgs"][0]["cg_id"]
    shutil.rmtree(directory / "model")
    assert cg.load_background(document["background_id"])["models"] == []
    assert cg.cg_image_path(document["background_id"], identity).read_bytes()


@pytest.mark.parametrize("failure", ["source.png", "Room_config.json", "Room_config.json.backup"])
def test_source_publication_failure_rolls_back_pixels_and_all_metadata(library, monkeypatch, failure):
    document, directory, _ = library
    before = snapshot(directory)
    original = background._atomic

    def fail(path, payload):
        if path.name == failure:
            raise OSError("Publication failed")
        original(path, payload)

    monkeypatch.setattr(background, "_atomic", fail)
    with pytest.raises(OSError, match="Publication failed"):
        background.add_image(document["background_id"], Image.new("RGB", (128, 64), "blue"), {})
    assert snapshot(directory) == before


def test_full_variant_runs_keep_one_current_set_and_partial_runs_preserve_other_files(library):
    document, directory, _ = library
    cards = document["editor"]["variants"]
    images = [Image.new("RGB", (128, 64), color) for color in ("red", "blue", "green", "yellow")]
    for _ in range(3):
        background.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, {}, None)
    document = background.load_background(document["background_id"])
    assert len(document["generated_sets"]) == 1
    assert document["generated_sets"][0]["version"] == ""
    assert len(json.loads((directory / "Room_config.json").read_bytes())["variant_outputs"]["images"]) == 4
    before = snapshot(directory / "Images")
    images[2] = Image.new("RGB", (128, 64), "white")
    background.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, {}, None, regenerate_indices=[2])
    after = snapshot(directory / "Images")
    assert {name for name in before if before[name] != after[name]} == {"02_Evening.png"}


def test_shared_job_registry_prevents_background_and_cg_writes_from_overlapping(monkeypatch):
    monkeypatch.setattr(cg, "_JOBS", {"job": {"background_id": "a" * 32, "status": "running"}})
    with pytest.raises(background.BackgroundConflictError, match="busy"):
        background.ensure_idle("a" * 32)
    background.ensure_idle("b" * 32)


def test_legacy_migration_is_explicit_idempotent_and_reversible(library):
    document, directory, tmp_path = library
    active = copy.deepcopy(document["images"][0])
    active.update(image_id="b" * 32, file="Sources/V2.png")
    (directory / "Sources").mkdir()
    Image.new("RGB", (128, 64), "blue").save(directory / active["file"])
    legacy = {**document, "images": [document["images"][0], active], "selected_image_id": active["image_id"],
              "models": [{"model_id": "c" * 32, "source_image_id": document["selected_image_id"], "settings": {}}]}
    (directory / "Room_config.json").write_bytes(storage.json_bytes(legacy))
    (directory / "Room_config.json.backup").write_bytes(storage.json_bytes(legacy))
    before = snapshot(directory)
    report = storage.migrate_directory(directory)
    assert report["sources_before"] == 2 and report["sources_after"] == 1
    assert report["models_before"] == 1 and report["models_after"] == 0
    assert snapshot(directory) == before
    backup = tmp_path / "migration-backup" / "Room"
    assert storage.migrate_directory(directory, backup, apply=True)["applied"]
    assert storage.migrate_directory(directory)["changes"] == []
    with Image.open(directory / "source.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    storage.restore_migration(directory, backup)
    assert snapshot(directory) == before


def test_reading_legacy_metadata_does_not_move_or_rewrite_any_files(library):
    document, directory, _ = library
    (directory / "Room_config.json").write_bytes(storage.json_bytes(document))
    before = snapshot(directory)
    cg.load_background(document["background_id"])
    assert snapshot(directory) == before


def test_unsigned_legacy_model_does_not_return_after_source_replacement(library):
    document, directory, tmp_path = library
    document = build_model(document, tmp_path)
    path = directory / document["models"][0]["file"]
    scene = json.loads(path.read_bytes())
    scene.pop("source_signature")
    path.write_bytes(storage.json_bytes(scene))
    (directory / "Room_config.json").write_bytes(storage.json_bytes(document))
    assert cg.load_background(document["background_id"])["models"]
    background.add_image(document["background_id"], Image.new("RGB", (128, 64), "green"), {})
    assert cg.load_background(document["background_id"])["models"] == []


def test_legacy_model_migrates_when_only_versioned_source_exists(library):
    document, directory, tmp_path = library
    document = build_model(document, tmp_path)
    path = directory / document["models"][0]["file"]
    scene = json.loads(path.read_bytes())
    scene.pop("source_signature")
    path.write_bytes(storage.json_bytes(scene))
    document["images"][0]["file"] = "Sources/V2.png"
    (directory / "Sources").mkdir()
    (directory / "source.png").rename(directory / "Sources/V2.png")
    (directory / "Room_config.json").write_bytes(storage.json_bytes(document))
    before = snapshot(directory)
    assert storage.migrate_directory(directory)["models_after"] == 1
    assert snapshot(directory) == before
    storage.migrate_directory(directory, tmp_path / "backup", apply=True)
    loaded = cg.load_background(document["background_id"])
    assert loaded["models"][0]["model_id"] == document["models"][0]["model_id"]
    assert loaded["images"][0]["file"] == "source.png"
    assert storage.migrate_directory(directory)["changes"] == []


def test_duplicate_variant_labels_keep_separate_model_folders(library):
    document, directory, tmp_path = library
    document = build_model(document, tmp_path)
    original = document["models"][0]
    cards = document["editor"]["variants"]
    cards[0]["name"] = "Original"
    document = background.save_background(document["background_id"], {**document["editor"], "variants": cards})
    images = [Image.new("RGB", (128, 64), color) for color in ("red", "blue", "green", "yellow")]
    background.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, {}, None)
    document = background.load_background(document["background_id"])
    group = document["generated_sets"][0]
    variant = {"set_id": group["set_id"], "index": 1, "variant_id": cards[0]["id"], "name": "Original"}
    scene = json.loads((directory / original["file"]).read_bytes())
    built = cg.commit_model(document["background_id"], document["selected_image_id"], directory / "model/Original", scene, {}, None, source_variant=variant)
    assert {model["file"] for model in built["models"]} == {"model/Original/scene.json", "model/Original (2)/scene.json"}
    assert len(cg.load_background(document["background_id"])["models"]) == 2


def test_interrupted_migration_rollback_can_resume(library, monkeypatch):
    document, directory, tmp_path = library
    (directory / "Room_config.json").write_bytes(storage.json_bytes(document))
    (directory / "Room_config.json.backup").write_bytes(storage.json_bytes(document))
    before = snapshot(directory)
    backup = tmp_path / "migration-backup"
    storage.migrate_directory(directory, backup, apply=True)
    write = storage.atomic
    count = 0

    def interrupt(path, payload):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("Interrupted restore")
        write(path, payload)

    monkeypatch.setattr(storage, "atomic", interrupt)
    with pytest.raises(OSError, match="Interrupted restore"):
        storage.restore_migration(directory, backup)
    monkeypatch.setattr(storage, "atomic", write)
    storage.restore_migration(directory, backup)
    storage.restore_migration(directory, backup)
    assert snapshot(directory) == before


@pytest.mark.parametrize("kind", ["source", "model", "variants"])
def test_interrupted_journal_cleanup_does_not_undo_committed_publication(library, monkeypatch, kind):
    document, directory, tmp_path = library
    if kind == "model":
        document = build_model(document, tmp_path)
    remove = shutil.rmtree
    failed = False

    def interrupt(path, *args, **kwargs):
        nonlocal failed
        if not failed and path.name == (".cg-assets.pending.completed" if kind == "model" else ".background-write.pending.completed"):
            failed = True
            for child in path.iterdir():
                if child.is_dir():
                    remove(child)
                    break
            raise OSError("Interrupted cleanup")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", interrupt)
    if kind == "source":
        published = background.add_image(document["background_id"], Image.new("RGB", (128, 64), "blue"), {})
    elif kind == "model":
        published = build_model(document, tmp_path, b"replacement")
    else:
        images = [Image.new("RGB", (128, 64), color) for color in ("blue", "green", "yellow", "red")]
        background.publish_background_set(document["background_id"], document["selected_image_id"], document["editor"]["variants"], images, {}, None)
        published = background.load_background(document["background_id"])
    assert failed, "The cleanup interruption was exercised"
    loaded = cg.load_background(document["background_id"])
    assert loaded["revision"] == document["revision"] + 1
    assert published == loaded
    if kind == "model":
        assert len(loaded["models"]) == 1
        assert loaded["models"][0]["model_id"] != document["models"][0]["model_id"]
    elif kind == "source":
        with Image.open(background.image_path(document["background_id"], loaded["selected_image_id"])) as image:
            assert image.getpixel((0, 0)) == (0, 0, 255)
    else:
        group = loaded["generated_sets"][0]
        for record, expected in zip(group["images"], images):
            with Image.open(directory / record["file"]) as image:
                assert image.tobytes() == expected.tobytes()
    assert not any(path.name.endswith(".completed") for path in directory.iterdir())
