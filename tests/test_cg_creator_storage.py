"""Shared scene persistence without ComfyUI, the main extension or model weights."""

import copy
import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from helpers.backend_package import service_package


load = service_package("vnccs_cg_storage_test")
storage = load("nodes.cg_creator")
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 42}


def path_for(document):
    return storage.root() / document["name"] / f"{document['name']}_config.json"


def files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def write_document(document):
    path = path_for(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2).encode() + b"\n"
    path.write_bytes(encoded)
    path.with_suffix(".json.backup").write_bytes(encoded)
    return path


@pytest.fixture
def scene(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    monkeypatch.setattr(storage.background, "root", storage.root)
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {}, raising=False)
    document = {"schema_version": 2, "background_id": "a" * 32, "name": "Quiet room", "revision": 7,
                "archived": False, "editor": {"description": "Keep this scene", "new_generator_option": {"x": 3}},
                "images": [{"image_id": "b" * 32, "file": "Sources/V1.png", "width": 64, "height": 32,
                            "settings": {"generator": "main", "seed": 3}, "preview_revision": 4},
                           {"image_id": "c" * 32, "file": "source.png", "width": 64, "height": 32,
                            "settings": {"generator": "main", "seed": 4}, "preview_revision": 1}],
                "selected_image_id": "c" * 32, "models": [], "cgs": [],
                "generator_settings": {"model": "qi2", "sampling": {"steps": 25}, "future": [1, 2]},
                "generated_sets": [],
                "future_owner_data": {"revision": "unrelated", "values": ["keep", "all"]}}
    write_document(document)
    for index, image in enumerate(document["images"]):
        path = storage.root() / document["name"] / image["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), "red" if index == 0 else "blue").save(path)
    return document


def factory_source(tmp_path):
    source = tmp_path / "factory-source"
    object_id = "d" * 32
    assets = source / "objects" / object_id
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "model.ply").write_bytes(b"immutable Gaussian fixture")
    Image.new("RGB", (64, 32), "green").save(assets / "reference.png")
    manifest = {"schema_version": 11, "scene_id": "e" * 32, "camera": CAMERA,
                "objects": [{"object_id": object_id, "source_camera": CAMERA,
                             "files": {"ply": f"objects/{object_id}/model.ply",
                                       "reference": f"objects/{object_id}/reference.png"}}]}
    return source, manifest


def add_model(document, tmp_path, image_id=None):
    source, manifest = factory_source(tmp_path)
    return storage.commit_model(document["background_id"], image_id or document["images"][0]["image_id"],
                                source, manifest, {"seed": 4}, document["revision"])


def test_no_background_generation_or_document_authoring_api(scene):
    for name in ("create_background", "add_image", "archive_background", "save_background", "save_editor"):
        assert not hasattr(storage, name)
    assert storage.list_backgrounds() == [scene]
    assert storage.load_background(scene["background_id"]) == scene


def test_save_only_cg_owned_settings_and_preserve_all_main_fields(scene):
    previous = path_for(scene).read_bytes()
    originals = {image["image_id"]: storage.image_path(scene["background_id"], image["image_id"]).read_bytes()
                 for image in scene["images"]}
    model_settings = {"seed": 14, "gaussians": 262144}
    cg_settings = {"prompt": "Complete the frame", "camera": copy.deepcopy(CAMERA), "resolution": [128, 64]}
    updated = storage._save_settings(scene["background_id"], model_settings, cg_settings, scene["revision"])
    assert updated == {**scene, "revision": scene["revision"] + 1,
                       "model_settings": model_settings, "cg_settings": cg_settings}
    assert path_for(scene).with_suffix(".json.backup").read_bytes() == previous
    model_settings["seed"] = 999
    cg_settings["camera"]["position"][0] = 999
    assert storage.load_background(scene["background_id"])["model_settings"]["seed"] == 14
    assert storage.load_background(scene["background_id"])["cg_settings"]["camera"] == CAMERA
    for image in scene["images"]:
        assert storage.image_path(scene["background_id"], image["image_id"]).read_bytes() == originals[image["image_id"]]


@pytest.mark.parametrize("preview_revision", [False, 0, -1, 1.5, "2"])
def test_invalid_main_preview_revision_is_not_silently_rewritten(scene, preview_revision):
    scene["images"][0]["preview_revision"] = preview_revision
    write_document(scene)
    before = files(storage.root())
    with pytest.raises(storage.BackgroundStorageError):
        storage.load_background(scene["background_id"])
    assert files(storage.root()) == before


def test_cg_deletion_preserves_other_data_assets_and_saved_workflow_references(scene, tmp_path):
    document = add_model(scene, tmp_path)
    raw = Image.new("RGB", (64, 32), "red")
    for name in ("Failed", "Keep"):
        document = storage.commit_cg(scene["background_id"], name, raw, raw, CAMERA, {},
                                     document["models"][0]["model_id"], document["revision"])
    target = document["cgs"][0]
    before = files(storage.root()); previous = path_for(document).read_bytes()
    updated = storage._archive_cg(scene["background_id"], target["cg_id"], document["revision"])
    assert updated == {**document, "revision": document["revision"] + 1,
                       "cgs": [{**target, "archived": True}, document["cgs"][1]]}
    assert path_for(updated).with_suffix(".json.backup").read_bytes() == previous
    for path, data in before.items():
        if not path.endswith((".json", ".json.backup")):
            assert (storage.root() / path).read_bytes() == data
    assert storage.cg_image_path(scene["background_id"], target["cg_id"]).is_file()
    before = files(storage.root())
    for cg_id, revision, error in ((target["cg_id"], document["revision"], storage.BackgroundConflictError),
                                   (target["cg_id"], updated["revision"], FileNotFoundError),
                                   ("0" * 32, updated["revision"], FileNotFoundError),
                                   ("../outside", updated["revision"], ValueError)):
        with pytest.raises(error):
            storage._archive_cg(scene["background_id"], cg_id, revision)
        assert files(storage.root()) == before


def test_failed_cg_deletion_keeps_the_original_document_and_assets(scene, tmp_path, monkeypatch):
    document = add_model(scene, tmp_path); raw = Image.new("RGB", (64, 32))
    document = storage.commit_cg(scene["background_id"], "Keep", raw, raw, CAMERA, {},
                                 document["models"][0]["model_id"], document["revision"])
    original = path_for(document).read_bytes(); atomic = storage.background._atomic
    def fail(path, data):
        if path == path_for(document):
            raise OSError("Disk write failed")
        return atomic(path, data)
    monkeypatch.setattr(storage.background, "_atomic", fail)
    with pytest.raises(OSError, match="Disk write failed"):
        storage._archive_cg(scene["background_id"], document["cgs"][0]["cg_id"], document["revision"])
    assert path_for(document).read_bytes() == original
    assert storage.load_background(scene["background_id"]) == document
    assert storage.cg_image_path(scene["background_id"], document["cgs"][0]["cg_id"]).is_file()


def test_preview_revision_remains_metadata_and_does_not_change_image_path(scene):
    image = scene["images"][0]
    path = storage.image_path(scene["background_id"], image["image_id"])
    assert path == storage.root() / scene["name"] / image["file"]
    assert storage.load_background(scene["background_id"])["images"][0]["preview_revision"] == 4


def test_models_and_cgs_keep_source_links_and_previous_variants(scene, tmp_path):
    original_fields = copy.deepcopy(scene)
    document = add_model(scene, tmp_path)
    first_model = document["models"][-1]
    first_model_path = storage.model_asset_path(document["background_id"], first_model["model_id"], "d" * 32, "ply")
    first_bytes = first_model_path.read_bytes()
    raw, finished = Image.new("RGB", (64, 32), "red"), Image.new("RGB", (128, 64), "blue")
    document = storage.commit_cg(document["background_id"], "Window", raw, finished, CAMERA,
                                 {"seed": 3}, first_model["model_id"], document["revision"])
    first_cg = document["cgs"][-1]
    assert (first_cg["width"], first_cg["height"]) == finished.size
    with Image.open(storage.cg_image_path(document["background_id"], first_cg["cg_id"], "raw")) as capture:
        assert capture.size == raw.size
    cg_bytes = storage.cg_image_path(document["background_id"], first_cg["cg_id"]).read_bytes()
    document = add_model(document, tmp_path, scene["images"][1]["image_id"])
    second_model = document["models"][-1]
    document = add_model(document, tmp_path, scene["images"][0]["image_id"])
    rebuilt_model = document["models"][-1]
    assert len(document["models"]) == 3
    assert [(model["model_id"], model["source_image_id"]) for model in document["models"]] == [
        (first_model["model_id"], scene["images"][0]["image_id"]),
        (second_model["model_id"], scene["images"][1]["image_id"]),
        (rebuilt_model["model_id"], scene["images"][0]["image_id"])]
    assert first_cg["source_image_id"] == first_model["source_image_id"]
    assert document["cgs"][0] == first_cg
    assert first_model_path.read_bytes() == first_bytes
    assert storage.cg_image_path(document["background_id"], first_cg["cg_id"]).read_bytes() == cg_bytes
    assert storage.model_scene(document["background_id"], first_model["model_id"])["source_image_id"] == first_model["source_image_id"]
    cg_root = storage.root() / scene["name"] / "CG" / "Window"
    assert set(path.name for path in cg_root.iterdir()) == {"raw.png", "image.png"}
    assert cg_root / "image.png" == storage.cg_image_path(document["background_id"], first_cg["cg_id"])
    for key in ("editor", "images", "selected_image_id", "generated_sets", "generator_settings", "future_owner_data"):
        assert document[key] == original_fields[key]


@pytest.mark.parametrize("operation", ["settings", "model", "cg"])
def test_stale_writes_preserve_every_file_byte(scene, tmp_path, operation):
    document = add_model(scene, tmp_path)
    source, manifest = factory_source(tmp_path)
    before = files(storage.root())
    raw = Image.new("RGB", (64, 32))
    mutations = {
        "settings": lambda: storage._save_settings(document["background_id"], {}, {}, document["revision"] - 1),
        "model": lambda: storage.commit_model(document["background_id"], document["selected_image_id"],
                                               source, manifest, {}, document["revision"] - 1),
        "cg": lambda: storage.commit_cg(document["background_id"], "Stale", raw, raw, CAMERA, {},
                                         document["models"][-1]["model_id"], document["revision"] - 1),
    }
    with pytest.raises(storage.BackgroundConflictError):
        mutations[operation]()
    assert files(storage.root()) == before


def test_archived_scene_assets_still_read_but_mutations_are_blocked(scene, tmp_path):
    scene = add_model(scene, tmp_path)
    scene["archived"] = True
    write_document(scene)
    assert storage.list_backgrounds() == []
    assert storage.load_background(scene["background_id"])["archived"] is True
    assert storage.image_path(scene["background_id"], scene["images"][0]["image_id"]).is_file()
    assert storage.model_scene(scene["background_id"], scene["models"][0]["model_id"])["background_id"] == scene["background_id"]
    before = files(storage.root())
    with pytest.raises(ValueError, match="archived"):
        storage._save_settings(scene["background_id"], {}, {}, scene["revision"])
    assert files(storage.root()) == before


def test_valid_backup_is_read_only_and_damaged_primary_is_preserved(scene):
    path = path_for(scene)
    path.write_bytes(b"{damaged")
    before = files(storage.root())
    recovered = storage.load_background(scene["background_id"])
    assert recovered["editor"] == scene["editor"]
    assert recovered["_recovery"]["source"] == f"{scene['name']}_config.json.backup"
    with pytest.raises(storage.BackgroundStorageError, match="read-only"):
        storage._save_settings(scene["background_id"], {}, {}, scene["revision"])
    assert files(storage.root()) == before


def test_corrupt_metadata_never_becomes_an_empty_scene_library(scene):
    path = path_for(scene)
    path.write_bytes(b"{bad")
    path.with_suffix(".json.backup").write_bytes(b"[]")
    before = files(storage.root())
    with pytest.raises(storage.BackgroundStorageError):
        storage.list_backgrounds()
    assert files(storage.root()) == before


def test_failed_json_replace_keeps_original_and_cleans_temporary_file(scene, monkeypatch):
    path = path_for(scene)
    original = path.read_bytes()
    replace = storage.os.replace

    def fail_primary(source, destination):
        if Path(destination) == path:
            raise OSError("Disk unavailable")
        replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_primary)
    with pytest.raises(OSError, match="Disk unavailable"):
        storage._save_settings(scene["background_id"], {}, {}, scene["revision"])
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_background_owner_is_the_local_authoritative_library(scene):
    assert storage.background_owner() is storage.background
    assert storage.BackgroundConflictError is storage.background.BackgroundConflictError
    with storage.background._LOCK:
        assert storage.load_background(scene["background_id"]) == scene



def test_shared_main_lock_prevents_cg_overwriting_a_new_main_revision(scene, monkeypatch):
    attempted = threading.Event()
    lock = threading.RLock()

    class MainLock:
        def __enter__(self):
            attempted.set()
            lock.acquire()

        def __exit__(self, *_args):
            lock.release()

    monkeypatch.setattr(storage.background, "_LOCK", MainLock())
    with ThreadPoolExecutor(max_workers=1) as pool:
        with lock:
            pending = pool.submit(storage._save_settings, scene["background_id"], {"seed": 1}, {}, scene["revision"])
            assert attempted.wait(2), "CG did not acquire the installed owner's lock"
            assert not pending.done()
            newer = copy.deepcopy(scene)
            newer["revision"] += 1
            newer["generator_settings"]["main_edit"] = "Keep the newest settings"
            write_document(newer)
        with pytest.raises(storage.BackgroundConflictError):
            pending.result(timeout=2)
    assert storage.load_background(scene["background_id"]) == newer


def test_local_lock_and_revisions_allow_only_one_concurrent_settings_writer(scene):
    def save(seed):
        try:
            return storage._save_settings(scene["background_id"], {"seed": seed}, {}, scene["revision"])
        except storage.BackgroundConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, [1, 2]))
    assert sum(result is not None for result in results) == 1
    assert storage.load_background(scene["background_id"])["revision"] == scene["revision"] + 1


def test_source_assets_and_model_paths_reject_symlinks_and_traversal(scene, tmp_path):
    image_id = scene["images"][0]["image_id"]
    image = storage.image_path(scene["background_id"], image_id)
    outside = tmp_path / "outside.png"
    outside.write_bytes(image.read_bytes())
    image.unlink()
    try:
        image.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links are not available on this platform")
    with pytest.raises(ValueError, match="symbolic links"):
        storage.image_path(scene["background_id"], image_id)
    source, manifest = factory_source(tmp_path)
    manifest["objects"][0]["files"]["ply"] = "../outside.ply"
    before = files(storage.root())
    with pytest.raises(ValueError, match="path"):
        storage.commit_model(scene["background_id"], image_id, source, manifest, {}, scene["revision"])
    assert files(storage.root()) == before


def test_splat_uses_shared_factory_cache_without_loading_models(scene, tmp_path, monkeypatch):
    document = add_model(scene, tmp_path)
    cache = tmp_path / "cached.splat"
    cache.write_bytes(b"cached SPLAT")
    calls = []
    factory = types.SimpleNamespace()

    def cached(path, *, ply_sha256=""):
        calls.append((path, ply_sha256))
        return cache

    factory._ensure_cached_splat = cached
    bridge = load("nodes.cg_creator")
    monkeypatch.setattr(bridge, "provider", lambda: {"storage": factory})
    result = storage.model_asset_path(document["background_id"], document["models"][0]["model_id"], "d" * 32, "splat")
    assert result == cache and calls[0][0].suffix == ".ply"


@pytest.fixture
def source_variants(scene):
    set_id = "7" * 32
    group = {"set_id": set_id, "version": "", "source_image_id": scene["images"][0]["image_id"],
             "variants": [{"id": "morning", "name": "Morning", "time": "Morning", "weather": "Clear", "additional_details": ""},
                          {"id": "night", "name": "Night", "time": "Night", "weather": "Clear", "additional_details": ""}],
             "settings": {"preserve_main_settings": True}, "images": []}
    for index, (name, color) in enumerate([("Original", "red"), ("Morning", "yellow"), ("Night", "navy")]):
        relative = f"Images/{index:02d}_{name}.png"
        path = path_for(scene).parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), color).save(path)
        group["images"].append({"name": name, "file": relative, "width": 64, "height": 32, "revision": scene["revision"], "settings": {}})
    scene["generated_sets"].append(group)
    write_document(scene)
    references = [{"set_id": set_id, "index": index, "variant_id": card["id"], "name": card["name"]}
                  for index, card in enumerate(group["variants"], start=1)]
    return scene, references


def test_current_main_documents_without_cg_arrays_are_read_without_rewriting(scene, tmp_path):
    scene.pop("models")
    scene.pop("cgs")
    write_document(scene)
    before = files(storage.root())
    assert storage.load_background(scene["background_id"]) == scene
    assert storage.list_backgrounds() == [scene]
    assert files(storage.root()) == before
    document = add_model(scene, tmp_path)
    assert len(document["models"]) == 1 and "cgs" not in document
    for key in ("editor", "images", "generated_sets", "generator_settings", "future_owner_data"):
        assert document[key] == scene[key]
    raw = Image.new("RGB", (64, 32))
    document = storage.commit_cg(scene["background_id"], "First CG", raw, raw, CAMERA, {},
                                 document["models"][0]["model_id"], document["revision"])
    assert len(document["cgs"]) == 1


def test_original_and_generated_variants_resolve_locally_without_changing_main_metadata(source_variants, monkeypatch):
    scene, (morning, night) = source_variants
    image_id = scene["images"][0]["image_id"]
    before = files(storage.root())
    owner = types.ModuleType("cg_test_unused_variant_owner")
    owner._LOCK = threading.RLock()

    def forbidden(*_args):
        raise AssertionError("CG variant paths must not depend on the main runtime")

    owner.load_background = owner.list_backgrounds = owner.image_path = owner.background_set_image_path = forbidden
    node = type("BackgroundCreator", (), {"__module__": owner.__name__})
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"VNCCS_BackgroundCreator": node})
    original = storage.image_path(scene["background_id"], image_id)
    assert storage.image_path(scene["background_id"], image_id, None) == original
    assert storage.load_background(scene["background_id"]) == scene
    assert storage.list_backgrounds() == [scene]
    with Image.open(storage.image_path(scene["background_id"], image_id, morning)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 0)
    with Image.open(storage.image_path(scene["background_id"], image_id, night)) as image:
        assert image.getpixel((0, 0)) == (0, 0, 128)
    assert files(storage.root()) == before


def test_morning_night_models_and_cgs_keep_exact_variant_references(source_variants, tmp_path):
    scene, references = source_variants
    original_metadata = copy.deepcopy(scene)
    source, manifest = factory_source(tmp_path)
    document = scene
    for variant in references:
        document = storage.commit_model(scene["background_id"], scene["images"][0]["image_id"], source, manifest,
                                        {}, document["revision"], variant)
        model = document["models"][-1]
        assert model["source_variant"] == variant
        assert storage.model_scene(scene["background_id"], model["model_id"])["source_variant"] == variant
        raw = Image.new("RGB", (64, 32))
        document = storage.commit_cg(scene["background_id"], variant["name"], raw, raw, CAMERA, {},
                                     model["model_id"], document["revision"])
        assert document["cgs"][-1]["source_variant"] == variant
    assert [storage._variant_key(model["source_variant"]) for model in document["models"]] == [
        (references[0]["set_id"], 1), (references[1]["set_id"], 2)]
    for key in ("editor", "images", "generated_sets", "generator_settings", "future_owner_data"):
        assert document[key] == original_metadata[key]
    # Saved models/CG remain usable if the owner later removes its generated set.
    document["generated_sets"] = []
    document["revision"] += 1
    write_document(document)
    assert storage.load_background(scene["background_id"])["models"] == document["models"]
    for model, cg in zip(document["models"], document["cgs"]):
        assert storage.model_scene(scene["background_id"], model["model_id"])["source_variant"] == model["source_variant"]
        assert storage.cg_image_path(scene["background_id"], cg["cg_id"]).is_file()


@pytest.mark.parametrize("invalid", [
    {"set_id": "../outside", "index": 1}, {"set_id": "7" * 32, "index": 0},
    {"set_id": "7" * 32, "index": True}, {"set_id": "7" * 32, "index": 3},
    {"set_id": "7" * 32, "index": 1, "variant_id": "night"},
    {"set_id": "7" * 32, "index": 1, "name": "Night"},
    {"set_id": "8" * 32, "index": 1},
])
def test_invalid_variant_references_do_not_write(source_variants, invalid):
    scene, _references = source_variants
    before = files(storage.root())
    with pytest.raises((ValueError, FileNotFoundError)):
        storage.image_path(scene["background_id"], scene["images"][0]["image_id"], invalid)
    assert files(storage.root()) == before


def test_variant_source_ownership_and_file_containment_are_checked(source_variants):
    scene, (morning, _night) = source_variants
    with pytest.raises(ValueError, match="another background source"):
        storage.image_path(scene["background_id"], scene["images"][1]["image_id"], morning)
    scene["generated_sets"][-1]["images"][1]["file"] = "../outside.png"
    write_document(scene)
    before = files(storage.root())
    with pytest.raises(storage.BackgroundStorageError):
        storage.image_path(scene["background_id"], scene["images"][0]["image_id"], morning)
    assert files(storage.root()) == before


def test_default_library_matches_the_latest_background_creator(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    assert storage.root() == tmp_path / "VNCCS" / "Backgrounds"


@pytest.fixture
def current_scene(scene):
    document = copy.deepcopy(scene)
    document.update(background_id="8" * 32, generated_sets=[], models=[], cgs=[])
    directory = storage.root() / document["name"]
    directory.mkdir(parents=True, exist_ok=True)
    for item, relative, color in zip(document["images"], ("Sources/V1.png", "source.png"), ("red", "blue")):
        item["file"] = relative
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), color).save(target)
    for set_id, version, color in (("6" * 32, "V1", "orange"), ("7" * 32, "", "yellow")):
        group = {"set_id": set_id, "version": version, "source_image_id": document["selected_image_id"],
                 "variants": [{"id": "morning", "name": "Morning", "time": "Morning", "weather": "Clear", "additional_details": ""}],
                 "settings": {"seed": 4}, "images": []}
        for index, name in enumerate(("Original", "Morning")):
            relative = f"Images/{version + '/' if version else ''}{index:02d}_{name}.png"
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 32), color).save(target)
            group["images"].append({"name": name, "file": relative, "width": 64, "height": 32,
                                    "revision": 7, "settings": {"seed": 4}})
        document["generated_sets"].append(group)
    path = directory / f"{document['name']}_config.json"
    def publish(value):
        encoded = json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n"
        path.write_bytes(encoded)
        path.with_suffix(".json.backup").write_bytes(encoded)
    publish(document)
    return types.SimpleNamespace(document=document, directory=directory, path=path, publish=publish)


def test_latest_sources_current_variants_and_archives_resolve_without_rewriting(current_scene):
    current = current_scene
    before = files(storage.root().parent)
    assert storage.load_background(current.document["background_id"]) == current.document
    assert storage.list_backgrounds() == [current.document]
    for image in current.document["images"]:
        assert storage.image_path(current.document["background_id"], image["image_id"]) == current.directory / image["file"]
    for group, expected in zip(current.document["generated_sets"], ((255, 165, 0), (255, 255, 0))):
        variant = {"set_id": group["set_id"], "index": 1, "variant_id": "morning", "name": "Morning"}
        path = storage.image_path(current.document["background_id"], current.document["selected_image_id"], variant)
        assert path == current.directory / group["images"][1]["file"]
        with Image.open(path) as image:
            assert image.getpixel((0, 0)) == expected
    assert files(storage.root().parent) == before


def test_models_cgs_and_config_use_the_current_background_layout(current_scene, tmp_path):
    current = current_scene
    document = add_model(current.document, tmp_path / "new")
    model = document["models"][-1]
    assert storage.model_asset_path(document["background_id"], model["model_id"], "d" * 32, "ply").is_relative_to(current.directory / "model")
    raw = Image.new("RGB", (64, 32))
    document = storage.commit_cg(document["background_id"], "Window", raw, raw, CAMERA, {}, model["model_id"], document["revision"])
    assert storage.cg_image_path(document["background_id"], document["cgs"][0]["cg_id"]).is_relative_to(current.directory / "CG")
    assert json.loads(current.path.read_bytes()) == document
    assert not (current.directory / f"{document['name']}.json").exists()
    for key in ("images", "editor", "generated_sets", "generator_settings", "future_owner_data"):
        assert document[key] == current.document[key]


@pytest.mark.parametrize("recoverable", [True, False])
def test_latest_config_recovery_never_falls_back_to_stale_old_json(current_scene, recoverable):
    current = current_scene
    (current.directory / f"{current.document['name']}.json").write_text(json.dumps({**current.document, "revision": 99}))
    current.path.write_bytes(b"{damaged")
    if not recoverable:
        current.path.with_suffix(".json.backup").write_bytes(b"{also damaged")
    before = files(storage.root().parent)
    if recoverable:
        document = storage.load_background(current.document["background_id"])
        assert document["_recovery"]["source"] == current.path.name + ".backup"
        assert document["revision"] == current.document["revision"]
        with pytest.raises(ValueError, match="read-only"):
            storage._save_settings(document["background_id"], {}, {}, document["revision"])
    else:
        with pytest.raises(storage.BackgroundStorageError):
            storage.load_background(current.document["background_id"])
    assert files(storage.root().parent) == before


@pytest.mark.parametrize("relative", ["../outside.png", "CG/frame.png", "Sources/V0.png", "Sources/../source.png"])
def test_latest_source_paths_are_bounded_and_invalid_documents_are_preserved(current_scene, relative):
    current = current_scene
    current.document["images"][0]["file"] = relative
    current.publish(current.document)
    before = files(storage.root().parent)
    with pytest.raises(storage.BackgroundStorageError):
        storage.load_background(current.document["background_id"])
    assert files(storage.root().parent) == before


@pytest.mark.parametrize("version", [None, False, "../Images", "V0", "V2"])
def test_latest_variant_version_cannot_escape_or_select_another_archive(current_scene, version):
    current = current_scene
    group = current.document["generated_sets"][0]
    group["version"] = version
    current.publish(current.document)
    before = files(storage.root().parent)
    with pytest.raises(storage.BackgroundStorageError):
        storage.background_set_image_path(current.document["background_id"], group["set_id"], 1)
    assert files(storage.root().parent) == before


def legacy_assets(document, tmp_path):
    """Keep the pre-migration wire identities and physical layout in one fixture."""
    document = add_model(document, tmp_path)
    raw = Image.new("RGB", (64, 32), "purple")
    document = storage.commit_cg(document["background_id"], "View 1", raw, raw, CAMERA, {},
                                 document["models"][0]["model_id"], document["revision"])
    directory = path_for(document).parent
    model, cg = document["models"][0], document["cgs"][0]
    model_path = directory / Path(model.pop("file")).parent
    legacy_model = directory / "model" / model["model_id"]
    model_path.rename(legacy_model)
    manifest_path = legacy_model / "scene.json"
    manifest = json.loads(manifest_path.read_bytes())
    for index, item in enumerate(manifest["objects"], 1):
        old, new = f"objects/Object {index}", f"objects/{item['object_id']}"
        (legacy_model / old).rename(legacy_model / new)
        item["files"] = {key: relative.replace(old + "/", new + "/", 1) for key, relative in item["files"].items()}
    manifest_path.write_text(json.dumps(manifest))
    (legacy_model / "user-note.txt").write_text("Preserve unknown files")
    cg_path = directory / Path(cg.pop("file")).parent
    cg.pop("raw_file")
    legacy_cg = cg_path / cg["cg_id"]
    legacy_cg.mkdir()
    for name in ("raw.png", "image.png"):
        (cg_path / name).rename(legacy_cg / name)
    (legacy_cg / "user-note.txt").write_text("Preserve unknown files")
    write_document(document)
    return document


def test_new_assets_have_readable_immutable_paths_with_collision_suffixes(scene, tmp_path):
    document = add_model(scene, tmp_path)
    first = document["models"][0]
    first_path = storage.model_asset_path(document["background_id"], first["model_id"], "d" * 32, "ply")
    assert first["file"] == "model/Original 1/scene.json"
    assert first_path.relative_to(path_for(document).parent).as_posix() == "model/Original 1/objects/Object 1/ply.ply"
    document = add_model(document, tmp_path)
    assert document["models"][1]["file"] == "model/Original 1 (2)/scene.json"
    raw = Image.new("RGB", (64, 32))
    for name, expected in (("View 1", "View 1"), ("view 1", "view 1 (2)"), ("View 1", "View 1 (3)")):
        document = storage.commit_cg(document["background_id"], name, raw, raw, CAMERA, {}, first["model_id"], None)
        assert document["cgs"][-1]["file"] == f"CG/{expected}/image.png"
        assert document["cgs"][-1]["raw_file"] == f"CG/{expected}/raw.png"
    assert first_path.read_bytes() == b"immutable Gaussian fixture"
    assert all(not storage._ID.fullmatch(path.name) for path in path_for(document).parent.rglob("*") if path.is_dir())


def test_legacy_assets_migrate_without_changing_workflow_identity_or_unrelated_data(scene, tmp_path):
    legacy = legacy_assets(scene, tmp_path)
    before = copy.deepcopy(legacy)
    directory = path_for(legacy).parent
    image_bytes = {item["file"]: (directory / item["file"]).read_bytes() for item in legacy["images"]}
    migrated = storage.load_background(legacy["background_id"])
    assert migrated["models"][0]["file"] == "model/Original 1/scene.json"
    assert migrated["cgs"][0]["file"] == "CG/View 1/image.png"
    for kind in ("models", "cgs"):
        for old, new in zip(before[kind], migrated[kind]):
            assert {key: new[key] for key in old} == old
    for key in before.keys() - {"models", "cgs", "revision"}:
        assert migrated[key] == before[key]
    model_id, cg_id = before["models"][0]["model_id"], before["cgs"][0]["cg_id"]
    manifest = storage.model_scene(legacy["background_id"], model_id)
    assert manifest["objects"][0]["object_id"] == "d" * 32
    assert manifest["objects"][0]["files"]["ply"] == "objects/Object 1/ply.ply"
    assert storage.model_asset_path(legacy["background_id"], model_id, "d" * 32, "ply").read_bytes() == b"immutable Gaussian fixture"
    assert storage.cg_image_path(legacy["background_id"], cg_id) == directory / "CG/View 1/image.png"
    for branch in ("model/Original 1", "CG/View 1"):
        assert (directory / branch / "user-note.txt").read_text() == "Preserve unknown files"
    assert all((directory / relative).read_bytes() == data for relative, data in image_bytes.items())
    assert all(not storage._ID.fullmatch(path.name) for path in directory.rglob("*") if path.is_dir())
    backup = json.loads(path_for(legacy).with_suffix(".json.backup").read_bytes())
    assert all((directory / item["file"]).is_file() for kind in ("models", "cgs") for item in backup[kind])
    after = files(directory)
    assert storage.load_background(legacy["background_id"]) == migrated
    assert files(directory) == after


@pytest.mark.parametrize("failure", ["root-move", "object-move", "manifest", "backup", "primary"])
def test_failed_legacy_model_migration_restores_every_original_byte(scene, tmp_path, monkeypatch, failure):
    legacy = legacy_assets(scene, tmp_path)
    directory = path_for(legacy).parent
    before = files(directory)
    replace = storage.os.replace
    fired = False
    def fail_once(source, destination):
        nonlocal fired
        destination = Path(destination)
        matches = failure == "root-move" and destination == directory / "model/Original 1"
        matches |= failure == "object-move" and destination.name == "Object 1"
        matches |= failure == "manifest" and destination == directory / "model/Original 1/scene.json"
        matches |= failure == "backup" and destination == path_for(legacy).with_suffix(".json.backup")
        matches |= failure == "primary" and destination == path_for(legacy)
        if matches and not fired:
            fired = True
            raise OSError("Injected migration interruption")
        return replace(source, destination)
    monkeypatch.setattr(storage.os, "replace", fail_once)
    with pytest.raises(OSError, match="Injected migration interruption"):
        storage.load_background(legacy["background_id"])
    assert fired
    assert files(directory) == before
    assert not (directory / ".cg-assets.pending").exists()
    assert storage.load_background(legacy["background_id"])["models"][0]["file"] == "model/Original 1/scene.json"


@pytest.mark.parametrize("failure", ["move", "primary"])
def test_legacy_cg_flattening_rolls_back_and_retries(scene, tmp_path, monkeypatch, failure):
    document = legacy_assets(scene, tmp_path)
    # Isolate the CG transaction from the preceding model transaction.
    model = document["models"][0]
    model["file"] = f"model/{model['model_id']}/scene.json"
    write_document(document)
    directory = path_for(document).parent
    before, replace, fired = files(directory), storage.os.replace, False
    def fail_once(source, target):
        nonlocal fired
        if not fired and Path(target) == (directory / "CG/View 1/raw.png" if failure == "move" else path_for(document)):
            fired = True
            raise OSError("CG migration failed")
        return replace(source, target)
    monkeypatch.setattr(storage.os, "replace", fail_once)
    with pytest.raises(OSError, match="CG migration failed"):
        storage.load_background(document["background_id"])
    assert fired and files(directory) == before
    assert storage.load_background(document["background_id"])["cgs"][0]["file"] == "CG/View 1/image.png"


def test_interrupted_legacy_migration_replays_journal_before_reopening(scene, tmp_path, monkeypatch):
    legacy = legacy_assets(scene, tmp_path)
    directory = path_for(legacy).parent
    recover, replace = storage._recover_assets, storage.os.replace
    monkeypatch.setattr(storage, "_recover_assets", lambda _directory: None)
    def interrupt(source, destination):
        if Path(destination).name == "Object 1":
            raise OSError("Process stopped after root rename")
        return replace(source, destination)
    monkeypatch.setattr(storage.os, "replace", interrupt)
    with pytest.raises(OSError, match="Process stopped"):
        storage.load_background(legacy["background_id"])
    assert (directory / ".cg-assets.pending/plan.json").exists()
    monkeypatch.setattr(storage, "_recover_assets", recover)
    monkeypatch.setattr(storage.os, "replace", replace)
    migrated = storage.load_background(legacy["background_id"])
    assert migrated["models"][0]["model_id"] == legacy["models"][0]["model_id"]
    assert migrated["models"][0]["file"] == "model/Original 1/scene.json"
    assert migrated["cgs"][0]["file"] == "CG/View 1/image.png"
    assert not (directory / ".cg-assets.pending").exists()


@pytest.mark.parametrize("operation", ["model", "cg"])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_interrupted_publication_recovers_on_next_load(scene, tmp_path, monkeypatch, operation, acknowledged):
    document = add_model(scene, tmp_path)
    directory = path_for(document).parent
    before = files(directory)
    recover, atomic = storage._recover_assets, storage.background._atomic
    monkeypatch.setattr(storage, "_recover_assets", lambda _directory: None)
    def fail_primary(path, data):
        if path == path_for(document):
            raise OSError("Interrupted before acknowledgement")
        return atomic(path, data)
    if not acknowledged:
        monkeypatch.setattr(storage.background, "_atomic", fail_primary)
    def publish():
        if operation == "model":
            return add_model(document, tmp_path)
        raw = Image.new("RGB", (64, 32))
        return storage.commit_cg(document["background_id"], "View 1", raw, raw, CAMERA, {}, document["models"][0]["model_id"], None)
    if acknowledged:
        expected = publish()
    else:
        with pytest.raises(OSError, match="Interrupted before acknowledgement"):
            publish()
        expected = document
    assert (directory / ".cg-assets.pending/plan.json").is_file()
    monkeypatch.setattr(storage, "_recover_assets", recover)
    monkeypatch.setattr(storage.background, "_atomic", atomic)
    assert storage.load_background(document["background_id"]) == expected
    assert not (directory / ".cg-assets.pending").exists()
    if not acknowledged:
        assert files(directory) == before


@pytest.mark.parametrize("operation", ["model", "cg"])
def test_failed_new_publication_leaves_no_assets_or_metadata(scene, tmp_path, monkeypatch, operation):
    document = add_model(scene, tmp_path)
    directory = path_for(document).parent
    before = files(directory)
    atomic = storage.background._atomic
    def fail(path, data):
        if path == path_for(document):
            raise OSError("Publication failed")
        return atomic(path, data)
    monkeypatch.setattr(storage.background, "_atomic", fail)
    with pytest.raises(OSError, match="Publication failed"):
        if operation == "model":
            add_model(document, tmp_path)
        else:
            raw = Image.new("RGB", (64, 32))
            storage.commit_cg(document["background_id"], "View 1", raw, raw, CAMERA, {}, document["models"][0]["model_id"], None)
    assert files(directory) == before
    assert not (directory / ".cg-assets.pending").exists()


@pytest.mark.parametrize("path", ["../outside/scene.json", "model/../scene.json", "model/name/../scene.json", "model\\name\\scene.json", "model/name/image.png", "C:/name/scene.json"])
def test_readable_model_paths_reject_escape_and_invalid_shapes(scene, tmp_path, path):
    document = add_model(scene, tmp_path)
    document["models"][0]["file"] = path
    write_document(document)
    before = files(storage.root())
    with pytest.raises(storage.BackgroundStorageError):
        storage.load_background(document["background_id"])
    assert files(storage.root()) == before


def test_asset_paths_cannot_alias_another_record_on_case_insensitive_systems(scene, tmp_path):
    document = add_model(scene, tmp_path)
    document["models"].append({**document["models"][0], "model_id": "9" * 32, "file": document["models"][0]["file"].upper().replace("MODEL/", "model/").replace("SCENE.JSON", "scene.json")})
    write_document(document)
    before = files(storage.root())
    with pytest.raises(storage.BackgroundStorageError):
        storage.load_background(document["background_id"])
    assert files(storage.root()) == before
