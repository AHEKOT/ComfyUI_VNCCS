"""CG jobs and HTTP routes preserve the main extension's scene ownership."""

import asyncio
import copy
import io
import json
import sys
import types
from unittest import mock

import pytest
from test_background_creator import web
from PIL import Image

from helpers.backend_package import service_package
from helpers.cg_factory_fixture import factory_fixture


load = service_package("vnccs_cg_service_test")
service = load("nodes.cg_creator")
library = load("nodes.cg_creator")
generation = load("nodes.cg_creator")
routes_module = load("nodes.cg_creator")
runtime, factory = factory_fixture()
CAMERA = {"position": [0, 0, 3], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 42}
MAIN_FIELDS = ("editor", "images", "selected_image_id", "generator_settings", "generated_sets", "future_owner_data")


def png(size=(128, 64), color="red"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def write_document(document):
    path = library.root() / document["name"] / f"{document['name']}_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2).encode() + b"\n"
    path.write_bytes(encoded)
    path.with_suffix(".json.backup").write_bytes(encoded)
    return path


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    monkeypatch.setattr(library.background, "root", library.root)
    monkeypatch.setattr(service, "provider", lambda: {"runtime": runtime, "storage": factory, "web_root": tmp_path / "web"})
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {}, raising=False)
    monkeypatch.setattr(factory, "_factory_root", lambda: tmp_path / "factory")
    monkeypatch.setattr(service, "_JOBS", {})
    factory_jobs = {}
    monkeypatch.setattr(runtime, "_JOBS", factory_jobs)
    monkeypatch.setattr(factory, "_JOBS", factory_jobs)
    worker = mock.Mock()
    monkeypatch.setattr(service, "_WORKER", worker)
    monkeypatch.setattr(runtime, "_emit", mock.Mock())
    monkeypatch.setattr(generation, "progress", lambda _job_id: {"stage": "unknown"})
    monkeypatch.setattr(generation, "interrupt", mock.Mock(return_value=False))
    generated = mock.Mock(return_value=(Image.new("RGB", (128, 64), "blue"),
        {"settings": generation.normalize_settings({}), "prompt": "Finished camera frame", "model": "qi2"}))
    monkeypatch.setattr(generation, "generate_image", generated)
    document = {"schema_version": 2, "background_id": "a" * 32, "name": "Quiet room", "revision": 7,
        "archived": False, "editor": {"description": "Keep the main generator scene", "future": {"x": 3}},
        "images": [{"image_id": "b" * 32, "file": "Sources/V1.png", "width": 128, "height": 64,
            "settings": {"generator": "main", "seed": 3}, "preview_revision": 4},
            {"image_id": "c" * 32, "file": "source.png", "width": 128, "height": 64,
            "settings": {"generator": "main", "seed": 4}, "preview_revision": 1}],
        "selected_image_id": "c" * 32, "models": [], "cgs": [],
        "generator_settings": {"model": "qi2", "sampling": {"steps": 25}, "future": [1, 2]},
        "generated_sets": [],
        "future_owner_data": {"revision": "unrelated", "values": ["keep", "all"]}}
    write_document(document)
    for index, image in enumerate(document["images"]):
        target = library.root() / document["name"] / image["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png(color="red" if index == 0 else "green"))

    def run_pending():
        function, job, callback = worker.submit.call_args.args
        assert function is runtime._run_job
        function(job, callback)
        return job

    return types.SimpleNamespace(document=document, worker=worker, generated=generated,
                                 run_pending=run_pending, tmp_path=tmp_path)


def with_generated_set(workspace, document=None):
    document = copy.deepcopy(document or workspace.document)
    set_id = "7" * 32
    group = {"set_id": set_id, "version": "", "source_image_id": document["images"][0]["image_id"],
        "variants": [{"id": "morning", "name": "Morning"}, {"id": "night", "name": "Night"}],
        "settings": {"future": "Keep main set settings"}, "images": []}
    for index, (name, color) in enumerate((("Original", "red"), ("Morning", "yellow"), ("Night", "navy"))):
        relative = f"Images/{index:02d}_{name}.png"
        target = library.root() / document["name"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png(color=color))
        group["images"].append({"name": name, "file": relative, "width": 128, "height": 64,
                                "revision": document["revision"], "settings": {"seed": index}})
    document["generated_sets"].append(group)
    write_document(document)
    return document


def source_variant(document, index=1):
    group = document["generated_sets"][-1]
    variant = group["variants"][index - 1]
    return {"set_id": group["set_id"], "index": index, "variant_id": variant["id"], "name": variant["name"]}


def with_model(workspace, document=None, image_id=None, source_variant=None):
    document = document or workspace.document
    source = workspace.tmp_path / "source-scene"
    object_id = "d" * 32
    assets = source / "objects" / object_id
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "model.ply").write_bytes(b"immutable Gaussian fixture")
    scene = {"schema_version": 11, "scene_id": "e" * 32, "camera": CAMERA,
        "objects": [{"object_id": object_id, "source_camera": CAMERA,
                     "files": {"ply": f"objects/{object_id}/model.ply"}}]}
    return library.commit_model(document["background_id"], image_id or document["images"][0]["image_id"],
                                source, scene, {"seed": 1}, document["revision"], source_variant=source_variant)


def job_payload(document, action="cg", **changes):
    model = document["models"][-1] if document.get("models") else {}
    return {"action": action, "revision": document["revision"],
        "image_id": model.get("source_image_id", document["images"][0]["image_id"]),
        "source_variant": copy.deepcopy(model.get("source_variant")),
        "model_id": model.get("model_id", ""), "name": "Window", "camera": copy.deepcopy(CAMERA),
        "settings": copy.deepcopy(generation.DEFAULT_SETTINGS), **changes}


def assert_main_fields_preserved(before, after):
    assert {key: after[key] for key in MAIN_FIELDS} == {key: before[key] for key in MAIN_FIELDS}


@pytest.mark.parametrize("name", [None, "", "   "])
def test_cg_creation_requires_an_explicit_scene_name(workspace, name):
    document = with_model(workspace)
    payload = job_payload(document, name=name)
    if name is None:
        payload.pop("name")
    with pytest.raises(ValueError, match="name"):
        service.start_job(document["background_id"], payload, png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("payload", [None, [], {"action": "background"}, {"action": "wizard"},
    {"action": "upload"}, {"action": "create"}, {"action": "export"}, {"action": "unknown"}])
def test_only_model_and_cg_actions_are_accepted(workspace, payload):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(workspace.document["background_id"]) == workspace.document


@pytest.mark.parametrize("revision", [None, True, 0, 999])
def test_job_uses_current_disk_revision_instead_of_workflow_metadata(workspace, revision):
    service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model", revision=revision))
    values = workspace.worker.submit.call_args.args[2]
    newer = copy.deepcopy(workspace.document)
    newer["revision"] += 1
    write_document(newer)
    assert workspace.run_pending()["status"] == "failed"
    assert library.load_background(newer["background_id"]) == newer



@pytest.mark.parametrize("condition", ["archived", "recovered"])
def test_read_only_scene_fails_before_inference(workspace, condition):
    document = workspace.document
    if condition == "archived":
        write_document({**document, "archived": True})
    else:
        (library.root() / document["name"] / f"{document['name']}_config.json").write_bytes(b"{damaged")
    with pytest.raises(ValueError, match="read-only"):
        service.start_job(document["background_id"], job_payload(document, "model"))
    workspace.worker.submit.assert_not_called()


def test_settings_save_changes_only_cg_owned_settings(workspace):
    document = workspace.document
    updated = service.save_settings(document["background_id"], {"revision": document["revision"],
        "model_settings": {"seed": 12, "provider": "trellis2", "remove_background": "1"},
        "cg_settings": {"prompt": "Repair the captured room", "generation": {"steps": 12}},
        "editor": {"description": "must not overwrite"}, "images": [], "generated_sets": []})
    saved = library.load_background(document["background_id"])
    assert_main_fields_preserved(document, saved)
    assert saved["revision"] == document["revision"] + 1
    assert saved["model_settings"]["seed"] == 12
    assert saved["model_settings"]["provider"] == "triposplat"
    assert saved["model_settings"]["remove_background"] is False
    assert saved["cg_settings"]["prompt"] == "Repair the captured room"
    assert updated["cg_settings"]["generation"]["steps"] == 12


def test_main_document_without_model_or_cg_arrays_is_read_without_rewriting(workspace):
    document = copy.deepcopy(workspace.document)
    document.pop("models")
    document.pop("cgs")
    path = write_document(document)
    original = path.read_bytes()
    public = service.public_document(library.load_background(document["background_id"]))
    assert public["models"] == [] and public["cgs"] == []
    assert "models" not in document and "cgs" not in document
    assert path.read_bytes() == original
    updated = service.save_settings(document["background_id"], {"revision": document["revision"]})
    assert updated["models"] == [] and updated["cgs"] == []
    saved = library.load_background(document["background_id"])
    assert "models" not in saved and "cgs" not in saved
    assert_main_fields_preserved(document, saved)
    status = service.start_job(document["background_id"], job_payload(saved, "model"))
    assert status["status"] == "queued"


def test_public_generated_set_urls_use_local_routes_and_preserve_main_metadata(workspace):
    document = with_generated_set(workspace)
    before = copy.deepcopy(document)
    public = service.public_document(document)
    group = public["generated_sets"][-1]
    base = f"{service.API_BASE}/backgrounds/{document['background_id']}/sets/{group['set_id']}"
    assert [image["url"] for image in group["images"]] == [f"{base}/{index}?v={document['revision']}" for index in range(3)]
    assert group["settings"] == document["generated_sets"][-1]["settings"]
    assert document == before == library.load_background(document["background_id"])
    assert public["images"][0]["url"].endswith("?v=4")


def test_regenerated_variant_revision_updates_the_preview_url_and_loaded_pixels(workspace):
    document = with_generated_set(workspace)
    group = document["generated_sets"][-1]
    previous_url = service.public_document(document)["generated_sets"][-1]["images"][1]["url"]
    entry = group["images"][1]
    entry["revision"] += 1
    document["revision"] += 1
    path = library.root() / document["name"] / entry["file"]
    path.write_bytes(png(color="blue"))
    write_document(document)
    current = service.public_document(library.load_background(document["background_id"]))
    current_url = current["generated_sets"][-1]["images"][1]["url"]
    assert current_url != previous_url and current_url.endswith(f"?v={entry['revision']}")
    with Image.open(library.image_path(document["background_id"], group["source_image_id"], source_variant(document))) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_cg_job_captures_settings_camera_and_frame_without_changing_main_variants(workspace):
    document = with_model(workspace)
    payload = job_payload(document)
    payload["settings"]["generation"]["steps"] = 12
    service.start_job(document["background_id"], payload, png())
    payload["camera"]["position"][0] = 99
    payload["settings"]["generation"]["steps"] = 99
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    settings, _job_id, _event, reference, background = workspace.generated.call_args.args
    assert settings["generation"]["steps"] == 12
    assert reference.size == (128, 64) and reference.getpixel((0, 0)) == (255, 0, 0)
    assert background.size == (128, 64) and background.getpixel((0, 0)) == (255, 0, 0)
    saved = library.load_background(document["background_id"])
    assert_main_fields_preserved(document, saved)
    assert saved["models"] == document["models"]
    cg = saved["cgs"][-1]
    assert cg["camera"] == CAMERA and cg["model_id"] == document["models"][-1]["model_id"]
    assert cg["source_image_id"] == document["models"][-1]["source_image_id"]
    assert cg["source_image_id"] != document["selected_image_id"]
    with Image.open(library.cg_image_path(document["background_id"], cg["cg_id"])) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    assert job["result"]["document"]["cgs"][-1]["url"].endswith(f"/{cg['cg_id']}/image")


def test_cg_rejects_model_from_another_source_variant(workspace):
    document = with_model(workspace)
    with pytest.raises(ValueError, match="another background variant"):
        service.start_job(document["background_id"], job_payload(document, image_id=document["images"][1]["image_id"]), png())
    workspace.worker.submit.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("requested", ["original", "night"])
def test_cg_rejects_model_from_another_generated_variant_of_the_same_image(workspace, requested):
    document = with_generated_set(workspace)
    morning = source_variant(document)
    document = with_model(workspace, document, source_variant=morning)
    different = source_variant(document, 2) if requested == "night" else None
    with pytest.raises(ValueError, match="another background variant"):
        service.start_job(document["background_id"], job_payload(document, source_variant=different), png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


def test_cg_job_binds_saved_frame_to_the_exact_generated_variant(workspace):
    document = with_generated_set(workspace)
    morning = source_variant(document)
    document = with_model(workspace, document, source_variant=morning)
    payload = job_payload(document)
    service.start_job(document["background_id"], payload, png())
    payload["source_variant"].update(source_variant(document, 2))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    reference, background = workspace.generated.call_args.args[-2:]
    assert background.getpixel((0, 0)) == (255, 255, 0)
    assert reference.getpixel((0, 0)) == (255, 0, 0)
    saved = library.load_background(document["background_id"])
    assert saved["cgs"][-1]["source_variant"] == morning
    assert saved["cgs"][-1]["source_image_id"] == document["images"][0]["image_id"]
    assert_main_fields_preserved(document, saved)


def test_model_job_rejects_a_generated_set_owned_by_another_base_image(workspace):
    document = with_generated_set(workspace)
    with pytest.raises(ValueError, match="another background source"):
        service.start_job(document["background_id"], job_payload(document, "model",
            image_id=document["images"][1]["image_id"], source_variant=source_variant(document)))
    workspace.worker.submit.assert_not_called()
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("variant", [[], {"set_id": "../outside", "index": 1},
    {"set_id": "7" * 32, "index": True}, {"set_id": "7" * 32, "index": 65}])
def test_invalid_variant_metadata_is_rejected_before_queuing(workspace, variant):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model", source_variant=variant))
    workspace.worker.submit.assert_not_called()


def test_cg_rejects_model_owned_by_another_background(workspace):
    document = with_model(workspace)
    other = {**copy.deepcopy(workspace.document), "background_id": "9" * 32, "name": "Other room"}
    write_document(other)
    other = with_model(workspace, other)
    with pytest.raises(FileNotFoundError):
        service.start_job(document["background_id"], job_payload(document, model_id=other["models"][-1]["model_id"]), png())
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()


@pytest.mark.parametrize("camera,frame,name", [(None, png(), "View"), ({}, png(), "View"),
    (CAMERA, None, "View"), (CAMERA, png((127, 64)), "View"), (CAMERA, png((32, 16)), "View"),
    (CAMERA, png(), "../bad")])
def test_invalid_cg_inputs_never_queue_inference(workspace, camera, frame, name):
    document = with_model(workspace)
    with pytest.raises((ValueError, TypeError)):
        service.start_job(document["background_id"], job_payload(document, camera=camera, name=name), frame)
    workspace.worker.submit.assert_not_called()
    workspace.generated.assert_not_called()


def test_busy_workspace_rejects_another_job_and_settings_save(workspace):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document), png())
    with pytest.raises(RuntimeError, match="busy"):
        service.start_job(document["background_id"], job_payload(document), png())
    with pytest.raises(RuntimeError, match="busy"):
        service.save_settings(document["background_id"], {"revision": document["revision"]})
    assert workspace.worker.submit.call_count == 1


def test_full_active_queue_rejects_work_without_dropping_jobs(workspace):
    service._JOBS.update({str(index): {"background_id": "other", "status": "queued"} for index in range(64)})
    with pytest.raises(RuntimeError, match="queue is full"):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model"))
    assert len(service._JOBS) == 64
    workspace.worker.submit.assert_not_called()


def test_cancelled_queued_cg_job_never_calls_models_or_writes(workspace):
    document = with_model(workspace)
    status = service.start_job(document["background_id"], job_payload(document), png())
    service.cancel_job(status["job_id"])
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    generation.interrupt.assert_called_once_with(status["job_id"])
    workspace.generated.assert_not_called()
    assert library.load_background(document["background_id"]) == document


def test_sampling_cancellation_is_translated_to_factory_job_cancellation(workspace):
    document = with_model(workspace)
    workspace.generated.side_effect = generation.GenerationCancelled("CG generation cancelled")
    service.start_job(document["background_id"], job_payload(document), png())
    job = workspace.run_pending()
    assert job["status"] == "cancelled" and job["error"] == "CG generation cancelled"
    assert library.load_background(document["background_id"]) == document


def test_stop_after_sampling_prevents_committing_a_finished_frame(workspace):
    document = with_model(workspace)
    def generate(_settings, _job_id, event, _reference, _background):
        event.set()
        return Image.new("RGB", (128, 64)), {}
    workspace.generated.side_effect = generate
    service.start_job(document["background_id"], job_payload(document), png())
    assert workspace.run_pending()["status"] == "cancelled"
    assert library.load_background(document["background_id"]) == document


def test_late_cg_cannot_replace_a_new_main_scene_revision(workspace):
    document = with_model(workspace)
    service.start_job(document["background_id"], job_payload(document), png())
    newer = copy.deepcopy(document)
    newer["revision"] += 1
    newer["editor"]["new_main_edit"] = "Keep the newest main scene"
    write_document(newer)
    job = workspace.run_pending()
    assert job["status"] == "failed" and "changed" in job["error"]
    assert library.load_background(document["background_id"]) == newer


@pytest.mark.parametrize("use_variant", [False, True], ids=["original", "generated-variant"])
def test_model_job_copies_snapshot_and_releases_active_factory_scene_before_cleanup(workspace, monkeypatch, use_variant):
    document = with_generated_set(workspace) if use_variant else workspace.document
    selected = source_variant(document) if use_variant else None
    expected_variant = copy.deepcopy(selected)
    created = factory.create_scene("Temporary CG reconstruction")
    monkeypatch.setattr(factory, "create_scene", lambda _name: created)
    source = factory.resolve_scene_dir(created["scene_id"])
    def reconstruct(job, encoded, object_id, name, settings):
        assert name == "Background" and settings["provider"] == "triposplat"
        assert settings["remove_background"] is False and settings["capture_source_camera"] is True
        assert settings["seed"] == 3
        with Image.open(io.BytesIO(encoded)) as image:
            assert image.getpixel((0, 0)) == ((255, 255, 0) if use_variant else (255, 0, 0))
        manifest = factory.load_scene(job["scene_id"])
        path = source / "objects" / object_id / "model.ply"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"committed Gaussian fixture")
        manifest["objects"] = [{"object_id": object_id, "source_camera": CAMERA,
            "files": {"ply": f"objects/{object_id}/model.ply"}}]
        factory._save_scene(manifest)
    monkeypatch.setattr(runtime, "_generate_object", mock.Mock(side_effect=reconstruct))
    payload = job_payload(document, "model", model_settings={"seed": 3}, source_variant=selected)
    service.start_job(document["background_id"], payload)
    payload["model_settings"]["seed"] = 99
    if use_variant:
        payload["source_variant"].update(source_variant(document, 2))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert job["scene_id"] == "" and not source.exists()
    saved = library.load_background(document["background_id"])
    assert len(saved["models"]) == 1
    model = library.model_scene(document["background_id"], saved["models"][-1]["model_id"])
    assert model["camera"] == CAMERA and model["lighting"]["preset"] == "off"
    assert model["source_image_id"] == document["images"][0]["image_id"]
    assert model.get("source_variant") == saved["models"][-1].get("source_variant") == expected_variant
    object_id = model["objects"][0]["object_id"]
    assert library.model_asset_path(document["background_id"], model["model_id"], object_id, "ply").read_bytes() == b"committed Gaussian fixture"
    assert_main_fields_preserved(document, saved)
    workspace.generated.assert_not_called()


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_or_cancelled_reconstruction_removes_its_temporary_scene(workspace, monkeypatch, cancelled):
    document = workspace.document
    created = factory.create_scene("Temporary reconstruction")
    source = factory.resolve_scene_dir(created["scene_id"])
    monkeypatch.setattr(factory, "create_scene", lambda _name: created)
    failure = runtime.JobCancelled("Cancelled") if cancelled else RuntimeError("Reconstruction failed")
    monkeypatch.setattr(runtime, "_generate_object", mock.Mock(side_effect=failure), raising=False)
    before = library.load_background(document["background_id"])
    service.start_job(document["background_id"], job_payload(document, "model"))
    job = workspace.run_pending()
    assert job["status"] == ("cancelled" if cancelled else "failed")
    assert job["scene_id"] == "" and not source.exists()
    assert library.load_background(document["background_id"]) == before


def test_cancel_after_publication_keeps_the_completed_result(workspace, monkeypatch):
    document = with_model(workspace)
    publish = service.public_document
    def acknowledge(saved):
        job = next(iter(service._JOBS.values()))
        assert job["published"]
        service.cancel_job(job["job_id"])
        assert not job["cancel_event"].is_set()
        return publish(saved)
    monkeypatch.setattr(service, "public_document", acknowledge)
    service.start_job(document["background_id"], job_payload(document), png())
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    assert len(job["result"]["document"]["cgs"]) == 1
    generation.interrupt.assert_not_called()


def test_optional_main_owner_busy_guard_is_respected(workspace, monkeypatch):
    owner = types.SimpleNamespace(ensure_idle=mock.Mock(side_effect=RuntimeError("Main background generator is busy")))
    monkeypatch.setattr(library, "background_owner", lambda: owner)
    with pytest.raises(RuntimeError, match="Main background generator"):
        service.start_job(workspace.document["background_id"], job_payload(workspace.document, "model"))
    workspace.worker.submit.assert_not_called()


def test_main_inference_slot_is_held_during_refinement_and_released(workspace, monkeypatch):
    document = with_model(workspace)
    slot = mock.Mock()
    slot.acquire.return_value = True
    monkeypatch.setattr(library, "background_owner", lambda: types.SimpleNamespace(inference_lock=slot))
    generated = workspace.generated.return_value
    def infer(*_args):
        slot.acquire.assert_called_once_with(timeout=0.25)
        slot.release.assert_not_called()
        return generated
    workspace.generated.side_effect = infer
    service.start_job(document["background_id"], job_payload(document), png())
    assert workspace.run_pending()["status"] == "completed"
    slot.release.assert_called_once_with()


def test_waiting_for_main_inference_slot_can_be_cancelled_without_releasing_its_lock(workspace, monkeypatch):
    document = with_model(workspace)
    status = service.start_job(document["background_id"], job_payload(document), png())
    job = service._JOBS[status["job_id"]]
    slot = mock.Mock()
    def acquire(*, timeout):
        job["cancel_event"].set()
        return False
    slot.acquire.side_effect = acquire
    monkeypatch.setattr(library, "background_owner", lambda: types.SimpleNamespace(inference_lock=slot))
    assert workspace.run_pending()["status"] == "cancelled"
    slot.release.assert_not_called()
    workspace.generated.assert_not_called()
    generation.interrupt.assert_not_called()


class Request:
    def __init__(self, payload=None, *, info=None, raw=None, multipart=None, headers=None):
        self.raw = json.dumps({} if payload is None else payload).encode() if raw is None else raw
        self.match_info = info or {}
        self.headers = {"Content-Length": str(len(self.raw)), "X-VNCCS-CSRF": "1", **(headers or {})}
        self.content_length = len(self.raw)
        self.content_type = "multipart/form-data" if multipart is not None else "application/json"
        self.multipart = multipart

    async def read(self):
        return self.raw

    async def post(self):
        return self.multipart


@pytest.fixture
def routes(workspace, monkeypatch):
    monkeypatch.setattr(routes_module, "_REGISTERED", False)
    monkeypatch.setattr(sys.modules["aiohttp"], "web", web)
    table = web.RouteTableDef()
    routes_module.register_routes(table)
    return {(route.method, route.path): route.handler for route in table}


def call(routes, method, suffix, request):
    return asyncio.run(routes[(method, service.API_BASE + suffix)](request))


def test_http_cg_deletion_hides_only_the_target_and_keeps_workflow_assets(routes, workspace):
    document = with_model(workspace); image = Image.new("RGB", (128, 64))
    for name in ("Failed", "Keep"):
        document = library.commit_cg(document["background_id"], name, image, image, CAMERA, {},
                                     document["models"][0]["model_id"], document["revision"])
    cg = document["cgs"][0]; info = {"background_id": document["background_id"], "cg_id": cg["cg_id"]}
    response = call(routes, "DELETE", "/backgrounds/{background_id}/cg/{cg_id}", Request({"revision": document["revision"]}, info=info))
    assert response.status == 200
    public = json.loads(response.text)
    assert [item["cg_id"] for item in public["cgs"]] == [document["cgs"][1]["cg_id"]]
    assert public["revision"] == document["revision"] + 1
    assert_main_fields_preserved(service.public_document(document), public)
    assert public["models"] == document["models"]
    stored = library.load_background(document["background_id"])
    assert_main_fields_preserved(document, stored)
    assert stored["cgs"][0] == {**cg, "archived": True}
    assert library.cg_image_path(document["background_id"], cg["cg_id"]).is_file()
    workspace.worker.submit.assert_not_called()


@pytest.mark.parametrize("condition,status", [("missing", 404), ("busy", 409), ("read-only", 400)])
def test_http_cg_deletion_rejects_unsafe_or_stale_requests(routes, workspace, condition, status):
    document = with_model(workspace); image = Image.new("RGB", (128, 64))
    document = library.commit_cg(document["background_id"], "Keep", image, image, CAMERA, {},
                                 document["models"][0]["model_id"], document["revision"])
    if condition == "read-only":
        document["archived"] = True; write_document(document)
    if condition == "busy":
        service._JOBS["busy"] = {"background_id": document["background_id"], "status": "running"}
    info = {"background_id": document["background_id"], "cg_id": "0" * 32 if condition == "missing" else document["cgs"][0]["cg_id"]}
    response = call(routes, "DELETE", "/backgrounds/{background_id}/cg/{cg_id}",
                    Request({"revision": document["revision"] - (condition == "stale")}, info=info))
    assert response.status == status
    assert library.load_background(document["background_id"]) == document


def test_http_inventory_has_no_background_authoring_actions(routes):
    expected = {("GET", "/config"), ("GET", "/backgrounds"), ("GET", "/backgrounds/{background_id}"),
        ("PATCH", "/backgrounds/{background_id}"), ("POST", "/backgrounds/{background_id}/jobs"),
        ("GET", "/jobs/{job_id}"), ("POST", "/jobs/{job_id}/cancel"),
        ("DELETE", "/backgrounds/{background_id}/cg/{cg_id}"),
        ("GET", "/backgrounds/{background_id}/images/{image_id}"),
        ("GET", "/backgrounds/{background_id}/sets/{set_id}/{index}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}"),
        ("GET", "/backgrounds/{background_id}/models/{model_id}/objects/{object_id}/{kind}"),
        ("GET", "/backgrounds/{background_id}/cg/{cg_id}/{kind}")}
    assert set(routes) == {(method, service.API_BASE + path) for method, path in expected}


def test_http_config_library_read_and_cg_settings_save(routes, workspace, monkeypatch):
    monkeypatch.setattr(generation, "configuration", lambda: {"settings": generation.DEFAULT_SETTINGS, "model": "QI2"})
    response = call(routes, "GET", "/config", Request())
    assert response.status == 200 and json.loads(response.text)["model"] == "QI2"
    response = call(routes, "GET", "/backgrounds", Request())
    public = json.loads(response.text)["backgrounds"][0]
    assert public["images"][0]["url"].endswith(f"/{workspace.document['images'][0]['image_id']}?v=4")
    info = {"background_id": workspace.document["background_id"]}
    response = call(routes, "PATCH", "/backgrounds/{background_id}", Request({
        "revision": workspace.document["revision"], "cg_settings": {"prompt": "Keep the camera"},
        "editor": {"description": "Do not overwrite"}}, info=info))
    assert response.status == 200
    saved = library.load_background(info["background_id"])
    assert_main_fields_preserved(workspace.document, saved)
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
        Request({"revision": workspace.document["revision"]}, info=info))
    assert response.status == 200
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
    assert json.loads(response.text)["revision"] == saved["revision"] + 1


def test_http_generated_set_image_uses_the_local_saved_set_resolver(routes, workspace):
    document = with_generated_set(workspace)
    info = {"background_id": document["background_id"], "set_id": document["generated_sets"][-1]["set_id"], "index": "1"}
    with mock.patch.object(library, "background_set_image_path", wraps=library.background_set_image_path) as resolver:
        response = call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info))
    resolver.assert_called_once_with(info["background_id"], info["set_id"], 1)
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"] == "no-store"
    assert library.load_background(document["background_id"]) == document


@pytest.mark.parametrize("action", ["background", "wizard", "upload", "create", "delete"])
def test_http_jobs_cannot_author_backgrounds(routes, workspace, action):
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs",
        Request(job_payload(workspace.document, action), info={"background_id": workspace.document["background_id"]}))
    assert response.status == 400
    workspace.worker.submit.assert_not_called()


@pytest.mark.parametrize("raw", [b"[]", b"null", b"false", b"{invalid"])
def test_http_invalid_json_is_rejected_before_work(routes, workspace, raw):
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
        Request(raw=raw, info={"background_id": workspace.document["background_id"]}))
    assert response.status == 400 and json.loads(response.text)["error"]
    workspace.worker.submit.assert_not_called()
    assert library.load_background(workspace.document["background_id"]) == workspace.document


def test_http_request_size_limits_apply_before_work(routes, workspace):
    info = {"background_id": workspace.document["background_id"]}
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        call(routes, "PATCH", "/backgrounds/{background_id}",
            Request(info=info, headers={"Content-Length": str(service.MAX_REQUEST_BYTES + 1)}))
    response = call(routes, "PATCH", "/backgrounds/{background_id}", Request(info=info, raw=b" " * (256 * 1024 + 1)))
    assert response.status == 400
    workspace.worker.submit.assert_not_called()


def test_http_capture_job_status_and_cancel_use_cg_only_actions(routes, workspace):
    document = with_model(workspace)
    form = {"data": json.dumps(job_payload(document)), "image": types.SimpleNamespace(file=io.BytesIO(png()))}
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs",
        Request(info={"background_id": document["background_id"]}, multipart=form))
    assert response.status == 200
    status = json.loads(response.text)
    job_info = {"job_id": status["job_id"]}
    response = call(routes, "GET", "/jobs/{job_id}", Request(info=job_info))
    assert json.loads(response.text)["status"] == "queued"
    response = call(routes, "POST", "/jobs/{job_id}/cancel", Request(info=job_info))
    assert response.status == 200 and service._JOBS[status["job_id"]]["cancel_event"].is_set()
    assert workspace.run_pending()["status"] == "cancelled"
    response = call(routes, "GET", "/jobs/{job_id}", Request(info={"job_id": "missing"}))
    assert response.status == 404


def test_http_owned_assets_are_immutable_and_do_not_expose_model_paths(routes, workspace):
    document = with_model(workspace)
    model = document["models"][-1]
    raw = Image.new("RGB", (128, 64), "red")
    document = library.commit_cg(document["background_id"], "Window", raw, raw, CAMERA, {}, model["model_id"], document["revision"])
    info = {"background_id": document["background_id"], "image_id": document["images"][0]["image_id"],
            "model_id": model["model_id"], "object_id": "d" * 32, "kind": "ply", "cg_id": document["cgs"][-1]["cg_id"]}
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}", Request(info=info))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"] == "no-store"
    response = call(routes, "GET", "/backgrounds/{background_id}/models/{model_id}", Request(info=info))
    item = json.loads(response.text)["objects"][0]
    assert item["locked"] is True and "files" not in item
    assert item["urls"]["splat"].startswith(service.API_BASE)
    response = call(routes, "GET", "/backgrounds/{background_id}/models/{model_id}/objects/{object_id}/{kind}", Request(info=info))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"].endswith("immutable")
    response = call(routes, "GET", "/backgrounds/{background_id}/cg/{cg_id}/{kind}", Request(info={**info, "kind": "image"}))
    assert isinstance(response, web.FileResponse) and response.headers["Cache-Control"].endswith("immutable")
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}", Request(info={**info, "image_id": "../outside"}))
    assert response.status == 404


def test_http_cross_origin_mutations_fail_before_parsing_or_work(routes, workspace):
    request = Request({"revision": workspace.document["revision"]}, info={"background_id": workspace.document["background_id"]},
                      headers={"Sec-Fetch-Site": "cross-site"})
    request.read = mock.AsyncMock(side_effect=AssertionError("Unsafe body must not be parsed"))
    response = call(routes, "PATCH", "/backgrounds/{background_id}", request)
    assert response.status == 403
    request.read.assert_not_called()
    workspace.worker.submit.assert_not_called()
