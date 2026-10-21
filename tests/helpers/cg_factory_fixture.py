"""In-process Factory double; GPU inference stays owned and tested by Utils."""

import copy
import io
import json
import shutil
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def factory_fixture():
    factory = SimpleNamespace(_factory_root=lambda: Path("/unused"), _JOBS={})
    runtime = SimpleNamespace(_JOBS={}, _emit=lambda *args: None)

    class JobCancelled(RuntimeError):
        pass

    def resolve(scene_id):
        return factory._factory_root() / scene_id

    def save(scene):
        directory = resolve(scene["scene_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scene.json").write_text(json.dumps(scene))

    def load(scene_id):
        return json.loads((resolve(scene_id) / "scene.json").read_text())

    def create(name):
        scene = {"schema_version": 11, "scene_id": uuid.uuid4().hex, "name": name, "objects": []}
        save(scene)
        return scene

    def update(scene_id, values):
        scene = {**load(scene_id), **values}
        save(scene)
        return scene

    def settings(value):
        return {"num_gaussians": 131072, "steps": 20, "guidance_scale": 3.0, "seed": -1,
                "conditioning_resolution": 1024, "prevent_upscale": False, **value,
                "provider": "triposplat", "remove_background": False}

    def new_job(kind):
        job = {"job_id": uuid.uuid4().hex, "kind": kind, "status": "queued", "progress": 0,
               "message": "Queued", "error": "", "result": None, "cancel_event": threading.Event()}
        runtime._JOBS[job["job_id"]] = job
        return job

    def check(job):
        if job["cancel_event"].is_set():
            raise JobCancelled("Job cancelled")

    def run(job, callback):
        job["status"] = "running"
        try:
            check(job)
            result = callback(job)
            check(job)
            job.update(result=result, status="completed")
        except JobCancelled as error:
            job.update(status="cancelled", error=str(error))
        except Exception as error:
            job.update(status="failed", error=str(error))

    factory.resolve_scene_dir = resolve
    factory._save_scene = save
    factory.load_scene = load
    factory.create_scene = create
    factory.update_scene = update
    factory.delete_scene = lambda scene_id: shutil.rmtree(resolve(scene_id))
    factory._generation_settings = settings
    def decode(payload):
        if not payload:
            raise ValueError("Image upload is empty")
        return Image.open(io.BytesIO(payload))
    factory._decode_image = decode
    runtime._new_job = new_job
    runtime._run_job = run
    runtime._check_cancel = check
    runtime._job_public = lambda job: {key: copy.deepcopy(value) for key, value in job.items() if key != "cancel_event"}
    runtime._generate_object = lambda *args: None
    runtime.JobCancelled = JobCancelled
    return runtime, factory
