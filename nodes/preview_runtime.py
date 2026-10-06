"""Run standalone preview jobs without submitting the workflow to ComfyUI."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from .runtime_cleanup import inference_lock
from ..operation_logger import log_event

# A single worker also prevents preview requests from racing shared model caches.
_preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vnccs-preview")


def _run_preview_job(callback, *args, **kwargs):
    # Hold the same lock as workflow generation, without merging stage cleanup.
    with inference_lock:
        return callback(*args, **kwargs)


async def run_preview_job(callback, *args, **kwargs):
    """Keep HTTP/progress processing responsive while one isolated job runs."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preview_executor, partial(_run_preview_job, callback, *args, **kwargs))


async def run_wizard_job(callback, payload, kind):
    """Serialize wizard inference with previews and publish scoped job stages."""
    def emit(status, message):
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("vnccs.wizard.stage", {
                "node_id": str(payload.get("node_id", "")),
                "request_id": str(payload.get("request_id", "")),
                "kind": kind, "status": status, "message": message,
            })
        except (AttributeError, ImportError):
            pass

    def perform():
        emit("running", "Loading model and generating description")
        try:
            response = callback(payload)
        except Exception:
            emit("error", "Description generation failed")
            raise
        failed = response.status >= 400
        emit("error" if failed else "done", "Description generation failed" if failed else "Description ready")
        return response

    emit("queued", "Waiting for the model worker")
    log_event("queued", component="Wizard", kind=kind, node_id=payload.get("node_id"),
              request_id=payload.get("request_id"))
    return await run_preview_job(perform)
