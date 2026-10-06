"""ComfyUI executor boundaries for inference performed inside VNCCS nodes."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock


_stage_depth = ContextVar("vnccs_inference_stage_depth", default=0)
# ponytail: one lock for shared ComfyUI model/allocator state; split only if the runtime isolates that state.
inference_lock = RLock()


def cleanup_runtime():
    """Release executor-owned resources without unloading reusable model weights."""
    try:
        import comfy.memory_management as memory_management
        import comfy.model_management as model_management
        import comfy.model_prefetch as model_prefetch
        import comfy_aimdo.model_vbar as model_vbar
    except ImportError:
        return  # Older ComfyUI versions do not use the dynamic allocator.
    if not getattr(memory_management, "aimdo_enabled", False):
        return
    model_prefetch.cleanup_prefetch_queues()
    model_management.reset_cast_buffers()
    model_vbar.vbars_reset_watermark_limits()


@contextmanager
def inference_stage():
    """Clean up after a synchronous stage, including failure and nested calls."""
    with inference_lock:
        depth = _stage_depth.get()
        token = _stage_depth.set(depth + 1)
        try:
            yield
        finally:
            _stage_depth.reset(token)
            if depth == 0:
                cleanup_runtime()
