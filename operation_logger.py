"""Compact VNCCS operation summaries through ComfyUI's standard logging setup.

Console messages use one line of key=value fields. LogRecord.vnccs contains
the same fields as a dictionary for handlers that need structured records.
DEBUG is off by default. Enable diagnostics with
logging.getLogger("VNCCS").setLevel(logging.DEBUG).
"""

from contextvars import ContextVar
from functools import wraps
import inspect
import json
import logging
import time
import uuid


logger = logging.getLogger("VNCCS")
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
_current_operation = ContextVar("vnccs_log_operation", default=None)


def log_event(event, *, component=None, level="info", exc_info=False, **fields):
    """Emit one searchable line; bound values and escape embedded newlines."""
    severity = getattr(logging, level.upper())
    if not logger.isEnabledFor(severity):
        return
    context = _current_operation.get() or {}
    values = {key: context[key] for key in ("component", "operation", "action", "node_id") if key in context}
    if component is not None:
        values["component"] = component
    values.update(event=event, **fields)
    parts = []
    structured = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str) and len(value) > 240:
            value = value[:237] + "..."
        structured[key] = value
        parts.append(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}")
    logger.log(severity, "[VNCCS] " + " ".join(parts),
               extra={"vnccs": structured},
               exc_info=True if exc_info and logger.isEnabledFor(logging.DEBUG) else None)


def log_stage(stage, status="running", *, current=None, total=None, **fields):
    """Log transitions and quarter milestones, never every image or poll."""
    context = _current_operation.get()
    elapsed = None
    if context is not None:
        if status == "error":
            return  # The operation boundary reports the exception once.
        context["stage"] = stage
        now = time.perf_counter()
        bucket = min(3, int(4 * current / total)) if current is not None and total and total > 0 else 0
        previous = context["stages"].get(stage)
        if previous and previous[:2] == (status, bucket):
            return
        started = previous[2] if previous and previous[0] != "done" else now
        context["stages"][stage] = (status, bucket, started)
        if status == "done":
            elapsed = round(now - started, 2)
    log_event("stage", stage=stage, status=status, current=current, total=total,
              duration_s=elapsed, level="error" if status == "error" else "info", **fields)


def logged_operation(component, action):
    """Keep synchronous jobs correlated, including nested jobs and failures."""
    def decorate(callback):
        signature = inspect.signature(callback)

        @wraps(callback)
        def perform(*args, **kwargs):
            arguments = signature.bind_partial(*args, **kwargs).arguments
            node_id = arguments.get("unique_id")
            payload = arguments.get("post", arguments.get("data"))
            if node_id is None and isinstance(payload, dict):
                node_id = payload.get("node_id")
            if isinstance(node_id, (list, tuple)) and len(node_id) == 1:
                node_id = node_id[0]
            context = {"component": component, "operation": uuid.uuid4().hex[:8],
                       "action": action, "node_id": node_id, "stages": {}}
            token = _current_operation.set(context)
            started = time.perf_counter()
            log_event("started")
            try:
                result = callback(*args, **kwargs)
                status = getattr(result, "status", None)
                failed = isinstance(status, int) and status >= 400
                error = None
                if failed:
                    try:
                        body = json.loads(getattr(result, "text", "") or "{}")
                        error = body.get("message") or body.get("error") if isinstance(body, dict) else None
                    except (ValueError, TypeError):
                        error = getattr(result, "text", None)
                log_event("failed" if failed else "completed", level="error" if failed else "info",
                          http_status=status, error=error, duration_s=round(time.perf_counter() - started, 2))
                return result
            except Exception as error:
                log_event("failed", level="error", stage=context.get("stage"),
                          error_type=type(error).__name__, error=str(error), exc_info=True,
                          duration_s=round(time.perf_counter() - started, 2))
                raise
            finally:
                _current_operation.reset(token)

        return perform
    return decorate
