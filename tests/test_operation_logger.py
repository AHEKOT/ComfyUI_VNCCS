"""Operation summaries must remain useful and quiet in lightweight CI."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import logging
from threading import Barrier
from types import SimpleNamespace

import pytest

from _vnccs.operation_logger import log_event, log_stage, logged_operation


def summaries(caplog):
    return [record.vnccs for record in caplog.records if record.name == "VNCCS"]


def test_operation_bounds_progress_and_preserves_result_and_signature(caplog):
    @logged_operation("Generator", "generate")
    def generate(unique_id, count=12):
        for current in range(count + 1):
            log_stage("sampling", current=current, total=count)
        log_stage("sampling", "done", current=count, total=count)
        return ("images", count)

    with caplog.at_level(logging.INFO, logger="VNCCS"):
        assert generate("42") == ("images", 12)
    events = summaries(caplog)
    assert list(inspect.signature(generate).parameters) == ["unique_id", "count"]
    assert len(events) == 7
    assert [event["current"] for event in events if event["event"] == "stage"] == [0, 3, 6, 9, 12]
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    assert len({event["operation"] for event in events}) == 1
    assert all(event["node_id"] == "42" for event in events)
    assert events[-1]["duration_s"] >= 0


def test_failure_keeps_stage_and_exception_without_default_traceback(caplog):
    error = RuntimeError("device failed")

    @logged_operation("Creator", "preview")
    def fail():
        log_stage("decoding")
        raise error

    with caplog.at_level(logging.INFO, logger="VNCCS"):
        with pytest.raises(RuntimeError) as caught:
            fail()
        log_event("after_failure", component="Storage")
    assert caught.value is error
    failed = next(event for event in summaries(caplog) if event["event"] == "failed")
    assert failed["stage"] == "decoding" and failed["error"] == "device failed"
    assert all(record.exc_info is None for record in caplog.records)
    assert "operation" not in summaries(caplog)[-1]


def test_debug_is_opt_in_and_messages_remain_single_line_and_bounded(caplog):
    with caplog.at_level(logging.INFO, logger="VNCCS"):
        log_event("raw_output", level="debug", output="hidden")
    assert summaries(caplog) == []
    with caplog.at_level(logging.DEBUG, logger="VNCCS"):
        log_event("raw_output", level="debug", output="line\n" + "x" * 400)
    assert len(summaries(caplog)[0]["output"]) == 240
    assert "\n" not in caplog.records[-1].getMessage()


def test_debug_failure_preserves_traceback(caplog):
    @logged_operation("Creator", "preview")
    def fail():
        raise ValueError("details")

    with caplog.at_level(logging.DEBUG, logger="VNCCS"):
        with pytest.raises(ValueError, match="details"):
            fail()
    assert caplog.records[-1].exc_info[0] is ValueError


def test_concurrent_and_nested_jobs_keep_separate_contexts(caplog):
    barrier = Barrier(2)

    @logged_operation("Wizard", "describe")
    def child():
        log_event("description_ready")

    @logged_operation("Creator", "preview")
    def parent(unique_id):
        barrier.wait(timeout=3)
        child()
        log_stage("saving")

    with caplog.at_level(logging.INFO, logger="VNCCS"), ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(parent, ["1", "2"]))
        log_event("outside", component="Storage")
    events = summaries(caplog)
    parent_events = [event for event in events if event["component"] == "Creator"]
    assert len({event["operation"] for event in parent_events}) == 2
    for node_id in ("1", "2"):
        assert len({event["operation"] for event in parent_events if event["node_id"] == node_id}) == 1
    child_events = [event for event in events if event["component"] == "Wizard"]
    assert len({event["operation"] for event in child_events}) == 2
    assert "operation" not in events[-1]


def test_http_error_is_a_failed_operation_with_its_reason(caplog):
    @logged_operation("Cloner", "wizard")
    def wizard(post):
        return SimpleNamespace(status=500, text=json.dumps({"error": "MODEL_MISSING", "message": "Download the model"}))

    with caplog.at_level(logging.INFO, logger="VNCCS"):
        assert wizard({"node_id": "7"}).status == 500
    last = summaries(caplog)[-1]
    assert last["event"] == "failed" and last["http_status"] == 500
    assert last["error"] == "Download the model" and last["node_id"] == "7"
