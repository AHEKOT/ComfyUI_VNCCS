"""Standalone preview scheduling must work in lightweight CI without torch."""

import asyncio
import threading
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from nodes.preview_runtime import run_preview_job
from runtime_cleanup_helpers import dynamic_runtime


@pytest.fixture
def progress_runtime(monkeypatch):
    import comfy.utils

    events, interruptions = [], []

    def workflow_hook(*args, **kwargs):
        events.append((args, kwargs))

    def check_interrupt():
        interruptions.append(threading.get_ident())

    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", workflow_hook, raising=False)
    monkeypatch.setattr(comfy.utils, "set_progress_bar_global_hook",
                        lambda hook: setattr(comfy.utils, "PROGRESS_BAR_HOOK", hook), raising=False)
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        SimpleNamespace(throw_exception_if_processing_interrupted=check_interrupt))

    class ProgressBar:
        # ComfyUI bars capture the global hook when constructed.
        def __init__(self):
            self.hook = comfy.utils.PROGRESS_BAR_HOOK

        def update(self, value, **kwargs):
            self.hook(value, 10, None, **kwargs)

    return comfy.utils, workflow_hook, ProgressBar, events, interruptions


@pytest.mark.parametrize("fail", [False, True])
def test_standalone_progress_does_not_enter_workflow_stream(progress_runtime, fail):
    utils, workflow_hook, ProgressBar, events, interruptions = progress_runtime
    started, release = threading.Event(), threading.Event()
    existing_bar = ProgressBar()
    owner = []

    def preview():
        owner.append(threading.get_ident())
        bar = ProgressBar()
        bar.update(1)
        bar.update(2, node_id="preview-node")
        started.set()
        assert release.wait(3)
        if fail:
            raise RuntimeError("sampler failed")
        return "image"

    async def scenario():
        job = asyncio.create_task(run_preview_job(preview))
        try:
            for _ in range(1000):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            assert started.is_set()
            assert events == [], "Standalone progress must not be assigned to the last workflow"
            existing_bar.update(3)
            concurrent_bar = ProgressBar()
            concurrent_bar.update(4, node_id="workflow-node")
        finally:
            release.set()
            if fail:
                with pytest.raises(RuntimeError, match="sampler failed"):
                    await job
            else:
                assert await job == "image"
        assert utils.PROGRESS_BAR_HOOK is workflow_hook
        concurrent_bar.update(5)
        ProgressBar().update(6)

    asyncio.run(scenario())
    assert events == [((3, 10, None), {}), ((4, 10, None), {"node_id": "workflow-node"}),
                      ((5, 10, None), {}), ((6, 10, None), {})]
    assert interruptions == owner * 2


def test_standalone_progress_preserves_interrupt_and_restores_hook(progress_runtime, monkeypatch):
    utils, workflow_hook, ProgressBar, _, _ = progress_runtime
    failure = RuntimeError("processing interrupted")

    def interrupt():
        raise failure

    monkeypatch.setattr(sys.modules["comfy.model_management"], "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run_preview_job(lambda: ProgressBar().update(1)))
    assert caught.value is failure
    assert utils.PROGRESS_BAR_HOOK is workflow_hook


def test_preview_cleanup_does_not_overwrite_a_new_hook(progress_runtime):
    utils, _, _, _, _ = progress_runtime
    replacement = lambda *args, **kwargs: None
    asyncio.run(run_preview_job(lambda: utils.set_progress_bar_global_hook(replacement)))
    assert utils.PROGRESS_BAR_HOOK is replacement


def test_preview_worker_keeps_event_loop_responsive_and_serializes_jobs():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls = []
        main_thread = threading.get_ident()

        def first():
            calls.append(("first", threading.get_ident()))
            started.set()
            assert release.wait(3), "HTTP loop could not release the preview worker"
            return "preview"

        one = asyncio.create_task(run_preview_job(first))
        try:
            for _ in range(1000):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            assert started.is_set()
            two = asyncio.create_task(run_preview_job(lambda: calls.append(("second", threading.get_ident()))))
            await asyncio.sleep(.01)
            assert len(calls) == 1
            assert calls[0][1] != main_thread
        finally:
            release.set()
        assert await one == "preview"
        await two
        assert [name for name, _ in calls] == ["first", "second"]

    asyncio.run(scenario())


def test_preview_failure_is_reported_and_does_not_poison_the_next_job():
    def fail():
        raise RuntimeError("sampler failed")

    async def scenario():
        with pytest.raises(RuntimeError, match="sampler failed"):
            await run_preview_job(fail)
        assert await run_preview_job(lambda: 42) == 42

    asyncio.run(scenario())


@pytest.mark.parametrize("first_kind", ["preview", "workflow"])
@pytest.mark.parametrize("fail", [False, True])
def test_preview_and_workflow_share_model_lock_and_preserve_stage_cleanup(dynamic_runtime, first_kind, fail):
    from nodes.generator_context import generator_execution_lock
    from nodes.runtime_cleanup import inference_stage

    _, events, pending = dynamic_runtime
    started, release, second_entered = threading.Event(), threading.Event(), threading.Event()

    def first():
        boundary = generator_execution_lock("first", "first:first") if first_kind == "workflow" else nullcontext()
        with boundary:
            # Preview Regenerate can nest a generator lock and native stages.
            with generator_execution_lock("first", "first:first"):
                with inference_stage():
                    pending.append(object())
            assert not pending
            assert events == ["prefetch", "cast_buffers", "watermarks"]
            # Whole jobs must remain isolated even between native stages.
            started.set()
            assert release.wait(3), "HTTP loop could not release model execution"
            if fail:
                raise RuntimeError("generation failed")
        return "first"

    def second():
        boundary = generator_execution_lock("second", "second:second") if first_kind == "preview" else nullcontext()
        with boundary:
            second_entered.set()
            assert not pending, "A previous inference stage still owns allocator resources"
            with inference_stage():
                pending.append(object())
            with inference_stage():
                assert not pending, "Preview job locking must preserve cleanup between stages"
                pending.append(object())
        return "second"

    async def scenario():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as workflow:
            def submit(kind, callback):
                if kind == "preview":
                    return asyncio.create_task(run_preview_job(callback))
                return loop.run_in_executor(workflow, callback)

            one = submit(first_kind, first)
            two = None
            try:
                for _ in range(1000):
                    if started.is_set():
                        break
                    await asyncio.sleep(.001)
                assert started.is_set()
                two = submit("workflow" if first_kind == "preview" else "preview", second)
                await asyncio.sleep(.05)
                assert not second_entered.is_set(), "Preview and workflow inference overlapped"
            finally:
                release.set()
                if fail:
                    with pytest.raises(RuntimeError, match="generation failed"):
                        await one
                else:
                    assert await one == "first"
                if two is not None:
                    assert await two == "second"

    asyncio.run(scenario())
    assert not pending
    assert events == ["prefetch", "cast_buffers", "watermarks"] * 3
