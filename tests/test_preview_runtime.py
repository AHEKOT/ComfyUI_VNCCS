"""Standalone preview scheduling must work in lightweight CI without torch."""

import asyncio
import threading

import pytest

from nodes.preview_runtime import run_preview_job


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
