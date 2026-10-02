"""Run standalone preview jobs without submitting the workflow to ComfyUI."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


# A single worker also prevents preview requests from racing shared model caches.
_preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vnccs-preview")


async def run_preview_job(callback, *args, **kwargs):
    """Keep HTTP/progress processing responsive while one isolated job runs."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preview_executor, partial(callback, *args, **kwargs))
