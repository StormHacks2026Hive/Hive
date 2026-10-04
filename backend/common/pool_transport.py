"""Adapter to the existing browser pool for its validated 1D compute ABI."""

import asyncio
from typing import Any


async def submit_compute(pool: Any, request: Any, timeout: float = 120.0) -> bytes:
    """Submit via existing leases/result validation; retain browser ABI bounds."""
    job = pool.create(request)
    try:
        async with asyncio.timeout(timeout):
            while job.status in ("queued", "running"):
                await asyncio.sleep(0.05)
        if job.status != "done":
            raise RuntimeError(job.error or job.status)
        return bytes(job.image)
    except (TimeoutError, asyncio.CancelledError):
        pool.finish(job, "cancelled", "Transport wait cancelled")
        await pool.sweep()
        raise
