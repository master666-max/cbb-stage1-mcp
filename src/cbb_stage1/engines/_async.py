"""Run a coroutine from sync code without assuming the caller owns the loop.

Used by engines that expose the synchronous ``Engine.extract`` contract while
their providers are async (AI Reader, LPMM, OpenIE).  If the current thread
already runs an event loop, the coroutine is executed in a fresh loop on a
short-lived worker thread; otherwise ``asyncio.run`` is used directly.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


def run_coroutine_sync(coro):
    """Run *coro* to completion and return its result."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
            box["error"] = exc

    thread = threading.Thread(target=_runner, name="cbb-sync-bridge")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
