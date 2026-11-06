"""Fire-and-forget events to the live agency board (board/server.py).

The call must never wait on, or break because of, the board.
"""

import asyncio
import logging
import os
import time

import aiohttp

logger = logging.getLogger("baton-board")
BOARD_URL = os.getenv("BATON_BOARD_URL", "http://localhost:8787/event")

_session: aiohttp.ClientSession | None = None
_tasks: set[asyncio.Task] = set()


async def _post(payload: dict) -> None:
    global _session
    try:
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.5))
        async with _session.post(BOARD_URL, json=payload):
            pass
    except Exception:
        logger.debug("board offline, skipped %s", payload.get("type"))


def emit(event: str, call: str = "", **data) -> None:
    payload = {"type": event, "call": call, "t": time.time(), **data}
    try:
        task = asyncio.get_running_loop().create_task(_post(payload))
    except RuntimeError:
        return
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
