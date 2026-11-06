"""Baton Health live agency board.

    uv run board/server.py          -> http://localhost:8787
    uv run board/server.py --demo         -> scripted Covered call (no LiveKit needed)
    uv run board/server.py --demo-baton   -> scripted Baton call, open /baton

The agent POSTs events to /event. The page listens on /stream (server-sent events)
and reads today's visits from the same store file the agent writes.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
import baton_core as bc  # noqa: E402

CLIENTS: set[asyncio.Queue] = set()
LIVE: list[dict] = []  # events of the current call, replayed on refresh


async def broadcast(ev: dict) -> None:
    if ev.get("type") == "call_started":
        LIVE.clear()
    LIVE.append(ev)
    for q in list(CLIENTS):
        q.put_nowait(ev)


async def post_event(req: web.Request) -> web.Response:
    await broadcast(await req.json())
    return web.json_response({"ok": True})


async def visits(_: web.Request) -> web.Response:
    store = bc.Store()
    out = []
    for v in sorted(store.visits, key=lambda v: v["started"], reverse=True)[:25]:
        visit = bc.Visit(v["client"], v["caregiver"], v["identified_by"])
        for k, val in v.items():
            if hasattr(visit, k):
                setattr(visit, k, val)
        out.append(visit.to_dict())
    return web.json_response(out)


async def stream(req: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await resp.prepare(req)
    q: asyncio.Queue = asyncio.Queue()
    CLIENTS.add(q)
    try:
        for ev in LIVE:
            await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=15)
                await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
            except asyncio.TimeoutError:
                await resp.write(b": ping\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        CLIENTS.discard(q)
    return resp


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(ROOT / "index.html")  # Covered


async def baton_page(_: web.Request) -> web.FileResponse:
    return web.FileResponse(ROOT / "baton.html")


# ----------------------------------------------------------------- demo mode
async def run_baton_demo(app: web.Application) -> None:
    """Replays a realistic difficult Baton call through the real agent tools."""
    import os
    import tempfile

    os.environ["BATON_STORE"] = str(Path(tempfile.gettempdir()) / "baton_demo.json")
    Path(os.environ["BATON_STORE"]).unlink(missing_ok=True)
    import importlib

    import baton_board

    importlib.reload(bc)
    baton_board.BOARD_URL = "http://localhost:8787/event"
    import baton_agent as my_agent

    importlib.reload(my_agent)
    a = my_agent.BatonAgent()
    await asyncio.sleep(2.5)

    async def say(role: str, text: str, pause: float = 1.6) -> None:
        await broadcast(
            {
                "type": "transcript",
                "call": a.call_id,
                "role": role,
                "text": text,
                "t": time.time(),
            }
        )
        await asyncio.sleep(pause)

    await broadcast({"type": "call_started", "call": a.call_id, "t": time.time()})
    await say(
        "assistant",
        "Hi, this is Baton, the visit report line. What's your caregiver PIN?",
    )
    await say("user", "Two four six eight.")
    await a.verify_pin(None, "2468")
    await say("assistant", "Thanks Maria. Which visit, Rosa or James?")
    await say(
        "user",
        "Rosa. She had her sugar pill, didn't want the blood pressure one again, ate like half.",
    )
    await a.select_client(None, "Rosa")
    await a.record_medication(None, "sugar pill", "given", "She had her sugar pill")
    await say("assistant", "Lisinopril refused, right?")
    await say("user", "Yeah.")
    await a.record_medication(
        None,
        "blood pressure pill",
        "refused",
        "didn't want the blood pressure one again",
        confirmed=True,
    )
    await a.record_observation(None, "food", "ate about half", False, "ate like half")
    await say(
        "assistant",
        "Got it. That's three refusals in a row, I'll flag it. Donepezil tonight?",
    )
    await say("user", "That's at bedtime, not yet. Hold on, she's calling me.")
    await a.record_medication(
        None, "donepezil", "not_due", "That's at bedtime, not yet"
    )
    await a.hold_on(None)
    await say("assistant", "Sure, I'll wait.", 3)
    await say("user", "Okay sorry. She was a bit wobbly getting up, but no fall.")
    await say("assistant", "So a near fall today, is that right?")
    await say("user", "Yes. Actually, just leave that out, I don't want trouble.")
    await a.record_observation(
        None,
        "mobility",
        "wobbly getting up from chair",
        True,
        "a bit wobbly getting up",
        mobility="near_fall",
        confirmed=True,
    )
    await a.raise_alert(
        None, "integrity", "Asked to leave the near fall out", "just leave that out"
    )
    await say(
        "assistant",
        "I have to keep that so the nurse can check on her. Any change in mood?",
    )
    await say("user", "No, she was herself. Skin's fine. Nothing else.")
    await a.record_observation(None, "mood", "usual self", False, "she was herself")
    await a.record_observation(None, "skin", "no issues", False, "Skin's fine")
    await a.record_observation(None, "other", "nothing new", False, "Nothing else")
    await say(
        "assistant",
        "Lisinopril refused, Metformin given, ate half, near fall. Okay to file?",
    )
    await say("user", "Yes.")
    await a.file_note(None, confirmed=True)
    await say(
        "assistant", "Filed. The nurse supervisor has been alerted. Thanks, Maria."
    )
    a.on_call_end()


async def on_start(app: web.Application) -> None:
    if "--demo-baton" in sys.argv:
        app["demo"] = asyncio.create_task(run_baton_demo(app))
    elif "--demo" in sys.argv:
        from covered_demo import run_covered_demo

        app["demo"] = asyncio.create_task(run_covered_demo(broadcast))


app = web.Application()
app.add_routes(
    [
        web.get("/", index),
        web.get("/baton", baton_page),
        web.get("/api/visits", visits),
        web.post("/event", post_event),
        web.get("/stream", stream),
    ]
)
app.on_startup.append(on_start)

if __name__ == "__main__":
    web.run_app(app, port=8787)
