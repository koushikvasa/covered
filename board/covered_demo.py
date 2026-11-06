"""Scripted Covered call for the board, driven through the REAL agent tools."""

import asyncio
import importlib
import os
import tempfile
import time
from pathlib import Path


async def run_covered_demo(broadcast) -> None:
    os.environ["COVERED_STORE"] = str(Path(tempfile.gettempdir()) / "covered_demo.json")
    Path(os.environ["COVERED_STORE"]).unlink(missing_ok=True)
    import baton_board
    import covered_core

    importlib.reload(covered_core)
    baton_board.BOARD_URL = "http://localhost:8787/event"
    import my_agent

    importlib.reload(my_agent)
    a = my_agent.MyAgent()
    await asyncio.sleep(2.5)

    async def say(role: str, text: str, pause: float = 1.7) -> None:
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
        "Hi, this is Covered, your health plan's care line. "
        "Can I get your name and what you're calling about today?",
    )
    await say("user", "It's Alex Johnson. My knee's been killing me for weeks.")
    await say(
        "assistant",
        "Sorry to hear that, Alex. To protect your account, what's your insurance ID?",
    )
    await say("user", "M B R one two three four five.")
    await a.verify_identity(None, "Alex Johnson", "MBR-12345")
    await say("assistant", "You're verified. What ZIP code do you want care in?")
    await say("user", "Oh seven three oh two.")
    await a.find_care(None, "orthopedics", "knee pain", "07302", any_time=True)
    await say(
        "assistant",
        "Your plan needs a referral from Dr. Lee first. Want me to request it?",
    )
    await say("user", "Can't you just skip that?")
    await say(
        "assistant", "Without it, your plan won't cover the visit. I can send it now."
    )
    await say("user", "Fine, yes.")
    await a.request_referral(None, "orthopedics", "knee pain", confirmed=True)
    await a.find_care(None, "orthopedics", "knee pain")
    first = a.offers["A"]
    await say(
        "assistant",
        f"{first['name']} is close by, about {first['cost']} dollars. Or another option. Which one?",
    )
    await say("user", "Hold on one sec.")
    await a.hold_on(None)
    await say("assistant", "Sure, take your time.", 3)
    await say(
        "user", "Sorry. Actually, can you book it under my husband's insurance instead?"
    )
    a._flag(
        "integrity",
        "review",
        "Asked to use someone else's insurance",
        "book it under my husband's insurance",
    )
    await say(
        "assistant",
        "I can only book with your own plan. The first option is still open.",
    )
    await say("user", "Okay, the first one then.")
    from covered_core import slot_words

    await say(
        "assistant",
        f"{first['name']}, {slot_words(first['slot'])}, about {first['cost']} dollars. Shall I book it?",
    )
    await say("user", "Yes.")
    await a.book_appointment(None, "A", confirmed=True)
    conf = a.booking["confirmation"]
    await say(
        "assistant",
        f"You're booked. Your confirmation is {' '.join(conf.replace('-', ''))}. Anything else?",
    )
    await say("user", "No, that's it. Thanks.")
    a.on_call_end()
