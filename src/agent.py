"""Entry point: wires the voice pipeline and picks which agent answers.

You mostly edit my_agent.py. Come here to change models, silence handling or
turn-taking.

  .env.local
    AGENT_PROFILE = booking | screener | mine     (which agent answers)
    MODEL_PRESET  = budget  | quality             (see models.py)
    REALTIME      = false   | true                (true: Gemini Live replaces STT, LLM, TTS)
    AGENT_NAME    = take2-agent                   (must match your dispatch rule)
"""

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from baton_agent import BatonAgent
from baton_core import CAREGIVERS, CLIENTS
from booking_agent import BookingAgent
from covered_core import HAND_PROVIDERS, MEMBERS
from models import Models, load_models
from my_agent import MyAgent
from screener_agent import ScreenerAgent

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENTS: dict[str, type[Agent]] = {
    "booking": BookingAgent,
    "screener": ScreenerAgent,
    "mine": MyAgent,  # Covered
    "baton": BatonAgent,
}

AGENT_NAME = os.getenv("AGENT_NAME", "take2-agent")

# Silence: after this many quiet seconds the caller counts as "away".
SILENCE_SECONDS = 8.0
# Check in this many times, then say goodbye and hang up.
MAX_CHECK_INS = 2


def pick_agent() -> Agent:
    profile = os.getenv("AGENT_PROFILE", "booking").strip().lower()
    if profile not in AGENTS:
        raise ValueError(
            f"Unknown AGENT_PROFILE '{profile}'. Choose one of: {', '.join(AGENTS)}"
        )
    return AGENTS[profile]()


def handle_silence(session: AgentSession) -> None:
    """Ask "are you still there?" when the caller goes quiet, then hang up."""
    check_ins = 0
    tasks: set[asyncio.Task[None]] = set()

    async def check_in() -> None:
        nonlocal check_ins
        # Baton: the caller said "hold on". Stay silent until the hold expires.
        agent = session.current_agent
        if getattr(agent, "on_hold", lambda: False)():
            await asyncio.sleep(agent.hold_remaining())
            if session.user_state != "away" or agent.on_hold():
                return
            agent.hold_until = 0.0
            await session.generate_reply(
                instructions="The caller has been on hold a while. In a few words, ask if they're back."
            )
            return
        check_ins += 1
        if check_ins <= MAX_CHECK_INS:
            await session.generate_reply(
                instructions="The caller has gone quiet. In a few words, ask if they are still there."
            )
        else:
            await session.generate_reply(
                instructions="The caller is not responding. Say a short, polite goodbye."
            )
            on_end = getattr(session.current_agent, "on_call_end", None)
            if on_end:
                on_end()
            session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal check_ins
        if ev.new_state == "speaking":
            check_ins = 0
        elif ev.new_state == "away":
            task = asyncio.create_task(check_in())
            tasks.add(task)
            task.add_done_callback(tasks.discard)


def session_options(models: Models) -> dict[str, Any]:
    """AgentSession settings for the pipeline chosen by REALTIME."""
    if models.realtime:
        # One model hears, thinks and speaks, and detects turns itself.
        return {"llm": models.llm, "user_away_timeout": SILENCE_SECONDS}
    return {
        # Ears: speech-to-text
        "stt": models.stt,
        # Bias transcription toward words it would otherwise misspell.
        "stt_context_options": STTContextOptions(
            keyterms=[
                "Take2",
                "LiveKit",
                "Baton",
                # Baton: client and medication names, so STT doesn't mangle them.
                *(c["name"] for c in CLIENTS.values()),
                *(m["name"] for c in CLIENTS.values() for m in c["meds"]),
                *(g["name"] for g in CAREGIVERS.values()),
                # Covered: provider and member names, plan words.
                "Covered",
                "insurance ID",
                *(m["member_id"] for m in MEMBERS.values()),
                "in-network",
                "copay",
                "deductible",
                "referral",
                # Hand-written only: the full directory is too long a list for STT.
                *(p["name"] for p in HAND_PROVIDERS),
                *(m["full_name"] for m in MEMBERS.values()),
            ],
            keyterm_detection={"enabled": True},
        ),
        # Brain: the LLM
        "llm": models.llm,
        # Voice: text-to-speech
        "tts": models.tts,
        "expressive": models.expressive,
        "turn_handling": TurnHandlingOptions(
            # Decides when the caller has finished speaking, using both the words
            # and the audio (pitch, pauses). Much better than a fixed silence timer.
            turn_detection=inference.TurnDetector(),
            # Start drafting the reply and its audio before the turn is confirmed.
            # Cuts latency, but the voice costs more when a draft gets thrown away.
            # Tools still wait for the confirmed turn, so a draft can't book anything.
            preemptive_generation={"enabled": True, "preemptive_tts": True},
        ),
        "user_away_timeout": SILENCE_SECONDS,
    }


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(**session_options(load_models()))

    handle_silence(session)

    agent = pick_agent()

    async def log_usage() -> None:
        # Shows model usage per call so you can see what you're spending.
        logger.info("session usage: %s", session.usage)
        # Baton: a dropped call keeps its partial note, marked incomplete.
        on_end = getattr(agent, "on_call_end", None)
        if on_end:
            on_end()

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Removes background noise and other voices before STT hears it.
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
            # Hang up the phone line when the session ends.
            delete_room_on_close=True,
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
