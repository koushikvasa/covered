"""Model presets for the voice pipeline.

Every model here runs through LiveKit Inference, so your LiveKit Cloud keys are
the only credentials you need. No Deepgram, Google or Cartesia accounts.

Pick a preset with MODEL_PRESET in .env.local, or edit one below.
Set REALTIME=true to replace STT, LLM and TTS with one Gemini Live model instead; it
costs the most, and MODEL_PRESET is then ignored.
Full model lists: https://docs.livekit.io/agents/models/
Prices:           https://livekit.com/pricing/inference

Rough cost per minute of conversation (our estimate, not LiveKit's):
  budget    ~ $0.015 / min   -> the free $2.50 credit lasts ~150 min
  quality   ~ $0.03  / min   -> the free $2.50 credit lasts ~80 min
Text-to-speech is usually the biggest line item.
"""

import os
from dataclasses import dataclass

from livekit.agents import inference

REALTIME_MODEL = "google/gemini-3.8-live"
REALTIME_VOICE = "Puck"
TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no", ""}


@dataclass
class Models:
    # In realtime mode one model hears and speaks, so stt and tts are None.
    stt: inference.STT | None
    llm: inference.LLM | inference.RealtimeModel
    tts: inference.TTS | None
    # Expressive mode lets the LLM add delivery tags (pacing, emotion) that the
    # TTS renders. Only turn it on for TTS models that support markup.
    expressive: bool
    realtime: bool = False


def budget() -> Models:
    """LiveKit's starter defaults. Cheapest, and good enough to win with."""
    return Models(
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        expressive=True,
    )


def quality() -> Models:
    """Deepgram + Gemini + Cartesia. About twice the cost of `budget`."""
    return Models(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="google/gemini-3.1-flash-lite"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        expressive=False,
    )


def realtime() -> Models:
    """Gemini Live through LiveKit Inference: no Google key or plugin needed."""
    return Models(
        stt=None,
        llm=inference.RealtimeModel(model=REALTIME_MODEL, voice=REALTIME_VOICE),
        tts=None,
        expressive=False,
        realtime=True,
    )


PRESETS = {"budget": budget, "quality": quality}


def realtime_enabled() -> bool:
    value = os.getenv("REALTIME", "false").strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"Unknown REALTIME '{value}'. Use true or false.")


def load_models() -> Models:
    if realtime_enabled():
        return realtime()
    name = os.getenv("MODEL_PRESET", "budget").strip().lower()
    if name not in PRESETS:
        raise ValueError(
            f"Unknown MODEL_PRESET '{name}'. Choose one of: {', '.join(PRESETS)}"
        )
    return PRESETS[name]()
