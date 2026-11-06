"""The REALTIME switch picks the pipeline. No network: models only connect in a session."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from livekit.agents import llm

import models


@pytest.fixture(autouse=True)
def livekit_credentials(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.invalid")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
    monkeypatch.delenv("MODEL_PRESET", raising=False)


@pytest.mark.parametrize("value", ["true", "TRUE", " True ", "1", "yes"])
def test_realtime_true_uses_gemini_with_no_separate_stt_or_tts(monkeypatch, value):
    monkeypatch.setenv("REALTIME", value)
    chosen = models.load_models()
    assert chosen.realtime is True
    assert isinstance(chosen.llm, llm.RealtimeModel)
    assert chosen.llm.model == "google/gemini-3.8-live"
    assert chosen.stt is None and chosen.tts is None and chosen.expressive is False


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", ""])
def test_realtime_false_keeps_the_current_pipeline(monkeypatch, value):
    monkeypatch.setenv("REALTIME", value)
    chosen = models.load_models()
    assert chosen.realtime is False
    assert chosen.stt is not None and chosen.tts is not None
    assert not isinstance(chosen.llm, llm.RealtimeModel)


def test_realtime_defaults_to_false(monkeypatch):
    monkeypatch.delenv("REALTIME", raising=False)
    assert models.load_models().realtime is False


def test_a_mistyped_realtime_value_fails_loudly(monkeypatch):
    monkeypatch.setenv("REALTIME", "maybe")
    with pytest.raises(ValueError, match="REALTIME"):
        models.load_models()


def test_session_options_match_the_pipeline(monkeypatch):
    import agent

    monkeypatch.setenv("REALTIME", "true")
    realtime = agent.session_options(models.load_models())
    assert "stt" not in realtime and "tts" not in realtime
    assert "turn_handling" not in realtime  # Gemini detects turns itself
    assert realtime["user_away_timeout"] == agent.SILENCE_SECONDS

    monkeypatch.setenv("REALTIME", "false")
    pipeline = agent.session_options(models.load_models())
    assert pipeline["stt"] is not None and pipeline["tts"] is not None
    assert pipeline["turn_handling"]["preemptive_generation"]["preemptive_tts"] is True
