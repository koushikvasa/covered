"""Drives Covered's tools the way the LLM would, with no LiveKit Cloud.
Proves the code-enforced rules hold no matter what the model tries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERED_STORE", str(tmp_path / "c.json"))
    import importlib

    import covered_core

    importlib.reload(covered_core)
    import my_agent

    importlib.reload(my_agent)
    return my_agent.MyAgent()


async def test_hmo_specialist_full_flow(agent):
    a, c = agent, None
    # nothing before identity
    assert "Not verified" in await a.find_care(c, "orthopedics", "knee pain")
    # wrong DOB never says which part was wrong
    r = await a.verify_identity(c, "Alex Johnson", "1990-04-13")
    assert "didn't match" in r and "Do not say which part" in r
    r = await a.verify_identity(c, "Alex Johnson", "1990-04-12")
    assert r.startswith("Verified Alex Johnson")

    # HMO: specialist needs a referral first; code refuses to skip it
    r = await a.find_care(c, "orthopedics", "knee pain")
    assert "referral" in r and "Dr. Lena Lee" in r
    r = await a.request_referral(c, "orthopedics", "knee pain")
    assert "Wait for a clear yes" in r
    r = await a.request_referral(c, "orthopedics", "knee pain", confirmed=True)
    assert "REF-" in r

    r = await a.find_care(c, "orthopedics", "knee pain")
    assert "Option A" in r and "Dr. Maya Stone" not in r  # ghost never offered
    assert "outdated listings were skipped" in r
    # booking needs a read-back first
    r = await a.book_appointment(c, "A")
    assert "Shall I book it" in r and "dollars" in r
    r = await a.book_appointment(c, "A", confirmed=True)
    assert r.startswith("Booked") and "confirmation" in r
    assert a.booking["referral"] == a.referral["id"]
    rec = a.store.data["intakes"][-1]
    assert (
        rec["status"] == "booked"
        and rec["caller"]["identity_verification_status"] == "verified"
    )


async def test_change_of_mind_and_place(agent):
    a, c = agent, None
    await a.verify_identity(c, "Alex Johnson", "April 12 1990")
    r = await a.find_care(c, "mental_health", "stress")
    assert "saw this provider before" in r  # remembers Dr. Patel
    first = a.offers["A"]["slot"]
    weekday = first.split(" ")[0]
    from datetime import date

    day = date.fromisoformat(weekday).strftime("%A").lower()
    r = await a.find_care(c, "mental_health", "stress", exclude_days=[day])
    assert all(day not in o["slot"] for o in a.offers.values())
    r = await a.find_care(
        c, "mental_health", "stress", location="I'm in Brooklyn this week"
    )
    assert "Sofia Reyes" in r
    r = await a.find_care(c, "mental_health", "stress", location="Chicago")
    assert "isn't in our search" in r


async def test_rep_rules_and_lockout(agent):
    a, c = agent, None
    r = await a.verify_identity(
        c, "Tom Garcia", "1948-09-03", "someone_else", "Elena Garcia"
    )
    assert "not an authorized representative" in r
    r = await a.verify_identity(
        c, "Maria Garcia", "1948-09-03", "someone_else", "Elena Garcia"
    )
    assert r.startswith("Verified Elena Garcia")
    # PPO: no referral needed for a specialist
    r = await a.find_care(c, "orthopedics", "hip pain")
    assert "Option A" in r


async def test_three_failures_lock(agent):
    for _ in range(3):
        r = await agent.verify_identity(None, "Alex Johnson", "2000-01-01")
    assert "Third failed attempt" in r
    r = await agent.verify_identity(None, "Alex Johnson", "1990-04-12")
    assert "locked" in r


async def test_coverage_only_from_data_and_handoff(agent):
    a, c = agent, None
    await a.verify_identity(c, "Alex Johnson", "1990-04-12")
    assert "prior authorization" in await a.check_coverage(c, "MRI")
    assert "Do not guess" in await a.check_coverage(c, "acupuncture")
    r = await a.transfer_to_human(c, "billing question")
    assert "won't have to repeat" in r
    assert a.store.data["intakes"][-1]["next_agent"] == "human_representative"


async def test_slot_taken_race(agent):
    a, c = agent, None
    await a.verify_identity(c, "Alex Johnson", "1990-04-12")
    await a.find_care(c, "mental_health", "stress")
    o = a.offers["A"]
    a.store.book("elena", o["id"], o["slot"], None)  # someone else books it first
    r = await a.book_appointment(c, "A", confirmed=True)
    assert "just taken" in r
