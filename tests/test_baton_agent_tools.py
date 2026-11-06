"""Drives the Baton agent's tools directly, the way the LLM would, with no LiveKit
Cloud. Proves the code-enforced rules: identity gate, confirmation, closed med
list, can't file early, integrity alerts, resume after a dropped call."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("BATON_STORE", str(tmp_path / "s.json"))
    import importlib

    import baton_core

    importlib.reload(baton_core)
    import baton_agent as my_agent

    importlib.reload(my_agent)
    return my_agent.BatonAgent()


async def test_full_call_with_edge_cases(agent):
    a, ctx = agent, None
    # no data before identity
    r = await a.select_client(ctx, "Rosa")
    assert "not verified" in r
    r = await a.record_medication(ctx, "lisinopril", "given", "gave it")
    assert "No report open" in r
    # wrong PIN, then right PIN
    assert "didn't match" in await a.verify_pin(ctx, "1111")
    assert "Verified Maria Santos" in await a.verify_pin(ctx, "2468")
    # not her client
    a2 = await a.select_client(ctx, "Walter")
    assert "Not one of their assigned clients" in a2
    r = await a.select_client(ctx, "Rosa")
    assert "Lisinopril" in r and "Next" in r

    # misheard med -> must ask
    r = await a.record_medication(ctx, "listen april", "given", "listen april given")
    assert "Did you mean Lisinopril" in r
    # refusal needs confirmation first
    r = await a.record_medication(
        ctx, "blood pressure pill", "refused", "she didn't want it again"
    )
    assert r.startswith("Confirm first")
    r = await a.record_medication(
        ctx,
        "blood pressure pill",
        "refused",
        "she didn't want it again",
        confirmed=True,
    )
    assert "Saved Lisinopril refused" in r
    assert "3 visits in a row" in r  # pattern across visits shows up live
    # unlisted med is never recorded as given
    r = await a.record_medication(ctx, "ibuprofen", "given", "gave her ibuprofen")
    assert "NOT record" in r
    assert "ibuprofen" not in a.visit.meds

    # can't file early
    r = await a.file_note(ctx, confirmed=True)
    assert r.startswith("Not done yet")

    await a.record_medication(ctx, "metformin", "given", "sugar pill yes")
    await a.record_medication(ctx, "donepezil", "not_due", "that's at night")
    await a.record_observation(
        ctx, "food", "ate half her lunch", False, "half her lunch"
    )
    # near fall needs confirmation
    r = await a.record_observation(
        ctx, "mobility", "wobbly standing up", True, "bit wobbly", mobility="near_fall"
    )
    assert r.startswith("Confirm first")
    await a.record_observation(
        ctx,
        "mobility",
        "wobbly standing up",
        True,
        "bit wobbly",
        mobility="near_fall",
        confirmed=True,
    )
    # integrity request
    r = await a.raise_alert(
        ctx, "integrity", "asked to remove the near fall", "just don't put that in"
    )
    assert "only record what actually happened" in r
    await a.record_observation(ctx, "mood", "usual self", False, "she was fine")
    await a.record_observation(ctx, "skin", "no issues", False, "skin fine")
    # hold mode
    await a.hold_on(ctx)
    assert a.on_hold()
    await a.record_observation(ctx, "other", "nothing new", False, "nothing else")

    r = await a.file_note(ctx)
    assert "Okay to file" in r and "Lisinopril refused" in r
    r = await a.file_note(ctx, confirmed=True)
    assert r.startswith("Filed")
    assert a.visit.status == "filed"
    assert a.visit.level() == "urgent"
    kinds = {f["kind"] for f in a.visit.flags}
    assert {"med", "fall", "integrity", "unlisted_med", "pattern"} <= kinds


async def test_dropped_call_resumes(agent, tmp_path):
    a, ctx = agent, None
    await a.verify_pin(ctx, "2468")
    await a.select_client(ctx, "James")
    await a.record_medication(ctx, "warfarin", "given", "blood thinner done")
    a.on_call_end()  # call drops
    import baton_agent as my_agent

    b = my_agent.BatonAgent()
    await b.verify_pin(ctx, "2468")
    r = await b.select_client(ctx, "James")
    assert "Resuming" in r
    assert "warfarin" in b.visit.meds
    assert "Furosemide" in r  # next missing item
    os.remove(b.store.path)


async def test_three_wrong_pins_ends(agent):
    for _ in range(2):
        await agent.verify_pin(None, "0000")
    r = await agent.verify_pin(None, "0000")
    assert "end the call" in r
