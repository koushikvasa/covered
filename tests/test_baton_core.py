import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import baton_core as bc


@pytest.fixture
def store(tmp_path):
    return bc.Store(tmp_path / "store.json")


def test_caller_id_and_pin():
    assert bc.identify_by_phone("+1 (201) 555-0199") == {
        "kind": "caregiver",
        "caregiver": "maria",
    }
    assert bc.identify_by_phone("2015550142") == {
        "kind": "client_home",
        "client": "rosa",
    }
    assert bc.identify_by_phone("+19998887777") == {"kind": "unknown"}
    assert (
        bc.caregiver_by_pin("two four six eight") is None
    )  # words are converted by the LLM
    assert bc.caregiver_by_pin("2 4 6 8") == "maria"
    assert bc.caregiver_by_pin("0000") is None


def test_only_assigned_clients():
    assert bc.resolve_client("maria", "Rosa") == "rosa"
    assert bc.resolve_client("devon", "Rosa") is None  # not Devon's client
    assert bc.resolve_client("devon", "James Okafor") == "james"


@pytest.mark.parametrize(
    ("spoken", "med", "conf"),
    [
        ("lisinopril", "lisinopril", "sure"),
        ("her blood pressure pill", "lisinopril", "sure"),
        ("listen april", "lisinopril", "ask"),
        ("metformin", "metformin", "sure"),
        ("the sugar pill", "metformin", "sure"),
        ("ibuprofen", None, "none"),
    ],
)
def test_closed_med_list(spoken, med, conf):
    m, c = bc.match_med("rosa", spoken)
    assert (m["id"] if m else None) == med
    assert c == conf


def test_emergency_and_negation():
    assert bc.detect_emergency("she fell and she can't get up") == "can't get up"
    assert bc.detect_emergency("Rosa is not breathing!") == "not breathing"
    assert bc.detect_emergency("she was breathing fine, no chest pain") is None
    assert bc.detect_safeguarding("I think her son hits her") is not None
    assert bc.detect_safeguarding("everything was normal") is None


def test_visit_flow_corrections_and_flags():
    v = bc.Visit("rosa", "maria", "pin")
    assert v.missing()[0] == "med:lisinopril"
    v.record_med("lisinopril", "given", "she took her bp pill")
    assert v.level() == "routine"
    corrected = v.record_med("lisinopril", "refused", "actually she refused it")
    assert corrected
    assert v.meds["lisinopril"]["history"][0]["value"] == "given"
    assert v.level() == "review"
    v.record_section(
        "mobility",
        "near_fall",
        "wobbly getting up",
        concern=True,
        detail="unsteady standing",
    )
    assert any(f["kind"] == "fall" for f in v.flags)
    # correcting the fall away clears the flag (code recomputes)
    v.record_section("mobility", "steady", "no she was fine", concern=False)
    assert not any(f["kind"] == "fall" for f in v.flags)
    v.record_section("mobility", "fall", "she fell in the bathroom", concern=True)
    assert v.level() == "urgent"


def test_cannot_finish_until_everything_covered():
    v = bc.Visit("james", "maria", "caller_id")
    v.record_med("warfarin", "given", "gave it")
    assert "med:furosemide" in v.missing()
    v.record_med("furosemide", "given", "water pill too")
    for s in ["food", "mobility", "mood", "skin", "other"]:
        assert v.missing()
        v.record_section(s, "ok", "fine", concern=False)
    assert v.missing() == []


def test_pattern_three_refusals(store):
    v = bc.Visit("rosa", "maria", "pin")
    v.record_med("lisinopril", "refused", "didn't want it again")
    texts = [p["text"] for p in bc.patterns(store, v)]
    assert any("Lisinopril" in t and "3 visits" in t for t in texts)


def test_pattern_falls(store):
    v = bc.Visit("james", "maria", "pin")
    v.record_section("mobility", "fall", "he fell", concern=True)
    assert any("falls" in p["text"] for p in bc.patterns(store, v))


def test_store_resume_dropped_call(store):
    v = bc.Visit("rosa", "maria", "pin")
    v.record_med("lisinopril", "given", "took it")
    store.upsert(v)
    again = bc.Store(store.path).open_visit("rosa", "maria")
    assert again and again["id"] == v.id
    os.remove(store.path)
