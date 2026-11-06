import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import covered_core as cc


@pytest.fixture
def store(tmp_path):
    return cc.Store(tmp_path / "c.json")


def test_identity_helpers():
    assert cc.member_by_phone("(201) 555-0111") == "alex"
    assert cc.member_by_phone("+19998887777") is None
    assert cc.rep_by_phone("+1 201 555 0122") == ("elena", "Maria Garcia")
    assert cc.member_by_name("Alex Johnson") == "alex"
    assert cc.normalize_dob("April 12, 1990") == "1990-04-12"
    assert cc.normalize_dob("04/12/1990") == "1990-04-12"
    assert cc.normalize_dob("sometime in spring") is None
    assert cc.is_authorized_rep("elena", "Maria Garcia", None)
    assert not cc.is_authorized_rep("elena", "Tom Garcia", "+19998887777")
    assert not cc.is_authorized_rep("alex", "Maria Garcia", None)


def test_plan_rules_and_costs():
    assert cc.needs_referral("alex", "orthopedics")
    assert not cc.needs_referral("alex", "mental_health")
    assert not cc.needs_referral("elena", "orthopedics")  # PPO
    assert cc.estimate_cost("alex", "dr-patel") == 30  # tier 1 copay
    # tier 2: $350 deductible left, allowed $220 -> all toward deductible
    assert cc.estimate_cost("alex", "dr-kim") == 220
    assert cc.estimate_cost("elena", "hudson-ortho") == 25  # deductible met, 10% of 250
    assert "prior authorization" in cc.coverage_answer("alex", "an MRI")
    assert cc.coverage_answer("alex", "acupuncture") is None  # not in data: no guessing


def test_search_skips_ghosts_and_prices(store):
    r = cc.search("alex", "mental_health", "07302", store.taken)
    names = [o["name"] for o in r["offers"]]
    ghost_names = {g["name"] for g in r["ghosts"]}
    assert ghost_names == {"Dr. Owen Hart", "Bayview Counseling"}
    assert "Dr. Owen Hart" not in names and len(r["offers"]) <= 2
    assert r["offers"][0]["cost"] <= r["offers"][-1]["cost"]  # cheapest first
    assert all(o["miles"] <= cc.MAX_MILES for o in r["offers"])


def test_search_respects_days_and_zip_override(store):
    r = cc.search(
        "alex",
        "mental_health",
        "07302",
        store.taken,
        exclude_days=["monday", "tuesday", "wednesday", "thursday", "friday"],
    )
    assert r["offers"] == []
    bk = cc.resolve_zip("I'm in Brooklyn this week", "07302")
    assert bk == "11201"
    r = cc.search("alex", "mental_health", bk, store.taken)
    assert any(o["name"] == "Dr. Sofia Reyes" for o in r["offers"])
    assert cc.resolve_zip("99999", "07302") is None


def test_booking_once_only(store):
    offer = cc.search("alex", "mental_health", "07302", store.taken)["offers"][0]
    b = store.book("alex", offer["id"], offer["slot"], None)
    assert b and b["confirmation"].startswith("CV-")
    assert (
        store.book("alex", offer["id"], offer["slot"], None) is None
    )  # double-book blocked
    again = cc.search("alex", "mental_health", "07302", store.taken)["offers"]
    assert all((o["id"], o["slot"]) != (offer["id"], offer["slot"]) for o in again)


def test_detectors():
    assert cc.detect_crisis("honestly I want to end my life")
    assert cc.detect_crisis("I'm stressed about work") is None
    assert cc.detect_integrity("just book it under my husband's insurance")
    assert cc.detect_integrity("say it's an emergency so it's covered")
    assert cc.detect_integrity("book me with Dr. Patel") is None
    assert cc.detect_emergency("I have chest pain right now")
    assert cc.slot_words("2026-11-10 10:00 AM") == "Tuesday November 10 at 10:00 AM"
