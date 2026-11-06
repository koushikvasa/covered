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
    assert cc.member_by_id("MBR-12345") == "alex"
    assert cc.member_by_id("mbr 12345") == "alex"
    assert cc.member_by_id("M B R 1 2 3 4 5") == "alex"
    assert cc.member_by_id("67890") == "elena"
    assert cc.member_by_id("MBR-99999") is None
    assert cc.member_by_id("") is None
    assert cc.name_matches("alex", "Alex Johnson")
    assert not cc.name_matches("alex", "Alex")  # the full name is half of the proof
    assert not cc.name_matches("alex", "Johnson")
    assert cc.name_matches("alex", "Alex Johnston")  # misheard by speech-to-text
    assert not cc.name_matches("alex", "Jordan Blake")
    assert not cc.name_matches("elena", "Maria Garcia")
    assert not cc.name_matches("elena", "Tom Garcia")
    assert not cc.name_matches("alex", "")
    assert cc.normalize_dob("April 12, 1990") == "1990-04-12"
    assert cc.normalize_dob("April 12th, 1990") == "1990-04-12"
    assert cc.normalize_dob("3rd September 1948") == "1948-09-03"
    assert cc.normalize_dob("04/12/1990") == "1990-04-12"
    assert cc.normalize_dob("sometime in spring") is None
    assert cc.is_authorized_rep("elena", "Maria Garcia", None)
    assert not cc.is_authorized_rep("elena", "Tom Garcia", "+19998887777")
    assert not cc.is_authorized_rep("alex", "Maria Garcia", None)
    assert not cc.is_authorized_rep("elena", "Garcia", None)


def test_plan_rules_and_costs():
    assert cc.needs_referral("alex", "orthopedics")
    assert not cc.needs_referral("alex", "mental_health")
    assert not cc.needs_referral("elena", "orthopedics")  # PPO
    assert cc.estimate_cost("alex", "dr-patel") == 30  # tier 1 copay
    # tier 2: $350 deductible left, allowed $220 -> all toward deductible
    assert cc.estimate_cost("alex", "dr-kim") == 220
    assert cc.estimate_cost("elena", "hudson-ortho") == 25  # deductible met, 10% of 250
    assert cc.copay_estimate("alex", "mental_health") == 30
    assert cc.copay_estimate("alex", "orthopedics") == 60  # specialist copay
    assert cc.copay_estimate("elena", "primary_care") == 20
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
    assert cc.resolve_zip("22030", "07302") == "22030"  # any real-looking ZIP is fine
    assert cc.resolve_zip("", "07302") == "07302"
    assert cc.resolve_zip("", None) is None
    assert cc.resolve_zip("Chicago", "07302") is None
    assert cc.resolve_zip("1234", None) is None


def test_booking_once_only(store):
    offer = cc.search("alex", "mental_health", "07302", store.taken)["offers"][0]
    b = store.book("alex", offer["id"], offer["slot"], None)
    assert b and b["confirmation"].startswith("CV-")
    assert (
        store.book("alex", offer["id"], offer["slot"], None) is None
    )  # double-book blocked
    again = cc.search("alex", "mental_health", "07302", store.taken)["offers"]
    assert all((o["id"], o["slot"]) != (offer["id"], offer["slot"]) for o in again)


def test_request_is_recorded_once_without_a_time_slot(store):
    first = store.add_request("alex", "npi-1", 30, None)
    assert first["confirmation"].startswith("RQ-") and first["slot"] is None
    assert store.add_request("alex", "npi-1", 30, None) == first
    assert store.add_request("alex", "npi-2", 30, None) != first
    assert len(store.data["requests"]) == 2
    assert store.taken == set()  # a request never holds a slot


def test_request_works_on_a_store_saved_before_requests_existed(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        '{"taken": [], "bookings": [], "referrals": [], "intakes": [], "ghost_reports": []}'
    )
    assert cc.Store(path).add_request("alex", "npi-1", 30, None)["slot"] is None


def test_new_specialties_follow_the_plan_rules():
    assert cc.needs_referral("alex", "neurology")
    assert cc.needs_referral("alex", "physical_therapy")
    assert not cc.needs_referral("alex", "pediatrics")
    assert not cc.needs_referral("elena", "neurology")
    assert cc.copay_estimate("alex", "physical_therapy") == 40
    assert cc.copay_estimate("elena", "physical_therapy") == 30
    assert cc.copay_estimate("alex", "neurology") == 60


def test_every_specialty_has_two_bookable_doctors_in_every_demo_zip():
    for specialty in cc.SPECIALTIES:
        for zip_code in cc.ZIPS:
            here = [
                p
                for p in cc.PROVIDERS
                if p["specialty"] == specialty
                and p["zip"] == zip_code
                and not p.get("ghost")
            ]
            assert len(here) == 2, (specialty, zip_code)


def test_changing_the_zip_changes_the_doctors(store):
    seen = {}
    for zip_code in cc.ZIPS:
        offers = cc.search("elena", "mental_health", zip_code, store.taken)["offers"]
        assert len(offers) == 2 and all(o["miles"] == 0 for o in offers)
        seen[zip_code] = {o["name"] for o in offers}
    assert len(set.union(*seen.values())) == 2 * len(cc.ZIPS)


def test_search_widens_only_when_the_zip_has_nothing_that_fits(store):
    local = cc.search("elena", "mental_health", "07030", store.taken)["offers"]
    taken = {f"{o['id']}|{s}" for o in local for s in cc.upcoming_slots(o["id"], set())}
    nearby = cc.search("elena", "mental_health", "07030", taken)["offers"]
    assert nearby and all(o["miles"] > 0 for o in nearby)


def test_costs_vary_between_doctors():
    costs = {cc.estimate_cost("alex", p["id"]) for p in cc.PROVIDERS}
    assert len(costs) >= 10
    assert cc.estimate_cost("alex", "dr-reyes") == 40  # $30 copay + $10 practice fee
    assert cc.estimate_cost("elena", "dr-reyes") == 30  # $20 copay + $10 practice fee


def test_every_bookable_doctor_has_a_background():
    for p in cc.PROVIDERS:
        if not p.get("ghost"):
            words = cc.background_words(p)
            assert "years in practice" in words or "group practice" in words, p["id"]
            assert "speaks" in words


def test_directory_is_the_same_on_every_run():
    import importlib

    before = [(p["id"], p["name"], p["tier"], p["allowed"]) for p in cc.PROVIDERS]
    importlib.reload(cc)
    assert before == [
        (p["id"], p["name"], p["tier"], p["allowed"]) for p in cc.PROVIDERS
    ]


def test_parse_time_understands_spoken_and_clock_forms():
    assert cc.parse_time("14:30") == 14 * 60 + 30
    assert cc.parse_time("2:30 PM") == 14 * 60 + 30
    assert cc.parse_time("2 pm") == 14 * 60
    assert cc.parse_time("9am") == 9 * 60
    assert cc.parse_time("sometime") is None
    assert cc.parse_time("") is None


def test_search_starts_from_the_preferred_date_and_nearest_time(store):
    from datetime import date, timedelta

    wanted = date.today() + timedelta(days=21)
    offers = cc.search(
        "alex",
        "mental_health",
        "07302",
        store.taken,
        prefer_provider="dr-patel",
        preferred_date=wanted,
        preferred_minutes=16 * 60,
    )["offers"]
    assert offers[0]["id"] == "dr-patel"
    assert offers[0]["slot"][:10] >= wanted.isoformat()
    assert offers[0]["slot"].endswith("4:00 PM")


def test_cost_reason_explains_how_the_cost_was_worked_out():
    assert "deductible" in cc.cost_reason("alex", "dr-kim")  # $350 left to pay
    assert "after their deductible" in cc.cost_reason("elena", "dr-kim")  # already met
    assert cc.cost_reason("alex", "dr-reyes") == "their plan copay plus a practice fee"
    assert cc.cost_reason("alex", "dr-patel") == "their plan copay"
    for p in cc.PROVIDERS:
        assert "network" not in cc.cost_reason("alex", p["id"])


def test_request_records_the_preferred_time(store):
    r = store.add_request("alex", "npi-1", 30, None, "Tuesday morning")
    assert r["preferred"] == "Tuesday morning"


def test_week_range_is_always_monday_to_friday():
    from datetime import date

    friday, wednesday = date(2026, 11, 6), date(2026, 11, 4)
    saturday = date(2026, 11, 7)
    assert cc.week_range("next_week", friday) == (date(2026, 11, 9), date(2026, 11, 13))
    assert cc.week_range("next_week", saturday) == (
        date(2026, 11, 9),
        date(2026, 11, 13),
    )
    assert cc.week_range("this_week", wednesday) == (wednesday, friday)


def test_week_window_keeps_slots_inside_it_and_spreads_them_across_days(store):
    first, last = cc.week_range("next_week")
    offers = cc.search(
        "elena",
        "mental_health",
        "07030",
        store.taken,
        preferred_date=first,
        window_end=last,
    )["offers"]
    for o in offers:
        assert first.isoformat() <= o["slot"][:10] <= last.isoformat()
        assert len(o["open_slots"]) <= cc.MAX_OPEN_SLOTS
        days = [s[:10] for s in o["open_slots"]]
        assert len(set(days)) == len(
            days
        )  # a choice of days comes before two on one day


def test_search_never_offers_a_slot_today_or_on_a_weekend(store):
    from datetime import date, datetime

    for p in cc.PROVIDERS[:25]:
        if p.get("ghost"):
            continue
        for s in cc.upcoming_slots(p["id"], store.taken):
            day = datetime.fromisoformat(s[:10]).date()
            assert day > date.today() and day.weekday() < 5


@pytest.mark.parametrize(
    "said",
    [
        "I have her insurance card right here",
        "It's his insurance, he's my son and I'm on the plan",
    ],
)
def test_mentioning_a_relatives_insurance_is_not_fraud(said):
    assert cc.detect_integrity(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "Book it under his insurance",
        "put me under her insurance",
        "my wife's insurance",
    ],
)
def test_using_someone_elses_insurance_is_still_flagged(said):
    assert cc.detect_integrity(said)


@pytest.mark.parametrize(
    "said",
    [
        "I had chest pain last year and want a heart checkup",
        "my dad was choking on his food last month",
        "I had a seizure two years ago, I need a neurologist",
    ],
)
def test_a_past_event_is_not_an_emergency(said):
    assert cc.detect_emergency(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "I have chest pain right now",
        "I had chest pain last week and it's back now",
        "I had chest pain last year and I still have it today",
        "she's choking",
        "I had chest pain",  # no sign it's over, so stay on the safe side
    ],
)
def test_a_current_or_unclear_event_is_still_an_emergency(said):
    assert cc.detect_emergency(said)


def test_detectors():
    assert cc.detect_crisis("honestly I want to end my life")
    assert cc.detect_crisis("I'm stressed about work") is None
    assert cc.detect_integrity("just book it under my husband's insurance")
    assert cc.detect_integrity("say it's an emergency so it's covered")
    assert cc.detect_integrity("book me with Dr. Patel") is None
    assert cc.detect_emergency("I have chest pain right now")
    assert cc.slot_words("2026-11-10 10:00 AM") == "Tuesday November 10 at 10:00 AM"
