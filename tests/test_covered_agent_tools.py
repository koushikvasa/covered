"""Drives Covered's tools the way the LLM would, with no LiveKit Cloud.
Proves the code-enforced rules hold no matter what the model tries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ALEX = ("Alex Johnson", "MBR-12345")
ELENA_ID = "MBR-67890"
HOME_ZIP = "07302"
LIVE_ZIP = "22030"  # outside the demo network, so it goes to the live registry

# Synthetic stand-ins for what live_providers.search_nearby returns.
LIVE_RESULTS = [
    {
        "id": "npi-1000000001",
        "name": "Dr. Test Provider",
        "address": "1 Sample Street, Fairfax, VA",
        "phone": "555-0100",
        "specialty": "mental_health",
        "background": "Psychologist, Ph.D., in the public registry since 2012",
    },
    {
        "id": "npi-1000000002",
        "name": "Casey Example",
        "address": "2 Sample Street, Fairfax, VA",
        "phone": "555-0101",
        "specialty": "mental_health",
        "background": "Psychologist, Ph.D., in the public registry since 2012",
    },
]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERED_STORE", str(tmp_path / "c.json"))
    import importlib

    import covered_core

    importlib.reload(covered_core)
    import my_agent

    importlib.reload(my_agent)
    return my_agent.MyAgent()


@pytest.fixture
def live_results(monkeypatch):
    """Make the live registry answer without touching the network."""
    import my_agent

    async def fake_search(specialty, zip_code):
        return [{**p, "specialty": specialty} for p in LIVE_RESULTS]

    monkeypatch.setattr(my_agent.live, "search_nearby", fake_search)


async def test_hmo_specialist_full_flow(agent):
    a, c = agent, None
    # nothing before identity
    assert "Not verified" in await a.find_care(
        c, "orthopedics", "knee pain", HOME_ZIP, any_time=True
    )
    # a wrong ID never says which part was wrong
    r = await a.verify_identity(c, "Alex Johnson", "MBR-99999")
    assert "didn't match" in r and "Do not say which part" in r
    r = await a.verify_identity(c, *ALEX)
    assert r.startswith("Verified Alex Johnson")

    # HMO: specialist needs a referral first; code refuses to skip it
    r = await a.find_care(c, "orthopedics", "knee pain", HOME_ZIP, any_time=True)
    assert "referral" in r and "Dr. Lena Lee" in r
    r = await a.request_referral(c, "orthopedics", "knee pain")
    assert "Wait for a clear yes" in r
    r = await a.request_referral(c, "orthopedics", "knee pain", confirmed=True)
    assert "REF-" in r

    r = await a.find_care(c, "orthopedics", "knee pain")  # ZIP is remembered
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


async def test_identity_is_name_and_insurance_id(agent):
    a, c = agent, None
    # the right name is useless with someone else's or no ID
    assert "didn't match" in await a.verify_identity(c, "Alex Johnson", "MBR-99999")
    assert "didn't match" in await a.verify_identity(c, "Alex Johnson", "unknown")
    assert a.member is None
    # spoken forms of the ID still work, and a misheard name doesn't block a member
    r = await a.verify_identity(c, "Alex Johnston", "M B R 12345")
    assert r.startswith("Verified Alex Johnson") and a.relationship == "self"


async def test_a_first_name_alone_is_not_enough(agent):
    # With no date of birth, the full name is half of the proof. Asking again is
    # not a failed attempt: a real member may just have given their first name.
    r = await agent.verify_identity(None, "Alex", "MBR-12345")
    assert "first and last name" in r and agent.member is None and agent.fails == 0
    r = await agent.verify_identity(None, "Alex Johnson", "MBR-12345")
    assert r.startswith("Verified Alex Johnson")


async def test_someone_elses_id_is_refused(agent):
    r = await agent.verify_identity(None, "Jordan Blake", "MBR-12345")
    assert "not an authorized representative" in r
    assert agent.member is None
    assert "Not verified" in await agent.find_care(
        None, "dermatology", "rash", "07302", any_time=True
    )


async def test_verified_result_does_not_reveal_a_zip_to_assume(agent):
    r = await agent.verify_identity(None, *ALEX)
    assert HOME_ZIP not in r and "ZIP code" in r


async def test_find_care_asks_for_a_zip_before_searching(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    assert "ZIP code" in await a.find_care(c, "mental_health", "stress")
    assert a.offers == {}
    assert "ZIP code" in await a.find_care(c, "mental_health", "stress", "Chicago")


async def test_change_of_mind_and_place(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(c, "mental_health", "stress", HOME_ZIP, any_time=True)
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


async def test_rep_rules(agent):
    a, c = agent, None
    r = await a.verify_identity(c, "Tom Garcia", ELENA_ID)
    assert "not an authorized representative" in r
    r = await a.verify_identity(c, "Maria Garcia", ELENA_ID)
    assert r.startswith("Verified Elena Garcia")
    assert a.relationship == "authorized_representative"
    # PPO: no referral needed for a specialist
    r = await a.find_care(c, "orthopedics", "hip pain", "07306", any_time=True)
    assert "Option A" in r


async def test_three_failures_lock(agent):
    for _ in range(3):
        r = await agent.verify_identity(None, "Alex Johnson", "MBR-99999")
    assert "Third failed attempt" in r
    r = await agent.verify_identity(None, *ALEX)
    assert "locked" in r


async def test_live_zip_offers_real_providers_as_requests(agent, live_results):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(c, "mental_health", "stress", LIVE_ZIP, any_time=True)
    assert "Option A: Dr. Test Provider" in r and "1 Sample Street" in r
    assert "NOT known" in r and "if in-network" in r  # no claim we can't back
    assert all(o["live"] and "slot" not in o for o in a.offers.values())

    r = await a.book_appointment(c, "A")
    assert "Shall I send it" in r and "set the time" in r
    assert a.booking is None
    r = await a.book_appointment(c, "A", confirmed=True)
    assert r.startswith("Request sent") and "booked" not in r.lower()
    assert a.booking["confirmation"].startswith("RQ-") and a.booking["slot"] is None
    assert a.store.taken == set()
    rec = a.store.data["intakes"][-1]
    assert rec["status"] == "requested" and rec["next_agent"] == "provider_office"


async def test_live_request_is_only_sent_once(agent, live_results):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    await a.find_care(c, "mental_health", "stress", LIVE_ZIP, any_time=True)
    await a.book_appointment(c, "A", confirmed=True)
    first = a.booking["confirmation"]
    await a.book_appointment(c, "A", confirmed=True)
    assert a.booking["confirmation"] == first
    assert len(a.store.data["requests"]) == 1


async def test_referral_rule_still_applies_to_live_providers(agent, live_results):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(c, "orthopedics", "knee pain", LIVE_ZIP, any_time=True)
    assert "referral" in r and a.offers == {}
    await a.request_referral(c, "orthopedics", "knee pain", confirmed=True)
    r = await a.find_care(c, "orthopedics", "knee pain")
    assert "about 60 dollars if in-network" in r  # Alex's specialist copay


async def test_live_registry_outage_never_breaks_the_call(agent, monkeypatch):
    import my_agent

    async def broken_search(specialty, zip_code):
        raise my_agent.live.ProviderLookupError("registry timed out")

    monkeypatch.setattr(my_agent.live, "search_nearby", broken_search)
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(c, "mental_health", "stress", LIVE_ZIP, any_time=True)
    assert "isn't answering" in r and a.offers == {}


async def test_live_zip_with_no_providers(agent, monkeypatch):
    import my_agent

    async def no_results(specialty, zip_code):
        return []

    monkeypatch.setattr(my_agent.live, "search_nearby", no_results)
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(c, "mental_health", "stress", "59001", any_time=True)
    assert "No providers" in r and a.offers == {}


async def test_coverage_only_from_data_and_handoff(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    assert "prior authorization" in await a.check_coverage(c, "MRI")
    assert "Do not guess" in await a.check_coverage(c, "acupuncture")
    r = await a.transfer_to_human(c, "billing question")
    assert "won't have to repeat" in r
    assert a.store.data["intakes"][-1]["next_agent"] == "human_representative"


async def test_slow_down_lasts_for_the_rest_of_the_call(agent):
    from livekit.agents import llm

    def caller_says(text):
        return llm.ChatMessage(role="user", content=[text])

    await agent.on_user_turn_completed(
        llm.ChatContext.empty(), caller_says("can you slow down please")
    )
    # The SDK gives each turn a fresh context, so an edit to one turn's is lost.
    later_turn = llm.ChatContext.empty()
    await agent.on_user_turn_completed(later_turn, caller_says("the first one"))

    assert any("slower pace" in (i.text_content or "") for i in later_turn.items)


def test_caller_phone_ignores_console_mock_participants(agent, monkeypatch):
    from unittest.mock import MagicMock

    import my_agent

    job = MagicMock()
    job.room.remote_participants = {"console": MagicMock()}
    monkeypatch.setattr(my_agent, "get_job_context", lambda: job)
    assert my_agent._caller_phone() is None

    job.room.remote_participants = {
        "sip": MagicMock(attributes={"sip.phoneNumber": "+12015550111"})
    }
    assert my_agent._caller_phone() == "+12015550111"


def test_board_transcript_hides_delivery_tags(agent, monkeypatch):
    from types import SimpleNamespace

    import my_agent

    sent = []
    monkeypatch.setattr(
        my_agent.board, "emit", lambda *_, **fields: sent.append(fields)
    )
    spoken = (
        '<expr type="expression" label="calm"/>Please '
        '<expr type="prosody" label="emphasis">hang up</expr> and call. '
        '<expr type="break" label="500ms"/> Okay.'
    )

    agent._on_item(
        SimpleNamespace(item=SimpleNamespace(role="assistant", text_content=spoken))
    )

    assert sent[0]["text"] == "Please hang up and call. Okay."


async def test_off_topic_steers_back_to_the_current_step(agent):
    a, c = agent, None
    r = await a.handle_off_topic(c, "the weather")
    assert "only help with booking care" in r and "insurance ID" in r

    await a.verify_identity(c, *ALEX)
    assert "what kind of care" in await a.handle_off_topic(c, "sports")

    await a.find_care(c, "orthopedics", "knee pain", HOME_ZIP, any_time=True)
    assert "referral" in await a.handle_off_topic(c, "a joke")

    await a.request_referral(c, "orthopedics", "knee pain", confirmed=True)
    await a.find_care(c, "orthopedics", "knee pain")
    assert "which option" in await a.handle_off_topic(c, "trivia")

    await a.book_appointment(c, "A", confirmed=True)
    assert "anything else" in await a.handle_off_topic(c, "the news")


async def test_repeated_off_topic_offers_a_person(agent):
    first = await agent.handle_off_topic(None, "the weather")
    second = await agent.handle_off_topic(None, "sports")
    third = await agent.handle_off_topic(None, "a joke")

    assert "a person" not in first and "a person" not in second
    assert "a person" in third


async def test_compare_gives_backgrounds_without_ranking(agent):
    a, c = agent, None
    assert "find_care" in await a.compare_doctors(c)  # nothing offered yet
    await a.verify_identity(c, *ALEX)
    await a.find_care(c, "mental_health", "stress", HOME_ZIP, any_time=True)

    r = await a.compare_doctors(c)

    assert all(o["name"] in r for o in a.offers.values())
    assert "years in practice" in r and "trained at" in r
    assert "Never say who is better" in r


async def test_compare_live_providers_only_uses_registry_facts(agent, live_results):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    await a.find_care(c, "mental_health", "stress", LIVE_ZIP, any_time=True)

    r = await a.compare_doctors(c)

    assert "in the public registry since 2012" in r
    assert "years in practice" not in r
    assert "Never say who is better" in r


def _weekday_after(days_ahead: int) -> str:
    from datetime import date, timedelta

    day = date.today() + timedelta(days=days_ahead)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


async def test_asks_for_a_date_and_time_before_suggesting_doctors(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)

    r = await a.find_care(c, "mental_health", "stress", HOME_ZIP)

    assert "date and time" in r and "Option A" not in r and a.offers == {}
    assert "date and time" in await a.handle_off_topic(c, "the weather")


async def test_search_uses_the_date_and_time_the_caller_asked_for(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    wanted = _weekday_after(14)

    r = await a.find_care(
        c,
        "mental_health",
        "stress",
        HOME_ZIP,
        preferred_date=wanted,
        preferred_time="15:30",
    )

    assert "Option A" in r
    patel = a.offers["A"]
    assert patel["name"] == "Dr. Priya Patel"
    assert patel["slot"][:10] >= wanted
    assert patel["slot"].endswith("4:00 PM")  # closer to 3:30 PM than her 10:00 AM


async def test_a_past_date_is_questioned_not_searched(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(
        c, "mental_health", "stress", HOME_ZIP, preferred_date="2020-01-06"
    )
    assert "already passed" in r and a.offers == {}


async def test_an_unreadable_date_is_asked_again(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    r = await a.find_care(
        c, "mental_health", "stress", HOME_ZIP, preferred_date="the fifth"
    )
    assert "date and time" in r and a.offers == {}


async def test_date_and_time_are_remembered_when_the_zip_changes(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    wanted = _weekday_after(14)
    await a.find_care(c, "mental_health", "stress", HOME_ZIP, preferred_date=wanted)

    r = await a.find_care(c, "mental_health", "stress", "11201")

    assert "Sofia Reyes" in r
    assert all(o["slot"][:10] >= wanted for o in a.offers.values())


async def test_every_cost_comes_with_its_true_reason(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)

    r = await a.find_care(c, "mental_health", "stress", HOME_ZIP, any_time=True)

    assert "Dr. Hana Vasquez" in r and "toward their remaining deductible" in r
    assert "All of these are in-network" in r


async def test_live_request_passes_along_the_preferred_time(agent, live_results):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    wanted = _weekday_after(14)
    await a.find_care(
        c,
        "mental_health",
        "stress",
        LIVE_ZIP,
        preferred_date=wanted,
        time_of_day="morning",
    )

    r = await a.book_appointment(c, "A")
    assert "pass along" in r and "morning" in r
    await a.book_appointment(c, "A", confirmed=True)
    assert "morning" in a.store.data["requests"][-1]["preferred"]


class FakeSession:
    def __init__(self):
        self.replies = []
        self.interrupted = 0

    def generate_reply(self, instructions=None):
        self.replies.append(instructions)

    def interrupt(self):
        self.interrupted += 1


def _realtime_agent(agent, monkeypatch):
    import my_agent

    session = FakeSession()
    monkeypatch.setattr(my_agent.MyAgent, "session", property(lambda self: session))
    return session


def _transcript(text, is_final=True):
    from types import SimpleNamespace

    return SimpleNamespace(transcript=text, is_final=is_final)


@pytest.mark.parametrize(
    ("said", "expected", "kind"),
    [
        ("I have chest pain right now", "nine one one", "emergency"),
        ("I want to end my life", "nine eight eight", "crisis"),
        ("book it under my husband's insurance", "improper request", "integrity"),
    ],
)
async def test_realtime_transcripts_get_the_same_safety_checks(
    agent, monkeypatch, said, expected, kind
):
    session = _realtime_agent(agent, monkeypatch)

    agent._on_realtime_transcript(_transcript(said))

    assert session.interrupted == 1
    assert expected in session.replies[0]
    assert agent.flags[-1]["kind"] == kind


async def test_realtime_ignores_partial_and_harmless_transcripts(agent, monkeypatch):
    session = _realtime_agent(agent, monkeypatch)

    agent._on_realtime_transcript(_transcript("chest pain", is_final=False))
    agent._on_realtime_transcript(_transcript("my knee hurts"))

    assert session.replies == [] and session.interrupted == 0 and agent.flags == []


async def test_realtime_slow_down_updates_the_instructions_once(agent, monkeypatch):
    import asyncio

    _realtime_agent(agent, monkeypatch)
    updates = []

    async def fake_update(text):
        updates.append(text)

    monkeypatch.setattr(agent, "update_instructions", fake_update)

    agent._on_realtime_transcript(_transcript("can you slow down"))
    agent._on_realtime_transcript(_transcript("please slow down again"))
    await asyncio.sleep(0)

    assert len(updates) == 1 and "slower pace" in updates[0]


async def test_slot_taken_race(agent):
    a, c = agent, None
    await a.verify_identity(c, *ALEX)
    await a.find_care(c, "mental_health", "stress", HOME_ZIP, any_time=True)
    o = a.offers["A"]
    a.store.book("elena", o["id"], o["slot"], None)  # someone else books it first
    r = await a.book_appointment(c, "A", confirmed=True)
    assert "just taken" in r
