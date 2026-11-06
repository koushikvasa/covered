"""Covered: a health plan's care line. Members call, and leave with a booked,
in-network appointment that is really available and priced before they say yes.

Design: the LLM listens and talks; covered_core.py decides.
  - Identity: full name and insurance ID are proof; reps must be on file.
  - Ghost-proof search: only providers with a real open slot are ever offered.
  - Outside the demo network, real providers come from the live NPI Registry as requests.
  - Cost: the member's own estimate (deductible, copay, coinsurance), from code.
  - Plan rules: HMO specialists need a referral; coverage answers come from data only.
  - Code-enforced confirmation: referral and booking need confirmed=true after a read-back.
  - Every caller turn is checked for emergency, crisis and fraud requests.
"""

import asyncio
import logging
import re
import time
from datetime import date
from typing import Literal

from livekit.agents import Agent, RunContext, function_tool, get_job_context, llm
from livekit.agents.beta.tools import EndCallTool

import baton_board as board
import covered_core as cc
import live_providers as live
from prompts import build_instructions

logger = logging.getLogger("covered")

HOLD_SECONDS = 180
# From this many off-topic questions on, also offer the caller a person.
OFF_TOPIC_LIMIT = 3
MAX_LIVE_OFFERS = 2
SLOWER_PACE_NOTE = (
    "Caller asked for a slower pace: give one detail per sentence, very short."
)
# A first miss may just be misheard, so the caller gets one retry before this.
NOT_ON_FILE_AFTER = 2
NOT_ON_FILE = (
    "No plan was found under those details; do not say which part. Say so kindly, and ask them to "
    "buy a Covered plan and call back once they have one. Do not say where to buy one. If they're sure they already have a plan, "
    "offer a person instead. Otherwise say a warm goodbye and end the call."
)
WEEK_WORDS = {"this_week": "this week", "next_week": "next week"}
ASK_FOR_TIMING = (
    "Kindly ask what date and time would suit them, or whether any time works, then call "
    "find_care again with their answer. Do not suggest doctors yet."
)
# Delivery tags the LLM adds for expressive TTS; never show them on the board.
DELIVERY_TAG = re.compile(r"</?expr\b[^>]*>")
Specialty = Literal[
    "primary_care",
    "mental_health",
    "orthopedics",
    "dermatology",
    "cardiology",
    "ob_gyn",
    "neurology",
    "ent",
    "ophthalmology",
    "gastroenterology",
    "endocrinology",
    "allergy",
    "pediatrics",
    "physical_therapy",
    "urology",
    "podiatry",
]

JOB_PROMPT = """
You are Covered, the care line for Covered Health, a health insurance plan.
Members call to get care. Your one job: book them an in-network appointment
they can actually get, and tell them what it will cost before they agree.

Callers may be in pain, worried, or worn out by insurance, so be warm and unhurried.
When they mention how they feel or what's wrong, first acknowledge it in a few kind words
that fit what they said, and don't reuse the same phrase in a call. Stay sincere, not gushing.
After that, speak in one short sentence, about fifteen words, with one question per turn.
Use plain words. Explain a term only if it matters ("in-network means your plan covers it").

1. Start with the caller's full name and why they're calling, in their own words.
2. Verify identity before any account detail. Ask for the insurance ID on their card,
   saying it's to protect the account. If they call for someone else, ask for that
   person's insurance ID. Then call verify_identity. If they don't have the ID with them
   or say they have no plan with us, call caller_without_id.
3. Map their reason to a type of provider; never diagnose. If unsure, choose primary_care.
   Ask for the ZIP code where they want care; never assume one. Then ask what date and
   time suit them. As soon as they answer, even vaguely ("next week", "any weekday"), call
   find_care without asking for more. Pass what they said even if it's a past date or a
   weekend: the tool checks it.
4. Do exactly what each tool result tells you. Offer at most two options, one sentence each.
5. Before requesting a referral or booking, read the details back and ask yes or no.
   Only after a clear yes, call the tool again with confirmed true. Silence is not a yes.
6. After booking, give the confirmation number slowly, ask if there's anything else,
   then say goodbye and end the call.

Never invent coverage, costs, providers or times: use only what tools return.
Never work out a weekday yourself: use the list of bookable days below.
Refer to doctors by name, never "he" or "she": you don't know their gender.
Never book for anyone you have not verified, and never use someone else's insurance.
If they want a person, call transfer_to_human. If they say "hold on", call hold_on.
If they ask about the doctors' backgrounds, or which one is better, call compare_doctors.
Callers can ask you to slow down or repeat at any time; do it without fuss.
"""


def _caller_phone() -> str | None:
    try:
        room = get_job_context().room
    except Exception:
        return None
    for p in room.remote_participants.values():
        phone = p.attributes.get("sip.phoneNumber")
        # Console mode uses a mock room, so only trust a real string.
        if phone and isinstance(phone, str):
            return phone
    return None


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(
                f"{JOB_PROMPT}\nBookable days: {', '.join(cc.bookable_days())}."
            ),
            tools=[
                EndCallTool(
                    ignore_on_enter=True,
                    # We say goodbye in the reply that calls this tool, so don't
                    # ask the LLM to generate a second one afterwards.
                    end_instructions=None,
                    extra_description=(
                        "Only after a booking or handoff is done, after an emergency, "
                        "after asking them to call back, or when the caller asks to stop. "
                        "Say your short goodbye in the same reply that calls this tool."
                    ),
                )
            ],
        )
        self.store = cc.Store()
        self.call_id = f"CALL-{int(time.time()) % 100000}"
        self.phone: str | None = None
        self.caller_name = ""
        self.member: str | None = None  # verified member id
        self.relationship = "self"
        self.fails = 0
        self.specialty: str | None = None
        self.need = ""
        self.zip: str | None = None
        self.offers: dict[str, dict] = {}
        self.booking: dict | None = None
        self.referral: dict | None = None
        self.flags: list[dict] = []
        self.prefs = {"slower_pace": False, "repeat_requested": False}
        self.hold_until = 0.0
        self.off_topic_count = 0
        self.timing: dict | None = None  # the date and time the caller wants
        self._realtime_tasks: set[asyncio.Task] = set()
        self._ended = False

    # ------------------------------------------------------------- helpers
    def _next_step(self) -> str:
        """Where the call left off, so an off-topic reply can return to it."""
        if self.booking:
            return "Then ask if there's anything else."
        if not self.member:
            return (
                "Then ask for whichever you still need: their full name, reason for "
                "calling, or insurance ID."
            )
        if (
            self.specialty
            and cc.needs_referral(self.member, self.specialty)
            and not self.store.has_referral(self.member, self.specialty)
        ):
            return "Then ask if they want you to request the referral."
        if self.zip and self.specialty and not self.timing:
            return "Then ask what date and time would suit them."
        if self.offers:
            return "Then ask which option they'd like."
        return "Then ask what kind of care they need and their ZIP code, whichever you're missing."

    def on_hold(self) -> bool:
        return time.time() < self.hold_until

    def hold_remaining(self) -> float:
        return max(0.0, self.hold_until - time.time())

    def _first(self) -> str:
        return cc.MEMBERS[self.member]["preferred_name"] if self.member else ""

    def intake(self, status: str, next_agent: str) -> dict:
        m = cc.MEMBERS.get(self.member or "", {})
        rec = {
            "call_id": self.call_id,
            "caller": {
                "full_name": self.caller_name or m.get("full_name"),
                "preferred_name": m.get("preferred_name")
                if self.relationship == "self"
                else None,
                "relationship_to_member": self.relationship,
                "member_id": m.get("member_id"),
                "identity_verification_status": "verified"
                if self.member
                else ("failed" if self.fails else "not_started"),
                "authorized_representative_status": "verified"
                if self.relationship != "self" and self.member
                else "not_applicable",
            },
            "call_details": {
                "intent": "provider_search_and_booking" if self.specialty else "other",
                "subcategory": self.specialty,
                "description": self.need,
                "urgency": "emergency"
                if any(f["kind"] == "emergency" for f in self.flags)
                else "normal",
                "desired_outcome": "booked_appointment",
            },
            "accessibility_preferences": {**self.prefs, "one_question_at_a_time": True},
            "references": {
                "confirmation": self.booking and self.booking["confirmation"],
                "provider_id": self.booking and self.booking["provider"],
                "referral_id": self.referral and self.referral["id"],
            },
            "flags": self.flags,
            "status": status,
            "next_agent": next_agent,
        }
        self.store.add_intake(rec)
        board.emit("intake", self.call_id, record=rec)
        return rec

    def _flag(self, kind: str, level: str, text: str, quote: str = "") -> None:
        self.flags.append({"kind": kind, "level": level, "text": text, "quote": quote})
        board.emit(
            "alert", self.call_id, level=level, kind=kind, text=text, quote=quote
        )

    def on_call_end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if not self.booking:
            self.intake("incomplete", "callback_queue")
        board.emit("call_ended", self.call_id)

    def _member_card(self) -> dict:
        m = cc.MEMBERS[self.member]
        p = cc.plan(self.member)
        return {
            "name": m["full_name"],
            "member_id": m["member_id"],
            "plan": p["name"],
            "zip": m["zip"],
            "area": cc.ZIPS[m["zip"]][2],
            "pcp": cc.PROVIDER[m["pcp"]]["name"],
            "deductible": m["deductible"],
            "deductible_met": m["deductible_met"],
            "relationship": self.relationship,
            "caller": self.caller_name,
            "how": "full name + insurance ID",
        }

    # ----------------------------------------------------------- lifecycle
    async def on_enter(self) -> None:
        self.session.on("conversation_item_added", self._on_item)
        if isinstance(self.session.llm, llm.RealtimeModel):
            self.session.on("user_input_transcribed", self._on_realtime_transcript)
        board.emit("call_started", self.call_id)
        self.phone = _caller_phone()
        await self.session.generate_reply(
            instructions=(
                "Say warmly: 'Hi, thanks for calling Covered, your health plan's care line. "
                "I'm your virtual assistant, and I can find you a doctor, book your visit, "
                "and tell you the cost up front. May I have your full name, and what can I "
                "help you with today?'"
            )
        )

    def _on_item(self, ev) -> None:
        item = getattr(ev, "item", None)
        text = getattr(item, "text_content", None)
        if text:
            board.emit(
                "transcript",
                self.call_id,
                role=getattr(item, "role", ""),
                text=" ".join(DELIVERY_TAG.sub("", text).split()),
            )

    def _note_requests(self, text: str) -> None:
        """Record what the caller asked of the conversation itself."""
        self.hold_until = 0.0
        low = text.lower()
        if any(p in low for p in ("slow down", "too fast", "slower")):
            self.prefs["slower_pace"] = True
        if any(p in low for p in ("repeat", "say that again", "what was that")):
            self.prefs["repeat_requested"] = True

    def _safety_note(self, text: str) -> str | None:
        """Flags an emergency, crisis or improper request, and says how to respond."""
        if cc.detect_emergency(text):
            self._flag("emergency", "emergency", "Possible medical emergency", text)
            return (
                "SAFETY CHECK: possible emergency. Reply only: 'Please hang up and call nine one one now.' "
                "Then end the call."
            )
        if cc.detect_crisis(text):
            self._flag("crisis", "emergency", "Caller mentioned self-harm", text)
            return (
                "SAFETY CHECK: the caller may be in crisis. Say warmly and briefly: 'I'm really glad you told me. "
                "You can call or text nine eight eight any time to talk to someone now, or call nine one one if "
                "you're in danger.' Ask if they're safe right now. Do not end the call. If they say they're safe, "
                "offer to keep helping them find a therapist."
            )
        if cc.detect_integrity(text):
            self._flag(
                "integrity",
                "review",
                "Asked to misuse insurance or misstate facts",
                text,
            )
            return (
                "SAFETY CHECK: improper request. Decline in one short, calm sentence: you can only book using the "
                "verified member's own plan and the real reason for the visit. Then offer what you can do."
            )
        return None

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        text = new_message.text_content or ""
        self._note_requests(text)
        if self.prefs["slower_pace"]:
            # turn_ctx edits only last one reply, so repeat this on every turn.
            turn_ctx.add_message(role="system", content=SLOWER_PACE_NOTE)
        safety = self._safety_note(text)
        if safety:
            turn_ctx.add_message(role="system", content=safety)

    def _on_realtime_transcript(self, ev) -> None:
        """A realtime model detects turns itself, so the SDK never calls
        on_user_turn_completed. Run the same checks on its final transcripts."""
        if not ev.is_final:
            return
        was_slow = self.prefs["slower_pace"]
        self._note_requests(ev.transcript)
        if self.prefs["slower_pace"] and not was_slow:
            task = asyncio.create_task(
                self.update_instructions(f"{self.instructions}\n{SLOWER_PACE_NOTE}")
            )
            self._realtime_tasks.add(task)
            task.add_done_callback(self._realtime_tasks.discard)
        safety = self._safety_note(ev.transcript)
        if safety:
            self.session.interrupt()
            self.session.generate_reply(instructions=safety)

    # --------------------------------------------------------------- tools
    @function_tool
    async def verify_identity(
        self,
        context: RunContext,
        caller_full_name: str,
        insurance_id: str,
    ) -> str:
        """Verify the caller before any account detail. Call once you have the caller's
        full name and the member's insurance ID. If the caller is phoning for someone
        else, use that person's insurance ID with the caller's own name.

        Args:
            caller_full_name: The first and last name of the person on the phone.
            insurance_id: The member's insurance ID as letters and digits, e.g. MBR-12345.
        """
        if self.fails >= 3:
            return "Verification is locked for this call. Offer transfer_to_human; share no account details."
        if len(caller_full_name.split()) < 2:
            return "Kindly ask for their first and last name, then call this again."
        self.caller_name = caller_full_name
        mid = cc.member_by_id(insurance_id)
        if not mid:
            self.fails += 1
            board.emit("identity_failed", self.call_id, attempts=self.fails)
            if self.fails >= NOT_ON_FILE_AFTER:
                self._flag("identity", "review", "Caller not found on file")
                return NOT_ON_FILE
            return (
                "That didn't match our records. Do not say which part. You already have their name, so kindly "
                "ask them to repeat just their insurance ID, then call verify_identity again."
            )
        member_name = cc.MEMBERS[mid]["full_name"]
        if not cc.name_matches(mid, caller_full_name):
            if not cc.is_authorized_rep(mid, caller_full_name, self.phone):
                self.fails += 1
                self._flag(
                    "identity",
                    "review",
                    f"{caller_full_name} not authorized for {member_name}",
                )
                return (
                    "Caller is not an authorized representative on file. Say you can't discuss that plan, "
                    "and that the member can call or add them as a representative. Share nothing else."
                )
            self.relationship = "authorized_representative"
        self.member = mid
        board.emit("member", self.call_id, member=self._member_card())
        return (
            f"Verified {member_name} ({cc.plan(mid)['name']}). Say thanks. If you don't know what care "
            "is needed yet, ask that. Otherwise ask for the ZIP code where they want care."
        )

    @function_tool
    async def find_care(
        self,
        context: RunContext,
        specialty: Specialty,
        need_summary: str,
        location: str = "",
        include_days: list[str] | None = None,
        exclude_days: list[str] | None = None,
        time_of_day: Literal["any", "morning", "afternoon"] = "any",
        preferred_date: str = "",
        preferred_time: str = "",
        any_time: bool = False,
        week: Literal["any", "this_week", "next_week"] = "any",
    ) -> str:
        """Find in-network providers with real open times near the member. Call this once
        you know what care they need, their ZIP code, and what date and time suit them.
        Call again whenever the caller changes the day, time or place.

        Args:
            specialty: primary_care, mental_health (therapist, counseling), orthopedics
                (bones, joints, knee), dermatology (skin), cardiology (heart), ob_gyn,
                neurology (headaches, nerves, memory), ent (ear, nose, throat, sinus),
                ophthalmology (eyes, vision), gastroenterology (stomach, digestion),
                endocrinology (diabetes, thyroid), allergy (allergies, asthma), pediatrics
                (a child), physical_therapy (rehab, back or neck strain), urology (bladder,
                kidney stones), podiatry (feet, ankles).
            need_summary: A few words in the caller's terms, e.g. "knee pain".
            location: The five-digit ZIP code the caller gave, or a place they named. Never
                assume one. Leave empty only if they already gave it and it hasn't changed.
            include_days: Weekday names they asked for, e.g. ["thursday"].
            exclude_days: Weekday names they can't do.
            time_of_day: any, morning, or afternoon.
            preferred_date: The date they want as YYYY-MM-DD, worked out from today's date.
                If they give only a day of the month, use the next time it falls.
            preferred_time: The time they want, e.g. 14:30.
            any_time: True if they said any date, any weekday, or the soonest works.
            week: this_week or next_week when they name a week instead of a day, with no
                preferred_date or preferred_time. The result suggests open times in that week.
        """
        if not self.member:
            return "Not verified yet. Verify identity first; share nothing."
        self.offers = {}  # a new search replaces them, so old ones can't be booked
        self.specialty, self.need = specialty, need_summary
        z = cc.resolve_zip(location, self.zip)
        if not z:
            return "Ask for the five-digit ZIP code where they want care, then call find_care with it."
        self.zip = z
        timing_problem = self._update_timing(
            preferred_date,
            preferred_time,
            include_days,
            exclude_days,
            time_of_day,
            any_time,
            week,
        )
        if timing_problem:
            return timing_problem
        pcp = cc.PROVIDER[cc.MEMBERS[self.member]["pcp"]]["name"]
        if cc.needs_referral(self.member, specialty) and not self.store.has_referral(
            self.member, specialty
        ):
            board.emit(
                "rule",
                self.call_id,
                text=f"Plan needs a referral from {pcp} for {cc.SPECIALTIES[specialty]}",
            )
            return (
                f"Their HMO needs a referral from their primary care doctor, {pcp}, before a "
                f"{cc.SPECIALTIES[specialty]} visit is covered. Ask: 'Your plan needs a referral from {pcp} first. "
                "Want me to request it and find you a time?' If yes, call request_referral with confirmed true, "
                "then find_care again."
            )
        if not self.timing:
            return ASK_FOR_TIMING
        if z not in cc.ZIPS:
            return await self._find_live(specialty, z)
        prev = cc.MEMBERS[self.member]["last_seen"].get(specialty)
        res = cc.search(
            self.member,
            specialty,
            z,
            self.store.taken,
            self.timing["include_days"],
            self.timing["exclude_days"],
            self.timing["time_of_day"],
            prev,
            self.timing["search_from"],
            self.timing["minutes"],
            self.timing["search_until"],
        )
        self.store.report_ghosts(res["ghosts"])
        self.offers = dict(zip("AB", res["offers"]))
        board.emit(
            "search",
            self.call_id,
            specialty=cc.SPECIALTIES[specialty],
            area=cc.ZIPS[z][2],
            offers=res["offers"],
            ghosts=res["ghosts"],
            no_match=res["no_match"],
        )
        if not self.offers:
            return (
                "No in-network openings match. Say so in one sentence and ask if another day, time, or a "
                "nearby area works. Do not invent options."
            )
        whole_week = self.timing["week"] != "any"
        lines = []
        for k, o in self.offers.items():
            dist = "under a mile" if o["miles"] < 0.5 else f"{o['miles']} miles"
            again = (
                " (they saw this provider before; mention it)"
                if o["id"] == prev
                else ""
            )
            wanted = self.timing["date"]
            closest = (
                " (the closest opening to the date they asked for; say so)"
                if wanted and o["slot"][:10] != wanted.isoformat()
                else ""
            )
            when = (
                "open "
                + "; ".join(
                    f"time {n}: {cc.slot_words(s)}"
                    for n, s in enumerate(o["open_slots"], start=1)
                )
                if whole_week
                else cc.slot_words(o["slot"])
            )
            lines.append(
                f"Option {k}: {o['name']}, {dist} away, {when}{closest}, "
                f"their cost about {o['cost']} dollars, {cc.cost_reason(self.member, o['id'])}{again}."
            )
        skipped = (
            f" {len(res['ghosts'])} outdated listings were skipped."
            if res["ghosts"]
            else ""
        )
        ask = (
            " Give each doctor's name, open times and cost briefly, then ask which doctor and time they'd "
            "like. Pass their choice to book_appointment as slot_number."
            if whole_week
            else " In two short sentences total, give each option's name, day, time and cost, then ask which one."
        )
        return (
            " ".join(lines)
            + skipped
            + ask
            + " Leave out distance and skipped listings. All of these are in-network. If a cost is high, "
            "explain it using only the reason given here."
        )

    def _update_timing(
        self,
        preferred_date: str,
        preferred_time: str,
        include_days: list[str] | None,
        exclude_days: list[str] | None,
        time_of_day: str,
        any_time: bool,
        week: str = "any",
    ) -> str | None:
        """Remember the date and time the caller asked for. Returns what to say instead
        of searching if the date can't be used; code decides, so no date is guessed."""
        stated = (
            preferred_date,
            preferred_time,
            include_days,
            exclude_days,
            time_of_day != "any",
            any_time,
            week != "any",
        )
        if not any(stated):
            return None  # keep what they said earlier, if anything
        if cc.WEEKEND_DAYS & {day.lower() for day in include_days or []}:
            return self._weekday_only("That day")
        wanted = None
        if preferred_date:
            wanted = cc.parse_date(preferred_date)
            if not wanted:
                return ASK_FOR_TIMING
            if wanted < date.today():
                return (
                    f"{cc.day_words(wanted)} has already passed. Say so kindly and ask what date and "
                    "time would suit them. Do not suggest doctors yet."
                )
            if cc.is_weekend(wanted):
                return self._weekday_only(cc.day_words(wanted))
        search_from, search_until = wanted, None
        minutes = cc.parse_time(preferred_time)
        if week != "any":
            # They named a whole week, so any day or time here was only the model's guess.
            search_from, search_until = cc.week_range(week)
            wanted, minutes = None, None
        self.timing = {
            "date": wanted,
            "week": week,
            "search_from": search_from,
            "search_until": search_until,
            "minutes": minutes,
            "include_days": include_days,
            "exclude_days": exclude_days,
            "time_of_day": time_of_day,
        }
        return None

    @staticmethod
    def _weekday_only(day: str) -> str:
        return (
            f"{day} is on a weekend, and appointments are Monday to Friday only. Say so kindly and "
            "ask which weekday and time would suit them. Do not suggest doctors yet."
        )

    def _timing_words(self) -> str:
        """What the caller asked for, in words an office could act on."""
        t = self.timing or {}
        words = []
        if t.get("date"):
            words.append(cc.day_words(t["date"]))
        if t.get("week") in WEEK_WORDS:
            words.append(WEEK_WORDS[t["week"]])
        words += [day.title() for day in t.get("include_days") or []]
        if t.get("time_of_day", "any") != "any":
            words.append(f"in the {t['time_of_day']}")
        if t.get("minutes") is not None:
            words.append(f"around {cc.clock_words(t['minutes'])}")
        return " ".join(words)

    async def _find_live(self, specialty: str, zip_code: str) -> str:
        """Real providers from the public registry, for a ZIP outside the demo network.

        The registry has no network status, open times or prices, so these can only be
        offered as a request. Never invent a time or a network claim for a real provider.
        """
        self.offers = {}
        try:
            found = await live.search_nearby(specialty, zip_code)
        except live.ProviderLookupError:
            logger.warning("live provider lookup failed", exc_info=True)
            return (
                "The live provider directory isn't answering. Apologize briefly, then ask for a "
                "nearby ZIP code or offer a person. Do not invent options."
            )
        cost = cc.copay_estimate(self.member, specialty)
        offers = [{**p, "cost": cost, "live": True} for p in found[:MAX_LIVE_OFFERS]]
        self.offers = dict(zip("AB", offers))
        board.emit(
            "search",
            self.call_id,
            specialty=cc.SPECIALTIES[specialty],
            area=f"ZIP {zip_code} · live registry",
            offers=offers,
            ghosts=[],
            no_match=[],
        )
        if not offers:
            return (
                "No providers of that type are listed in that ZIP. Say so in one sentence and ask for a "
                "nearby ZIP code. Do not invent options."
            )
        lines = " ".join(
            f"Option {k}: {o['name']}, at {o['address']}."
            for k, o in self.offers.items()
        )
        return (
            f"{lines} These are from the live public registry, so their network status and open times are "
            f"NOT known. Their copay would be about {cost} dollars if in-network. In two short sentences, "
            "give each name and street, say you can send an appointment request and the office sets the "
            "time, then ask which one."
        )

    @function_tool
    async def request_referral(
        self,
        context: RunContext,
        specialty: Specialty,
        reason: str,
        confirmed: bool = False,
    ) -> str:
        """Request a referral from the member's primary care doctor (HMO plans).

        Args:
            specialty: The type of specialist the referral is for.
            reason: A few words, e.g. "knee pain".
            confirmed: True only after the caller said yes to requesting it.
        """
        if not self.member:
            return "Not verified yet."
        if not cc.needs_referral(self.member, specialty):
            return "No referral is needed for this plan. Call find_care."
        pcp = cc.PROVIDER[cc.MEMBERS[self.member]["pcp"]]["name"]
        if not confirmed:
            return f"Ask: 'Should I send a referral request to {pcp} for your {reason}?' Wait for a clear yes."
        self.referral = self.store.add_referral(self.member, specialty, reason)
        board.emit("referral", self.call_id, referral={**self.referral, "pcp": pcp})
        return f"Referral {self.referral['id']} requested from {pcp}. Say it's sent in a few words, then call find_care."

    @function_tool
    async def book_appointment(
        self,
        context: RunContext,
        option: Literal["A", "B"],
        confirmed: bool = False,
        slot_number: int = 0,
    ) -> str:
        """Book one of the options find_care returned. As soon as the caller picks an
        option, call this with confirmed false and say the read-back it returns.

        Args:
            option: A or B, from the latest find_care result.
            confirmed: Leave false on the first call. True only on a later turn, after
                the caller said yes to the read-back this tool returned.
            slot_number: Only when find_care listed numbered times: the one they chose.
                Leave 0 when confirming a time you already read back.
        """
        if not self.member:
            return "Not verified yet."
        o = self.offers.get(option)
        if not o:
            return "That option isn't available. Call find_care first."
        specialty = (
            o["specialty"] if o.get("live") else cc.PROVIDER[o["id"]]["specialty"]
        )
        if cc.needs_referral(self.member, specialty) and not self.store.has_referral(
            self.member, specialty
        ):
            return "A referral is required first. Ask to request it, then call request_referral."
        if o.get("live"):
            return self._request_live(o, confirmed)
        if slot_number:
            open_slots = o.get("open_slots", [])
            if not 1 <= slot_number <= len(open_slots):
                return "That isn't one of the times offered. Ask which of the listed times they'd like."
            o = {**o, "slot": open_slots[slot_number - 1]}
            self.offers = {**self.offers, option: o}
        if not confirmed:
            return (
                f"Read back and ask: '{o['name']}, {cc.slot_words(o['slot'])}, at {o['address']}, about "
                f"{o['cost']} dollars. Shall I book it?' Wait for a clear yes."
            )
        b = self.store.book(
            self.member, o["id"], o["slot"], self.referral and self.referral["id"]
        )
        if not b:
            return "That time was just taken. Apologize briefly and call find_care again for new times."
        self.booking = b
        conf = " ".join(b["confirmation"].replace("-", ""))
        board.emit(
            "booking",
            self.call_id,
            booking={
                **b,
                "provider_name": o["name"],
                "address": o["address"],
                "when": cc.slot_words(o["slot"]),
            },
        )
        self.intake("booked", "none")
        ref = (
            " Note the referral is pending with their primary care doctor."
            if self.referral
            else ""
        )
        return (
            f"Booked. Say: 'You're booked with {o['name']}, {cc.slot_words(o['slot'])}. Your cost is about "
            f"{o['cost']} dollars. Your confirmation is {conf}.'{ref} Ask if there's anything else."
        )

    def _request_live(self, o: dict, confirmed: bool) -> str:
        """Send an appointment request to a live provider, after a read-back and yes."""
        preference = self._timing_words()
        if not confirmed:
            pass_along = (
                f" I'll pass along that you'd like {preference}." if preference else ""
            )
            return (
                f"Read back and ask: 'I'd request an appointment with {o['name']} at {o['address']}. The "
                f"office calls you to set the time, and it's about {o['cost']} dollars if they're in your "
                f"network.{pass_along} Shall I send it?' Wait for a clear yes."
            )
        self.booking = self.store.add_request(
            self.member,
            o["id"],
            o["cost"],
            self.referral and self.referral["id"],
            preference,
        )
        conf = " ".join(self.booking["confirmation"].replace("-", ""))
        board.emit(
            "booking",
            self.call_id,
            booking={
                **self.booking,
                "provider_name": o["name"],
                "address": o["address"],
                "when": "Office will call to set a time",
            },
        )
        self.intake("requested", "provider_office")
        return (
            f"Request sent. Say: 'Your request to {o['name']} is in, and the office will call you to set a "
            f"time. Your reference number is {conf}.' Ask if there's anything else."
        )

    @function_tool
    async def check_coverage(self, context: RunContext, service: str) -> str:
        """Answer whether a service is covered, from the member's plan data only.

        Args:
            service: The service in plain words, e.g. "MRI", "therapy", "flu shot".
        """
        if not self.member:
            return "Verify identity first."
        ans = cc.coverage_answer(self.member, service)
        if not ans:
            return (
                "Not in the plan data. Say you can't confirm that here and offer a benefits specialist "
                "(transfer_to_human). Do not guess."
            )
        return f"From the plan: {ans} Say this in one short sentence."

    @function_tool
    async def transfer_to_human(self, context: RunContext, reason: str) -> str:
        """The caller wants a person, or the request is outside this line (claims, billing,
        unclear coverage, failed verification).

        Args:
            reason: A few words on why.
        """
        self.need = self.need or reason
        self.intake("handed_off", "human_representative")
        board.emit("handoff", self.call_id, reason=reason)
        return (
            "Say: 'I'm passing this to a member services representative with everything you told me, "
            "so you won't have to repeat it.' Then say goodbye and end the call."
        )

    @function_tool
    async def hold_on(self, context: RunContext) -> str:
        """The caller needs a moment ("hold on", "one sec")."""
        self.hold_until = time.time() + HOLD_SECONDS
        board.emit("hold", self.call_id)
        return "Say only: 'Sure, take your time.' Then stay silent until they speak."

    @function_tool
    async def caller_without_id(
        self, context: RunContext, situation: Literal["cant_find_id", "no_plan"]
    ) -> str:
        """The caller can't give their insurance ID, or says they have no plan with us.

        Args:
            situation: cant_find_id if they have a plan but not the ID with them; no_plan
                if they aren't a member.
        """
        board.emit("rule", self.call_id, text=f"Asked to call back: {situation}")
        if situation == "no_plan":
            return (
                "Kindly ask them to buy a Covered plan and call back once they have one. Do not say "
                "where to buy one. Say a warm "
                "goodbye and end the call."
            )
        return (
            "Kindly say their insurance ID is on the front of their insurance card. If they can get it "
            "now, call hold_on and wait. If they can't, ask them to call back once they have it, say a "
            "warm goodbye and end the call."
        )

    @function_tool
    async def compare_doctors(self, context: RunContext) -> str:
        """The caller wants to compare the doctors you offered, asks about a doctor's
        background or experience, or asks which one is better or who you'd pick."""
        if not self.offers:
            return "No doctors have been offered yet. Call find_care first."
        facts = " ".join(
            f"{o['name']}: {o['background']}." for o in self.offers.values()
        )
        return (
            f"{facts} Share these facts kindly, one short sentence per doctor, keeping each fact separate "
            "and adding nothing: board certified does not mean certified in their focus. Never say who is better, "
            "and never recommend one, even if they press you: say the choice is theirs, then ask what "
            "matters most to them or which they'd like."
        )

    @function_tool
    async def handle_off_topic(self, context: RunContext, topic: str) -> str:
        """The caller asked something unrelated to getting care through this line: weather,
        sports, news, jokes, trivia, general questions, or medical advice. Call this instead
        of answering. Do not use it for coverage, costs, appointments, whether you are an AI,
        emergencies, or asking for a person.

        Args:
            topic: A few words on what they asked, e.g. "the weather".
        """
        self.off_topic_count += 1
        board.emit("off_topic", self.call_id, topic=topic)
        offer = (
            " Offer a person too, if they'd rather."
            if self.off_topic_count >= OFF_TOPIC_LIMIT
            else ""
        )
        return (
            "In one short, friendly sentence, say you can only help with booking care here; "
            f"do not answer the question.{offer} {self._next_step()}"
        )

    @function_tool
    async def raise_alert(
        self,
        context: RunContext,
        kind: Literal["integrity", "emergency", "crisis"],
        reason: str,
        caller_quote: str,
    ) -> str:
        """Log a safety or integrity concern: using someone else's insurance, misstating facts
        to get coverage (integrity), a medical emergency, or a mental health crisis.

        Args:
            kind: integrity, emergency, or crisis.
            reason: A few words.
            caller_quote: The caller's exact words.
        """
        self._flag(
            kind, "review" if kind == "integrity" else "emergency", reason, caller_quote
        )
        return {
            "integrity": "Logged. Calmly say you can only use the verified member's own plan and real visit reason, then continue.",
            "emergency": "Logged. Make sure they call nine one one, then end the call.",
            "crisis": "Logged. Share nine eight eight, check they're safe, and stay with them.",
        }[kind]
