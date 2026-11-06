"""Covered: a health plan's care line. Members call, and leave with a booked,
in-network appointment that is really available and priced before they say yes.

Design: the LLM listens and talks; covered_core.py decides.
  - Identity: caller ID is a hint, date of birth is proof; reps must be on file.
  - Ghost-proof search: only providers with a real open slot are ever offered.
  - Cost: the member's own estimate (deductible, copay, coinsurance), from code.
  - Plan rules: HMO specialists need a referral; coverage answers come from data only.
  - Code-enforced confirmation: referral and booking need confirmed=true after a read-back.
  - Every caller turn is checked for emergency, crisis and fraud requests.
"""

import logging
import time
from typing import Literal

from livekit.agents import Agent, RunContext, function_tool, get_job_context, llm
from livekit.agents.beta.tools import EndCallTool

import baton_board as board
import covered_core as cc
from prompts import build_instructions

logger = logging.getLogger("covered")

HOLD_SECONDS = 180
Specialty = Literal[
    "primary_care",
    "mental_health",
    "orthopedics",
    "dermatology",
    "cardiology",
    "ob_gyn",
]

JOB_PROMPT = """
You are Covered, the care line for Covered Health, a health insurance plan.
Members call to get care. Your one job: book them an in-network appointment
they can actually get, and tell them what it will cost before they agree.

Speak in one short sentence, about fifteen words, with one question per turn.
Use plain words. Explain a term only if it matters ("in-network means your plan covers it").

1. Verify identity before any account detail. Ask for full name, then date of birth,
   saying it's to protect their account. If they call for someone else, ask that
   person's full name too. Then call verify_identity.
2. Ask what care they need, in their own words. Map it to a type of provider; never
   diagnose. If unsure, choose primary_care. Then call find_care.
3. Do exactly what each tool result tells you. Offer at most two options, one sentence each.
4. Before requesting a referral or booking, read the details back and ask yes or no.
   Only after a clear yes, call the tool again with confirmed true. Silence is not a yes.
5. After booking, give the confirmation number slowly, ask if there's anything else,
   then say goodbye and end the call.

Never invent coverage, costs, providers or times: use only what tools return.
Never book for anyone you have not verified, and never use someone else's insurance.
If they want a person, call transfer_to_human. If they say "hold on", call hold_on.
Callers can ask you to slow down or repeat at any time; do it without fuss.
"""


def _caller_phone() -> str | None:
    try:
        room = get_job_context().room
    except Exception:
        return None
    for p in room.remote_participants.values():
        if p.attributes.get("sip.phoneNumber"):
            return p.attributes["sip.phoneNumber"]
    return None


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[
                EndCallTool(
                    ignore_on_enter=True,
                    extra_description=(
                        "Only after a booking or handoff is done, after an emergency, "
                        "or when the caller asks to stop."
                    ),
                )
            ],
        )
        self.store = cc.Store()
        self.call_id = f"CALL-{int(time.time()) % 100000}"
        self.phone: str | None = None
        self.phone_hint: str | None = None  # member id suggested by caller ID
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
        self._ended = False

    # ------------------------------------------------------------- helpers
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
            "how": "caller ID + date of birth"
            if self.phone_hint == self.member
            else "name + date of birth",
        }

    # ----------------------------------------------------------- lifecycle
    async def on_enter(self) -> None:
        self.session.on("conversation_item_added", self._on_item)
        board.emit("call_started", self.call_id)
        self.phone = _caller_phone()
        self.phone_hint = cc.member_by_phone(self.phone)
        rep = cc.rep_by_phone(self.phone)
        if self.phone_hint:
            first = cc.MEMBERS[self.phone_hint]["preferred_name"]
            greet = f"Say: 'Hi, this is Covered, your health plan's care line. Am I speaking with {first}?'"
        elif rep:
            greet = f"Say: 'Hi, this is Covered, your health plan's care line. Is this {rep[1].split()[0]}?'"
        else:
            greet = (
                "Say: 'Hi, this is Covered, your health plan's care line. I can find and book care for you. "
                "What's your full name?'"
            )
        await self.session.generate_reply(instructions=greet)

    def _on_item(self, ev) -> None:
        item = getattr(ev, "item", None)
        text = getattr(item, "text_content", None)
        if text:
            board.emit(
                "transcript", self.call_id, role=getattr(item, "role", ""), text=text
            )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        text = new_message.text_content or ""
        self.hold_until = 0.0
        low = text.lower()
        if any(p in low for p in ("slow down", "too fast", "slower")):
            self.prefs["slower_pace"] = True
            turn_ctx.add_message(
                role="system",
                content="Caller asked for a slower pace: from now on give one detail per sentence, very short.",
            )
        if any(p in low for p in ("repeat", "say that again", "what was that")):
            self.prefs["repeat_requested"] = True
        if cc.detect_emergency(text):
            self._flag("emergency", "emergency", "Possible medical emergency", text)
            turn_ctx.add_message(
                role="system",
                content=(
                    "SAFETY CHECK: possible emergency. Reply only: 'Please hang up and call nine one one now.' "
                    "Then end the call."
                ),
            )
            return
        if cc.detect_crisis(text):
            self._flag("crisis", "emergency", "Caller mentioned self-harm", text)
            turn_ctx.add_message(
                role="system",
                content=(
                    "SAFETY CHECK: the caller may be in crisis. Say warmly and briefly: 'I'm really glad you told me. "
                    "You can call or text nine eight eight any time to talk to someone now, or call nine one one if "
                    "you're in danger.' Ask if they're safe right now. Do not end the call. If they say they're safe, "
                    "offer to keep helping them find a therapist."
                ),
            )
            return
        if cc.detect_integrity(text):
            self._flag(
                "integrity",
                "review",
                "Asked to misuse insurance or misstate facts",
                text,
            )
            turn_ctx.add_message(
                role="system",
                content=(
                    "SAFETY CHECK: improper request. Decline in one short, calm sentence: you can only book using the "
                    "verified member's own plan and the real reason for the visit. Then offer what you can do."
                ),
            )

    # --------------------------------------------------------------- tools
    @function_tool
    async def verify_identity(
        self,
        context: RunContext,
        caller_full_name: str,
        date_of_birth: str,
        calling_for: Literal["self", "someone_else"] = "self",
        member_full_name: str = "",
    ) -> str:
        """Verify the caller before any account detail. Call after you have the caller's
        name and the member's date of birth.

        Args:
            caller_full_name: The caller's full name.
            date_of_birth: The MEMBER's date of birth as YYYY-MM-DD.
            calling_for: self, or someone_else if calling about another person's plan.
            member_full_name: Only when calling_for is someone_else: that person's full name.
        """
        if self.fails >= 3:
            return "Verification is locked for this call. Offer transfer_to_human; share no account details."
        self.caller_name = caller_full_name
        dob = cc.normalize_dob(date_of_birth)
        if calling_for == "self":
            mid = self.phone_hint or cc.member_by_name(caller_full_name)
            if (
                mid
                and self.phone_hint
                and cc.member_by_name(caller_full_name) not in (None, mid)
            ):
                mid = cc.member_by_name(
                    caller_full_name
                )  # someone else using this phone
        else:
            mid = cc.member_by_name(member_full_name)
            if mid and not cc.is_authorized_rep(mid, caller_full_name, self.phone):
                self.fails += 1
                self._flag(
                    "identity",
                    "review",
                    f"{caller_full_name} not authorized for {member_full_name}",
                )
                return (
                    "Caller is not an authorized representative on file. Say you can't discuss that plan, "
                    "and that the member can call or add them as a representative. Share nothing else."
                )
            self.relationship = "authorized_representative"
        if not mid or not dob or dob != cc.MEMBERS[mid]["dob"]:
            self.fails += 1
            board.emit("identity_failed", self.call_id, attempts=self.fails)
            if self.fails >= 3:
                self._flag("identity", "review", "Three failed verification attempts")
                return "Third failed attempt. Say you can't verify them, offer a person, share no details."
            return (
                "That didn't match our records. Do not say which part. Ask them to repeat their full name "
                "and date of birth."
            )
        self.member = mid
        self.zip = cc.MEMBERS[mid]["zip"]
        board.emit("member", self.call_id, member=self._member_card())
        who = "you" if calling_for == "self" else cc.MEMBERS[mid]["preferred_name"]
        return (
            f"Verified {cc.MEMBERS[mid]['full_name']} ({cc.plan(mid)['name']}, ZIP {self.zip}). "
            f"Say thanks, and ask what kind of care {who} need today."
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
    ) -> str:
        """Find in-network providers with real open times near the member. Call again
        whenever the caller changes the day, time or place.

        Args:
            specialty: primary_care, mental_health (therapist, counseling), orthopedics
                (bones, joints, knee, back), dermatology (skin), cardiology (heart), ob_gyn.
            need_summary: A few words in the caller's terms, e.g. "knee pain".
            location: Only if they are somewhere other than home: a ZIP or place name.
            include_days: Weekday names they asked for, e.g. ["thursday"].
            exclude_days: Weekday names they can't do.
            time_of_day: any, morning, or afternoon.
        """
        if not self.member:
            return "Not verified yet. Verify identity first; share nothing."
        self.specialty, self.need = specialty, need_summary
        home = cc.MEMBERS[self.member]["zip"]
        z = cc.resolve_zip(location, home)
        if not z:
            return "That area isn't in our search. Ask for a ZIP in Jersey City, Hoboken, Manhattan, Harlem or Brooklyn."
        self.zip = z
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
        prev = cc.MEMBERS[self.member]["last_seen"].get(specialty)
        res = cc.search(
            self.member,
            specialty,
            z,
            self.store.taken,
            include_days,
            exclude_days,
            time_of_day,
            prev,
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
        lines = []
        for k, o in self.offers.items():
            dist = "under a mile" if o["miles"] < 0.5 else f"{o['miles']} miles"
            again = (
                " (they saw this provider before; mention it)"
                if o["id"] == prev
                else ""
            )
            lines.append(
                f"Option {k}: {o['name']}, {dist} away, {cc.slot_words(o['slot'])}, "
                f"their cost about {o['cost']} dollars{again}."
            )
        skipped = (
            f" {len(res['ghosts'])} outdated listings were skipped."
            if res["ghosts"]
            else ""
        )
        return (
            " ".join(lines)
            + skipped
            + " Offer these briefly, one sentence each, then ask which one. If a cost is high, say why in a few words."
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
        self, context: RunContext, option: Literal["A", "B"], confirmed: bool = False
    ) -> str:
        """Book one of the options find_care returned.

        Args:
            option: A or B, from the latest find_care result.
            confirmed: True only after you read back provider, day, time and cost and the caller said yes.
        """
        if not self.member:
            return "Not verified yet."
        o = self.offers.get(option)
        if not o:
            return "That option isn't available. Call find_care first."
        if cc.needs_referral(
            self.member, cc.PROVIDER[o["id"]]["specialty"]
        ) and not self.store.has_referral(
            self.member, cc.PROVIDER[o["id"]]["specialty"]
        ):
            return "A referral is required first. Ask to request it, then call request_referral."
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
