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

import logging
import re
import time
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
   person's insurance ID. Then call verify_identity.
3. Map their reason to a type of provider; never diagnose. If unsure, choose primary_care.
   Ask for the ZIP code where they want care; never assume one. Then call find_care.
4. Do exactly what each tool result tells you. Offer at most two options, one sentence each.
5. Before requesting a referral or booking, read the details back and ask yes or no.
   Only after a clear yes, call the tool again with confirmed true. Silence is not a yes.
6. After booking, give the confirmation number slowly, ask if there's anything else,
   then say goodbye and end the call.

Never invent coverage, costs, providers or times: use only what tools return.
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
            instructions=build_instructions(JOB_PROMPT),
            tools=[
                EndCallTool(
                    ignore_on_enter=True,
                    # We say goodbye in the reply that calls this tool, so don't
                    # ask the LLM to generate a second one afterwards.
                    end_instructions=None,
                    extra_description=(
                        "Only after a booking or handoff is done, after an emergency, "
                        "or when the caller asks to stop. Say your short goodbye in "
                        "the same reply that calls this tool."
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
        board.emit("call_started", self.call_id)
        self.phone = _caller_phone()
        await self.session.generate_reply(
            instructions=(
                "Say: 'Hi, this is Covered, your health plan's care line. "
                "Can I get your full name and what you're calling about today?'"
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

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        text = new_message.text_content or ""
        self.hold_until = 0.0
        low = text.lower()
        if any(p in low for p in ("slow down", "too fast", "slower")):
            self.prefs["slower_pace"] = True
        if self.prefs["slower_pace"]:
            # turn_ctx edits only last one reply, so repeat this on every turn.
            turn_ctx.add_message(
                role="system",
                content="Caller asked for a slower pace: give one detail per sentence, very short.",
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
            if self.fails >= 3:
                self._flag("identity", "review", "Three failed verification attempts")
                return "Third failed attempt. Say you can't verify them, offer a person, share no details."
            return (
                "That didn't match our records. Do not say which part. Kindly ask them to repeat their full "
                "name and insurance ID."
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
    ) -> str:
        """Find in-network providers with real open times near the member. Call this as
        soon as you know what kind of care they need; do not ask about days or times
        first. Call again whenever the caller changes the day, time or place.

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
        """
        if not self.member:
            return "Not verified yet. Verify identity first; share nothing."
        self.specialty, self.need = specialty, need_summary
        z = cc.resolve_zip(location, self.zip)
        if not z:
            return "Ask for the five-digit ZIP code where they want care, then call find_care with it."
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
        if z not in cc.ZIPS:
            return await self._find_live(specialty, z)
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
            + " In two short sentences total, give each option's name, day, time and cost, then ask which one."
            " Leave out distance and skipped listings. If a cost is high, say why in a few words."
        )

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
        self, context: RunContext, option: Literal["A", "B"], confirmed: bool = False
    ) -> str:
        """Book one of the options find_care returned. As soon as the caller picks an
        option, call this with confirmed false and say the read-back it returns.

        Args:
            option: A or B, from the latest find_care result.
            confirmed: Leave false on the first call. True only on a later turn, after
                the caller said yes to the read-back this tool returned.
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
        if not confirmed:
            return (
                f"Read back and ask: 'I'd request an appointment with {o['name']} at {o['address']}. The "
                f"office calls you to set the time, and it's about {o['cost']} dollars if they're in your "
                "network. Shall I send it?' Wait for a clear yes."
            )
        self.booking = self.store.add_request(
            self.member, o["id"], o["cost"], self.referral and self.referral["id"]
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
