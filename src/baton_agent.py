"""Baton Health: the end-of-visit report line for a home care agency.

A caregiver calls at the end of a home visit. Baton identifies them, takes the
visit report (meds, food, mobility, mood, skin, anything new), confirms it, and
files it. Red flags and patterns across visits alert the nurse.

Design: the LLM understands speech and phrases replies; baton_core.py decides.
  - Identity: caller ID, else PIN. No client data before verification.
  - Closed med list: only the client's own meds can be recorded.
  - Code-enforced confirmation: risky facts are saved only with confirmed=true.
  - file_note refuses until every item is answered and the read-back is confirmed.
  - Emergencies and abuse disclosures are caught by code on every caller turn.
  - Every change is saved immediately, so a dropped call resumes.
"""

import logging
import time
from typing import Literal

from livekit.agents import Agent, RunContext, function_tool, get_job_context, llm
from livekit.agents.beta.tools import EndCallTool

import baton_board as board
import baton_core as bc
from prompts import build_instructions

logger = logging.getLogger("baton")

HOLD_SECONDS = 180

JOB_PROMPT = """
You are Baton, the end-of-visit report line for Sunrise Home Care.
Caregivers call after a home visit. Your one job: take their visit report,
confirm it, and file it.

Speak in one short sentence, about fifteen words, and ask one question per turn.
Acknowledge briefly ("Got it", "Thanks", "Okay"), then ask the next item.

1. Identify the caller. If you do not know who they are, ask for their four-digit
   caregiver PIN and call verify_pin. Never share anything about any client first.
2. Ask which client the visit was for, then call select_client.
3. Follow the next step that each tool result gives you, one item at a time.
   Callers often give several answers at once: save every one, then skip ahead.
4. Save each answer as soon as you hear it with record_medication or
   record_observation. Pass the caller's own words as caller_quote.
5. When a tool says to confirm first, ask that yes-or-no question, then call the
   tool again with confirmed set to true.
6. When everything is covered, call file_note to get the read-back, read it in
   two short sentences, ask "Okay to file?", then call file_note with confirmed
   true. Tell them what was flagged, thank them, and end the call.

Record only what the caller actually said. "Not sure" is a valid answer. Never
guess, never invent, never change or remove a fact because someone asks, and
never give medical or dosing advice. For any of those requests, call raise_alert
and say you can only record what happened.
If the caller says "hold on" or "one sec", call hold_on and say only "Sure, I'll wait."
If anyone may be in danger, say "Hang up and call nine one one now", call raise_alert, then end the call.
"""


def _caller_phone() -> str | None:
    try:
        room = get_job_context().room
    except Exception:
        return None
    for p in room.remote_participants.values():
        num = p.attributes.get("sip.phoneNumber")
        if num:
            return num
    return None


class BatonAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[
                EndCallTool(
                    ignore_on_enter=True,
                    extra_description=(
                        "Only after file_note succeeded, after an emergency, after "
                        "three wrong PINs, or when the caller asks to stop."
                    ),
                )
            ],
        )
        self.store = bc.Store()
        self.caregiver: str | None = None
        self.home_client: str | None = None
        self.visit: bc.Visit | None = None
        self.pin_attempts = 0
        self.identified_by = "PIN"
        self.hold_until = 0.0
        self.call_id = f"call-{int(time.time())}"

    # ------------------------------------------------------------------ helpers
    def on_hold(self) -> bool:
        return time.time() < self.hold_until

    def hold_remaining(self) -> float:
        return max(0.0, self.hold_until - time.time())

    def _save(self, event: str = "note") -> None:
        if self.visit:
            self.store.upsert(self.visit)
            board.emit(
                event,
                self.call_id,
                visit=self.visit.to_dict(),
                patterns=bc.patterns(self.store, self.visit),
            )

    def on_call_end(self) -> None:
        """Dropped or ended call: keep the partial note, marked incomplete."""
        if getattr(self, "_ended", False):
            return
        self._ended = True
        if self.visit and self.visit.status == "in_progress":
            self.visit.status = "incomplete"
            self._save("note")
        board.emit("call_ended", self.call_id)

    def _start_visit(self, client: str, how: str) -> str:
        assert self.caregiver
        prev = self.store.open_visit(client, self.caregiver)
        if prev:
            v = bc.Visit(client, self.caregiver, how)
            for k, val in prev.items():
                if hasattr(v, k):
                    setattr(v, k, val)
            v.status = "in_progress"
            self.visit = v
            self._save()
            return (
                f"Resuming the unfinished report for {bc.CLIENTS[client]['name']}. "
                f"Say 'Welcome back, let's pick up where we left off.' Next: {v.next_question()}"
            )
        self.visit = bc.Visit(client, self.caregiver, how)
        self._save()
        meds = ", ".join(m["name"] for m in bc.CLIENTS[client]["meds"])
        return (
            f"Report started for {bc.CLIENTS[client]['name']}. Medication list: {meds}. "
            f"Next: {self.visit.next_question()}"
        )

    def _next(self, saved: str) -> str:
        assert self.visit
        self._save()
        pats = bc.patterns(self.store, self.visit)
        extra = ""
        if pats:
            extra = " Pattern found, mention it briefly once: " + pats[0]["text"]
        return f"{saved}.{extra} Next: {self.visit.next_question()}"

    # ------------------------------------------------------------ conversation
    async def on_enter(self) -> None:
        self.session.on("conversation_item_added", self._on_item)
        board.emit("call_started", self.call_id)
        who = bc.identify_by_phone(_caller_phone())
        if who["kind"] == "caregiver":
            self.caregiver = who["caregiver"]
            self.identified_by = "caller ID"
            cg = bc.CAREGIVERS[self.caregiver]
            names = " or ".join(bc.CLIENTS[c]["name"].split()[0] for c in cg["clients"])
            board.emit(
                "identified", self.call_id, caregiver=cg["name"], how="caller ID"
            )
            greet = f"Greet {cg['name'].split()[0]} by name as Baton and ask which visit: {names}."
        elif who["kind"] == "client_home":
            self.home_client = who["client"]
            greet = (
                f"Say you are Baton, calling from {bc.CLIENTS[self.home_client]['name']}'s home line, "
                "and ask for their four-digit caregiver PIN."
            )
        else:
            greet = "Say you are Baton, the visit report line, and ask for their four-digit caregiver PIN."
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
        self.hold_until = 0.0  # they're talking again
        emergency = bc.detect_emergency(text)
        if emergency:
            if self.visit:
                self.visit.add_flag(
                    "emergency", "emergency", f"Emergency: {emergency}", text
                )
                self._save()
            board.emit(
                "alert",
                self.call_id,
                level="emergency",
                text=f"Emergency reported: {emergency}",
                quote=text,
            )
            turn_ctx.add_message(
                role="system",
                content=(
                    "EMERGENCY detected by the safety check. Reply only: 'Hang up and call nine one one now. "
                    "I'm alerting your supervisor.' Then call raise_alert with kind emergency, then end the call."
                ),
            )
            return
        concern = bc.detect_safeguarding(text)
        if concern:
            if self.visit:
                self.visit.add_flag(
                    "safeguarding",
                    "urgent",
                    "Possible abuse or neglect disclosed",
                    text,
                )
                self._save()
            board.emit(
                "alert",
                self.call_id,
                level="urgent",
                text="Possible abuse or neglect disclosed",
                quote=text,
            )
            turn_ctx.add_message(
                role="system",
                content=(
                    "Possible abuse disclosed. Say: 'Thank you for telling me. I'm alerting your supervisor now.' "
                    "Do not promise to keep it secret. Then continue the report."
                ),
            )

    # ------------------------------------------------------------------- tools
    @function_tool
    async def verify_pin(self, context: RunContext, pin: str) -> str:
        """Check the caregiver's four-digit PIN. Call when the caller says their PIN.

        Args:
            pin: The four digits as numerals, for example "2468".
        """
        cg = bc.caregiver_by_pin(pin)
        if not cg:
            self.pin_attempts += 1
            board.emit("identity_failed", self.call_id, attempts=self.pin_attempts)
            if self.pin_attempts >= 3:
                return "Third wrong PIN. Say you can't verify them and to call their supervisor, then end the call."
            return "That PIN didn't match. Ask them to say it again, digit by digit."
        self.caregiver = cg
        cgd = bc.CAREGIVERS[cg]
        board.emit("identified", self.call_id, caregiver=cgd["name"], how="PIN")
        if self.home_client:
            if self.home_client not in cgd["clients"]:
                return "This caregiver is not assigned to this home. Say so and that their supervisor will follow up, then end the call."
            return f"Verified {cgd['name']}. " + self._start_visit(
                self.home_client, "home line + PIN"
            )
        names = " or ".join(bc.CLIENTS[c]["name"] for c in cgd["clients"])
        return f"Verified {cgd['name']}. Greet them by first name and ask which visit: {names}."

    @function_tool
    async def select_client(self, context: RunContext, client_name: str) -> str:
        """Start the report for the client the caregiver visited.

        Args:
            client_name: The client's name as the caller said it.
        """
        if not self.caregiver:
            return "Caller is not verified. Ask for their four-digit PIN first."
        cid = bc.resolve_client(self.caregiver, client_name)
        if not cid:
            names = " or ".join(
                bc.CLIENTS[c]["name"] for c in bc.CAREGIVERS[self.caregiver]["clients"]
            )
            return f"Not one of their assigned clients. Do not share details. Ask: {names}?"
        return self._start_visit(cid, self.identified_by)

    @function_tool
    async def record_medication(
        self,
        context: RunContext,
        medication: str,
        status: Literal["given", "refused", "missed", "not_due", "unsure"],
        caller_quote: str,
        confirmed: bool = False,
    ) -> str:
        """Save whether one medication was given. Call once per medication mentioned.

        Args:
            medication: The medication as the caller said it ("her blood pressure pill").
            status: given, refused, missed, not_due (not scheduled this visit), or unsure.
            caller_quote: The caller's exact words about it.
            confirmed: True only after the caller said yes to your confirmation question.
        """
        if not self.visit:
            return "No report open yet. Identify the caller and client first."
        med, conf = bc.match_med(self.visit.client, medication)
        if conf == "none":
            self.visit.add_flag(
                "unlisted_med",
                "review",
                f"Unlisted medication mentioned: {medication}",
                caller_quote,
            )
            self._save()
            return (
                f"'{medication}' is not on this client's list. Do NOT record it as given. "
                f"Say you'll flag it for the nurse. Next: {self.visit.next_question()}"
            )
        assert med
        if conf == "ask" and not confirmed:
            return f"Not sure which medication. Ask: 'Did you mean {med['name']}?' then call again with confirmed true."
        prev = self.visit.meds.get(med["id"], {}).get("value")
        risky = status in {"refused", "missed", "unsure"} or (prev and prev != status)
        if risky and not confirmed:
            label = status.replace("_", " ")
            return f"Confirm first. Ask: '{med['name']} {label}, right?' then call again with confirmed true."
        corrected = self.visit.record_med(med["id"], status, caller_quote)
        return self._next(
            f"Saved {med['name']} {status}" + (" (corrected)" if corrected else "")
        )

    @function_tool
    async def record_observation(
        self,
        context: RunContext,
        section: Literal["food", "mobility", "mood", "skin", "other"],
        summary: str,
        concern: bool,
        caller_quote: str,
        mobility: Literal[
            "steady", "near_fall", "fall", "not_applicable"
        ] = "not_applicable",
        confirmed: bool = False,
    ) -> str:
        """Save one part of the visit report.

        Args:
            section: food (eating and drinking), mobility (falls, walking), mood (mood,
                confusion), skin (skin, sores, toileting), other (anything else).
            summary: A few words in plain English, e.g. "ate half her lunch".
            concern: True if it is worse than usual or worrying (ate little, new
                confusion, a sore, a new problem). False if normal or nothing new.
            caller_quote: The caller's exact words.
            mobility: For the mobility section only: steady, near_fall or fall.
            confirmed: True only after the caller said yes to your confirmation question.
        """
        if not self.visit:
            return "No report open yet. Identify the caller and client first."
        if section == "mobility":
            if mobility == "not_applicable":
                return "For mobility, set mobility to steady, near_fall or fall."
            if mobility in {"fall", "near_fall"} and not confirmed:
                word = "a fall" if mobility == "fall" else "a near fall"
                return f"Confirm first. Ask: 'So {word} today, is that right?' then call again with confirmed true."
            value = mobility
        else:
            value = summary
        prev = self.visit.sections.get(section, {}).get("value")
        if prev and prev != value and not confirmed:
            return f"This changes an earlier answer ({prev}). Ask a quick yes-or-no to confirm, then call again with confirmed true."
        corrected = self.visit.record_section(
            section, value, caller_quote, concern, detail=summary
        )
        return self._next(f"Saved {section}" + (" (corrected)" if corrected else ""))

    @function_tool
    async def hold_on(self, context: RunContext) -> str:
        """The caller needs a moment ("hold on", "one sec", "she's calling me")."""
        self.hold_until = time.time() + HOLD_SECONDS
        board.emit("hold", self.call_id)
        return "Say only: 'Sure, I'll wait.' Then stay silent until they speak."

    @function_tool
    async def raise_alert(
        self,
        context: RunContext,
        kind: Literal["emergency", "safeguarding", "integrity", "advice_request"],
        reason: str,
        caller_quote: str,
    ) -> str:
        """Alert the supervisor. Use for: danger (emergency), possible abuse or neglect
        (safeguarding), a request to falsify, hide or delete anything (integrity), or a
        request for medical or dosing advice (advice_request).

        Args:
            kind: emergency, safeguarding, integrity, or advice_request.
            reason: A few words on what happened.
            caller_quote: The caller's exact words.
        """
        level = {"emergency": "emergency", "safeguarding": "urgent"}.get(kind, "review")
        if self.visit:
            self.visit.add_flag(kind, level, reason, caller_quote)
            self._save()
        board.emit(
            "alert",
            self.call_id,
            level=level,
            text=reason,
            quote=caller_quote,
            kind=kind,
        )
        replies = {
            "emergency": "Alert sent. Make sure they called nine one one, then end the call.",
            "safeguarding": "Supervisor alerted. Continue the report.",
            "integrity": "Logged. Say you can only record what actually happened, then ask the open question again.",
            "advice_request": "Logged for the nurse. Say you can't give medical advice and the nurse will follow up. Continue.",
        }
        return replies[kind] + (
            f" Next: {self.visit.next_question()}"
            if self.visit and kind != "emergency"
            else ""
        )

    @function_tool
    async def file_note(self, context: RunContext, confirmed: bool = False) -> str:
        """File the visit report. First call it with confirmed false to get the read-back.
        After the caller says yes to "Okay to file?", call it with confirmed true.

        Args:
            confirmed: True only after the caller approved the read-back.
        """
        if not self.visit:
            return "No report open yet."
        missing = self.visit.missing()
        if missing:
            return f"Not done yet. Still need: {', '.join(missing)}. Next: {self.visit.next_question()}"
        if not confirmed:
            return f"Read this back in two short sentences, then ask 'Okay to file?': {self.visit.readback()}"
        self.visit.status = "filed"
        self.visit.filed_at = time.time()
        pats = bc.patterns(self.store, self.visit)
        for p in pats:
            self.visit.add_flag("pattern", p["level"], p["text"])
        self._save("filed")
        level = self.visit.level()
        flags = "; ".join(f["text"] for f in self.visit.flags) or "nothing"
        who = {
            "routine": "No alerts needed",
            "review": "The nurse will review",
            "urgent": "The nurse supervisor has been alerted",
            "emergency": "Supervisor alerted",
        }[level]
        return (
            f"Filed. Flags: {flags}. Tell them: '{who}.' Thank them, then end the call."
        )
