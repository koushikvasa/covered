"""Baton Health core: the parts that must never be wrong live here, in plain code.

The LLM understands speech and phrases replies. This module decides:
  - who is calling (caller ID / PIN), and which clients they may report on
  - which medications exist for a client (closed list, fuzzy-matched)
  - what is still missing before a note can be filed
  - red flags, triage level, emergencies, and patterns across visits

No LiveKit imports, so it is unit-tested in tests/test_baton_core.py.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Demo agency data (fake). Swap for the agency's real records.
# --------------------------------------------------------------------------

CLIENTS: dict[str, dict] = {
    "rosa": {
        "name": "Rosa Delgado",
        "landline": "+12015550142",
        "meds": [
            {
                "id": "lisinopril",
                "name": "Lisinopril",
                "purpose": "blood pressure",
                "aliases": ["blood pressure pill", "bp pill", "pressure pill"],
                "due": "morning",
                "critical": True,
            },
            {
                "id": "metformin",
                "name": "Metformin",
                "purpose": "diabetes",
                "aliases": ["sugar pill", "diabetes pill"],
                "due": "morning and evening",
                "critical": True,
            },
            {
                "id": "donepezil",
                "name": "Donepezil",
                "purpose": "memory",
                "aliases": ["memory pill", "aricept"],
                "due": "bedtime",
                "critical": False,
            },
        ],
    },
    "james": {
        "name": "James Okafor",
        "landline": "+12015550177",
        "meds": [
            {
                "id": "warfarin",
                "name": "Warfarin",
                "purpose": "blood thinner",
                "aliases": ["blood thinner", "coumadin"],
                "due": "evening",
                "critical": True,
            },
            {
                "id": "furosemide",
                "name": "Furosemide",
                "purpose": "fluid",
                "aliases": ["water pill", "lasix"],
                "due": "morning",
                "critical": False,
            },
        ],
    },
}

CAREGIVERS: dict[str, dict] = {
    "maria": {
        "name": "Maria Santos",
        "phone": "+12015550199",
        "pin": "2468",
        "clients": ["rosa", "james"],
    },
    "devon": {
        "name": "Devon Price",
        "phone": "+12015550133",
        "pin": "1357",
        "clients": ["james"],
    },
}

# Extra caregiver phones for the demo (e.g. your own mobile), "+1555...=maria,+1..=devon"
for pair in filter(None, os.getenv("BATON_DEMO_PHONES", "").split(",")):
    num, _, cg = pair.partition("=")
    if cg.strip() in CAREGIVERS:
        CAREGIVERS[cg.strip()].setdefault("extra_phones", []).append(num.strip())

SECTIONS = ["meds", "food", "mobility", "mood", "skin", "other"]
SECTION_QUESTIONS = {
    "food": "How did {first} eat and drink today?",
    "mobility": "Any falls, near falls, or new unsteadiness today?",
    "mood": "Any change in mood or new confusion?",
    "skin": "Any skin problems, sores, or toileting issues?",
    "other": "Anything else the nurse or next caregiver should know?",
}
MED_STATUSES = {"given", "refused", "missed", "not_due", "unsure"}
CRITICAL_SECTIONS = {"mobility"}  # changes here need an explicit yes before saving

LEVELS = ["routine", "review", "urgent", "emergency"]

# --------------------------------------------------------------------------
# Speech helpers
# --------------------------------------------------------------------------

EMERGENCY_PHRASES = [
    "not breathing",
    "isn't breathing",
    "stopped breathing",
    "can't breathe",
    "cannot breathe",
    "unresponsive",
    "won't wake up",
    "not waking up",
    "passed out",
    "chest pain",
    "can't get up",
    "cannot get up",
    "bleeding a lot",
    "lot of blood",
    "having a stroke",
    "seizure",
    "call 911",
    "call nine one one",
    "choking",
    "hit her head",
    "hit his head",
]
SAFEGUARDING_PHRASES = [
    "hits her",
    "hits him",
    "hit her",
    "hit him",
    "slapped",
    "hurting her",
    "hurting him",
    "abuse",
    "abusing",
    "bruises from",
    "pushed her",
    "pushed him",
    "threatens",
    "scared of her son",
    "scared of his son",
    "takes her money",
    "takes his money",
]
NEGATIONS = {"not", "no", "never", "didn't", "doesn't", "wasn't", "isn't", "nothing"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower()).strip()


def _negated(text: str, start: int) -> bool:
    prev = text[:start].split()[-3:]
    return any(w in NEGATIONS for w in prev)


def _find(text: str, phrases: list[str]) -> str | None:
    t = _norm(text)
    for p in phrases:
        i = t.find(p)
        if i >= 0 and not _negated(t, i):
            return p
    return None


def detect_emergency(text: str) -> str | None:
    """Unambiguous emergency phrase in the caller's words, ignoring negations."""
    return _find(text, EMERGENCY_PHRASES)


def detect_safeguarding(text: str) -> str | None:
    """Possible abuse or neglect disclosure."""
    return _find(text, SAFEGUARDING_PHRASES)


def match_med(client_id: str, spoken: str) -> tuple[dict | None, str]:
    """Match a spoken medication against the client's own list only.

    Returns (med, confidence) where confidence is "sure", "ask" or "none".
    """
    meds = CLIENTS[client_id]["meds"]
    s = _norm(spoken)
    if not s:
        return None, "none"
    for m in meds:
        names = [m["id"], _norm(m["name"]), m["purpose"], *m["aliases"]]
        if any(n and (n == s or n in s or (len(s) > 3 and s in n)) for n in names):
            return m, "sure"
    best, score = None, 0.0
    for m in meds:
        for n in [m["id"], *m["aliases"]]:
            r = difflib.SequenceMatcher(None, s, n).ratio()
            # also compare word by word ("listen april" vs "lisinopril")
            r = max(
                r,
                difflib.SequenceMatcher(
                    None, s.replace(" ", ""), n.replace(" ", "")
                ).ratio(),
            )
            if r > score:
                best, score = m, r
    if score >= 0.8:
        return best, "sure"
    if score >= 0.55:
        return best, "ask"
    return None, "none"


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def _digits(phone: str) -> str:
    d = re.sub(r"\D", "", phone or "")
    return d[-10:]


def identify_by_phone(phone: str | None) -> dict:
    """Caller ID lookup. Returns {"kind": "caregiver"|"client_home"|"unknown", ...}."""
    d = _digits(phone or "")
    if not d:
        return {"kind": "unknown"}
    for cid, cg in CAREGIVERS.items():
        if d in {
            _digits(cg["phone"]),
            *(_digits(p) for p in cg.get("extra_phones", [])),
        }:
            return {"kind": "caregiver", "caregiver": cid}
    for cid, c in CLIENTS.items():
        if d == _digits(c["landline"]):
            return {"kind": "client_home", "client": cid}
    return {"kind": "unknown"}


def caregiver_by_pin(pin: str) -> str | None:
    p = re.sub(r"\D", "", pin or "")
    for cid, cg in CAREGIVERS.items():
        if p == cg["pin"]:
            return cid
    return None


def resolve_client(caregiver_id: str, spoken: str) -> str | None:
    """Only clients assigned to this caregiver can be selected."""
    s = _norm(spoken)
    for cid in CAREGIVERS[caregiver_id]["clients"]:
        full = _norm(CLIENTS[cid]["name"])
        if s and (
            s in full or full in s or any(part in s.split() for part in full.split())
        ):
            return cid
    return None


# --------------------------------------------------------------------------
# Visit note
# --------------------------------------------------------------------------


@dataclass
class Entry:
    value: str
    quote: str
    concern: bool = False
    detail: str = ""
    at: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)  # superseded values


@dataclass
class Visit:
    client: str
    caregiver: str
    identified_by: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.time)
    status: str = "in_progress"  # in_progress | filed | incomplete
    meds: dict[str, dict] = field(default_factory=dict)
    sections: dict[str, dict] = field(default_factory=dict)
    flags: list[dict] = field(default_factory=list)
    filed_at: float | None = None

    # ---- recording -------------------------------------------------------
    def _set(self, bucket: dict, key: str, entry: Entry) -> bool:
        prev = bucket.get(key)
        corrected = False
        if prev:
            hist = [
                *prev.get("history", []),
                {"value": prev["value"], "quote": prev["quote"], "at": prev["at"]},
            ]
            entry.history = hist
            corrected = prev["value"] != entry.value
        bucket[key] = asdict(entry)
        return corrected

    def record_med(
        self, med_id: str, status: str, quote: str, detail: str = ""
    ) -> bool:
        med = next(m for m in CLIENTS[self.client]["meds"] if m["id"] == med_id)
        concern = status in {"refused", "missed"} and med["critical"]
        corrected = self._set(self.meds, med_id, Entry(status, quote, concern, detail))
        self._reflag()
        return corrected

    def record_section(
        self, section: str, value: str, quote: str, concern: bool, detail: str = ""
    ) -> bool:
        corrected = self._set(
            self.sections, section, Entry(value, quote, concern, detail)
        )
        self._reflag()
        return corrected

    def add_flag(self, kind: str, level: str, text: str, quote: str = "") -> None:
        if not any(f["kind"] == kind and f["text"] == text for f in self.flags):
            self.flags.append(
                {
                    "kind": kind,
                    "level": level,
                    "text": text,
                    "quote": quote,
                    "at": time.time(),
                }
            )

    def _reflag(self) -> None:
        """Recompute data-driven flags from current values (corrections can clear them)."""
        manual = [
            f
            for f in self.flags
            if f["kind"]
            in {
                "emergency",
                "safeguarding",
                "integrity",
                "unlisted_med",
                "advice_request",
            }
        ]
        self.flags = manual
        name = {m["id"]: m["name"] for m in CLIENTS[self.client]["meds"]}
        for mid, e in self.meds.items():
            if e["concern"]:
                self.add_flag("med", "review", f"{name[mid]} {e['value']}", e["quote"])
        mob = self.sections.get("mobility")
        if mob and mob["value"] == "fall":
            self.add_flag(
                "fall",
                "urgent",
                "Fall reported" + (f": {mob['detail']}" if mob["detail"] else ""),
                mob["quote"],
            )
        elif mob and mob["value"] == "near_fall":
            self.add_flag(
                "fall",
                "review",
                "Near fall / unsteady"
                + (f": {mob['detail']}" if mob["detail"] else ""),
                mob["quote"],
            )
        for sec in ("food", "mood", "skin", "other"):
            e = self.sections.get(sec)
            if e and e["concern"]:
                self.add_flag(
                    sec, "review", f"{sec.capitalize()}: {e['value']}", e["quote"]
                )

    # ---- progress --------------------------------------------------------
    def missing(self) -> list[str]:
        out = [
            f"med:{m['id']}"
            for m in CLIENTS[self.client]["meds"]
            if m["id"] not in self.meds
        ]
        out += [s for s in SECTIONS[1:] if s not in self.sections]
        return out

    def next_question(self) -> str:
        first = CLIENTS[self.client]["name"].split()[0]
        miss = self.missing()
        if not miss:
            return "Everything is covered. Read the note back and ask if it is okay to file."
        m = miss[0]
        if m.startswith("med:"):
            med = next(x for x in CLIENTS[self.client]["meds"] if x["id"] == m[4:])
            return f"Ask about {med['name']} ({med['purpose']}, due {med['due']}): did {first} take it?"
        return "Ask: " + SECTION_QUESTIONS[m].format(first=first)

    def level(self) -> str:
        return max(
            (f["level"] for f in self.flags), key=LEVELS.index, default="routine"
        )

    def readback(self) -> str:
        name = {m["id"]: m["name"] for m in CLIENTS[self.client]["meds"]}
        meds = ", ".join(
            f"{name[k]} {v['value'].replace('_', ' ')}" for k, v in self.meds.items()
        )
        secs = "; ".join(f"{k} {v['value']}" for k, v in self.sections.items())
        return f"Meds: {meds}. {secs}."

    def to_dict(self) -> dict:
        d = asdict(self)
        d["client_name"] = CLIENTS[self.client]["name"]
        d["caregiver_name"] = CAREGIVERS[self.caregiver]["name"]
        d["level"] = self.level()
        d["missing"] = self.missing()
        d["med_names"] = {m["id"]: m["name"] for m in CLIENTS[self.client]["meds"]}
        return d


# --------------------------------------------------------------------------
# Store + patterns across visits
# --------------------------------------------------------------------------

STORE_PATH = Path(
    os.getenv(
        "BATON_STORE",
        Path(__file__).resolve().parent.parent / "data" / "baton_store.json",
    )
)


def _seed() -> list[dict]:
    now = datetime.now()

    def past(days: int, client: str, meds: dict, sections: dict) -> dict:
        t = (now - timedelta(days=days)).replace(hour=14, minute=0).timestamp()

        def mk(v, c=False):
            return {
                "value": v,
                "quote": "(earlier visit)",
                "concern": c,
                "detail": "",
                "at": t,
                "history": [],
            }

        return {
            "client": client,
            "caregiver": "maria",
            "identified_by": "seed",
            "id": uuid.uuid4().hex[:8],
            "started": t,
            "status": "filed",
            "filed_at": t,
            "flags": [],
            "meds": {k: mk(v, v in {"refused", "missed"}) for k, v in meds.items()},
            "sections": {k: mk(v[0], v[1]) for k, v in sections.items()},
        }

    ok = {
        "mood": ("usual", False),
        "skin": ("no issues", False),
        "other": ("nothing new", False),
    }
    return [
        past(
            2,
            "rosa",
            {"lisinopril": "refused", "metformin": "given", "donepezil": "not_due"},
            {"food": ("ate about half", False), "mobility": ("steady", False), **ok},
        ),
        past(
            1,
            "rosa",
            {"lisinopril": "refused", "metformin": "given", "donepezil": "not_due"},
            {"food": ("ate a little", True), "mobility": ("steady", False), **ok},
        ),
        past(
            3,
            "james",
            {"warfarin": "given", "furosemide": "given"},
            {"food": ("ate well", False), "mobility": ("near_fall", False), **ok},
        ),
    ]


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.visits: list[dict] = json.loads(self.path.read_text())
        else:
            self.visits = _seed()
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.visits, indent=1))
        tmp.replace(self.path)  # atomic

    def upsert(self, visit: Visit) -> None:
        d = asdict(visit)
        self.visits = [v for v in self.visits if v["id"] != visit.id] + [d]
        self.save()

    def open_visit(self, client: str, caregiver: str) -> dict | None:
        """An unfinished note from earlier today (dropped call) to resume."""
        today = datetime.now().date()
        for v in reversed(self.visits):
            if (
                v["client"] == client
                and v["caregiver"] == caregiver
                and v["status"] in {"in_progress", "incomplete"}
                and datetime.fromtimestamp(v["started"]).date() == today
            ):
                return v
        return None

    def history(self, client: str, exclude: str | None = None) -> list[dict]:
        hs = [
            v
            for v in self.visits
            if v["client"] == client and v["id"] != exclude and v["status"] == "filed"
        ]
        return sorted(hs, key=lambda v: v["started"])


def patterns(store: Store, visit: Visit) -> list[dict]:
    """Rules across visits. Code, not the LLM, decides what counts as a pattern."""
    hist = [*store.history(visit.client, exclude=visit.id), asdict(visit)]
    out: list[dict] = []
    names = {m["id"]: m["name"] for m in CLIENTS[visit.client]["meds"]}
    for mid, name in names.items():
        streak = 0
        for v in reversed(hist):
            st = v["meds"].get(mid, {}).get("value")
            if st in {"refused", "missed"}:
                streak += 1
            else:
                break
        if streak >= 3:
            out.append(
                {
                    "kind": "pattern",
                    "level": "urgent",
                    "text": f"{name} refused or missed {streak} visits in a row. Nurse review needed.",
                }
            )
    week_ago = time.time() - 7 * 86400
    falls = sum(
        1
        for v in hist
        if v["started"] >= week_ago
        and v["sections"].get("mobility", {}).get("value") in {"fall", "near_fall"}
    )
    if falls >= 2:
        out.append(
            {
                "kind": "pattern",
                "level": "urgent",
                "text": f"{falls} falls or near falls in 7 days. Fall-risk review needed.",
            }
        )
    recent_food = [v["sections"].get("food", {}).get("concern") for v in hist[-3:]]
    if len(recent_food) == 3 and all(recent_food):
        out.append(
            {
                "kind": "pattern",
                "level": "review",
                "text": "Poor eating 3 visits in a row.",
            }
        )
    return out
