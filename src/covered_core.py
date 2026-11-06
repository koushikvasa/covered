"""Covered: the parts of a health-plan care line that must never be wrong.

The LLM understands the caller and phrases replies. This module decides:
  - identity: caller ID is a hint; date of birth proves it; reps must be on file
  - plan rules: referrals, prior authorization, what is or isn't in the data
  - search: in-network, near the ZIP, accepting patients, with a REAL open slot
    (ghost listings are skipped and logged, never offered)
  - cost: the member's own estimate from deductible, copay and coinsurance
  - booking: only an offered slot, only once, only after confirmation

No LiveKit imports; tested in tests/test_covered_core.py. All data is synthetic.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from baton_core import detect_emergency  # shared, tested emergency detector
from covered_directory import background_words, fill_gaps

__all__ = ["background_words", "detect_emergency"]

# Close enough to be the same name misheard, without matching a relative's name.
NAME_MATCH_THRESHOLD = 0.8

# ----------------------------------------------------------------- reference
ZIPS = {
    "07302": (40.7196, -74.0434, "Downtown Jersey City"),
    "07306": (40.7323, -74.0659, "Journal Square"),
    "07030": (40.7445, -74.0324, "Hoboken"),
    "10001": (40.7506, -73.9972, "Chelsea, Manhattan"),
    "10027": (40.8116, -73.9533, "Harlem"),
    "11201": (40.6955, -73.9893, "Brooklyn Heights"),
}
PLACE_TO_ZIP = {
    "jersey city": "07302",
    "journal square": "07306",
    "hoboken": "07030",
    "manhattan": "10001",
    "chelsea": "10001",
    "harlem": "10027",
    "brooklyn": "11201",
}

SPECIALTIES = {
    "primary_care": "primary care doctor",
    "mental_health": "therapist",
    "orthopedics": "orthopedic doctor",
    "dermatology": "dermatologist",
    "cardiology": "heart doctor",
    "ob_gyn": "OB-GYN",
    "neurology": "neurologist",
    "ent": "ear, nose and throat doctor",
    "ophthalmology": "eye doctor",
    "gastroenterology": "digestive health doctor",
    "endocrinology": "diabetes and hormone doctor",
    "allergy": "allergy doctor",
    "pediatrics": "pediatrician",
    "physical_therapy": "physical therapist",
    "urology": "urologist",
    "podiatry": "foot doctor",
}
# Specialists an HMO member needs a referral for.
REFERRAL_SPECIALTIES = {
    "orthopedics",
    "dermatology",
    "cardiology",
    "neurology",
    "ent",
    "ophthalmology",
    "gastroenterology",
    "endocrinology",
    "allergy",
    "physical_therapy",
    "urology",
    "podiatry",
}

PLANS = {
    "silver_hmo": {
        "name": "Covered Silver HMO",
        "needs_referral": REFERRAL_SPECIALTIES,
        "copay": {
            "primary_care": 25,
            "mental_health": 30,
            "specialist": 60,
            "ob_gyn": 25,
            "pediatrics": 25,
            "physical_therapy": 40,
        },
        "coinsurance": 0.2,
        "services": {
            "therapy": "Covered with a thirty dollar copay at in-network therapists. No referral needed.",
            "mri": "Covered after prior authorization. Your doctor submits the request.",
            "physical therapy": "Covered, up to twenty visits a year, with a referral.",
            "annual physical": "Covered at no cost once a year with your primary care doctor.",
            "flu shot": "Covered at no cost.",
        },
    },
    "gold_ppo": {
        "name": "Covered Gold PPO",
        "needs_referral": set(),
        "copay": {
            "primary_care": 20,
            "mental_health": 20,
            "specialist": 40,
            "ob_gyn": 20,
            "pediatrics": 20,
            "physical_therapy": 30,
        },
        "coinsurance": 0.1,
        "services": {
            "therapy": "Covered with a twenty dollar copay at in-network therapists.",
            "mri": "Covered after prior authorization. Your doctor submits the request.",
            "annual physical": "Covered at no cost once a year.",
            "flu shot": "Covered at no cost.",
        },
    },
}

MEMBERS = {
    "alex": {
        "full_name": "Alex Johnson",
        "preferred_name": "Alex",
        "member_id": "MBR-12345",
        "dob": "1990-04-12",
        "phone": "+12015550111",
        "zip": "07302",
        "plan": "silver_hmo",
        "pcp": "dr-lee",
        "deductible": 2000,
        "deductible_met": 1650,
        "authorized_reps": [],
        "last_seen": {"mental_health": "dr-patel"},
    },
    "elena": {
        "full_name": "Elena Garcia",
        "preferred_name": "Elena",
        "member_id": "MBR-67890",
        "dob": "1948-09-03",
        "phone": "+12015550144",
        "zip": "07306",
        "plan": "gold_ppo",
        "pcp": "dr-lee",
        "deductible": 1000,
        "deductible_met": 1000,
        "authorized_reps": [{"name": "Maria Garcia", "phone": "+12015550122"}],
        "last_seen": {},
    },
}

# Demo: map your own phone to a member, e.g. COVERED_DEMO_PHONES=+1646XXXXXXX=alex
for _pair in filter(None, os.getenv("COVERED_DEMO_PHONES", "").split(",")):
    _num, _, _mid = _pair.partition("=")
    if _mid.strip() in MEMBERS:
        MEMBERS[_mid.strip()].setdefault("extra_phones", []).append(_num.strip())

# tier 1 = copay only; tier 2 = deductible then coinsurance on the allowed amount
# ghost = what a real secret-shopper call would find
HAND_PROVIDERS = [
    {
        "id": "dr-lee",
        "name": "Dr. Lena Lee",
        "specialty": "primary_care",
        "zip": "07302",
        "address": "120 Grand Street, Jersey City",
        "tier": 1,
        "allowed": 160,
        "langs": ["English", "Korean"],
        "times": ["9:00 AM", "1:30 PM"],
    },
    {
        "id": "dr-okafor",
        "name": "Dr. Ada Okafor",
        "specialty": "primary_care",
        "zip": "07306",
        "address": "30 Journal Square Plaza, Jersey City",
        "tier": 2,
        "allowed": 180,
        "langs": ["English"],
        "times": ["10:00 AM", "3:00 PM"],
    },
    {
        "id": "dr-patel",
        "name": "Dr. Priya Patel",
        "specialty": "mental_health",
        "zip": "07302",
        "address": "75 Montgomery Street, Jersey City",
        "tier": 1,
        "allowed": 150,
        "langs": ["English", "Hindi"],
        "times": ["10:00 AM", "4:00 PM"],
    },
    {
        "id": "dr-kim",
        "name": "Dr. Daniel Kim",
        "specialty": "mental_health",
        "zip": "07030",
        "address": "200 Washington Street, Hoboken",
        "tier": 2,
        "allowed": 220,
        "langs": ["English", "Korean"],
        "times": ["11:00 AM", "6:00 PM"],
    },
    {
        "id": "dr-hart",
        "name": "Dr. Owen Hart",
        "specialty": "mental_health",
        "zip": "07302",
        "address": "15 Exchange Place, Jersey City",
        "tier": 1,
        "allowed": 150,
        "langs": ["English"],
        "times": [],
        "ghost": "phone number disconnected",
    },
    {
        "id": "bayview",
        "name": "Bayview Counseling",
        "specialty": "mental_health",
        "zip": "07306",
        "address": "88 Sip Avenue, Jersey City",
        "tier": 1,
        "allowed": 140,
        "langs": ["English", "Spanish"],
        "times": [],
        "ghost": "not accepting new patients",
    },
    {
        "id": "dr-reyes",
        "name": "Dr. Sofia Reyes",
        "specialty": "mental_health",
        "zip": "11201",
        "address": "45 Court Street, Brooklyn",
        "tier": 1,
        "allowed": 150,
        "langs": ["English", "Spanish"],
        "times": ["9:30 AM", "2:30 PM"],
    },
    {
        "id": "dr-brennan",
        "name": "Dr. Liam Brennan",
        "specialty": "orthopedics",
        "zip": "07302",
        "address": "50 Newark Avenue, Jersey City",
        "tier": 1,
        "allowed": 260,
        "langs": ["English"],
        "times": ["8:30 AM", "2:00 PM"],
    },
    {
        "id": "hudson-ortho",
        "name": "Hudson Orthopedic Group",
        "specialty": "orthopedics",
        "zip": "07030",
        "address": "1 Hudson Place, Hoboken",
        "tier": 2,
        "allowed": 250,
        "langs": ["English", "Spanish"],
        "times": ["9:00 AM", "12:00 PM"],
    },
    {
        "id": "dr-stone",
        "name": "Dr. Maya Stone",
        "specialty": "orthopedics",
        "zip": "07306",
        "address": "2 Journal Square, Jersey City",
        "tier": 1,
        "allowed": 260,
        "langs": ["English"],
        "times": [],
        "ghost": "moved, no longer at this address",
    },
    {
        "id": "dr-nguyen",
        "name": "Dr. Anh Nguyen",
        "specialty": "dermatology",
        "zip": "07302",
        "address": "101 Hudson Street, Jersey City",
        "tier": 1,
        "allowed": 200,
        "langs": ["English", "Vietnamese"],
        "times": ["11:30 AM"],
    },
    {
        "id": "dr-adams",
        "name": "Dr. Ruth Adams",
        "specialty": "cardiology",
        "zip": "10001",
        "address": "350 West 23rd Street, New York",
        "tier": 1,
        "allowed": 300,
        "langs": ["English"],
        "times": ["10:30 AM"],
    },
    {
        "id": "dr-silva",
        "name": "Dr. Ines Silva",
        "specialty": "ob_gyn",
        "zip": "07030",
        "address": "77 River Street, Hoboken",
        "tier": 1,
        "allowed": 180,
        "langs": ["English", "Portuguese", "Spanish"],
        "times": ["9:00 AM", "1:00 PM"],
    },
]
# Hand-written providers' backgrounds and practice fees (all invented).
_HAND_DETAILS = {
    "dr-lee": (0, 18, "Eastmere Medical School", "preventive care"),
    "dr-okafor": (0, 9, "Lakeshire School of Medicine", "family medicine"),
    "dr-patel": (0, 14, "Stonebridge University", "anxiety and stress"),
    "dr-kim": (0, 22, "Westhollow College of Medicine", "depression"),
    "dr-reyes": (10, 7, "Riverton Medical College", "life transitions"),
    "dr-brennan": (0, 25, "Ashgrove University", "joint pain"),
    "dr-nguyen": (15, 11, "Eastmere Medical School", "skin checks"),
    "dr-adams": (25, 30, "Stonebridge University", "heart rhythm"),
    "dr-silva": (5, 16, "Lakeshire School of Medicine", "prenatal care"),
}
for _p in HAND_PROVIDERS:
    if _p["id"] in _HAND_DETAILS:
        _fee, _years, _school, _focus = _HAND_DETAILS[_p["id"]]
        _p["practice_fee"] = _fee
        _p["background"] = {
            "years": _years,
            "training": _school,
            "board_certified": True,
            "focus": _focus,
        }
    elif not _p.get("ghost"):  # Hudson Orthopedic Group
        _p["background"] = {"group": True, "focus": "sports injuries and joint pain"}

PROVIDERS = HAND_PROVIDERS + fill_gaps(HAND_PROVIDERS, SPECIALTIES, ZIPS)
PROVIDER = {p["id"]: p for p in PROVIDERS}
MAX_MILES = 8.0
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# ------------------------------------------------------------------- speech
CRISIS_PHRASES = [
    "kill myself",
    "end my life",
    "suicidal",
    "want to die",
    "hurt myself",
    "don't want to be alive",
    "dont want to be alive",
    "no reason to live",
]
INTEGRITY_PHRASES = [
    "my husband's insurance",
    "my wife's insurance",
    "his insurance",
    "her insurance card",
    "my sister's insurance",
    "my brother's insurance",
    "someone else's insurance",
    "say it's an emergency",
    "say it was an emergency",
    "pretend it's",
    "pretend i'm",
    "say i'm him",
    "say i'm her",
    "lie about",
    "fake",
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower()).strip()


def detect_crisis(text: str) -> str | None:
    t = _norm(text)
    return next((p for p in CRISIS_PHRASES if p in t), None)


def detect_integrity(text: str) -> str | None:
    t = _norm(text)
    return next((p for p in INTEGRITY_PHRASES if p in t), None)


# ----------------------------------------------------------------- identity
def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")[-10:]


def member_by_phone(phone: str | None) -> str | None:
    d = _digits(phone or "")
    if not d:
        return None
    for mid, m in MEMBERS.items():
        if d in {_digits(m["phone"]), *(_digits(p) for p in m.get("extra_phones", []))}:
            return mid
    return None


def rep_by_phone(phone: str | None) -> tuple[str, str] | None:
    """(member_id, rep_name) when the number belongs to an authorized representative."""
    d = _digits(phone or "")
    for mid, m in MEMBERS.items():
        for r in m["authorized_reps"]:
            if d and d == _digits(r["phone"]):
                return mid, r["name"]
    return None


def member_by_name(name: str) -> str | None:
    n = _norm(name)
    for mid, m in MEMBERS.items():
        full = _norm(m["full_name"])
        if n and (n == full or n in full.split() or full in n):
            return mid
    return None


def member_by_id(insurance_id: str) -> str | None:
    """Accepts 'MBR-12345', 'mbr 12345', 'M B R 1 2 3 4 5' or just the digits."""
    spoken = re.sub(r"[^A-Z0-9]", "", (insurance_id or "").upper())
    if not spoken:
        return None
    for mid, m in MEMBERS.items():
        on_file = re.sub(r"[^A-Z0-9]", "", m["member_id"])
        if spoken == on_file or spoken == re.sub(r"\D", "", on_file):
            return mid
    return None


def name_matches(member_id: str, name: str) -> bool:
    """Is this caller the member? Tolerant of speech-to-text mishearing a name."""
    return _same_person(name, MEMBERS[member_id]["full_name"])


def _same_person(spoken: str, on_file: str) -> bool:
    """Full name required: with no date of birth, it is half of the proof."""
    n, full = _norm(spoken), _norm(on_file)
    if not n:
        return False
    if full in n:
        return True
    return SequenceMatcher(None, n, full).ratio() >= NAME_MATCH_THRESHOLD


def normalize_dob(text: str) -> str | None:
    """Accepts '1990-04-12', '04/12/1990', 'April 12th 1990', '12 April 1990'."""
    t = (text or "").strip().replace(",", " ")
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t, flags=re.IGNORECASE)
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(" ".join(t.split()), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def is_authorized_rep(
    member_id: str, caller_name: str, caller_phone: str | None
) -> bool:
    for r in MEMBERS[member_id]["authorized_reps"]:
        phone_ok = bool(caller_phone) and _digits(caller_phone) == _digits(r["phone"])
        if _same_person(caller_name, r["name"]) or phone_ok:
            return True
    return False


# ---------------------------------------------------------------- plan rules
def plan(member_id: str) -> dict:
    return PLANS[MEMBERS[member_id]["plan"]]


def needs_referral(member_id: str, specialty: str) -> bool:
    return specialty in plan(member_id)["needs_referral"]


def coverage_answer(member_id: str, service: str) -> str | None:
    """Only answers from plan data. None means 'not in the data, don't guess'."""
    s = _norm(service)
    for key, ans in plan(member_id)["services"].items():
        if key in s or s in key:
            return ans
    return None


def copay_estimate(member_id: str, specialty: str) -> int:
    copays = plan(member_id)["copay"]
    return int(copays[specialty if specialty in copays else "specialist"])


def estimate_cost(member_id: str, provider_id: str) -> int:
    m, p = MEMBERS[member_id], PROVIDER[provider_id]
    pl = plan(member_id)
    if p["tier"] == 1:
        return copay_estimate(member_id, p["specialty"]) + p.get("practice_fee", 0)
    remaining = max(0, m["deductible"] - m["deductible_met"])
    toward_deductible = min(remaining, p["allowed"])
    return round(
        toward_deductible + pl["coinsurance"] * (p["allowed"] - toward_deductible)
    )


# -------------------------------------------------------------------- search
def resolve_zip(text: str | None, default: str) -> str | None:
    if not text:
        return default
    digits = re.sub(r"\D", "", text)
    if len(digits) == 5:
        return digits  # ZIPs outside the demo network go to the live registry
    t = _norm(text)
    return next((z for place, z in PLACE_TO_ZIP.items() if place in t), None)


def miles(z1: str, z2: str) -> float:
    (a1, o1, _), (a2, o2, _) = ZIPS[z1], ZIPS[z2]
    x = math.radians(o2 - o1) * math.cos(math.radians((a1 + a2) / 2))
    y = math.radians(a2 - a1)
    return round(3959 * math.hypot(x, y), 1)


def upcoming_slots(
    provider_id: str, taken: set[str], start: date | None = None
) -> list[str]:
    """ISO 'YYYY-MM-DD HH:MM AM' slots over the next 7 weekdays, minus booked ones."""
    p = PROVIDER[provider_id]
    out, day = [], start or date.today()
    while len(out) < 7 * len(p["times"]) and p["times"]:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
        # stagger: providers aren't open every day
        if (day.toordinal() + sum(map(ord, provider_id))) % 3 == 0:
            continue
        out += [f"{day.isoformat()} {t}" for t in p["times"]]
        if len(out) > 12:
            break
    return [s for s in out if f"{provider_id}|{s}" not in taken]


def slot_words(slot: str) -> str:
    d, t = slot.split(" ", 1)
    dt = datetime.fromisoformat(d)  # no %-d: it breaks on Windows
    return f"{dt.strftime('%A %B')} {dt.day} at {t}"


def search(
    member_id: str,
    specialty: str,
    zip_code: str,
    taken: set[str],
    include_days: list[str] | None = None,
    exclude_days: list[str] | None = None,
    time_of_day: str = "any",
    prefer_provider: str | None = None,
) -> dict:
    """In-network, in this ZIP (or nearby if nothing fits), really bookable.
    Returns offers (max 2, cheapest first) and skipped ghosts."""
    inc = {d.lower() for d in include_days or [] if d.lower() in DAYS}
    exc = {d.lower() for d in exclude_days or [] if d.lower() in DAYS}
    offers, ghosts, no_match = [], [], []
    for p in PROVIDERS:
        if p["specialty"] != specialty:
            continue
        dist = miles(zip_code, p["zip"])
        if dist > MAX_MILES:
            continue
        if p.get("ghost"):
            ghosts.append(
                {"id": p["id"], "name": p["name"], "reason": p["ghost"], "miles": dist}
            )
            continue
        slots = []
        for s in upcoming_slots(p["id"], taken):
            wd = datetime.fromisoformat(s.split(" ")[0]).strftime("%A").lower()
            if (inc and wd not in inc) or wd in exc:
                continue
            hour_pm = s.endswith("PM") and not s.split(" ")[1].startswith("12")
            if (time_of_day == "morning" and hour_pm) or (
                time_of_day == "afternoon" and not hour_pm
            ):
                continue
            slots.append(s)
        if not slots:
            no_match.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "reason": "no open times that fit",
                    "miles": dist,
                }
            )
            continue
        offers.append(
            {
                "id": p["id"],
                "name": p["name"],
                "miles": dist,
                "address": p["address"],
                "slot": slots[0],
                "cost": estimate_cost(member_id, p["id"]),
                "tier": p["tier"],
                "background": background_words(p),
            }
        )
    # Doctors in the caller's own ZIP come first, so a new ZIP means new doctors.
    offers = [o for o in offers if o["miles"] == 0] or offers
    offers.sort(
        key=lambda o: (o["id"] != prefer_provider, o["cost"], o["slot"], o["miles"])
    )
    return {"offers": offers[:2], "ghosts": ghosts, "no_match": no_match}


# --------------------------------------------------------------------- store
STORE_PATH = Path(
    os.getenv(
        "COVERED_STORE",
        Path(__file__).resolve().parent.parent / "data" / "covered_store.json",
    )
)


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = Path(path)
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "taken": [],
                "bookings": [],
                "referrals": [],
                "intakes": [],
                "ghost_reports": [],
            }
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    @property
    def taken(self) -> set[str]:
        return set(self.data["taken"])

    def has_referral(self, member_id: str, specialty: str) -> bool:
        return any(
            r["member"] == member_id and r["specialty"] == specialty
            for r in self.data["referrals"]
        )

    def add_referral(self, member_id: str, specialty: str, reason: str) -> dict:
        r = {
            "id": "REF-" + uuid.uuid4().hex[:5].upper(),
            "member": member_id,
            "specialty": specialty,
            "to_pcp": MEMBERS[member_id]["pcp"],
            "reason": reason,
            "status": "requested",
            "at": time.time(),
        }
        self.data["referrals"].append(r)
        self.save()
        return r

    def book(
        self, member_id: str, provider_id: str, slot: str, referral: str | None
    ) -> dict | None:
        key = f"{provider_id}|{slot}"
        if key in self.data["taken"]:
            return None  # someone else got it first
        self.data["taken"].append(key)
        b = {
            "confirmation": "CV-" + uuid.uuid4().hex[:6].upper(),
            "member": member_id,
            "provider": provider_id,
            "slot": slot,
            "cost": estimate_cost(member_id, provider_id),
            "referral": referral,
            "at": time.time(),
        }
        self.data["bookings"].append(b)
        self.save()
        return b

    def add_request(
        self, member_id: str, provider_id: str, cost: int, referral: str | None
    ) -> dict:
        """An appointment request to a live provider: no time is claimed or held."""
        requests = self.data.setdefault("requests", [])
        for existing in requests:
            if (existing["member"], existing["provider"]) == (member_id, provider_id):
                return existing
        r = {
            "confirmation": "RQ-" + uuid.uuid4().hex[:6].upper(),
            "member": member_id,
            "provider": provider_id,
            "slot": None,
            "cost": cost,
            "referral": referral,
            "at": time.time(),
        }
        requests.append(r)
        self.save()
        return r

    def report_ghosts(self, ghosts: list[dict]) -> None:
        known = {g["id"] for g in self.data["ghost_reports"]}
        self.data["ghost_reports"] += [
            {**g, "at": time.time()} for g in ghosts if g["id"] not in known
        ]
        self.save()

    def add_intake(self, record: dict) -> None:
        self.data["intakes"] = [
            r for r in self.data["intakes"] if r["call_id"] != record["call_id"]
        ] + [record]
        self.save()
