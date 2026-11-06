"""Synthetic doctors, so every specialty has two bookable options in every demo ZIP.

The hand-written providers in covered_core.py stay as they are, ghost listings included;
this only fills the gaps. Every value comes from a hash of the doctor's slot, so the
directory is identical on every run. All names, schools and backgrounds are invented.
"""

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

DOCTORS_PER_ZIP = 2
ROUND_ALLOWED_TO = 5

FIRST_NAMES = (
    "Amara", "Ben", "Carmen", "Dev", "Esther", "Farah", "Gabriel", "Hana", "Ivan", "Joy",
    "Kofi", "Lucia", "Marcus", "Nadia", "Omar", "Paula", "Quinn", "Rosa", "Samir", "Tessa",
    "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zane", "Aiko", "Bruno", "Celia", "Dmitri",
)  # fmt: skip
LAST_NAMES = (
    "Abara", "Bianchi", "Castellano", "Dhillon", "Ekstrom", "Fontaine", "Gallagher", "Haddad",
    "Ibarra", "Jansen", "Kowalski", "Lindqvist", "Morales", "Nakamura", "Oyelaran", "Petrov",
    "Quintero", "Rahman", "Sorensen", "Tran", "Ueda", "Vasquez", "Whitfield", "Xiong",
    "Yilmaz", "Zielinski", "Adeyemi", "Bergstrom", "Chaudhry", "Delacroix",
)  # fmt: skip
# Invented on purpose: a real school would imply a real person's record.
SCHOOLS = (
    "Eastmere Medical School",
    "Lakeshire School of Medicine",
    "Stonebridge University",
    "Westhollow College of Medicine",
    "Riverton Medical College",
    "Ashgrove University",
)
EXTRA_LANGUAGES = (
    None,
    None,
    "Spanish",
    "Mandarin",
    "Hindi",
    "Korean",
    "Arabic",
    "Portuguese",
)
TIME_SETS = (
    ("9:00 AM", "2:00 PM"),
    ("8:30 AM", "1:30 PM"),
    ("10:30 AM", "4:30 PM"),
    ("11:00 AM", "3:15 PM"),
    ("9:45 AM", "5:30 PM"),
)
# Tier 1 visits cost the plan copay plus this practice fee.
PRACTICE_FEES = (0, 0, 5, 10, 15, 20, 25)
# Tier 2 visits are priced from the allowed amount, which varies by doctor.
ALLOWED_SPREAD = (0.85, 1.0, 1.15, 1.3)
TIER_2_ODDS = (True, False, False)
YEARS_IN_PRACTICE = range(4, 34)
BOARD_CERTIFIED_ODDS = (True, True, True, False)

AREAS = {
    "07302": (
        "Jersey City",
        ("Grove Street", "Newark Avenue", "Christopher Columbus Drive"),
    ),
    "07306": ("Jersey City", ("Kennedy Boulevard", "Sip Avenue", "Summit Avenue")),
    "07030": ("Hoboken", ("Washington Street", "Hudson Street", "Willow Avenue")),
    "10001": ("New York", ("West 28th Street", "Seventh Avenue", "West 34th Street")),
    "10027": ("New York", ("West 125th Street", "Amsterdam Avenue", "Broadway")),
    "11201": ("Brooklyn", ("Atlantic Avenue", "Montague Street", "Henry Street")),
}
SPECIALTY_DETAILS: dict[str, tuple[int, tuple[str, str, str]]] = {
    # specialty: (typical allowed amount, what a doctor might focus on)
    "primary_care": (
        170,
        ("preventive care", "long-term conditions", "family medicine"),
    ),
    "mental_health": (180, ("anxiety and stress", "depression", "life transitions")),
    "orthopedics": (260, ("sports injuries", "joint pain", "spine care")),
    "dermatology": (210, ("skin checks", "acne and rashes", "eczema")),
    "cardiology": (300, ("heart rhythm", "blood pressure", "preventive heart care")),
    "ob_gyn": (190, ("prenatal care", "women's health", "menopause care")),
    "neurology": (320, ("headaches and migraines", "memory concerns", "nerve pain")),
    "ent": (240, ("sinus problems", "hearing", "throat and voice")),
    "ophthalmology": (220, ("eye exams", "cataracts", "glaucoma")),
    "gastroenterology": (290, ("reflux and digestion", "colon health", "liver care")),
    "endocrinology": (270, ("diabetes", "thyroid conditions", "hormone health")),
    "allergy": (200, ("seasonal allergies", "asthma", "food allergies")),
    "pediatrics": (150, ("newborn care", "childhood illness", "teen health")),
    "physical_therapy": (130, ("back and neck pain", "injury recovery", "balance")),
    "urology": (250, ("kidney stones", "bladder health", "men's health")),
    "podiatry": (180, ("foot pain", "diabetic foot care", "ankle injuries")),
}


def _pick(seed: str, options: Sequence[Any], salt: str) -> Any:
    digest = hashlib.sha256(f"{seed}|{salt}".encode()).hexdigest()
    return options[int(digest, 16) % len(options)]


def make_provider(specialty: str, zip_code: str, index: int) -> dict:
    seed = f"{specialty}|{zip_code}|{index}"
    city, streets = AREAS[zip_code]
    typical_allowed, focuses = SPECIALTY_DETAILS[specialty]
    spread = _pick(seed, ALLOWED_SPREAD, "allowed")
    extra_language = _pick(seed, EXTRA_LANGUAGES, "language")
    street_number = _pick(seed, range(10, 400), "number")
    return {
        "id": f"{specialty}-{zip_code}-{index}",
        "name": (
            f"Dr. {_pick(seed, FIRST_NAMES, 'first')} {_pick(seed, LAST_NAMES, 'last')}"
        ),
        "specialty": specialty,
        "zip": zip_code,
        "address": f"{street_number} {_pick(seed, streets, 'street')}, {city}",
        "tier": 2 if _pick(seed, TIER_2_ODDS, "tier") else 1,
        "allowed": round(typical_allowed * spread / ROUND_ALLOWED_TO)
        * ROUND_ALLOWED_TO,
        "practice_fee": _pick(seed, PRACTICE_FEES, "fee"),
        "langs": ["English", extra_language] if extra_language else ["English"],
        "times": list(_pick(seed, TIME_SETS, "times")),
        "background": {
            "years": _pick(seed, YEARS_IN_PRACTICE, "years"),
            "training": _pick(seed, SCHOOLS, "school"),
            "board_certified": _pick(seed, BOARD_CERTIFIED_ODDS, "board"),
            "focus": _pick(seed, focuses, "focus"),
        },
    }


def fill_gaps(
    providers: list[dict], specialties: Iterable[str], zip_codes: Iterable[str]
) -> list[dict]:
    """Providers needed so each specialty has DOCTORS_PER_ZIP bookable doctors per ZIP."""
    zip_codes = list(zip_codes)
    added = []
    for specialty in specialties:
        for zip_code in zip_codes:
            bookable = sum(
                p["specialty"] == specialty
                and p["zip"] == zip_code
                and not p.get("ghost")
                for p in providers
            )
            added += [
                make_provider(specialty, zip_code, index)
                for index in range(bookable, DOCTORS_PER_ZIP)
            ]
    return added


def background_words(provider: dict) -> str:
    """Facts only, in words the agent can say. No ranking and no opinion."""
    background = provider["background"]
    if background.get("group"):
        intro = "a group practice"
    else:
        intro = f"{background['years']} years in practice, trained at {background['training']}"
    certified = ", board certified" if background.get("board_certified") else ""
    languages = ", ".join(provider["langs"])
    return f"{intro}{certified}, focuses on {background['focus']}, speaks {languages}"
