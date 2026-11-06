"""Live provider lookup: the public CMS NPI Registry, the same source ClearCare uses.

It lists real licensed providers for any US ZIP. It has no network status, open times
or prices, so these providers are only ever offered as an appointment REQUEST with no
invented time (see MyAgent._find_live). Records are fetched during a call and never
committed; tests use synthetic ones.
"""

import asyncio
import re

import aiohttp

NPI_URL = "https://npiregistry.cms.hhs.gov/api/"
TIMEOUT_SECONDS = 2.5  # a caller is waiting on the line
# Fetch more than we offer: many results practice outside the ZIP and get dropped.
FETCH_LIMIT = 20
ACTIVE_STATUS = "A"
DOCTORAL_CREDENTIALS = {"MD", "DO", "PHD", "PSYD", "DPM", "DNP"}
LOCATION_PURPOSE = "LOCATION"

# Our specialties mapped to NPI taxonomy descriptions.
TAXONOMY = {
    "primary_care": "Family Medicine",
    "mental_health": "Psychologist",
    "orthopedics": "Orthopaedic Surgery",
    "dermatology": "Dermatology",
    "cardiology": "Cardiovascular Disease",
    "ob_gyn": "Obstetrics & Gynecology",
    "neurology": "Neurology",
    "ent": "Otolaryngology",
    "ophthalmology": "Ophthalmology",
    "gastroenterology": "Gastroenterology",
    "endocrinology": "Endocrinology",
    "allergy": "Allergy & Immunology",
    "pediatrics": "Pediatrics",
    "physical_therapy": "Physical Therapist",
    "urology": "Urology",
    "podiatry": "Podiatrist",
}


class ProviderLookupError(Exception):
    """The registry could not be reached or returned something unreadable."""


def _display_name(basic: dict) -> str:
    first = (basic.get("first_name") or "").strip().title()
    last = (basic.get("last_name") or "").strip().title()
    if not first or not last:
        return ""
    letters = re.sub(r"[.\s]", "", basic.get("credential") or "").upper()
    is_doctor = bool(DOCTORAL_CREDENTIALS & set(re.split(r"[,;/]", letters)))
    return f"Dr. {first} {last}" if is_doctor else f"{first} {last}"


def _background(record: dict) -> str:
    """Only what the registry states; it says nothing about quality or experience."""
    basic = record.get("basic", {})
    primary = next(
        (t.get("desc") for t in record.get("taxonomies", []) if t.get("primary")), ""
    )
    year = (basic.get("enumeration_date") or "")[:4]
    facts = [fact for fact in (primary, basic.get("credential")) if fact]
    if year:
        facts.append(f"in the public registry since {year}")
    return ", ".join(facts) or "Listed in the public registry"


def _practice_location(record: dict, zip_code: str) -> dict | None:
    for address in record.get("addresses", []):
        in_zip = (address.get("postal_code") or "")[:5] == zip_code
        if address.get("address_purpose") == LOCATION_PURPOSE and in_zip:
            return address
    return None


def parse_results(data: dict, zip_code: str, specialty: str) -> list[dict]:
    """Active individuals who really practice in this ZIP, in a voice-friendly shape."""
    providers = []
    for record in data.get("results", []):
        basic = record.get("basic", {})
        name = _display_name(basic)
        location = _practice_location(record, zip_code)
        if not name or not location or basic.get("status") != ACTIVE_STATUS:
            continue
        street = (location.get("address_1") or "").title()
        city = (location.get("city") or "").title()
        providers.append(
            {
                "id": f"npi-{record.get('number')}",
                "name": name,
                "address": f"{street}, {city}, {location.get('state', '')}",
                "phone": location.get("telephone_number", ""),
                "specialty": specialty,
                "background": _background(record),
            }
        )
    return providers


async def _fetch(params: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(NPI_URL, params=params) as response,
    ):
        response.raise_for_status()
        return await response.json(content_type=None)


async def search_nearby(specialty: str, zip_code: str) -> list[dict]:
    """Providers near a ZIP. Raises ProviderLookupError so a call can recover in words."""
    params = {
        "version": "2.1",
        "limit": FETCH_LIMIT,
        "enumeration_type": "NPI-1",
        "taxonomy_description": TAXONOMY[specialty],
        "postal_code": zip_code,
    }
    try:
        data = await _fetch(params)
    except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError) as exc:
        raise ProviderLookupError(f"NPI lookup failed: {exc!r}") from exc
    return parse_results(data, zip_code, specialty)
