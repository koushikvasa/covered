"""Parsing of NPI Registry responses. All records here are synthetic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import live_providers as live

ZIP = "22030"


def record(number, first, last, credential, location_zip=ZIP, status="A"):
    return {
        "number": number,
        "basic": {
            "first_name": first,
            "last_name": last,
            "credential": credential,
            "status": status,
            "enumeration_date": "2012-03-04",
        },
        "taxonomies": [
            {"desc": "Secondary Thing", "primary": False},
            {"desc": "Cardiovascular Disease", "primary": True},
        ],
        "addresses": [
            {
                "address_purpose": "MAILING",
                "address_1": "PO BOX 1",
                "city": "ELSEWHERE",
                "state": "MD",
                "postal_code": "20001",
                "telephone_number": "555-0199",
            },
            {
                "address_purpose": "LOCATION",
                "address_1": "1 SAMPLE STREET",
                "city": "FAIRFAX",
                "state": "VA",
                "postal_code": f"{location_zip}1234",
                "telephone_number": "555-0100",
            },
        ],
    }


def test_parse_uses_the_practice_location_not_the_mailing_address():
    found = live.parse_results(
        {"results": [record(1, "TEST", "PROVIDER", "M.D.")]}, ZIP, "cardiology"
    )
    assert found == [
        {
            "id": "npi-1",
            "name": "Dr. Test Provider",
            "address": "1 Sample Street, Fairfax, VA",
            "phone": "555-0100",
            "specialty": "cardiology",
            "background": "Cardiovascular Disease, M.D., in the public registry since 2012",
        }
    ]


def test_background_copes_with_missing_registry_fields():
    bare = record(1, "TEST", "PROVIDER", None)
    del bare["taxonomies"]
    del bare["basic"]["enumeration_date"]
    found = live.parse_results({"results": [bare]}, ZIP, "cardiology")
    assert found[0]["background"] == "Listed in the public registry"


@pytest.mark.parametrize(
    ("credential", "expected"),
    [
        ("M.D.", "Dr. Casey Example"),
        ("DO", "Dr. Casey Example"),
        ("Ph.D.", "Dr. Casey Example"),
        ("PsyD", "Dr. Casey Example"),
        ("LCSW", "Casey Example"),
        ("", "Casey Example"),
        (None, "Casey Example"),
    ],
)
def test_only_doctoral_credentials_get_the_title(credential, expected):
    found = live.parse_results(
        {"results": [record(1, "CASEY", "EXAMPLE", credential)]}, ZIP, "mental_health"
    )
    assert found[0]["name"] == expected


def test_parse_drops_other_zips_inactive_and_nameless_records():
    data = {
        "results": [
            record(1, "ELSE", "WHERE", "MD", location_zip="10001"),
            record(2, "NOT", "ACTIVE", "MD", status="D"),
            record(3, "", "", "MD"),
            record(4, "KEEP", "ME", "MD"),
        ]
    }
    assert [p["id"] for p in live.parse_results(data, ZIP, "cardiology")] == ["npi-4"]


def test_parse_handles_empty_and_error_responses():
    assert live.parse_results({}, ZIP, "cardiology") == []
    assert live.parse_results({"Errors": [{"description": "bad"}]}, ZIP, "x") == []


def test_every_specialty_has_a_registry_taxonomy():
    import covered_core as cc

    assert set(live.TAXONOMY) == set(cc.SPECIALTIES)


async def test_search_turns_any_network_problem_into_lookup_failed(monkeypatch):
    async def timeout(params):
        raise TimeoutError

    monkeypatch.setattr(live, "_fetch", timeout)
    with pytest.raises(live.ProviderLookupError):
        await live.search_nearby("cardiology", ZIP)


async def test_search_asks_for_individuals_with_the_right_taxonomy(monkeypatch):
    seen = {}

    async def capture(params):
        seen.update(params)
        return {"results": [record(1, "TEST", "PROVIDER", "MD")]}

    monkeypatch.setattr(live, "_fetch", capture)
    found = await live.search_nearby("orthopedics", ZIP)

    assert seen["postal_code"] == ZIP
    assert seen["taxonomy_description"] == "Orthopaedic Surgery"
    assert seen["enumeration_type"] == "NPI-1"
    assert found[0]["specialty"] == "orthopedics"
