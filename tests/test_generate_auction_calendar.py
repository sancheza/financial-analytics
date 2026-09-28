#!/usr/bin/env python3
"""Unit tests for generate_auction_calendar.py.

Path: tests/test_generate_auction_calendar.py (relative to the repo root)

Covers auction type normalization, reissue/reopening pattern matching for
10Y/20Y/30Y/5Y notes/bonds and TIPS, unannounced upcoming auction merging,
and iCalendar (.ics) generation with synthetic data (no network access).

Usage:
    pytest tests/test_generate_auction_calendar.py -v
"""

import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

BOND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bond-analytics"
)
sys.path.insert(0, BOND_DIR)

import generate_auction_calendar as gac  # noqa: E402


def test_normalize_security_type_nominal():
    """Verify nominal note and bond types are formatted properly."""
    record = {
        "security_type": "Note",
        "security_term": "10-Year",
        "inflation_index_security": "No",
    }
    assert gac.normalize_security_type(record) == "Note 10-Year"

    reissue_record = {
        "security_type": "Note",
        "security_term": "9-Year 10-Month",
        "inflation_index_security": "No",
    }
    assert gac.normalize_security_type(reissue_record) == "Note 9-Year 10-Month"


def test_normalize_security_type_tips():
    """Verify TIPS are distinguished from nominal securities even when labeled as Note/Bond."""
    # From auctions_query where security_type is 'Note' but inflation_index_security is 'Yes'
    tips_query_record = {
        "security_type": "Note",
        "security_term": "10-Year",
        "inflation_index_security": "Yes",
    }
    assert gac.normalize_security_type(tips_query_record) == "TIPS Note 10-Year"

    # TIPS reissue
    tips_reissue_record = {
        "security_type": "Note",
        "security_term": "9-Year 10-Month",
        "inflation_index_security": "Yes",
    }
    assert gac.normalize_security_type(tips_reissue_record) == "TIPS Note 9-Year 10-Month"

    # From upcoming_auctions where security_type already has 'TIPS'
    tips_upcoming_record = {
        "security_type": "TIPS Note",
        "security_term": "9-Year 10-Month",
    }
    assert gac.normalize_security_type(tips_upcoming_record) == "TIPS Note 9-Year 10-Month"


def test_10y_note_reissue_pattern_matching():
    """Verify 10-Year Note patterns match original issues and reissues without matching TIPS."""
    patterns = [
        r"^note.*10.year",
        r"^10.year.*note",
        r"^note.*9.year.*(8|9|10|11).month",
    ]

    valid_10y_cases = [
        "note 10-year",
        "10-year note",
        "note 9-year 11-month",
        "note 9-year 10-month",
        "note 9-year 8-month",
    ]
    for case in valid_10y_cases:
        assert any(re.search(p, case) for p in patterns), f"Failed to match: {case}"

    invalid_cases = [
        "tips note 10-year",
        "tips note 9-year 10-month",
        "bond 19-year 10-month",
        "note 2-year",
        "note 5-year",
    ]
    for case in invalid_cases:
        assert not any(re.search(p, case) for p in patterns), f"Should not match: {case}"


def test_20y_and_30y_bond_reissue_pattern_matching():
    """Verify 20-Year and 30-Year Bond patterns match reissues."""
    p_20y = [
        r"^bond.*20.year",
        r"^20.year.*bond",
        r"^bond.*19.year.*(8|9|10|11).month",
    ]
    p_30y = [
        r"^bond.*30.year",
        r"^30.year.*bond",
        r"^bond.*29.year.*(8|9|10|11).month",
    ]

    assert any(re.search(p, "bond 20-year") for p in p_20y)
    assert any(re.search(p, "bond 19-year 11-month") for p in p_20y)
    assert any(re.search(p, "bond 19-year 10-Month".lower()) for p in p_20y)
    assert not any(re.search(p, "bond 29-year 10-month") for p in p_20y)

    assert any(re.search(p, "bond 30-year") for p in p_30y)
    assert any(re.search(p, "bond 29-year 11-month") for p in p_30y)
    assert any(re.search(p, "bond 29-year 10-month") for p in p_30y)
    assert not any(re.search(p, "tips bond 29-year 6-month") for p in p_30y)


def test_tips_pattern_matching():
    """Verify TIPS patterns match both new issues and reissues."""
    p_10y_tips = [
        r"tips.*10.year",
        r"10.year.*tips",
        r"tips.*9.year.*(8|9|10|11).month",
    ]
    p_5y_tips = [
        r"tips.*5.year",
        r"5.year.*tips",
        r"tips.*4.year.*1[01].month",
    ]

    assert any(re.search(p, "tips note 10-year") for p in p_10y_tips)
    assert any(re.search(p, "tips note 9-year 10-month") for p in p_10y_tips)
    assert any(re.search(p, "tips note 9-year 8-month") for p in p_10y_tips)
    assert not any(re.search(p, "note 9-year 10-month") for p in p_10y_tips)

    assert any(re.search(p, "tips note 5-year") for p in p_5y_tips)
    assert any(re.search(p, "tips note 4-year 10-month") for p in p_5y_tips)
    assert not any(re.search(p, "note 5-year") for p in p_5y_tips)


def test_fetch_auctions_merges_upcoming():
    """Verify fetch_auctions merges unannounced upcoming auctions without duplicating."""
    mock_query_data = {
        "data": [
            {
                "auction_date": "2026-09-23",
                "security_type": "Note",
                "security_term": "5-Year",
                "cusip": "91282CRN3",
                "offering_amt": "70000000000",
                "issue_date": "2026-09-30",
                "reopening": "No",
                "inflation_index_security": "No",
            }
        ]
    }
    mock_upcoming_data = {
        "data": [
            # Duplicate of the announced auction
            {
                "auction_date": "2026-09-23",
                "security_type": "Note",
                "security_term": "5-Year",
                "cusip": "91282CRN3",
                "offering_amt": "70000000000",
                "issue_date": "2026-09-30",
                "reopening": "No",
            },
            # Unannounced upcoming 10Y reissue
            {
                "auction_date": "2026-10-07",
                "security_type": "Note",
                "security_term": "9-Year 10-Month",
                "cusip": "91282CRF0",
                "offering_amt": "null",
                "announcemt_date": "2026-10-01",
                "auction_date": "2026-10-07",
                "issue_date": "2026-10-15",
                "reopening": "Yes",
            },
        ]
    }

    def mock_get(url, params=None, headers=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "auctions_query" in url:
            resp.json.return_value = mock_query_data
        elif "upcoming_auctions" in url:
            resp.json.return_value = mock_upcoming_data
        return resp

    with patch("requests.get", side_effect=mock_get):
        auctions = gac.fetch_auctions(2026)

    assert len(auctions) == 2
    dates = [a["auction_date"] for a in auctions]
    assert dates == ["2026-09-23", "2026-10-07"]

    oct_reissue = auctions[1]
    assert oct_reissue["security_type"] == "Note 9-Year 10-Month"
    assert oct_reissue["is_announced"] is False
    assert "Offering Amount: TBD (Unannounced)" in oct_reissue["details"]
    assert "Reopening: Yes" in oct_reissue["details"]


def test_format_ics_event_generation():
    """Verify format_ics generates valid VEVENT blocks with reminders."""
    events = [
        {
            "title": "10-Year Note Auction",
            "date": "20261007",
            "desc": "Status: TENTATIVE\nDetails: Reopening: Yes",
        }
    ]
    ics_text = gac.format_ics(events)

    assert "BEGIN:VCALENDAR" in ics_text
    assert "BEGIN:VEVENT" in ics_text
    assert "SUMMARY:10-Year Note Auction" in ics_text
    assert "DTSTART;VALUE=DATE:20261007" in ics_text
    assert "TRIGGER:-P2D" in ics_text
    assert "END:VEVENT" in ics_text
    assert "END:VCALENDAR" in ics_text
