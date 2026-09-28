#!/usr/bin/env python3
"""Unit tests for generate_auction_calendar.py.

Path: tests/test_generate_auction_calendar.py (relative to the repo root)

Covers auction type normalization, reissue/reopening pattern matching for
10Y/20Y/30Y/5Y notes/bonds and TIPS, unannounced upcoming auction merging,
Treasury Tentative Auction Schedule XML ingestion, canonical deduplication,
issue type determination (New vs Reopening), and iCalendar (.ics) generation
with synthetic data (no network access).

Usage:
    pytest tests/test_generate_auction_calendar.py -v
"""

import os
import re
import sys
from datetime import datetime, timedelta
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
    today = datetime.now()
    future_date_1 = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    future_date_2 = (today + timedelta(days=9)).strftime("%Y-%m-%d")

    mock_query_data = {
        "data": [
            {
                "auction_date": future_date_1,
                "security_type": "Note",
                "security_term": "5-Year",
                "cusip": "91282CRN3",
                "offering_amt": "70000000000",
                "issue_date": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
                "reopening": "No",
                "inflation_index_security": "No",
            }
        ]
    }
    mock_upcoming_data = {
        "data": [
            # Duplicate of the announced auction
            {
                "auction_date": future_date_1,
                "security_type": "Note",
                "security_term": "5-Year",
                "cusip": "91282CRN3",
                "offering_amt": "70000000000",
                "issue_date": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
                "reopening": "No",
            },
            # Unannounced upcoming 10Y reissue
            {
                "auction_date": future_date_2,
                "security_type": "Note",
                "security_term": "9-Year 10-Month",
                "cusip": "91282CRF0",
                "offering_amt": "null",
                "announcemt_date": (today + timedelta(days=3)).strftime("%Y-%m-%d"),
                "auction_date": future_date_2,
                "issue_date": (today + timedelta(days=14)).strftime("%Y-%m-%d"),
                "reopening": "Yes",
            },
        ]
    }

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "auctions_query" in url:
            resp.json.return_value = mock_query_data
        elif "upcoming_auctions" in url:
            resp.json.return_value = mock_upcoming_data
        elif "Tentative-Auction-Schedule.xml" in url:
            resp.content = b"<AuctionCalendar></AuctionCalendar>"
        return resp

    with patch("requests.get", side_effect=mock_get):
        auctions = gac.fetch_auctions(today.year)

    assert len(auctions) == 2
    dates = [a["auction_date"] for a in auctions]
    assert dates == [future_date_1, future_date_2]

    oct_reissue = auctions[1]
    assert oct_reissue["security_type"] == "Note 9-Year 10-Month"
    assert oct_reissue["is_announced"] is False
    assert oct_reissue["reopening"] == "Yes"
    assert "Offering Amount: TBD (Unannounced)" in oct_reissue["details"]
    assert "Reopening: Yes" in oct_reissue["details"]


def test_fetch_auctions_excludes_past_auctions():
    """Verify fetch_auctions strictly ignores auctions scheduled before today."""
    today = datetime.now()
    past_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    future_date = (today + timedelta(days=5)).strftime("%Y-%m-%d")

    mock_query_data = {
        "data": [
            {
                "auction_date": past_date,
                "security_type": "Note",
                "security_term": "10-Year",
                "cusip": "91282COLD1",
                "offering_amt": "35000000000",
                "reopening": "No",
            },
            {
                "auction_date": future_date,
                "security_type": "Note",
                "security_term": "10-Year",
                "cusip": "91282CNEW2",
                "offering_amt": "38000000000",
                "reopening": "No",
            },
        ]
    }

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "auctions_query" in url:
            resp.json.return_value = mock_query_data
        elif "upcoming_auctions" in url:
            resp.json.return_value = {"data": []}
        elif "Tentative-Auction-Schedule.xml" in url:
            resp.content = b"<AuctionCalendar></AuctionCalendar>"
        return resp

    with patch("requests.get", side_effect=mock_get):
        auctions = gac.fetch_auctions(today.year)

    assert len(auctions) == 1
    assert auctions[0]["auction_date"] == future_date


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


def test_determine_issue_type_explicit_reopening():
    """Verify determine_issue_type identifies New vs Reopening using reopening field."""
    assert gac.determine_issue_type({"reopening": "Yes"}) == "Reopening"
    assert gac.determine_issue_type({"reopening": "No"}) == "New"
    assert gac.determine_issue_type({"reopening": "yes"}) == "Reopening"
    assert gac.determine_issue_type({"reopening": "no"}) == "New"


def test_determine_issue_type_details_fallback():
    """Verify determine_issue_type inspects details string when reopening field is omitted."""
    assert gac.determine_issue_type({"details": "CUSIP: 91282CRF0, Reopening: Yes"}) == "Reopening"
    assert gac.determine_issue_type({"details": "CUSIP: 91282CRN3, Reopening: No"}) == "New"


def test_determine_issue_type_term_fallback():
    """Verify determine_issue_type uses security term month pattern as fallback."""
    assert gac.determine_issue_type({"security_type": "Note 9-Year 10-Month"}) == "Reopening"
    assert gac.determine_issue_type({"security_type": "Bond 29-Year 10-Month"}) == "Reopening"
    assert gac.determine_issue_type({"security_term": "19-Year 11-Month"}) == "Reopening"
    assert gac.determine_issue_type({"security_type": "Note 10-Year"}) == "New"


def test_main_summary_output_issue_type(capsys, tmp_path):
    """Verify main() prints [term] [date] [New|Reopening] in the stdout summary."""
    today = datetime.now()
    date_new = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    date_reopening = (today + timedelta(days=4)).strftime("%Y-%m-%d")

    mock_auctions = [
        {
            "security_type": "Note 10-Year",
            "auction_date": date_new,
            "year": today.year,
            "is_announced": True,
            "details": "Reopening: No",
            "reopening": "No",
        },
        {
            "security_type": "Note 9-Year 10-Month",
            "auction_date": date_reopening,
            "year": today.year,
            "is_announced": True,
            "details": "Reopening: Yes",
            "reopening": "Yes",
        },
    ]

    with patch("generate_auction_calendar.fetch_auctions", return_value=mock_auctions), \
         patch("sys.argv", ["generate_auction_calendar.py", str(today.year)]), \
         patch("generate_auction_calendar.open_file"):
        gac.main()

    captured = capsys.readouterr()
    expected_new = f"[10-Year Note] [{date_new}] [New]"
    expected_reopening = f"[10-Year Note] [{date_reopening}] [Reopening]"

    assert expected_new in captured.out
    assert expected_reopening in captured.out


def test_canonical_security_key():
    """Verify canonical_security_key normalizes terms and remaining-month terms to families."""
    assert gac.canonical_security_key("Note 10-Year") == "note 10-year"
    assert gac.canonical_security_key("Note 9-Year 10-Month") == "note 10-year"
    assert gac.canonical_security_key("Bond 30-Year") == "bond 30-year"
    assert gac.canonical_security_key("Bond 29-Year 10-Month") == "bond 30-year"
    assert gac.canonical_security_key("Bond 20-Year") == "bond 20-year"
    assert gac.canonical_security_key("Bond 19-Year 11-Month") == "bond 20-year"
    assert gac.canonical_security_key("Note 5-Year") == "note 5-year"
    assert gac.canonical_security_key("Note 4-Year 10-Month") == "note 5-year"
    assert gac.canonical_security_key("TIPS Note 10-Year") == "tips note 10-year"
    assert gac.canonical_security_key("TIPS Note 9-Year 10-Month") == "tips note 10-year"
    assert gac.canonical_security_key("Bill 13-Week") == "bill 13-week"


def test_fetch_auctions_ingests_xml_tentative_schedule():
    """Verify fetch_auctions ingests Tentative-Auction-Schedule.xml and dedupes properly."""
    today = datetime.now()
    d1 = (today + timedelta(days=5)).strftime("%Y-%m-%d")
    d2 = (today + timedelta(days=15)).strftime("%Y-%m-%d")

    mock_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <AuctionCalendar>
        <!-- Duplicate of upcoming auction on d1 (10Y Note reopening) -->
        <AuctionCalendarDate>
            <SecurityTermWeekYear>10-Year</SecurityTermWeekYear>
            <SecurityType>NOTE</SecurityType>
            <ReOpeningIndicator>Y</ReOpeningIndicator>
            <TIPS>N</TIPS>
            <AuctionDate>{d1}</AuctionDate>
            <AnnouncementDate>{d1}</AnnouncementDate>
            <SettlementDate>{d1}</SettlementDate>
        </AuctionCalendarDate>
        <!-- Forward refunding auction on d2 (New 30Y Bond) -->
        <AuctionCalendarDate>
            <SecurityTermWeekYear>30-Year</SecurityTermWeekYear>
            <SecurityType>BOND</SecurityType>
            <ReOpeningIndicator>N</ReOpeningIndicator>
            <TIPS>N</TIPS>
            <AuctionDate>{d2}</AuctionDate>
            <AnnouncementDate>{d2}</AnnouncementDate>
            <SettlementDate>{d2}</SettlementDate>
        </AuctionCalendarDate>
    </AuctionCalendar>
    """.encode("utf-8")

    mock_upcoming_data = {
        "data": [
            {
                "auction_date": d1,
                "security_type": "Note",
                "security_term": "9-Year 10-Month",
                "cusip": "91282CRF0",
                "offering_amt": "null",
                "reopening": "Yes",
            }
        ]
    }

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "auctions_query" in url:
            resp.json.return_value = {"data": []}
        elif "upcoming_auctions" in url:
            resp.json.return_value = mock_upcoming_data
        elif "Tentative-Auction-Schedule.xml" in url:
            resp.content = mock_xml
        return resp

    with patch("requests.get", side_effect=mock_get):
        auctions = gac.fetch_auctions(today.year)

    # d1 duplicate from XML should be dropped; d2 should be added
    assert len(auctions) == 2
    assert auctions[0]["auction_date"] == d1
    assert auctions[0]["security_type"] == "Note 9-Year 10-Month"
    assert auctions[0]["reopening"] == "Yes"

    assert auctions[1]["auction_date"] == d2
    assert auctions[1]["security_type"] == "Bond 30-Year"
    assert auctions[1]["reopening"] == "No"
    assert "Tentative Schedule" in auctions[1]["details"]
