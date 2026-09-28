#!/usr/bin/env python3
"""generate_auction_calendar.py: Generate an iCalendar (.ics) file of Treasury auctions.

Fetches real-time auction schedules from Treasury FiscalData APIs and the official
Treasury Tentative Auction Schedule XML, merging announced, near-term upcoming,
and forward-looking refunding auctions. Filters auctions by security type
(standard, minimum, or all) and matches original issues and reissues/reopenings.
Exports results to an iCalendar (.ics) file with reminder alarms and reports added
events (term, date, and issue type: New or Reopening) to stdout.

Usage:
    python generate_auction_calendar.py [OPTIONS] [YEAR]
    python generate_auction_calendar.py 2026 --standard
    python generate_auction_calendar.py 2026 --all
    python generate_auction_calendar.py --import
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import requests


# Configure logging with option to disable file logging
def setup_logging(log_to_file: bool = True) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        log_to_file: Whether to also log to a file (default: True)

    Returns:
        Configured logger instance
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.append(logging.FileHandler("auction_calendar.log"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

VERSION = "1.0.9"
# v1.04: Renamed --open to --import for clarity on adding events to calendar.
# v1.05: Switched to auctions_query API (from upcoming_auctions) to get Reopening
# status and Original Issue Date in event details.
# v1.06: Fixed reopening pattern matching for 10Y/20Y/30Y/5Y notes/bonds and TIPS;
# merged upcoming_auctions to include unannounced scheduled auctions.
# v1.07: Restricted calendar generation strictly to future events (today or afterward).
# v1.08: Specified New vs Reopening issue type in stdout summary output.
# v1.09: Ingested Treasury Tentative-Auction-Schedule.xml for full forward refunding calendar.


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the script.

    Returns:
        Namespace containing parsed arguments (year, all, minimum, standard, debug, no_log_file, import_calendar)
    """
    # ANSI escape codes for formatting
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    description_text = f"""{GREEN}{BOLD}Treasury Auction Calendar Generator v{VERSION}{RESET}

{CYAN}{BOLD}OVERVIEW:{RESET}
  This script generates a personalized iCalendar (.ics) file containing upcoming
  U.S. Treasury security auctions. It fetches real-time data from the official
  FiscalData Treasury API, allowing you to easily import auction schedules into
  your favorite calendar application (Google Calendar, Outlook, Apple Calendar, etc.).
  You can filter auctions by specific security types or include all.
  Each event's details include the CUSIP, offering amount, issue date, and
  whether the security is a reopening (with the original issue date if so).

{CYAN}{BOLD}USAGE:{RESET}
  {YELLOW}python generate_auction_calendar.py [OPTIONS] [YEAR]{RESET}

{CYAN}{BOLD}EXAMPLES:{RESET}
  {GREEN}# Generate calendar for the current year with standard securities (default){RESET}
  {YELLOW}python generate_auction_calendar.py{RESET}

  {GREEN}# Generate calendar for 2024 including ALL Treasury auctions{RESET}
  {YELLOW}python generate_auction_calendar.py 2024 --all{RESET}

  {GREEN}# Generate calendar for 2023 with only MINIMUM key securities and debug info{RESET}
  {YELLOW}python generate_auction_calendar.py 2023 --minimum --debug{RESET}

  {GREEN}# Generate calendar for the current year and import it automatically{RESET}
  {YELLOW}python generate_auction_calendar.py --import{RESET}
"""

    parser = argparse.ArgumentParser(
        description=description_text, formatter_class=argparse.RawTextHelpFormatter
    )

    # Year argument
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=datetime.now().year,
        help="Year to generate calendar for (default: current year)",
    )

    # Filter mode group
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--minimum",
        action="store_true",
        help="Include only minimum security types (5-Year TIPS, 10-Year TIPS, "
        "10-Year Note, 20-Year Bond)",
    )
    filter_group.add_argument(
        "--standard",
        action="store_true",
        help="Include standard security types (default): 5-Year TIPS, 10-Year TIPS, "
        "10-Year Note, 20-Year Bond, 5-Year Note, 30-Year Bond",
    )
    filter_group.add_argument(
        "--all", action="store_true", help="Include all Treasury auctions"
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging and save raw API response",
    )
    parser.add_argument(
        "--no-log-file", action="store_true", help="Disable logging to file"
    )
    parser.add_argument(
        "--import",
        action="store_true",
        dest="import_calendar",
        help="Automatically import the generated .ics file into your default calendar application.",
    )

    # Version and help
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Treasury Auction Calendar Generator version {VERSION}",
    )

    args = parser.parse_args()
    return args


def normalize_security_type(auction: dict) -> str:
    """
    Format security type string to distinguish TIPS from nominal securities
    and combine with security term.

    Args:
        auction: Raw dictionary representing an auction record from the API.

    Returns:
        Formatted string representing the security type and term.
    """
    is_tips = (
        auction.get("inflation_index_security") == "Yes"
        or "tips" in auction.get("security_type", "").lower()
        or "tips" in auction.get("security_term", "").lower()
    )
    raw_type = auction.get("security_type", "")
    if is_tips and not raw_type.lower().startswith("tips"):
        raw_type = f"TIPS {raw_type}"
    sec_term = auction.get("security_term", "")
    return f"{raw_type} {sec_term}".strip()


def determine_issue_type(auction: dict) -> str:
    """Determine whether an auction represents a New issue or a Reopening.

    Checks the 'reopening' metadata field, the 'details' string, and security
    term patterns (e.g., remaining month terms such as '9-Year 10-Month').

    Args:
        auction: Raw or processed auction record dictionary.

    Returns:
        'Reopening' if the auction is a reopening, otherwise 'New'.
    """
    reopening = str(auction.get("reopening", "")).strip().lower()
    if reopening == "yes":
        return "Reopening"
    if reopening == "no":
        return "New"

    details = str(auction.get("details", "")).lower()
    if "reopening: yes" in details:
        return "Reopening"
    if "reopening: no" in details:
        return "New"

    sec_type = str(auction.get("security_type", "")).lower()
    sec_term = str(auction.get("security_term", "")).lower()
    if re.search(r"\d+-month", sec_type) or re.search(r"\d+-month", sec_term):
        return "Reopening"

    return "New"


def canonical_security_key(security_type: str) -> str:
    """Normalize a security type and term into a canonical family key for deduplication.

    Maps reopening/remaining terms (e.g., '9-Year 10-Month' for a 10-Year Note)
    to their original term family (e.g., 'note 10-year') so records from
    announced, unannounced, and tentative schedule sources match.

    Args:
        security_type: Formatted security type string.

    Returns:
        Canonical lowercase string representing the security family.
    """
    s = security_type.lower()
    is_tips = "tips" in s
    prefix = "tips " if is_tips else ""

    if "bill" in s:
        m = re.search(r"\b(\d+)-week", s)
        if m:
            return f"bill {m.group(1)}-week"
        return s

    if "note" in s or "bond" in s:
        kind = "note" if "note" in s else "bond"
        if re.search(r"\b(29|30)[ -]year", s):
            return f"{prefix}{kind} 30-year"
        if re.search(r"\b(19|20)[ -]year", s):
            return f"{prefix}{kind} 20-year"
        if re.search(r"\b(9|10)[ -]year", s):
            return f"{prefix}{kind} 10-year"
        if re.search(r"\b(6|7)[ -]year", s):
            return f"{prefix}{kind} 7-year"
        if re.search(r"\b(4|5)[ -]year", s):
            return f"{prefix}{kind} 5-year"
        if re.search(r"\b(2|3)[ -]year", s):
            return f"{prefix}{kind} 3-year"
        if re.search(r"\b(1|2)[ -]year", s):
            return f"{prefix}{kind} 2-year"

    return s


def fetch_auctions(year: int, debug: bool = False) -> list:
    """Fetch auction data from Treasury FiscalData APIs and Tentative Auction Schedule XML.

    Combines announced auctions from FiscalData auctions_query, near-term unannounced
    auctions from FiscalData upcoming_auctions, and forward-looking refunding schedules
    from Treasury's official Tentative-Auction-Schedule.xml.

    Args:
        year: Year to fetch auctions for.
        debug: Whether to save raw API response and print debug info.

    Returns:
        List of processed auction records sorted by auction date.
    """
    logger.info(f"Fetching auctions for year {year}...")

    # Define the date range (only future events: today or afterward)
    today_str = datetime.now().strftime("%Y-%m-%d")
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"

    start_date = max(year_start, today_str)
    end_date = year_end

    if start_date > end_date:
        logger.info(
            f"Requested year {year} is in the past; no future auctions to fetch (today is {today_str})."
        )
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    processed_auctions = []
    seen_keys = set()

    # 1. Fetch announced and historical auctions from auctions_query
    params_query = {
        "filter": f"auction_date:gte:{start_date},auction_date:lte:{end_date}",
        "sort": "auction_date",
        "page[size]": "1000",
    }

    if debug:
        print(f"API parameters (auctions_query): {params_query}")

    try:
        api_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        response = requests.get(api_url, params=params_query, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if debug:
            with open(f"api_response_{year}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved raw API response to api_response_{year}.json")

        auctions_data = data.get("data", [])
        logger.info(f"Retrieved {len(auctions_data)} announced/historical auction records")

        for auction in auctions_data:
            auction_date = auction.get("auction_date")
            if not auction_date or auction_date < today_str:
                continue

            try:
                date_obj = datetime.strptime(auction_date, "%Y-%m-%d")
                security_type = normalize_security_type(auction)
                reopening = auction.get("reopening", "N/A")
                details = (
                    f"CUSIP: {auction.get('cusip', 'N/A')}, "
                    f"Offering Amount: {auction.get('offering_amt', 'N/A')}, "
                    f"Issue Date: {auction.get('issue_date', 'N/A')}, "
                    f"Reopening: {reopening}"
                )
                if reopening == "Yes":
                    original_issue_date = auction.get("original_issue_date")
                    if original_issue_date and original_issue_date != "null":
                        details += f" (Original Issue Date: {original_issue_date})"

                processed_auction = {
                    "security_type": security_type,
                    "auction_date": auction_date,
                    "year": date_obj.year,
                    "is_announced": True,
                    "details": details,
                    "reopening": reopening,
                }
                processed_auctions.append(processed_auction)

                cusip = auction.get("cusip")
                if cusip:
                    seen_keys.add((auction_date, cusip))
                seen_keys.add((auction_date, security_type))
                seen_keys.add((auction_date, canonical_security_key(security_type)))
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing auction record: {e}")
                if debug:
                    print(f"Error processing auction record: {e}")

    except requests.exceptions.RequestException as e:
        logger.error(f"auctions_query API request failed: {e}")
        if debug:
            print(f"auctions_query API request failed: {e}")

    # 2. Fetch unannounced upcoming auctions from upcoming_auctions for future dates
    upcoming_start = max(start_date, today_str)
    if upcoming_start <= end_date:
        params_upcoming = {
            "filter": f"auction_date:gte:{upcoming_start},auction_date:lte:{end_date}",
            "sort": "auction_date",
            "page[size]": "1000",
        }
        if debug:
            print(f"API parameters (upcoming_auctions): {params_upcoming}")

        try:
            upcoming_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
            response_upcoming = requests.get(upcoming_url, params=params_upcoming, headers=headers, timeout=15)
            response_upcoming.raise_for_status()
            upcoming_data = response_upcoming.json().get("data", [])

            unannounced_count = 0
            for auction in upcoming_data:
                auction_date = auction.get("auction_date")
                if not auction_date or auction_date < today_str:
                    continue

                cusip = auction.get("cusip")
                security_type = normalize_security_type(auction)
                canon_key = (auction_date, canonical_security_key(security_type))

                if (
                    (auction_date, cusip) in seen_keys
                    or (auction_date, security_type) in seen_keys
                    or canon_key in seen_keys
                ):
                    continue

                try:
                    date_obj = datetime.strptime(auction_date, "%Y-%m-%d")
                    reopening = auction.get("reopening", "N/A")
                    offering_amt = auction.get("offering_amt")
                    if not offering_amt or offering_amt == "null":
                        offering_amt = "TBD (Unannounced)"

                    announcemt_date = auction.get("announcemt_date")
                    is_announced = bool(announcemt_date and announcemt_date <= today_str)

                    details = (
                        f"CUSIP: {cusip or 'N/A'}, "
                        f"Offering Amount: {offering_amt}, "
                        f"Issue Date: {auction.get('issue_date', 'N/A')}, "
                        f"Reopening: {reopening}"
                    )
                    if announcemt_date:
                        details += f", Announcement Date: {announcemt_date}"

                    processed_auction = {
                        "security_type": security_type,
                        "auction_date": auction_date,
                        "year": date_obj.year,
                        "is_announced": is_announced,
                        "details": details,
                        "reopening": reopening,
                    }
                    processed_auctions.append(processed_auction)
                    if cusip:
                        seen_keys.add((auction_date, cusip))
                    seen_keys.add((auction_date, security_type))
                    seen_keys.add(canon_key)
                    unannounced_count += 1
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing upcoming auction record: {e}")
                    if debug:
                        print(f"Error processing upcoming auction record: {e}")

            if unannounced_count > 0:
                logger.info(f"Retrieved {unannounced_count} unannounced upcoming records from upcoming_auctions")

        except requests.exceptions.RequestException as e:
            logger.error(f"upcoming_auctions API request failed: {e}")
            if debug:
                print(f"upcoming_auctions API request failed: {e}")

    # 3. Fetch longer-term tentative schedule from Treasury Tentative-Auction-Schedule.xml
    if start_date <= end_date:
        xml_url = "https://home.treasury.gov/system/files/221/Tentative-Auction-Schedule.xml"
        if debug:
            print(f"Fetching XML schedule: {xml_url}")

        try:
            xml_resp = requests.get(xml_url, headers=headers, timeout=15)
            xml_resp.raise_for_status()
            root = ET.fromstring(xml_resp.content)
            xml_count = 0

            for elem in root.findall("AuctionCalendarDate"):
                auction_date = elem.findtext("AuctionDate")
                if not auction_date or auction_date < start_date or auction_date > end_date:
                    continue

                sec_type_raw = (elem.findtext("SecurityType") or "").strip().title()
                sec_term_raw = (elem.findtext("SecurityTermWeekYear") or "").strip()
                if not sec_type_raw or not sec_term_raw:
                    continue

                is_tips = (elem.findtext("TIPS") or "").strip().upper() == "Y"
                security_type = (
                    f"TIPS {sec_type_raw} {sec_term_raw}"
                    if is_tips
                    else f"{sec_type_raw} {sec_term_raw}"
                )
                canon_key = (auction_date, canonical_security_key(security_type))
                if (
                    (auction_date, security_type) in seen_keys
                    or canon_key in seen_keys
                ):
                    continue

                try:
                    date_obj = datetime.strptime(auction_date, "%Y-%m-%d")
                    reopen_raw = (
                        elem.findtext("ReOpeningIndicator") or ""
                    ).strip().upper()
                    reopening = "Yes" if reopen_raw == "Y" else "No"
                    announcemt_date = elem.findtext("AnnouncementDate")
                    settlement_date = elem.findtext("SettlementDate")
                    is_announced = bool(
                        announcemt_date and announcemt_date <= today_str
                    )

                    details = (
                        f"Offering Amount: TBD (Tentative Schedule), "
                        f"Issue Date: {settlement_date or 'N/A'}, "
                        f"Reopening: {reopening}"
                    )
                    if announcemt_date:
                        details += f", Announcement Date: {announcemt_date}"

                    processed_auction = {
                        "security_type": security_type,
                        "auction_date": auction_date,
                        "year": date_obj.year,
                        "is_announced": is_announced,
                        "details": details,
                        "reopening": reopening,
                    }
                    processed_auctions.append(processed_auction)
                    seen_keys.add((auction_date, security_type))
                    seen_keys.add(canon_key)
                    xml_count += 1
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing XML auction record: {e}")
                    if debug:
                        print(f"Error processing XML auction record: {e}")

            if xml_count > 0:
                logger.info(
                    f"Retrieved {xml_count} tentative schedule records from Tentative-Auction-Schedule.xml"
                )

        except (requests.exceptions.RequestException, ET.ParseError) as e:
            logger.warning(
                f"Could not fetch or parse Tentative-Auction-Schedule.xml: {e}"
            )
            if debug:
                print(f"Tentative-Auction-Schedule.xml fetch warning: {e}")

    processed_auctions.sort(key=lambda x: x["auction_date"])
    logger.info(f"Successfully processed {len(processed_auctions)} total auction records")
    return processed_auctions


def format_ics(events: list) -> str:
    """
    Format events into iCalendar (.ics) format.

    Args:
        events: List of event dictionaries with 'title', 'date', and 'desc' keys

    Returns:
        String containing the complete iCalendar formatted text
    """
    now = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    prodid = f"-//Treasury Auction Calendar Generator v{VERSION}//EN"

    ics = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"PRODID:{prodid}",
        f"X-WR-CALNAME:Treasury Auctions",
        f"X-WR-CALDESC:U.S. Treasury Security Auctions",
        f"X-WR-TIMEZONE:UTC",
    ]

    for e in events:
        # Create a consistent UID based on the event details
        event_id = f"{e['date']}-{e['title'].replace(' ', '_')}"
        uid = f"{event_id}@treasury.auction.calendar"

        ics.append("BEGIN:VEVENT")
        ics.append(f"UID:{uid}")
        ics.append(f"DTSTAMP:{now}")
        ics.append(f"DTSTART;VALUE=DATE:{e['date']}")
        ics.append(f"SUMMARY:{e['title']}")
        # Format multi-line description properly
        desc = e["desc"].replace("\n", "\\n")
        ics.append(f"DESCRIPTION:{desc}")
        ics.append(f"CATEGORIES:Treasury Auction")
        ics.append(f"TRANSP:TRANSPARENT")  # Does not block time in calendar

        # Add an alarm for 2 days before the auction
        ics.append("BEGIN:VALARM")
        ics.append("ACTION:DISPLAY")
        ics.append(f"DESCRIPTION:Reminder: {e['title']} is coming up")
        ics.append("TRIGGER:-P2D")  # 2 days before the event
        ics.append("END:VALARM")

        ics.append("END:VEVENT")

    ics.append("END:VCALENDAR")
    return "\n".join(ics)


def open_file(filepath: str) -> None:
    """Open a file with the default application for the current OS.

    Args:
        filepath: Path to the calendar file to open.
    """
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", filepath], check=True)
        elif system == "Windows":
            os.startfile(filepath)
        else:  # Linux and other Unix-like
            subprocess.run(["xdg-open", filepath], check=True)
        logger.info(f"Attempting to open calendar file: {filepath}")
    except (FileNotFoundError, subprocess.CalledProcessError, AttributeError) as e:
        logger.error(f"Failed to open calendar file automatically: {e}")
        print(f"\nCould not open {filepath} automatically.")
        print("Please open the file manually in your calendar application.")


def main() -> None:
    """Main function to generate the Treasury auction calendar.

    Fetches auction data from Treasury FiscalData APIs and the official
    Treasury Tentative Auction Schedule XML, filters by security type
    including reopenings, outputs added events to stdout, and generates
    an iCalendar (.ics) file for import into calendar applications.
    """
    # Parse command line arguments
    args = parse_arguments()
    year = args.year
    include_all = args.all
    minimum_only = args.minimum
    debug_mode = args.debug
    standard = not (
        include_all or minimum_only
    )  # Standard is the default if no other option is chosen
    output_file = f"auction_calendar_{year}.ics"

    # Setup logging based on command-line options
    global logger
    logger = setup_logging(not args.no_log_file)

    print(f"Treasury Auction Calendar Generator v{VERSION}")
    print(f"Generating calendar for year: {year}")
    print(f"Output will be saved to: {output_file}")

    if include_all:
        print(f"Filter mode: ALL auctions")
    elif minimum_only:
        print(f"Filter mode: MINIMUM securities only")
    else:
        print(f"Filter mode: STANDARD securities (default)")

    # Define minimum security types (original list)
    # Patterns use regex to match API security_term values. Reopenings/reissues
    # are reported with their remaining term (e.g., 9-Year 10-Month for 10-Year Notes,
    # 19-Year 10-Month for 20-Year Bonds, 29-Year 10-Month for 30-Year Bonds).
    minimum_security_types = [
        {
            "name": "5-Year TIPS",
            "patterns": [
                r"tips.*5.year",
                r"5.year.*tips",
                r"tips.*4.year.*1[01].month",
            ],
        },
        {
            "name": "10-Year TIPS",
            "patterns": [
                r"tips.*10.year",
                r"10.year.*tips",
                r"tips.*9.year.*(8|9|10|11).month",
            ],
        },
        {
            "name": "10-Year Note",
            "patterns": [
                r"^note.*10.year",
                r"^10.year.*note",
                r"^note.*9.year.*(8|9|10|11).month",
            ],
        },
        {
            "name": "20-Year Bond",
            "patterns": [
                r"^bond.*20.year",
                r"^20.year.*bond",
                r"^bond.*19.year.*(8|9|10|11).month",
            ],
        },
    ]

    # Define standard security types (original plus 5-Year Note and 30-Year Bond)
    standard_security_types = minimum_security_types + [
        {
            "name": "5-Year Note",
            "patterns": [
                r"^note.*5.year",
                r"^5.year.*note",
                r"^note.*4.year.*(8|9|10|11).month",
            ],
        },
        {
            "name": "30-Year Bond",
            "patterns": [
                r"^bond.*30.year",
                r"^30.year.*bond",
                r"^bond.*29.year.*(8|9|10|11).month",
            ],
        },
    ]

    print("Fetching Treasury auction data...")
    auctions = fetch_auctions(year, debug=debug_mode)

    # If no auctions were retrieved, exit gracefully
    if not auctions:
        print(
            "\nCould not retrieve any auction data from the API. No calendar file will be generated."
        )
        return

    print(f"Retrieved {len(auctions)} auctions in total")

    events = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for auction in auctions:
        auction_date = auction.get("auction_date", "")
        # Only add future events (today or afterward)
        if auction_date < today_str:
            continue

        security_type = auction["security_type"].lower()
        if debug_mode:
            print(
                f"Processing: {auction['security_type']} on {auction['auction_date']}"
            )

        # If --all flag is used, include all auctions
        if include_all:
            try:
                date_str = auction["auction_date"].replace("-", "")
                title = f"{auction['security_type']} Auction"
                status = (
                    "CONFIRMED" if auction.get("is_announced", False) else "TENTATIVE"
                )
                desc = f"Status: {status}\nDetails: {auction.get('details', 'N/A')}"
                issue_type = determine_issue_type(auction)
                events.append(
                    {
                        "title": title,
                        "date": date_str,
                        "desc": desc,
                        "term": auction["security_type"],
                        "auction_date": auction["auction_date"],
                        "issue_type": issue_type,
                    }
                )
                if debug_mode:
                    print(f"Added event: {title}")
            except Exception as e:
                logger.error(f"Error processing auction {auction}: {e}")
                if debug_mode:
                    print(f"Error processing auction {auction}: {e}")

            # Continue to next auction
            continue

        # Determine which security types to filter by based on command line options
        if minimum_only:
            security_types_to_use = minimum_security_types
            filter_description = "minimum securities"
        else:  # Standard is the default
            security_types_to_use = standard_security_types
            filter_description = "standard securities"

        # Check if this auction matches any of our security types of interest
        for interest_type in security_types_to_use:
            matched = False
            for pattern in interest_type["patterns"]:
                if re.search(pattern, security_type):
                    try:
                        date_str = auction["auction_date"].replace("-", "")
                        # Use the friendly name from our config
                        title = f"{interest_type['name']} Auction"
                        status = (
                            "CONFIRMED"
                            if auction.get("is_announced", False)
                            else "TENTATIVE"
                        )
                        desc = (
                            f"Status: {status}\n"
                            f"Original Security Type: {auction['security_type']}\n"
                            f"Details: {auction.get('details', 'N/A')}"
                        )
                        issue_type = determine_issue_type(auction)
                        events.append(
                            {
                                "title": title,
                                "date": date_str,
                                "desc": desc,
                                "term": interest_type["name"],
                                "auction_date": auction["auction_date"],
                                "issue_type": issue_type,
                            }
                        )
                        if debug_mode:
                            print(f"Added event: {title}")
                        matched = True
                        break
                    except Exception as e:
                        logger.error(f"Error processing auction {auction}: {e}")
                        if debug_mode:
                            print(f"Error processing auction {auction}: {e}")

            if matched:
                break

    print(
        f"Found {len(events)} events matching {filter_description if not include_all else 'all auctions'}"
    )

    if events:
        ics_text = format_ics(events)
        with open(output_file, "w") as f:
            f.write(ics_text)
        print(f"✅ Calendar saved as: {output_file} with {len(events)} events\n")
        print(f"Events added to {output_file}:")
        for event in events:
            term = event.get("term", event.get("title", ""))
            auction_date = event.get("auction_date", event.get("date", ""))
            issue_type = event.get("issue_type", "New")
            print(f"  [{term}] [{auction_date}] [{issue_type}]")
        if args.import_calendar:
            open_file(output_file)
    else:
        print(f"No future auctions found for {year} (from {today_str} onwards)")
        # Create an empty calendar file
        with open(output_file, "w") as f:
            f.write(format_ics([]))
        print(f"✅ Empty calendar file created: {output_file}")
        if args.import_calendar:
            open_file(output_file)


if __name__ == "__main__":
    main()
