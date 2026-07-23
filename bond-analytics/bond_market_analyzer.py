#!/usr/bin/env python3
"""Fetch, calculate, and track secondary-market Treasury bond pricing by CUSIP.

Default mode fetches a live quote (coupon, maturity, bid/ask/last, yield) for a
CUSIP from Webull, cross-checks the coupon against Treasury's own auction record,
calculates Yield to Worst, and -- after showing you the entry and asking for
confirmation -- appends it to a dated history in a local JSON file
(data/json/treasuries_secondary_market.json). Guided manual entry (--manual) and
raw copy-paste entry (--copypaste) are available when no fetch source has data
for a CUSIP. See fetch_add(), interactive_add(), and parse_input() respectively.
"""
import argparse
import getpass
import json
import os
import sys
from statistics import median
from pathlib import Path
from datetime import datetime
from calculate_YTW import calculate_ytw_bey
import finra_bond_fetcher
import webull_bond_fetcher

VERSION = "1.4.0"
# v1.4.0: Switched default fetch source from FINRA to Webull (live bid/ask/last;
#         FINRA's coupon field was found to be unreliable and its trade prints
#         can be stale). FINRA fetcher kept dormant, reachable via --source finra.
# v1.3.0: Default mode now fetches price/yield from FINRA by CUSIP. Guided manual
#         entry (formerly the default) moved to --manual.
# v1.2.0: Added --interactive mode for guided entry.

# Colors for snazzy output -- disabled when stdout isn't a terminal (e.g. redirected
# to a log file by launchd for --daily runs), so log files stay plain, readable text.
_USE_COLOR = sys.stdout.isatty()
CYAN = "\033[96m" if _USE_COLOR else ""
GREEN = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

_NOTE_COLORS = {"info": CYAN, "warn": YELLOW, "error": RED}

def _print_notes(notes, indent=""):
    """Print (level, message) tuples from _fetch_bond_entry, color-coded by level."""
    for level, message in notes:
        print(f"{indent}{_NOTE_COLORS.get(level, '')}{message}{RESET}")

# Use absolute path or path relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "data", "json", "treasuries_secondary_market.json")

# Coupons within this many percentage points of Treasury's official figure are
# treated as a match (guards against float formatting noise, not real disagreement).
COUPON_MATCH_TOLERANCE_PCT = 0.001

# launchd job that runs `--daily` on a schedule -- see print_help()'s LAUNCHD section
# and setup_daily_fetch_launchd.py, which generates and installs the plist below.
LAUNCHD_LABEL = f"com.{getpass.getuser()}.bondanalytics.dailyfetch"
LAUNCHD_PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
DAILY_LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "daily_fetch.log")

def parse_input():
    """Parse input data for treasury bonds.
    Reads multi-line input directly from stdin.
    """
    print("Copy and paste your treasury data below (press Ctrl+D when finished):")
    
    # Read all lines from stdin until EOF (Ctrl+D)
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    # Process the input data
    result = []

    def is_valid_ytw(ytw_str):
        """True if ytw_str parses to a plausible YTW percentage (0 < x < 30)."""
        try:
            ytw_val = float(ytw_str.replace('%', ''))
            return 0 < ytw_val < 30
        except Exception:
            return False

    def needs_outlier_confirmation(coupon_str, ytw_str, threshold=2.5):
        """True if YTW deviates from the coupon rate by more than threshold points."""
        try:
            coupon_val = float(coupon_str.replace('%', ''))
            ytw_val = float(ytw_str.replace('%', ''))
            return abs(ytw_val - coupon_val) > threshold
        except Exception:
            return False

    # If we have data
    if lines:
        # Join all non-empty lines with spaces
        all_text = " ".join([line.strip() for line in lines if line.strip()])
        all_fields = all_text.split()

        # If we have at least 4 fields
        if len(all_fields) >= 4:
            try:
                coupon = all_fields[0]
                maturity = all_fields[1]
                price = all_fields[2]
                ytw = all_fields[3]  # Default to 4th field

                # If we have at least 7 fields, use field 7 instead
                if len(all_fields) >= 7:
                    ytw = all_fields[6]

                # Validate YTW
                if not is_valid_ytw(ytw):
                    print(f"Warning: YTW value '{ytw}' is out of expected range (0 < YTW < 30). Entry skipped.")
                    return []

                # Check if YTW is an outlier compared to coupon
                outlier = needs_outlier_confirmation(coupon, ytw)

                # Create an entry with today's date
                entry = {
                    "Coupon": coupon,
                    "Maturity": maturity,
                    "Price": price,
                    "YTW": ytw,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }

                # Show entry and request confirmation
                print("\nParsed bond entry:")
                for k, v in entry.items():
                    print(f"{k}: {v}")
                if outlier:
                    print(f"\nWarning: YTW ({ytw}) differs from coupon rate ({coupon}) by more than 2.5%. Please confirm this is correct.")
                confirm = input("\nAdd this entry to JSON? (Y/N): ").strip().lower()
                if confirm == 'y':
                    result.append(entry)
                else:
                    print("Entry not added.")
            except Exception as e:
                print(f"Warning: Could not parse data: {e}")

    return result

def is_numeric(value):
    """Check if a string represents a numeric value"""
    try:
        float(value.replace('$', '').replace(',', '').replace('%', ''))
        return True
    except (ValueError, AttributeError):
        return False

def load_data():
    """Load the saved bond history from DATA_FILE, or {} if it doesn't exist yet or fails to parse."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading data: {e}")
            return {}
    return {}

def save_data(data):
    """Save data to the JSON file, preserving existing data not in the current dataset."""
    # Ensure directory exists
    data_dir = os.path.dirname(DATA_FILE)
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Check if the file exists and has content
        existing_data = {}
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
            try:
                with open(DATA_FILE, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing data file is corrupted, creating new file")
        
        # Merge the existing data with new data
        merged_data = existing_data.copy()
        for cusip, entries in data.items():
            if cusip in merged_data:
                existing_dates = {entry.get('Date') for entry in merged_data[cusip] if 'Date' in entry}
                for entry in entries:
                    if entry.get('Date') and entry.get('Date') not in existing_dates:
                        merged_data[cusip].append(entry)
                    elif '_comment' in entry:
                        merged_data[cusip].append(entry)
            else:
                merged_data[cusip] = entries

        # For each CUSIP, ensure only one _comment entry is kept
        for cusip in merged_data:
            comments = []
            others = []
            for entry in merged_data[cusip]:
                if '_comment' in entry:
                    if not comments:
                        comments.append(entry)
                else:
                    others.append(entry)
            merged_data[cusip] = comments + others

        # Save the merged data back to file
        with open(DATA_FILE, "w") as f:
            json.dump(merged_data, f, indent=2)
    except IOError as e:
        print(f"Error saving data: {e}")

def show_info(cusip, data):
    """Print low/high/median/latest YTW and price-vs-average stats for a CUSIP's saved history."""
    if cusip not in data:
        print(f"No data for CUSIP {cusip}")
        return

    try:
        # Convert YTW values to float, removing any '%' characters
        ytws = []
        prices = []
        for entry in data[cusip]:
            try:
                ytw_str = entry["YTW"].replace('%', '')
                ytws.append(float(ytw_str))
            except (ValueError, KeyError):
                pass
            try:
                price_str = entry["Price"].replace('$', '').replace(',', '')
                prices.append(float(price_str))
            except (ValueError, KeyError):
                pass

        if not ytws:
            print(f"No valid YTW values found for CUSIP {cusip}")
            return

        print(f"CUSIP: {cusip}")
        print(f"Low YTW: {min(ytws):.3f}%")
        print(f"High YTW: {max(ytws):.3f}%")
        print(f"Median YTW: {median(ytws):.3f}%")
        print(f"Latest YTW: {ytws[-1]:.3f}%")
        print(f"Number of entries: {len(ytws)}")

        # Price comparison logic
        if prices:
            latest_price = prices[-1]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            pct_vs_avg = ((latest_price - avg_price) / avg_price) * 100 if avg_price else 0
            print(f"Latest Price: ${latest_price:.3f}")
            print(f"Average Price: ${avg_price:.3f}")
            print(f"Lowest Price: ${min_price:.3f}")
            print(f"Highest Price: ${max_price:.3f}")
            if abs(pct_vs_avg) < 0.01:
                print(f"Current price is equal to the average for this CUSIP.")
            elif pct_vs_avg > 0:
                print(f"Current price is {pct_vs_avg:.2f}% more expensive than average for this CUSIP.")
            else:
                print(f"Current price is {abs(pct_vs_avg):.2f}% less expensive than average for this CUSIP.")
            if latest_price == max_price:
                print("This is the most expensive price recorded for this CUSIP.")
            if latest_price == min_price:
                print("This is the least expensive price recorded for this CUSIP.")
    except Exception as e:
        print(f"Error analyzing data for CUSIP {cusip}: {str(e)}")

def interactive_add(data):
    """Interactively add a new bond entry by prompting for fields."""
    try:
        import inquirer
    except ImportError:
        print(f"{RED}Error: 'inquirer' library is required for interactive mode.{RESET}")
        print("Please install it by running: pip install inquirer")
        return

    try:
        # 1. Prompt for CUSIP, highlighting owned bonds
        owned_cusips = set()
        for cusip, entries in data.items():
            for entry in entries:
                if "_comment" in entry and "paid" in entry.get("_comment", ""):
                    owned_cusips.add(cusip)

        cusip_choices = [("[ New CUSIP ]", "[ New CUSIP ]")]
        for cusip in sorted(data.keys()):
            if cusip in owned_cusips:
                display_text = f"{GREEN}{cusip} (owned){RESET}"
                cusip_choices.append((display_text, cusip))
            else:
                cusip_choices.append((cusip, cusip))

        questions = [
            inquirer.List('cusip',
                          message="Select a CUSIP or add a new one",
                          choices=cusip_choices,
                          ),
        ]
        answers = inquirer.prompt(questions)
        if not answers:  # User pressed Ctrl+C
            print(f"{YELLOW}Selection cancelled.{RESET}")
            return

        cusip = answers['cusip']

        if cusip == "[ New CUSIP ]":
            cusip = input(f"{BOLD}Enter new CUSIP: {RESET}").strip().upper()

        if not cusip:
            print(f"{RED}Error: CUSIP cannot be empty.{RESET}")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        # Check for existing entry for today
        if cusip in data:
            for existing_entry in data[cusip]:
                if existing_entry.get("Date") == today_str:
                    print(f"{YELLOW}Entry for CUSIP {cusip} on {today_str} already exists. Not adding duplicate.{RESET}")
                    return

        # 2. Check if CUSIP exists in the data at all
        if cusip in data and data[cusip]:
            # CUSIP exists, use its data to pre-fill
            print(f"{CYAN}CUSIP {cusip} found. Using existing data for Coupon and Maturity.{RESET}")
            # Find the most recent non-comment entry to get coupon/maturity
            latest_entry = None
            for entry in reversed(data[cusip]):
                if '_comment' not in entry:
                    latest_entry = entry
                    break
            
            if not latest_entry:
                print(f"{RED}Error: No valid previous data entries found for CUSIP {cusip} to source coupon/maturity from.{RESET}")
                return

            coupon_str = latest_entry['Coupon']
            maturity_str = latest_entry['Maturity']
            print(f"  - Coupon: {GREEN}{coupon_str}{RESET}")
            print(f"  - Maturity: {GREEN}{maturity_str}{RESET}")

            # Prompt for price
            while True:
                try:
                    price_str = input(f"{BOLD}Enter Current Price: {RESET}")
                    price = float(price_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid price. Please enter a number.{RESET}")

            # Calculate YTW
            print(f"{CYAN}Calculating YTW...{RESET}")
            ytw_pct = calculate_ytw_bey(
                coupon_rate=float(coupon_str),
                maturity_date=maturity_str,
                price=price
            )
            ytw = f"{ytw_pct:.3f}"
            print(f"Calculated YTW: {GREEN}{ytw}%{RESET}")

            new_entry = {
                "Coupon": coupon_str,
                "Maturity": maturity_str,
                "Price": f"{price:.3f}",
                "YTW": ytw,
                "Date": today_str
            }

        else:
            # CUSIP does not exist, prompt for everything
            print(f"{CYAN}CUSIP {cusip} not found. Please provide all details for this new bond.{RESET}")

            # Prompt for coupon
            while True:
                try:
                    coupon_str = input(f"{BOLD}Enter Coupon Rate: {RESET}").strip()
                    coupon = float(coupon_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid coupon. Please enter a number.{RESET}")

            # Prompt for maturity
            while True:
                try:
                    maturity_str = input(f"{BOLD}Enter Maturity Date (MM/DD/YYYY): {RESET}").strip()
                    datetime.strptime(maturity_str, "%m/%d/%Y")  # just for validation
                    break
                except ValueError:
                    print(f"{RED}Invalid date format. Please use MM/DD/YYYY.{RESET}")

            # Prompt for price
            while True:
                try:
                    price_str = input(f"{BOLD}Enter Price: {RESET}").strip()
                    price = float(price_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid price. Please enter a number.{RESET}")

            # Calculate YTW
            print(f"{CYAN}Calculating YTW...{RESET}")
            ytw_pct = calculate_ytw_bey(
                coupon_rate=coupon,
                maturity_date=maturity_str,
                price=price
            )
            ytw_str = f"{ytw_pct:.3f}"
            print(f"Calculated YTW: {GREEN}{ytw_str}%{RESET}")

            new_entry = {
                "Coupon": f"{coupon:.3f}",
                "Maturity": maturity_str,
                "Price": f"{price:.3f}",
                "YTW": ytw_str,
                "Date": today_str
            }

        # Confirmation before adding
        print(f"\n{BOLD}{CYAN}New entry to be added:{RESET}")
        for k, v in new_entry.items():
            print(f"  {k}: {GREEN}{v}{RESET}")
        confirm = input(f"\n{BOLD}Add this entry? [Y/n]: {RESET}").strip().lower()
        if confirm == 'n':
            print(f"{YELLOW}Entry not added.{RESET}")
            return

        # Add to data structure and save
        if cusip not in data:
            data[cusip] = []
        data[cusip].append(new_entry)
        save_data(data)
        print(f"{GREEN}Saved new entry for CUSIP {cusip}.{RESET}")

    except (ValueError, RuntimeError) as e:
        print(f"\n{RED}Error: {e}. Aborting interactive add.{RESET}")
    except Exception as e:
        print(f"\n{RED}An unexpected error occurred: {e}. Aborting.{RESET}")
        import traceback
        traceback.print_exc()

SOURCE_MODULES = {
    "webull": webull_bond_fetcher,
    "finra": finra_bond_fetcher,
}

def _fetch_bond_entry(cusip, source="webull"):
    """Fetch coupon/maturity/price for cusip from source and cross-check the coupon
    against Treasury's own auction record. Never raises for expected failure modes
    (network error, no record, no price, coupon mismatch) -- those come back as a
    None entry plus an explanatory note instead, so this is safe to call from both
    an interactive prompt and an unattended batch job.

    Returns {"entry": dict|None, "clean": bool, "notes": [(level, message), ...]}.
    entry is the ready-to-save Coupon/Maturity/Price/YTW/Date dict, or None if no
    usable price could be determined. clean is True only when the fetch succeeded,
    a price was found, AND the coupon was independently confirmed against Treasury's
    auction record -- an unverified (no Treasury record) or mismatched coupon marks
    clean=False even though entry may still be populated, so callers that want to
    auto-save unattended (see fetch_all_owned) can require clean=True.
    """
    fetcher = SOURCE_MODULES[source]
    notes = []

    try:
        record = fetcher.fetch_treasury_price(cusip)
    except Exception as e:
        return {"entry": None, "clean": False, "notes": [("error", f"Error fetching data from {source}: {e}")]}

    if not record:
        return {"entry": None, "clean": False, "notes": [("error", f"No {source} record found for CUSIP {cusip}.")]}

    coupon = record.get("couponRate")
    maturity_raw = record.get("maturityDate")
    price = record.get("price")

    if coupon is None or maturity_raw is None:
        return {"entry": None, "clean": False,
                "notes": [("error", f"{source} record for {cusip} is missing coupon/maturity data.")]}

    coupon = float(coupon)
    maturity_str = datetime.strptime(maturity_raw, "%Y-%m-%d").strftime("%m/%d/%Y")

    # Cross-check coupon against Treasury's own auction record regardless of source,
    # since FINRA's TreasurySecurities coupon field has already been caught wrong once.
    try:
        official_coupon = finra_bond_fetcher.fetch_official_coupon(cusip)
    except Exception:
        official_coupon = None

    clean = True
    if official_coupon is not None:
        if abs(official_coupon - coupon) > COUPON_MATCH_TOLERANCE_PCT:
            notes.append(("error", f"Coupon mismatch: {source} reports {coupon}%, but Treasury's own auction "
                                    f"record says {official_coupon}%. Using Treasury's figure."))
            coupon = official_coupon
            clean = False
        else:
            notes.append(("info", f"Coupon verified against Treasury's own auction record: {coupon}%"))
    else:
        notes.append(("warn", f"Could not verify coupon against Treasury (no auction record found) -- "
                               f"using {source}'s reported coupon: {coupon}%. Double-check this value."))
        clean = False

    notes.append(("info", f"{source} reference data -- issuer: {record.get('issuerName', 'n/a')}, "
                           f"coupon: {coupon}%, maturity: {maturity_str}"))

    if price is None:
        notes.append(("warn", f"{source} has no current price for CUSIP {cusip}."))
        return {"entry": None, "clean": False, "notes": notes}

    price = float(price)

    if source == "webull":
        notes.append(("info", f"Last: {price:.3f} | Bid: {record.get('bidPrice')} | Ask: {record.get('askPrice')} "
                               f"| as of {record.get('tradeTime')} (yield: {record.get('yield')}%)"))
    else:
        notes.append(("info", f"Last trade: {price:.3f} on {record.get('tradeDate')} {record.get('tradeTime')} "
                               f"(yield: {record.get('yield')}%)"))

    ytw_pct = calculate_ytw_bey(coupon_rate=coupon, maturity_date=maturity_str, price=price)

    entry = {
        "Coupon": f"{coupon:.3f}",
        "Maturity": maturity_str,
        "Price": f"{price:.3f}",
        "YTW": f"{ytw_pct:.3f}",
        "Date": datetime.now().strftime("%Y-%m-%d"),
    }

    return {"entry": entry, "clean": clean, "notes": notes}

def fetch_add(data, cusip=None, source="webull"):
    """Interactively fetch a CUSIP's price via _fetch_bond_entry and save on confirmation.

    source: "webull" (default, live bid/ask/last) or "finra" (dormant -- see
    finra_bond_fetcher.py's module docstring for why it was demoted).
    """
    cusip = (cusip or input("Enter CUSIP: ").strip()).strip().upper()
    if not cusip:
        print(f"{RED}Error: CUSIP cannot be empty.{RESET}")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    if cusip in data:
        for existing_entry in data[cusip]:
            if existing_entry.get("Date") == today_str:
                print(f"{YELLOW}Entry for CUSIP {cusip} on {today_str} already exists. Not adding duplicate.{RESET}")
                show_info(cusip, data)
                return

    print(f"{CYAN}Fetching {cusip} from {source}...{RESET}")
    result = _fetch_bond_entry(cusip, source)
    _print_notes(result["notes"])

    new_entry = result["entry"]
    if not new_entry:
        print(f"Use {BOLD}--manual{RESET} to enter this bond's details by hand instead.")
        return

    print(f"\n{BOLD}{CYAN}Entry to be added for {cusip}:{RESET}")
    for k, v in new_entry.items():
        print(f"  {k}: {GREEN}{v}{RESET}")

    confirm = input(f"\n{BOLD}Add this entry? [Y/n]: {RESET}").strip().lower()
    if confirm == 'n':
        print(f"{YELLOW}Entry not added.{RESET}")
        return

    if cusip not in data:
        data[cusip] = []
    data[cusip].append(new_entry)
    save_data(data)
    print(f"{GREEN}Saved new entry for CUSIP {cusip}.{RESET}")

def fetch_all_owned(data, source="webull"):
    """Non-interactively fetch and save today's price for every owned CUSIP.

    "Owned" means a CUSIP with a _comment entry starting with "paid" (the same
    marker interactive_add() uses to highlight owned bonds). For each one: skip
    if today's entry already exists; otherwise fetch via _fetch_bond_entry and
    save ONLY if clean=True (price found and coupon independently confirmed
    against Treasury) -- anything else (fetch error, no price, unverified or
    mismatched coupon) is skipped and reported in the summary for manual review
    via `--CUSIP <cusip>`, rather than guessed at.

    Intended for unattended/scheduled runs -- see --daily and the LAUNCHD section
    of --help. Saves incrementally after each CUSIP so a crash partway through
    doesn't lose already-fetched entries.
    """
    owned = sorted(cusip for cusip, entries in data.items()
                   if any(e.get("_comment", "").startswith("paid") for e in entries))

    if not owned:
        print(f"{YELLOW}No owned CUSIPs found (no _comment starting with 'paid').{RESET}")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"{BOLD}{CYAN}Daily fetch for {len(owned)} owned CUSIP(s) -- {today_str} ({source}){RESET}")

    saved, skipped = [], []
    for cusip in owned:
        print(f"\n{BOLD}{cusip}{RESET}")

        if any(e.get("Date") == today_str for e in data[cusip]):
            print(f"{YELLOW}  Already have an entry for today. Skipping.{RESET}")
            skipped.append((cusip, "already up to date"))
            continue

        result = _fetch_bond_entry(cusip, source)
        _print_notes(result["notes"], indent="  ")

        if result["entry"] and result["clean"]:
            data[cusip].append(result["entry"])
            save_data(data)
            print(f"{GREEN}  Saved: {result['entry']}{RESET}")
            saved.append(cusip)
        else:
            reason = "fetch failed or no price available" if not result["entry"] else "coupon unverified or mismatched"
            print(f"{RED}  Skipped -- {reason}.{RESET}")
            skipped.append((cusip, reason))

    print(f"\n{BOLD}{CYAN}Daily fetch summary -- {today_str}{RESET}")
    print(f"  Saved ({len(saved)}): {', '.join(saved) if saved else 'none'}")
    print(f"  Skipped ({len(skipped)}):")
    for cusip, reason in skipped:
        print(f"    {cusip}: {reason} -- review with --CUSIP {cusip}")
    if not skipped:
        print("    none")

def print_help():
    """Print a formatted help screen: usage, modes with examples, options, data sources, and file location."""
    script_name = os.path.basename(sys.argv[0])
    rule = f"{CYAN}{'-' * 66}{RESET}"

    def opt(flag, desc):
        if flag:
            print(f"  {BOLD}{flag:<24}{RESET}{desc}")
        else:
            print(f"  {'':<24}{desc}")

    def mode(flag, desc, example):
        print(f"  {BOLD}{flag:<14}{RESET}{desc}")
        print(f"{'':16}{YELLOW}$ {script_name} {example}{RESET}")

    print(f"\n{rule}")
    print(f"{BOLD}{GREEN}  Bond Market Analyzer  {RESET}{CYAN}v{VERSION}{RESET}")
    print(rule)
    print("  Fetches live secondary-market Treasury bond pricing by CUSIP,")
    print("  calculates Yield to Worst, and saves a dated history to JSON.")

    print(f"\n{BOLD}{CYAN}USAGE{RESET}")
    print(f"  {script_name} [OPTIONS]")

    print(f"\n{BOLD}{CYAN}MODES{RESET}  (pick one; default shown first)")
    mode("(default)", "Fetch price/yield automatically for a CUSIP.", "--CUSIP 912810UF3")
    mode("--manual", "Guided prompts for coupon/maturity/price (no network calls).", "--manual")
    mode("--copypaste", "Paste raw bond values copied from a broker screen.", "--CUSIP 912810UF3 --copypaste")
    mode("--getinfo", "Show saved low/high/median/latest YTW stats for a CUSIP.", "--CUSIP 912810UF3 --getinfo")
    mode("--daily", "Unattended: fetch + save today's price for every owned CUSIP.", "--daily")

    print(f"\n{BOLD}{CYAN}OPTIONS{RESET}")
    opt("--CUSIP CUSIP", "9-digit CUSIP identifier.")
    opt("--source {webull,finra}", "Price source for default fetch mode (default: webull).")
    opt("", "finra is dormant -- see finra_bond_fetcher.py.")
    opt("--manual", "Use guided manual entry mode instead of fetching.")
    opt("--copypaste", "Use copy/paste mode instead of fetching.")
    opt("--getinfo", "Show statistics for a CUSIP instead of adding an entry.")
    opt("--daily", "Fetch/save every owned CUSIP non-interactively (no y/N prompt).")
    opt("-h, --help", "Show this help message and exit.")
    opt("-v, --version", "Show script version and exit.")

    print(f"\n{BOLD}{CYAN}DATA SOURCES{RESET}")
    print(f"  {GREEN}webull{RESET} (default)  Live bid/ask/last quote, server-rendered, no login required.")
    print(f"  {YELLOW}finra{RESET}  (dormant)  TRACE trade history; can be months stale for thinly-traded")
    print("                     CUSIPs, and its coupon reference field has been observed wrong.")
    print("                     See finra_bond_fetcher.py's module docstring for the full story.")
    print("  Every fetch cross-checks the coupon against Treasury's own auction record")
    print("  (api.fiscaldata.treasury.gov) regardless of source, and flags any mismatch")
    print("  instead of silently trusting either source.")

    print(f"\n{BOLD}{CYAN}DATA FILE{RESET}")
    print(f"  {os.path.relpath(DATA_FILE, SCRIPT_DIR)} (relative to this script)")
    print("  A CUSIP counts as \"owned\" (and is included in --daily) if it has a")
    print("  _comment entry starting with \"paid\" -- the same marker interactive_add()")
    print("  uses to show \"(owned)\" in the CUSIP picker.")

    print(f"\n{BOLD}{CYAN}LAUNCHD{RESET}  (scheduled daily runs)")
    print("  --daily is meant to run unattended once a day via a macOS LaunchAgent")
    print("  rather than cron -- launchd catches up a run that was missed because the")
    print("  Mac was asleep or off, which plain cron does not.")
    print(f"  Set it up with: {YELLOW}python3 setup_daily_fetch_launchd.py{RESET}")
    print()
    print(f"  {BOLD}Plist location:{RESET}  {LAUNCHD_PLIST_PATH}")
    print(f"  {BOLD}Log file:{RESET}        {DAILY_LOG_FILE}")
    print(f"  {BOLD}Label:{RESET}           {LAUNCHD_LABEL}")
    print()
    print(f"  {BOLD}Check it's loaded:{RESET}")
    print(f"    launchctl list | grep {LAUNCHD_LABEL}")
    print(f"  {BOLD}Trigger a run right now (for testing):{RESET}")
    print(f"    launchctl kickstart -p gui/$(id -u)/{LAUNCHD_LABEL}")
    print(f"  {BOLD}Tail the log:{RESET}")
    print(f"    tail -f {DAILY_LOG_FILE}")
    print(f"  {BOLD}Disable temporarily{RESET} (stays installed, won't fire):")
    print(f"    launchctl bootout gui/$(id -u)/{LAUNCHD_LABEL}")
    print(f"  {BOLD}Re-enable after disabling:{RESET}")
    print(f"    launchctl bootstrap gui/$(id -u) {LAUNCHD_PLIST_PATH}")
    print(f"  {BOLD}Change the schedule:{RESET} edit SCHEDULE_HOUR/SCHEDULE_MINUTE at the top of")
    print("    setup_daily_fetch_launchd.py, then re-run it -- don't hand-edit the plist,")
    print("    the next re-run of that script will silently overwrite it.")
    print(f"  {BOLD}Remove entirely:{RESET} bootout, then delete the plist file.")

    print(f"\n{BOLD}{CYAN}NOTES{RESET}")
    print("  - Interactive modes show every entry for review before saving -- nothing is")
    print("    written to the JSON file without an explicit y/N confirmation. --daily")
    print("    saves automatically, but ONLY when the coupon was independently verified")
    print("    against Treasury's own auction record; anything unverified, mismatched,")
    print("    or fetch-failed is skipped and listed in the run summary for you to")
    print("    review by hand (e.g. bond_market_analyzer.py --CUSIP <cusip>).")
    print("  - Duplicate entries for the same CUSIP + date are detected and skipped.")
    print()

def main():
    """Parse CLI args and dispatch to the selected mode: fetch (default), --manual, --copypaste, or --getinfo."""
    parser = argparse.ArgumentParser(description="Bond Market Analyzer", add_help=False)
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--CUSIP", type=str, help="CUSIP identifier")
    parser.add_argument("--getinfo", action="store_true")
    parser.add_argument("--manual", action="store_true", help="Use guided manual entry mode (previously the default) instead of fetching a price.")
    parser.add_argument("--copypaste", action="store_true", help="Use copy/paste mode for data entry instead of the default fetch mode.")
    parser.add_argument("--source", choices=["webull", "finra"], default="webull", help="Price data source for the default fetch mode (default: webull). finra is dormant -- see finra_bond_fetcher.py.")
    parser.add_argument("--daily", action="store_true", help="Non-interactively fetch and save today's price for every owned CUSIP. For scheduled/unattended runs -- see the LAUNCHD section of --help.")
    args = parser.parse_args()

    if args.version:
        print(f"Bond Market Analyzer version {VERSION}")
        return
    if args.help:
        print_help()
        return

    try:
        data = load_data()

        if args.getinfo:
            cusip = args.CUSIP or input("Enter CUSIP: ").strip()
            show_info(cusip, data)
            return

        if args.manual:
            interactive_add(data)
            return

        if args.daily:
            fetch_all_owned(data, source=args.source)
            return

        if not args.copypaste:
            # Default mode: fetch price/yield from the configured source (--source, default webull)
            fetch_add(data, args.CUSIP, source=args.source)
            return

        # --copypaste mode
        cusip = args.CUSIP or input("Enter CUSIP: ").strip().upper()
        if not cusip:
            print("Error: CUSIP cannot be empty")
            return
            
        entries = parse_input()
        if not entries:
            print("No valid entries found.")
            return
            
        if cusip not in data:
            data[cusip] = []
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if we already have an entry for this CUSIP for today
        entry_exists = False
        for existing_entry in data[cusip]:
            if existing_entry.get("Date") == today:
                entry_exists = True
                print(f"Entry for CUSIP {cusip} on {today} already exists. Not adding duplicate.")
                
                # Show the existing entry
                print("\nExisting entry:")
                print(f"Coupon: {existing_entry['Coupon']}, Maturity: {existing_entry['Maturity']}, "
                      f"Price: {existing_entry['Price']}, YTW: {existing_entry['YTW']}, Date: {existing_entry['Date']}")
                # Also show price comparison summary
                print("\nSummary for CUSIP:")
                show_info(cusip, data)
                break
        
        # Only add new entries if no entry exists for today
        if not entry_exists:
            # Add the new entries to the data
            data[cusip].extend(entries)
            save_data(data)
            print(f"Saved {len(entries)} entries for CUSIP {cusip}.")

            # Show only the information just entered, with correct labels
            print("\nEntry added:")
            for entry in entries:
                print(f"Coupon: {entry['Coupon']}, Maturity: {entry['Maturity']}, "
                      f"Price: {entry['Price']}, YTW: {entry['YTW']}, Date: {entry['Date']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()