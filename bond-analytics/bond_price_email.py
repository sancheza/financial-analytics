#!/usr/bin/env python3
"""Fetch live prices for a set of Treasury CUSIPs and email them to you.

Designed to be run from crontab twice a day local time: at 08:15 it reports each
CUSIP's price shortly after the market open (a few minutes after 08:00, once opening
trades have printed), and at 17:15 it reports the day's last price (market closes
at 17:00).

Configuration lives in this directory's .env file:

    CUSIP=91282CMM0,912810UF3,...   comma-separated list to report
    GMAIL_USER=you@gmail.com         Gmail address used to log in and send
    GMAIL_PASS=...                   Gmail app password (not the login password)
    EMAIL_TO=you@gmail.com           optional; defaults to GMAIL_USER

Data source, in priority order per CUSIP:
  1. Webull's live quote (webull_bond_fetcher.fetch_treasury_price) -- price
     is the real thing, timestamped to the second. This is the normal path.
  2. If Webull shows a YTW/reference data but no price, the price is estimated
     from that YTW with bond_return_calc.price_from_yield and flagged ESTIMATE.
  3. If Webull has no listing at all, FINRA's last TRACE-reported trade is used
     as a (potentially stale) reference price. If FINRA has a YTW but no price,
     that yields an ESTIMATE too.
A CUSIP with no price from any source is listed as unpriced in the email rather
than silently dropped.

Each email line reports price, YTW (yield to worst), and maturity (MM/YY); a single
"as of <time> (Source: ...)" footer -- the most recent timestamp among the CUSIPs --
follows at the end. Run log (one line per CUSIP plus a send line) is appended to
logs/bond_price_email.log. For cron, redirect stdout/stderr to /dev/null and let
the script's own log be the record.

CRONTAB (crontab -e):

    SHELL=/bin/bash
    15 8,17 * * * python3 /path/to/bond_price_email.py >/path/to/bond_price_email.log 2>&1

(replace /absolute/path/to with the directory that contains this script.) The script
uses absolute paths internally, so the cron entry does not depend on cron's working
directory or a non-default PATH.
"""

import argparse
import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText

from dotenv import load_dotenv

import webull_bond_fetcher
import finra_bond_fetcher
from bond_return_calc import price_from_yield

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "bond_price_email.log")

VERSION = "1.0.1"

# Colors for the help screen -- disabled when stdout isn't a terminal (e.g. output
# is redirected to /dev/null by cron), so logs stay plain, readable text.
_USE_COLOR = sys.stdout.isatty()
CYAN = "\033[96m" if _USE_COLOR else ""
GREEN = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
REQUEST_TIMEOUT = 10


def _as_float(value) -> float | None:
    """Convert a numeric-typed string/float to float, or None if not a real number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(utc_timestamp: str | None) -> datetime | None:
    """Parse Webull's UTC timestamp ("2026-09-11T20:59:47.020+0000") into an aware datetime, or None."""
    if not utc_timestamp:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(utc_timestamp, fmt)
        except ValueError:
            continue
    return None


def _local_time_string(utc_timestamp: str) -> str:
    """Return a local-time string for Webull's UTC timestamp, or the raw value if unparseable."""
    parsed = _parse_timestamp(utc_timestamp)
    if parsed is None:
        return utc_timestamp if utc_timestamp else ""
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD date (FINRA's tradeDate) into a datetime, or None."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _format_maturity(maturity_date: str | None) -> str:
    """Format a YYYY-MM-DD maturity date as MM/YY ('2035-02-15' -> '02/35'), or '' if absent/unparseable."""
    if not maturity_date:
        return ""
    try:
        return datetime.strptime(maturity_date, "%Y-%m-%d").strftime("%m/%y")
    except ValueError:
        return maturity_date


def _estimate_price(coupon_rate: str | float, maturity_date: str, ytw_pct: float | None) -> float | None:
    """Estimate clean price from a yield using bond_return_calc.price_from_yield."""
    if ytw_pct is None:
        return None
    try:
        maturity = datetime.strptime(maturity_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    coupon = _as_float(coupon_rate)
    if coupon is None:
        return None
    return price_from_yield(ytw_pct, coupon, maturity)


def resolve_price(cusip: str) -> dict:
    """Fetch the best available price for one CUSIP.

    Returns a dict describing the result:
      price: float or None
      ytw_pct: float or None
      maturity: MM/YY string (e.g. "02/35") or "" if unknown
      time: display "as of" string or None
      time_dt: epoch-seconds float for the same timestamp, or None (for choosing the
               most recent across CUSIPs); None when unknown
      source: "Webull" or "FINRA" (the data source) or ""
      kind: "live" (Webull last), "mid" (Webull bid/ask mid), "estimate",
            "stale" (FINRA last trade), or "none"
      note: human-readable annotation (data source, ESTIMATE flag, staleness warning).
    """
    note = ""
    try:
        webull = webull_bond_fetcher.fetch_treasury_price(cusip, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        webull = None
        note = f"Webull fetch failed ({e}); "
    if webull:
        price = _as_float(webull.get("price"))
        ytw_pct = _as_float(webull.get("ytw") or webull.get("ytm") or webull.get("yield"))
        maturity = _format_maturity(webull.get("maturityDate"))
        time_str = _local_time_string(webull.get("tradeTime"))
        parsed_time = _parse_timestamp(webull.get("tradeTime"))
        time_dt = parsed_time.timestamp() if parsed_time else None
        if price is not None:
            return {"cusip": cusip, "price": price, "ytw_pct": ytw_pct, "maturity": maturity,
                    "time": time_str, "time_dt": time_dt, "source": "Webull", "kind": "live",
                    "note": "Source: Webull"}
        bid = _as_float(webull.get("bidPrice"))
        ask = _as_float(webull.get("askPrice"))
        if bid is not None and ask is not None:
            return {"cusip": cusip, "price": round((bid + ask) / 2.0, 6), "ytw_pct": ytw_pct,
                    "maturity": maturity, "time": time_str, "time_dt": time_dt, "source": "Webull",
                    "kind": "mid", "note": f"Source: Webull; bid/ask mid (bid {bid}, ask {ask})"}
        estimated = _estimate_price(webull.get("couponRate"), webull.get("maturityDate"), ytw_pct)
        if estimated is not None:
            return {"cusip": cusip, "price": estimated, "ytw_pct": ytw_pct, "maturity": maturity,
                    "time": time_str, "time_dt": time_dt, "source": "Webull", "kind": "estimate",
                    "note": "Source: Webull; ESTIMATE — no live price; computed from yield"}
        note += "Webull listed CUSIP but gave no price or yield-based price; "

    try:
        finra = finra_bond_fetcher.fetch_treasury_price(cusip, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        finra = None
        note += f"FINRA fetch failed ({e}); "
    if finra:
        price = _as_float(finra.get("price"))
        ytw_pct = _as_float(finra.get("yield"))
        maturity = _format_maturity(finra.get("maturityDate"))
        trade_date = finra.get("tradeDate")
        parsed_date = _parse_date(trade_date)
        time_dt = parsed_date.timestamp() if parsed_date else None
        if price is not None:
            return {"cusip": cusip, "price": price, "ytw_pct": ytw_pct, "maturity": maturity,
                    "time": trade_date or "", "time_dt": time_dt, "source": "FINRA", "kind": "stale",
                    "note": f"Source: FINRA; last TRACE trade (may be stale; reported {trade_date or 'unknown date'})"}
        estimated = _estimate_price(finra.get("couponRate"), finra.get("maturityDate"), ytw_pct)
        if estimated is not None:
            return {"cusip": cusip, "price": estimated, "ytw_pct": ytw_pct, "maturity": maturity,
                    "time": trade_date or "", "time_dt": time_dt, "source": "FINRA", "kind": "estimate",
                    "note": "Source: FINRA; ESTIMATE — no TRACE price; computed from trade yield"}
        note += f"FINRA has reference data but no trade for {cusip}; "

    return {"cusip": cusip, "price": None, "ytw_pct": None, "maturity": "",
            "time": None, "time_dt": None, "source": "", "kind": "none",
            "note": (note or "No price available from any source").rstrip("; ")}


def _now_local() -> datetime:
    return datetime.now().astimezone()


def build_body(records: list[dict]) -> str:
    lines = [f"Bond prices fetched {_now_local().strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    newest = None
    for r in records:
        maturity_part = f" ({r['maturity']})" if r["maturity"] else ""
        if r["price"] is None:
            lines.append(f"{r['cusip']}{maturity_part}: no price available ({r['note']})")
            continue
        ytw_part = f"  YTW: {r['ytw_pct']:.3f}%" if r["ytw_pct"] is not None else ""
        label = "Est. price" if r["kind"] == "estimate" else "price"
        stale_marker = "  (last TRACE trade)" if r["kind"] == "stale" else ""
        lines.append(f"{r['cusip']}{maturity_part}  {label}: {r['price']:.3f}{ytw_part}{stale_marker}")
        if r["time_dt"] is not None and (newest is None or r["time_dt"] > newest[0]):
            newest = (r["time_dt"], r["time"], r["source"])
    if newest:
        lines.append("")
        lines.append(f"as of {newest[1]} (Source: {newest[2]})")
    return "\n".join(lines)


def send_email(records: list[dict]) -> None:
    user = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASS")
    to_addr = os.getenv("EMAIL_TO") or user
    if not user or not password:
        raise RuntimeError("Missing GMAIL_USER/GMAIL_PASS in .env; cannot send email.")

    body = build_body(records)
    msg = MIMEText(body)
    msg["Subject"] = f"Bond prices {_now_local().strftime('%Y-%m-%d %H:%M')}"
    msg["From"] = user
    msg["To"] = to_addr

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=REQUEST_TIMEOUT) as server:
        server.login(user, password)
        server.send_message(msg)


def log(message: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{_now_local().strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def print_help():
    """Print a formatted help screen: usage, crontab setup, .env config, data sources, and options."""
    script_name = os.path.basename(sys.argv[0])
    rule = f"{CYAN}{'-' * 66}{RESET}"

    def opt(flag, desc):
        print(f"  {BOLD}{flag:<16}{RESET}{desc}")

    print(f"\n{rule}")
    print(f"{BOLD}{GREEN}  Bond Price Email  {RESET}{CYAN}v{VERSION}{RESET}")
    print(rule)
    print("  Fetches a live price for each CUSIP in .env and emails it to you.")
    print("  Made to run from crontab shortly after the market open and after the close.")

    print(f"\n{BOLD}{CYAN}USAGE{RESET}")
    print(f"  {script_name} [OPTIONS]")
    print(f"  {BOLD}{script_name}{RESET}  fetches prices and emails them (no args = run).")

    print(f"\n{BOLD}{CYAN}CRONTAB{RESET}  (crontab -e)")
    print("  15 8,17 * * * /opt/homebrew/bin/python3 /absolute/path/to/{0} >/dev/null 2>&1".format(script_name))
    print("  Replace /absolute/path/to with the directory containing this script. Runs at")
    print("  08:15 (a few minutes after the 08:00 open, once opening trades have printed)")
    print("  and 17:15 (after the 17:00 close, showing the day's last print). The script")
    print("  uses absolute paths internally, so it does not depend on cron's working directory.")
    print("  Run log: logs/bond_price_email.log")

    print(f"\n{BOLD}{CYAN}CONFIG{RESET}  (this directory's .env)")
    print("  CUSIP=91282CMM0,912810UF3,...   comma-separated list to report")
    print("  GMAIL_USER=you@gmail.com         Gmail address used to log in and send")
    print("  GMAIL_PASS=...                   Gmail app password (not the login password)")
    print("  EMAIL_TO=you@gmail.com           optional; defaults to GMAIL_USER")

    print(f"\n{BOLD}{CYAN}DATA SOURCES{RESET}  (per CUSIP, in priority order)")
    print(f"  {GREEN}webull{RESET} (primary)    Live quote -- real price, timestamped to the second.")
    print("               No login or API key needed; same fetcher as bond_market_analyzer.py.")
    print(f"  {YELLOW}finra{RESET}   (fallback)   Last TRACE-reported trade; can be stale for thinly-")
    print("               traded CUSIPs (see finra_bond_fetcher.py).")
    print("  estimate   If a source has a YTW but no price, price is computed from that YTW")
    print("             with bond_return_calc.price_from_yield and flagged ESTIMATE.")
    print("  Each line reports price, YTW, and maturity (MM/YY). A single 'as of <time>")
    print("  (Source: ...)' footer -- the most recent timestamp -- follows at the end.")
    print("  A CUSIP with no price from any source is listed as unpriced.")

    print(f"\n{BOLD}{CYAN}OPTIONS{RESET}")
    opt("-h, --help", "Show this help message and exit.")
    opt("-v, --version", "Show script version and exit.")
    print()

def main():
    """Parse CLI args (--version/--help), then fetch prices for the configured CUSIPs and email them."""
    parser = argparse.ArgumentParser(description="Bond Price Email", add_help=False)
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(f"Bond Price Email version {VERSION}")
        return 0
    if args.help:
        print_help()
        return 0

    load_dotenv(ENV_FILE)

    raw_cusips = os.getenv("CUSIP", "")
    cusips = [c.strip().upper() for c in raw_cusips.split(",") if c.strip()]
    if not cusips:
        log(f"ERROR: no CUSIPs configured (set CUSIP=x,y,z in {ENV_FILE})")
        print(f"ERROR: no CUSIPs configured (set CUSIP=x,y,z in {ENV_FILE})", file=sys.stderr)
        return 1

    records = []
    for cusip in cusips:
        record = resolve_price(cusip)
        records.append(record)
        log(f"{cusip}: {record['kind']} price={record['price']} ytw={record['ytw_pct']} note='{record['note']}'")
        print(f"{cusip}: {record['kind']} price={record['price']} ytw={record['ytw_pct']} note='{record['note']}'")

    try:
        send_email(records)
    except Exception as e:
        log(f"ERROR: email send failed: {e}")
        print(f"ERROR: email send failed: {e}", file=sys.stderr)
        return 1

    log(f"Email sent to {os.getenv('EMAIL_TO') or os.getenv('GMAIL_USER')}")
    print(f"Email sent to {os.getenv('EMAIL_TO') or os.getenv('GMAIL_USER')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())