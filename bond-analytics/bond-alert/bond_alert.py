#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bond Alert - Monitor Treasury Bond Yields and Send Email Alerts

This script monitors various US Treasury bond yields (10Y, 20Y, 30Y bonds and 10Y TIPS)
using the FRED API. It checks if any yields exceed configured thresholds and sends
email alerts when thresholds are breached.

Author: Anonymous
License: MIT
Version: 1.0.0
"""

import requests
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import argparse
from typing import Dict, Tuple
import sys
from datetime import datetime

VERSION = "1.0"

# Thresholds for different bond types (in %)
THRESHOLDS = {
    "2Y_NOTE":  4.00,    # 2-year Treasury Note
    "10Y_BILL": 4.55,    # 10-year Treasury Bill
    "20Y_BOND": 4.95,    # 20-year Treasury Bond
    "30Y_BOND": 5.00,    # 30-year Treasury Bond
    "10Y_TIPS": 2.30     # 10-year TIPS (buy at 2.5% and load up at 3.0%)
}

# FRED Series IDs for different bonds
SERIES_IDS = {
    "2Y_NOTE":  "DGS2",     # 2-year Treasury Constant Maturity Rate
    "10Y_BILL": "DGS10",    # 10-year Treasury Constant Maturity Rate
    "20Y_BOND": "DGS20",    # 20-year Treasury Constant Maturity Rate
    "30Y_BOND": "DGS30",    # 30-year Treasury Constant Maturity Rate
    "10Y_TIPS": "DFII10"    # 10-year Treasury Inflation-Indexed Security
}

# Load API key and email creds from .env
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

DURATION_LABELS = {
    "2Y_NOTE":  "2-Year",     # 2-year Treasury Note
    "10Y_BILL": "10-Year",    # 10-year Treasury Bill
    "20Y_BOND": "20-Year",    # 20-year Treasury Bond
    "30Y_BOND": "30-Year",    # 30-year Treasury Bond
    "10Y_TIPS": "10-Year"     # 10-year TIPS
}

HELP_TEXT = """
get_bond_yield.py - Retrieve auction yield for a given bond duration and date
Examples:
  ./get_bond_yield.py --duration 20Y               # Most recent 20Y auction
  ./get_bond_yield.py --duration 10Y --date 06012024
  ./get_bond_yield.py -h
  ./get_bond_yield.py -v

Note: Yield is retrieved from Treasury auction results via TreasuryDirect.gov
"""

API_URL = "https://www.treasurydirect.gov/auctions/xml/td_xml_security_data.xml"

def fetch_auction_yield(security_term, target_date):
    try:
        r = requests.get(API_URL)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"URL attempted: {e.response.url}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Treasury API: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing API response: {e}")
        sys.exit(1)

    # Filter auctions for the requested security type
    auctions = [
        auction for auction in data
        if auction.get("securityTerm") == security_term and auction.get("highYield")
    ]

    if not auctions:
        return None

    # Parse auction dates and find closest
    for auction in auctions:
        auction_date_str = auction.get("auctionDate")
        if auction_date_str:
            try:
                auction["auction_date"] = datetime.strptime(auction_date_str, "%Y-%m-%dT%H:%M:%S").date()
            except ValueError:
                continue

    # Filter out auctions without valid dates
    auctions = [a for a in auctions if "auction_date" in a]
    if not auctions:
        return None

    closest = min(auctions, key=lambda x: abs(x["auction_date"] - target_date.date()))
    return closest["auction_date"], float(closest["highYield"])

def get_yield(series_id: str) -> float:
    """Get the latest yield value for a given series ID from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    return float(data['observations'][0]['value'])

def check_all_yields() -> Dict[str, Tuple[float, bool]]:
    """Check all bond yields and return their values and threshold status."""
    results = {}
    for bond_type, series_id in SERIES_IDS.items():
        current_yield = get_yield(series_id)
        is_above_threshold = current_yield >= THRESHOLDS[bond_type]
        results[bond_type] = (current_yield, is_above_threshold)
        print(f"{bond_type}: Current yield = {current_yield:.2f}% (Threshold: {THRESHOLDS[bond_type]:.2f}%)")
    return results

def send_email_alert(alerts: Dict[str, Tuple[float, bool]]):
    """Send email alert for bonds that exceeded their thresholds."""
    exceeded_bonds = {k: v[0] for k, v in alerts.items() if v[1]}
    if not exceeded_bonds:
        return

    body = "The following bonds have exceeded their thresholds:\n\n"
    for bond_type, yield_value in exceeded_bonds.items():
        body += f"{bond_type}: {yield_value:.2f}% (Threshold: {THRESHOLDS[bond_type]:.2f}%)\n"

    msg = MIMEText(body)
    msg['Subject'] = f"Alert: Bond Yield Thresholds Exceeded"
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

def main():
    parser = argparse.ArgumentParser(description='Monitor bond yields and send alerts when thresholds are exceeded.')
    parser.add_argument('-v', '--version', action='version', version=f'Bond Alert v{VERSION}')
    parser.add_argument('--showsettings', action='store_true', help='Display current threshold settings')
    args = parser.parse_args()

    if args.showsettings:
        print("\nCurrent Threshold Settings:")
        print("=" * 50)
        for bond_type, threshold in THRESHOLDS.items():
            print(f"{bond_type:<10}: {threshold:>5.2f}% (Series ID: {SERIES_IDS[bond_type]})")
        print("=" * 50)
        return

    alerts = check_all_yields()
    if any(alert[1] for alert in alerts.values()):
        send_email_alert(alerts)

if __name__ == "__main__":
    main()
