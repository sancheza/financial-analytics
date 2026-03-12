#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bond Alert - Monitor Treasury Bond Yields and Send Email Alerts

This script monitors various US Treasury bond yields (10Y, 20Y, 30Y bonds and 10Y TIPS)
using the FRED API. It checks if any yields exceed configured thresholds and sends
email alerts when thresholds are breached.

Author: sancheza
License: MIT
Version: 1.0.2
"""

import requests
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import argparse
from typing import Dict, Tuple, Optional
import sys
from datetime import datetime

VERSION = "1.0.2"

# ANSI escape codes for formatting
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[33m"
RED = "\033[91m"
RESET = "\033[0m"

# Thresholds for different bond types (in %)
THRESHOLDS = {
    "2Y_NOTE":  4.00,    # 2-year Treasury Note
    "10Y_BILL": 4.50,    # 10-year Treasury Bill
    "20Y_BOND": 5.00,    # 20-year Treasury Bond
    "30Y_BOND": 5.10,    # 30-year Treasury Bond
    "10Y_TIPS": 2.30     # 10-year TIPS
}

# FRED Series IDs for different bonds
SERIES_IDS = {
    "2Y_NOTE":  "DGS2",     # 2-year Treasury Constant Maturity Rate
    "10Y_BILL": "DGS10",    # 10-year Treasury Constant Maturity Rate
    "20Y_BOND": "DGS20",    # 20-year Treasury Constant Maturity Rate
    "30Y_BOND": "DGS30",    # 30-year Treasury Constant Maturity Rate
    "10Y_TIPS": "DFII10"    # 10-year Treasury Inflation-Indexed Security
}

DESCRIPTION = f"""
{GREEN}{BOLD}Bond Alert version {VERSION}{RESET}

{CYAN}{BOLD}OVERVIEW:{RESET}
  This script monitors various U.S. Treasury bond yields using data from the
  Federal Reserve Economic Data (FRED) API. It checks the latest yields for
  key securities against user-defined thresholds and sends an email alert if
  any of those thresholds are breached.

  The script requires a {BOLD}.env{RESET} file with your FRED API key and email credentials.

{CYAN}{BOLD}DATA SOURCE & YIELD EXPLANATION:{RESET}
  The yields retrieved by this script are {BOLD}Constant Maturity Treasury (CMT){RESET} rates from the
  Federal Reserve Economic Data (FRED) service.

  {BOLD}What does the yield represent?{RESET}
  The yield for a given date is derived from the {BOLD}secondary market{RESET}. It is read from a
  yield curve constructed daily by the U.S. Treasury based on the closing market bid
  yields of actively traded Treasury securities. This provides a consistent measure of
  the market yield for a specific maturity (e.g., exactly 10 years), even if no
  single security has that exact maturity.

  This is different from an auction yield, which is the result of a specific, primary
  auction event. This script provides a daily snapshot of general market sentiment.
"""

# Generate strings for help text using the default values before they are potentially overridden
default_thresholds_str = "; ".join([f"{THRESHOLDS[key]:.2f}%" for key in SERIES_IDS.keys()])
default_order_str = ", ".join(SERIES_IDS.keys())

EPILOG = f"""
{CYAN}{BOLD}USAGE:{RESET}
  {GREEN}# Run the monitor to check current yields against thresholds{RESET}
    {YELLOW}python bond_alert.py{RESET}

  {GREEN}# Display the current threshold settings from the script{RESET}
    {YELLOW}python bond_alert.py --showsettings{RESET}

{CYAN}{BOLD}SETUP:{RESET}
  Create a {BOLD}.env{RESET} file in the same directory with the following content:
    {YELLOW}FRED_API_KEY="your_fred_api_key"{RESET}
    {YELLOW}EMAIL_FROM="your_email@example.com"{RESET}
    {YELLOW}EMAIL_TO="recipient_email@example.com"{RESET}
    {YELLOW}SMTP_USER="your_smtp_username_or_email"{RESET}
    {YELLOW}SMTP_PASS="your_smtp_password_or_app_password"{RESET}

  {CYAN}{BOLD}OPTIONAL THRESHOLD OVERRIDE:{RESET}
    You can override the default yield thresholds by setting the {BOLD}YIELD_THRESHOLD{RESET}
    environment variable. It should be a semicolon-separated list of yields.

    Example:
      {YELLOW}YIELD_THRESHOLD="4.1%;4.6%;5.1%;5.2%;2.4%"{RESET}

    The order must match the internal bond order: {default_order_str}
    Default values: {default_thresholds_str}
"""

# Load API key and email creds from .env
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
YIELD_THRESHOLD_ENV = os.getenv("YIELD_THRESHOLD")

# Check for environment variable override for thresholds
if YIELD_THRESHOLD_ENV:
    try:
        # Split by semicolon, remove percentage signs, and filter out empty strings
        yield_values_str = [v.strip().replace('%', '') for v in YIELD_THRESHOLD_ENV.split(';')]
        yield_values_float = [float(v) for v in yield_values_str if v]
        threshold_keys = list(THRESHOLDS.keys())
        if len(yield_values_float) == len(threshold_keys):
            for i, key in enumerate(threshold_keys):
                THRESHOLDS[key] = yield_values_float[i]
            print(f"{CYAN}Using custom thresholds from YIELD_THRESHOLD environment variable.{RESET}")
        else:
            print(f"{YELLOW}Warning: YIELD_THRESHOLD variable has an incorrect number of values ({len(yield_values_float)} instead of {len(threshold_keys)}). Using defaults.{RESET}")
    except (ValueError, TypeError) as e:
        print(f"{YELLOW}Warning: Could not parse YIELD_THRESHOLD environment variable: {e}. Using defaults.{RESET}")

def get_yield(series_id: str) -> Tuple[str, float]:
    """
    Get the latest yield value for a given series ID from FRED.

    Args:
        series_id (str): The FRED series ID for the security.

    Returns:
        Tuple[str, float]: A tuple containing the date (YYYY-MM-DD) and the
                           latest yield value as a percentage.

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
        KeyError: If the response JSON is not in the expected format.
    """
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
    latest_observation = data['observations'][0]
    return latest_observation['date'], float(latest_observation['value'])

def check_all_yields() -> Dict[str, Tuple[str, float, bool]]:
    """
    Check all configured bond yields against their thresholds.

    Returns:
        Dict[str, Tuple[str, float, bool]]: A dictionary where keys are bond types and
                                            values are tuples of (yield_date, current_yield, is_above_threshold).
    """
    results = {}
    for bond_type, series_id in SERIES_IDS.items():
        try:
            yield_date, current_yield = get_yield(series_id)
            threshold = THRESHOLDS[bond_type]
            is_above_threshold = current_yield >= threshold
            results[bond_type] = (yield_date, current_yield, is_above_threshold)

            # Determine color based on comparison to threshold
            if current_yield > threshold:
                color = GREEN
            elif current_yield < threshold:
                color = RED
            else:  # Exactly matches
                color = YELLOW
            print(f"{color}{bond_type}: Yield as of {yield_date} = {current_yield:.2f}% (Threshold: {threshold:.2f}%){RESET}")
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"{RED}Could not retrieve yield for {bond_type}: {e}{RESET}")
    return results

def send_email_alert(alerts: Dict[str, Tuple[str, float, bool]]) -> None:
    """
    Send an email alert for bonds that have exceeded their thresholds.

    Args:
        alerts (Dict[str, Tuple[str, float, bool]]): The results from check_all_yields.
    """
    # Exceeded bonds will be a dict of {bond_type: (date, yield)}
    exceeded_bonds = {k: (v[0], v[1]) for k, v in alerts.items() if v[2]}
    if not exceeded_bonds:
        return

    body = "The following bonds have exceeded their thresholds:\n\n"
    for bond_type, (yield_date, yield_value) in exceeded_bonds.items():
        body += f"{bond_type} on {yield_date}: {yield_value:.2f}% (Threshold: {THRESHOLDS[bond_type]:.2f}%)\n"

    msg = MIMEText(body)
    msg['Subject'] = f"Alert: Bond Yield Thresholds Exceeded"
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

def main() -> None:
    """
    Parses command-line arguments and runs the bond yield check.
    """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--version', action='version', version=f'Bond Alert version {VERSION}')
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
    if any(alert[2] for alert in alerts.values()):
        send_email_alert(alerts)

if __name__ == "__main__":
    main()
