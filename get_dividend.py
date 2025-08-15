#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

VERSION = "1.0.2"

def get_dividend(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all list items that might contain the Forward Dividend Yield value
    for li in soup.find_all("li", class_="yf-1hwifs8"):
        label = li.find("span", class_="label")
        value = li.find("span", class_="value")
        if label and value and "Forward Dividend & Yield" in label.text:
            raw_value = value.text.strip()
            # If the value is "--", return "N/A"
            if raw_value == "--":
                return "N/A"
            # If the value is empty or invalid, return an empty string
            if not raw_value or raw_value == "-":
                return ""
            # Extract the part inside parentheses and remove them
            if "(" in raw_value and ")" in raw_value:
                yield_value = raw_value.split("(")[1].split(")")[0]
                return yield_value.strip()
            return ""  # Fallback if the value is not in the expected format

    # If no "Forward Dividend & Yield" entry is found, return an empty string
    return ""

def print_help():
        help_text = """
get_dividend.py - Fetch Forward Dividend Yield from Yahoo Finance

Overview:
    This script retrieves the Forward Dividend Yield for a given stock ticker from Yahoo Finance.
    It is designed for fast, simple, and reliable extraction of dividend yield data for use in
    screening, analysis, or automation workflows.

Objectives:
    - Fetch the Forward Dividend Yield for a specified stock ticker
    - Output the yield value (as a percentage) or "N/A" if unavailable
    - Provide a minimal, scriptable interface for integration with other tools

Usage:
    python get_dividend.py TICKER
        Fetches the Forward Dividend Yield for the given ticker symbol.

Options:
    -h, --help      Show this help message and exit
    -v, --version   Show the version of the script and exit

Arguments:
    TICKER          The stock ticker symbol to fetch the dividend information for

Examples:
    python get_dividend.py AAPL
    python get_dividend.py --help
    python get_dividend.py --version
"""
        print(help_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("")  # Output an empty string for invalid usage
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg in ("-h", "--help"):
        print_help()
        sys.exit(0)
    elif arg in ("-v", "--version"):
        print(f"get_dividend.py version {VERSION}")
        sys.exit(0)

    ticker = sys.argv[1].upper()
    result = get_dividend(ticker)
    if result == "":
        print("")  # Output an empty string if no valid dividend yield is found
    elif result == "N/A":
        print("N/A")  # Output "N/A" if the source returns "--"
    else:
        print(result)

