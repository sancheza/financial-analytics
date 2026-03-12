#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

VERSION = "1.0.1"
# Updated scraping logic to be more resilient to Yahoo Finance page changes.

def get_earnings_date(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    raw_value = None

    # Method 1: Use the `data-test` attribute, which is the most stable selector.
    earnings_element = soup.find(attrs={"data-test": "EARNINGS_DATE-value"})
    if earnings_element:
        spans = earnings_element.find_all("span")
        if len(spans) > 1:
            raw_value = spans[1].text.strip()

    # Method 2 (Fallback): Find the label by its text content if the first method fails.
    if not raw_value or raw_value in ["", "--", "-"]:
        label_span = soup.find("span", string="Earnings Date")
        if label_span:
            # The value is usually in the next sibling span.
            value_span = label_span.find_next_sibling("span")
            if value_span:
                raw_value = value_span.text.strip()

    # Process the extracted value
    if raw_value:
        if raw_value == "--":
            return "N/A"
        if raw_value == "-":
            return ""
        return raw_value

    # If no value was found by any method, return an empty string.
    return ""

def print_help():
    help_text = """
Usage: python get_earnings_date.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol to fetch the Earnings Date information for
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
        print(f"get_earnings_date.py version {VERSION}")
        sys.exit(0)

    ticker = sys.argv[1].upper()
    result = get_earnings_date(ticker)
    if result == "":
        print("")  # Output an empty string if no valid Earnings Date is found
    elif result == "N/A":
        print("N/A")  # Output "N/A" if the source returns "--"
    else:
        print(result)
