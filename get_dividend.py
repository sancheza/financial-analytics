#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

VERSION = "1.0.1"

def get_dividend(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all list items that might contain the Forward Dividend Yield value
    for li in soup.find_all("li", class_="yf-1jj98ts"):
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
Usage: python get_dividend.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol to fetch the dividend information for
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

