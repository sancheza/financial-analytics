#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

VERSION = "1.0.3"

def get_pb(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the Price/Book row in the table
    # Look for the cell containing "Price/Book" text
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            # Check if the first cell contains "Price/Book"
            if cells[0] and "Price/Book" in cells[0].get_text():
                # Get the value from the second cell
                raw_value = cells[1].get_text().strip()
                # If the value is "--", return "N/A"
                if raw_value == "--":
                    return "N/A"
                # If the value is empty or invalid, return "N/A"
                if not raw_value or raw_value == "-":
                    return "N/A"
                return raw_value

    # If no "Price/Book" entry is found, return "N/A"
    return "N/A"

def print_help():
    help_text = """
get_pb.py - Fetch Price-to-Book ratio for stock tickers from Yahoo Finance

This script retrieves the Price-to-Book (P/B) ratio for a given stock ticker
by scraping the key statistics page from Yahoo Finance. The P/B ratio compares
a company's market value to its book value, helping investors assess whether
a stock is undervalued or overvalued.

Usage: python get_pb.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol for which to fetch price-to-book ratio

Examples:
  python get_pb.py AAPL     # Get P/B ratio for Apple Inc.
  python get_pb.py MSFT     # Get P/B ratio for Microsoft
  python get_pb.py --help   # Show this help message

Output:
  Returns the P/B ratio as a number (e.g., "1.23") or "N/A" if the data
  is unavailable or the ticker is invalid.
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("N/A")  # Output "N/A" for invalid usage
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg in ("-h", "--help"):
        print_help()
        sys.exit(0)
    elif arg in ("-v", "--version"):
        print(f"get_pb.py version {VERSION}")
        sys.exit(0)

    ticker = sys.argv[1].upper()
    result = get_pb(ticker)
    print(result)  # Will always be either a valid P/B value or "N/A"