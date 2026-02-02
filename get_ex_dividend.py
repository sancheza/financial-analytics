#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

VERSION = "1.0.2"
# Updated scraping logic to be more resilient to Yahoo Finance page changes.


def get_ex_dividend(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    raw_value = None

    # Method 1: Use the `data-test` attribute, which is the most stable selector.
    ex_div_element = soup.find(attrs={"data-test": "EX_DIVIDEND_DATE-value"})
    if ex_div_element:
        spans = ex_div_element.find_all("span")
        if len(spans) > 1:
            raw_value = spans[1].text.strip()

    # Method 2 (Fallback): Find the label by its text content if the first method fails.
    if not raw_value or raw_value in ["", "--", "-"]:
        # Search for text containing "Ex-Dividend Date" to handle <sup> tags or whitespace
        label = soup.find(string=lambda text: text and "Ex-Dividend Date" in text)
        if label:
            # Check if we are in a table row (new layout)
            tr = label.find_parent("tr")
            if tr:
                val_td = tr.find("td", class_=lambda x: x and "value" in x)
                if val_td:
                    raw_value = val_td.text.strip()

            # Fallback for div/span layout (legacy) or if table lookup failed
            if not raw_value and label.parent:
                value_elem = label.parent.find_next_sibling()
                if value_elem:
                    raw_value = value_elem.text.strip()

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
Usage: python get_ex_dividend.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol to fetch the ex_dividend information for
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
        print(f"get_ex_dividend.py version {VERSION}")
        sys.exit(0)

    ticker = sys.argv[1].upper()
    result = get_ex_dividend(ticker)
    if result == "":
        print("")  # Output an empty string if no valid ex_dividend is found
    elif result == "N/A":
        print("N/A")  # Output "N/A" if the source returns "--"
    else:
        print(result)
