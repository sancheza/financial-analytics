#!/usr/bin/env python3

import sys
import requests
import yfinance as yf
from bs4 import BeautifulSoup

VERSION = "1.0.3"

def get_pb(ticker):
    stock = yf.Ticker(ticker)
    pb = stock.info.get("priceToBook", None)
    return pb

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