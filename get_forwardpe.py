#!/usr/bin/env python3

import sys
import yfinance as yf

VERSION = "1.1.0"
# Switched to yfinance library to avoid scraping and improve reliability.

def get_forward_pe(ticker: str) -> str:
    """
    Fetches the Forward P/E for a given stock ticker using the yfinance library.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # yfinance returns a small dict for invalid tickers, check for a real key like 'symbol'
        if 'symbol' not in stock.info:
            return "" # Invalid ticker

        forward_pe = stock.info.get("forwardPE")

        if forward_pe is not None:
            return f"{forward_pe:.2f}"
        else:
            # If forwardPE is None, it's often represented as "N/A" on the site.
            return "N/A"
    except Exception:
        # This can happen for various reasons, like network errors or delisted tickers
        # where yfinance raises an exception.
        return ""

def print_help():
    help_text = """
Usage: python get_forward_pe.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol to fetch the Forward P/E for.
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
        print(f"get_forward_pe.py version {VERSION}")
        sys.exit(0)

    ticker = sys.argv[1].upper()
    result = get_forward_pe(ticker)
    print(result)
