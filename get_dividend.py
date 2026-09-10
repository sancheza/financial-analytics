#!/usr/bin/env python3
import sys
import re
import json
import requests
import yfinance as yf
import argparse

VERSION = "1.3.0"

# ANSI escape codes for formatting
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# iShares publishes per-fund SEC yields as JSON-LD on its product pages.
# The URL path carries a stable numeric fund id, so maintain a small map for
# the most common iShares bond funds. Unknown tickers fall through to the
# other sources.
ISHARES_PRODUCTS = {
    "TIP": "https://www.ishares.com/us/products/239667/ishares-tips-bond-etf",
    "AGG": "https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf",
    "LQD": "https://www.ishares.com/us/products/239566/ishares-iboxx-investment-grade-corporate-bond-etf",
    "IEF": "https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf",
    "HYG": "https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf",
    "TLT": "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf",
    "SHY": "https://www.ishares.com/us/products/239452/ishares-13-year-treasury-bond-etf",
    "MUB": "https://www.ishares.com/us/products/239766/ishares-national-amtfree-muni-bond-etf",
}


def _sec_yield_vanguard(ticker):
    """Vanguard's public profile JSON; 404 means the ticker is not a Vanguard fund."""
    url = f"https://investor.vanguard.com/irr/funds/profile/{ticker.upper()}"
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    price = resp.json().get("price", {})
    if not price.get("secYield"):
        return None
    sec_day = (price.get("yieldNote") or [{}])[0].get("secDay")
    return price["secYield"], sec_day, price.get("secYieldAsOfDate")


def _sec_yield_ishares(ticker):
    """Parses the '30 Day SEC Yield' JSON-LD property from an iShares product page."""
    url = ISHARES_PRODUCTS.get(ticker.upper())
    if url is None:
        return None
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    for block in re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', resp.text, re.S
    ):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        values = []
        def walk(node):
            if isinstance(node, dict):
                if node.get("@type") == "PropertyValue":
                    values.append((node.get("name", ""), node.get("value")))
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)
        walk(data)
        for i, (name, value) in enumerate(values):
            if "SEC Yield" in name:
                as_of = values[i + 1][1] if i + 1 < len(values) and values[i + 1][0] == "As of Dates" else None
                return value.strip(), "30", as_of
    return None


def _sec_yield_schwab_official(ticker):
    """Reads the 'SEC Yield' row from the official Schwab fund page.

    The sponsor's own pages are Akamai-blocked to automation, so the page is
    fetched through a reader proxy that returns rendered markdown. The row
    carries its own day count (7 or 30) and as-of date. Returns None for any
    ticker that is not a Schwab fund (the site serves a 'not found' page)."""
    page_url = f"https://www.schwabassetmanagement.com/products/{ticker.lower()}"
    reader_url = f"https://r.jina.ai/{page_url}"
    try:
        resp = requests.get(reader_url, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    m = re.search(
        r'SEC Yield \((?P<day>\d+) Day\).*?\|\s*[0-9/]+\s*\|\s*(?P<value>-?[0-9.]+%\s*)',
        resp.text,
        re.S,
    )
    if not m:
        return None
    value = m.group("value").strip()
    as_of = re.search(r"SEC Yield \(\d+ Day\)\*\*\s*As of ([0-9/]+)", resp.text)
    return value, m.group("day"), as_of.group(1) if as_of else None


def _sec_yield_schwab(ticker):
    """Parses the 'SEC Yield (30 Day)' row from Schwab's ETF research page.

    The page also serves non-Schwab ETFs; tickers without a yield return '--'.
    This research page is known to report unreliable values for some TIPS
    funds (e.g. SCHP), so it is only used as a fallback."""
    url = f"https://www.schwab.wallst.com/Prospect/Research/etfs/summary.asp?symbol={ticker}"
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    m = re.search(
        r'<tr><th>SEC Yield <span class="nowrap">\(30 Day\)</span></th><td>([^<]*)</td></tr>',
        resp.text,
    )
    if not m:
        return None
    value = m.group(1).strip().rstrip("*").strip()
    if not value or value in ("--", "-"):
        return None
    return value, "30", None


def fetch_sec_yield(ticker: str):
    """Returns a fund's SEC yield from the first source that has it.

    Tries, in order: iShares (for mapped tickers), Vanguard's own API, the
    official Schwab fund page, then Schwab's research page as a fallback.
    Returns (sec_yield_pct, sec_day, as_of_date), or None if no source has
    SEC yield data for the ticker.
    """
    for source in (
        _sec_yield_ishares,
        _sec_yield_vanguard,
        _sec_yield_schwab_official,
        _sec_yield_schwab,
    ):
        result = source(ticker)
        if result is not None:
            return result
    return None


# This function is the unified data fetcher for both single and batch modes.
def fetch_dividend_yield(ticker: str) -> str:
    """
    Fetches the dividend yield for a stock ticker, or the SEC yield for a
    fund, using fund sponsors' public data (iShares, Vanguard, Schwab) and
    yfinance as a fallback.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A string with the formatted yield (e.g., "1.55%" or "3.70% (7-day
        SEC yield as of 09/09/2026)"), "No Dividend" if none is offered,
        "Invalid Ticker" if the ticker is not found, or "Error" if another
        issue occurs.
    """
    try:
        # Try SEC-yield sources (iShares, Vanguard, Schwab) first; those
        # cover funds, where a dividend yield isn't meaningful.
        sec_yield = fetch_sec_yield(ticker)
        if sec_yield is not None:
            yield_pct, sec_day, as_of = sec_yield
            label = f"{yield_pct} ({sec_day}-day SEC yield"
            if as_of:
                label += f" as of {as_of}"
            return label + ")"

        stock = yf.Ticker(ticker)
        info = stock.info

        # Check for invalid ticker. yfinance returns a small dict for invalid tickers.
        # A valid ticker should have a 'symbol' and some price data.
        # If 'symbol' is missing or 'regularMarketPrice' is not available, it's likely invalid.
        if 'symbol' not in info or not info.get('regularMarketPrice'):
            return "Invalid Ticker"

        dividend_yield = info.get('dividendYield')

        if dividend_yield is None:
            return "No Dividend"

        # yfinance returns dividendYield already in percentage units
        # (e.g., 0.47 for 0.47%), so use it as is.
        return f"{dividend_yield:.2f}%"

    except Exception:
        # Catch any other exceptions (e.g., network issues, malformed ticker that yfinance can't handle)
        return "Error"

def process_ticker_file(input_file: str, output_file: str) -> None:
    """Reads tickers from an input file, gets their dividend yield, and writes to a CSV."""
    print(f"Reading tickers from: {CYAN}{input_file}{RESET}")
    try:
        with open(input_file, "r") as f:
            # Read tickers and filter out any empty lines
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"\n{YELLOW}Error: Input file '{input_file}' not found.{RESET}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing results to: {CYAN}{output_file}{RESET}")
    with open(output_file, "w") as f:
        f.write("Ticker,DividendYield\n")  # Add a header row for the CSV
        for ticker in tickers:
            dividend = fetch_dividend_yield(ticker)
            f.write(f"{ticker},{dividend}\n")
            print(f"  {ticker:<10} -> {dividend}")

    print(f"\n{GREEN}Successfully wrote dividend data to {output_file}{RESET}")

def print_help() -> None:
    """Prints a formatted, colorful help message for the unified script."""
    print(f"""
{GREEN}{BOLD}Get Dividend Yield v{VERSION}{RESET}

{CYAN}{BOLD}OVERVIEW:{RESET}
  Fetches the forward dividend yield for a single stock ticker
  or for a list of tickers from a file.

{CYAN}{BOLD}MODES OF OPERATION:{RESET}

  {BOLD}1. Single Ticker Mode (Default):{RESET}
     Fetches the dividend for a single ticker and prints it to the console.
     Designed for quick lookups and integration with other scripts.

     {CYAN}USAGE:{RESET}
       {YELLOW}python get_dividend.py [TICKER]{RESET}
     {CYAN}EXAMPLE:{RESET}
       {YELLOW}python get_dividend.py AAPL{RESET}

  {BOLD}2. Batch File Mode:{RESET}
     Reads a list of tickers from an input file (one per line) and saves
     the results to a CSV file.

     {CYAN}USAGE:{RESET}
       {YELLOW}python get_dividend.py -i <input_file> -o <output_file>{RESET}
     {CYAN}EXAMPLE:{RESET}
       {YELLOW}python get_dividend.py -i my_stocks.txt -o my_dividends.csv{RESET}

{CYAN}{BOLD}ARGUMENTS & OPTIONS:{RESET}
  {BOLD}TICKER{RESET}               The stock ticker symbol (for single ticker mode).

  {BOLD}-i, --input FILE{RESET}     Path to the input file containing tickers (for batch mode).
  {BOLD}-o, --output FILE{RESET}    Path to the output CSV file (for batch mode).
                         Default: dividends.csv

  {BOLD}-h, --help{RESET}           Show this help message and exit.
  {BOLD}-v, --version{RESET}        Show the script version and exit.
""")

def main():
    """Main function to parse arguments and run the correct mode."""
    parser = argparse.ArgumentParser(
        description="Fetch dividend yield for one or more stock tickers.",
        add_help=False  # Use custom help
    )
    # Use argparse.SUPPRESS to hide this from the standard help message
    parser.add_argument('ticker', nargs='?', default=None, help=argparse.SUPPRESS)
    parser.add_argument('-i', '--input', help='Input file with tickers.')
    parser.add_argument('-o', '--output', default='dividends.csv', help='Output CSV file.')
    parser.add_argument('-h', '--help', action='store_true', help='Show help message.')
    parser.add_argument('-v', '--version', action='version', version=f'get-dividend version {VERSION}')

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    # --- Mode Selection ---
    if args.input:
        # Batch File Mode
        process_ticker_file(args.input, args.output)
    elif args.ticker:
        # Single Ticker Mode
        ticker = args.ticker.upper()
        result = fetch_dividend_yield(ticker)

        # Maintain original script's output behavior for compatibility
        if result in ["Invalid Ticker", "Error"]:
            print("")
        elif result == "No Dividend":
            print("N/A")
        else:
            print(result)
    else:
        # If no ticker or input file is provided, show the help message.
        print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
