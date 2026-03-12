#!/usr/bin/env python3

import sys
import yfinance as yf
import argparse

VERSION = "1.3.0"

# ANSI escape codes for formatting
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# This function is the unified data fetcher for both single and batch modes.
def fetch_dividend_yield(ticker: str) -> str:
    """
    Fetches the forward dividend yield for a stock ticker using yfinance.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A string with the formatted dividend yield (e.g., "1.55%"),
        "No Dividend" if none is offered, "Invalid Ticker" if the ticker
        is not found, or "Error" if another issue occurs.
    """
    try:
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

        # The yfinance library typically returns dividendYield as a ratio (e.g., 0.069 for 6.9%).
        # To safeguard against cases where it might be returned as a percentage (e.g., 6.9),
        # we check if the value is a ratio (<1.0) before multiplying.
        if dividend_yield < 1.0:
            # Value is a ratio, so convert to percentage.
            percent_yield = dividend_yield * 100
        else:
            # Value is likely already a percentage, use as is.
            percent_yield = dividend_yield

        return f"{percent_yield:.2f}%"

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
