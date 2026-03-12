#!/usr/bin/env python3

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, date

VERSION = "1.0.1"

def get_yearly_performance(ticker, year):
    """
    Get stock performance data for a given ticker and year.
    
    Returns:
        dict: Contains start_price, end_price, dollar_change, percent_change
    """
    try:
        # Create date range - extend slightly to ensure we get data
        start_date = f"{year-1}-12-15"  # Start earlier to ensure we get first trading day
        end_date = f"{year+1}-01-15"    # End later to ensure we get last trading day
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
        
        if hist.empty:
            return None
        
        # Filter to exact year
        year_data = hist[hist.index.year == year]
        
        if year_data.empty:
            return None
            
        # Get first and last trading day prices (using Close, not Adj Close)
        first_price = year_data['Close'].iloc[0]
        last_price = year_data['Close'].iloc[-1]
        
        # Calculate changes
        dollar_change = last_price - first_price
        percent_change = (dollar_change / first_price) * 100
        
        return {
            'start_price': round(first_price, 2),
            'end_price': round(last_price, 2),
            'dollar_change': round(dollar_change, 2),
            'percent_change': round(percent_change, 2),
            'first_date': year_data.index[0].strftime('%Y-%m-%d'),
            'last_date': year_data.index[-1].strftime('%Y-%m-%d'),
            'debug_info': {
                'total_days': len(year_data),
                'raw_first_price': first_price,
                'raw_last_price': last_price
            }
        }
        
    except Exception as e:
        print(f"Debug: Error occurred - {str(e)}")
        return None

def print_help():
    help_text = """
yearly_performance.py - Get yearly stock performance data

This script retrieves stock price performance for a given ticker symbol
and calendar year, showing the opening and closing prices along with
total gains/losses in both dollar amount and percentage.

Usage: python yearly_performance.py [OPTION] TICKER YEAR

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit
  -d, --debug     Show debug information

Arguments:
  TICKER          The stock ticker symbol (e.g., AAPL, MSFT)
  YEAR            The calendar year (e.g., 2023, 2022)

Examples:
  python yearly_performance.py GS 2019       # Goldman Sachs performance for 2019
  python yearly_performance.py AAPL 2023     # Apple performance for 2023
  python yearly_performance.py --help        # Show this help message

Output:
  Displays the first trading day price, last trading day price, dollar change,
  and percentage change for the specified year. Returns "N/A" if data is
  unavailable or the ticker/year is invalid.
"""
    print(help_text)

if __name__ == "__main__":
    debug_mode = False
    
    if len(sys.argv) < 3:
        if len(sys.argv) == 2:
            arg = sys.argv[1].lower()
            if arg in ("-h", "--help"):
                print_help()
                sys.exit(0)
            elif arg in ("-v", "--version"):
                print(f"yearly_performance.py version {VERSION}")
                sys.exit(0)
        
        print("Usage: yearly_performance.py TICKER YEAR")
        print("Use --help for more information")
        sys.exit(1)

    # Check for debug flag
    if "--debug" in sys.argv or "-d" in sys.argv:
        debug_mode = True
        sys.argv.remove("--debug" if "--debug" in sys.argv else "-d")

    ticker = sys.argv[1].upper()
    
    try:
        year = int(sys.argv[2])
        current_year = datetime.now().year
        
        if year < 1970 or year > current_year:
            print(f"Error: Year must be between 1970 and {current_year}")
            sys.exit(1)
            
    except ValueError:
        print("Error: Year must be a valid integer")
        sys.exit(1)

    result = get_yearly_performance(ticker, year)
    
    if result is None:
        print("N/A - Unable to retrieve data for the specified ticker and year")
    else:
        print(f"Stock Performance for {ticker} in {year}")
        print(f"{'='*50}")
        print(f"First trading day ({result['first_date']}): ${result['start_price']}")
        print(f"Last trading day ({result['last_date']}):  ${result['end_price']}")
        print(f"{'='*50}")
        print(f"Dollar change: ${result['dollar_change']}")
        print(f"Percent change: {result['percent_change']}%")
        
        if debug_mode:
            print(f"\nDebug Info:")
            print(f"Trading days in year: {result['debug_info']['total_days']}")
            print(f"Raw first price: {result['debug_info']['raw_first_price']}")
            print(f"Raw last price: {result['debug_info']['raw_last_price']}")
        
        if result['dollar_change'] > 0:
            print(f"Result: GAIN of ${result['dollar_change']} ({result['percent_change']}%)")
        elif result['dollar_change'] < 0:
            print(f"Result: LOSS of ${abs(result['dollar_change'])} ({result['percent_change']}%)")
        else:
            print("Result: NO CHANGE")