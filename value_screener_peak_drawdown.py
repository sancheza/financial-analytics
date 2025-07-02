#!/usr/bin/env python3
"""
S&P 500 Peak Drawdown Screener

Finds S&P 500 stocks down X% or more from their peak over the last N years,
with a minimum market cap filter. Uses yfinance for data and caches results.

Usage:
    python value_screener_peak_drawdown.py [--percent 70] [--years 10] [--marketcap 10B] [--forceupdate]

Example:
    python value_screener_peak_drawdown.py --percent 75 --years 5 --marketcap 5B
"""

import yfinance as yf
import pandas as pd
import argparse
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import re
import wcwidth
import warnings
from io import StringIO

__version__ = "1.03"

# --- Utility Functions ---

def parse_marketcap(s):
    """Parse market cap string like '10B', '500M', '1T' to integer."""
    s = s.strip().upper()
    match = re.match(r'^(\d+(\.\d+)?)([MBT])$', s)
    if not match:
        raise ValueError("Market cap must be like 10B, 500M, 1T")
    num = float(match.group(1))
    mult = {'M': 1e6, 'B': 1e9, 'T': 1e12}[match.group(3)]
    return int(num * mult)

def get_sp500_tickers_for_year(target_year):
    """Get S&P 500 tickers for a given year using hanshof/sp500_constituents cache logic."""
    cache_dir = Path("./data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sp500_historical_constituents.csv"
    metadata_file = cache_dir / "sp500_cache_metadata.json"
    github_url = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv"

    def should_download_fresh_data():
        if not cache_file.exists() or not metadata_file.exists():
            return True
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            cached_years = metadata.get('available_years', [])
            if target_year not in cached_years:
                return True
            cache_date = datetime.fromisoformat(metadata['last_updated'])
            if (datetime.now() - cache_date).days > 7:
                return True
            return False
        except Exception:
            return True

    def download_and_cache_data():
        try:
            print(f"Downloading S&P 500 historical data...")
            import requests
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            with open(cache_file, 'w') as f:
                f.write(response.text)
            df = pd.read_csv(StringIO(response.text))
            df['date'] = pd.to_datetime(df['date'])
            available_years = sorted(df['date'].dt.year.unique().tolist())
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'available_years': available_years
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Data cached: {len(df)} records")
            return True
        except Exception as e:
            print(f"❌ Error downloading S&P 500 data: {e}")
            return False

    def get_constituents_from_cache(year):
        try:
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            target_date = datetime(year, 12, 31)
            available_dates = df[df['date'] <= target_date]['date']
            if available_dates.empty:
                print(f"❌ No S&P 500 data for {year}")
                return []
            most_recent_date = available_dates.max()
            row = df[df['date'] == most_recent_date].iloc[0]
            tickers = [t.strip().replace('.', '-') for t in row['tickers'].split(',') if t.strip()]
            return tickers
        except Exception as e:
            print(f"❌ Error reading S&P 500 cache: {e}")
            return []

    if should_download_fresh_data():
        download_and_cache_data()
    return get_constituents_from_cache(target_year)

def print_pretty_results_table(results, percent, years, marketcap_str):
    """Pretty print the results table, inspired by value_price_screener.py."""
    import os

    class Colors:
        HEADER = ''
        OKBLUE = ''
        OKCYAN = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''

    # Disable colors if not a TTY
    use_colors = sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
    if use_colors:
        Colors.HEADER = '\033[95m'
        Colors.OKBLUE = '\033[94m'
        Colors.OKCYAN = '\033[96m'
        Colors.OKGREEN = '\033[92m'
        Colors.WARNING = '\033[93m'
        Colors.FAIL = '\033[91m'
        Colors.ENDC = '\033[0m'
        Colors.BOLD = '\033[1m'

    def colorize(text, color_code):
        return f"{color_code}{text}{Colors.ENDC}" if use_colors and color_code else text

    def get_display_width(text):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        return sum(wcwidth.wcwidth(char) for char in clean_text)

    def pad_text(text, width, align='center'):
        display_width = get_display_width(text)
        padding_needed = width - display_width
        if align == 'center':
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            return ' ' * left_pad + text + ' ' * right_pad
        elif align == 'left':
            return text + ' ' * padding_needed
        else:
            return ' ' * padding_needed + text

    if not results:
        print(colorize("\n❌ No stocks met the criteria.", Colors.FAIL))
        return

    headers = ["Ticker", "Peak Price", "Peak Date", "Current Price", "% Down", "Market Cap"]
    col_widths = [8, 12, 13, 14, 9, 14]
    table_width = sum(col_widths) + len(col_widths) + 1
    header_width = table_width - 2

    title = f"S&P 500 Stocks Down {percent}%+ From {years}Y Peak (Cap ≥ {marketcap_str})"
    print("\n" + "┌" + "─" * header_width + "┐")
    print("│" + title.center(header_width) + "│")
    print("├" + "─" * col_widths[0] + "┬" + "─" * col_widths[1] + "┬" + "─" * col_widths[2] + "┬" +
          "─" * col_widths[3] + "┬" + "─" * col_widths[4] + "┬" + "─" * col_widths[5] + "┤")
    print("│" + "".join([pad_text(h, w) + "│" for h, w in zip(headers, col_widths)]))
    print("├" + "─" * col_widths[0] + "┼" + "─" * col_widths[1] + "┼" + "─" * col_widths[2] + "┼" +
          "─" * col_widths[3] + "┼" + "─" * col_widths[4] + "┼" + "─" * col_widths[5] + "┤")

    for row in results:
        ticker = colorize(str(row['Ticker']), Colors.BOLD + (Colors.FAIL if row['% Down'] < -80 else Colors.OKGREEN))
        down_str = f"{row['% Down']:.1f}%"
        down_col = Colors.FAIL if row['% Down'] < -80 else Colors.WARNING if row['% Down'] < -70 else Colors.OKGREEN
        print("│" +
              pad_text(ticker, col_widths[0]) + "│" +
              pad_text(f"${row['Peak Price']:.2f}", col_widths[1]) + "│" +
              pad_text(row['Peak Date'], col_widths[2]) + "│" +
              pad_text(f"${row['Current Price']:.2f}", col_widths[3]) + "│" +
              pad_text(colorize(down_str, down_col), col_widths[4]) + "│" +
              pad_text(row['Market Cap'], col_widths[5]) + "│")
    print("└" + "─" * col_widths[0] + "┴" + "─" * col_widths[1] + "┴" + "─" * col_widths[2] + "┴" +
          "─" * col_widths[3] + "┴" + "─" * col_widths[4] + "┴" + "─" * col_widths[5] + "┘")

# --- Main Logic ---

def main(percent, years, marketcap_str, forceupdate, verbose=False):
    percent = float(percent)
    years = int(years)
    min_marketcap = parse_marketcap(marketcap_str)
    cache_dir = Path("./data/json")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sp500_history_cache.json"

    # Use current S&P 500 tickers, not historical
    tickers = get_sp500_tickers_for_year(datetime.now().year)
    tickers.sort()

    # Load or build cache
    cache = {}
    if cache_file.exists() and not forceupdate:
        with open(cache_file, "r") as f:
            cache = json.load(f)

    results = []
    skipped = 0
    insufficient_history = 0
    skipped_tickers = []
    insufficient_history_tickers = []
    total = len(tickers)
    min_days = int(years * 365 * 0.95)  # Require at least 95% of expected days

    for idx, ticker in enumerate(tickers, 1):
        try:
            if ticker not in cache or forceupdate:
                stock = yf.Ticker(ticker)
                start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = stock.history(start=start, end=end, auto_adjust=True)
                if hist.empty or 'Close' not in hist:
                    skipped += 1
                    skipped_tickers.append(ticker)
                    if verbose:
                        print(f"[{ticker}] Skipped: No price data found.")
                    continue
                # Check for sufficient history
                if (hist.index[-1] - hist.index[0]).days < min_days:
                    insufficient_history += 1
                    years_available = (hist.index[-1] - hist.index[0]).days / 365.25
                    insufficient_history_tickers.append(f"{ticker} ({years_available:.2f}y)")
                    if verbose:
                        print(f"[{ticker}] Skipped: Insufficient price history ({years_available:.2f} years).")
                    continue
                peak_row = hist['Close'].idxmax()
                peak_price = float(hist['Close'].max())
                peak_date = str(peak_row.date())
                current_price = float(hist['Close'].iloc[-1])
                info = stock.info
                mcap = info.get('marketCap', 0)
                if mcap >= 1e12:
                    mcap_disp = f"{mcap/1e12:.2f}T"
                elif mcap >= 1e9:
                    mcap_disp = f"{mcap/1e9:.2f}B"
                elif mcap >= 1e6:
                    mcap_disp = f"{mcap/1e6:.2f}M"
                else:
                    mcap_disp = str(mcap)
                cache[ticker] = {
                    "peak_price": peak_price,
                    "peak_date": peak_date,
                    "current_price": current_price,
                    "marketcap": mcap,
                    "marketcap_disp": mcap_disp,
                    "history_start": str(hist.index[0].date()),
                    "history_end": str(hist.index[-1].date())
                }
                time.sleep(0.7)
            else:
                peak_price = cache[ticker]["peak_price"]
                peak_date = cache[ticker]["peak_date"]
                current_price = cache[ticker]["current_price"]
                mcap = cache[ticker]["marketcap"]
                mcap_disp = cache[ticker]["marketcap_disp"]
                # Check for sufficient history in cache
                if "history_start" in cache[ticker] and "history_end" in cache[ticker]:
                    hist_days = (
                        datetime.fromisoformat(cache[ticker]["history_end"]) -
                        datetime.fromisoformat(cache[ticker]["history_start"])
                    ).days
                    if hist_days < min_days:
                        insufficient_history += 1
                        years_available = hist_days / 365.25
                        insufficient_history_tickers.append(f"{ticker} ({years_available:.2f}y)")
                        if verbose:
                            print(f"[{ticker}] Skipped: Insufficient price history ({years_available:.2f} years).")
                        continue

            if mcap is None or mcap < min_marketcap:
                if verbose:
                    print(f"[{ticker}] Skipped: Market cap {mcap_disp} below threshold.")
                continue
            pct_down = ((current_price - peak_price) / peak_price) * 100
            if pct_down <= -percent:
                results.append({
                    "Ticker": ticker,
                    "Peak Price": peak_price,
                    "Peak Date": peak_date,
                    "Current Price": current_price,
                    "% Down": pct_down,
                    "Market Cap": mcap_disp
                })
                if verbose:
                    print(f"[{ticker}] Peak: ${peak_price:.2f} on {peak_date} | Current: ${current_price:.2f} | % Down: {pct_down:.1f}% | Market Cap: {mcap_disp}")
            elif verbose:
                print(f"[{ticker}] Not down enough: {pct_down:.1f}% from peak.")
        except Exception as e:
            skipped += 1
            skipped_tickers.append(ticker)
            if verbose:
                print(f"[{ticker}] Skipped: Exception: {e}")
            continue
        if idx % 10 == 0 or idx == total:
            print(f"\rProcessed {idx}/{total} tickers... Skipped: {skipped}  Insufficient history: {insufficient_history}", end="", flush=True)
    print()
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    results.sort(key=lambda x: x["% Down"])
    print_pretty_results_table(results, percent, years, marketcap_str)
    print(f"\nSkipped {skipped} tickers due to missing or delisted data.")
    if skipped_tickers:
        print("  Skipped tickers:", ", ".join(skipped_tickers))
    if insufficient_history > 0:
        print(f"⚠️  {insufficient_history} tickers skipped due to insufficient price history for the requested {years} years.")
        print("  Insufficient history tickers:", ", ".join(insufficient_history_tickers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
S&P 500 Peak Drawdown Screener

This script analyzes the current S&P 500 constituents and identifies stocks that are down a specified percentage or more from their peak price over the past N years, with an optional minimum market capitalization filter.

For each S&P 500 ticker:
- Downloads historical daily price data for the specified number of years (default: 10).
- Finds the highest closing price (the peak) and its date in that period.
- Compares the latest closing price to the peak and calculates the percentage drawdown.
- Retrieves the latest market capitalization from yfinance.
- Filters results to only include stocks that are down at least the specified percentage (default: 70%) and have a market cap above the specified threshold (default: 10B).
- Results are cached in a single JSON file for faster subsequent runs. Use --forceupdate to refresh the cache.
- The output is a formatted table showing: Ticker, Peak Price, Peak Date, Current Price, % Down, and Market Cap.
- At the end, the script reports any tickers skipped due to missing/delisted data or insufficient price history, listing the affected tickers.

Example usage:
    python value_screener_peak_drawdown.py --percent 75 --years 5 --marketcap 5B

""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--percent", type=float, default=70, help="Percent down from peak (default: 70)")
    parser.add_argument("--years", type=int, default=10, help="Number of years for peak search (default: 10)")
    parser.add_argument("--marketcap", type=str, default="10B", help="Minimum market cap (e.g. 10B, 500M, 1T; default: 10B)")
    parser.add_argument("--forceupdate", action="store_true", help="Force update cache from yfinance")
    parser.add_argument("--verbose", action="store_true", help="Show progress and metrics for each ticker")
    parser.add_argument("-v", "--version", action="store_true", help="Show script version and exit")
    args = parser.parse_args()

    if args.version:
        print(f"S&P 500 Peak Drawdown Screener version {__version__}")
        sys.exit(0)

    main(args.percent, args.years, args.marketcap, args.forceupdate, verbose=args.verbose)