#!/usr/bin/env python3

import sys
from playwright.sync_api import sync_playwright

VERSION = "1.0.0"

def get_projected_fcf(ticker, debug=False):
    url = f"https://www.gurufocus.com/term/intrinsic-value-projected-fcf/{ticker}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        try:
            if debug:
                print(f"[DEBUG] Navigating to: {url}")
            page.goto(url, timeout=30000)
            page.wait_for_selector('p strong', timeout=15000)  # Wait up to 15s for the data
            html = page.content()
            elements = page.query_selector_all('p')
            if debug:
                print(f"[DEBUG] Downloaded HTML (first 1000 chars):\n{html[:1000]}\n...")
                print(f"[DEBUG] Found {len(elements)} <p> tags.")
            if len(elements) >= 2:
                strongs = elements[1].query_selector_all('strong')
                if debug:
                    print(f"[DEBUG] Found {len(strongs)} <strong> tags in 2nd <p>.")
                if len(strongs) >= 2:
                    raw_value = strongs[1].inner_text().strip()
                    if debug:
                        print(f"[DEBUG] Extracted value: '{raw_value}'")
                    if raw_value == "--":
                        browser.close()
                        return "N/A"
                    if not raw_value or raw_value == "-":
                        browser.close()
                        return ""
                    browser.close()
                    return raw_value
            browser.close()
            return ""
        except Exception as e:
            if debug:
                print(f"[DEBUG] Exception: {e}")
            browser.close()
            return ""

def print_help():
    help_text = """
Usage: python get_projected_fcf.py [OPTION] TICKER

Options:
  -h, --help      Show this help message and exit
  -v, --version   Show the version of the script and exit

Arguments:
  TICKER          The stock ticker symbol to fetch the Projected FCF information for
"""
    print(help_text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Projected FCF from GuruFocus.")
    parser.add_argument("ticker", nargs="?", help="Stock ticker symbol")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    args, unknown = parser.parse_known_args()

    if args.version:
        print(f"get_projected_fcf.py version {VERSION}")
        sys.exit(0)
    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    ticker = args.ticker.upper()
    result = get_projected_fcf(ticker, debug=args.debug)
    if result == "":
        print("")
    elif result == "N/A":
        print("N/A")
    else:
        print(result)
