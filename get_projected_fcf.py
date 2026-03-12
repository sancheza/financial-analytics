#!/usr/bin/env python3

import sys
import re
from playwright.sync_api import sync_playwright

VERSION = "1.0.0"


def get_projected_fcf(ticker, debug=False):
    url = f"https://www.gurufocus.com/term/intrinsic-value-projected-fcf/{ticker}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        try:
            if debug:
                print(f"[DEBUG] Navigating to: {url}")
            page.goto(url, timeout=30000)
            page.wait_for_timeout(3000)

            # Find the projected FCF value - it's in a span with font-size: 24px and color #337ab7
            # Pattern: <span style="font-size: 24px; font-weight: 700; color: #337ab7">: $XX.XX (As of ...)</span>
            elements = page.query_selector_all("span")
            raw_value = None
            for elem in elements:
                style = elem.get_attribute("style") or ""
                if "font-size: 24px" in style and "#337ab7" in style:
                    text = elem.inner_text().strip()
                    # Extract just the dollar amount: ": $43.06 (As of Mar. 12, 2026)" -> "$43.06"
                    match = re.search(r":\s*(\$\d+(?:\.\d+)?)", text)
                    if match:
                        raw_value = match.group(1)
                        if debug:
                            print(f"[DEBUG] Found projected FCF: '{raw_value}'")
                        break

            if raw_value:
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
