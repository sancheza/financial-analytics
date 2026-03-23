#!/usr/bin/env python3

import argparse
import re
import random
import time
import sys
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

VERSION = "1.1.0"

# --- Constants ---
BASE_DELAY = 1.5 # Base delay in seconds

# ANSI escape codes for colors
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Set up logging
logger = logging.getLogger(__name__)

class FairValueScraper:
    """
    A scraper class to manage a persistent Playwright browser instance.
    This is more efficient as it reuses a single browser for all requests.
    """
    def __init__(self, headless: bool = True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        # Use a common user agent and viewport to appear more like a real user
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )

    def get_fair_value_alphaspread(self, ticker: str) -> str | None:
        """Fetches the fair value from AlphaSpread, returning the raw value string (e.g., '$123.45 USD')."""
        url = f"https://www.alphaspread.com/security/nasdaq/{ticker.lower()}/summary"
        page = self.context.new_page()
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector("div[class*='intrinsic-value']", timeout=30000)
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            intrinsic_div = soup.select_one("div[class*='intrinsic-value']")
            if intrinsic_div:
                value_text = intrinsic_div.text.strip()
                logger.info(f"AlphaSpread: Found intrinsic value text: {value_text}")
                return value_text
            else:
                logger.warning("Intrinsic value element not found in the page")
                return "Intrinsic value element not found"
        except Exception as e:
            logger.error(f"Error fetching AlphaSpread data: {str(e)}")
            return f"Error: {e}"
        finally:
            page.close()

    def get_fair_value_valueinvesting_io(self, ticker: str) -> str | None:
        """Fetches the fair value from ValueInvesting.io, returning the raw value string (e.g., '$123.45')."""
        url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
        page = self.context.new_page()
        try:
            page.goto(url, timeout=60000)
            # Per your instruction, using the specific XPath.
            selector = "xpath=//*[@id='app1']/div[2]/div[2]/div[1]/div[2]/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div[1]"
            page.wait_for_selector(selector, timeout=30000)

            element_text = page.locator(selector).inner_text()

            if element_text:
                logger.info(f"ValueInvesting.io: Found fair value text: {element_text}")
                return element_text.strip()

            logger.warning(f"ValueInvesting.io: Fair value element not found for {ticker}")
            return "Fair value not found"
        except Exception as e:
            logger.error(f"Error fetching ValueInvesting.io data: {str(e)}")
            return f"Error: {e}"
        finally:
            page.close()

    def get_fair_value_gurufocus(self, ticker: str) -> str | None:
        """Fetches the fair value from GuruFocus DCF page."""
        url = f"https://www.gurufocus.com/stock/{ticker.upper()}/dcf"
        page = self.context.new_page()
        try:
            # __NUXT__ is server-side rendered, so domcontentloaded is sufficient —
            # no need to wait for networkidle or dismiss the cookie banner.
            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            # Extract iv_dcf directly from the Nuxt.js server-side state.
            # The DOM's "Fair Value" element is not reliably hydrated, so we read
            # the raw data instead of relying on a fragile CSS/SVG selector.
            iv_dcf = page.evaluate("""() => {
                try {
                    const fetchData = window.__NUXT__.fetch;
                    for (const key of Object.keys(fetchData)) {
                        const entry = fetchData[key];
                        if (entry && entry.stock && entry.stock.iv_dcf !== undefined) {
                            return entry.stock.iv_dcf;
                        }
                    }
                } catch(e) { return null; }
                return null;
            }""")

            if iv_dcf:
                logger.info(f"GuruFocus: iv_dcf = {iv_dcf}")
                return f"${iv_dcf}"

            # iv_dcf is 0 or missing — the DCF value is computed client-side by Vue.
            # Re-load the page in non-headless mode so the component fully renders,
            # then read the Fair Value directly from the DOM.
            logger.info(f"GuruFocus: iv_dcf unavailable for {ticker}, retrying in non-headless mode.")
        except Exception as e:
            logger.error(f"Error fetching GuruFocus data: {str(e)}")
            return f"Error: {e}"
        finally:
            page.close()

        nh_browser = self.playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            nh_context = nh_browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            nh_context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            nh_page = nh_context.new_page()
            nh_page.goto(url, timeout=60000)
            try:
                nh_page.wait_for_selector('button:has-text("Accept All")', timeout=5000)
                nh_page.click('button:has-text("Accept All")')
            except Exception:
                pass
            nh_page.wait_for_load_state("networkidle", timeout=60000)

            dom_value = nh_page.evaluate("""() => {
                for (const row of document.querySelectorAll('.dcf-table-row')) {
                    if (row.textContent.includes('Fair Value')) {
                        const cell = row.querySelector('[class*="text-right"]');
                        if (cell) {
                            return Array.from(cell.childNodes)
                                .filter(n => n.nodeType === 3)
                                .map(n => n.textContent.trim())
                                .filter(t => t.length > 0)
                                .join(' ');
                        }
                    }
                }
                return null;
            }""")

            if dom_value:
                logger.info(f"GuruFocus: DOM Fair Value = {dom_value!r}")
                cleaned = re.sub(r'\s+', '', dom_value)
                return cleaned

            logger.warning(f"GuruFocus: Fair Value not found in DOM for {ticker}")
            return "DCF value not found"
        except Exception as e:
            logger.error(f"Error fetching GuruFocus data (non-headless): {str(e)}")
            return f"Error: {e}"
        finally:
            nh_browser.close()

    def get_fair_value_simplywallst(self, ticker: str) -> str | None:
        """Fetches the intrinsic fair value from Simply Wall St.

        Simply Wall St blocks headless browsers via Cloudflare, so this method
        launches a separate non-headless browser. It resolves the stock's canonical
        URL via the SWS GraphQL API (trying common US exchange prefixes), then
        extracts share_price and intrinsic_discount from the embedded React Query
        state to compute: fair_value = share_price / (1 - intrinsic_discount / 100).
        """
        sws_browser = self.playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            sws_context = sws_browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                viewport={'width': 1440, 'height': 900},
                locale='en-US',
            )
            sws_context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page = sws_context.new_page()

            # Step 1: Load any SWS page to obtain Cloudflare clearance.
            # Wait for the challenge URL to resolve rather than sleeping a fixed duration.
            page.goto("https://simplywall.st/stocks/us/market-cap-large", timeout=60000, wait_until="domcontentloaded")
            try:
                page.wait_for_url(lambda url: "/challenge" not in url, timeout=20000)
            except Exception:
                pass  # No challenge page, or already resolved
            logger.info("Simply Wall St: Cloudflare clearance obtained.")

            # Step 2: Resolve the canonical URL via GraphQL.
            # SWS unique symbols follow the pattern "EXCHANGE:TICKER".
            # We try common US exchange prefixes in priority order.
            exchange_prefixes = ["NYSE", "NasdaqGS", "NasdaqGM", "NasdaqCM"]
            canonical_url = None
            for exchange in exchange_prefixes:
                unique_symbol = f"{exchange}:{ticker.upper()}"
                canonical_url = page.evaluate(
                    """async (uniqueSymbol) => {
                        try {
                            const r = await fetch('/graphql', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    query: `{ Company(id: "${uniqueSymbol}") { canonicalURL } }`
                                })
                            });
                            const data = await r.json();
                            return data?.data?.Company?.canonicalURL || null;
                        } catch(e) { return null; }
                    }""",
                    unique_symbol
                )
                if canonical_url:
                    logger.info(f"Simply Wall St: Resolved {unique_symbol} -> {canonical_url}")
                    break

            if not canonical_url:
                logger.warning(f"Simply Wall St: Could not resolve ticker {ticker.upper()} to a SWS URL.")
                return "Stock not found on Simply Wall St"

            # Step 3: Navigate to the stock page.
            page.goto(f"https://simplywall.st{canonical_url}", timeout=60000, wait_until="domcontentloaded")

            # Wait for the React Query state to be populated with analysis data.
            page.wait_for_function(
                """() => {
                    try {
                        return window.__REACT_QUERY_STATE__?.queries?.some(
                            q => q?.state?.data?.data?.analysis?.data?.share_price !== undefined
                        );
                    } catch(e) { return false; }
                }""",
                timeout=30000
            )

            # Step 4: Extract share_price and intrinsic_discount from React Query state.
            data = page.evaluate("""() => {
                try {
                    for (const query of window.__REACT_QUERY_STATE__.queries) {
                        const a = query?.state?.data?.data?.analysis?.data;
                        if (a && a.share_price !== undefined && a.intrinsic_discount !== undefined) {
                            return { share_price: a.share_price, intrinsic_discount: a.intrinsic_discount };
                        }
                    }
                } catch(e) {}
                return null;
            }""")

            if data:
                fair_value = data['share_price'] / (1 - data['intrinsic_discount'] / 100)
                logger.info(
                    f"Simply Wall St: share_price={data['share_price']}, "
                    f"intrinsic_discount={data['intrinsic_discount']}%, "
                    f"fair_value={fair_value:.2f}"
                )
                return f"${fair_value:.2f}"

            logger.warning(f"Simply Wall St: Fair value data not found for {ticker}")
            return "Fair value not found"
        except Exception as e:
            logger.error(f"Error fetching Simply Wall St data: {str(e)}")
            return f"Error: {e}"
        finally:
            sws_browser.close()

    def fetch_all_parallel(self, ticker: str) -> dict[str, str | None]:
        """Fetch fair values from all four sources concurrently.

        Playwright's sync API is not thread-safe for shared objects, so each
        worker thread gets its own isolated FairValueScraper instance (and thus
        its own playwright/browser). The wall-clock time becomes roughly equal
        to the slowest single source instead of the sum of all four.
        """
        sources = [
            ("AlphaSpread",       "get_fair_value_alphaspread"),
            ("ValueInvesting.io", "get_fair_value_valueinvesting_io"),
            ("GuruFocus",         "get_fair_value_gurufocus"),
            ("Simply Wall St",    "get_fair_value_simplywallst"),
        ]

        def run_isolated(method_name: str) -> str | None:
            isolated = FairValueScraper(headless=True)
            try:
                return getattr(isolated, method_name)(ticker)
            finally:
                isolated.close()

        results: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {
                executor.submit(run_isolated, method): source
                for source, method in sources
            }
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results[source] = future.result()
                except Exception as e:
                    logger.error(f"Parallel fetch error for {source}: {e}")
                    results[source] = f"Error: {e}"
        return results

    def close(self):
        """Closes the browser and stops the Playwright instance."""
        self.browser.close()
        self.playwright.stop()

def process_ticker_file_fair_value(input_file: str, output_file: str, scraper: FairValueScraper, debug: bool = False) -> None:
    """Reads tickers from an input file, gets their fair value, and writes to a CSV."""
    print(f"Reading tickers from: {BColors.OKCYAN}{input_file}{BColors.ENDC}")
    try:
        with open(input_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"\n{BColors.FAIL}Error: Input file '{input_file}' not found.{BColors.ENDC}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing results to: {BColors.OKCYAN}{output_file}{BColors.ENDC}")
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "AlphaSpread", "ValueInvesting.io", "GuruFocus", "Simply Wall St"])
        for i, ticker in enumerate(tickers):
            print(f"  Processing {ticker.upper()}...")
            results = scraper.fetch_all_parallel(ticker)

            alpha_result_text       = results.get("AlphaSpread")
            valueinvesting_result_text = results.get("ValueInvesting.io")
            gurufocus_result_text   = results.get("GuruFocus")
            simplywallst_result_text = results.get("Simply Wall St")

            def csv_val(text):
                v = parse_value_from_string(text)
                return str(v) if v is not None else ""

            writer.writerow([
                ticker.upper(),
                csv_val(alpha_result_text),
                csv_val(valueinvesting_result_text),
                csv_val(gurufocus_result_text),
                csv_val(simplywallst_result_text),
            ])
            print(f"    AlphaSpread: {alpha_result_text or 'Not found'}")
            print(f"    ValueInvesting.io: {valueinvesting_result_text or 'Not found'}")
            print(f"    GuruFocus: {gurufocus_result_text or 'Not found'}")
            print(f"    Simply Wall St: {simplywallst_result_text or 'Not found'}")
            # Brief pause between tickers to avoid hammering sites back-to-back
            if i < len(tickers) - 1:
                time.sleep(random.uniform(1.0, 2.5))

    print(f"\n{BColors.OKGREEN}Successfully wrote fair value data to {output_file}{BColors.ENDC}")

def parse_value_from_string(s: str) -> float | None:
    """
    Parses a numerical value from a string, ignoring currency symbols and text.
    Handles strings like '$123.45', '1,234.56 USD', or just '123.45'.
    Returns None if no valid number is found or if the string contains an error message.
    """
    if not isinstance(s, str):
        return None

    # Avoid parsing numbers from common error messages to prevent false positives (e.g., from error codes).
    if "error" in s.lower() or "not found" in s.lower():
        return None

    # Regex to find the first floating-point or integer number in the string.
    # It handles commas as thousand separators.
    match = re.search(r'(\d[\d,]*\.?\d*)', s)
    if match:
        try:
            # Clean the matched number string by removing commas before converting to float.
            number_str = match.group(1).replace(',', '')
            return float(number_str)
        except (ValueError, IndexError):
            return None
    return None

def main():
    description_text = f"""{BColors.BOLD}Get fair value estimates for a stock from AlphaSpread, ValueInvesting.io, and GuruFocus.{BColors.ENDC}

{BColors.HEADER}Overview:{BColors.ENDC}
  This script fetches the intrinsic/fair value of a given stock ticker from three
  online sources: AlphaSpread, ValueInvesting.io, and GuruFocus. It uses Playwright to
  automate a browser and bypass anti-bot protections to scrape the required data.

{BColors.HEADER}How it works:{BColors.ENDC}
  - You provide a stock ticker symbol (e.g., {BColors.OKGREEN}MRNA{BColors.ENDC}).
  - The script uses Playwright to load AlphaSpread, ValueInvesting.io, and GuruFocus pages
    and extract the fair value estimate for the stock.
  - Results are printed to the console.

{BColors.HEADER}Requirements:{BColors.ENDC}
  - Python package '{BColors.OKGREEN}playwright{BColors.ENDC}' must be installed.
  - Run '{BColors.OKGREEN}playwright install{BColors.ENDC}' after installing the package.
  - Internet connection.

{BColors.HEADER}Modes of Operation:{BColors.ENDC}
  {BColors.BOLD}1. Single Ticker Mode:{BColors.ENDC}
     Fetches and displays fair values for a single ticker with formatted output.
     {BColors.OKCYAN}USAGE:{BColors.ENDC} {BColors.OKGREEN}python get_fair_value.py MRNA{BColors.ENDC}

  {BColors.BOLD}2. Numeric-Only Mode:{BColors.ENDC}
     Outputs only the numeric fair value(s) for a single ticker, useful for scripting.
     Can specify one or both sources; output order matches argument order.
     {BColors.OKCYAN}USAGE:{BColors.ENDC} {BColors.OKGREEN}python get_fair_value.py MRNA -as -vi{BColors.ENDC}

  {BColors.BOLD}3. Batch File Mode:{BColors.ENDC}
     Reads tickers from an input file, fetches fair values for each, and saves
     results to a CSV file.
     {BColors.OKCYAN}USAGE:{BColors.ENDC} {BColors.OKGREEN}python get_fair_value.py -i tickers.txt -o values.csv{BColors.ENDC}

{BColors.HEADER}About the Sources:{BColors.ENDC}
  {BColors.BOLD}AlphaSpread{BColors.ENDC}
     Uses a dual-model approach: DCF (Intrinsic) and Relative Valuation (Multiples).
     Accuracy is considered {BColors.WARNING}Moderate{BColors.ENDC}.

  {BColors.BOLD}ValueInvesting.io{BColors.ENDC}
     Applies a heavy focus on DCF and WACC calculations. Accuracy is considered
     {BColors.WARNING}Moderate{BColors.ENDC} but can be overly optimistic on growth stocks, i.e. tech stocks
     where Terminal Value is 80% of the score.

  {BColors.BOLD}GuruFocus{BColors.ENDC}
     Uses multiple models: GF Value, Peter Lynch Fair Value, and Project FCF.
     Accuracy is considered {BColors.OKGREEN}High{BColors.ENDC}. It is less prone to extreme automated DCF swings.

  {BColors.BOLD}Simply Wall St{BColors.ENDC}
     Uses a 2-Stage FCFE (Free Cash Flow to Equity) model. Accuracy is considered
     {BColors.FAIL}Low to Moderate{BColors.ENDC} as it's criticized for being too optimistic. Because it is
     fully automated, it frequently misses one-time accounting charges or structural
     industry shifts.
"""
    
    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("ticker", nargs='?', default=None, help=f"Stock ticker symbol (e.g., {BColors.OKGREEN}MRNA{BColors.ENDC}). Used in single-ticker modes.")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    
    # Batch mode arguments
    parser.add_argument("-i", "--input", help="Path to the input file containing tickers (for batch mode).")
    parser.add_argument("-o", "--output", default='fair_values.csv', help="Path to the output CSV file (for batch mode).")

    # Flags for numeric-only output. Can be combined.
    parser.add_argument("-as", "--alphaspread-only", action="store_true",
                       help="Output only the numeric value from AlphaSpread.")
    parser.add_argument("-vi", "--valueinvesting-only", action="store_true",
                       help="Output only the numeric value from ValueInvesting.io.")
    parser.add_argument("-gf", "--gurufocus-only", action="store_true",
                       help="Output only the numeric value from GuruFocus.")
    parser.add_argument("-sw", "--simplywallst-only", action="store_true",
                       help="Output only the numeric value from Simply Wall St.")
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging to show retrieval details (ignored in numeric-only or batch modes).")
    args = parser.parse_args()

    # Handle modes
    if not args.ticker and not args.input:
        parser.print_help()
        sys.exit(1)

    # Configure logging based on debug flag
    log_format = '%(levelname)s:%(name)s:%(message)s'
    if args.debug:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        # Only show warnings and errors by default
        logging.basicConfig(level=logging.WARNING, format=log_format)

    is_numeric_only_mode = args.alphaspread_only or args.valueinvesting_only or args.gurufocus_only or args.simplywallst_only
    is_batch_mode = args.input is not None

    # Initialize the scraper. It will be used in all modes.
    # Show the browser window only when running a full report for a single ticker in debug mode.
    show_browser = args.debug and not is_numeric_only_mode and not is_batch_mode
    scraper = FairValueScraper(headless=not show_browser)
    try:
        if args.input:
            # Batch File Mode
            process_ticker_file_fair_value(args.input, args.output, scraper, debug=args.debug)
        elif args.ticker:
            # All single-ticker modes
            if args.alphaspread_only or args.valueinvesting_only or args.gurufocus_only or args.simplywallst_only:
                # Numeric-only mode. Process in a fixed, predictable order.
                outputs = []
                if args.alphaspread_only:
                    result_text = scraper.get_fair_value_alphaspread(args.ticker)
                    numeric_value = parse_value_from_string(result_text)
                    outputs.append(f"{numeric_value if numeric_value is not None else ''}")
                
                if args.valueinvesting_only:
                    result_text = scraper.get_fair_value_valueinvesting_io(args.ticker)
                    numeric_value = parse_value_from_string(result_text)
                    outputs.append(f"{numeric_value if numeric_value is not None else ''}")

                if args.gurufocus_only:
                    result_text = scraper.get_fair_value_gurufocus(args.ticker)
                    numeric_value = parse_value_from_string(result_text)
                    outputs.append(f"{numeric_value if numeric_value is not None else ''}")

                if args.simplywallst_only:
                    result_text = scraper.get_fair_value_simplywallst(args.ticker)
                    numeric_value = parse_value_from_string(result_text)
                    outputs.append(f"{numeric_value if numeric_value is not None else ''}")
                
                print('\n'.join(outputs))
                return

            # Full report mode (default single-ticker mode)
            print(f"\nFetching fair value estimates for: {BColors.BOLD}{args.ticker.upper()}{BColors.ENDC}\n")

            results = scraper.fetch_all_parallel(args.ticker)
            alpha_result_text        = results.get("AlphaSpread")
            valueinvesting_result_text = results.get("ValueInvesting.io")
            gurufocus_result_text    = results.get("GuruFocus")
            simplywallst_result_text = results.get("Simply Wall St")

            ticker_upper = args.ticker.upper()
            ticker_lower = args.ticker.lower()
            results_data = [
                {"source": "AlphaSpread", "text": alpha_result_text, "value": parse_value_from_string(alpha_result_text), "url": f"https://www.alphaspread.com/security/nasdaq/{ticker_lower}/summary"},
                {"source": "ValueInvesting.io", "text": valueinvesting_result_text, "value": parse_value_from_string(valueinvesting_result_text), "url": f"https://valueinvesting.io/{ticker_upper}/valuation/fair-value"},
                {"source": "GuruFocus", "text": gurufocus_result_text, "value": parse_value_from_string(gurufocus_result_text), "url": f"https://www.gurufocus.com/stock/{ticker_upper}/dcf"},
                {"source": "Simply Wall St", "text": simplywallst_result_text, "value": parse_value_from_string(simplywallst_result_text), "url": f"https://simplywall.st/search?q={ticker_upper}"},
            ]
        
            # --- Analyze and print results with colors ---
            valid_values = [r['value'] for r in results_data if r['value'] is not None]
            min_val = min(valid_values) if valid_values else None
            max_val = max(valid_values) if valid_values else None
        
            def hyperlink(url, text):
                return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"

            for result in results_data:
                color = ''
                display_text = f"{result['source']}: "

                if result['value'] is not None:
                    display_text += f"Fair Value = ${result['value']}"
                    # Only color if there are at least two different values to compare
                    if min_val is not None and max_val is not None and min_val != max_val:
                        if result['value'] == max_val:
                            color = BColors.OKGREEN
                        elif result['value'] == min_val:
                            color = BColors.FAIL
                elif "Error" in str(result['text']) or "not found" in str(result['text']):
                    color = BColors.WARNING
                    display_text += result['text']
                else:
                    display_text += result['text']

                print(f"{color}{display_text}{BColors.ENDC if color else ''}")
                if result['url']:
                    print(f"  {hyperlink(result['url'], result['url'])}")
        
            print()
    finally:
        # Ensure the browser is closed gracefully
        scraper.close()

if __name__ == "__main__":
    main()
