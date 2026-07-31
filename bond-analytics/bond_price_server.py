#!/usr/bin/env python3
"""Serve an interactive, multi-bond price comparison chart in the browser.

Serves the static page in web/ (checkboxes to show/hide bonds, 1M/3M/6M/1Y/All
range buttons, line chart) and a small JSON API the page calls on every load:
  GET /api/watchlist  -- the CUSIP list from data/json/bond_price_watchlist.json
  GET /api/history    -- fetches each of those CUSIPs' historical price/yield
                          series from Webull server-side and returns it

The Webull fetch happens server-side (here), not in the page's own JS, because
a browser-side fetch straight to quotes-gw.webullfintech.com from this page
would be blocked by CORS -- Webull's API doesn't grant this origin permission.
Routing it through this server sidesteps that; the page only ever talks to
itself.

Each successful fetch is cached to data/json/bond_price_cache.json, keyed by
CUSIP. If a later fetch for some CUSIP fails (network error, Webull has no
listing, etc.) that CUSIP's last-good cached entry is served instead, marked
stale, rather than silently dropping it from the chart -- see fetch_one().

Reuses webull_bond_fetcher.py's page-scrape (fetch_treasury_price for the
coupon/maturity/issuer label, fetch_price_history for the historical series --
see that module's docstring for how fetch_price_history's endpoint was found
and its own reliability caveats, since it's an undocumented API with no SLA,
same class of fragility as the rest of that module's Webull scraping).

Stdlib only -- no pip install needed, so this runs the same way here and on a
bare Linux box.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import webull_bond_fetcher

VERSION = "1.0.0"

# Colors for the --help screen -- disabled when stdout isn't a terminal.
_USE_COLOR = sys.stdout.isatty()
CYAN = "\033[96m" if _USE_COLOR else ""
GREEN = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web")
WATCHLIST_FILE = os.path.join(SCRIPT_DIR, "data", "json", "bond_price_watchlist.json")
CACHE_FILE = os.path.join(SCRIPT_DIR, "data", "json", "bond_price_cache.json")


def load_watchlist():
    with open(WATCHLIST_FILE) as f:
        return json.load(f).get("cusips", [])


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_one(cusip, cache):
    """Fetch label + price history for a CUSIP. Falls back to the cached entry
    (marked stale) if the live fetch fails, so a transient error doesn't blank
    out a bond that fetched fine on a previous run.
    """
    try:
        quote = webull_bond_fetcher.fetch_treasury_price(cusip)
        history = webull_bond_fetcher.fetch_price_history(cusip)
        if not quote or not history:
            raise RuntimeError("no data returned")

        coupon = float(quote["couponRate"]) if quote.get("couponRate") else None
        coupon_str = f"{coupon:g}" if coupon is not None else "?"
        issuer = quote.get("issuerName") or cusip
        maturity = quote.get("maturityDate") or "?"

        entry = {
            "cusip": cusip,
            "issuer": issuer,
            "coupon": coupon,
            "maturity": maturity,
            "label": f"{issuer} {coupon_str}% {maturity}",
            "history": history,
            "fetchedAt": datetime.now().isoformat(),
            "stale": False,
            "error": None,
        }
        cache[cusip] = entry
        return entry
    except Exception as e:
        if cusip in cache:
            stale = dict(cache[cusip])
            stale["stale"] = True
            stale["error"] = str(e)
            return stale
        return {"cusip": cusip, "label": cusip, "history": [], "stale": True, "error": str(e)}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/watchlist":
            self._send_json({"cusips": load_watchlist()})
            return
        if self.path == "/api/history":
            cache = load_cache()
            results = {cusip: fetch_one(cusip, cache) for cusip in load_watchlist()}
            save_cache(cache)
            self._send_json(results)
            return
        super().do_GET()

    def _send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def print_help():
    """Print a formatted help screen: overview, usage, options, watchlist format, examples."""
    script_name = os.path.basename(sys.argv[0])
    rule = f"{CYAN}{'-' * 66}{RESET}"

    def opt(flag, desc):
        print(f"  {BOLD}{flag:<22}{RESET}{desc}")

    def example(desc, cmd):
        print(f"  {desc}")
        print(f"    {YELLOW}$ {cmd}{RESET}")

    print(f"\n{rule}")
    print(f"{BOLD}{GREEN}  Bond Price Comparison Server  {RESET}{CYAN}v{VERSION}{RESET}")
    print(rule)
    print("  Serves a local web page that plots price history for several bonds")
    print("  at once, with checkboxes to show/hide each one and buttons to change")
    print("  the visible time range. Data comes from Webull, fetched fresh on")
    print("  every page load and cached locally so a failed fetch for one CUSIP")
    print("  doesn't blank out the others.")

    print(f"\n{BOLD}{CYAN}USAGE{RESET}")
    print(f"  {script_name} [OPTIONS]")
    print(f"  Then open the printed URL in a browser. The page fetches its own")
    print(f"  data on load -- there's nothing else to run.")

    print(f"\n{BOLD}{CYAN}OPTIONS{RESET}")
    opt("--host HOST", "Bind address (default 127.0.0.1, local machine only).")
    opt("", "Use 0.0.0.0 to allow other machines on the network to view")
    opt("", "it too -- e.g. running this on a headless Linux box and")
    opt("", "viewing the chart from your laptop's browser.")
    opt("--port PORT", "Port to listen on (default 8765).")
    opt("-h, --help", "Show this help message and exit.")
    opt("-v, --version", "Show script version and exit.")

    print(f"\n{BOLD}{CYAN}WHICH CUSIPS ARE SHOWN{RESET}")
    print(f"  Edit {GREEN}{os.path.relpath(WATCHLIST_FILE, SCRIPT_DIR)}{RESET} (relative to this script) --")
    print(f"  that's the only place the bond list is defined. Format:")
    print(f'''  {YELLOW}{{"cusips": ["912810UF3", "912810UV8"]}}{RESET}''')
    print("  Add or remove CUSIPs there, then reload the page (or click Refresh)")
    print("  -- no restart needed, the file is re-read on every /api/watchlist")
    print("  and /api/history request.")

    print(f"\n{BOLD}{CYAN}DATA FILES{RESET}")
    print(f"  {os.path.relpath(WATCHLIST_FILE, SCRIPT_DIR):<40} bond list (hand-edited)")
    print(f"  {os.path.relpath(CACHE_FILE, SCRIPT_DIR):<40} last-good fetch per CUSIP (auto-written)")

    print(f"\n{BOLD}{CYAN}EXAMPLES{RESET}")
    example("Run locally, view in this machine's browser:", f"{script_name}")
    example("Run on a headless box, view from another machine:", f"{script_name} --host 0.0.0.0")
    example("Use a different port:", f"{script_name} --port 9000")

    print(f"\n{BOLD}{CYAN}NOTES{RESET}")
    print("  - The Webull fetch happens on THIS server, not in the page's own JS,")
    print("    because a browser-side fetch straight to Webull's API would be")
    print("    blocked by CORS. The page only ever talks to this server.")
    print("  - Webull's historical-price endpoint is undocumented (found by")
    print("    inspecting network requests, not a published API) -- see")
    print("    webull_bond_fetcher.py's module docstring for the full caveat.")
    print("  - A newly-issued or newly-reopened CUSIP may show only one data")
    print("    point -- that's it having little trading history yet, not a bug.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Bond price comparison chart server", add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.help:
        print_help()
        return
    if args.version:
        print(f"Bond Price Comparison Server version {VERSION}")
        return

    handler = partial(Handler, directory=WEB_DIR)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{BOLD}{GREEN}Serving bond price chart at http://{args.host}:{args.port}/{RESET}")
    print(f"Watchlist: {WATCHLIST_FILE}  (edit this file to add/remove CUSIPs)")
    print("Press Ctrl+C to stop. Run with --help for more.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
