#!/usr/bin/env python3
"""run_quote_server.py

Serve cached stock and Treasury bond prices to spreadsheets over local HTTP.

The server binds to 127.0.0.1 only. Every request is answered from memory; a
background thread refreshes each symbol that has ever been requested.

Usage:
    run_quote_server.py
    run_quote_server.py --refresh-seconds 60
    run_quote_server.py --port 8800 --cache-file /path/to/cache.json
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher

VERSION = "1.0.0"

# Seconds between full refreshes of every known symbol. Override with --refresh-seconds.
REFRESH_SECONDS = 300

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "quote_server_cache.json"


def main() -> int:
    """Parse arguments and run the server."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port on 127.0.0.1 (default: %(default)s)")
    parser.add_argument("--refresh-seconds", type=float, default=REFRESH_SECONDS,
                        help="seconds between full refreshes (default: %(default)s)")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
                        help="file that keeps the symbol list across restarts (default: %(default)s)")
    args = parser.parse_args()
    if args.refresh_seconds < 5:
        parser.error("--refresh-seconds must be at least 5")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    cache = QuoteCache(args.cache_file)
    loaded = cache.load()
    logging.info("quote_server %s starting on %s:%d, refresh every %gs, %d symbols loaded",
                 VERSION, HOST, args.port, args.refresh_seconds, loaded)
    app = create_app(cache, Refresher(cache, args.refresh_seconds))
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
