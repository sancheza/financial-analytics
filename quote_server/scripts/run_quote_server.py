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
import functools
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn

from quote_server import providers
from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher
from quote_server.watchdog import Watchdog

VERSION = "1.0.0"

# Seconds between full refreshes of every known symbol. Override with --refresh-seconds.
REFRESH_SECONDS = 300

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
BLOCK_FILE_NAME = "quote_server_blocks.json"   # kept next to the cache file
DEFAULT_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "quote_server_cache.json"


LEVEL_COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[38;5;214m",
                "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}   # dev ADR 0020
COLOR_RESET = "\033[0m"
EVENT_FIELDS = ("scope", "action", "outcome", "target")           # dev ADR 0008


class EventFormatter(logging.Formatter):
    """UTC ISO 8601 timestamp, aligned level label, message, then the event fields."""

    converter = time.gmtime

    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
        self._color = color

    def formatMessage(self, record: logging.LogRecord) -> str:
        level = record.levelname
        shown = f"{LEVEL_COLORS.get(level, '')}{level}{COLOR_RESET}" if self._color else level
        label = f"{shown}:{' ' * max(1, 8 - len(level))} "
        fields = " ".join(f"{name}={getattr(record, name)}" for name in EVENT_FIELDS
                          if getattr(record, name, None))
        return f"{record.asctime} {label}{record.message}" + (f"  [{fields}]" if fields else "")


def use_color() -> bool:
    """Color only an interactive terminal, unless NO_COLOR or FORCE_COLOR says otherwise."""
    if "NO_COLOR" in os.environ:
        return False
    return "FORCE_COLOR" in os.environ or sys.stdout.isatty()


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
    parser.add_argument("--request-delay", type=float, default=providers.REQUEST_DELAY_SECONDS,
                        help="seconds between the starts of two requests to one price source "
                             "(default: %(default)s)")
    parser.add_argument("--restart-on-hang", action="store_true",
                        help="exit when the refresh stops or hangs, so a supervisor (the LaunchAgent) "
                             "starts a new server; without it the problem is logged and notified")
    args = parser.parse_args()
    if args.refresh_seconds < 5:
        parser.error("--refresh-seconds must be at least 5")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(EventFormatter(use_color()))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.captureWarnings(True)
    cache = QuoteCache(args.cache_file)
    loaded = cache.load()
    logging.info("quote_server %s starting on %s:%d, refresh every %gs, %d symbols loaded",
                 VERSION, HOST, args.port, args.refresh_seconds, loaded)
    if args.request_delay < 0:
        parser.error("--request-delay must not be negative")
    client = providers.make_client(block_file=args.cache_file.with_name(BLOCK_FILE_NAME),
                                   request_delay=args.request_delay)
    refresher = Refresher(cache, args.refresh_seconds,
                          fetch_many=functools.partial(providers.fetch_many, client=client))
    watchdog = Watchdog(refresher, args.refresh_seconds, restart=args.restart_on_hang, save=cache.save)
    app = create_app(cache, refresher, watchdog, client)
    try:
        uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
