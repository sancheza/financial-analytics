#!/usr/bin/env python3
"""install_launch_agent.py

Install (or remove) the macOS LaunchAgent that starts the quote server at login
and restarts it if it exits, or if its refresh stops or hangs (--restart-on-hang).

Safe to re-run: the plist is regenerated and reloaded each time. The installer waits for
the old job to finish unloading, loads the new one (retrying if launchd is still busy),
and checks that the server answers before it reports success.

Usage:
    install_launch_agent.py
    install_launch_agent.py --refresh-seconds 60
    install_launch_agent.py --dry-run
    install_launch_agent.py --uninstall
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO / "quote_server" / "scripts" / "run_quote_server.py"
LABEL = f"com.{getpass.getuser()}.quoteserver"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_FILE = REPO / "logs" / "quote_server.log"

DEFAULT_PORT = 8765                # the server's own default (run_quote_server.py)

# After `launchctl bootout`, the job takes about 5 seconds to disappear (measured 3.8 to 5.3 s),
# and `launchctl bootstrap` fails with "5: Input/output error" until it has.
UNLOAD_TIMEOUT_SECONDS = 30.0
UNLOAD_POLL_SECONDS = 0.25
LOAD_ATTEMPTS = 4
LOAD_RETRY_SECONDS = 2.0           # the wait before attempt n+1 is n times this
START_TIMEOUT_SECONDS = 30.0
START_POLL_SECONDS = 0.5

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}
    </array>
    <key>WorkingDirectory</key>
    <string>{workdir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"""


def build_plist(extra_args: list[str]) -> str:
    """Return the plist text that runs the server with the venv's Python."""
    arguments = [sys.executable, str(SERVER_SCRIPT), *extra_args]
    return PLIST_TEMPLATE.format(
        label=LABEL,
        arguments="\n".join(f"        <string>{a}</string>" for a in arguments),
        workdir=REPO,
        log=LOG_FILE,
    )


def launchctl(*args: str) -> subprocess.CompletedProcess:
    """Run launchctl and capture its output."""
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def is_loaded(domain: str) -> bool:
    """Return True while launchd still has the job."""
    return launchctl("print", f"{domain}/{LABEL}").returncode == 0


def unload(domain: str, sleep=time.sleep, clock=time.monotonic) -> bool:
    """Unload the job and wait until launchd has finished removing it.

    Returns:
        True when the job is gone (or was never loaded); False if it is still there
        after UNLOAD_TIMEOUT_SECONDS.
    """
    launchctl("bootout", f"{domain}/{LABEL}")   # fails harmlessly when not loaded
    deadline = clock() + UNLOAD_TIMEOUT_SECONDS
    while is_loaded(domain):
        if clock() >= deadline:
            return False
        sleep(UNLOAD_POLL_SECONDS)
    return True


def load(domain: str, sleep=time.sleep, clock=time.monotonic) -> subprocess.CompletedProcess:
    """Load the job, unloading any old copy first and retrying when launchd refuses.

    Returns:
        The result of the last `launchctl bootstrap`; returncode 0 means the job is loaded.
    """
    result = None
    for attempt in range(1, LOAD_ATTEMPTS + 1):
        if not unload(domain, sleep, clock):
            print(f"{LABEL} is still loaded after {UNLOAD_TIMEOUT_SECONDS:g} s; trying to load anyway")
        result = launchctl("bootstrap", domain, str(PLIST_PATH))
        if result.returncode == 0:
            return result
        if attempt < LOAD_ATTEMPTS:
            wait = LOAD_RETRY_SECONDS * attempt
            reason = (result.stderr.strip().splitlines() or ["no message"])[0]
            print(f"launchctl bootstrap failed ({reason}); attempt {attempt} of {LOAD_ATTEMPTS}, "
                  f"retrying in {wait:g} s")
            sleep(wait)
    return result


def wait_for_server(port: int, sleep=time.sleep, clock=time.monotonic) -> dict | None:
    """Wait for the server's /health to answer.

    Returns:
        The decoded /health reply, or None if nothing answered within START_TIMEOUT_SECONDS.
    """
    deadline = clock() + START_TIMEOUT_SECONDS
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            if clock() >= deadline:
                return None
            sleep(START_POLL_SECONDS)


def main() -> int:
    """Parse arguments and install or remove the LaunchAgent."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--refresh-seconds", type=float,
                        help="pass --refresh-seconds to the server (default: the server's own)")
    parser.add_argument("--port", type=int, help="pass --port to the server")
    parser.add_argument("--dry-run", action="store_true", help="print the plist; change nothing")
    parser.add_argument("--uninstall", action="store_true", help="unload and delete the LaunchAgent")
    args = parser.parse_args()

    domain = f"gui/{os.getuid()}"
    if args.uninstall:
        if args.dry_run:
            print(f"Would unload {LABEL} and delete {PLIST_PATH}")
            return 0
        if not unload(domain):
            print(f"{LABEL} is still loaded after {UNLOAD_TIMEOUT_SECONDS:g} s", file=sys.stderr)
            return 1
        PLIST_PATH.unlink(missing_ok=True)
        print(f"Removed {LABEL}")
        return 0

    extra: list[str] = ["--restart-on-hang"]
    if args.refresh_seconds is not None:
        extra += ["--refresh-seconds", str(args.refresh_seconds)]
    if args.port is not None:
        extra += ["--port", str(args.port)]
    plist = build_plist(extra)
    if args.dry_run:
        print(f"Would write {PLIST_PATH} and load it:\n")
        print(plist)
        return 0

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist)
    result = load(domain)
    if result.returncode != 0:
        print(f"launchctl could not load {LABEL} after {LOAD_ATTEMPTS} attempts: "
              f"{result.stderr.strip() or 'no message'}\n"
              f"The server is not running. Load it with:\n  launchctl bootstrap {domain} {PLIST_PATH}",
              file=sys.stderr)
        return 1
    print(f"Loaded {LABEL}. Log: {LOG_FILE}")
    health = wait_for_server(args.port or DEFAULT_PORT)
    if health is None:
        print(f"The server did not answer on port {args.port or DEFAULT_PORT} within "
              f"{START_TIMEOUT_SECONDS:g} s. Read the log: {LOG_FILE}", file=sys.stderr)
        return 1
    print(f"Server answering on port {args.port or DEFAULT_PORT}: refresh every "
          f"{health.get('refresh_seconds')} s, {health.get('symbols')} symbols")
    print(f"Status:  launchctl list | grep {LABEL}")
    print(f"Restart: launchctl kickstart -k {domain}/{LABEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
