#!/usr/bin/env python3
"""install_launch_agent.py

Install (or remove) the macOS LaunchAgent that starts the quote server at login
and restarts it if it exits.

Safe to re-run: the plist is regenerated and reloaded each time.

Usage:
    install_launch_agent.py
    install_launch_agent.py --refresh-seconds 60
    install_launch_agent.py --dry-run
    install_launch_agent.py --uninstall
"""

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO / "quote_server" / "scripts" / "run_quote_server.py"
LABEL = f"com.{getpass.getuser()}.quoteserver"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_FILE = REPO / "logs" / "quote_server.log"

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
        launchctl("bootout", f"{domain}/{LABEL}")
        PLIST_PATH.unlink(missing_ok=True)
        print(f"Removed {LABEL}")
        return 0

    extra: list[str] = []
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
    launchctl("bootout", f"{domain}/{LABEL}")  # fails harmlessly when not loaded
    result = launchctl("bootstrap", domain, str(PLIST_PATH))
    if result.returncode != 0:
        print(f"launchctl bootstrap failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"Loaded {LABEL}. Log: {LOG_FILE}")
    print(f"Status:  launchctl list | grep {LABEL}")
    print(f"Restart: launchctl kickstart -k {domain}/{LABEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
