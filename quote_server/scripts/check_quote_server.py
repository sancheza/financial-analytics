#!/usr/bin/env python3
"""check_quote_server.py

Report the state of the quote server LaunchAgent: whether launchd has it, whether its
process is running, how many files it has open, whether /health answers, and what the
log says since the last start. Changes nothing.

Usage:
    check_quote_server.py
    check_quote_server.py --port 8800
    check_quote_server.py --log-lines 20

Exit status: 0 when no problem is found, 1 when at least one is.
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[2]
LABEL = f"com.{getpass.getuser()}.quoteserver"
LOG_FILE = REPO / "logs" / "quote_server.log"
DEFAULT_PORT = 8765

HEALTH_TIMEOUT_SECONDS = 3.0
FILES_WARNING_FRACTION = 0.8       # report a problem when this share of the file limit is open
LOG_TAIL_BYTES = 2_000_000         # how much of the end of the log is read

START_LINE = "starting on"
LOG_PROBLEMS = (                   # text in a log line -> what it means
    ("Too many open files", "ran out of open files"),
    ("refresh crashed", "a refresh crashed"),
    ("Traceback", "an exception was logged"),
)


def launchctl_print(domain: str) -> str | None:
    """Return launchd's description of the job, or None when launchd does not have it."""
    result = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def parse_launchd(text: str) -> dict[str, str]:
    """Return the `state`, `pid`, `runs`, `last exit code` and `path` lines of `launchctl print`."""
    wanted = ("state", "pid", "runs", "last exit code", "path")
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\t(\w[\w ]*?) = (.*)$", line)     # one tab: the job's own settings
        if match and match.group(1) in wanted and match.group(1) not in found:
            found[match.group(1)] = match.group(2).strip()
    return found


def count_descriptors(lsof_text: str) -> tuple[int, dict[str, int], int]:
    """Count the file descriptors in `lsof -p` output.

    Only rows with a numbered FD count; `cwd`, `txt` and `mem` rows are not descriptors.

    Returns:
        (descriptors, count by type such as PIPE or IPv4, sockets in CLOSE_WAIT)
    """
    by_type: dict[str, int] = {}
    close_wait = 0
    for line in lsof_text.splitlines()[1:]:
        columns = line.split()
        if len(columns) < 5 or not re.fullmatch(r"\d+[urw]?", columns[3]):
            continue
        by_type[columns[4]] = by_type.get(columns[4], 0) + 1
        close_wait += "(CLOSE_WAIT)" in line
    return sum(by_type.values()), by_type, close_wait


def read_health(port: int) -> tuple[dict | None, str]:
    """Ask /health.

    Returns:
        (the decoded reply, "") or (None, why there was none)
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=HEALTH_TIMEOUT_SECONDS) as reply:
            return json.loads(reply.read().decode("utf-8")), ""
    except urllib.error.URLError as exc:
        return None, str(exc.reason)
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def log_since_last_start(path: Path) -> list[str]:
    """Return the lines of the log from the most recent server start."""
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - LOG_TAIL_BYTES))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if START_LINE in line]
    return lines[starts[-1]:] if starts else lines


def log_problems(lines: list[str]) -> list[str]:
    """Return one sentence per kind of problem found in the log lines, with its count."""
    found = []
    for text, meaning in LOG_PROBLEMS:
        count = sum(text in line for line in lines)
        if count:
            found.append(f"{meaning} ({count} log lines contain '{text}')")
    return found


def process_age(pid: str) -> str:
    """Return how long the process has been running, as `ps` prints it ([dd-][hh:]mm:ss)."""
    age = subprocess.run(["ps", "-o", "etime=", "-p", pid], capture_output=True, text=True).stdout.strip()
    return age or "unknown"


def main() -> int:
    """Print the report and return 1 when a problem was found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port the server listens on (default: %(default)s)")
    parser.add_argument("--log-lines", type=int, default=5,
                        help="how many of the last ERROR and WARNING log lines to show (default: %(default)s)")
    args = parser.parse_args()

    problems: list[str] = []
    domain = f"gui/{os.getuid()}"

    print(f"== launchd: {LABEL}")
    described = launchctl_print(domain)
    pid = None
    if described is None:
        print("not loaded")
        problems.append(f"launchd does not have {LABEL}; install it with scripts/install_launch_agent.py")
    else:
        job = parse_launchd(described)
        for key in ("state", "pid", "runs", "last exit code", "path"):
            print(f"{key + ':':16}{job.get(key, '-')}")
        pid = job.get("pid")
        if job.get("state") != "running" or pid is None:
            problems.append("launchd has the job but no process is running")

    print("\n== process")
    if pid:
        print(f"{'running for:':16}{process_age(pid)}")
        lsof = subprocess.run(["lsof", "-nP", "-p", pid], capture_output=True, text=True).stdout
        descriptors, by_type, close_wait = count_descriptors(lsof)
        breakdown = ", ".join(f"{name} {n}" for name, n in sorted(by_type.items(), key=lambda i: -i[1]))
        print(f"{'open files:':16}{descriptors}  ({breakdown})")
        print(f"{'CLOSE_WAIT:':16}{close_wait} sockets the server has not closed")
    else:
        print("no process")

    print(f"\n== http://127.0.0.1:{args.port}/health")
    health, why = read_health(args.port)
    if health is None:
        print(f"no answer: {why}")
        problems.append("the server does not answer /health"
                        + (" although its process is running" if pid else ""))
    else:
        used, limit = health.get("open_files"), health.get("open_files_limit")
        for key in ("status", "version", "symbols", "refresher_alive", "last_cycle_at"):
            print(f"{key + ':':16}{health.get(key)}")
        print(f"{'open files:':16}{used} of {limit}")
        if health.get("status") != "ok":
            problems.append("/health reports status " + str(health.get("status")))
        if used and limit and used >= FILES_WARNING_FRACTION * limit:
            problems.append(f"{used} of {limit} files are open")

    print(f"\n== log since the last start: {LOG_FILE}")
    lines = log_since_last_start(LOG_FILE)
    if not lines:
        print("no log")
    else:
        problems += log_problems(lines)
        recent = [line for line in lines if " ERROR: " in line or " WARNING: " in line][-args.log_lines:]
        print("\n".join(recent) if recent else "no ERROR or WARNING lines")

    print("\n== result")
    if problems:
        print("\n".join(f"PROBLEM: {p}" for p in problems))
        print(f"Restart: launchctl kickstart -k {domain}/{LABEL}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
