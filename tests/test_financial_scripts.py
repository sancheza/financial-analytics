#!/usr/bin/env python3

"""
Financial Analytics Test Suite
------------------------------
Path: tests/test_financial_scripts.py (relative to the repo root)
Version: run `python3 tests/test_financial_scripts.py --version` to see the
         current version (also usable directly with pytest as a plugin file).

This docstring is also this script's --help output (get_args() passes __doc__
straight to argparse) and the pydoc output (`python3 -m pydoc
./tests/test_financial_scripts.py`) - there is one copy of this guide, not three.

WHAT THIS IS:
    A centralized validation framework for the financial-analytics project.
    Rather than unit-testing each script's internals, it runs every script as
    a real subprocess against its real .env configuration and checks the
    result the way a human would: exit code 0, plus an expected shape in
    stdout (or, for a couple of scripts, an expected file side effect). That
    makes it both a regression suite (did a change break this script?) and an
    early-warning system for upstream drift (did FRED / Yahoo Finance / Alpha
    Vantage change their response shape under us?).

HOW IT WORKS:
    - SCRIPT_TESTS is a list of (script_path, args, validation_regex, needs_env)
      tuples, parametrized into test_script_execution via
      @pytest.mark.parametrize. Each entry runs `python3 <script_path> <args>`
      in an isolated subprocess, then asserts returncode == 0 and
      re.search(validation_regex, stdout).
    - needs_env is a narrow flag: True means "fail explicitly, with a clear
      message, if ALPHA_VANTAGE_KEY is missing" instead of letting the
      subprocess fail more cryptically on its own. It does not generalize to
      other keys - see ADDING A NEW TEST CASE below.
    - Scripts whose success is a file side effect rather than distinctive
      stdout (e.g. generate_auction_calendar.py's .ics output) get their own
      dedicated test function instead of a SCRIPT_TESTS entry - see
      test_auction_calendar_side_effects.
    - BASE_DIR/BOND_DIR are derived from this file's own location
      (repo_root/tests/this_file), not hardcoded, so the suite works no
      matter where the repo is cloned.
    - Every run logs to both the console and test_summary.log, written to
      whatever directory you invoked pytest from (typically the repo root).

GETTING STARTED (new contributors start here):
    1. Install dependencies (pytest, python-dotenv, plus whatever the target
       scripts themselves need - requests, beautifulsoup4, etc.):
           pip install -r requirements.txt   # or install pytest/python-dotenv directly

    2. Provide the .env files the scripts under test read from. Missing keys
       cause the *target script* to exit non-zero, which shows up here as a
       test failure - not a bug in this suite:
           <repo_root>/.env                ALPHA_VANTAGE_KEY=...
           <repo_root>/bond-analytics/.env FRED_API_KEY=...
       (EMAIL_FROM/EMAIL_TO/SMTP_USER/SMTP_PASS are only needed if you run
       bond_alert.py's live email-alert path yourself; this suite's bond_alert.py
       test case doesn't exercise that path.)

    3. Run it:
           pytest tests/test_financial_scripts.py -v                 # full suite
           pytest tests/test_financial_scripts.py -v -m "not slow"   # skip slow cases
           pytest tests/test_financial_scripts.py -v -k bond_alert   # one script only
           pytest tests/test_financial_scripts.py -v --log-file=test_results.log
           python3 tests/test_financial_scripts.py --run-now [--fast]  # no pytest CLI needed

    NOTE ON NETWORK ACCESS: almost every test here runs a real script that hits
    a live API or scrapes a live web page (FRED, Yahoo Finance, Alpha Vantage,
    etc.) - there are no mocks. A failure often means the upstream source
    changed its response shape, not that this suite regressed. Network access
    and valid API keys are required to get a clean run.

SCRIPTS COVERED:
    - Equity Analytics: get_dividend.py, get_earnings_date.py,
      get_ex_dividend.py, get_fair_value.py, get_forwardpe.py, get_pb.py,
      get_projected_fcf.py, get_ticker_range.py, get_yearly_performance.py,
      value_price_screener.py, value_screener_peak_drawdown.py, value_screener.py
    - Bond Analytics: bond_alert.py, bond_market_analyzer.py,
      bond_market_analyzer_viewer.py, bond_return_calc.py, calculate_YTW.py,
      generate_auction_calendar.py, get_bond_yield.py
    See SCRIPT_TESTS below for the exact args/pattern used per script.

CORE VALIDATIONS:
    - Process Integrity: Verifies scripts exit with code 0.
    - Data Integrity: Matches stdout against script-specific Regex patterns.
    - Env Integrity: Checks for mandatory keys like ALPHA_VANTAGE_KEY.
    - Side Effects: Confirms file generation (e.g., .ics files).

ADDING A NEW TEST CASE:
    Append a tuple to SCRIPT_TESTS: (script_path, args, validation_regex, needs_env).
        - script_path:      absolute path, built from BASE_DIR or BOND_DIR.
        - args:             list of CLI args to pass to the script.
        - validation_regex: pattern that must appear somewhere in stdout.
        - needs_env:        True only if the script requires ALPHA_VANTAGE_KEY
                             specifically - this flag doesn't generalize to other
                             keys. If you add a script that needs a different key
                             (e.g. FRED_API_KEY), extend the env check inside
                             test_script_execution rather than reusing this flag.
    Wrap the tuple in pytest.param(..., marks=pytest.mark.slow) for anything
    that's expensive to run (see value_price_screener.py below).
    For scripts whose success shows up as a file side effect rather than
    distinctive stdout, add a dedicated test function instead of a SCRIPT_TESTS
    entry (see test_auction_calendar_side_effects for the pattern).

EXIT CODES:
    0: Success / help displayed
    1: Test failure or environment error

DEPENDENCIES:
    - pytest
    - python-dotenv
    - subprocess (stdlib)
    - re (stdlib)
"""

import subprocess
import pytest
import os
import re
import argparse
import sys
import logging
from dotenv import load_dotenv

# Setup Logging
# Configured to provide both persistent file logs and real-time console output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("test_summary.log"), logging.StreamHandler()],
)

# Load environment variables from .env
# Required for scripts that interface with the Alpha Vantage API.
load_dotenv()

# Constants
# BASE_DIR is derived from this file's own location (repo_root/tests/this_file)
# rather than a hardcoded path, so the suite works regardless of where the repo
# is cloned.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOND_DIR = os.path.join(BASE_DIR, "bond-analytics")
VERSION = "1.3.0"


def pytest_configure(config):
    """
    Register custom markers to suppress PytestUnknownMarkWarning.
    This informs pytest that 'slow' is a valid marker for deselecting
    long-running scripts using the -m flag.

    NOTE: This hook is executed by pytest during the configuration phase.
    If the warning persists, it indicates pytest is collecting markers
    before this hook completes, requiring a pytest.ini registration.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def get_args():
    """
    Build the argparse parser used for direct (non-pytest) invocation.

    Reuses this module's own docstring as the --help description instead of
    maintaining a second copy of the same guide, so the module docstring,
    `--help` output, and `pydoc` output are always identical.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    parser.add_argument(
        "--run-now", action="store_true", help="Trigger pytest execution immediately."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Skip long-running scripts (marked 'slow')."
    )
    return parser


# Updated Regex for "Month DD, YYYY"
# This pattern matches standard US financial date formatting used in reports.
DATE_PATTERN = r"[A-Z][a-z]{2}\s\d{1,2},\s\d{4}"

# Configuration Mapping: (script_path, args, validation_regex, needs_env)
# This list defines the test matrix for the entire suite.
# Entries with pytest.param allow for metadata injection (e.g., markers).
SCRIPT_TESTS = [
    (os.path.join(BASE_DIR, "get_earnings_date.py"), ["PFE"], DATE_PATTERN, False),
    (os.path.join(BASE_DIR, "get_ex_dividend.py"), ["PFE"], DATE_PATTERN, False),
    (os.path.join(BASE_DIR, "get_dividend.py"), ["PFE"], r"\d+\.?\d*", False),
    (os.path.join(BASE_DIR, "get_fair_value.py"), ["PFE"], r"\$?\d+", False),
    (os.path.join(BASE_DIR, "get_forwardpe.py"), ["PFE"], r"\d+\.?\d*", False),
    (os.path.join(BASE_DIR, "get_pb.py"), ["PFE"], r"\d+\.?\d*", False),
    (os.path.join(BASE_DIR, "get_projected_fcf.py"), ["PFE"], r"\$?\d+", False),
    (os.path.join(BASE_DIR, "get_ticker_range.py"), [], r".+", False),
    (
        os.path.join(BASE_DIR, "get_yearly_performance.py"),
        ["PFE", "2023"],
        r".+",
        False,
    ),
    # Long-running script marked as slow for use with the -m "not slow" flag.
    pytest.param(
        os.path.join(BASE_DIR, "value_price_screener.py"),
        ["2024"],
        r".+",
        False,
        marks=pytest.mark.slow,
    ),
    (os.path.join(BASE_DIR, "value_screener_peak_drawdown.py"), [], r".+", True),
    (os.path.join(BASE_DIR, "value_screener.py"), ["--verbosity", "2"], r".+", False),
    # Bond Analytics
    (os.path.join(BOND_DIR, "bond_alert.py"), [], r"\d+\.?\d*%", False),
    (os.path.join(BOND_DIR, "bond_market_analyzer_viewer.py"), [], r".+", False),
    (os.path.join(BOND_DIR, "bond_market_analyzer.py"), ["-h"], r".+", False),
    (
        os.path.join(BOND_DIR, "bond_return_calc.py"),
        ["4.625", "01/31/2027", "99.75"],
        r".+",
        False,
    ),
    (
        os.path.join(BOND_DIR, "calculate_YTW.py"),
        ["4.625", "01/31/2027", "99.75"],
        r"\d+\.?\d*%",
        False,
    ),
    (os.path.join(BOND_DIR, "get_bond_yield.py"), [], r"\d+\.?\d*%", False),
]


@pytest.mark.parametrize("script_path, args, pattern, needs_env", SCRIPT_TESTS)
def test_script_execution(script_path, args, pattern, needs_env):
    """
    Executes scripts and validates exit codes and output patterns.
    Ensures that each script is functional in its standalone state.
    """
    logging.info(f"Testing script: {os.path.basename(script_path)} with args {args}")

    if not os.path.exists(script_path):
        logging.error(f"Path not found: {script_path}")
        pytest.fail(f"Script not found: {script_path}")

    if needs_env and not os.getenv("ALPHA_VANTAGE_KEY"):
        logging.error(
            f"Environment check failed: ALPHA_VANTAGE_KEY missing for {script_path}."
        )
        pytest.fail("Missing ALPHA_VANTAGE_KEY")

    # Isolation check using subprocess
    result = subprocess.run(
        [sys.executable, script_path] + args,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        logging.error(f"Execution Failed: {script_path}\nStderr: {result.stderr}")

    assert (
        result.returncode == 0
    ), f"Script {script_path} failed with return code {result.returncode}"
    assert re.search(
        pattern, result.stdout
    ), f"Output of {script_path} did not match pattern {pattern}"
    logging.info(f"Successfully validated {os.path.basename(script_path)}")


def test_auction_calendar_side_effects():
    """
    Validates generate_auction_calendar.py specifically for file creation.
    Ensures an .ics file is generated and contains at least one VEVENT.
    Cleans up generated files after validation to maintain a clean environment.
    """
    logging.info("Testing side effects for generate_auction_calendar.py")
    script_path = os.path.join(BOND_DIR, "generate_auction_calendar.py")

    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")

    subprocess.run([sys.executable, script_path, "--minimum"], check=True)

    ics_files = [f for f in os.listdir(".") if f.endswith(".ics")]
    try:
        assert len(ics_files) > 0, "No .ics file was generated."
        with open(ics_files[0], "r") as f:
            content = f.read()
            assert "BEGIN:VEVENT" in content, "Generated ICS file is empty of events."
        logging.info(
            f"Successfully validated side effects for {os.path.basename(script_path)}"
        )
    finally:
        for f in ics_files:
            os.remove(f)


if __name__ == "__main__":
    parser = get_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args()

    if parsed_args.run_now:
        # Trigger pytest programmatically
        pytest_args = [__file__, "-v"]
        if parsed_args.fast:
            pytest_args.extend(["-m", "not slow"])
        sys.exit(pytest.main(pytest_args))
