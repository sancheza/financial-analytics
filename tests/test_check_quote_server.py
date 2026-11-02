"""Tests for check_quote_server.py: reading launchctl, lsof and log text."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "quote_server" / "scripts" / "check_quote_server.py"
spec = importlib.util.spec_from_file_location("check_quote_server", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

LAUNCHCTL = """gui/501/com.me.quoteserver = {
\tactive count = 1
\tpath = /Users/me/Library/LaunchAgents/com.me.quoteserver.plist
\ttype = LaunchAgent
\tstate = running

\tenvironment = {
\t\tstate => ignored
\t}

\truns = 3
\tpid = 18432
\tlast exit code = 70
}
"""

LSOF = """COMMAND   PID USER   FD      TYPE             DEVICE SIZE/OFF NODE NAME
Python  18432 me  cwd       DIR               1,16     1888 30811614 /repo
Python  18432 me  txt       REG               1,16    33568 268451482 /bin/python
Python  18432 me  mem       REG               1,16    33568 268451482 /lib/x.so
Python  18432 me    0u      CHR              16,2      0t0      306 /dev/null
Python  18432 me    5u     IPv4 0xaa      0t0      TCP 1.2.3.4:5->5.6.7.8:443 (CLOSE_WAIT)
Python  18432 me    6u     IPv4 0xbb      0t0      TCP 127.0.0.1:8765 (LISTEN)
Python  18432 me    7      PIPE 0xcc    16384                     ->0xdd
Python  18432 me    8r      REG               1,16     8192 123 /cache/tkr-tz.db
"""


def test_parse_launchd_reads_the_jobs_own_lines_only():
    job = checker.parse_launchd(LAUNCHCTL)
    assert job == {"path": "/Users/me/Library/LaunchAgents/com.me.quoteserver.plist", "state": "running",
                   "runs": "3", "pid": "18432", "last exit code": "70"}


def test_count_descriptors_skips_rows_that_are_not_descriptors():
    total, by_type, close_wait = checker.count_descriptors(LSOF)
    assert total == 5 and by_type == {"CHR": 1, "IPv4": 2, "PIPE": 1, "REG": 1}
    assert close_wait == 1


def test_log_since_last_start_ignores_earlier_runs(tmp_path):
    log = tmp_path / "q.log"
    log.write_text("old ERROR: Too many open files\n"
                   "x quote_server 1.0.0 starting on 127.0.0.1:8765\n"
                   "x INFO: fine\n")
    assert checker.log_since_last_start(log) == ["x quote_server 1.0.0 starting on 127.0.0.1:8765",
                                                 "x INFO: fine"]
    assert checker.log_since_last_start(tmp_path / "missing.log") == []


def test_log_problems_name_each_kind_once_with_a_count():
    lines = ["OSError: [Errno 24] Too many open files: 'x'"] * 2 + ["refresh crashed (1 in a row)", "ok"]
    found = checker.log_problems(lines)
    assert found[0].startswith("ran out of open files (2 ") and found[1].startswith("a refresh crashed (1 ")
    assert checker.log_problems(["all fine"]) == []
