"""Tests for install_launch_agent.py: unloading, loading with retries, and the start check."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "quote_server" / "scripts" / "install_launch_agent.py"
spec = importlib.util.spec_from_file_location("install_launch_agent", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

BOOTSTRAP_5 = "Bootstrap failed: 5: Input/output error\nTry re-running the command as root for richer errors."


class FakeLaunchd:
    """Stands in for launchctl. After `bootout`, the job stays loaded for `unload_polls` checks;
    `bootstrap` fails with error 5 while the job is loaded or `extra_failures` remain."""

    def __init__(self, loaded=True, unload_polls=0, extra_failures=0, never_unloads=False):
        self.loaded, self.polls_left, self.extra_failures = loaded, 0, extra_failures
        self.unload_polls, self.never_unloads = unload_polls, never_unloads
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args[0])
        verb = args[0]
        if verb == "bootout":
            if self.loaded:
                self.polls_left = self.unload_polls
                self.loaded_until_polled = True
            return subprocess.CompletedProcess(args, 0 if self.loaded else 3, "", "")
        if verb == "print":
            if self.loaded and self.polls_left > 0 or self.never_unloads:
                self.polls_left -= 1
                return subprocess.CompletedProcess(args, 0, "", "")
            self.loaded = False
            return subprocess.CompletedProcess(args, 113, "", "Could not find service")
        if verb == "bootstrap":
            if self.loaded or self.extra_failures > 0:
                self.extra_failures -= 1
                return subprocess.CompletedProcess(args, 5, "", BOOTSTRAP_5)
            self.loaded = True
            self.polls_left = 0
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected launchctl {args}")


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(installer, "PLIST_PATH", tmp_path / "agent.plist")
    monkeypatch.setattr(installer, "LOG_FILE", tmp_path / "logs" / "q.log")
    monkeypatch.setattr(installer.time, "sleep", clock.sleep)
    monkeypatch.setattr(installer.time, "monotonic", clock.monotonic)
    # the default arguments of unload/load/wait_for_server were bound at import: rebind them
    for name in ("unload", "load", "wait_for_server"):
        func = getattr(installer, name)
        monkeypatch.setattr(func, "__defaults__", (clock.sleep, clock.monotonic))
    return clock


def run_main(monkeypatch, launchd, args=(), health=({"refresh_seconds": 600.0, "symbols": 50},)):
    monkeypatch.setattr(installer, "launchctl", launchd)
    answers = list(health)
    monkeypatch.setattr(installer, "wait_for_server", lambda port: answers[0] if answers[0] else None)
    monkeypatch.setattr(sys, "argv", ["install_launch_agent.py", *args])
    return installer.main()


def test_a_loaded_job_is_unloaded_before_the_new_one_is_loaded(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=True, unload_polls=6)             # launchd needs 6 checks to finish
    assert run_main(monkeypatch, launchd, ["--refresh-seconds", "600"]) == 0
    assert launchd.calls.count("print") == 7                        # polled until gone, then one more is not needed
    assert launchd.calls[0] == "bootout" and launchd.calls[-1] == "bootstrap"
    assert launchd.calls.count("bootstrap") == 1                    # no failed attempt: it waited
    assert "Server answering on port 8765: refresh every 600.0 s, 50 symbols" in capsys.readouterr().out


def test_a_first_install_needs_no_wait(env, monkeypatch):
    launchd = FakeLaunchd(loaded=False)
    assert run_main(monkeypatch, launchd) == 0
    assert launchd.calls == ["bootout", "print", "bootstrap"] and env.sleeps == []


def test_bootstrap_error_5_is_retried_and_the_old_job_is_unloaded_again(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=False, extra_failures=2)
    assert run_main(monkeypatch, launchd) == 0
    assert launchd.calls.count("bootstrap") == 3 and launchd.calls.count("bootout") == 3
    assert env.sleeps == [2.0, 4.0]                                 # the wait grows with each attempt
    out = capsys.readouterr().out
    assert "Bootstrap failed: 5: Input/output error); attempt 1 of 4, retrying in 2 s" in out


def test_giving_up_says_the_server_is_not_running_and_how_to_start_it(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=False, extra_failures=99)
    assert run_main(monkeypatch, launchd, ["--port", "8800"]) == 1
    captured = capsys.readouterr()
    assert launchd.calls.count("bootstrap") == installer.LOAD_ATTEMPTS
    assert "after 4 attempts" in captured.err and "The server is not running" in captured.err
    assert f"launchctl bootstrap gui/{installer.os.getuid()} {installer.PLIST_PATH}" in captured.err
    assert "Loaded" not in captured.out


def test_a_job_that_never_unloads_is_reported_and_loading_is_still_tried(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=True, never_unloads=True)
    assert run_main(monkeypatch, launchd) == 1
    assert f"still loaded after {installer.UNLOAD_TIMEOUT_SECONDS:g} s" in capsys.readouterr().out
    assert env.now >= installer.UNLOAD_TIMEOUT_SECONDS


def test_a_server_that_does_not_answer_is_a_failure(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=False)
    assert run_main(monkeypatch, launchd, ["--port", "8800"], health=(None,)) == 1
    captured = capsys.readouterr()
    assert "Loaded" in captured.out
    assert "did not answer on port 8800" in captured.err and str(installer.LOG_FILE) in captured.err


def test_uninstall_waits_for_the_unload_before_deleting_the_plist(env, monkeypatch):
    launchd = FakeLaunchd(loaded=True, unload_polls=4)
    installer.PLIST_PATH.write_text("x")
    assert run_main(monkeypatch, launchd, ["--uninstall"]) == 0
    assert not installer.PLIST_PATH.exists() and launchd.calls.count("print") == 5


def test_uninstall_reports_a_job_that_will_not_unload(env, monkeypatch, capsys):
    launchd = FakeLaunchd(loaded=True, never_unloads=True)
    installer.PLIST_PATH.write_text("x")
    assert run_main(monkeypatch, launchd, ["--uninstall"]) == 1
    assert installer.PLIST_PATH.exists() and "still loaded" in capsys.readouterr().err


def test_dry_run_runs_no_launchctl(env, monkeypatch, capsys):
    launchd = FakeLaunchd()
    assert run_main(monkeypatch, launchd, ["--dry-run", "--refresh-seconds", "60"]) == 0
    assert launchd.calls == [] and not installer.PLIST_PATH.exists()
    assert "--restart-on-hang" in capsys.readouterr().out


def test_wait_for_server_polls_until_it_answers(env, monkeypatch):
    replies = [OSError("refused"), OSError("refused"), None]

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"refresh_seconds": 5.0, "symbols": 1}'

    def urlopen(url, timeout):
        item = replies.pop(0)
        if isinstance(item, OSError):
            raise item
        return Reply()

    monkeypatch.setattr(installer.urllib.request, "urlopen", urlopen)
    assert installer.wait_for_server(8765) == {"refresh_seconds": 5.0, "symbols": 1}
    assert env.sleeps == [installer.START_POLL_SECONDS] * 2


def test_wait_for_server_gives_up_after_the_timeout(env, monkeypatch):
    def urlopen(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(installer.urllib.request, "urlopen", urlopen)
    assert installer.wait_for_server(8765) is None
    assert env.now >= installer.START_TIMEOUT_SECONDS
