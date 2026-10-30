"""Tests for the quote_server watchdog: when it decides the refresh is broken, and what it does."""

import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from quote_server import watchdog as watchdog_module
from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher
from quote_server.watchdog import CONFIRM_SECONDS, GRACE_SECONDS, Watchdog


class FakeRefresher:
    def __init__(self, alive=True, last_cycle_at=None):
        self.alive = alive
        self.last_cycle_at = last_cycle_at


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(refresher, restart=True, interval=300.0):
    clock, events = Clock(), []
    dog = Watchdog(refresher, interval, restart=restart, save=lambda: events.append("save"),
                   notify=lambda title, message: events.append(("notify", message)),
                   exit_process=lambda code: events.append(("exit", code)), clock=clock)
    return dog, clock, events


def test_healthy_refresh_has_no_problem():
    dog, clock, _ = make(FakeRefresher(last_cycle_at=1_000_000.0 - 100))
    assert dog.problem(clock()) is None


def test_stopped_thread_is_a_problem():
    dog, clock, _ = make(FakeRefresher(alive=False))
    assert dog.problem(clock()) == "the refresh thread has stopped"


def test_overdue_refresh_is_a_problem_only_after_three_intervals_and_grace():
    dog, clock, _ = make(FakeRefresher(last_cycle_at=None))
    clock.now += 3 * 300 + GRACE_SECONDS - 1
    assert dog.problem(clock()) is None
    clock.now += 2
    assert "no full refresh for 16 min" in dog.problem(clock())


def test_a_problem_must_last_before_the_server_restarts():
    dog, clock, events = make(FakeRefresher(alive=False))
    dog.check()                                   # first sighting: logged only
    clock.now += CONFIRM_SECONDS - 1
    dog.check()
    assert events == []
    clock.now += 2
    dog.check()
    assert events == [("notify", "Restarting: the refresh thread has stopped."), "save",
                      ("exit", watchdog_module.EXIT_CODE)]


def test_a_recovery_resets_the_wait():
    refresher = FakeRefresher(alive=False)
    dog, clock, events = make(refresher)
    dog.check()
    clock.now += CONFIRM_SECONDS - 10
    refresher.alive = True
    dog.check()                                   # healthy again (a Mac that just woke)
    refresher.alive = False
    clock.now += 100
    dog.check()                                   # a new episode starts here
    clock.now += CONFIRM_SECONDS - 1
    dog.check()
    assert events == []


def test_without_restart_it_notifies_once_and_keeps_running():
    dog, clock, events = make(FakeRefresher(alive=False), restart=False)
    dog.check()
    clock.now += CONFIRM_SECONDS + 1
    dog.check()
    dog.check()
    assert events == [("notify", "Not refreshing: the refresh thread has stopped. Restart it.")]


def test_exit_does_not_wait_forever_for_a_stuck_save(monkeypatch):
    monkeypatch.setattr(watchdog_module, "SAVE_TIMEOUT_SECONDS", 0.05)
    release, events = threading.Event(), []
    dog = Watchdog(FakeRefresher(alive=False), 300.0, restart=True, save=release.wait,
                   notify=lambda *a: None, exit_process=events.append, clock=Clock())
    dog._restart_server("test")
    release.set()
    assert events == [watchdog_module.EXIT_CODE]


def test_notification_passes_text_as_arguments_not_script(monkeypatch):
    seen = []
    monkeypatch.setattr(watchdog_module.sys, "platform", "darwin")
    monkeypatch.setattr(watchdog_module.subprocess, "run",
                        lambda command, **kw: seen.append(command) or subprocess.CompletedProcess(command, 0))
    assert watchdog_module.notify_macos("Quote server", 'a "quoted" message') is True
    assert seen[0][-2:] == ['a "quoted" message', "Quote server"]
    assert not any('quoted' in part for part in seen[0][:-2])


def test_notification_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(watchdog_module.sys, "platform", "darwin")

    def missing(command, **kw):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(watchdog_module.subprocess, "run", missing)
    assert watchdog_module.notify_macos("t", "m") is False


def test_notification_is_skipped_off_macos(monkeypatch):
    monkeypatch.setattr(watchdog_module.sys, "platform", "linux")
    assert watchdog_module.notify_macos("t", "m") is False


def test_thread_runs_checks(monkeypatch):
    monkeypatch.setattr(watchdog_module, "CHECK_SECONDS", 0.01)
    monkeypatch.setattr(watchdog_module, "CONFIRM_SECONDS", 0.05)
    exited = threading.Event()
    dog = Watchdog(FakeRefresher(alive=False), 300.0, restart=True, notify=lambda *a: None,
                   exit_process=lambda code: exited.set())
    dog.start()
    try:
        assert exited.wait(3.0)
    finally:
        dog.stop()


def test_health_reports_watchdog_settings(tmp_path):
    cache = QuoteCache(tmp_path / "c.json")
    refresher = Refresher(cache, interval=300, fetch_many=lambda kinds: {})
    dog = Watchdog(refresher, 300.0, restart=True)
    with TestClient(create_app(cache, refresher, dog)) as client:
        body = client.get("/health").json()
    assert body["restart_on_hang"] is True and body["started_at"] == pytest.approx(time.time(), abs=60)
