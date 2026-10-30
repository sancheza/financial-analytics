"""Tests for core.http_client: classification, pacing, retries, the breaker, and persistent blocks."""

import json
import subprocess
import sys
import threading
import time

import pytest
import requests

from core import http_client
from core.http_client import (BlockStore, HostBreaker, HostSettings, HttpClient, HttpClientError, Pacer,
                              classify_exception)


class FakeResponse:
    def __init__(self, status_code=200, content=b"ok", headers=None):
        self.status_code, self.content, self.headers = status_code, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Answers each get() from a script: a response, or an exception to raise."""

    def __init__(self, *script):
        self.script, self.calls, self.closed = list(script), [], False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make_client(*script, sleeps=None, clock=None, **kw):
    sleeps = [] if sleeps is None else sleeps
    session = FakeSession(*script)
    kw.setdefault("retry_delays", (1.0, 3.0))
    client = HttpClient(session=session, sleep=sleeps.append, clock=clock or Clock(), **kw)
    return client, session, sleeps


def status_error(status, headers=None):
    return requests.HTTPError(f"HTTP {status}", response=FakeResponse(status, headers=headers))


# --- classification --------------------------------------------------------------------------

@pytest.mark.parametrize("error,kind,retryable,blocks", [
    (status_error(429), "rate_limited", False, True),
    (status_error(403), "refused", False, True),
    (status_error(401), "refused", False, True),
    (status_error(503), "server_error", True, False),
    (status_error(404), "http_status", False, False),
    (requests.ReadTimeout("x"), "timeout", True, False),
    (requests.ConnectionError("x"), "network", True, False),
])
def test_classify_exception(error, kind, retryable, blocks):
    result = classify_exception(error, "Webull")
    assert (result.kind, result.retryable, result.blocks_host) == (kind, retryable, blocks)
    assert "Webull" in result.message


def test_classify_leaves_the_callers_own_exceptions_alone():
    assert classify_exception(ValueError("x"), "Webull") is None
    assert classify_exception(KeyError("x"), "Webull") is None


def test_retry_after_is_read_and_capped():
    assert classify_exception(status_error(503, {"Retry-After": "7"}), "h").retry_after == 7.0
    assert classify_exception(status_error(503, {"Retry-After": "9999"}), "h").retry_after == 60.0
    assert classify_exception(status_error(503, {"Retry-After": "soon"}), "h").retry_after is None


# --- Pacer -----------------------------------------------------------------------------------

def test_pacer_spaces_request_starts():
    clock, sleeps = [0.0], []
    pacer = Pacer(HostSettings(min_delay=0.5), jitter=0.0, clock=lambda: clock[0], sleep=sleeps.append)
    for _ in range(3):
        with pacer.slot():
            pass
    assert sleeps == [0.5, 1.0]            # the first goes at once; the clock never moved


def test_pacer_does_not_wait_after_an_idle_period():
    clock, sleeps = [0.0], []
    pacer = Pacer(HostSettings(min_delay=0.5), jitter=0.0, clock=lambda: clock[0], sleep=sleeps.append)
    with pacer.slot():
        pass
    clock[0] = 10.0
    with pacer.slot():
        pass
    assert sleeps == []


def test_pacer_jitter_stays_inside_its_range():
    values = iter([0.75, 1.25, 1.0])
    pacer = Pacer(HostSettings(min_delay=1.0), jitter=0.25, clock=lambda: 0.0, sleep=lambda s: None,
                  uniform=lambda low, high: next(values))
    waits = [pacer._reserve() for _ in range(3)]
    assert waits == [0.0, 0.75, 2.0]      # each gap is min_delay times that request's jitter


def test_pacer_caps_requests_in_flight():
    pacer = Pacer(HostSettings(max_concurrent=2))
    inside, peak, lock = 0, 0, threading.Lock()

    def work():
        nonlocal inside, peak
        with pacer.slot():
            with lock:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.03)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak == 2


# --- HostBreaker -----------------------------------------------------------------------------

def test_breaker_opens_after_threshold_and_allows_a_trial_after_cooldown():
    now = [0.0]
    breaker = HostBreaker("h", threshold=3, cooldown=20.0, clock=lambda: now[0])
    error = HttpClientError("network", "cannot reach h", host="h")
    for _ in range(2):
        breaker.record_failure(error)
    assert breaker.open_error() is None
    breaker.record_failure(error)
    assert breaker.open_error() is error
    now[0] = 21.0
    assert breaker.open_error() is None            # trial allowed
    breaker.record_failure(error)                  # the trial failed: open again at once
    assert breaker.is_open
    now[0] = 50.0
    breaker.record_success()
    breaker.record_failure(error)
    assert not breaker.is_open                     # the count restarted


# --- BlockStore ------------------------------------------------------------------------------

def refused(host="h"):
    return HttpClientError("rate_limited", "h is limiting requests (HTTP 429)", host=host,
                           retryable=False, blocks_host=True)


def test_block_cooldown_grows_with_each_refusal_after_the_cooldown():
    clock = Clock()
    store = BlockStore(None, cooldowns=(300.0, 900.0, 3600.0), clock=clock)
    first = store.record_block("h", refused())
    assert first["cooldown_until"] == "2027-01-15T08:05:00Z" and first["level"] == 0
    clock.now += 301
    assert store.active("h") is None
    second = store.record_block("h", refused())
    assert second["level"] == 1 and second["since"] == first["since"] and second["refused"] == 2
    clock.now += 901
    third = store.record_block("h", refused())
    clock.now += 3601
    fourth = store.record_block("h", refused())
    assert (third["level"], fourth["level"]) == (2, 2)   # the last cooldown repeats


def test_refusals_during_a_cooldown_are_counted_but_do_not_lengthen_it():
    store = BlockStore(None, clock=Clock())
    first = store.record_block("h", refused())
    again = store.record_block("h", refused())
    assert again["cooldown_until"] == first["cooldown_until"] and again["refused"] == 2


def test_block_survives_a_restart_and_restores_after_success(tmp_path):
    path, clock = tmp_path / "blocks.json", Clock()
    BlockStore(path, clock=clock).record_block("h", refused())
    reopened = BlockStore(path, clock=clock)
    assert reopened.active("h")["kind"] == "rate_limited"
    reopened.restore_if_due("h")
    assert reopened.active("h") is not None        # cooldown still running: not restored
    clock.now += 301
    reopened.restore_if_due("h")
    assert reopened.record_for("h") is None
    events = json.loads(path.read_text())["events"]
    assert [e["event"] for e in events] == ["limited", "restored"]
    assert all(e["at"].endswith("Z") and e["at"][4] == "-" for e in events)   # ISO 8601 UTC


@pytest.mark.parametrize("text", ["{not json", "[]", '{"hosts": {"h": {}}, "events": []}', "null"])
def test_unreadable_block_file_starts_empty(tmp_path, text):
    path = tmp_path / "blocks.json"
    path.write_text(text)
    assert BlockStore(path).active("h") is None


# --- HttpClient.get --------------------------------------------------------------------------

def test_get_returns_the_response_and_sends_host_headers():
    client, session, _ = make_client(FakeResponse(), headers_by_host={"Webull": {"User-Agent": "UA"}})
    response = client.get("https://example.test/x", host="Webull", params={"a": 1}, headers={"X": "y"})
    assert response.status_code == 200
    kwargs = session.calls[0][1]
    assert kwargs["headers"] == {"User-Agent": "UA", "X": "y"} and kwargs["params"] == {"a": 1}


def test_host_defaults_to_the_url_host():
    client, _, _ = make_client(FakeResponse())
    client.get("https://example.test/x")
    assert client.status()["example.test"]["requests"] == 1


def test_transient_failures_are_retried_with_growing_waits():
    client, session, sleeps = make_client(requests.ConnectionError("x"), requests.ConnectionError("x"),
                                          FakeResponse())
    assert client.get("https://h.test/", host="h").status_code == 200
    assert len(session.calls) == 3
    assert 0.8 <= sleeps[0] <= 1.2 and 2.4 <= sleeps[1] <= 3.6


def test_retry_after_lengthens_the_wait():
    client, _, sleeps = make_client(status_error(503, {"Retry-After": "10"}), FakeResponse())
    client.get("https://h.test/", host="h")
    assert sleeps == [10.0]


def test_all_attempts_failing_raises_with_the_cause():
    client, session, _ = make_client(requests.ConnectionError("x"))
    with pytest.raises(HttpClientError) as caught:
        client.get("https://h.test/", host="h")
    assert caught.value.kind == "network" and len(session.calls) == 3
    assert isinstance(caught.value.__cause__, requests.ConnectionError)


def test_a_404_is_an_answer_not_a_failure():
    client, session, _ = make_client(status_error(404), breaker_threshold=1)
    for _ in range(3):
        with pytest.raises(HttpClientError) as caught:
            client.get("https://h.test/", host="h")
        assert caught.value.status == 404 and caught.value.kind == "http_status"
    assert len(session.calls) == 3                 # not retried, and the breaker (threshold 1) never opened
    assert client.status()["h"]["status"] == "ok" and client.status()["h"]["failed_calls"] == 0


def test_a_refusal_stops_the_host_without_more_requests(tmp_path):
    client, session, _ = make_client(status_error(429), block_file=tmp_path / "b.json")
    with pytest.raises(HttpClientError) as first:
        client.get("https://h.test/", host="h")
    assert first.value.kind == "rate_limited" and first.value.cooldown_until
    with pytest.raises(HttpClientError) as second:
        client.get("https://h.test/", host="h")
    assert second.value.kind == "rate_limited" and "stopped until" in second.value.message
    assert len(session.calls) == 1                 # not retried, and not asked again
    assert client.status()["h"]["status"] == "limited" and client.status()["h"]["refused"] == 1


def test_a_block_from_an_earlier_run_is_honored(tmp_path):
    path, clock = tmp_path / "b.json", Clock()
    first, _, _ = make_client(status_error(403), block_file=path, clock=clock)
    with pytest.raises(HttpClientError):
        first.get("https://h.test/", host="h")
    second, session, _ = make_client(FakeResponse(), block_file=path, clock=clock)
    with pytest.raises(HttpClientError):
        second.get("https://h.test/", host="h")
    assert session.calls == []


def test_first_success_after_the_cooldown_restores_the_host(tmp_path):
    clock = Clock()
    client, _, _ = make_client(status_error(429), FakeResponse(), block_file=tmp_path / "b.json", clock=clock)
    with pytest.raises(HttpClientError):
        client.get("https://h.test/", host="h")
    clock.now += 301
    assert client.get("https://h.test/", host="h").status_code == 200
    assert client.status()["h"]["status"] == "ok"


def test_empty_202_is_treated_as_a_refusal():
    client, _, _ = make_client(FakeResponse(202, b""))
    with pytest.raises(HttpClientError) as caught:
        client.get("https://h.test/", host="h")
    assert caught.value.kind == "refused" and caught.value.blocks_host


def test_breaker_skips_a_host_that_keeps_failing_then_recovers():
    now = [0.0]
    client, session, _ = make_client(requests.ConnectionError("x"), retry_delays=(), breaker_threshold=3,
                                     breaker_cooldown=20.0)
    client._state("h").breaker._clock = lambda: now[0]
    for _ in range(3):
        with pytest.raises(HttpClientError):
            client.get("https://h.test/", host="h")
    assert len(session.calls) == 3
    with pytest.raises(HttpClientError) as skipped:
        client.get("https://h.test/", host="h")
    assert skipped.value.kind == "network" and len(session.calls) == 3      # no request sent
    assert client.status()["h"]["status"] == "unreachable"
    now[0] = 25.0
    session.script = [FakeResponse()]
    assert client.get("https://h.test/", host="h").status_code == 200
    assert client.status()["h"]["status"] == "ok"


def test_hosts_are_independent():
    client, session, _ = make_client(status_error(429))
    with pytest.raises(HttpClientError):
        client.get("https://a.test/", host="a")
    session.script = [FakeResponse()]
    assert client.get("https://b.test/", host="b").status_code == 200


# --- HttpClient.call -------------------------------------------------------------------------

def test_call_passes_the_callers_own_exception_through_and_counts_it_as_an_answer():
    client, _, _ = make_client(retry_delays=(), breaker_threshold=1)
    client._state("h").breaker.record_failure(HttpClientError("network", "x", host="h"))
    client._state("h").breaker._open_until = 0.0

    def answer():
        raise KeyError("no such symbol")

    with pytest.raises(KeyError):
        client.call("h", answer)
    assert client.status()["h"]["status"] == "ok"


def test_call_uses_the_callers_classifier():
    class Library429(Exception):
        pass

    def classify(exc, host):
        if isinstance(exc, Library429):
            return HttpClientError("rate_limited", "limited", host=host, retryable=False, blocks_host=True)
        return classify_exception(exc, host)

    client, _, _ = make_client()
    calls = []

    def fn():
        calls.append(1)
        raise Library429()

    with pytest.raises(HttpClientError) as caught:
        client.call("h", fn, classify=classify)
    assert caught.value.blocks_host and calls == [1]
    with pytest.raises(HttpClientError):
        client.call("h", fn, classify=classify)
    assert calls == [1]                            # stopped


def test_listeners_see_each_attempt_and_a_broken_one_is_harmless():
    events = []

    def broken(event):
        raise RuntimeError("listener bug")

    client, _, _ = make_client(requests.ConnectionError("x"), FakeResponse(), listeners=(broken, events.append))
    client.get("https://h.test/", host="h")
    assert [(e["http.attempt"], e["event.outcome"], e["error.type"]) for e in events] == [
        (1, "failure", "network"), (2, "success", None)]
    assert events[0]["http.host"] == "h" and events[0]["event.duration"] >= 0


# --- lifecycle and import (dev ADR 0023 sections 2.2 and 2.10) -------------------------------

def test_close_releases_the_session_it_made_and_works_in_a_with_block():
    with HttpClient() as client:
        session = client._get_session()
    assert session.adapters is not None and client._closed
    with pytest.raises(RuntimeError):
        client._get_session()


def test_close_leaves_a_session_it_was_given():
    session = FakeSession(FakeResponse())
    with HttpClient(session=session):
        pass
    assert session.closed is False


def test_import_creates_nothing_and_adds_only_a_null_handler(tmp_path):
    code = ("import logging, os; os.chdir(%r); import core.http_client; "
            "print(sorted(os.listdir('.')), [type(h).__name__ for h in logging.getLogger('core').handlers], "
            "logging.getLogger().handlers)" % str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(http_client.__file__.rsplit("/core/", 1)[0])).stdout.strip()
    assert out == "[] ['NullHandler'] []"


def test_works_with_no_listener_and_no_logging_setup():
    client, _, _ = make_client(FakeResponse())
    assert client.get("https://h.test/", host="h").status_code == 200
