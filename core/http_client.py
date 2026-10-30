"""Shared HTTP client: pacing, retries with backoff, a per-host circuit breaker, and blocks that last.

Every call goes to a named host. For each host the client:
  paces       spaces requests by a minimum delay (with jitter) and caps requests in flight
  retries     repeats a failure that may pass, waiting longer each time (only the failed call)
  breaks      skips the host for a short time after several failed calls in a row
  remembers   after a refusal or rate limit (HTTP 401, 403, 429, or an empty 202/204 reply)
              stops all requests to the host for a cooldown that grows with each repeat, and
              keeps that record in a file so a restart does not forget it

`HttpClient.get` makes a GET request. `HttpClient.call` applies the same rules to any
function, such as a third-party library's own network call.

All settings are keyword arguments to the constructor; nothing is read at import (dev ADR 0023).
Log events: logger `core.http_client`, scope `http_client`, action `http_request`.
Dates in the block file are ISO 8601 UTC.
"""

import json
import logging
import os
import random
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

__all__ = [
    "BLOCK_COOLDOWNS", "BlockStore", "HostBreaker", "HostSettings", "HttpClient",
    "HttpClientError", "Pacer", "classify_exception",
    "NETWORK", "TIMEOUT", "RATE_LIMITED", "REFUSED", "SERVER_ERROR", "HTTP_STATUS",
]

log = logging.getLogger(__name__)

SCOPE = "http_client"
ACTION_HTTP_REQUEST = "http_request"

# Error kinds. A caller may add its own kinds through the `classify` argument of `call`.
NETWORK = "network"            # connection failed, DNS failed
TIMEOUT = "timeout"            # the host did not answer in time
RATE_LIMITED = "rate_limited"  # HTTP 429
REFUSED = "refused"            # HTTP 401 or 403, or an empty 202/204 reply
SERVER_ERROR = "server_error"  # HTTP 5xx
HTTP_STATUS = "http_status"    # any other HTTP error status

DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRY_DELAYS = (1.0, 3.0)       # waits before the 2nd and 3rd attempt (each varies by +/-20%)
DEFAULT_BREAKER_THRESHOLD = 5           # failed calls in a row that open the breaker
DEFAULT_BREAKER_COOLDOWN = 20.0         # seconds the breaker stays open
BLOCK_COOLDOWNS = (300.0, 900.0, 3600.0)  # seconds without requests after the 1st, 2nd, 3rd+ refusal
MAX_RETRY_AFTER = 60.0
MAX_EVENTS = 200


class HttpClientError(Exception):
    """A call failed in a way that concerns the host, not the answer it would have given.

    Attributes:
        kind: One of the kind constants above (or a caller's own kind).
        message: One sentence naming the cause.
        host: The host name the call was made for.
        status: HTTP status, when there was one.
        retryable: True when asking again soon may succeed.
        blocks_host: True when the host refused us; requests to it stop for a cooldown.
        retry_after: Seconds the host asked us to wait, when it said.
        cooldown_until: ISO 8601 UTC time before which no request is sent, when blocked.
    """

    def __init__(self, kind: str, message: str, *, host: str, status: int | None = None,
                 retryable: bool = True, blocks_host: bool = False,
                 retry_after: float | None = None, cooldown_until: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.host = host
        self.status = status
        self.retryable = retryable
        self.blocks_host = blocks_host
        self.retry_after = retry_after
        self.cooldown_until = cooldown_until


def _retry_after(response) -> float | None:
    """Return the Retry-After header as seconds (capped), or None."""
    value = getattr(response, "headers", None) and response.headers.get("Retry-After")
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_exception(exc: BaseException, host: str) -> HttpClientError | None:
    """Describe a requests or curl_cffi error as an HttpClientError.

    Returns:
        The HttpClientError, or None when `exc` is not a network or HTTP error (the host
        answered; the exception is the caller's own).
    """
    if isinstance(exc, HttpClientError):
        return exc
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or None
    retry_after = _retry_after(response)
    if status == 429:
        return HttpClientError(RATE_LIMITED, f"{host} is limiting requests (HTTP 429)", host=host,
                               status=status, retryable=False, blocks_host=True, retry_after=retry_after)
    if status in (401, 403):
        return HttpClientError(REFUSED, f"{host} refused the request (HTTP {status})", host=host,
                               status=status, retryable=False, blocks_host=True)
    if status and status >= 500:
        return HttpClientError(SERVER_ERROR, f"{host} answered HTTP {status}", host=host,
                               status=status, retry_after=retry_after)
    if status:
        return HttpClientError(HTTP_STATUS, f"{host} answered HTTP {status}", host=host,
                               status=status, retryable=False)
    if "Timeout" in {cls.__name__ for cls in type(exc).__mro__}:
        return HttpClientError(TIMEOUT, f"{host} did not answer in time", host=host)
    if isinstance(exc, OSError):
        return HttpClientError(NETWORK, f"cannot reach {host}", host=host)
    return None


@dataclass(frozen=True)
class HostSettings:
    """Pacing for one host.

    Attributes:
        min_delay: Minimum seconds between the starts of two requests. 0 means no spacing.
        max_concurrent: Most requests in flight at once. None means no cap.
    """

    min_delay: float = 0.0
    max_concurrent: int | None = None


class Pacer:
    """Spaces the start of requests to one host and caps how many are in flight."""

    def __init__(self, settings: HostSettings, jitter: float = 0.25, clock=time.monotonic,
                 sleep=time.sleep, uniform=random.uniform) -> None:
        self._min_delay = settings.min_delay
        self._jitter = jitter
        self._clock, self._sleep, self._uniform = clock, sleep, uniform
        self._lock = threading.Lock()
        self._next_start = 0.0
        self._gate = threading.BoundedSemaphore(settings.max_concurrent) if settings.max_concurrent else None

    def _reserve(self) -> float:
        """Claim the next start time; return how long to wait for it."""
        if self._min_delay <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + self._min_delay * self._uniform(1 - self._jitter, 1 + self._jitter)
            return start - now

    @contextmanager
    def slot(self):
        """Wait for this request's turn; hold an in-flight place until the block ends."""
        if self._gate is not None:
            self._gate.acquire()
        try:
            wait = self._reserve()
            if wait > 0:
                self._sleep(wait)
            yield
        finally:
            if self._gate is not None:
                self._gate.release()


class HostBreaker:
    """Opens after `threshold` failed calls in a row; the next call after `cooldown` is a trial."""

    def __init__(self, host: str, threshold: int = DEFAULT_BREAKER_THRESHOLD,
                 cooldown: float = DEFAULT_BREAKER_COOLDOWN, clock=time.monotonic) -> None:
        self._host = host
        self._threshold = threshold
        self._cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self._error: HttpClientError | None = None

    def open_error(self) -> HttpClientError | None:
        """Return the error to raise while the breaker is open, else None."""
        with self._lock:
            return self._error if self._clock() < self._open_until else None

    def record_success(self) -> None:
        """The host answered; close the breaker."""
        with self._lock:
            self._failures, self._open_until, self._error = 0, 0.0, None

    def record_failure(self, error: HttpClientError) -> None:
        """A call failed after its retries; open the breaker when enough have in a row."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                newly_open = self._clock() >= self._open_until
                self._open_until = self._clock() + self._cooldown
                self._error = error
            else:
                return
        if newly_open:
            log.warning("%s: %d calls failed in a row (%s); not asking for %g s", self._host,
                        self._failures, error.kind, self._cooldown,
                        extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST, "outcome": "failure",
                               "actor": "system", "target": self._host})

    @property
    def is_open(self) -> bool:
        """True while calls are being skipped."""
        return self.open_error() is not None


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(iso: str) -> float:
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


class BlockStore:
    """Hosts that refused us, kept in a file so a restart honors the cooldown.

    Each refusal after the cooldown ends makes the next cooldown longer. The first
    successful call after a cooldown restores the host. Every change is a dated event.

    Args:
        path: JSON file for the record; None keeps it in memory only.
        cooldowns: Seconds without requests after the 1st, 2nd, ... refusal (the last repeats).
        clock: Returns the current time in epoch seconds.
    """

    def __init__(self, path: Path | None = None, cooldowns: tuple[float, ...] = BLOCK_COOLDOWNS,
                 clock=time.time) -> None:
        self._path = path
        self._cooldowns = cooldowns
        self._clock = clock
        self._lock = threading.Lock()
        self._hosts: dict[str, dict] = {}
        self._events: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            hosts, events = data["hosts"], data["events"]
            for record in hosts.values():
                _epoch(record["cooldown_until"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            log.warning("block file %s is unreadable; starting with no blocks", self._path,
                        extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST,
                               "outcome": "failure", "actor": "system"})
            return
        self._hosts, self._events = hosts, events

    def _save(self) -> None:
        if self._path is None:
            return
        payload = json.dumps({"hosts": self._hosts, "events": self._events[-MAX_EVENTS:]}, indent=1)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
            os.replace(tmp, self._path)
        except OSError as exc:
            log.warning("block file %s not written: %s", self._path, type(exc).__name__,
                        extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST,
                               "outcome": "failure", "actor": "system"})

    def active(self, host: str) -> dict | None:
        """Return the host's record while its cooldown runs, else None."""
        with self._lock:
            record = self._hosts.get(host)
            if record and self._clock() < _epoch(record["cooldown_until"]):
                return dict(record)
            return None

    def record_block(self, host: str, error: HttpClientError) -> dict:
        """Record a refusal. Returns the host's record.

        Refusals that arrive during a running cooldown (calls already in flight) are
        counted but do not lengthen it.
        """
        with self._lock:
            now = self._clock()
            record = self._hosts.get(host)
            if record and now < _epoch(record["cooldown_until"]):
                record["refused"] += 1
                self._save()
                return dict(record)
            level = min(record["level"] + 1, len(self._cooldowns) - 1) if record else 0
            cooldown = self._cooldowns[level]
            record = {"status": "limited", "kind": error.kind, "message": error.message,
                      "since": record["since"] if record else _iso(now), "level": level,
                      "refused": (record["refused"] if record else 0) + 1,
                      "last_refused_at": _iso(now), "cooldown_until": _iso(now + cooldown)}
            self._hosts[host] = record
            self._events.append({"at": _iso(now), "host": host, "event": "limited",
                                 "kind": error.kind, "cooldown_seconds": cooldown})
            self._save()
        log.warning("%s: %s; no requests for %g s", host, error.message, cooldown,
                    extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST, "outcome": "failure",
                           "actor": "system", "target": host})
        return dict(record)

    def restore_if_due(self, host: str) -> None:
        """Clear the host's record after a successful call once its cooldown has ended."""
        with self._lock:
            record = self._hosts.get(host)
            now = self._clock()
            if not record or now < _epoch(record["cooldown_until"]):
                return
            del self._hosts[host]
            self._events.append({"at": _iso(now), "host": host, "event": "restored",
                                 "limited_since": record["since"], "refused": record["refused"]})
            self._save()
        log.info("%s: requests are being answered again (limited since %s)", host, record["since"],
                 extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST, "outcome": "success",
                        "actor": "system", "target": host})

    def record_for(self, host: str) -> dict | None:
        """Return a copy of the host's record, running or past its cooldown, or None."""
        with self._lock:
            record = self._hosts.get(host)
            return dict(record) if record else None


class _HostState:
    def __init__(self, settings: HostSettings, host: str, breaker_threshold: int,
                 breaker_cooldown: float) -> None:
        self.pacer = Pacer(settings)
        self.breaker = HostBreaker(host, breaker_threshold, breaker_cooldown)
        self.lock = threading.Lock()
        self.requests = 0
        self.failures = 0


class HttpClient:
    """Makes HTTP requests with the rules in the module docstring.

    Args:
        timeout: Seconds before a GET gives up.
        retry_delays: Waits before the 2nd, 3rd, ... attempt of one call. () means no retry.
        host_settings: host name -> HostSettings. A host not listed has no pacing.
        breaker_threshold: Failed calls in a row that open a host's breaker.
        breaker_cooldown: Seconds a breaker stays open.
        block_file: JSON file that keeps refusals across restarts; None keeps them in memory.
        block_cooldowns: Seconds without requests after the 1st, 2nd, ... refusal.
        headers_by_host: host name -> headers sent on every GET to that host.
        listeners: Functions called with one dict per attempt (`http.host`, `http.attempt`,
            `event.outcome`, `error.type`, `event.duration`). With none, nothing is recorded.
        session: A requests.Session to use; by default one is created on first use.
        sleep: Wait function; tests replace it.
        clock: Returns epoch seconds; tests replace it.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT,
                 retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
                 host_settings: dict[str, HostSettings] | None = None,
                 breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
                 breaker_cooldown: float = DEFAULT_BREAKER_COOLDOWN,
                 block_file: Path | None = None,
                 block_cooldowns: tuple[float, ...] = BLOCK_COOLDOWNS,
                 headers_by_host: dict[str, dict[str, str]] | None = None,
                 listeners: tuple = (), session: requests.Session | None = None,
                 sleep=time.sleep, clock=time.time) -> None:
        self._timeout = timeout
        self._delays = tuple(retry_delays)
        self._host_settings = dict(host_settings or {})
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._headers_by_host = dict(headers_by_host or {})
        self._listeners = tuple(listeners)
        self._session = session
        self._owns_session = session is None
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, _HostState] = {}
        self._blocks = BlockStore(block_file, block_cooldowns, clock)
        self._closed = False

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Release the session. The client cannot be used afterward."""
        with self._lock:
            session, self._session, self._closed = self._session, None, True
        if session is not None and self._owns_session:
            session.close()

    def _state(self, host: str) -> _HostState:
        with self._lock:
            state = self._states.get(host)
            if state is None:
                state = self._states[host] = _HostState(
                    self._host_settings.get(host, HostSettings()), host,
                    self._breaker_threshold, self._breaker_cooldown)
            return state

    def _get_session(self) -> requests.Session:
        with self._lock:
            if self._closed:
                raise RuntimeError("HttpClient is closed")
            if self._session is None:
                self._session = requests.Session()
            return self._session

    def get(self, url: str, *, host: str | None = None, params: dict | None = None,
            headers: dict | None = None, timeout: float | None = None) -> requests.Response:
        """GET a URL under the host's rules.

        Args:
            url: The address.
            host: Name the rules and statistics are kept under; default the URL's host.
            params: Query parameters.
            headers: Extra headers, added to the host's own.
            timeout: Seconds before giving up; default the client's.

        Returns:
            The response, whose status is a success.

        Raises:
            HttpClientError: the call failed, was refused, or the host is stopped.
        """
        host = host or urlparse(url).netloc
        merged = {**self._headers_by_host.get(host, {}), **(headers or {})}

        def once() -> requests.Response:
            response = self._get_session().get(url, params=params, headers=merged or None,
                                               timeout=timeout or self._timeout)
            if response.status_code in (202, 204) and not response.content:
                raise HttpClientError(REFUSED, f"{host} sent an empty reply (HTTP {response.status_code}), "
                                      "which it uses to refuse requests", host=host,
                                      status=response.status_code, retryable=False, blocks_host=True)
            response.raise_for_status()
            return response

        return self.call(host, once)

    def call(self, host: str, fn, *, classify=classify_exception):
        """Run `fn()` under the host's rules and return its result.

        Args:
            host: Name the rules and statistics are kept under.
            fn: A function with no arguments that makes the network call.
            classify: `classify(exc, host)` returns an HttpClientError for an error that
                concerns the host, or None for an exception that is the caller's own
                answer (it is raised unchanged and counts as the host having answered).

        Raises:
            HttpClientError: every attempt failed, or the host is blocked or its breaker is open.
        """
        state = self._state(host)
        for attempt in range(len(self._delays) + 1):
            self._raise_if_stopped(host, state)
            started = time.monotonic()
            try:
                with state.pacer.slot():
                    with state.lock:
                        state.requests += 1
                    result = fn()
            except Exception as exc:  # isolation point: sorted into the host's error or the caller's own
                error = classify(exc, host)
                if error is None:
                    self._answered(host, state, attempt, started)
                    raise
                final = not error.retryable or attempt == len(self._delays)
                self._emit(host, attempt, started, "failure", error.kind)
                if not final:
                    wait = max(self._delays[attempt] * random.uniform(0.8, 1.2), error.retry_after or 0.0)
                    log.info("%s: %s; retrying in %.1f s (attempt %d of %d)", host, error.message, wait,
                             attempt + 2, len(self._delays) + 1,
                             extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST, "outcome": "unknown",
                                    "actor": "system", "target": host})
                    self._sleep(wait)
                    continue
                if error.kind == HTTP_STATUS:   # a 4xx such as 404: the host answered, so it is not failing
                    state.breaker.record_success()
                    raise error from exc
                with state.lock:
                    state.failures += 1
                state.breaker.record_failure(error)
                if error.blocks_host:
                    record = self._blocks.record_block(host, error)
                    error.cooldown_until = record["cooldown_until"]
                raise error from exc
            self._answered(host, state, attempt, started)
            return result
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _answered(self, host: str, state: _HostState, attempt: int, started: float) -> None:
        state.breaker.record_success()
        self._blocks.restore_if_due(host)
        self._emit(host, attempt, started, "success", None)

    def _raise_if_stopped(self, host: str, state: _HostState) -> None:
        block = self._blocks.active(host)
        if block is not None:
            raise HttpClientError(block["kind"], f"{host} is stopped until {block['cooldown_until']}: "
                                  f"{block['message']}", host=host, retryable=False, blocks_host=True,
                                  cooldown_until=block["cooldown_until"])
        open_error = state.breaker.open_error()
        if open_error is not None:
            raise HttpClientError(open_error.kind, open_error.message, host=host,
                                  status=open_error.status, retryable=open_error.retryable)

    def _emit(self, host: str, attempt: int, started: float, outcome: str, error_kind: str | None) -> None:
        if not self._listeners:
            return
        event = {"http.host": host, "http.attempt": attempt + 1, "event.outcome": outcome,
                 "error.type": error_kind, "event.duration": time.monotonic() - started}
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # isolation point: a broken listener must not fail the request
                log.warning("listener failed", exc_info=True,
                            extra={"scope": SCOPE, "action": ACTION_HTTP_REQUEST,
                                   "outcome": "failure", "actor": "system"})

    def status(self) -> dict[str, dict]:
        """Return each host's state: `ok`, `limited` (refused, in cooldown), or `unreachable`.

        Each entry also has `requests` and `failed_calls` counts, and for a host with a
        block record: `since`, `cooldown_until`, `refused`, `kind`.
        """
        with self._lock:
            states = dict(self._states)
        report = {}
        for host in sorted(set(states) | set(self._host_settings)):
            state = states.get(host)
            block = self._blocks.record_for(host)
            entry = {"status": "ok", "requests": 0, "failed_calls": 0}
            if state is not None:
                with state.lock:
                    entry.update(requests=state.requests, failed_calls=state.failures)
                if state.breaker.is_open:
                    entry["status"] = "unreachable"
            if block is not None:
                if self._blocks.active(host) is not None:
                    entry["status"] = "limited"
                entry.update({k: block[k] for k in ("since", "cooldown_until", "refused", "kind")})
            report[host] = entry
        return report
