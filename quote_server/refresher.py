"""Background thread that keeps every known symbol's cached quote current.

Log events: logger `quote_server.refresher`, scope `quote_server`, action `price_refresh`
(one summary per refresh, one line when a symbol starts or stops failing).
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from quote_server import providers
from quote_server.cache import QuoteCache

log = logging.getLogger("quote_server.refresher")

SCOPE = "quote_server"
ACTION_PRICE_REFRESH = "price_refresh"

# Longest single wait. The schedule is kept in wall-clock time and checked this often,
# so a refresh that came due while the computer slept runs soon after it wakes.
POLL_SECONDS = 15.0

# Seconds before the 1st, 2nd, 3rd, ... re-fetch of symbols that failed with a cause
# that may pass (no network, timeout). The last value repeats until the full refresh.
RETRY_DELAYS_SECONDS = (15.0, 30.0, 60.0, 120.0)

# Symbols fetched before the queue of newly requested symbols is checked.
CHUNK_SIZE = 10

# A new failure shared by more symbols than this is logged once, in the summary line.
MAX_SYMBOL_LOG_LINES = 3


@dataclass
class _Tally:
    """What one refresh did, across its chunks."""

    total: int = 0
    counts: dict = field(default_factory=lambda: {"ok": 0, "not_found": 0, "failed": 0})
    failure_counts: dict = field(default_factory=dict)
    new_failures: dict = field(default_factory=dict)
    retry: list = field(default_factory=list)


class Refresher:
    """Refreshes all symbols every `interval` seconds, and new symbols as soon as they appear.

    A symbol whose fetch failed for a passing cause is fetched again after a short wait
    instead of waiting for the next full refresh.
    """

    def __init__(self, cache: QuoteCache, interval: float,
                 fetch_many=providers.fetch_many) -> None:
        self._cache = cache
        self._interval = interval
        self._fetch_many = fetch_many
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._urgent: set[str] = set()
        self._lock = threading.Lock()
        self._retry_symbols: set[str] = set()
        self._retry_at: float | None = None
        self._retry_step = 0
        self._thread = threading.Thread(target=self._run, name="refresher", daemon=True)
        self.last_cycle_at: float | None = None

    @property
    def interval(self) -> float:
        """Seconds between full refresh cycles."""
        return self._interval

    @property
    def alive(self) -> bool:
        """True while the refresh thread is running."""
        return self._thread.is_alive()

    def start(self) -> None:
        """Start the thread. The first cycle refreshes every cached symbol immediately."""
        with self._lock:
            self._urgent.update(s for s, _ in self._cache.symbols())
        self._wake.set()
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for it to exit."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=15)

    def request(self, symbols: list[str]) -> None:
        """Queue symbols for an immediate refresh."""
        with self._lock:
            self._urgent.update(symbols)
        self._wake.set()

    def refresh_all_async(self) -> threading.Event:
        """Refresh every known symbol on a new thread.

        Returns:
            An Event that is set when the refresh has finished.
        """
        done = threading.Event()

        def work() -> None:
            try:
                symbols = [s for s, _ in self._cache.symbols()]
                self._schedule_retry(symbols, self._refresh_guarded(symbols))
            finally:
                done.set()

        threading.Thread(target=work, name="refresh-all", daemon=True).start()
        return done

    def refresh(self, symbols: list[str]) -> list[str]:
        """Fetch the given symbols now and store the results.

        The symbols are fetched CHUNK_SIZE at a time. Between chunks, symbols that were
        requested in the meantime (a new symbol typed in a sheet) are fetched first, so
        a paced refresh of many symbols does not make a new one wait for all of them.

        Returns:
            The symbols that failed with a cause worth retrying soon.
        """
        kinds = dict(self._cache.symbols())
        pending = [s for s in symbols if s in kinds]
        if not pending:
            return []
        started = time.monotonic()
        tally = _Tally()
        while pending:
            chunk, pending = pending[:CHUNK_SIZE], pending[CHUNK_SIZE:]
            self._fetch_and_store(chunk, kinds, tally)
            urgent = sorted(self._take_urgent() - set(pending))
            if urgent:
                kinds = dict(self._cache.symbols())
                self._fetch_and_store([s for s in urgent if s in kinds], kinds, tally)
        self._cache.save()
        for code, group in tally.new_failures.items():
            if len(group) <= MAX_SYMBOL_LOG_LINES:   # a wider failure is named once, in the summary
                for symbol, failure in group:
                    self._log_symbol(symbol, "fetch failed: %s (%s)", failure.message, code,
                                     level=logging.WARNING, outcome="failure")
        causes = [f"{code} x{n}: {message}" for (code, message), n in sorted(tally.failure_counts.items())]
        self._log_summary(tally.total, tally.counts, causes, time.monotonic() - started)
        return tally.retry

    def _fetch_and_store(self, symbols: list[str], kinds: dict[str, str], tally: "_Tally") -> None:
        wanted = {s: kinds[s] for s in symbols}
        if not wanted:
            return
        tally.total += len(wanted)
        extras = {s: f for s, f in self._cache.wanted().items() if s in wanted}
        results = self._fetch_many(wanted, extras=extras) if extras else self._fetch_many(wanted)
        for symbol, result in results.items():
            before = self._cache.get(symbol)
            if isinstance(result, dict):
                errors = getattr(result, "errors", {})
                self._cache.set_ok(symbol, result, {f: {"code": e.code, "message": e.message, "retryable": e.retryable}
                                                    for f, e in errors.items()})
                tally.counts["ok"] += 1
                if any(e.retryable for e in errors.values()):
                    tally.retry.append(symbol)
                for field_name, failure in errors.items():
                    known = (before or {}).get("field_errors", {}).get(field_name)
                    if known and known["code"] == failure.code:
                        continue   # logged when it first appeared
                    self._log_symbol(symbol, "%s not obtained: %s (%s)", field_name, failure.message,
                                     failure.code, level=logging.WARNING, outcome="failure")
                if before and before["error_code"] not in (None, "NOT_FOUND"):
                    self._log_symbol(symbol, "recovered after %s", before["error_code"],
                                     level=logging.INFO, outcome="success")
            elif isinstance(result, providers.NotFound):
                self._cache.set_not_found(symbol, result.message)
                tally.counts["not_found"] += 1
            else:
                failure = providers.as_failure(result, providers.SOURCE_BY_KIND[wanted[symbol]])
                self._cache.set_failure(symbol, failure.code, failure.message)
                tally.counts["failed"] += 1
                cause = (failure.code, failure.message)
                tally.failure_counts[cause] = tally.failure_counts.get(cause, 0) + 1
                if failure.retryable:
                    tally.retry.append(symbol)
                if before is None or before["error_code"] != failure.code:
                    tally.new_failures.setdefault(failure.code, []).append((symbol, failure))

    def _log_symbol(self, symbol: str, message: str, *args, level: int, outcome: str) -> None:
        log.log(level, "%s " + message, symbol, *args,
                extra={"scope": SCOPE, "action": ACTION_PRICE_REFRESH,
                       "outcome": outcome, "actor": "system", "target": symbol})

    def _log_summary(self, total: int, counts: dict[str, int], causes: list[str],
                     seconds: float) -> None:
        failed = counts["failed"]
        log.log(logging.WARNING if failed else logging.INFO,
                "refreshed %d symbols in %.1f s: %d ok, %d not found, %d failed%s",
                total, seconds, counts["ok"], counts["not_found"], failed,
                f" ({'; '.join(causes)})" if causes else "",
                extra={"scope": SCOPE, "action": ACTION_PRICE_REFRESH,
                       "outcome": "failure" if failed else "success", "actor": "system"})

    def _refresh_guarded(self, symbols: list[str]) -> list[str]:
        """Run refresh(); an unexpected error is logged and the symbols are retried."""
        try:
            return self.refresh(symbols)
        except Exception:  # isolation point: one failed refresh must not end the thread
            log.error("refresh crashed; will retry", exc_info=True,
                      extra={"scope": SCOPE, "action": ACTION_PRICE_REFRESH,
                             "outcome": "failure", "actor": "system"})
            return list(symbols)

    def _schedule_retry(self, batch: list[str], failed: list[str]) -> None:
        """Update the symbols to re-fetch soon after a refresh of `batch`.

        Symbols in `batch` leave the plan unless they are in `failed`. A plan with
        symbols and no time set gets the next wait from RETRY_DELAYS_SECONDS; a plan
        that empties resets the waits.
        """
        with self._lock:
            self._retry_symbols = (self._retry_symbols - set(batch)) | set(failed)
            if not self._retry_symbols:
                self._retry_at, self._retry_step = None, 0
            elif self._retry_at is None:
                delay = RETRY_DELAYS_SECONDS[min(self._retry_step, len(RETRY_DELAYS_SECONDS) - 1)]
                self._retry_step += 1
                self._retry_at = time.time() + delay

    def _take_urgent(self) -> set[str]:
        with self._lock:
            batch, self._urgent = self._urgent, set()
        return batch

    def _next_wait(self, next_full: float) -> float:
        with self._lock:
            retry_at = self._retry_at
        due = next_full if retry_at is None else min(next_full, retry_at)
        return max(0.0, min(due - time.time(), POLL_SECONDS))

    def _run(self) -> None:
        next_full = time.time() + self._interval
        while not self._stop.is_set():
            self._wake.wait(timeout=self._next_wait(next_full))
            if self._stop.is_set():
                return
            self._wake.clear()
            batch = self._take_urgent()
            now = time.time()
            full = now >= next_full
            with self._lock:
                retry_due = self._retry_at is not None and now >= self._retry_at
                if full:
                    batch = {s for s, _ in self._cache.symbols()}
                elif retry_due:
                    batch |= self._retry_symbols
                if full or retry_due:
                    self._retry_at = None
            if not batch:
                continue
            ordered = sorted(batch)
            self._schedule_retry(ordered, self._refresh_guarded(ordered))
            if full:
                self.last_cycle_at = time.time()
                next_full = self.last_cycle_at + self._interval
