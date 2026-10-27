"""Background thread that keeps every known symbol's cached quote current."""

import logging
import threading
import time

from quote_server import providers
from quote_server.cache import QuoteCache

log = logging.getLogger("quote_server")


class Refresher:
    """Refreshes all symbols every `interval` seconds, and new symbols as soon as they appear."""

    def __init__(self, cache: QuoteCache, interval: float,
                 fetch_many=providers.fetch_many) -> None:
        self._cache = cache
        self._interval = interval
        self._fetch_many = fetch_many
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._urgent: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="refresher", daemon=True)
        self.last_cycle_at: float | None = None

    @property
    def interval(self) -> float:
        """Seconds between full refresh cycles."""
        return self._interval

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
                self.refresh([s for s, _ in self._cache.symbols()])
            finally:
                done.set()

        threading.Thread(target=work, name="refresh-all", daemon=True).start()
        return done

    def refresh(self, symbols: list[str]) -> None:
        """Fetch the given symbols now and store the results."""
        kinds = dict(self._cache.symbols())
        wanted = {s: kinds[s] for s in symbols if s in kinds}
        if not wanted:
            return
        for symbol, result in self._fetch_many(wanted).items():
            if isinstance(result, dict):
                self._cache.set_ok(symbol, result)
            elif isinstance(result, providers.NotFound):
                self._cache.set_not_found(symbol)
            else:
                log.warning("fetch failed symbol=%s error=%s: %s",
                            symbol, type(result).__name__, result)
                self._cache.set_failure(symbol, f"{type(result).__name__}: {result}")
        self._cache.save()

    def _run(self) -> None:
        while not self._stop.is_set():
            woke = self._wake.wait(timeout=self._interval)
            if self._stop.is_set():
                return
            self._wake.clear()
            if woke:
                with self._lock:
                    batch, self._urgent = list(self._urgent), set()
            else:
                batch = [s for s, _ in self._cache.symbols()]
            self.refresh(batch)
            if not woke:
                self.last_cycle_at = time.time()
