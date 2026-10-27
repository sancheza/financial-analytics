"""Thread-safe quote cache with disk persistence."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

OK = "OK"
PENDING = "PENDING"
NOT_FOUND = "NOT_FOUND"
UPSTREAM = "UPSTREAM"


class QuoteCache:
    """Holds one entry per symbol. Every method takes the lock for its whole body."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._path = path

    def register(self, symbol: str, kind: str) -> bool:
        """Add a symbol as PENDING if unknown.

        Returns:
            True when the symbol was new.
        """
        with self._lock:
            if symbol in self._entries:
                return False
            self._entries[symbol] = {
                "kind": kind, "status": PENDING, "fields": {},
                "fetched_at": None, "failures": 0, "error": None,
            }
            return True

    def get(self, symbol: str) -> dict | None:
        """Return a copy of the entry, or None when unknown."""
        with self._lock:
            entry = self._entries.get(symbol)
            return None if entry is None else {**entry, "fields": dict(entry["fields"])}

    def symbols(self) -> list[tuple[str, str]]:
        """Return (symbol, kind) for every entry."""
        with self._lock:
            return [(s, e["kind"]) for s, e in self._entries.items()]

    def set_ok(self, symbol: str, fields: dict) -> None:
        """Store a successful fetch."""
        with self._lock:
            entry = self._entries[symbol]
            entry.update(status=OK, fields=dict(fields), fetched_at=time.time(),
                         failures=0, error=None)

    def set_not_found(self, symbol: str) -> None:
        """Record that the provider has no data for the symbol."""
        with self._lock:
            entry = self._entries[symbol]
            entry.update(status=NOT_FOUND, fields={}, fetched_at=None,
                         error="provider has no listing")

    def set_failure(self, symbol: str, error: str) -> None:
        """Record a fetch error. Any earlier value is kept and served as stale."""
        with self._lock:
            entry = self._entries[symbol]
            entry["failures"] += 1
            entry["error"] = error
            if not entry["fields"]:
                entry["status"] = UPSTREAM

    def save(self) -> None:
        """Write the cache to disk atomically. No-op without a path."""
        if self._path is None:
            return
        with self._lock:
            payload = json.dumps(self._entries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, self._path)

    def load(self) -> int:
        """Load entries from disk. Returns the number loaded; 0 if missing or unreadable."""
        if self._path is None or not self._path.exists():
            return 0
        try:
            loaded = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return 0
        with self._lock:
            self._entries.update(loaded)
            return len(loaded)
