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
                "fetched_at": None, "failures": 0, "error_code": None, "error": None,
                "wants": [], "field_errors": {},
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

    def want(self, symbol: str, field: str) -> bool:
        """Record that a sheet asked for an optional field, so each refresh fetches it.

        Returns:
            True when the field was not wanted before.
        """
        with self._lock:
            wants = self._entries[symbol]["wants"]
            if field in wants:
                return False
            wants.append(field)
            return True

    def wanted(self) -> dict[str, list[str]]:
        """Return symbol -> optional fields asked for, for symbols that have any."""
        with self._lock:
            return {s: list(e["wants"]) for s, e in self._entries.items() if e["wants"]}

    def set_ok(self, symbol: str, fields: dict, field_errors: dict | None = None) -> None:
        """Store a successful fetch.

        Args:
            symbol: A registered symbol.
            fields: The values obtained.
            field_errors: field -> {"code", "message", "retryable"} for optional fields that were
                asked for and could not be obtained, while the rest was.
        """
        with self._lock:
            entry = self._entries[symbol]
            entry.update(status=OK, fields=dict(fields), fetched_at=time.time(),
                         failures=0, error_code=None, error=None,
                         field_errors=dict(field_errors or {}))

    def set_not_found(self, symbol: str, message: str) -> None:
        """Record that the provider answered that it has no listing for the symbol."""
        with self._lock:
            entry = self._entries[symbol]
            entry.update(status=NOT_FOUND, fields={}, fetched_at=None, failures=0,
                         error_code=NOT_FOUND, error=message)

    def set_failure(self, symbol: str, code: str, message: str) -> None:
        """Record a failed fetch. Any earlier value is kept and served as stale.

        Args:
            symbol: A registered symbol.
            code: Failure code shown in the spreadsheet (NO_NETWORK, RATE_LIMITED, ...).
            message: One sentence in plain words naming the cause.
        """
        with self._lock:
            entry = self._entries[symbol]
            entry["failures"] += 1
            entry.update(error_code=code, error=message)
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
        if not isinstance(loaded, dict) or not all(isinstance(e, dict) for e in loaded.values()):
            return 0
        for entry in loaded.values():
            entry.setdefault("error_code", None)   # files written before failure codes existed
            entry.setdefault("wants", [])          # files written before optional fields existed
            entry.setdefault("field_errors", {})
        with self._lock:
            self._entries.update(loaded)
            return len(loaded)
