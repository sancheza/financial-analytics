"""HTTP API. Every route answers from memory and never waits on an upstream fetch."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.http_client import HttpClient
from quote_server import cache as cache_mod
from quote_server import providers
from quote_server import symbols
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher
from quote_server.watchdog import Watchdog

VERSION = "1.0.0"
STALE_AFTER_CYCLES = 1.5   # older than this many refresh intervals: flagged stale
EXPIRED_AFTER_CYCLES = 3   # older than this many: served as an error
MAX_BATCH = 500
MAX_WAIT_SECONDS = 30.0

FIELDS = {symbols.STOCK: ("price", "dividend_yield", "yield"),
          symbols.BOND: ("price", "term", "ytw", "yield")}
# "yield" means the yield that fits the symbol: yield to worst for a bond, dividend yield for a stock.
YIELD_FIELD = {symbols.STOCK: "dividend_yield", symbols.BOND: "ytw"}
# Fields fetched only for symbols a sheet asked them for, because each needs its own request.
OPTIONAL_FIELDS = {symbols.STOCK: ("dividend_yield",), symbols.BOND: ()}

log = logging.getLogger("quote_server")


class BatchRequest(BaseModel):
    """Body of POST /batch-refresh."""

    symbols: list[str] = Field(max_length=MAX_BATCH)
    field: str = "price"


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _source_name(entry: dict) -> str:
    return providers.SOURCE_BY_KIND[entry["kind"]]


def _age_text(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _stale_message(symbol: str, age: float, last_error: str | None, refresher_alive: bool) -> str:
    """Say how old the price is and why it was not refreshed."""
    if not refresher_alive:
        reason = "the quote server's refresh has stopped; restart the server"
    elif last_error:
        reason = f"the last refresh failed: {last_error}"
    else:
        reason = "no refresh has completed since"
    return f"{symbol} price is {_age_text(age)} old; {reason}"


def _lookup(cache: QuoteCache, refresher: Refresher, raw: str, field: str) -> dict:
    """Return the quote dict for one symbol, or raise HTTPException."""
    symbol = symbols.resolve(raw)
    kind = symbols.classify(symbol)
    if kind is None:
        raise _error(400, *symbols.bad_symbol_reason(raw, symbol))
    if field not in FIELDS[kind]:
        raise _error(400, "BAD_FIELD", f"{kind} symbols support: {', '.join(FIELDS[kind])}")
    field = YIELD_FIELD[kind] if field == "yield" else field
    newly_registered = cache.register(symbol, kind)
    newly_wanted = field in OPTIONAL_FIELDS[kind] and cache.want(symbol, field)
    if newly_registered or newly_wanted:
        refresher.request([symbol])
    entry = cache.get(symbol)
    if entry["status"] == cache_mod.NOT_FOUND:
        raise _error(404, "NOT_FOUND", entry["error"] or f"no price found for {symbol}")
    if entry["status"] == cache_mod.PENDING:
        if not refresher.alive:
            raise _error(503, "REFRESH_STOPPED",
                         f"{symbol} cannot be fetched: the quote server's refresh has stopped; restart the server")
        raise _error(202, "PENDING", f"{symbol} is being fetched; retry shortly")
    if entry["status"] == cache_mod.UPSTREAM:
        raise _error(503, entry["error_code"] or "UPSTREAM",
                     entry["error"] or f"no price for {symbol}: the last fetch failed")
    age = time.time() - entry["fetched_at"]
    if age > refresher.interval * EXPIRED_AFTER_CYCLES:
        raise _error(503, "STALE", _stale_message(symbol, age, entry["error"], refresher.alive))
    if field not in entry["fields"]:
        failed = entry["field_errors"].get(field)
        if failed:
            raise _error(503 if failed.get("retryable") else 404, failed["code"], failed["message"])
        if field in OPTIONAL_FIELDS[kind]:
            raise _error(202, "PENDING", f"{symbol} {field.replace('_', ' ')} is being fetched; retry shortly")
        raise _error(404, "NO_FIELD", f"{symbol} has no {field}; {_source_name(entry)} did not supply it")
    return {
        "symbol": symbol,
        "type": kind,
        "field": field,
        "value": entry["fields"][field],
        "fields": entry["fields"],
        "stale": age > refresher.interval * STALE_AFTER_CYCLES,
        "age_seconds": round(age, 1),
    }


def create_app(cache: QuoteCache, refresher: Refresher, watchdog: Watchdog | None = None,
               client: HttpClient | None = None) -> FastAPI:
    """Build the FastAPI app around a cache, its refresher, and an optional watchdog.

    `client` is the HTTP client the price lookups use; `/health` reports its per-source state.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        refresher.start()
        if watchdog is not None:
            watchdog.start()
        yield
        if watchdog is not None:
            watchdog.stop()
        refresher.stop()

    app = FastAPI(title="Quote Server", version=VERSION, lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def api_error(_, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "code": "HTTP_ERROR", "message": str(exc.detail)}
        return JSONResponse({"error": detail}, status_code=exc.status_code)

    @app.get("/quote")
    async def quote(symbol: str, field: str = "price") -> dict:
        """Return one cached value for a ticker or CUSIP."""
        return _lookup(cache, refresher, symbol, field.lower())

    @app.post("/batch-refresh")
    async def batch_refresh(body: BatchRequest) -> dict:
        """Register symbols, queue them for an immediate refresh, return what is cached."""
        results: dict[str, dict] = {}
        for raw in body.symbols:
            try:
                results[raw] = _lookup(cache, refresher, raw, body.field.lower())
            except HTTPException as exc:
                results[raw] = {"error": exc.detail}
        refresher.request([symbols.resolve(r) for r in body.symbols
                           if symbols.classify(symbols.resolve(r))])
        return {"results": results}

    @app.post("/refresh-all")
    async def refresh_all(wait: float = 5.0) -> dict:
        """Refresh every known symbol now; wait up to `wait` seconds for it to finish."""
        done = refresher.refresh_all_async()
        complete = await asyncio.to_thread(done.wait, min(max(wait, 0.0), MAX_WAIT_SECONDS))
        return {"symbols": len(cache.symbols()), "complete": complete}

    @app.get("/health")
    async def health() -> dict:
        """Report liveness and cache size."""
        return {"status": "ok" if refresher.alive else "degraded", "version": VERSION,
                "symbols": len(cache.symbols()), "refresher_alive": refresher.alive,
                "refresh_seconds": refresher.interval,
                "last_cycle_at": refresher.last_cycle_at,
                "started_at": watchdog.started_at if watchdog is not None else None,
                "restart_on_hang": watchdog.restart if watchdog is not None else False,
                "sources": client.status() if client is not None else {}}

    return app
