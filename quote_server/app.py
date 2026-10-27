"""HTTP API. Every route answers from memory and never waits on an upstream fetch."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from quote_server import cache as cache_mod
from quote_server import symbols
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher

VERSION = "1.0.0"
STALE_AFTER_CYCLES = 1.5   # older than this many refresh intervals: flagged stale
EXPIRED_AFTER_CYCLES = 3   # older than this many: served as an error
MAX_BATCH = 500
MAX_WAIT_SECONDS = 30.0

FIELDS = {symbols.STOCK: ("price",), symbols.BOND: ("price", "term", "ytw")}

log = logging.getLogger("quote_server")


class BatchRequest(BaseModel):
    """Body of POST /batch-refresh."""

    symbols: list[str] = Field(max_length=MAX_BATCH)
    field: str = "price"


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _lookup(cache: QuoteCache, refresher: Refresher, raw: str, field: str) -> dict:
    """Return the quote dict for one symbol, or raise HTTPException."""
    symbol = symbols.resolve(raw)
    kind = symbols.classify(symbol)
    if kind is None:
        raise _error(400, *symbols.bad_symbol_reason(raw, symbol))
    if field not in FIELDS[kind]:
        raise _error(400, "BAD_FIELD", f"{kind} symbols support: {', '.join(FIELDS[kind])}")
    if cache.register(symbol, kind):
        refresher.request([symbol])
    entry = cache.get(symbol)
    if entry["status"] == cache_mod.NOT_FOUND:
        raise _error(404, "NOT_FOUND", f"no price found for {symbol}")
    if entry["status"] == cache_mod.PENDING:
        raise _error(202, "PENDING", f"{symbol} is being fetched; retry shortly")
    if entry["status"] == cache_mod.UPSTREAM:
        raise _error(503, "UPSTREAM", f"price source failed for {symbol}: {entry['error']}")
    age = time.time() - entry["fetched_at"]
    if age > refresher.interval * EXPIRED_AFTER_CYCLES:
        raise _error(503, "STALE", f"{symbol} price is {int(age)} s old")
    if field not in entry["fields"]:
        raise _error(404, "NOT_FOUND", f"{symbol} has no {field}")
    return {
        "symbol": symbol,
        "type": kind,
        "field": field,
        "value": entry["fields"][field],
        "fields": entry["fields"],
        "stale": age > refresher.interval * STALE_AFTER_CYCLES,
        "age_seconds": round(age, 1),
    }


def create_app(cache: QuoteCache, refresher: Refresher) -> FastAPI:
    """Build the FastAPI app around a cache and its refresher."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        refresher.start()
        yield
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
        return {"status": "ok", "version": VERSION, "symbols": len(cache.symbols()),
                "refresh_seconds": refresher.interval,
                "last_cycle_at": refresher.last_cycle_at}

    return app
