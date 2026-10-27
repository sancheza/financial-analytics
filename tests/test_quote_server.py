"""Tests for quote_server: symbols, cache, refresher, and the HTTP API."""

import threading
import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from quote_server import providers, symbols
from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher


def fake_fetch_many(kinds, **_):
    """Stand-in for providers.fetch_many with fixed answers."""
    out = {}
    for symbol, kind in kinds.items():
        if symbol == "MISSING":
            out[symbol] = providers.NotFound(symbol)
        elif symbol == "BROKEN":
            out[symbol] = ValueError("boom")
        elif kind == "bond":
            out[symbol] = {"price": 99.5, "ytw": 4.5, "term": 10.0}
        else:
            out[symbol] = {"price": 123.45}
    return out


@pytest.fixture
def client(tmp_path):
    cache = QuoteCache(tmp_path / "cache.json")
    refresher = Refresher(cache, interval=300, fetch_many=fake_fetch_many)
    with TestClient(create_app(cache, refresher)) as test_client:
        test_client.cache, test_client.refresher = cache, refresher
        yield test_client


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", symbols.STOCK), ("BRK-B", symbols.STOCK), ("^GSPC", symbols.STOCK),
    ("912810UF3", symbols.BOND), ("91282CMM0", symbols.BOND),
    ("912810UF4", None), ("", None), ("BAD SYM", None), ("A" * 12, None),
    ("0700.HK", symbols.STOCK), ("7203.T", symbols.STOCK), ("123", None), ("12.5", None),
])
def test_classify(symbol, expected):
    assert symbols.classify(symbol) == expected


@pytest.mark.parametrize("raw,code", [
    ("5", "NEED_TICKER_OR_CUSIP"), ("1234.5", "NEED_TICKER_OR_CUSIP"), ("-3", "NEED_TICKER_OR_CUSIP"),
    ("1e5", "NEED_TICKER_OR_CUSIP"), ("12345678", "NEED_TICKER_OR_CUSIP"),
    ("123456789", "NEED_TICKER_OR_CUSIP"),   # nine digits that fail the CUSIP check digit
    ("Apple Inc", "BAD_SYMBOL"), ("912810UF4", "BAD_SYMBOL"), ("AA PL", "BAD_SYMBOL"),
])
def test_bad_symbol_reason(raw, code):
    assert symbols.bad_symbol_reason(raw, symbols.normalize(raw))[0] == code


@pytest.mark.parametrize("text,expected", [
    ("5Y UST 91282CJF9", "91282CJF9"),
    ("5y ust 91282cjf9", "91282CJF9"),
    ("UST-91282CJF9", "91282CJF9"),
    ("(91282CJF9) 5Y", "91282CJF9"),
    ("10Y note 912810UF3 coupon 4.625", "912810UF3"),
    ("AAPL", "AAPL"),
    ("91282CJF9", "91282CJF9"),
    ("91282CJF9 91282CJF9", "91282CJF9"),   # the same CUSIP twice is not ambiguous
])
def test_resolve_finds_one_cusip_in_text(text, expected):
    assert symbols.resolve(text) == expected


@pytest.mark.parametrize("text", [
    "5Y UST 91282CJF8",       # check digit wrong
    "Apple Inc", "10 yr note", "ALPHABETS corp",   # no CUSIP; a 9-letter word is not one
])
def test_resolve_leaves_text_without_a_valid_cusip(text):
    resolved = symbols.resolve(text)
    assert symbols.classify(resolved) is None
    assert symbols.bad_symbol_reason(text, resolved)[0] == "BAD_SYMBOL"


def test_two_different_cusips_are_ambiguous():
    text = "912810UF3 and 91282CJF9"
    code, message = symbols.bad_symbol_reason(text, symbols.resolve(text))
    assert code == "AMBIGUOUS_SYMBOL" and "912810UF3" in message and "91282CJF9" in message


def test_leading_zero_hint_only_for_eight_digits():
    assert "Text" in symbols.bad_symbol_reason("37833100", "37833100")[1]
    assert "Text" not in symbols.bad_symbol_reason("5", "5")[1]


def test_cache_register_and_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    cache = QuoteCache(path)
    assert cache.register("AAPL", "stock") is True
    assert cache.register("AAPL", "stock") is False
    cache.set_ok("AAPL", {"price": 1.5})
    cache.save()
    other = QuoteCache(path)
    assert other.load() == 1
    assert other.get("AAPL")["fields"] == {"price": 1.5}


def test_cache_load_ignores_corrupt_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert QuoteCache(path).load() == 0


def test_cache_failure_keeps_last_value():
    cache = QuoteCache()
    cache.register("AAPL", "stock")
    cache.set_ok("AAPL", {"price": 2.0})
    cache.set_failure("AAPL", "net down")
    entry = cache.get("AAPL")
    assert entry["status"] == "OK" and entry["fields"] == {"price": 2.0}
    assert entry["failures"] == 1


def test_cache_is_thread_safe():
    cache = QuoteCache()

    def worker(n):
        for i in range(200):
            sym = f"S{n}X{i}"
            cache.register(sym, "stock")
            cache.set_ok(sym, {"price": float(i)})
            cache.get(sym)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(cache.symbols()) == 1600


def test_fetch_bond_computes_term(monkeypatch):
    monkeypatch.setattr(providers.webull_bond_fetcher, "fetch_treasury_price",
                        lambda cusip, timeout: {"price": "91.8", "ytw": "5.36",
                                                "maturityDate": "2036-01-01"})
    fields = providers.fetch_bond("912810UF3", today=date(2026, 1, 1))
    assert fields == {"price": 91.8, "ytw": 5.36, "term": 10.0}


def test_fetch_bond_without_listing_raises(monkeypatch):
    monkeypatch.setattr(providers.webull_bond_fetcher, "fetch_treasury_price",
                        lambda cusip, timeout: None)
    with pytest.raises(providers.NotFound):
        providers.fetch_bond("912810UF3")


def test_fetch_many_reports_errors_per_symbol():
    def stock(symbol):
        if symbol == "BAD":
            raise providers.NotFound(symbol)
        return {"price": 1.0}

    out = providers.fetch_many({"OK": "stock", "BAD": "stock"}, fetch_stock_fn=stock)
    assert out["OK"] == {"price": 1.0} and isinstance(out["BAD"], providers.NotFound)


def test_unknown_symbol_returns_pending_then_value(client):
    first = client.get("/quote", params={"symbol": "aapl"})
    assert first.status_code == 202 and first.json()["error"]["code"] == "PENDING"
    assert wait_for(lambda: client.get("/quote", params={"symbol": "AAPL"}).status_code == 200)
    body = client.get("/quote", params={"symbol": "AAPL"}).json()
    assert body["value"] == 123.45 and body["type"] == "stock" and body["stale"] is False


def test_bond_fields(client):
    client.get("/quote", params={"symbol": "912810UF3"})
    assert wait_for(lambda: client.get("/quote", params={"symbol": "912810UF3"}).status_code == 200)
    for field, expected in (("price", 99.5), ("ytw", 4.5), ("term", 10.0)):
        assert client.get("/quote", params={"symbol": "912810UF3", "field": field}).json()["value"] == expected


def test_text_around_a_cusip_is_accepted(client):
    first = client.get("/quote", params={"symbol": "5Y UST 91282CJF9"})
    assert first.status_code == 202
    assert wait_for(lambda: client.get("/quote", params={"symbol": "5Y UST 91282CJF9"}).status_code == 200)
    body = client.get("/quote", params={"symbol": "5y ust 91282cjf9", "field": "ytw"}).json()
    assert body["symbol"] == "91282CJF9" and body["type"] == "bond" and body["value"] == 4.5
    assert [s for s, _ in client.cache.symbols()] == ["91282CJF9"]   # one entry, not one per phrasing


def test_two_cusips_in_one_cell_report_ambiguity(client):
    response = client.get("/quote", params={"symbol": "912810UF3 and 91282CJF9"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "AMBIGUOUS_SYMBOL"


def test_batch_refresh_resolves_text(client):
    body = {"symbols": ["5Y UST 91282CJF9"]}
    client.post("/batch-refresh", json=body)
    assert wait_for(lambda: "value" in client.post("/batch-refresh", json=body).json()["results"]["5Y UST 91282CJF9"])
    assert client.post("/batch-refresh", json=body).json()["results"]["5Y UST 91282CJF9"]["symbol"] == "91282CJF9"


def test_number_gets_its_own_message(client):
    response = client.get("/quote", params={"symbol": "42"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "NEED_TICKER_OR_CUSIP" and "ticker" in error["message"] and "CUSIP" in error["message"]


def test_error_codes(client):
    assert client.get("/quote", params={"symbol": "bad sym"}).status_code == 400
    assert client.get("/quote", params={"symbol": "AAPL", "field": "ytw"}).json()["error"]["code"] == "BAD_FIELD"
    client.get("/quote", params={"symbol": "MISSING"})
    assert wait_for(lambda: client.get("/quote", params={"symbol": "MISSING"}).status_code == 404)
    client.get("/quote", params={"symbol": "BROKEN"})
    assert wait_for(lambda: client.get("/quote", params={"symbol": "BROKEN"}).status_code == 503)
    assert client.get("/quote", params={"symbol": "BROKEN"}).json()["error"]["code"] == "UPSTREAM"


def test_stale_and_expired(client):
    client.get("/quote", params={"symbol": "AAPL"})
    assert wait_for(lambda: client.get("/quote", params={"symbol": "AAPL"}).status_code == 200)
    with client.cache._lock:
        client.cache._entries["AAPL"]["fetched_at"] = time.time() - 600   # 2 cycles old
    body = client.get("/quote", params={"symbol": "AAPL"}).json()
    assert body["stale"] is True and body["value"] == 123.45
    with client.cache._lock:
        client.cache._entries["AAPL"]["fetched_at"] = time.time() - 1200  # 4 cycles old
    expired = client.get("/quote", params={"symbol": "AAPL"})
    assert expired.status_code == 503 and expired.json()["error"]["code"] == "STALE"


def test_batch_refresh(client):
    body = {"symbols": ["AAPL", "912810UF3", "bad sym"]}
    first = client.post("/batch-refresh", json=body).json()["results"]
    assert first["bad sym"]["error"]["code"] == "BAD_SYMBOL"
    assert wait_for(lambda: "value" in client.post("/batch-refresh", json=body).json()["results"]["AAPL"])
    results = client.post("/batch-refresh", json=body).json()["results"]
    assert results["AAPL"]["value"] == 123.45 and results["912810UF3"]["value"] == 99.5


def test_batch_refresh_rejects_oversized_request(client):
    assert client.post("/batch-refresh", json={"symbols": ["AAPL"] * 501}).status_code == 422


def test_refresh_all_waits_for_completion(client):
    for symbol in ("AAPL", "912810UF3"):
        client.get("/quote", params={"symbol": symbol})
    body = client.post("/refresh-all", params={"wait": 5}).json()
    assert body == {"symbols": 2, "complete": True}
    assert client.get("/quote", params={"symbol": "AAPL"}).status_code == 200


def test_refresh_all_with_no_symbols(client):
    assert client.post("/refresh-all").json() == {"symbols": 0, "complete": True}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["refresh_seconds"] == 300


def test_symbols_survive_restart(tmp_path):
    path = tmp_path / "c.json"
    cache = QuoteCache(path)
    cache.register("AAPL", "stock")
    cache.save()
    restarted = QuoteCache(path)
    restarted.load()
    refresher = Refresher(restarted, interval=300, fetch_many=fake_fetch_many)
    refresher.start()
    try:
        assert wait_for(lambda: restarted.get("AAPL")["status"] == "OK")
    finally:
        refresher.stop()
