"""Tests for quote_server failure handling: what is NotFound, what is retried, what the cell says."""

import sqlite3
import threading
import time

import pandas
import pytest
import requests
import yfinance
from fastapi.testclient import TestClient

from core import http_client
from core.http_client import HttpClientError
from quote_server import providers, refresher as refresher_module
from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}


def http_error(status):
    return requests.HTTPError(f"HTTP {status}", response=FakeResponse(status))


def fast_client(**options):
    """A client with no pacing and no real waiting."""
    return providers.make_client(request_delay=0, retry_delays=(0.0, 0.0), sleep=lambda _: None, **options)


def fake_ticker(history=None, error=None, calls=None):
    """Return a stand-in for yfinance.Ticker whose history() returns a frame or raises.

    `error` may be one exception, or a list of outcomes used in turn (exception or frame).
    """
    outcomes = list(error) if isinstance(error, list) else None

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **_):
            if calls is not None:
                calls.append(self.symbol)
            if outcomes is not None:
                outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
            if error is not None:
                raise error
            return history
    return Ticker


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- fetch_stock: what counts as "not found" ---------------------------------------------

def test_stock_price_is_latest_close(monkeypatch):
    frame = pandas.DataFrame({"Close": [10.0, 11.5, float("nan")]})
    monkeypatch.setattr(providers.yfinance, "Ticker", fake_ticker(history=frame))
    assert providers.fetch_stock("AAPL", fast_client()) == {"price": 11.5}


@pytest.mark.parametrize("error", [
    yfinance.exceptions.YFPricesMissingError("ZZZ", "5d"),
    http_error(404),
])
def test_stock_is_not_found_only_when_yahoo_says_so(monkeypatch, error):
    monkeypatch.setattr(providers.yfinance, "Ticker", fake_ticker(error=error))
    with pytest.raises(providers.NotFound) as caught:
        providers.fetch_stock("ZZZ", fast_client())
    assert caught.value.message == "Yahoo Finance has no listing for ZZZ; check the symbol"


def test_stock_network_failure_is_not_not_found(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=requests.ConnectionError("dns"), calls=calls))
    with pytest.raises(HttpClientError) as caught:
        providers.fetch_stock("AMZN", fast_client())
    assert len(calls) == 3                                    # three attempts
    failure = providers.as_failure(caught.value, providers.STOCK_SOURCE)
    assert failure.code == providers.NO_NETWORK and failure.retryable


def test_stock_recovers_when_a_retry_succeeds(monkeypatch):
    frame = pandas.DataFrame({"Close": [7.0]})
    calls = []
    monkeypatch.setattr(providers.yfinance, "Ticker", fake_ticker(
        error=[requests.ConnectionError("x"), requests.ReadTimeout("x"), frame], calls=calls))
    assert providers.fetch_stock("AMZN", fast_client()) == {"price": 7.0}
    assert len(calls) == 3


def test_stock_with_no_price_rows_is_a_failure_not_not_found(monkeypatch):
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(history=pandas.DataFrame({"Close": []})))
    with pytest.raises(providers.FetchFailed) as caught:
        providers.fetch_stock("AMZN", fast_client())
    assert caught.value.code == providers.NO_DATA and caught.value.retryable


def test_unknown_ticker_does_not_count_against_the_source(monkeypatch):
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=yfinance.exceptions.YFPricesMissingError("ZZZ", "5d")))
    client = fast_client(breaker_threshold=1)
    for _ in range(3):
        with pytest.raises(providers.NotFound):
            providers.fetch_stock("ZZZ", client)
    assert client.status()["Yahoo Finance"]["status"] == "ok"


def test_yahoo_rate_limit_stops_asking_yahoo(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=yfinance.exceptions.YFRateLimitError(), calls=calls))
    client = fast_client(block_file=tmp_path / "blocks.json")
    with pytest.raises(HttpClientError) as first:
        providers.fetch_stock("AMZN", client)
    with pytest.raises(HttpClientError):
        providers.fetch_stock("GOOG", client)
    assert calls == ["AMZN"]                                  # not retried, and not asked again
    failure = providers.as_failure(first.value, providers.STOCK_SOURCE)
    assert failure.code == providers.RATE_LIMITED and not failure.retryable
    assert "not asking again until" in failure.message
    assert client.status()["Yahoo Finance"]["status"] == "limited"


# --- fetch_bond ----------------------------------------------------------------------------

def patch_webull(monkeypatch, outcome):
    def fetch(cusip, timeout, get):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(providers.webull_bond_fetcher, "fetch_treasury_price", fetch)


def test_bond_unlisted_cusip_is_not_found(monkeypatch):
    patch_webull(monkeypatch, HttpClientError(http_client.HTTP_STATUS, "x", host="Webull", status=404,
                                              retryable=False))
    with pytest.raises(providers.NotFound) as caught:
        providers.fetch_bond("912810UF3", fast_client())
    assert "Webull" in caught.value.message


def test_bond_request_goes_through_the_client(monkeypatch):
    seen = []

    def fetch(cusip, timeout, get):
        seen.append(get)
        return {"price": "99.5"}

    monkeypatch.setattr(providers.webull_bond_fetcher, "fetch_treasury_price", fetch)
    client = fast_client()
    assert providers.fetch_bond("912810UF3", client) == {"price": 99.5}
    assert seen[0].func == client.get and seen[0].keywords == {"host": "Webull"}


def test_bond_changed_page_is_source_changed_and_not_retried(monkeypatch):
    patch_webull(monkeypatch, RuntimeError("Could not find window.__initState__"))
    with pytest.raises(providers.FetchFailed) as caught:
        providers.fetch_bond("912810UF3", fast_client())
    assert caught.value.code == providers.SOURCE_CHANGED and not caught.value.retryable
    assert "__initState__" not in caught.value.message


def test_bond_unreadable_price_is_source_changed(monkeypatch):
    patch_webull(monkeypatch, {"price": "n/a"})
    with pytest.raises(providers.FetchFailed) as caught:
        providers.fetch_bond("912810UF3", fast_client())
    assert caught.value.code == providers.SOURCE_CHANGED


# --- as_failure: cause -> code and message -------------------------------------------------

def client_error(kind, status=None, retryable=True, blocks=False, until=None):
    return HttpClientError(kind, "generic text", host="Yahoo Finance", status=status,
                           retryable=retryable, blocks_host=blocks, cooldown_until=until)


@pytest.mark.parametrize("error,code,retryable,words", [
    (client_error(http_client.NETWORK), providers.NO_NETWORK, True, "internet"),
    (client_error(http_client.TIMEOUT), providers.SOURCE_TIMEOUT, True, "did not answer"),
    (client_error(http_client.RATE_LIMITED, 429, False, True), providers.RATE_LIMITED, False, "trying again later"),
    (client_error(http_client.RATE_LIMITED, 429, False, True, "2026-10-30T18:30:00Z"),
     providers.RATE_LIMITED, False, "not asking again until"),
    (client_error(http_client.REFUSED, 403, False, True), providers.SOURCE_REFUSED, False, "HTTP 403"),
    (client_error(http_client.SERVER_ERROR, 503), providers.SOURCE_DOWN, True, "HTTP 503"),
    (client_error(http_client.HTTP_STATUS, 400, False), providers.SOURCE_ERROR, False, "HTTP 400"),
    (requests.ConnectionError("x"), providers.NO_NETWORK, True, "internet"),
    (requests.ReadTimeout("x"), providers.SOURCE_TIMEOUT, True, "did not answer"),
    (yfinance.exceptions.YFRateLimitError(), providers.RATE_LIMITED, False, "limiting"),
    (sqlite3.OperationalError("unable to open database file"), providers.LOCAL_CACHE, True, "cache"),
    (ValueError("secret detail"), providers.INTERNAL, False, "ValueError"),
])
def test_as_failure_maps_causes(error, code, retryable, words):
    failure = providers.as_failure(error, "Yahoo Finance")
    assert (failure.code, failure.retryable) == (code, retryable)
    assert words in failure.message
    assert "secret detail" not in failure.message and "generic text" not in failure.message


# --- fetch_many: one batch ------------------------------------------------------------------

def test_failing_source_is_skipped_after_a_few_symbols(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=requests.ConnectionError("down"), calls=calls))
    kinds = {f"S{i}": "stock" for i in range(40)}
    results = providers.fetch_many(kinds, client=fast_client())
    assert len(calls) < 40 * 3
    assert len(results) == 40
    assert all(isinstance(r, providers.FetchFailed) and r.code == providers.NO_NETWORK
               for r in results.values())


def test_rate_limit_stops_that_source_only(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=yfinance.exceptions.YFRateLimitError(), calls=calls))
    patch_webull(monkeypatch, {"price": "99.0"})
    kinds = {f"S{i}": "stock" for i in range(30)} | {"912810UF3": "bond"}
    results = providers.fetch_many(kinds, client=fast_client())
    assert len(calls) <= providers.MAX_WORKERS
    assert results["912810UF3"] == {"price": 99.0}
    assert {r.code for s, r in results.items() if s != "912810UF3"} == {providers.RATE_LIMITED}


def test_unexpected_error_in_one_symbol_does_not_stop_the_batch():
    def fetch(symbol):
        if symbol == "BAD":
            raise AttributeError("bug")
        return {"price": 1.0}

    results = providers.fetch_many({"BAD": "stock", "OK": "stock"}, fetch_stock_fn=fetch,
                                   fetch_bond_fn=fetch)
    assert results["OK"] == {"price": 1.0}
    assert results["BAD"].code == providers.INTERNAL


def test_fetch_many_with_both_lookups_given_makes_no_client(monkeypatch):
    monkeypatch.setattr(providers, "make_client", lambda *a, **k: pytest.fail("no client needed"))
    out = providers.fetch_many({"A": "stock"}, fetch_stock_fn=lambda s: {"price": 1.0},
                               fetch_bond_fn=lambda s: {"price": 2.0})
    assert out == {"A": {"price": 1.0}}


# --- Refresher: results, survival, retry timing --------------------------------------------

def make_refresher(fetch_many, interval=300):
    cache = QuoteCache()
    cache.register("AMZN", "stock")
    return cache, Refresher(cache, interval=interval, fetch_many=fetch_many)


def test_network_failure_never_becomes_not_found():
    cache, refresher = make_refresher(lambda kinds: {
        s: providers.FetchFailed(providers.NO_NETWORK, "cannot reach Yahoo Finance") for s in kinds})
    assert refresher.refresh(["AMZN"]) == ["AMZN"]
    entry = cache.get("AMZN")
    assert entry["status"] == "UPSTREAM" and entry["error_code"] == "NO_NETWORK"
    assert entry["error"] == "cannot reach Yahoo Finance"


def test_failure_keeps_the_last_price():
    cache, refresher = make_refresher(lambda kinds: {
        s: providers.FetchFailed(providers.SOURCE_TIMEOUT, "Yahoo Finance did not answer in time")
        for s in kinds})
    cache.set_ok("AMZN", {"price": 200.0})
    refresher.refresh(["AMZN"])
    entry = cache.get("AMZN")
    assert entry["status"] == "OK" and entry["fields"] == {"price": 200.0}
    assert entry["error_code"] == "SOURCE_TIMEOUT"


def test_recovery_clears_the_error():
    answers = iter([providers.FetchFailed(providers.NO_NETWORK, "down"), {"price": 5.0}])
    cache, refresher = make_refresher(lambda kinds: {s: next(answers) for s in kinds})
    refresher.refresh(["AMZN"])
    refresher.refresh(["AMZN"])
    entry = cache.get("AMZN")
    assert entry["status"] == "OK" and entry["error_code"] is None and entry["failures"] == 0


def test_non_retryable_failure_is_not_scheduled_for_a_quick_retry():
    cache, refresher = make_refresher(lambda kinds: {
        s: providers.FetchFailed(providers.RATE_LIMITED, "limiting", retryable=False) for s in kinds})
    assert refresher.refresh(["AMZN"]) == []


def test_log_names_the_cause(caplog):
    cache, refresher = make_refresher(lambda kinds: {
        s: providers.FetchFailed(providers.NO_NETWORK, "cannot reach Yahoo Finance") for s in kinds})
    with caplog.at_level("INFO", logger="quote_server"):
        refresher.refresh(["AMZN"])
        refresher.refresh(["AMZN"])
    failures = [r for r in caplog.records if r.getMessage().startswith("AMZN fetch failed")]
    assert len(failures) == 1                                  # logged when it starts, not every cycle
    assert failures[0].scope == "quote_server" and failures[0].outcome == "failure"
    summary = [r for r in caplog.records if r.getMessage().startswith("refreshed")]
    assert "NO_NETWORK x1: cannot reach Yahoo Finance" in summary[0].getMessage()


def test_wide_failure_is_one_summary_line_not_one_line_per_symbol(caplog):
    cache = QuoteCache()
    for n in range(20):
        cache.register(f"S{n}", "stock")
    refresher = Refresher(cache, interval=300, fetch_many=lambda kinds: {
        s: providers.FetchFailed(providers.NO_NETWORK, "cannot reach Yahoo Finance") for s in kinds})
    with caplog.at_level("INFO", logger="quote_server"):
        refresher.refresh([s for s, _ in cache.symbols()])
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "20 failed (NO_NETWORK x20: cannot reach Yahoo Finance)" in caplog.records[0].getMessage()


def test_thread_survives_a_crashing_fetch_and_retries(monkeypatch):
    monkeypatch.setattr(refresher_module, "RETRY_DELAYS_SECONDS", (0.05,))
    calls = []

    def fetch_many(kinds):
        calls.append(sorted(kinds))
        if len(calls) == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return {s: {"price": 9.0} for s in kinds}

    cache, refresher = make_refresher(fetch_many)
    refresher.start()
    try:
        assert wait_for(lambda: cache.get("AMZN")["status"] == "OK")
        assert refresher.alive and len(calls) == 2
    finally:
        refresher.stop()


def test_failed_symbols_are_fetched_again_soon(monkeypatch):
    monkeypatch.setattr(refresher_module, "RETRY_DELAYS_SECONDS", (0.05,))
    calls = []

    def fetch_many(kinds):
        calls.append(sorted(kinds))
        if len(calls) == 1:
            return {s: providers.FetchFailed(providers.NO_NETWORK, "down") for s in kinds}
        return {s: {"price": 3.0} for s in kinds}

    cache, refresher = make_refresher(fetch_many)
    refresher.start()
    try:
        assert wait_for(lambda: cache.get("AMZN")["status"] == "OK")
        time.sleep(0.2)
        assert len(calls) == 2                                 # no further retries once it succeeds
    finally:
        refresher.stop()


def test_full_refresh_runs_on_schedule(monkeypatch):
    calls = []
    cache, refresher = make_refresher(lambda kinds: calls.append(1) or {s: {"price": 1.0} for s in kinds},
                                      interval=0.1)
    refresher.start()
    try:
        assert wait_for(lambda: len(calls) >= 4)
        assert refresher.last_cycle_at is not None
    finally:
        refresher.stop()


def test_wait_never_exceeds_the_poll_interval():
    cache, refresher = make_refresher(lambda kinds: {}, interval=300)
    assert refresher._next_wait(time.time() + 300) <= refresher_module.POLL_SECONDS
    assert refresher._next_wait(time.time() - 5) == 0.0        # overdue (for example after sleep)


def test_refresh_all_survives_a_crash():
    def fetch_many(kinds):
        raise RuntimeError("bug")

    cache, refresher = make_refresher(fetch_many)
    assert refresher.refresh_all_async().wait(3.0)


# --- API messages --------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    cache = QuoteCache()
    refresher = Refresher(cache, interval=300, fetch_many=lambda kinds: {})
    with TestClient(create_app(cache, refresher)) as test_client:
        test_client.cache = cache
        yield test_client


def register(api, symbol, kind="stock"):
    api.cache.register(symbol, kind)


def error_of(api, symbol, **params):
    response = api.get("/quote", params={"symbol": symbol, **params})
    return response.status_code, response.json()["error"]


def test_upstream_error_carries_its_code_and_cause(api):
    register(api, "AMZN")
    api.cache.set_failure("AMZN", "NO_NETWORK", "cannot reach Yahoo Finance; check the internet connection")
    status, error = error_of(api, "AMZN")
    assert status == 503 and error["code"] == "NO_NETWORK"
    assert error["message"] == "cannot reach Yahoo Finance; check the internet connection"


def test_not_found_names_the_source(api):
    register(api, "ZZZ")
    api.cache.set_not_found("ZZZ", "Yahoo Finance has no listing for ZZZ; check the symbol")
    status, error = error_of(api, "ZZZ")
    assert status == 404 and error["code"] == "NOT_FOUND" and "Yahoo Finance" in error["message"]


def test_stale_message_gives_age_and_last_failure(api):
    register(api, "AMZN")
    api.cache.set_ok("AMZN", {"price": 1.0})
    api.cache.set_failure("AMZN", "NO_NETWORK", "cannot reach Yahoo Finance; check the internet connection")
    with api.cache._lock:
        api.cache._entries["AMZN"]["fetched_at"] = time.time() - 1800
    status, error = error_of(api, "AMZN")
    assert status == 503 and error["code"] == "STALE"
    assert error["message"] == ("AMZN price is 30 min old; the last refresh failed: "
                                "cannot reach Yahoo Finance; check the internet connection")


def test_stale_message_without_a_failure(api):
    register(api, "AMZN")
    api.cache.set_ok("AMZN", {"price": 1.0})
    with api.cache._lock:
        api.cache._entries["AMZN"]["fetched_at"] = time.time() - 1800
    assert "no refresh has completed since" in error_of(api, "AMZN")[1]["message"]


def test_missing_field_says_which_source_did_not_supply_it(api):
    register(api, "912810UF3", "bond")
    api.cache.set_ok("912810UF3", {"price": 90.0})
    status, error = error_of(api, "912810UF3", field="ytw")
    assert status == 404 and error["code"] == "NO_FIELD"
    assert error["message"] == "912810UF3 has no ytw; Webull did not supply it"


def test_stopped_refresher_is_reported():
    cache = QuoteCache()
    unstarted = Refresher(cache, interval=300, fetch_many=lambda kinds: {})
    client = TestClient(create_app(cache, unstarted))          # no lifespan: the thread never starts
    assert client.get("/health").json()["status"] == "degraded"
    status, error = error_of(client, "AMZN")
    assert status == 503 and error["code"] == "REFRESH_STOPPED" and "restart" in error["message"]


# --- cache files written before failure codes existed ---------------------------------------

def test_cache_loads_a_file_from_before_failure_codes(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"AMZN": {"kind": "stock", "status": "NOT_FOUND", "fields": {}, '
                    '"fetched_at": null, "failures": 0, "error": "provider has no listing"}}')
    cache = QuoteCache(path)
    assert cache.load() == 1
    assert cache.get("AMZN")["error_code"] is None


@pytest.mark.parametrize("text", ["[]", '{"AMZN": 5}', "null"])
def test_cache_ignores_a_file_of_the_wrong_shape(tmp_path, text):
    path = tmp_path / "c.json"
    path.write_text(text)
    assert QuoteCache(path).load() == 0


# --- paced refresh: a new symbol is not stuck behind the rest ------------------------------

def test_new_symbol_is_fetched_between_chunks(monkeypatch):
    monkeypatch.setattr(refresher_module, "CHUNK_SIZE", 2)
    cache = QuoteCache()
    for n in range(6):
        cache.register(f"S{n}", "stock")
    order = []

    def fetch_many(kinds):
        order.append(sorted(kinds))
        if len(order) == 1:                                   # during the first chunk, a sheet asks for NEW
            cache.register("NEW", "stock")
            refresher.request(["NEW"])
        return {s: {"price": 1.0} for s in kinds}

    refresher = Refresher(cache, interval=300, fetch_many=fetch_many)
    refresher.refresh([f"S{n}" for n in range(6)])
    assert order == [["S0", "S1"], ["NEW"], ["S2", "S3"], ["S4", "S5"]]
    assert cache.get("NEW")["status"] == "OK"


def test_requested_symbol_already_in_the_batch_is_not_fetched_twice(monkeypatch):
    monkeypatch.setattr(refresher_module, "CHUNK_SIZE", 2)
    cache = QuoteCache()
    for n in range(4):
        cache.register(f"S{n}", "stock")
    order = []

    def fetch_many(kinds):
        order.append(sorted(kinds))
        if len(order) == 1:
            refresher.request(["S3"])
        return {s: {"price": 1.0} for s in kinds}

    refresher = Refresher(cache, interval=300, fetch_many=fetch_many)
    refresher.refresh(["S0", "S1", "S2", "S3"])
    assert order == [["S0", "S1"], ["S2", "S3"]]


# --- wiring: pacing settings and /health sources -------------------------------------------

def test_make_client_paces_both_sources():
    client = providers.make_client(request_delay=0.4)
    assert client._host_settings[providers.STOCK_SOURCE] == http_client.HostSettings(0.4, providers.MAX_IN_FLIGHT)
    assert client._host_settings[providers.BOND_SOURCE] == http_client.HostSettings(0.4, providers.MAX_IN_FLIGHT)
    client.close()


def test_health_reports_each_source(monkeypatch, tmp_path):
    monkeypatch.setattr(providers.yfinance, "Ticker",
                        fake_ticker(error=yfinance.exceptions.YFRateLimitError()))
    client = fast_client(block_file=tmp_path / "blocks.json")
    with pytest.raises(HttpClientError):
        providers.fetch_stock("AMZN", client)
    cache = QuoteCache()
    refresher = Refresher(cache, interval=300, fetch_many=lambda kinds: {})
    with TestClient(create_app(cache, refresher, client=client)) as api:
        sources = api.get("/health").json()["sources"]
    assert sources["Yahoo Finance"]["status"] == "limited" and sources["Yahoo Finance"]["refused"] == 1
    assert sources["Yahoo Finance"]["cooldown_until"].endswith("Z")
    assert sources["Webull"]["status"] == "ok"


def test_summary_names_each_source_separately(caplog):
    cache = QuoteCache()
    for n in range(5):
        cache.register(f"S{n}", "stock")
    for cusip in ("912810UF3", "91282CJF9", "91282CMM0", "912810UJ5"):
        cache.register(cusip, "bond")

    def fetch_many(kinds):
        return {s: providers.FetchFailed(providers.NO_NETWORK, "cannot reach " + (
            "Webull" if kind == "bond" else "Yahoo Finance")) for s, kind in kinds.items()}

    refresher = Refresher(cache, interval=300, fetch_many=fetch_many)
    with caplog.at_level("INFO", logger="quote_server"):
        refresher.refresh([s for s, _ in cache.symbols()])
    text = caplog.records[-1].getMessage()
    assert "NO_NETWORK x4: cannot reach Webull" in text and "NO_NETWORK x5: cannot reach Yahoo Finance" in text
