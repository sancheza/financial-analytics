"""Tests for the yield field: yield to worst for a bond, dividend yield for a stock."""

import time

import pandas
import pytest
import requests
from fastapi.testclient import TestClient

from quote_server import providers
from quote_server.app import create_app
from quote_server.cache import QuoteCache
from quote_server.refresher import Refresher


def info(**values):
    base = {"dividendYield": None, "dividendRate": None, "trailingAnnualDividendRate": None}
    return {**base, **values}


# --- dividend_yield_from: when is 0 confirmed ---------------------------------------------------

@pytest.mark.parametrize("values,price,expected", [
    (info(dividendYield=6.48, dividendRate=2.83, trailingAnnualDividendRate=2.81), 44.74, 6.48),
    (info(dividendYield=5.0), 81.59, 5.0),                       # funds report no dividendRate: accepted
    (info(dividendYield=0.31, dividendRate=1.08), 349.0, 0.31),
    (info(dividendYield=0.0, dividendRate=None, trailingAnnualDividendRate=0.0), 238.0, 0.0),
    (info(dividendYield=0, dividendRate=0, trailingAnnualDividendRate=0), 238.0, 0.0),
])
def test_dividend_yield_values(values, price, expected):
    assert providers.dividend_yield_from(values, price, "X") == expected


@pytest.mark.parametrize("values,words", [
    (info(), "unknown whether it pays one"),                                     # no value at all
    (info(dividendYield=float("nan")), "unknown whether it pays one"),
    (info(dividendYield=-1.0), "unknown whether it pays one"),
    (info(dividendYield="6.4"), "unknown whether it pays one"),
    (info(dividendYield=True), "unknown whether it pays one"),
    (info(dividendYield=0.0), "does not confirm"),                               # 0, but no 12-month total
    (info(dividendYield=0.0, trailingAnnualDividendRate=2.0), "dividends in the last 12 months"),
    (info(dividendYield=0.0, dividendRate=1.0, trailingAnnualDividendRate=0.0), "does not confirm"),
])
def test_dividend_yield_not_confirmed(values, words):
    with pytest.raises(providers.FetchFailed) as caught:
        providers.dividend_yield_from(values, 100.0, "X")
    assert caught.value.code == providers.NO_DIVIDEND_DATA and not caught.value.retryable
    assert words in caught.value.message


@pytest.mark.parametrize("value", [0.0648, 648.0])   # a fraction, or a number 100 times too large
def test_a_change_of_units_is_caught(value):
    with pytest.raises(providers.FetchFailed) as caught:
        providers.dividend_yield_from(info(dividendYield=value, dividendRate=2.83), 44.74, "VZ")
    assert caught.value.code == providers.SOURCE_CHANGED and "units" in caught.value.message


# --- fetch_stock with the optional field -----------------------------------------------------

class FakeTicker:
    """Ticker with a price history and an `info` that is a dict or raises."""

    info_calls = 0

    def __init__(self, symbol, *, close=100.0, info_value=None):
        self.symbol, self._close, self._info = symbol, close, info_value

    def history(self, **_):
        return pandas.DataFrame({"Close": [self._close]})

    @property
    def info(self):
        type(self).info_calls += 1
        if isinstance(self._info, BaseException):
            raise self._info
        return self._info


def patch_ticker(monkeypatch, info_value, close=100.0):
    FakeTicker.info_calls = 0
    monkeypatch.setattr(providers.yfinance, "Ticker", lambda symbol: FakeTicker(symbol, close=close, info_value=info_value))


def fast_client():
    return providers.make_client(request_delay=0, retry_delays=(0.0, 0.0), sleep=lambda _: None)


def test_without_the_field_no_info_request_is_made(monkeypatch):
    patch_ticker(monkeypatch, info(dividendYield=5.0))
    assert providers.fetch_stock("X", fast_client()) == {"price": 100.0}
    assert FakeTicker.info_calls == 0


def test_dividend_yield_is_fetched_when_wanted(monkeypatch):
    patch_ticker(monkeypatch, info(dividendYield=5.0, dividendRate=5.0))
    result = providers.fetch_stock("X", fast_client(), wants=("dividend_yield",))
    assert result == {"price": 100.0, "dividend_yield": 5.0} and result.errors == {}


def test_a_confirmed_zero_is_a_value(monkeypatch):
    patch_ticker(monkeypatch, info(dividendYield=0.0, trailingAnnualDividendRate=0.0))
    assert providers.fetch_stock("X", fast_client(), wants=("dividend_yield",))["dividend_yield"] == 0.0


def test_unconfirmed_yield_keeps_the_price_and_names_the_cause(monkeypatch):
    patch_ticker(monkeypatch, info())
    result = providers.fetch_stock("X", fast_client(), wants=("dividend_yield",))
    assert result == {"price": 100.0} and result.errors["dividend_yield"].code == providers.NO_DIVIDEND_DATA


def test_network_failure_on_the_yield_request_is_retryable_and_keeps_the_price(monkeypatch):
    patch_ticker(monkeypatch, requests.ConnectionError("down"))
    result = providers.fetch_stock("X", fast_client(), wants=("dividend_yield",))
    failure = result.errors["dividend_yield"]
    assert result == {"price": 100.0} and failure.code == providers.NO_NETWORK and failure.retryable
    assert FakeTicker.info_calls == 3                                      # retried like any request


def test_unexpected_error_on_the_yield_request_keeps_the_price(monkeypatch):
    patch_ticker(monkeypatch, KeyError("bug"))
    result = providers.fetch_stock("X", fast_client(), wants=("dividend_yield",))
    assert result == {"price": 100.0} and result.errors["dividend_yield"].code == providers.INTERNAL


def test_fetch_many_passes_wants_only_for_symbols_that_asked():
    calls = []

    def stock(symbol, wants=()):
        calls.append((symbol, wants))
        return {"price": 1.0}

    providers.fetch_many({"A": "stock", "B": "stock", "912810UF3": "bond"}, fetch_stock_fn=stock,
                         fetch_bond_fn=lambda c: {"price": 2.0}, extras={"A": ["dividend_yield"]})
    assert sorted(calls) == [("A", ("dividend_yield",)), ("B", ())]


# --- cache and refresher ---------------------------------------------------------------------

def test_cache_remembers_wanted_fields_across_a_restart(tmp_path):
    path = tmp_path / "c.json"
    cache = QuoteCache(path)
    cache.register("VZ", "stock")
    assert cache.want("VZ", "dividend_yield") is True and cache.want("VZ", "dividend_yield") is False
    cache.save()
    reloaded = QuoteCache(path)
    reloaded.load()
    assert reloaded.wanted() == {"VZ": ["dividend_yield"]}


def test_cache_file_from_before_optional_fields_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"VZ": {"kind": "stock", "status": "OK", "fields": {"price": 1.0}, "fetched_at": 1.0, '
                    '"failures": 0, "error": null}}')
    cache = QuoteCache(path)
    assert cache.load() == 1
    entry = cache.get("VZ")
    assert entry["wants"] == [] and entry["field_errors"] == {} and cache.wanted() == {}


def test_refresher_stores_field_errors_logs_them_once_and_retries_a_temporary_one(caplog):
    cache = QuoteCache()
    cache.register("VUSXX", "stock")
    cache.register("VZ", "stock")
    cache.want("VUSXX", "dividend_yield")
    cache.want("VZ", "dividend_yield")
    seen = []

    def fetch_many(kinds, extras=None):
        seen.append(extras)
        return {
            "VUSXX": providers.Fields({"price": 1.0}, errors={"dividend_yield": providers.FetchFailed(
                providers.NO_DIVIDEND_DATA, "unknown", retryable=False)}),
            "VZ": providers.Fields({"price": 44.0}, errors={"dividend_yield": providers.FetchFailed(
                providers.NO_NETWORK, "cannot reach Yahoo Finance", retryable=True)}),
        }

    refresher = Refresher(cache, interval=300, fetch_many=fetch_many)
    with caplog.at_level("INFO", logger="quote_server"):
        retry = refresher.refresh(["VUSXX", "VZ"])
        refresher.refresh(["VUSXX", "VZ"])
    assert seen[0] == {"VUSXX": ["dividend_yield"], "VZ": ["dividend_yield"]}
    assert retry == ["VZ"]                                                      # only the temporary one
    assert cache.get("VUSXX")["field_errors"]["dividend_yield"] == {
        "code": providers.NO_DIVIDEND_DATA, "message": "unknown", "retryable": False}
    assert cache.get("VUSXX")["status"] == "OK" and cache.get("VUSXX")["fields"] == {"price": 1.0}
    notices = [r for r in caplog.records if "dividend_yield not obtained" in r.getMessage()]
    assert len(notices) == 2                                                    # once per symbol, not per cycle


# --- API -------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    cache = QuoteCache(tmp_path / "c.json")

    def fetch_many(kinds, extras=None):
        out = {}
        for symbol, kind in kinds.items():
            wants = (extras or {}).get(symbol, [])
            if kind == "bond":
                out[symbol] = {"price": 99.5, "ytw": 4.5, "term": 10.0}
            elif symbol == "NODIV":
                out[symbol] = providers.Fields({"price": 50.0, "dividend_yield": 0.0})
            elif symbol == "UNKNOWN":
                out[symbol] = providers.Fields({"price": 1.0}, errors={"dividend_yield": providers.FetchFailed(
                    providers.NO_DIVIDEND_DATA, "Yahoo Finance gives no dividend yield for UNKNOWN; "
                    "it is unknown whether it pays one", retryable=False)})
            elif symbol == "FLAKY":
                out[symbol] = providers.Fields({"price": 1.0}, errors={"dividend_yield": providers.FetchFailed(
                    providers.NO_NETWORK, "cannot reach Yahoo Finance; check the internet connection")})
            else:
                fields = providers.Fields({"price": 123.45})
                if "dividend_yield" in wants:
                    fields["dividend_yield"] = 4.2
                out[symbol] = fields
        return out

    refresher = Refresher(cache, interval=300, fetch_many=fetch_many)
    with TestClient(create_app(cache, refresher)) as client:
        client.cache = cache
        yield client


def get(api, symbol, field):
    return api.get("/quote", params={"symbol": symbol, "field": field})


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_yield_of_a_bond_is_yield_to_worst(api):
    get(api, "912810UF3", "yield")
    assert wait_for(lambda: get(api, "912810UF3", "yield").status_code == 200)
    body = get(api, "912810UF3", "yield").json()
    assert body["value"] == 4.5 and body["field"] == "ytw"
    assert get(api, "912810UF3", "ytw").json()["value"] == 4.5


def test_yield_of_a_stock_is_its_dividend_yield(api):
    assert get(api, "VZ", "yield").status_code == 202
    assert wait_for(lambda: get(api, "VZ", "yield").status_code == 200)
    body = get(api, "VZ", "yield").json()
    assert body["value"] == 4.2 and body["field"] == "dividend_yield" and body["type"] == "stock"
    assert get(api, "VZ", "dividend_yield").json()["value"] == 4.2
    assert get(api, "VZ", "price").json()["value"] == 123.45


def test_asking_for_the_yield_of_a_stock_already_cached_fetches_it(api):
    get(api, "VZ", "price")
    assert wait_for(lambda: get(api, "VZ", "price").status_code == 200)
    first = get(api, "VZ", "yield")
    assert first.status_code == 202 and first.json()["error"]["code"] == "PENDING"
    assert "dividend yield is being fetched" in first.json()["error"]["message"]
    assert wait_for(lambda: get(api, "VZ", "yield").status_code == 200)


def test_a_stock_without_a_dividend_shows_zero(api):
    get(api, "NODIV", "yield")
    assert wait_for(lambda: get(api, "NODIV", "yield").status_code == 200)
    assert get(api, "NODIV", "yield").json()["value"] == 0.0


def test_unconfirmed_dividend_is_an_error_that_says_why(api):
    get(api, "UNKNOWN", "yield")
    assert wait_for(lambda: get(api, "UNKNOWN", "yield").status_code != 202)
    response = get(api, "UNKNOWN", "yield")
    error = response.json()["error"]
    assert response.status_code == 404 and error["code"] == "NO_DIVIDEND_DATA"
    assert "unknown whether it pays one" in error["message"]
    assert get(api, "UNKNOWN", "price").json()["value"] == 1.0                 # the price is unaffected


def test_a_temporary_failure_of_the_yield_is_a_503(api):
    get(api, "FLAKY", "yield")
    assert wait_for(lambda: get(api, "FLAKY", "yield").status_code != 202)
    response = get(api, "FLAKY", "yield")
    assert response.status_code == 503 and response.json()["error"]["code"] == "NO_NETWORK"


def test_fields_that_do_not_fit_the_symbol_type_are_rejected(api):
    stock = get(api, "VZ", "ytw")
    assert stock.status_code == 400 and stock.json()["error"]["code"] == "BAD_FIELD"
    assert "dividend_yield" in stock.json()["error"]["message"] and "yield" in stock.json()["error"]["message"]
    bond = get(api, "912810UF3", "dividend_yield")
    assert bond.status_code == 400 and bond.json()["error"]["code"] == "BAD_FIELD"
    assert "ytw" in bond.json()["error"]["message"]


def test_a_bond_never_asks_for_optional_fields(api):
    get(api, "912810UF3", "yield")
    assert api.cache.wanted() == {}
