"""Upstream price lookups: yfinance for stocks, Webull for Treasury CUSIPs.

A lookup ends in one of three ways:
  a dict of fields   the source answered with a price
  NotFound           the source answered that it has no such security
  FetchFailed        the source could not be asked or did not give a usable answer

Only an answer from the source is NotFound. A network, timeout, rate-limit, or page
problem is always FetchFailed, so a failed lookup never replaces a known price with
"not found".

Every request to a source goes through one core.http_client.HttpClient, which paces the
requests, retries a failure that may pass, skips a source that keeps failing, and stops
asking a source that refused us (README section 8.1).

Log events: logger `quote_server.providers`, scope `quote_server`, action `price_fetch`.
"""

import functools
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import yfinance

from core import http_client, webull_bond_fetcher
from core.http_client import HostSettings, HttpClient, HttpClientError

log = logging.getLogger("quote_server.providers")

SCOPE = "quote_server"
ACTION_PRICE_FETCH = "price_fetch"

# yfinance calls pandas options that pandas has deprecated; the warning repeats on every call.
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")
# By default yfinance hides network and HTTP errors and returns an empty result, which is
# indistinguishable from an unknown ticker. Raising them lets fetch_stock tell them apart.
yfinance.config.debug.hide_exceptions = False
# yfinance logs an ERROR line for every unknown ticker on every cycle; this module logs
# the outcome of each lookup itself.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

MAX_WORKERS = 8
REQUEST_TIMEOUT = 10.0

# Spacing between the starts of two requests to one source, and requests in flight at once.
# Sequential runs of 100 Yahoo Finance requests and 48 Webull requests at 1.0, 0.5, 0.25 and
# 0.1 s spacing (2026-10-30) drew no refusal; 0.25 s is the spacing used.
REQUEST_DELAY_SECONDS = 0.25
MAX_IN_FLIGHT = 2

STOCK_SOURCE = "Yahoo Finance"
BOND_SOURCE = "Webull"
SOURCE_BY_KIND = {"stock": STOCK_SOURCE, "bond": BOND_SOURCE}

# Failure codes. They reach the spreadsheet as `#ERR: <code>`.
NO_NETWORK = "NO_NETWORK"
SOURCE_TIMEOUT = "SOURCE_TIMEOUT"
RATE_LIMITED = "RATE_LIMITED"
SOURCE_REFUSED = "SOURCE_REFUSED"
SOURCE_DOWN = "SOURCE_DOWN"
SOURCE_ERROR = "SOURCE_ERROR"
SOURCE_CHANGED = "SOURCE_CHANGED"
NO_DATA = "NO_DATA"
LOCAL_CACHE = "LOCAL_CACHE"
INTERNAL = "INTERNAL"
NO_DIVIDEND_DATA = "NO_DIVIDEND_DATA"

LOCAL_KIND = "local_cache"   # an http_client error kind of this module's own

_CODE_BY_KIND = {
    http_client.NETWORK: NO_NETWORK,
    http_client.TIMEOUT: SOURCE_TIMEOUT,
    http_client.RATE_LIMITED: RATE_LIMITED,
    http_client.REFUSED: SOURCE_REFUSED,
    http_client.SERVER_ERROR: SOURCE_DOWN,
    http_client.HTTP_STATUS: SOURCE_ERROR,
    LOCAL_KIND: LOCAL_CACHE,
}


class NotFound(Exception):
    """The source answered that it has no listing for the symbol."""

    def __init__(self, symbol: str, source: str = "the price source") -> None:
        super().__init__(symbol)
        self.symbol = symbol
        self.source = source
        self.message = f"{source} has no listing for {symbol}; check the symbol"


class Fields(dict):
    """Fields obtained for a symbol, plus errors for optional fields that were asked for and failed.

    Attributes:
        errors: field -> FetchFailed.
    """

    def __init__(self, *args, errors: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.errors = dict(errors or {})


class FetchFailed(Exception):
    """A lookup failed for a reason other than the source having no listing.

    Attributes:
        code: Short code shown in the spreadsheet (NO_NETWORK, RATE_LIMITED, ...).
        message: One sentence in plain words naming the cause. No exception text.
        retryable: True when asking again soon may succeed.
    """

    def __init__(self, code: str, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def make_client(block_file: Path | None = None, request_delay: float = REQUEST_DELAY_SECONDS,
                **client_options) -> HttpClient:
    """Build the HTTP client both sources are called through.

    Args:
        block_file: Where refusals are kept across restarts; None keeps them in memory.
        request_delay: Seconds between the starts of two requests to one source.
        **client_options: Passed to HttpClient (tests set `sleep`, `retry_delays`, ...).
    """
    settings = HostSettings(min_delay=request_delay, max_concurrent=MAX_IN_FLIGHT)
    return HttpClient(timeout=REQUEST_TIMEOUT, block_file=block_file,
                      host_settings={STOCK_SOURCE: settings, BOND_SOURCE: settings}, **client_options)


def classify_error(exc: BaseException, host: str) -> HttpClientError | None:
    """Classify an error from yfinance or Webull; None when it is not about the host."""
    if isinstance(exc, yfinance.exceptions.YFRateLimitError):
        return HttpClientError(http_client.RATE_LIMITED, f"{host} is limiting requests", host=host,
                               retryable=False, blocks_host=True)
    if type(exc).__module__.split(".")[0] in ("sqlite3", "peewee"):
        return HttpClientError(LOCAL_KIND, "yfinance could not open its local cache file", host=host)
    return http_client.classify_exception(exc, host)


def _local_time(iso: str) -> str:
    stamp = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return stamp.astimezone().strftime("%H:%M")


def _failure_from(error: HttpClientError, source: str) -> FetchFailed:
    """Turn an HttpClientError into the plain-words FetchFailed shown in the spreadsheet."""
    until = f"; not asking again until {_local_time(error.cooldown_until)}" if error.cooldown_until else ""
    status = f" (HTTP {error.status})" if error.status else ""
    messages = {
        http_client.NETWORK: f"cannot reach {source}; check the internet connection",
        http_client.TIMEOUT: f"{source} did not answer in time",
        http_client.RATE_LIMITED: f"{source} is limiting requests{until or '; trying again later'}",
        http_client.REFUSED: f"{source} refused the request{status}{until}",
        http_client.SERVER_ERROR: f"{source} is having problems{status}",
        http_client.HTTP_STATUS: f"{source} answered HTTP {error.status}",
        LOCAL_KIND: "yfinance could not open its local cache file (~/Library/Caches/py-yfinance)",
    }
    code = _CODE_BY_KIND.get(error.kind, INTERNAL)
    return FetchFailed(code, messages.get(error.kind, error.message), retryable=error.retryable)


def as_failure(exc: BaseException, source: str) -> FetchFailed:
    """Describe any lookup error as a FetchFailed with a plain-words message.

    Args:
        exc: The error a lookup raised.
        source: Name of the source that was asked ("Yahoo Finance", "Webull").
    """
    if isinstance(exc, FetchFailed):
        return exc
    error = exc if isinstance(exc, HttpClientError) else classify_error(exc, source)
    if error is not None:
        return _failure_from(error, source)
    return FetchFailed(INTERNAL, f"unexpected {type(exc).__name__} in the quote server; "
                                 "see the server log", retryable=False)


def dividend_yield_from(info: dict, price: float, symbol: str) -> float:
    """Return the dividend yield in percent from Yahoo's `info` values, or raise.

    A yield of 0 is returned only when Yahoo reports both a 0 yield and 0 dividends over
    the last 12 months. A missing value, or values that disagree, are not a "no dividend"
    answer. A positive yield is checked against dividendRate / price, so a change of
    units in yfinance cannot give a result off by a factor of 100.

    Raises:
        FetchFailed: NO_DIVIDEND_DATA when Yahoo gives no usable yield or does not confirm
            that there is no dividend; SOURCE_CHANGED when the value disagrees with the rate.
    """
    value, rate, trailing = (info.get(key) for key in ("dividendYield", "dividendRate",
                                                       "trailingAnnualDividendRate"))

    def number(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x

    unconfirmed = f"{STOCK_SOURCE} does not confirm whether {symbol} pays a dividend"
    if not number(value) or value < 0:
        raise FetchFailed(NO_DIVIDEND_DATA, f"{STOCK_SOURCE} gives no dividend yield for {symbol}; "
                          "it is unknown whether it pays one", retryable=False)
    if value == 0:
        if number(trailing) and trailing == 0 and (rate is None or rate == 0):
            return 0.0
        raise FetchFailed(NO_DIVIDEND_DATA, f"{unconfirmed}: it reports a 0% yield "
                          f"but {'dividends in the last 12 months' if number(trailing) and trailing > 0 else 'no 12-month total'}",
                          retryable=False)
    if number(rate) and rate > 0 and price > 0:
        expected = rate / price * 100
        if not expected / 3 <= value <= expected * 3:
            raise FetchFailed(SOURCE_CHANGED, f"{STOCK_SOURCE}'s dividend yield for {symbol} ({value:g}) "
                              f"disagrees with its dividend rate ({expected:.2f}%); the units may have changed",
                              retryable=False)
    return round(float(value), 4)


def fetch_dividend_yield(symbol: str, price: float, client: HttpClient) -> float:
    """Return the dividend yield of a stock in percent (see dividend_yield_from).

    Raises:
        FetchFailed: the yield is unknown or not confirmed.
        HttpClientError: the request failed, or the source is stopped (see as_failure).
    """
    def lookup() -> dict:
        info = yfinance.Ticker(symbol).info
        return {key: info.get(key) for key in ("dividendYield", "dividendRate",
                                               "trailingAnnualDividendRate")}

    return dividend_yield_from(client.call(STOCK_SOURCE, lookup, classify=classify_error), price, symbol)


def fetch_stock(symbol: str, client: HttpClient, wants: tuple[str, ...] = ()) -> dict:
    """Return {"price": float} for a ticker: the latest close from the last 5 days.

    Args:
        symbol: The ticker.
        client: The HTTP client.
        wants: Optional fields to fetch as well; only "dividend_yield" exists. When one of
            them cannot be obtained the price is still returned, in a Fields whose
            `errors` names the field and the cause.

    Raises:
        NotFound: Yahoo Finance answered that the ticker does not exist.
        FetchFailed: no price was obtained for any other reason. This includes a
            response with no price rows, which is not proof the ticker is unknown.
        HttpClientError: the request failed, or the source is stopped (see as_failure).
    """
    def lookup() -> dict:
        try:
            history = yfinance.Ticker(symbol).history(period="5d", auto_adjust=False,
                                                      timeout=REQUEST_TIMEOUT)
        except yfinance.exceptions.YFTickerMissingError as exc:
            raise NotFound(symbol, STOCK_SOURCE) from exc
        except OSError as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                raise NotFound(symbol, STOCK_SOURCE) from exc
            raise
        closes = history["Close"].dropna()
        if closes.empty:
            raise FetchFailed(NO_DATA, f"{STOCK_SOURCE} returned no prices for {symbol}")
        return {"price": round(float(closes.iloc[-1]), 4)}

    fields = client.call(STOCK_SOURCE, lookup, classify=classify_error)
    if "dividend_yield" not in wants:
        return fields
    result = Fields(fields)
    try:
        result["dividend_yield"] = fetch_dividend_yield(symbol, fields["price"], client)
    except Exception as exc:  # isolation point: a failed optional field must not lose the price
        failure = as_failure(exc, STOCK_SOURCE)
        if failure.code == INTERNAL:
            log.error("unexpected error fetching the dividend yield of %s", symbol, exc_info=True,
                      extra={"scope": SCOPE, "action": ACTION_PRICE_FETCH, "outcome": "failure",
                             "actor": "system", "target": symbol})
        result.errors["dividend_yield"] = failure
    return result


def fetch_bond(cusip: str, client: HttpClient, today: date | None = None) -> dict:
    """Return {"price", "term", "ytw"} for a Treasury CUSIP.

    `term` is years to maturity. `ytw` is a percent. Either is omitted when Webull
    does not supply it.

    Raises:
        NotFound: Webull has no listing, or no price.
        FetchFailed: Webull's page could not be read (its layout changed).
        HttpClientError: the request failed, or the source is stopped (see as_failure).
    """
    try:
        quote = webull_bond_fetcher.fetch_treasury_price(
            cusip, timeout=REQUEST_TIMEOUT, get=functools.partial(client.get, host=BOND_SOURCE))
    except HttpClientError as exc:
        if exc.status == 404:
            raise NotFound(cusip, BOND_SOURCE) from exc
        raise
    except (RuntimeError, ValueError) as exc:
        raise FetchFailed(SOURCE_CHANGED, f"{BOND_SOURCE}'s page could not be read; "
                          "its layout may have changed", retryable=False) from exc
    if not quote or quote.get("price") in (None, ""):
        raise NotFound(cusip, BOND_SOURCE)
    try:
        fields = {"price": float(quote["price"])}
        if quote.get("ytw") not in (None, ""):
            fields["ytw"] = float(quote["ytw"])
        if quote.get("maturityDate"):
            days = (date.fromisoformat(quote["maturityDate"][:10]) - (today or date.today())).days
            fields["term"] = round(max(days, 0) / 365.25, 2)
    except (ValueError, TypeError) as exc:
        raise FetchFailed(SOURCE_CHANGED, f"{BOND_SOURCE} sent a price that could not be "
                          "read", retryable=False) from exc
    return fields


def fetch_many(kind_by_symbol: dict[str, str], fetch_stock_fn=None, fetch_bond_fn=None,
               client: HttpClient | None = None,
               extras: dict[str, list[str]] | None = None) -> dict[str, dict | NotFound | FetchFailed]:
    """Fetch every symbol concurrently.

    Args:
        kind_by_symbol: symbol -> "stock" or "bond".
        extras: symbol -> optional fields to fetch as well (see fetch_stock). A lookup
            function that is given is called `fn(symbol, wants=(...))` for those symbols.
        fetch_stock_fn: Lookup for a stock ticker, `fn(symbol)`. Default fetch_stock on `client`.
        fetch_bond_fn: Lookup for a CUSIP, `fn(cusip)`. Default fetch_bond on `client`.
        client: The HTTP client for the default lookups. Default: a new client for this call.

    Returns:
        symbol -> fields dict, NotFound, or FetchFailed.
    """
    own_client = client is None and (fetch_stock_fn is None or fetch_bond_fn is None)
    if own_client:
        client = make_client()
    stock_fn = fetch_stock_fn or functools.partial(fetch_stock, client=client)
    bond_fn = fetch_bond_fn or functools.partial(fetch_bond, client=client)
    results: dict[str, dict | NotFound | FetchFailed] = {}

    def run(item: tuple[str, str]) -> None:
        symbol, kind = item
        fetch = bond_fn if kind == "bond" else stock_fn
        wants = tuple((extras or {}).get(symbol, ())) if kind == "stock" else ()
        try:
            results[symbol] = fetch(symbol, wants=wants) if wants else fetch(symbol)
        except (NotFound, FetchFailed) as outcome:
            results[symbol] = outcome
        except Exception as exc:  # isolation point: one symbol's error must not stop the batch
            failure = as_failure(exc, SOURCE_BY_KIND[kind])
            if failure.code == INTERNAL:
                log.error("unexpected error fetching %s", symbol, exc_info=True,
                          extra={"scope": SCOPE, "action": ACTION_PRICE_FETCH, "outcome": "failure",
                                 "actor": "system", "target": symbol})
            results[symbol] = failure

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(run, kind_by_symbol.items()))
    finally:
        if own_client:
            client.close()
    return results
