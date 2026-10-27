"""Upstream price lookups: yfinance for stocks, Webull for Treasury CUSIPs."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests
import yfinance

from core import webull_bond_fetcher

# yfinance calls pandas options that pandas has deprecated; the warning repeats on every call.
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")

MAX_WORKERS = 8
REQUEST_TIMEOUT = 10.0

# Errors that mean "this lookup failed" and are retried next cycle.
FETCH_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError,
                OSError, yfinance.exceptions.YFException)


class NotFound(Exception):
    """The provider has no listing for the symbol."""


def fetch_stock(symbol: str) -> dict:
    """Return {"price": float} for a ticker.

    Raises:
        NotFound: yfinance returns no price. An unknown ticker makes yfinance raise
            KeyError('currentTradingPeriod'), which is reported as not found.
    """
    try:
        price = yfinance.Ticker(symbol).fast_info["last_price"]
    except KeyError as exc:
        raise NotFound(symbol) from exc
    if price is None or price != price:
        raise NotFound(symbol)
    return {"price": round(float(price), 4)}


def fetch_bond(cusip: str, today: date | None = None) -> dict:
    """Return {"price", "term", "ytw"} for a Treasury CUSIP.

    `term` is years to maturity. `ytw` is a percent. Either is omitted when Webull
    does not supply it.

    Raises:
        NotFound: Webull has no listing, or no price.
    """
    quote = webull_bond_fetcher.fetch_treasury_price(cusip, timeout=REQUEST_TIMEOUT)
    if not quote or quote.get("price") in (None, ""):
        raise NotFound(cusip)
    fields = {"price": float(quote["price"])}
    if quote.get("ytw") not in (None, ""):
        fields["ytw"] = float(quote["ytw"])
    if quote.get("maturityDate"):
        days = (date.fromisoformat(quote["maturityDate"][:10]) - (today or date.today())).days
        fields["term"] = round(max(days, 0) / 365.25, 2)
    return fields


def fetch_many(kind_by_symbol: dict[str, str], fetch_stock_fn=fetch_stock,
               fetch_bond_fn=fetch_bond) -> dict[str, dict | Exception]:
    """Fetch every symbol concurrently.

    Args:
        kind_by_symbol: symbol -> "stock" or "bond".

    Returns:
        symbol -> fields dict, or the exception raised (NotFound or a FETCH_ERRORS member).
    """
    results: dict[str, dict | Exception] = {}

    def run(item: tuple[str, str]) -> None:
        symbol, kind = item
        fn = fetch_bond_fn if kind == "bond" else fetch_stock_fn
        try:
            results[symbol] = fn(symbol)
        except (NotFound, *FETCH_ERRORS) as exc:
            results[symbol] = exc

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(run, kind_by_symbol.items()))
    return results
