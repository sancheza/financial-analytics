#!/usr/bin/env python3
"""Fetch near-real-time Treasury bond quotes from Webull's public quote page by CUSIP.

Webull server-renders the full quote (last price, bid/ask, yield, coupon) into a
`window.__initState__` JSON blob embedded directly in the page HTML -- no login,
no API key, no JS execution required, just a GET + regex extraction.

This is the active/default bond price source (see bond_market_analyzer.py). It
replaced finra_bond_fetcher.py as the default after two problems surfaced there:
  1. FINRA's TreasuryTradeHistory (the actual TRACE trade prints) is only as fresh
     as the last reported trade, which for thinly-traded off-the-run CUSIPs can be
     months stale. Webull's quote is a live bid/ask/last, timestamped to the second.
  2. FINRA's TreasurySecurities reference table returned a flatly wrong coupon for
     a real CUSIP (912810UF3: FINRA said 5.0%, Treasury's own auction record and
     the bondholder's actual purchase confirmation both say 4.625%). Webull's
     embedded coupon matched Treasury's official figure exactly.
See finra_bond_fetcher.py's module docstring for the full writeup of that failure.
finra_bond_fetcher.py is kept in the codebase but is dormant -- not called by
default -- pending any future decision to re-enable it (e.g. --source finra).
"""

import json
import re
import requests
from datetime import datetime

QUOTE_URL_TEMPLATE = "https://www.webull.com/quote/bond-{cusip}"

# Undocumented endpoint discovered by inspecting network requests fired by the quote
# page's own chart -- same reverse-engineered-with-no-SLA class as the __initState__
# scrape above. Returns real historical price/yield bars (not just the live snapshot)
# for a bond, keyed by Webull's internal tickerId rather than CUSIP. period controls
# both range and bar granularity: "d1"/"d5" return minute bars for the current/last
# few trading days; "m1"/"y1"/"y5" return progressively coarser (daily/weekly) bars
# reaching back to whenever the bond started trading -- "y5" does not mean "exactly
# 5 years back," it means "the longest-range bucket," which for a bond younger than
# 5 years is just its whole trading life (confirmed: a bond issued days earlier
# returns exactly 1 row).
TREND_URL = "https://quotes-gw.webullfintech.com/api/bonds/charts/trend"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

_INIT_STATE_RE = re.compile(r"window\.__initState__\s*=\s*(\{.*?\})\s*;?\s*(?:</script>|window\.)", re.DOTALL)


def _extract_init_state(html: str) -> dict:
    """Pull the server-rendered window.__initState__ JSON blob out of a Webull quote page."""
    match = _INIT_STATE_RE.search(html)
    if not match:
        raise RuntimeError("Could not find window.__initState__ in Webull's page; the page layout may have changed.")
    return json.loads(match.group(1))


def fetch_treasury_price(cusip: str, timeout: float = 10.0) -> dict | None:
    """Look up a Treasury security's live quote by CUSIP.

    Returns a dict with coupon/maturity/issuer reference fields plus live pricing
    (last, bid, ask, yield, YTM/YTW, trade timestamp), or None if Webull has no
    listing for this CUSIP.
    """
    url = QUOTE_URL_TEMPLATE.format(cusip=cusip.lower())
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    state = _extract_init_state(response.text)
    ticker_map = state.get("tickerMap") or {}
    if not ticker_map:
        return None

    ticker = next(iter(ticker_map.values()))
    info = ticker.get("tickerInfo", {})
    rt = ticker.get("tickerRT", {})

    if rt.get("cusip", "").upper() != cusip.upper():
        return None

    ask = (rt.get("askList") or [{}])[0]
    bid = (rt.get("bidList") or [{}])[0]

    coupon_fraction = rt.get("coupon")
    coupon_pct = float(coupon_fraction) * 100 if coupon_fraction is not None else None

    return {
        "cusip": rt.get("cusip"),
        "isin": rt.get("isin"),
        "issueSymbolIdentifier": rt.get("disSymbol") or info.get("disSymbol"),
        "couponRate": f"{coupon_pct:.6f}" if coupon_pct is not None else None,
        "maturityDate": rt.get("expDate"),
        "issuerName": rt.get("issuerName"),
        "couponFrequency": rt.get("couponFreqDesc"),
        "price": rt.get("close"),
        "yield": rt.get("bondYield"),
        "bidPrice": bid.get("price"),
        "bidYield": bid.get("bondYield"),
        "askPrice": ask.get("price"),
        "askYield": ask.get("bondYield"),
        "ytm": rt.get("YTM"),
        "ytw": rt.get("YTW"),
        "tradeTime": rt.get("tradeTime"),
        "status": rt.get("status"),
    }


def fetch_ticker_id(cusip: str, timeout: float = 10.0) -> str | None:
    """Look up the Webull internal tickerId for a CUSIP (the tickerMap key in
    window.__initState__), needed to call TREND_URL. Returns None if Webull has
    no listing for this CUSIP.
    """
    url = QUOTE_URL_TEMPLATE.format(cusip=cusip.lower())
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    state = _extract_init_state(response.text)
    ticker_map = state.get("tickerMap") or {}
    if not ticker_map:
        return None
    return next(iter(ticker_map.keys()))


def fetch_price_history(cusip: str, period: str = "y1", count: int = 800, timeout: float = 10.0) -> list[dict] | None:
    """Fetch a CUSIP's historical price/yield bars from Webull's TREND_URL.

    period: "d1"/"d5" (minute bars, current/last few days), "m1"/"y1" (daily bars,
    the default -- 800 bars covers roughly 3 years), or "y5" (weekly bars, the
    longest-range bucket Webull offers, trading resolution for reach). The default
    favors resolution over maximum lookback since callers charting short ranges
    (e.g. "1M") get no benefit from a weekly bar covering the whole month.

    Returns a list of {date, price, yield} dicts oldest-first, or None if Webull has
    no listing for this CUSIP or returns no bars (e.g. this CUSIP hasn't traded yet).
    """
    ticker_id = fetch_ticker_id(cusip, timeout)
    if not ticker_id:
        return None

    response = requests.get(
        TREND_URL,
        params={"tickerIds": ticker_id, "period": period, "count": count},
        headers=REQUEST_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None

    row = payload[0]
    price_bars = row.get("data") or []
    yield_bars = row.get("yieldData") or []
    # Each bar is "timestamp,close,open,high,low,prevClose,volume,...". Close is what
    # Webull's own Price-mode chart plots, so that's what's used here.
    yields_by_ts = {}
    for bar in yield_bars:
        fields = bar.split(",")
        yields_by_ts[fields[0]] = float(fields[1])

    history = []
    for bar in price_bars:
        fields = bar.split(",")
        ts, close = fields[0], fields[1]
        history.append({
            "date": datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d"),
            "price": float(close),
            "yield": yields_by_ts.get(ts),
        })
    # Webull returns bars newest-first; flip to oldest-first for charting.
    history.reverse()
    return history


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <CUSIP>")
        sys.exit(1)

    result = fetch_treasury_price(sys.argv[1].strip().upper())
    print(json.dumps(result, indent=2) if result else "No record found.")
