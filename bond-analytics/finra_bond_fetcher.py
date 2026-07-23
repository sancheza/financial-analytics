#!/usr/bin/env python3
"""Fetch Treasury security reference and last-trade data from FINRA's public Fixed Income data service.

STATUS: DORMANT. This was the original data source for bond_market_analyzer.py's
default fetch mode and has been superseded by webull_bond_fetcher.py. It is kept
in the codebase (reachable via `--source finra`) but is NOT the default, because
it has two confirmed reliability problems:

1. UNRELIABLE REFERENCE DATA: the TreasurySecurities dataset's couponRate field
   returned 5.0% for CUSIP 912810UF3. The actual coupon, per Treasury's own
   auction record (api.fiscaldata.treasury.gov auctions_query, int_rate field)
   and the bondholder's real purchase confirmation, is 4.625%. This is not a
   yield/coupon mixup -- couponRate was queried specifically. There is no reason
   to trust this field for any other CUSIP either.
2. STALE PRICING: TreasuryTradeHistory only reflects the last TRACE-reported
   trade. For thinly-traded, off-the-run Treasuries that can be many months old
   (observed: a print from 2025-02-19 returned as the "latest" trade on
   2026-07-23) even though it's the correct field to query -- it just isn't
   real-time by nature, since it depends on someone having actually traded the
   CUSIP recently.

TreasuryTradeHistory itself (the actual TRACE trade-reporting pipeline, legally
mandated under FINRA Rule 6730) is a different, more trustworthy data pipeline
than the TreasurySecurities reference table -- the coupon bug does not by itself
prove the trade prints are wrong -- but staleness alone makes it a poor fit for
near-real-time pricing, which is why Webull (server-rendered live bid/ask/last)
is now the active source.

This hits the same undocumented endpoint the free CUSIP lookup widget at
https://www.finra.org/finra-data/fixed-income uses. No login or API key is required,
just a CSRF-style handshake: GET the endpoint once to receive an XSRF-TOKEN cookie,
then echo it back as the X-XSRF-TOKEN header on the POST.

Two datasets under the same group are used:
  - TreasurySecurities: coupon, maturity, issuer, benchmark term (security reference data)
  - TreasuryTradeHistory: individual TRACE-reported trades (this is where actual
    last-trade price/yield lives; TreasurySecurities' own lastSalePrice field is
    reliably null on this endpoint)
"""

import json
import requests

BASE_URL = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name"
SECURITIES_URL = f"{BASE_URL}/TreasurySecurities"
TRADE_HISTORY_URL = f"{BASE_URL}/TreasuryTradeHistory"

# FINRA's TreasurySecurities.couponRate has been observed wrong for at least one real
# CUSIP (912810UF3: FINRA reports 5.0%, Treasury's own auction record and the bondholder's
# actual purchase confirmation both say 4.625%). Treasury's auctions_query is the issuer
# of record for these securities, so it's used to verify/override FINRA's coupon whenever
# it has a record for the CUSIP.
TREASURY_AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

SECURITY_FIELDS = [
    "cusip",
    "issueSymbolIdentifier",
    "couponRate",
    "maturityDate",
    "issuerName",
    "benchmarkTermCode",
]

TRADE_FIELDS = [
    "cusip",
    "tradeDate",
    "tradeTime",
    "lastSalePrice",
    "lastSaleYield",
    "reportedTradeVolume",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _cusip_filter(cusip: str) -> dict:
    """Build the orFilters/compareFilters clause this API expects to match a CUSIP by equality."""
    return {"orFilters": [{"compareFilters": [{"fieldName": "cusip", "fieldValue": cusip, "compareType": "EQUAL"}]}]}


def _post(session: requests.Session, url: str, payload: dict, timeout: float) -> dict:
    """POST a query payload with the session's XSRF-TOKEN echoed back, and return the decoded record list."""
    token = session.cookies.get("XSRF-TOKEN")
    if not token:
        raise RuntimeError("Could not obtain FINRA XSRF-TOKEN cookie; the endpoint may have changed.")

    headers = dict(REQUEST_HEADERS)
    headers["X-XSRF-TOKEN"] = token

    response = session.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.json()

    if body.get("status") != "success":
        raise RuntimeError(f"FINRA API returned an error: {body.get('statusMessage')}")

    return json.loads(body["returnBody"]["data"])


def fetch_official_coupon(cusip: str, timeout: float = 10.0) -> float | None:
    """Look up the coupon rate straight from Treasury's own auction record.

    Returns None if Treasury has no auction record for this CUSIP (e.g. it isn't
    a fixed-rate coupon security, or the fiscaldata API has a gap) rather than
    raising, since this is a cross-check, not the primary lookup.
    """
    params = {"filter": f"cusip:eq:{cusip}", "fields": "cusip,int_rate", "page[size]": 1}
    response = requests.get(TREASURY_AUCTIONS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows or not rows[0].get("int_rate"):
        return None
    return float(rows[0]["int_rate"])


def fetch_treasury_price(cusip: str, timeout: float = 10.0) -> dict | None:
    """Look up a Treasury security's reference data and most recent trade by CUSIP.

    Returns a dict merging TreasurySecurities fields (coupon, maturity, issuer,
    benchmarkTerm) with the latest TreasuryTradeHistory print (price, yield, tradeDate,
    tradeTime), or None if the CUSIP isn't found in the security reference dataset.

    price/yield/tradeDate/tradeTime will be None when FINRA has no TRACE-reported
    trade for this CUSIP -- common for thinly-traded, off-the-run Treasuries. That's
    a real absence of trading activity, not a fetch failure.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": REQUEST_HEADERS["User-Agent"]})
    session.get(SECURITIES_URL, timeout=timeout)  # establishes XSRF-TOKEN cookie

    security_records = _post(
        session,
        SECURITIES_URL,
        {"fields": SECURITY_FIELDS, **_cusip_filter(cusip), "offset": 0, "limit": 1},
        timeout,
    )
    if not security_records:
        return None
    record = security_records[0]

    record["couponRateFinra"] = record.get("couponRate")
    try:
        official_coupon = fetch_official_coupon(cusip, timeout)
    except requests.RequestException:
        official_coupon = None
    if official_coupon is not None:
        record["couponRate"] = f"{official_coupon:.10f}"
        record["couponSource"] = "treasury"
    else:
        record["couponSource"] = "finra"

    trade_records = _post(
        session,
        TRADE_HISTORY_URL,
        {
            "fields": TRADE_FIELDS,
            **_cusip_filter(cusip),
            "sortFields": ["-tradeDate", "-tradeTime"],
            "offset": 0,
            "limit": 1,
        },
        timeout,
    )

    if trade_records:
        trade = trade_records[0]
        record["price"] = trade.get("lastSalePrice")
        record["yield"] = trade.get("lastSaleYield")
        record["tradeDate"] = trade.get("tradeDate")
        record["tradeTime"] = trade.get("tradeTime")
        record["tradeVolume"] = trade.get("reportedTradeVolume")
    else:
        record["price"] = None
        record["yield"] = None
        record["tradeDate"] = None
        record["tradeTime"] = None
        record["tradeVolume"] = None

    return record


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <CUSIP>")
        sys.exit(1)

    result = fetch_treasury_price(sys.argv[1].strip().upper())
    print(json.dumps(result, indent=2) if result else "No record found.")
