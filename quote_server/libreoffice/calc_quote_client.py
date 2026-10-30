"""Calc side of the quote server: GETQUOTE lookups and the Refresh Prices command.

LibreOffice loads this file from its profile's Scripts/python/ (a link to this file,
made by install_calc_integration.py). It uses only the standard library, because
LibreOffice runs it with its own bundled Python.

Entry points, called by LibreOffice through script URLs:
  getquote          Called by the GETQUOTE() cell function (QuoteServer.bas).
  refresh_prices    Called by the Quotes menu and the Refresh Prices toolbar button.
  on_load_finished  Called when a document finishes loading (Events.xcu).
"""

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "1.0.0"

SERVER_URL = os.environ.get("QUOTE_SERVER_URL", "http://127.0.0.1:8765")

# A cell lookup never waits longer than this for the server.
TIMEOUT_SEC = 0.5

# After one failed lookup, further lookups answer OFFLINE at once for this long.
OFFLINE_RETRY_SECONDS = 10.0

# Refresh Prices and document open wait at most this long for fresh prices.
REFRESH_WAIT_SECONDS = 5.0

OFFLINE_MESSAGE = ("the quote server is not reachable at {url}; start it, "
                   "then Quotes > Refresh Prices")

FORMULA_MARKER = "GETQUOTE("
CELL_FLAG_FORMULA = 16  # com.sun.star.sheet.CellFlags.FORMULA

_blocked_until = 0.0


def _block() -> None:
    global _blocked_until
    _blocked_until = time.monotonic() + OFFLINE_RETRY_SECONDS


def _unblock() -> None:
    global _blocked_until
    _blocked_until = 0.0


def _blocked() -> bool:
    return time.monotonic() < _blocked_until


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (socket.timeout, TimeoutError)) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError)))


def _err(code: str, message: str) -> str:
    """Format an error cell: the code to test for, then the cause in plain words."""
    return f"#ERR: {code} - {message}"


def _call(method: str, path: str, params: dict | None = None,
          timeout: float | None = None) -> dict:
    """Send one request and return the decoded JSON body.

    An HTTP error status still returns its JSON body (the server puts the reason in it).

    Raises:
        urllib.error.URLError, OSError: the server is unreachable or too slow.
        ValueError: the response is not JSON.
    """
    url = SERVER_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, method=method,
                                     headers={"User-Agent": "LibreOffice-Calc-Client"})
    try:
        with urllib.request.urlopen(request, timeout=timeout or TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode("utf-8"))


def getquote(symbol, field="price"):
    """Return a price for a ticker or CUSIP, or an '#ERR: <CODE> - <cause>' string.

    Args:
        symbol: Ticker or CUSIP. A blank symbol returns an empty string.
        field: price (default), ytw, or term. Bonds only for ytw and term.

    Returns:
        A float, '' for a blank symbol, or '#ERR: <CODE> - <cause in plain words>'.
        CODE is a server error code (PENDING, NOT_FOUND, NO_FIELD, BAD_SYMBOL,
        NO_NETWORK, RATE_LIMITED, STALE, ...; see the README) or one of OFFLINE,
        TIMEOUT, BAD_RESPONSE, NO_VALUE. The cause comes from the server.
    """
    symbol = str(symbol if symbol is not None else "").strip().upper()
    if not symbol:
        return ""
    field = str(field if field is not None else "").strip().lower() or "price"
    if _blocked():
        return _err("OFFLINE", OFFLINE_MESSAGE.format(url=SERVER_URL))
    try:
        body = _call("GET", "/quote", {"symbol": symbol, "field": field})
    except (urllib.error.URLError, OSError) as exc:
        _block()
        if _is_timeout(exc):
            return _err("TIMEOUT", f"the quote server did not answer within {TIMEOUT_SEC:g} s; "
                                   "try Quotes > Refresh Prices")
        return _err("OFFLINE", OFFLINE_MESSAGE.format(url=SERVER_URL))
    except ValueError:
        return _err("BAD_RESPONSE", f"the reply from {SERVER_URL} is not a quote server reply; "
                                    "check the port")
    if "error" in body:
        error = body["error"] if isinstance(body["error"], dict) else {}
        return _err(str(error.get("code", "ERROR")),
                    str(error.get("message") or "the quote server reported an error"))
    value = body.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return _err("NO_VALUE", f"the quote server returned no number for {field}")


def _is_spreadsheet(doc) -> bool:
    return doc is not None and doc.supportsService("com.sun.star.sheet.SpreadsheetDocument")


def _quote_cells(doc) -> list:
    """Return every cell whose formula calls GETQUOTE."""
    cells = []
    for index in range(doc.Sheets.Count):
        sheet = doc.Sheets.getByIndex(index)
        found = sheet.queryContentCells(CELL_FLAG_FORMULA)
        enumeration = found.Cells.createEnumeration()
        while enumeration.hasMoreElements():
            cell = enumeration.nextElement()
            if FORMULA_MARKER in cell.Formula.upper():
                cells.append(cell)
    return cells


def _recalculate(cells: list) -> None:
    """Re-enter each formula so Calc evaluates only those cells again."""
    for cell in cells:
        cell.Formula = cell.Formula


def _refresh_server(wait: float) -> bool:
    """Ask the server to refresh everything now. Returns False when it is unreachable."""
    try:
        body = _call("POST", "/refresh-all", {"wait": wait}, timeout=wait + 1.0)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return "error" not in body


def refresh_document(doc, wait: float = REFRESH_WAIT_SECONDS) -> int:
    """Register and refresh every GETQUOTE cell in a spreadsheet.

    The first pass registers the symbols with the server. The server then fetches
    fresh prices (waiting up to `wait` seconds), and a second pass shows them.

    Returns:
        The number of GETQUOTE cells found.
    """
    cells = _quote_cells(doc)
    if not cells:
        return 0
    indicator = None
    try:
        indicator = doc.CurrentController.StatusIndicator
        indicator.start("Refreshing prices", 0)
    except AttributeError:
        indicator = None
    try:
        _unblock()
        _recalculate(cells)
        if not _blocked() and _refresh_server(wait):
            _recalculate(cells)
    finally:
        if indicator is not None:
            indicator.end()
    return len(cells)


def refresh_prices(*args):
    """Refresh the active spreadsheet (Quotes menu and toolbar button)."""
    doc = XSCRIPTCONTEXT.getDocument()  # noqa: F821  (provided by LibreOffice)
    if _is_spreadsheet(doc):
        refresh_document(doc)


def on_load_finished(event=None, *args):
    """Refresh a spreadsheet that has just finished loading."""
    doc = getattr(event, "Source", None)
    if _is_spreadsheet(doc):
        refresh_document(doc)


g_exportedScripts = (getquote, refresh_prices, on_load_finished)
