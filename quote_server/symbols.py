"""Classify a symbol as a stock ticker or a Treasury CUSIP."""

import re

STOCK = "stock"
BOND = "bond"

_CUSIP_RE = re.compile(r"^[0-9A-Z*@#]{9}$")
_TICKER_RE = re.compile(r"^[A-Z^][A-Z0-9.\-=]{0,9}$")
_EXCHANGE_TICKER_RE = re.compile(r"^[0-9]{1,6}\.[A-Z]{1,3}$")   # 7203.T, 0700.HK
_CUSIP_IN_TEXT_RE = re.compile(r"(?<![0-9A-Z])[0-9A-Z]{9}(?![0-9A-Z])")
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)(E[+-]?\d+)?$")


def normalize(symbol: str) -> str:
    """Return the symbol upper-cased with surrounding whitespace removed."""
    return (symbol or "").strip().upper()


def cusip_check_digit(body: str) -> int:
    """Compute the CUSIP check digit for the first 8 characters.

    Args:
        body: The first 8 characters of a CUSIP.

    Returns:
        The expected 9th character as an integer 0-9.
    """
    total = 0
    for index, char in enumerate(body):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char) - 55
        else:
            value = {"*": 36, "@": 37, "#": 38}[char]
        if index % 2 == 1:
            value *= 2
        total += value // 10 + value % 10
    return (10 - total % 10) % 10


def classify(symbol: str) -> str | None:
    """Return STOCK, BOND, or None when the symbol is neither.

    A 9-character symbol is a CUSIP and must pass the check-digit test.

    Args:
        symbol: A normalized symbol.
    """
    if _CUSIP_RE.match(symbol):
        if symbol[8].isdigit() and int(symbol[8]) == cusip_check_digit(symbol[:8]):
            return BOND
        return None
    if _TICKER_RE.match(symbol) or _EXCHANGE_TICKER_RE.match(symbol):
        return STOCK
    return None


def cusips_in(text: str) -> list[str]:
    """Return the distinct valid CUSIPs found as separate words in normalized text.

    A 9-character word counts only when it passes the CUSIP check digit, so ordinary
    words are not mistaken for CUSIPs.
    """
    found: list[str] = []
    for word in _CUSIP_IN_TEXT_RE.findall(text):
        if classify(word) == BOND and word not in found:
            found.append(word)
    return found


def resolve(raw: str) -> str:
    """Return the symbol to look up for text from a cell.

    A ticker or CUSIP alone is used as is. Text around exactly one valid CUSIP
    ("5Y UST 91282CJF9") resolves to that CUSIP. Anything else is returned
    normalized, for classify() and bad_symbol_reason() to judge.
    """
    symbol = normalize(raw)
    if classify(symbol) is not None:
        return symbol
    found = cusips_in(symbol)
    return found[0] if len(found) == 1 else symbol


def is_number(symbol: str) -> bool:
    """Return True when the symbol reads as a number (5, 1234.5, -3, 1E5)."""
    return bool(_NUMBER_RE.match(symbol))


def bad_symbol_reason(raw: str, symbol: str) -> tuple[str, str]:
    """Explain why a symbol that classify() rejected is unusable.

    Args:
        raw: The symbol as submitted.
        symbol: The normalized symbol.

    Returns:
        (error code, message). Several CUSIPs give AMBIGUOUS_SYMBOL; a number gives
        NEED_TICKER_OR_CUSIP; anything else BAD_SYMBOL.
    """
    found = cusips_in(symbol)
    if len(found) > 1:
        return "AMBIGUOUS_SYMBOL", f"'{raw}' contains more than one CUSIP ({', '.join(found)}). Keep one."
    if is_number(symbol):
        message = f"'{raw}' is a number. Submit a ticker (such as AAPL) or a CUSIP (such as 912810UF3)."
        if symbol.isdigit() and len(symbol) == 8:
            message += " A CUSIP has 9 characters; a leading zero may have been dropped, so format the cell as Text."
        return "NEED_TICKER_OR_CUSIP", message
    return "BAD_SYMBOL", f"'{raw}' is not a ticker and contains no valid CUSIP"
