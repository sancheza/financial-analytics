#!/usr/bin/env python3

import sys
import requests
from bs4 import BeautifulSoup

def get_forward_pe(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all list items that might contain the Forward P/E value
    for li in soup.find_all("li", class_="yf-i6syij"):
        label = li.find("p", class_="label")
        value = li.find("p", class_="value")
        if label and value and "Forward P/E" in label.text:
            return value.text.strip()

    return "N/A"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_forward_pe.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    print(get_forward_pe(ticker))

