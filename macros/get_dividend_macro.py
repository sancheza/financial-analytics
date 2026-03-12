#!/usr/bin/env python3
"""
LibreOffice Calc macro to populate dividend yield data for stock tickers.

This macro reads ticker symbols from column A and populates the corresponding
"Dividend Yield" column by calling an external script to fetch dividend yield
data from financial sources. Numeric values are stored as numbers for calculations,
while "N/A" entries are stored as text.
"""

import subprocess
import os
import uno

VERSION = "1.0.3"

def get_dividend_yield_macro():
    doc = XSCRIPTCONTEXT.getDocument()
    sheet = doc.CurrentController.ActiveSheet

    # Dynamically resolve user home path
    home = os.path.expanduser("~")
    script_path = os.path.join(home, "dev", "financial-analytics", "run_get_dividend.sh")

    # Find "Dividend Yield" column index from first row
    dividend_yield_col = -1
    for col in range(0, 100):  # Arbitrary limit to 100 columns
        cell = sheet.getCellByPosition(col, 0)  # row 0 = header
        if cell.getString().strip().lower() == "dividend yield":
            dividend_yield_col = col
            break

    if dividend_yield_col == -1:
        raise Exception("Column with header 'Dividend Yield' not found.")

    row = 1  # Start from second row (index 1)

    while True:
        ticker_cell = sheet.getCellByPosition(0, row)  # Column A = index 0
        ticker = ticker_cell.getString().strip().upper()
        if not ticker:
            break

        try:
            result = subprocess.check_output(
                [script_path, ticker],
                stderr=subprocess.STDOUT,
                timeout=5
            )
            dividend_yield = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            dividend_yield = None  # Set to None on error
        except Exception:
            dividend_yield = None  # Set to None on error

        # Update the cell if dividend_yield is valid or "N/A"
        if dividend_yield and dividend_yield.strip() and not dividend_yield.lower().startswith("error"):
            output_cell = sheet.getCellByPosition(dividend_yield_col, row)
            
            # Try to convert to float for numeric values
            try:
                if dividend_yield.strip().upper() == "N/A":
                    output_cell.setString("N/A")  # Keep as text
                else:
                    # Handle percentage values - remove % sign if present
                    clean_value = dividend_yield.strip().rstrip('%')
                    numeric_value = float(clean_value)
                    output_cell.setValue(numeric_value / 100) # convert to decimal value
            except ValueError:
                # If conversion fails, set as string
                output_cell.setString(dividend_yield)

        row += 1
