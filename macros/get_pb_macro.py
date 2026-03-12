#!/usr/bin/env python3
"""
LibreOffice Calc macro to populate Price-to-Book (P/B) ratio data for stock tickers.

This macro reads ticker symbols from column A and populates the corresponding
"P/B" column by calling an external script to fetch price-to-book ratio
data from financial sources. Numeric values are stored as numbers for calculations,
while "N/A" entries are stored as text.
"""

import subprocess
import os
import uno

VERSION = "1.0.2"

def get_pb_macro():
    doc = XSCRIPTCONTEXT.getDocument()
    sheet = doc.CurrentController.ActiveSheet

    # Dynamically resolve user home path
    home = os.path.expanduser("~")
    script_path = os.path.join(home, "dev", "financial-analytics", "run_get_pb.sh")

    # Find "P/B" column index from first row
    pb_col = -1
    for col in range(0, 100):  # Arbitrary limit to 100 columns
        cell = sheet.getCellByPosition(col, 0)  # row 0 = header
        if cell.getString().strip().lower() == "p/b":
            pb_col = col
            break

    if pb_col == -1:
        raise Exception("Column with header 'P/B' not found.")

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
            pb = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            pb = None  # Set to None on error
        except Exception:
            pb = None  # Set to None on error

        # Update the cell if pb is valid or "N/A"
        if pb and pb.strip() and not pb.lower().startswith("error"):
            output_cell = sheet.getCellByPosition(pb_col, row)
            
            # Try to convert to float for numeric values
            try:
                if pb.strip().upper() == "N/A":
                    output_cell.setString("N/A")  # Keep as text
                else:
                    numeric_value = float(pb.strip())
                    output_cell.setValue(numeric_value)  # Set as number
            except ValueError:
                # If conversion fails, set as string
                output_cell.setString(pb)

        row += 1
