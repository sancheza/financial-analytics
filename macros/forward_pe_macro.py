#!/usr/bin/env python3
"""
LibreOffice Calc macro to populate Forward P/E ratio data for stock tickers.

This macro reads ticker symbols from column A and populates the corresponding
"Forward PE" column by calling an external script to fetch forward price-to-earnings
data from financial sources. Numeric values are stored as numbers for calculations,
while "N/A" entries and error messages are stored as text.
"""

import subprocess
import os
import uno

VERSION = "1.0.2"

def get_forward_pe_macro():
    doc = XSCRIPTCONTEXT.getDocument()
    sheet = doc.CurrentController.ActiveSheet

    # Dynamically resolve user home path
    home = os.path.expanduser("~")
    script_path = os.path.join(home, "dev", "financial-analytics", "run_forward_pe.sh")

    # Find "Forward PE" column index from first row
    forward_pe_col = -1
    for col in range(0, 100):  # Arbitrary limit to 100 columns
        cell = sheet.getCellByPosition(col, 0)  # row 0 = header
        if cell.getString().strip().lower() == "forward pe":
            forward_pe_col = col
            break

    if forward_pe_col == -1:
        raise Exception("Column with header 'Forward PE' not found.")

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
            pe_value = result.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            pe_value = f"Error: {e.output.decode('utf-8').strip()}"
        except Exception as e:
            pe_value = f"Error: {str(e)}"

        output_cell = sheet.getCellByPosition(forward_pe_col, row)
        
        # Try to convert to float for numeric values
        try:
            if pe_value.strip().upper() == "N/A":
                output_cell.setString("N/A")  # Keep as text
            elif pe_value.lower().startswith("error"):
                output_cell.setString(pe_value)  # Keep error messages as text
            else:
                numeric_value = float(pe_value.strip())
                output_cell.setValue(numeric_value)  # Set as number
        except ValueError:
            # If conversion fails, set as string
            output_cell.setString(pe_value)

        row += 1
