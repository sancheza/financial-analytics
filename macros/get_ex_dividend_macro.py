import subprocess
import os

VERSION = "1.0.0"

def get_ex_dividend_date_macro():
    doc = XSCRIPTCONTEXT.getDocument()
    sheet = doc.CurrentController.ActiveSheet

    # Dynamically resolve user home path
    home = os.path.expanduser("~")
    script_path = os.path.join(home, "dev", "financial-analytics", "run_get_ex_dividend.sh")

    # Find "Ex-Dividend" column index from first row
    ex_dividend_col = -1
    for col in range(0, 100):  # Arbitrary limit to 100 columns
        cell = sheet.getCellByPosition(col, 0)  # row 0 = header
        if cell.getString().strip().lower() == "ex-dividend":
            ex_dividend_col = col
            break

    if ex_dividend_col == -1:
        raise Exception("Column with header 'Ex-Dividend' not found.")

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
            ex_dividend = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            ex_dividend = None  # Set to None on error
        except Exception:
            ex_dividend = None  # Set to None on error

        # Update the cell if ex_dividend is valid or "N/A"
        if ex_dividend and ex_dividend.strip() and not ex_dividend.lower().startswith("error"):
            output_cell = sheet.getCellByPosition(ex_dividend_col, row)
            output_cell.setString(ex_dividend)

        row += 1