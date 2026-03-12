#!/usr/bin/env python3
"""
LibreOffice Calc macro to populate fair value data for stock tickers.

This macro reads ticker symbols from column A and populates the corresponding
"Fair Value" columns by calling an external helper script which then calls the
main retrieval script to fetch fair value from financial sources. Numeric values
are stored as numbers for calculations, while "N/A" entries are stored as text.
"""

import subprocess
import os
import uno
import time

VERSION = "1.3.1"

def find_column_indices(sheet, column_mappings):
    """
    Finds the column indices for each header in the sheet.
    """
    fair_value_cols = {}
    for col in range(0, 100):
        cell = sheet.getCellByPosition(col, 0)
        header = cell.getString().strip()

        for arg, header_text in column_mappings.items():
            if header.lower() == header_text.lower():
                fair_value_cols[arg] = col
                break
    return fair_value_cols

def get_fair_value_macro():
    doc = XSCRIPTCONTEXT.getDocument()
    sheet = doc.CurrentController.ActiveSheet

    # Define the arguments and their corresponding column headers
    column_mappings = {
        "-as": "Fair Value (AS)",
        "-vi": "Fair Value (VI)",
        "-gf": "Fair Value (GF)"
    }
    
    # --- NEW CODE: Initialize a cache to store results ---
    fair_value_cache = {}
    # --- END OF NEW CODE ---

    fair_value_cols = find_column_indices(sheet, column_mappings)
    # Check if all required columns were found
    if len(fair_value_cols) != len(column_mappings):
        missing_args = set(column_mappings.keys()) - set(fair_value_cols.keys())
        missing_headers = [column_mappings[arg] for arg in missing_args]
        raise Exception(f"The following required columns were not found: {', '.join(missing_headers)}")
    # Dynamically resolve user home path
    home = os.path.expanduser("~")
    script_path = os.path.join(home, "dev", "financial-analytics", "run_get_fair_value.sh")
    
    # --- NEW: Count total tickers for the progress bar ---
    total_tickers = 0
    r = 1
    while True:
        cell = sheet.getCellByPosition(0, r)  # Ticker is in column 0
        if not cell.getString().strip():
            break
        total_tickers += 1
        r += 1
    # --- END OF NEW CODE ---

    # --- NEW: Initialize status indicator ---
    status_indicator = doc.CurrentController.StatusIndicator
    status_indicator.start("Fetching fair values...", total_tickers)
    # --- END OF NEW CODE ---

    row = 1  # Start from second row (index 1)

    while True:
        ticker_cell = sheet.getCellByPosition(0, row)  # Column A = index 0
        ticker = ticker_cell.getString().strip().upper()
        if not ticker:
            break

        # Clean the ticker symbol by removing exchange suffixes
        if ':' in ticker:
            ticker = ticker.split(':')[0]
        
        # Get the cell from Column B (index 1)
        asset_type_cell = sheet.getCellByPosition(1, row)
        asset_type = asset_type_cell.getString().strip().lower()

        # --- NEW: Update status indicator ---
        status_indicator.setText(f"Processing {row}/{total_tickers}: {ticker}")
        status_indicator.setValue(row)
        # --- END OF NEW CODE ---
        
        # If the asset type is "bond", fill with "N/A" and skip this row
        if asset_type == "bond":
            for arg, _ in column_mappings.items():
                output_col = fair_value_cols[arg]
                output_cell = sheet.getCellByPosition(output_col, row)
                output_cell.setString("N/A")
            
            row += 1
            continue

        # --- OPTIMIZATION: Make one call per ticker for all sources ---
        cache_key = ticker  # Use just the ticker as the key
        if cache_key in fair_value_cache:
            all_fair_values = fair_value_cache[cache_key]
        else:
            try:
                # Get the list of arguments to pass to the script
                args_to_pass = list(column_mappings.keys())

                # --- MODIFICATION: Use Popen to avoid UI freeze ---
                # Start the subprocess in a non-blocking way
                process = subprocess.Popen(
                    [script_path, ticker] + args_to_pass,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8'
                )

                # Poll the process to keep the UI responsive
                start_time = time.time()
                while process.poll() is None:
                    if time.time() - start_time > 90:
                        process.kill()
                        process.communicate()  # Clean up streams
                        raise subprocess.TimeoutExpired(process.args, 90)
                    time.sleep(0.1)  # Yield to allow UI updates

                # Get output and check for errors
                output, _ = process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args, output=output)
                # --- END OF MODIFICATION ---

                # The script returns values separated by newlines
                # Robustly split the output, handling potential trailing newlines.
                all_fair_values = output.split('\n')
                if all_fair_values and all_fair_values[-1] == '':
                    all_fair_values.pop()

                # Store the list of results in the cache
                fair_value_cache[cache_key] = all_fair_values
                
            except subprocess.TimeoutExpired:
                all_fair_values = ["Error: Timeout"] * len(column_mappings)
            except subprocess.CalledProcessError as e:
                error_output = e.output.strip().replace('\n', ' ')
                all_fair_values = [f"Error: {error_output}"] * len(column_mappings)
            except Exception as e:
                all_fair_values = [f"Error: {e}"] * len(column_mappings)

        # Map the results back to the corresponding columns
        ordered_args = list(column_mappings.keys())
        
        # Ensure we have the expected number of results
        if len(all_fair_values) != len(ordered_args):
            all_fair_values = [f"Error: Mismatched results"] * len(ordered_args)

        for i, arg in enumerate(ordered_args):
            fair_value = all_fair_values[i]
            output_col = fair_value_cols[arg]
            output_cell = sheet.getCellByPosition(output_col, row)

            if fair_value and not fair_value.lower().startswith("error"):
                try:
                    if fair_value.strip().upper() == "N/A":
                        output_cell.setString("N/A")
                    else:
                        numeric_value = float(fair_value.strip().replace('$', '').replace(',', ''))
                        output_cell.setValue(numeric_value)
                except ValueError:
                    output_cell.setString(fair_value)
            else:
                output_cell.setString(fair_value)
        row += 1

    # --- NEW: End the status indicator when done ---
    status_indicator.end()
    # --- END OF NEW CODE ---