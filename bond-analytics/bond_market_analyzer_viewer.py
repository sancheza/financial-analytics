#!/usr/bin/env python3
import json
import os
import argparse

def display_treasury_data(file_path):
    """
    Reads a JSON file containing treasury bond data, extracts the values,
    and prints them in a well-formatted table, with a separate row for each entry.
    
    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        # Check if the file exists before trying to open it
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return

        # Open and load the JSON data from the file
        with open(file_path, 'r') as f:
            treasury_data = json.load(f)

        # Define the ANSI escape codes for coloring the output
        # These codes may not work in all terminals.
        COLOR_DEFAULT = "\033[0m"       # Reset to default color (white in most terminals)
        COLOR_ALT = "\033[36m"          # A cyan color for alternating rows
        COLOR_GREEN = "\033[32m"        # Green for favorable pricing
        COLOR_RED = "\033[31m"          # Red for unfavorable pricing
        
        current_color = COLOR_DEFAULT
        last_cusip = None

        # Define the headers for the table
        headers = ["CUSIP", "Coupon", "Maturity", "Price", "Bought At", "YTW", "Date"]
        
        # Print the table header with proper spacing
        print(f"{headers[0]:<12} {headers[1]:<8} {headers[2]:<15} {headers[3]:<8} {headers[4]:<12} {headers[5]:<8} {headers[6]:<12}")
        print("-" * 87) # A separator line adjusted for the new column order

        # Sort the CUSIP keys before iterating
        sorted_cusips = sorted(treasury_data.keys())

        for cusip in sorted_cusips:
            bond_list = treasury_data[cusip]
            
            # Check if this is a new CUSIP and switch colors if it is
            if last_cusip is None or cusip != last_cusip:
                current_color = COLOR_ALT if current_color == COLOR_DEFAULT else COLOR_DEFAULT
                last_cusip = cusip

            # Initialize bought_at for each new CUSIP
            bought_at = "N/A"

            # Iterate through the list of dictionaries for each CUSIP
            for item in bond_list:
                if "_comment" in item:
                    # If a comment is found, extract the price and store it
                    comment_text = item.get("_comment", "")
                    if "paid" in comment_text:
                        # Extract the numeric value from a string like "paid 100.9063"
                        bought_at = comment_text.split()[-1]
                    else:
                        bought_at = comment_text
                else:
                    # If it's not a comment, it's a new data entry. Print it as a new row.
                    # This ensures every data entry is displayed.
                    coupon = item.get("Coupon", "N/A")
                    maturity = item.get("Maturity", "N/A")
                    price = item.get("Price", "N/A")
                    ytw = item.get("YTW", "N/A")
                    date = item.get("Date", "N/A")
                    
                    bought_at_color = COLOR_DEFAULT
                    try:
                        bought_at_float = float(bought_at)
                        price_float = float(price)
                        if bought_at_float < price_float:
                            bought_at_color = COLOR_GREEN
                        elif bought_at_float > price_float:
                            bought_at_color = COLOR_RED
                    except (ValueError, TypeError):
                        # If conversion fails (e.g., 'N/A'), keep the default color
                        pass
                    
                    print(f"{current_color}{cusip:<12} {coupon:<8} {maturity:<15} {price:<8} {bought_at_color}{bought_at:<12}{COLOR_DEFAULT} {ytw:<8} {date:<12}{COLOR_DEFAULT}")
    
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A script to display treasury bond data from a JSON file.")
    
    # Add the --jsonfile argument
    parser.add_argument(
        "--jsonfile",
        type=str,
        help="Path to a custom JSON file to use instead of the default."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Determine the file path based on the argument
    if args.jsonfile:
        json_file_path = args.jsonfile
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, 'data', 'json', 'treasuries_secondary_market.json')

    # Call the function to display the data
    display_treasury_data(json_file_path)
