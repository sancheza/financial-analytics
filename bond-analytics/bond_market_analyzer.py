#!/usr/bin/env python3
import argparse
import json
import os
import sys
from statistics import median
from pathlib import Path
from datetime import datetime
from calculate_YTW import calculate_ytw_bey

VERSION = "1.2.0"
# v1.2.0: Added --interactive mode for guided entry.

# Colors for snazzy output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[33m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Use absolute path or path relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "data", "json", "treasuries_secondary_market.json")

def parse_input():
    """Parse input data for treasury bonds.
    Reads multi-line input directly from stdin.
    """
    print("Copy and paste your treasury data below (press Ctrl+D when finished):")
    
    # Read all lines from stdin until EOF (Ctrl+D)
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    # Process the input data
    result = []

    def is_valid_ytw(ytw_str):
        try:
            ytw_val = float(ytw_str.replace('%', ''))
            return 0 < ytw_val < 30
        except Exception:
            return False

    def needs_outlier_confirmation(coupon_str, ytw_str, threshold=2.5):
        try:
            coupon_val = float(coupon_str.replace('%', ''))
            ytw_val = float(ytw_str.replace('%', ''))
            return abs(ytw_val - coupon_val) > threshold
        except Exception:
            return False

    # If we have data
    if lines:
        # Join all non-empty lines with spaces
        all_text = " ".join([line.strip() for line in lines if line.strip()])
        all_fields = all_text.split()

        # If we have at least 4 fields
        if len(all_fields) >= 4:
            try:
                coupon = all_fields[0]
                maturity = all_fields[1]
                price = all_fields[2]
                ytw = all_fields[3]  # Default to 4th field

                # If we have at least 7 fields, use field 7 instead
                if len(all_fields) >= 7:
                    ytw = all_fields[6]

                # Validate YTW
                if not is_valid_ytw(ytw):
                    print(f"Warning: YTW value '{ytw}' is out of expected range (0 < YTW < 30). Entry skipped.")
                    return []

                # Check if YTW is an outlier compared to coupon
                outlier = needs_outlier_confirmation(coupon, ytw)

                # Create an entry with today's date
                entry = {
                    "Coupon": coupon,
                    "Maturity": maturity,
                    "Price": price,
                    "YTW": ytw,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }

                # Show entry and request confirmation
                print("\nParsed bond entry:")
                for k, v in entry.items():
                    print(f"{k}: {v}")
                if outlier:
                    print(f"\nWarning: YTW ({ytw}) differs from coupon rate ({coupon}) by more than 2.5%. Please confirm this is correct.")
                confirm = input("\nAdd this entry to JSON? (Y/N): ").strip().lower()
                if confirm == 'y':
                    result.append(entry)
                else:
                    print("Entry not added.")
            except Exception as e:
                print(f"Warning: Could not parse data: {e}")

    return result

def is_numeric(value):
    """Check if a string represents a numeric value"""
    try:
        float(value.replace('$', '').replace(',', '').replace('%', ''))
        return True
    except (ValueError, AttributeError):
        return False

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading data: {e}")
            return {}
    return {}

def save_data(data):
    """Save data to the JSON file, preserving existing data not in the current dataset."""
    # Ensure directory exists
    data_dir = os.path.dirname(DATA_FILE)
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Check if the file exists and has content
        existing_data = {}
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
            try:
                with open(DATA_FILE, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing data file is corrupted, creating new file")
        
        # Merge the existing data with new data
        merged_data = existing_data.copy()
        for cusip, entries in data.items():
            if cusip in merged_data:
                existing_dates = {entry.get('Date') for entry in merged_data[cusip] if 'Date' in entry}
                for entry in entries:
                    if entry.get('Date') and entry.get('Date') not in existing_dates:
                        merged_data[cusip].append(entry)
                    elif '_comment' in entry:
                        merged_data[cusip].append(entry)
            else:
                merged_data[cusip] = entries

        # For each CUSIP, ensure only one _comment entry is kept
        for cusip in merged_data:
            comments = []
            others = []
            for entry in merged_data[cusip]:
                if '_comment' in entry:
                    if not comments:
                        comments.append(entry)
                else:
                    others.append(entry)
            merged_data[cusip] = comments + others

        # Save the merged data back to file
        with open(DATA_FILE, "w") as f:
            json.dump(merged_data, f, indent=2)
    except IOError as e:
        print(f"Error saving data: {e}")

def show_info(cusip, data):
    if cusip not in data:
        print(f"No data for CUSIP {cusip}")
        return

    try:
        # Convert YTW values to float, removing any '%' characters
        ytws = []
        prices = []
        for entry in data[cusip]:
            try:
                ytw_str = entry["YTW"].replace('%', '')
                ytws.append(float(ytw_str))
            except (ValueError, KeyError):
                pass
            try:
                price_str = entry["Price"].replace('$', '').replace(',', '')
                prices.append(float(price_str))
            except (ValueError, KeyError):
                pass

        if not ytws:
            print(f"No valid YTW values found for CUSIP {cusip}")
            return

        print(f"CUSIP: {cusip}")
        print(f"Low YTW: {min(ytws):.3f}%")
        print(f"High YTW: {max(ytws):.3f}%")
        print(f"Median YTW: {median(ytws):.3f}%")
        print(f"Latest YTW: {ytws[-1]:.3f}%")
        print(f"Number of entries: {len(ytws)}")

        # Price comparison logic
        if prices:
            latest_price = prices[-1]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            pct_vs_avg = ((latest_price - avg_price) / avg_price) * 100 if avg_price else 0
            print(f"Latest Price: ${latest_price:.3f}")
            print(f"Average Price: ${avg_price:.3f}")
            print(f"Lowest Price: ${min_price:.3f}")
            print(f"Highest Price: ${max_price:.3f}")
            if abs(pct_vs_avg) < 0.01:
                print(f"Current price is equal to the average for this CUSIP.")
            elif pct_vs_avg > 0:
                print(f"Current price is {pct_vs_avg:.2f}% more expensive than average for this CUSIP.")
            else:
                print(f"Current price is {abs(pct_vs_avg):.2f}% less expensive than average for this CUSIP.")
            if latest_price == max_price:
                print("This is the most expensive price recorded for this CUSIP.")
            if latest_price == min_price:
                print("This is the least expensive price recorded for this CUSIP.")
    except Exception as e:
        print(f"Error analyzing data for CUSIP {cusip}: {str(e)}")

def interactive_add(data):
    """Interactively add a new bond entry by prompting for fields."""
    try:
        import inquirer
    except ImportError:
        print(f"{RED}Error: 'inquirer' library is required for interactive mode.{RESET}")
        print("Please install it by running: pip install inquirer")
        return

    try:
        # 1. Prompt for CUSIP, highlighting owned bonds
        owned_cusips = set()
        for cusip, entries in data.items():
            for entry in entries:
                if "_comment" in entry and "paid" in entry.get("_comment", ""):
                    owned_cusips.add(cusip)

        cusip_choices = [("[ New CUSIP ]", "[ New CUSIP ]")]
        for cusip in sorted(data.keys()):
            if cusip in owned_cusips:
                display_text = f"{GREEN}{cusip} (owned){RESET}"
                cusip_choices.append((display_text, cusip))
            else:
                cusip_choices.append((cusip, cusip))

        questions = [
            inquirer.List('cusip',
                          message="Select a CUSIP or add a new one",
                          choices=cusip_choices,
                          ),
        ]
        answers = inquirer.prompt(questions)
        if not answers:  # User pressed Ctrl+C
            print(f"{YELLOW}Selection cancelled.{RESET}")
            return

        cusip = answers['cusip']

        if cusip == "[ New CUSIP ]":
            cusip = input(f"{BOLD}Enter new CUSIP: {RESET}").strip().upper()

        if not cusip:
            print(f"{RED}Error: CUSIP cannot be empty.{RESET}")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        # Check for existing entry for today
        if cusip in data:
            for existing_entry in data[cusip]:
                if existing_entry.get("Date") == today_str:
                    print(f"{YELLOW}Entry for CUSIP {cusip} on {today_str} already exists. Not adding duplicate.{RESET}")
                    return

        # 2. Check if CUSIP exists in the data at all
        if cusip in data and data[cusip]:
            # CUSIP exists, use its data to pre-fill
            print(f"{CYAN}CUSIP {cusip} found. Using existing data for Coupon and Maturity.{RESET}")
            # Find the most recent non-comment entry to get coupon/maturity
            latest_entry = None
            for entry in reversed(data[cusip]):
                if '_comment' not in entry:
                    latest_entry = entry
                    break
            
            if not latest_entry:
                print(f"{RED}Error: No valid previous data entries found for CUSIP {cusip} to source coupon/maturity from.{RESET}")
                return

            coupon_str = latest_entry['Coupon']
            maturity_str = latest_entry['Maturity']
            print(f"  - Coupon: {GREEN}{coupon_str}{RESET}")
            print(f"  - Maturity: {GREEN}{maturity_str}{RESET}")

            # Prompt for price
            while True:
                try:
                    price_str = input(f"{BOLD}Enter Current Price: {RESET}")
                    price = float(price_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid price. Please enter a number.{RESET}")

            # Calculate YTW
            print(f"{CYAN}Calculating YTW...{RESET}")
            ytw_pct = calculate_ytw_bey(
                coupon_rate=float(coupon_str),
                maturity_date=maturity_str,
                price=price
            )
            ytw = f"{ytw_pct:.3f}"
            print(f"Calculated YTW: {GREEN}{ytw}%{RESET}")

            new_entry = {
                "Coupon": coupon_str,
                "Maturity": maturity_str,
                "Price": f"{price:.3f}",
                "YTW": ytw,
                "Date": today_str
            }

        else:
            # CUSIP does not exist, prompt for everything
            print(f"{CYAN}CUSIP {cusip} not found. Please provide all details for this new bond.{RESET}")

            # Prompt for coupon
            while True:
                try:
                    coupon_str = input(f"{BOLD}Enter Coupon Rate: {RESET}").strip()
                    coupon = float(coupon_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid coupon. Please enter a number.{RESET}")

            # Prompt for maturity
            while True:
                try:
                    maturity_str = input(f"{BOLD}Enter Maturity Date (MM/DD/YYYY): {RESET}").strip()
                    datetime.strptime(maturity_str, "%m/%d/%Y")  # just for validation
                    break
                except ValueError:
                    print(f"{RED}Invalid date format. Please use MM/DD/YYYY.{RESET}")

            # Prompt for price
            while True:
                try:
                    price_str = input(f"{BOLD}Enter Price: {RESET}").strip()
                    price = float(price_str)
                    break
                except ValueError:
                    print(f"{RED}Invalid price. Please enter a number.{RESET}")

            # Calculate YTW
            print(f"{CYAN}Calculating YTW...{RESET}")
            ytw_pct = calculate_ytw_bey(
                coupon_rate=coupon,
                maturity_date=maturity_str,
                price=price
            )
            ytw_str = f"{ytw_pct:.3f}"
            print(f"Calculated YTW: {GREEN}{ytw_str}%{RESET}")

            new_entry = {
                "Coupon": f"{coupon:.3f}",
                "Maturity": maturity_str,
                "Price": f"{price:.3f}",
                "YTW": ytw_str,
                "Date": today_str
            }

        # Confirmation before adding
        print(f"\n{BOLD}{CYAN}New entry to be added:{RESET}")
        for k, v in new_entry.items():
            print(f"  {k}: {GREEN}{v}{RESET}")
        confirm = input(f"\n{BOLD}Add this entry? [Y/n]: {RESET}").strip().lower()
        if confirm == 'n':
            print(f"{YELLOW}Entry not added.{RESET}")
            return

        # Add to data structure and save
        if cusip not in data:
            data[cusip] = []
        data[cusip].append(new_entry)
        save_data(data)
        print(f"{GREEN}Saved new entry for CUSIP {cusip}.{RESET}")

    except (ValueError, RuntimeError) as e:
        print(f"\n{RED}Error: {e}. Aborting interactive add.{RESET}")
    except Exception as e:
        print(f"\n{RED}An unexpected error occurred: {e}. Aborting.{RESET}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Bond Market Analyzer", add_help=False)
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--CUSIP", type=str, help="CUSIP identifier")
    parser.add_argument("--getinfo", action="store_true")
    parser.add_argument("--copypaste", action="store_true", help="Use copy/paste mode for data entry instead of the default interactive mode.")
    args = parser.parse_args()

    if args.version:
        print(f"Bond Market Analyzer version {VERSION}")
        return
    if args.help:
        script_name = os.path.basename(sys.argv[0])
        print(f"\n{BOLD}{GREEN}Bond Market Analyzer v{VERSION}{RESET}")
        print("This script collects values for a bond offering, saves them to a local JSON file, and provides simple analytics.")
        print("It supports both an interactive guided entry mode and a manual copy-paste mode.")
        
        print(f"\n{BOLD}{CYAN}USAGE:{RESET}")
        print(f"  {script_name} [OPTIONS]")
        print(f"  Run without arguments for interactive mode to add a new bond.")
        print(f"  Use '--copypaste' for manual data entry.")

        print(f"\n{BOLD}{CYAN}EXAMPLE (Copy/Paste Mode):{RESET}")
        print("When using --copypaste, you will be prompted to paste bond values, e.g.:")
        print(f"{YELLOW}4.85\t11/01/2064\t95.445 Empty--\t\t5.12 Empty--\t5.12 Empty--\t{RESET}\n")

        print(f"\n{BOLD}{CYAN}OPTIONS:{RESET}")
        print(f"  {BOLD}--copypaste{RESET}\t\tUse copy/paste mode for data entry instead of interactive mode.")
        print(f"  {BOLD}--CUSIP CUSIP{RESET}\tSpecify CUSIP identifier for adding or retrieving data.")
        print(f"  {BOLD}--getinfo{RESET}\t\tShow statistics (low, high, median, latest YTW) for a CUSIP.")
        print(f"  {BOLD}-h, --help{RESET}\t\tShow this help message and exit.")
        print(f"  {BOLD}-v, --version{RESET}\t\tShow script version and exit.")

        print(f"\n{BOLD}{CYAN}NOTE:{RESET}")
        print("The script is designed for simplicity, using manual entry instead of API calls.")
        print("It will detect if an entry for a given CUSIP and date already exists to avoid duplicates.")
        return

    try:
        data = load_data()

        if args.getinfo:
            cusip = args.CUSIP or input("Enter CUSIP: ").strip()
            show_info(cusip, data)
            return
        
        if not args.copypaste:
            interactive_add(data)
            return

        # Default mode: copy-paste entry
        cusip = args.CUSIP or input("Enter CUSIP: ").strip().upper()
        if not cusip:
            print("Error: CUSIP cannot be empty")
            return
            
        entries = parse_input()
        if not entries:
            print("No valid entries found.")
            return
            
        if cusip not in data:
            data[cusip] = []
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if we already have an entry for this CUSIP for today
        entry_exists = False
        for existing_entry in data[cusip]:
            if existing_entry.get("Date") == today:
                entry_exists = True
                print(f"Entry for CUSIP {cusip} on {today} already exists. Not adding duplicate.")
                
                # Show the existing entry
                print("\nExisting entry:")
                print(f"Coupon: {existing_entry['Coupon']}, Maturity: {existing_entry['Maturity']}, "
                      f"Price: {existing_entry['Price']}, YTW: {existing_entry['YTW']}, Date: {existing_entry['Date']}")
                # Also show price comparison summary
                print("\nSummary for CUSIP:")
                show_info(cusip, data)
                break
        
        # Only add new entries if no entry exists for today
        if not entry_exists:
            # Add the new entries to the data
            data[cusip].extend(entries)
            save_data(data)
            print(f"Saved {len(entries)} entries for CUSIP {cusip}.")

            # Show only the information just entered, with correct labels
            print("\nEntry added:")
            for entry in entries:
                print(f"Coupon: {entry['Coupon']}, Maturity: {entry['Maturity']}, "
                      f"Price: {entry['Price']}, YTW: {entry['YTW']}, Date: {entry['Date']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()