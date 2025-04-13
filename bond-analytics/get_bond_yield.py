#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime, timedelta
import requests

VERSION = "1.04"

# Dictionary of duration codes to labels
DURATION_LABELS = {
    # Treasury Bills
    "4W": "4-Week",
    "8W": "8-Week",
    "13W": "13-Week",
    "17W": "17-Week",
    "26W": "26-Week",
    "52W": "52-Week",
    # Treasury Notes
    "2Y": "2-Year",
    "3Y": "3-Year",
    "5Y": "5-Year",
    "7Y": "7-Year",
    "10Y": "10-Year",
    # Treasury Bonds
    "20Y": "20-Year",
    "30Y": "30-Year",
    # TIPS
    "5Y TIPS": "5-Year TIPS",
    "10Y TIPS": "10-Year TIPS",
    "30Y TIPS": "30-Year TIPS",
    # FRNs
    "2Y FRN": "2-Year FRN"
}

# Dictionary to determine the correct security_type for each duration
SECURITY_TYPES = {
    # Treasury Bills
    "4W": "Bill",
    "8W": "Bill",
    "13W": "Bill",
    "17W": "Bill",
    "26W": "Bill",
    "52W": "Bill",
    # Treasury Notes
    "2Y": "Note",
    "3Y": "Note",
    "5Y": "Note",
    "7Y": "Note",
    "10Y": "Note",
    # Treasury Bonds
    "20Y": "Bond",
    "30Y": "Bond",
    # TIPS - directly use API types
    "5Y TIPS": "TIPS Note",
    "10Y TIPS": "TIPS Note",
    "30Y TIPS": "TIPS Bond",
    # FRNs
    "2Y FRN": "FRN Note"
}

DESCRIPTION = f"""
Treasury Bond Yield Fetcher v{VERSION}
Retrieves real historical auction yields for Treasury bills, notes, bonds, TIPS, and FRNs.
"""

EPILOG = """
Examples:
  Get the latest 10-Year Treasury Note yield:
    python get_bond_yield.py --duration 10Y
    
  Get the 2-Year Treasury Note yield for a specific date:
    python get_bond_yield.py --duration 2Y --date 05152023
    
  Get the 13-Week Treasury Bill yield:
    python get_bond_yield.py --duration 13W
    
  Get the 5-Year TIPS yield:
    python get_bond_yield.py --duration "5Y TIPS"
"""

# Use the FiscalData API (known to work)
API_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
AUCTION_DATA_ENDPOINT = f"{API_BASE_URL}/v1/accounting/od/auctions_query"

def fetch_auction_yield(duration_code, target_date):
    """
    Fetch the auction yield for a specific Treasury security using the FiscalData API.
    
    Args:
        duration_code (str): Duration code like "10Y" or "5Y TIPS"
        target_date (datetime): Target date to fetch yield for
        
    Returns:
        tuple or None: (security_type, security_term, cusip, auction_date, yield_value) if found,
                    None if not found or error occurs
    """
    # Get the security type and term label
    security_type = SECURITY_TYPES.get(duration_code)
    security_term_label = DURATION_LABELS.get(duration_code)
    
    if not security_type or not security_term_label:
        print(f"Invalid duration code: {duration_code}")
        return None
    
    # Determine if this is a special security type (TIPS, FRN)
    is_tips = "TIPS" in security_type
    is_frn = "FRN" in security_type
    
    try:
        # Format date for API query
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        # Extract target year/week number from the term
        term_parts = security_term_label.split('-')
        if len(term_parts) >= 2:
            target_num_str = term_parts[0]
            try:
                target_num = int(target_num_str)
            except ValueError:
                print(f"Error: Could not parse number from term '{security_term_label}'")
                return None
        else:
            print(f"Error: Invalid term format '{security_term_label}'")
            return None
            
        # Request parameters
        params = {
            "fields": "auction_date,security_type,security_term,high_yield,cusip",
            "sort": "-auction_date",
            "format": "json",
            "page[size]": "300"
        }
        
        # For special security types (TIPS, FRN), don't filter by security_type in the API call
        # as we'll handle this in the client-side filtering
        if not is_tips and not is_frn:
            params["filter"] = f"security_type:eq:{security_type}"
        
        print(f"\nQuerying Treasury API for {security_type}s with term related to {security_term_label} around {target_date_str}")
        r = requests.get(AUCTION_DATA_ENDPOINT, params=params)
        
        print(f"API URL: {r.url}")
        
        r.raise_for_status()
        data = r.json()
        
        if not data.get("data"):
            print(f"No auction data returned from API.")
            return None
        
        auctions = data.get("data", [])    
        print(f"\nReceived {len(auctions)} auctions. Filtering for {security_type} with terms related to {security_term_label}...")
        
        # Filter for relevant auctions based on term pattern and security type
        relevant_auctions = []
        for auction in auctions:
            auction_term = auction.get("security_term", "")
            auction_security_type = auction.get("security_type", "")
            
            # Skip if missing term or type
            if not auction_term or not auction_security_type:
                continue
                
            # For TIPS and FRNs, ensure we match the correct security_type
            if is_tips and "TIPS" not in auction_security_type:
                continue
                
            if is_frn and "FRN" not in auction_security_type:
                continue
                
            if not is_tips and not is_frn:
                if auction_security_type != security_type:
                    continue
            
            # Handle different security types differently
            is_match = False
            
            # For Bills: match exact week term (e.g., "4-Week", "13-Week")
            if security_type == "Bill":
                if auction_term == security_term_label:
                    is_match = True
                    
                # For Bills, add all matching auctions even if yield is null
                # since we'll prioritize ones with valid yields later
                if is_match:
                    relevant_auctions.append(auction)
                    continue
            
            # For Notes and Bonds: Check for exact match or Year-1 patterns
            elif security_type in ["Note", "Bond"] or is_tips or is_frn:
                # For TIPS, use base term like "10-Year" from "10-Year TIPS"
                check_term = security_term_label
                if is_tips or is_frn:
                    # Extract just the year portion for comparison
                    parts = security_term_label.split()
                    if len(parts) > 0:
                        check_term = parts[0]  # Get "10-Year" from "10-Year TIPS"
                
                # Case 1: Exact match or prefix match
                if auction_term == check_term or auction_term.startswith(check_term):
                    is_match = True
                # Case 2: Year-1 with months pattern
                elif auction_term.startswith(f"{target_num-1}-Year "):
                    is_match = True
                # Case 3: Special case handling for reopenings
                elif f"{target_num}" in auction_term:
                    is_match = True
                    
            # Add to relevant auctions if it matches and has yield data
            if is_match:
                high_yield = auction.get("high_yield")
                
                # For FRNs, always accept matches since they have null yields
                if is_frn:
                    relevant_auctions.append(auction)
                # For other securities, only include those with valid yield data
                elif high_yield is not None and high_yield != "null":
                    relevant_auctions.append(auction)
        
        if not relevant_auctions:
            print(f"No matching {security_type} auctions found with term pattern related to {security_term_label}.")
            return None
            
        print(f"Found {len(relevant_auctions)} matching {security_type} auctions. Finding closest to {target_date_str}...")
        
        # Find auction closest to target date
        closest_auction = None
        min_diff = timedelta.max
        
        for auction in relevant_auctions:
            try:
                auction_date = datetime.strptime(auction['auction_date'], "%Y-%m-%d")
                diff = abs(auction_date - target_date)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_auction = auction
                    
            except (ValueError, KeyError) as e:
                print(f"Skipping auction with error: {e}")
                continue
        
        if not closest_auction:
            print(f"Could not determine the closest auction with a valid yield.")
            return None

        # Print the details of the closest auction
        print(f"Closest auction found: Date={closest_auction['auction_date']}, "
              f"Term={closest_auction['security_term']}, "
              f"Yield={closest_auction['high_yield']}% "
              f"(Difference: {min_diff.days} days)")

        # Handle null yield values
        if closest_auction["high_yield"] == "null" or closest_auction["high_yield"] is None:
            if is_frn:
                print("\nNote: FRNs do not have a fixed 'high_yield' value as they use floating rates based on auction-determined spreads.")
                print("The spread to the index rate would be a more appropriate measure for FRNs.")
                
                # Return FRN data with a placeholder yield
                return (
                    closest_auction["security_type"],
                    closest_auction.get("security_term", "N/A"),
                    closest_auction["cusip"],
                    datetime.strptime(closest_auction["auction_date"], "%Y-%m-%d"),
                    0.0  # Placeholder yield
                )
            else:
                print("\nWarning: The closest auction has a null yield value.")
                print("This is common with recent Treasury Bill auctions.")
                print("The yield data may not be available yet or is reported differently.")
                
                # For display purposes only - don't use in calculations
                return (
                    closest_auction["security_type"],
                    closest_auction.get("security_term", "N/A"),
                    closest_auction["cusip"],
                    datetime.strptime(closest_auction["auction_date"], "%Y-%m-%d"),
                    0.0  # Placeholder yield
                )

        # Return the auction details
        return (
            closest_auction["security_type"],
            closest_auction.get("security_term", "N/A"),
            closest_auction["cusip"],
            datetime.strptime(closest_auction["auction_date"], "%Y-%m-%d"),
            float(closest_auction["high_yield"])
        )
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        print(f"URL attempted: {e.response.url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Treasury API: {e}")
        return None
    except (ValueError, KeyError) as e:
        print(f"Error parsing API response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add command line arguments
    parser.add_argument('--duration', default="20Y", 
                        help='Bond duration code (e.g., 13W, 10Y, 20Y). Default: 20Y')
    parser.add_argument('--date', help='Target date (MMDDYYYY format), default: today')
    
    args = parser.parse_args()
    
    # Normalize the duration code to uppercase
    duration_code = args.duration.upper()
    
    # Validate duration
    if duration_code not in DURATION_LABELS:
        valid_durations = ', '.join(sorted(DURATION_LABELS.keys()))
        print(f"Error: '{args.duration}' is not a valid duration code.")
        print(f"Valid duration codes: {valid_durations}")
        return 1
    
    # Handle date parameter
    if args.date:
        try:
            # Parse date in MMDDYYYY format
            target_date = datetime.strptime(args.date, "%m%d%Y")
        except ValueError:
            print(f"Error: Date '{args.date}' is not in the required format MMDDYYYY")
            return 1
    else:
        # Use current date if none provided
        target_date = datetime.now()
    
    # Fetch auction yield
    result = fetch_auction_yield(duration_code, target_date)
    
    # Exit if we couldn't get a valid yield
    if result is None:
        return 1
    
    sec_type, sec_term, cusip, auc_date, yield_value = result
    
    # Print the security details
    print(f"\nTreasury {sec_type} ({sec_term}):")
    print(f"  CUSIP: {cusip}")
    print(f"  Auction Date: {auc_date.strftime('%Y-%m-%d')}")
    print(f"  Yield: {yield_value:.3f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
