#!/usr/bin/env python3

import sys
import argparse
from datetime import datetime, timedelta
import requests

VERSION = "1.0"

DURATION_LABELS = {
    "2Y": "2-Year",     # 2-year Treasury Note
    "10Y": "10-Year",   # 10-year Treasury Note
    "20Y": "20-Year",   # 20-year Treasury Bond
    "30Y": "30-Year"    # 30-year Treasury Bond
}

DESCRIPTION = """Retrieve auction yield for a given bond duration and date.
Yields are retrieved from Treasury auction results via Fiscal Data Treasury API."""

EPILOG = """examples:
  %(prog)s --duration 20Y               # Most recent 20Y auction
  %(prog)s --duration 10Y --date 06012024
  %(prog)s -h                          # Show this help message
  %(prog)s -v                          # Show version"""

API_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
# Keep using the working endpoint
AUCTION_DATA_ENDPOINT = f"{API_BASE_URL}/v1/accounting/od/auctions_query"

def fetch_auction_yield(security_term_with_unit, target_date):
    """ Fetches auction yield for a given security term (e.g., '10-Year') 
    handling the variations in how terms are labeled in the API. """
    
    # Extract target year number from the term (e.g., 20 from "20-Year")
    target_year_str = security_term_with_unit.split('-')[0]
    try:
        target_year_num = int(target_year_str)
    except ValueError:
        print(f"Error: Could not parse year number from term '{security_term_with_unit}'")
        sys.exit(1)
    
    # Determine security type based on term
    # Generally, 2Y, 3Y, 5Y, 7Y, 10Y are Notes while 20Y, 30Y are Bonds
    expected_security_type = "Note" if target_year_num <= 10 else "Bond"
        
    try:
        # Format date for API query
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        # Request parameters - use a larger page size to ensure we capture all relevant auctions
        params = {
            "fields": "auction_date,security_type,security_term,high_yield,cusip",
            "sort": "-auction_date", # Sort by date descending
            "format": "json",
            "page[size]": "300"  # Larger page size to capture enough data
        }
        
        print(f"\nQuerying Treasury API for {security_term_with_unit} securities around {target_date_str}")
        r = requests.get(AUCTION_DATA_ENDPOINT, params=params)
        
        print(f"API URL: {r.url}")
        
        r.raise_for_status()
        data = r.json()
        
        if not data.get("data"):
            print(f"No auction data returned from API.")
            return None
        
        auctions = data.get("data", [])    
        print(f"\nReceived {len(auctions)} total auctions. Filtering for {security_term_with_unit} securities...")
        
        # Filter for relevant auctions based on security type and term pattern
        relevant_auctions = []
        for auction in auctions:
            auction_term = auction.get("security_term", "")
            auction_type = auction.get("security_type", "")
            
            # Skip if missing term or type
            if not auction_term or not auction_type:
                continue
                
            # Check if this auction matches our target term pattern
            is_match = False
            
            # Case 1: Exact match (e.g., "20-Year")
            if auction_term == security_term_with_unit:
                is_match = True
            # Case 2: Year-1 with months (e.g., "19-Year 10-Month", "19-Year 11-Month")
            elif auction_term.startswith(f"{target_year_num-1}-Year "):
                is_match = True
            # Case 3: Special case handling for reopenings that might have different labels
            elif (f"{target_year_num}" in auction_term) and auction_type == expected_security_type:
                is_match = True
                
            # Add to relevant auctions if it matches and has the expected security type
            if is_match and auction_type == expected_security_type:
                high_yield = auction.get("high_yield")
                if high_yield is not None:  # Only include auctions with yield data
                    relevant_auctions.append(auction)
        
        if not relevant_auctions:
            print(f"No matching {expected_security_type} auctions found with term pattern related to {security_term_with_unit}.")
            return None
            
        print(f"Found {len(relevant_auctions)} matching {expected_security_type} auctions. Finding closest to {target_date_str}...")
        
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

        # Return the auction details
        return (
            closest_auction["security_type"],
            closest_auction.get("security_term", "N/A"),  # Use term in place of description
            closest_auction["cusip"],
            datetime.strptime(closest_auction["auction_date"], "%Y-%m-%d"),
            float(closest_auction["high_yield"])
        )
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        print(f"URL attempted: {e.response.url}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Treasury API: {e}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"Error parsing API response: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", default="20Y", 
                       help="Bond duration (e.g., 20Y, 10Y, default: 20Y)")
    parser.add_argument("--date", help="Target date (MMDDYYYY), default: most recent", 
                       default=None)
    parser.add_argument("-v", "--version", action="version", 
                       version=f"%(prog)s {VERSION}")
    
    args = parser.parse_args()
    
    # Use the full duration label (e.g., "10-Year") for the API call
    security_term_label = DURATION_LABELS.get(args.duration.upper())
    if not security_term_label:
        print(f"Invalid duration key: {args.duration}")
        print(f"Valid duration keys: {', '.join(DURATION_LABELS.keys())}")
        sys.exit(1)
    
    # Handle date parameter
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%m%d%Y")
        except ValueError:
            print(f"Invalid date format: {args.date}. Use MMDDYYYY")
            sys.exit(1)
    else:
        # Use current date to get most recent auction
        target_date = datetime.now()
        
    # Pass the full label (e.g., "10-Year") to the fetch function
    result = fetch_auction_yield(security_term_label, target_date)
    if result:
        security_type, security_desc, cusip, auction_date, yield_value = result
        print(f"Security: {security_type} {security_desc} (CUSIP: {cusip})")
        print(f"Auction Date: {auction_date.strftime('%Y-%m-%d')}")
        print(f"High Yield: {yield_value:.3f}%")
    else:
        print("No yield data found")
        sys.exit(1)

if __name__ == "__main__":
    main()
