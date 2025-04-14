#!/usr/bin/env python3
import requests
from datetime import datetime
import os
import sys
import logging
import re
import json
import argparse

# Configure logging with option to disable file logging
def setup_logging(log_to_file=True):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.append(logging.FileHandler('auction_calendar.log'))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

VERSION = "1.01"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate a calendar of Treasury security auctions.')
    
    # Year argument
    parser.add_argument('year', nargs='?', type=int, default=datetime.now().year,
                      help='Year to generate calendar for (default: current year)')
    
    # Filter mode group
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--minimum', action='store_true',
                           help='Include only minimum security types (5-Year TIPS, 10-Year TIPS, '
                                '10-Year Note, 20-Year Bond)')
    filter_group.add_argument('--standard', action='store_true',
                           help='Include standard security types (default): 5-Year TIPS, 10-Year TIPS, '
                                '10-Year Note, 20-Year Bond, 5-Year Note, 30-Year Bond')
    filter_group.add_argument('--all', action='store_true',
                           help='Include all Treasury auctions')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with additional logging and save raw API response')
    parser.add_argument('--no-log-file', action='store_true',
                       help='Disable logging to file')
    
    # Version and help
    parser.add_argument('-v', '--version', action='version',
                      version=f'Treasury Auction Calendar Generator v{VERSION}')
    
    args = parser.parse_args()
    return args

def fetch_auctions(year, debug=False):
    """
    Fetch auction data from Treasury API
    
    Args:
        year: Year to fetch auctions for
        debug: Whether to save raw API response and print debug info
        
    Returns:
        List of processed auction records
    """
    logger.info(f"Fetching auctions for year {year}...")
    
    # Define the date range for the whole year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    params = {
        "filter": f"auction_date:gte:{start_date},auction_date:lte:{end_date}",
        "sort": "auction_date",
        "page[size]": "1000"
    }
    
    if debug:
        print(f"API parameters: {params}")
    
    try:
        # API URL for Treasury auctions
        api_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
        
        # Headers for requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        if debug:
            print("Making API request...")
        
        response = requests.get(api_url, params=params, headers=headers)
        
        if debug:
            print(f"API response status code: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        if debug:
            print("Successfully parsed JSON response")
            # Save the raw API response for debugging
            with open(f"api_response_{year}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved raw API response to api_response_{year}.json")
        
        if "data" not in data:
            logger.warning(f"No data field found in API response")
            if debug:
                print(f"API response structure: {list(data.keys())}")
            return []
            
        auctions_data = data["data"]
        logger.info(f"Successfully retrieved {len(auctions_data)} auction records")
        
        # Process the auction data
        processed_auctions = []
        for auction in auctions_data:
            try:
                # Extract relevant fields from API response
                auction_date = auction.get("auction_date")
                if not auction_date:
                    continue
                    
                # Parse the date to get the year
                date_obj = datetime.strptime(auction_date, "%Y-%m-%d")
                
                security_type = f"{auction.get('security_type', '')} {auction.get('security_term', '')}"
                
                processed_auction = {
                    'security_type': security_type.strip(),
                    'auction_date': auction_date,
                    'year': date_obj.year,
                    'is_announced': True,
                    'details': f"CUSIP: {auction.get('cusip', 'N/A')}, Offering Amount: {auction.get('offering_amt', 'N/A')}, Issue Date: {auction.get('issue_date', 'N/A')}"
                }
                processed_auctions.append(processed_auction)
            except Exception as e:
                logger.error(f"Error processing auction record: {e}")
                if debug:
                    print(f"Error processing auction record: {e}")
                
        return processed_auctions
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if debug:
            print(f"API request failed: {e}")
        
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            if debug:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text[:500]}...")
        
        return []

def format_ics(events):
    """Format events into iCalendar format"""
    ics = ["BEGIN:VCALENDAR", "VERSION:2.0", "CALSCALE:GREGORIAN", "METHOD:PUBLISH"]
    for e in events:
        ics.append("BEGIN:VEVENT")
        ics.append(f"SUMMARY:{e['title']}")
        ics.append(f"DTSTART;VALUE=DATE:{e['date']}")
        ics.append(f"DESCRIPTION:{e['desc']}")
        ics.append("END:VEVENT")
    ics.append("END:VCALENDAR")
    return "\n".join(ics)

def main():
    """Main function to generate the auction calendar"""
    # Parse command line arguments
    args = parse_arguments()
    year = args.year
    include_all = args.all
    minimum_only = args.minimum
    debug_mode = args.debug
    standard = not (include_all or minimum_only)  # Standard is the default if no other option is chosen
    output_file = f"auction_calendar_{year}.ics"
    
    # Setup logging based on command-line options
    global logger
    logger = setup_logging(not args.no_log_file)
    
    print(f"Generating calendar for year: {year}")
    print(f"Output will be saved to: {output_file}")
    
    if include_all:
        print(f"Filter mode: ALL auctions")
    elif minimum_only:
        print(f"Filter mode: MINIMUM securities only")
    else:
        print(f"Filter mode: STANDARD securities (default)")
    
    # Define minimum security types (original list)
    minimum_security_types = [
        {"name": "5-Year TIPS", "patterns": ["tips.*5.year", "5.year.*tips"]},
        {"name": "10-Year TIPS", "patterns": ["tips.*10.year", "10.year.*tips"]},
        {"name": "10-Year Note", "patterns": ["note.*10.year", "10.year.*note"]},
        {"name": "20-Year Bond", "patterns": ["bond.*20.year", "20.year.*bond", "bond.*19.year.*10.month"]}
    ]
    
    # Define standard security types (original plus 5-Year Note and 30-Year Bond)
    standard_security_types = minimum_security_types + [
        {"name": "5-Year Note", "patterns": ["note.*5.year", "5.year.*note"]},
        {"name": "30-Year Bond", "patterns": ["bond.*30.year", "30.year.*bond", "bond.*29.year.*10.month"]}
    ]
    
    print("Fetching Treasury auction data...")
    auctions = fetch_auctions(year, debug=debug_mode)
    print(f"Retrieved {len(auctions)} auctions in total")
    
    events = []
    
    for auction in auctions:
        security_type = auction['security_type'].lower()
        if debug_mode:
            print(f"Processing: {auction['security_type']} on {auction['auction_date']}")
        
        # If --all flag is used, include all auctions
        if include_all:
            try:
                date_str = auction['auction_date'].replace('-', '')
                title = f"{auction['security_type']} Auction"
                status = "CONFIRMED" if auction.get('is_announced', False) else "TENTATIVE"
                desc = (
                    f"Status: {status}\n"
                    f"Details: {auction.get('details', 'N/A')}"
                )
                events.append({"title": title, "date": date_str, "desc": desc})
                if debug_mode:
                    print(f"Added event: {title}")
            except Exception as e:
                logger.error(f"Error processing auction {auction}: {e}")
                if debug_mode:
                    print(f"Error processing auction {auction}: {e}")
            
            # Continue to next auction
            continue
            
        # Determine which security types to filter by based on command line options
        if minimum_only:
            security_types_to_use = minimum_security_types
            filter_description = "minimum securities"
        else:  # Standard is the default
            security_types_to_use = standard_security_types
            filter_description = "standard securities"
        
        # Check if this auction matches any of our security types of interest
        for interest_type in security_types_to_use:
            matched = False
            for pattern in interest_type["patterns"]:
                if re.search(pattern, security_type):
                    try:
                        date_str = auction['auction_date'].replace('-', '')
                        # Use the friendly name from our config
                        title = f"{interest_type['name']} Auction"
                        status = "CONFIRMED" if auction.get('is_announced', False) else "TENTATIVE"
                        desc = (
                            f"Status: {status}\n"
                            f"Original Security Type: {auction['security_type']}\n"
                            f"Details: {auction.get('details', 'N/A')}"
                        )
                        events.append({"title": title, "date": date_str, "desc": desc})
                        if debug_mode:
                            print(f"Added event: {title}")
                        matched = True
                        break
                    except Exception as e:
                        logger.error(f"Error processing auction {auction}: {e}")
                        if debug_mode:
                            print(f"Error processing auction {auction}: {e}")
                
            if matched:
                break
    
    print(f"Found {len(events)} events matching {filter_description if not include_all else 'all auctions'}")
    
    if events:
        ics_text = format_ics(events)
        with open(output_file, "w") as f:
            f.write(ics_text)
        print(f"✅ Calendar saved as: {output_file} with {len(events)} events")
    else:
        print(f"No relevant auctions found for {year}")
        # Create an empty calendar file
        with open(output_file, "w") as f:
            f.write(format_ics([]))
        print(f"✅ Empty calendar file created: {output_file}")

if __name__ == "__main__":
    main()
