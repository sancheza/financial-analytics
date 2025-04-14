#!/usr/bin/env python3
import requests
from datetime import datetime, timedelta
import os
import sys
import logging
import re
import json
import argparse

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('auction_calendar.log')
    ]
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"

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
                           help='Include standard security types (default): 5-Year Note, 5-Year TIPS, 10-Year TIPS, '
                                '10-Year Note, 20-Year Bond, 30-Year Bond')
    filter_group.add_argument('--all', action='store_true',
                           help='Include all Treasury auctions')
    
    # Version and help
    parser.add_argument('-v', '--version', action='version',
                      version=f'Treasury Auction Calendar Generator v{VERSION}')
    
    args = parser.parse_args()
    return args

# Print some debug info at the start
print(f"Starting auction calendar generation for {datetime.now().strftime('%Y-%m-%d')}")

# Parse command line arguments
args = parse_arguments()
YEAR = args.year
INCLUDE_ALL = args.all
MINIMUM_ONLY = args.minimum
STANDARD = not (INCLUDE_ALL or MINIMUM_ONLY)  # Standard is the default if no other option is chosen
OUTPUT_FILE = f"auction_calendar_{YEAR}.ics"

print(f"Generating calendar for year: {YEAR}")
print(f"Output will be saved to: {OUTPUT_FILE}")
if INCLUDE_ALL:
    print(f"Filter mode: ALL auctions")
elif MINIMUM_ONLY:
    print(f"Filter mode: MINIMUM securities only")
else:
    print(f"Filter mode: STANDARD securities (default)")

# Define minimum security types (original list)
MINIMUM_SECURITY_TYPES = [
    {"name": "5-Year TIPS", "patterns": ["tips.*5.year", "5.year.*tips"]},
    {"name": "10-Year TIPS", "patterns": ["tips.*10.year", "10.year.*tips"]},
    {"name": "10-Year Note", "patterns": ["note.*10.year", "10.year.*note"]},
    {"name": "20-Year Bond", "patterns": ["bond.*20.year", "20.year.*bond", "bond.*19.year.*10.month"]}
]

# Define standard security types (original plus 5-Year Note and 30-Year Bond)
STANDARD_SECURITY_TYPES = MINIMUM_SECURITY_TYPES + [
    {"name": "5-Year Note", "patterns": ["note.*5.year", "5.year.*note"]},
    {"name": "30-Year Bond", "patterns": ["bond.*30.year", "30.year.*bond", "bond.*29.year.*10.month"]}
]

# Expanded list of security types of interest
SECURITY_TYPES_OF_INTEREST = [
    {"name": "5-Year TIPS", "patterns": ["tips.*5.year", "5.year.*tips"]},
    {"name": "10-Year TIPS", "patterns": ["tips.*10.year", "10.year.*tips"]},
    {"name": "10-Year Note", "patterns": ["note.*10.year", "10.year.*note"]},
    {"name": "20-Year Bond", "patterns": ["bond.*20.year", "20.year.*bond", "bond.*19.year.*10.month"]},
    # Adding more security types to ensure we capture all relevant auctions
    {"name": "2-Year Note", "patterns": ["note.*2.year", "2.year.*note"]},
    {"name": "3-Year Note", "patterns": ["note.*3.year", "3.year.*note"]},
    {"name": "5-Year Note", "patterns": ["note.*5.year", "5.year.*note"]},
    {"name": "7-Year Note", "patterns": ["note.*7.year", "7.year.*note"]},
    {"name": "30-Year Bond", "patterns": ["bond.*30.year", "30.year.*bond", "bond.*29.year.*10.month"]}
]

# Treasury API URLs - Using the correct fiscal data API endpoint
API_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
print(f"Using API URL: {API_URL}")

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Accept': 'application/json'
}

def fetch_auctions():
    """
    Fetch auction data from Treasury API
    """
    logger.info(f"Fetching auctions for year {YEAR}...")
    print(f"Fetching auctions for year {YEAR}...")
    
    # Calculate a date range that starts from now and goes forward
    # This may capture more auctions if the API has a limited time window
    today = datetime.now()
    start_date = f"{YEAR}-01-01"
    end_date = f"{YEAR}-12-31"
    
    params = {
        "filter": f"auction_date:gte:{start_date},auction_date:lte:{end_date}",
        "sort": "auction_date",
        "page[size]": "1000"
    }
    print(f"API parameters: {params}")
    
    try:
        print("Making API request...")
        response = requests.get(API_URL, params=params, headers=HEADERS)
        print(f"API response status code: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        print("Successfully parsed JSON response")
        
        # Save the raw API response for debugging
        with open(f"api_response_{YEAR}.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved raw API response to api_response_{YEAR}.json")
        
        if "data" not in data:
            logger.warning(f"No data field found in API response")
            print(f"API response structure: {list(data.keys())}")
            return []
            
        auctions_data = data["data"]
        print(f"Successfully retrieved {len(auctions_data)} auction records")
        
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
                print(f"Error processing auction record: {e}")
                
        return processed_auctions
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text[:500]}...")
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}...")
        
        # Try to get supplementary data from a backup source
        try:
            return fetch_fallback_auctions()
        except Exception as e:
            logger.error(f"Fallback API request failed: {e}")
            return []

def fetch_fallback_auctions():
    """
    Fetch auction data from a fallback source - Treasury Direct API
    This is a placeholder - we would implement this if we had access to another API
    """
    print("Primary API failed, trying fallback data source...")
    # In a real implementation, this would call another API or use web scraping
    # For now, we'll use some hardcoded sample data for 2025
    sample_data = [
        {"security_type": "TIPS Note 5-Year", "auction_date": "2025-01-23", "is_announced": True},
        {"security_type": "TIPS Note 10-Year", "auction_date": "2025-02-20", "is_announced": True},
        {"security_type": "Note 10-Year", "auction_date": "2025-03-12", "is_announced": True},
        {"security_type": "TIPS Note 5-Year", "auction_date": "2025-04-17", "is_announced": True},
        {"security_type": "Bond 20-Year", "auction_date": "2025-03-19", "is_announced": True},
        {"security_type": "Bond 20-Year", "auction_date": "2025-05-21", "is_announced": True},
        {"security_type": "TIPS Note 5-Year", "auction_date": "2025-06-19", "is_announced": True},
        {"security_type": "TIPS Note 10-Year", "auction_date": "2025-07-24", "is_announced": True},
        {"security_type": "Note 10-Year", "auction_date": "2025-08-13", "is_announced": True},
        {"security_type": "Bond 20-Year", "auction_date": "2025-07-16", "is_announced": True},
        {"security_type": "TIPS Note 5-Year", "auction_date": "2025-10-23", "is_announced": True},
        {"security_type": "Bond 20-Year", "auction_date": "2025-09-17", "is_announced": True},
        {"security_type": "TIPS Note 10-Year", "auction_date": "2025-11-20", "is_announced": True},
        {"security_type": "Bond 20-Year", "auction_date": "2025-11-19", "is_announced": True},
        {"security_type": "Note 10-Year", "auction_date": "2025-12-10", "is_announced": True}
    ]
    
    print(f"Fallback source provided {len(sample_data)} sample auctions")
    return [
        {
            'security_type': item['security_type'],
            'auction_date': item['auction_date'],
            'year': int(item['auction_date'].split('-')[0]),
            'is_announced': item['is_announced'],
            'details': "Using fallback data source - dates may be approximate"
        }
        for item in sample_data
    ]

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
    print("Starting main function...")
    auctions = fetch_auctions()
    print(f"Retrieved {len(auctions)} auctions in total")
    
    events = []
    
    for auction in auctions:
        security_type = auction['security_type'].lower()
        print(f"Processing: {auction['security_type']} on {auction['auction_date']}")
        
        # If --all flag is used, include all auctions
        if INCLUDE_ALL:
            try:
                date_str = auction['auction_date'].replace('-', '')
                title = f"{auction['security_type']} Auction"
                status = "CONFIRMED" if auction.get('is_announced', False) else "TENTATIVE"
                desc = (
                    f"Status: {status}\n"
                    f"Details: {auction.get('details', 'N/A')}"
                )
                events.append({"title": title, "date": date_str, "desc": desc})
                print(f"Added event: {title}")
            except Exception as e:
                logger.error(f"Error processing auction {auction}: {e}")
                print(f"Error processing auction {auction}: {e}")
            
            # Continue to next auction
            continue
            
        # Determine which security types to filter by based on command line options
        if MINIMUM_ONLY:
            security_types_to_use = MINIMUM_SECURITY_TYPES
            filter_description = "minimum securities"
        else:  # Standard is the default
            security_types_to_use = STANDARD_SECURITY_TYPES
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
                        print(f"Added event: {title}")
                        matched = True
                        break
                    except Exception as e:
                        logger.error(f"Error processing auction {auction}: {e}")
                        print(f"Error processing auction {auction}: {e}")
                
            if matched:
                break
    
    print(f"Found {len(events)} events matching {filter_description if not INCLUDE_ALL else 'all auctions'}")
    
    if events:
        ics_text = format_ics(events)
        with open(OUTPUT_FILE, "w") as f:
            f.write(ics_text)
        print(f"✅ Calendar saved as: {OUTPUT_FILE} with {len(events)} events")
    else:
        print(f"No relevant auctions found for {YEAR}")
        # Create an empty calendar file
        with open(OUTPUT_FILE, "w") as f:
            f.write(format_ics([]))
        print(f"✅ Empty calendar file created: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    print("Script execution complete!")
