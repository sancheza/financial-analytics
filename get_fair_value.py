#!/usr/bin/env python3

import argparse
import requests
import re
from bs4 import BeautifulSoup
import logging
import time
import json

VERSION = "1.05"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_alphaspread_fair_value(ticker):
    url = f"https://www.alphaspread.com/security/nasdaq/{ticker.lower()}/summary"
    
    try:
        # FlareSolverr endpoint
        solver_url = "http://localhost:8191/v1"
        
        # Request payload for FlareSolverr
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 60000
        }
        
        resp = requests.post(solver_url, json=payload)
        resp.raise_for_status()
        
        result = resp.json()
        
        if result.get("status") != "ok":
            raise Exception(f"FlareSolverr error: {result.get('message')}")
            
        html_content = result["solution"]["response"]
        soup = BeautifulSoup(html_content, "html.parser")
        
        logger.debug(f"Response status code: {result['solution']['status']}")
        
        # Look for intrinsic value div with more flexible selection
        intrinsic_div = soup.select_one("div[class*='intrinsic-value']")
        if intrinsic_div:
            intrinsic_value = intrinsic_div.text.strip().replace("USD", "").strip()
            logger.info(f"Found intrinsic value: {intrinsic_value}")
            return f"AlphaSpread: Intrinsic Fair Value = ${intrinsic_value}"
        else:
            logger.warning("Intrinsic value element not found in the page")
            return "AlphaSpread: Intrinsic value element not found"
    except Exception as e:
        logger.error(f"Error fetching AlphaSpread data: {str(e)}")
        return f"AlphaSpread: Error - {e}"

def get_valueinvesting_io_fair_value(ticker):
    url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
    
    try:
        # FlareSolverr endpoint
        solver_url = "http://localhost:8191/v1"
        
        # Create a new session
        session_payload = {
            "cmd": "sessions.create",
            "maxTimeout": 60000
        }
        
        session_resp = requests.post(solver_url, json=session_payload)
        session_result = session_resp.json()
        
        if session_result.get("status") != "ok":
            raise Exception(f"Failed to create FlareSolverr session: {session_result.get('message')}")
            
        session_id = session_result["session"]
        
        try:
            # Request payload for FlareSolverr using the session
            payload = {
                "cmd": "request.get",
                "url": url,
                "maxTimeout": 120000,  # Increased timeout
                "session": session_id
            }
            
            logger.info(f"Fetching {url} via FlareSolverr")
            resp = requests.post(solver_url, json=payload)
            resp.raise_for_status()
            
            result = resp.json()
            
            if result.get("status") != "ok":
                raise Exception(f"FlareSolverr error: {result.get('message')}")
                
            # Get the HTML content from the solution
            html_content = result["solution"]["response"]
            
            # Save response for debugging
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            if result["solution"]["status"] != 200:
                raise Exception(f"Bad response status: {result['solution']['status']}")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for elements containing fair value information
            selectors = [
                'div.norm', 
                'div[class*="value"]', 
                'span[class*="value"]',
                'div[class*="price"]',
                'span[class*="price"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if 'USD' in text:
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            fair_value = match.group(1)
                            return f"ValueInvesting.io: Fair Value = ${fair_value}"
            
            raise Exception("Fair value not found in page content")
            
        finally:
            # Destroy the session
            destroy_payload = {
                "cmd": "sessions.destroy",
                "session": session_id
            }
            requests.post(solver_url, json=destroy_payload)
            
    except Exception as e:
        logger.error(f"Error fetching ValueInvesting.io data: {str(e)}")
        if 'result' in locals():
            logger.error(f"FlareSolverr response: {json.dumps(result, indent=2)}")
        return f"ValueInvesting.io: Error - {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description="Get fair value estimates for a stock from AlphaSpread and ValueInvesting.io"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., MRNA)")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    args = parser.parse_args()

    print(f"\n🔍 Fetching fair value estimates for: {args.ticker.upper()}\n")
    
    # Try AlphaSpread first
    alphaspread_result = get_alphaspread_fair_value(args.ticker)
    print(alphaspread_result)
    
    # Add a small delay between requests
    time.sleep(2)
    
    # Then try ValueInvesting.io
    valueinvesting_result = get_valueinvesting_io_fair_value(args.ticker)
    print(valueinvesting_result)
    
    print("Simply Wall St: (not yet implemented)\n")

if __name__ == "__main__":
    main()
