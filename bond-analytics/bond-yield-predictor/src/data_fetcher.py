#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fetcher module for retrieving Treasury bond data from FRED and TreasuryDirect.
"""

import os
import requests
import json
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union, Tuple
from src.utils import setup_logger
import pandas_datareader.data as web
from datetime import datetime, timedelta

# Configure logging
logger = setup_logger(__name__, "../data/log.txt")

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    logger.error("FRED API key not found in environment variables")

# Bond type to FRED series ID mapping
FRED_SERIES_MAP = {
    '1M': 'DGS1MO',
    '3M': 'DGS3MO',
    '6M': 'DGS6MO',
    '1Y': 'DGS1',
    '2Y': 'DGS2',
    '5Y': 'DGS5',
    '7Y': 'DGS7',
    '10Y': 'DGS10',
    '20Y': 'DGS20',
    '30Y': 'DGS30'
}

# Mapping from bond type to typical security term used by the Treasury API
TREASURY_TERM_MAP = {
    '1M': '4-Week',
    '3M': '13-Week',
    '6M': '26-Week',
    '1Y': '52-Week',
    '2Y': '2-Year',
    '3Y': '3-Year',
    '5Y': '5-Year',
    '7Y': '7-Year',
    '10Y': '10-Year',
    '20Y': '20-Year',
    '30Y': '30-Year'
}

# Mapping from bond type to security type in Treasury API
TREASURY_TYPE_MAP = {
    '1M': 'Bill',
    '3M': 'Bill',
    '6M': 'Bill',
    '1Y': 'Bill',
    '2Y': 'Note',
    '3Y': 'Note',
    '5Y': 'Note',
    '7Y': 'Note',
    '10Y': 'Note',
    '20Y': 'Bond',
    '30Y': 'Bond'
}

class DataFetcher:
    """Class to handle data fetching operations."""
    
    def __init__(self, cache_dir: str = "../data/json"):
        """
        Initialize the DataFetcher.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.treasury_api_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_fred_data(self, series_id: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from FRED API for a given series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing the yield data
        """
        if not FRED_API_KEY:
            raise ValueError("FRED API key not found")
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if not start_date:
            # Default to 1 year of data
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            response = requests.get(self.fred_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            observations = data.get('observations', [])
            if not observations:
                logger.warning(f"No data found for series {series_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(observations)
            # Convert date to datetime and value to float
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Filter out missing values
            df = df.dropna(subset=['value'])
            
            return df[['date', 'value']]
        
        except requests.RequestException as e:
            logger.error(f"Error fetching data from FRED: {str(e)}")
            return pd.DataFrame()
    
    def get_treasury_data(self, bond_type: str, force_update: bool = False, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get Treasury yield data for a specific bond type.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            end_date: Optional end date for data retrieval (for back-testing)
            
        Returns:
            DataFrame with yield data
        """
        if bond_type not in FRED_SERIES_MAP:
            raise ValueError(f"Unsupported bond type: {bond_type}")
        
        series_id = FRED_SERIES_MAP[bond_type]
        cache_file = os.path.join(self.cache_dir, f"{bond_type}.json")
        
        # Check cache first if not forcing update
        if not force_update and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                last_fetch = datetime.fromisoformat(cached_data['timestamp'])
                now = datetime.now()
                
                # Use cache if it's less than a day old
                if (now - last_fetch).days < 1 and 'data' in cached_data:
                    logger.info(f"Using cached data for {bond_type}")
                    df = pd.DataFrame(cached_data['data'])
                    
                    # Apply end_date filter if specified (for back-testing)
                    if end_date and 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df[df['date'] <= end_date]
                    
                    return df
            
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading cache file for {bond_type}: {str(e)}")
        
        # Fetch fresh data from FRED
        logger.info(f"Fetching fresh data for {bond_type}")
        df = self._get_fred_data(series_id, end_date=end_date)
        
        if df.empty:
            return df
        
        # Prepare data for caching - convert dates to strings for JSON serialization
        data_to_cache = {
            'bond_type': bond_type,
            'timestamp': datetime.now().isoformat(),
            'data': df.assign(date=df['date'].dt.strftime('%Y-%m-%d')).to_dict('records')
        }
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(data_to_cache, f, indent=2)
                
            logger.info(f"Saved {bond_type} data to cache")
        except Exception as e:
            logger.error(f"Error saving {bond_type} data to cache: {str(e)}")
        
        return df
    
    def get_treasury_direct_auctions(self, bond_type: str, force_update: bool = False) -> List[Dict[str, Any]]:
        """
        Get historical auction data from TreasuryDirect website.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            
        Returns:
            List of auction data dictionaries with real historical data
        """
        cache_file = os.path.join(self.cache_dir, f"{bond_type}_auctions.json")
        
        # Check cache first if not forcing update
        if not force_update and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                last_fetch = datetime.fromisoformat(cached_data['timestamp'])
                now = datetime.now()
                
                # Use cache if it's less than a week old
                if (now - last_fetch).days < 7 and 'auctions' in cached_data:
                    logger.info(f"Using cached auction data for {bond_type}")
                    return cached_data['auctions']
            
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading auction cache file for {bond_type}: {str(e)}")
        
        logger.info(f"Fetching real TreasuryDirect auction data for {bond_type}")
        
        try:
            # Map bond type to TreasuryDirect security type and term
            security_type = TREASURY_TYPE_MAP.get(bond_type)
            security_term = TREASURY_TERM_MAP.get(bond_type)
            
            if not security_type or not security_term:
                logger.warning(f"No TreasuryDirect mapping for {bond_type}")
                return []
            
            # TreasuryDirect auction results API endpoint
            url = "https://www.treasurydirect.gov/GA-FI/FedInvest/todaySecurityPriceData"
            
            # Parameters for the request
            params = {
                "pageNum": "1",
                "security_type": security_type,  # "Note", "Bond", "Bill"
                "security_term": security_term,  # "10-Year", "20-Year", etc.
                "recordsPerPage": "20"
            }
            
            # Make request to TreasuryDirect
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get("https://www.treasurydirect.gov/TA_WS/securities/auctioned", headers=headers)
            response.raise_for_status()
            
            # Parse the results - actual structure depends on TreasuryDirect API
            all_auctions = response.json()
            
            # Filter to get relevant auctions matching our bond type
            term_key = security_term.lower().replace('-', '')
            type_key = security_type.lower()
            
            # Special case for 20Y bonds which might have different naming
            if bond_type == '20Y':
                matching_auctions = [
                    a for a in all_auctions 
                    if (a.get('securityType', '').lower() == type_key and
                        ('20' in a.get('securityTerm', '').lower() or 
                         '19' in a.get('securityTerm', '').lower()))
                ]
            else:
                # Standard matching for other bond types
                matching_auctions = [
                    a for a in all_auctions
                    if (a.get('securityType', '').lower() == type_key and
                        term_key in a.get('securityTerm', '').lower().replace('-', '').replace(' ', ''))
                ]
            
            # Convert to standard format
            auctions = []
            for auction in matching_auctions:
                try:
                    # Parse date correctly from ISO format
                    date_str = auction.get('auctionDate', '')
                    if date_str:
                        # Format date as YYYY-MM-DD
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                    else:
                        continue
                    
                    # Get yield information
                    high_yield = auction.get('highRate') or auction.get('highYield') or auction.get('highDiscountRate')
                    
                    # Build auction record
                    auction_record = {
                        'date': formatted_date,
                        'cusip': auction.get('cusip', ''),
                        'high_yield': str(high_yield) if high_yield else '0.0'
                    }
                    auctions.append(auction_record)
                except Exception as e:
                    logger.warning(f"Error processing auction: {e}")
                    continue
            
            # If we couldn't get data from TreasuryDirect API, try web scraping as fallback
            if not auctions:
                logger.warning("No auctions found from API, trying web scraping...")
                auctions = self._scrape_treasury_direct_auctions(bond_type)
            
            # Sort by date (newest first)
            auctions.sort(key=lambda x: x['date'], reverse=True)
            
            # Save to cache
            data_to_cache = {
                'bond_type': bond_type,
                'timestamp': datetime.now().isoformat(),
                'auctions': auctions
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data_to_cache, f, indent=2)
            
            logger.info(f"Saved real {bond_type} auction data to cache: {len(auctions)} auctions")
            return auctions
            
        except requests.RequestException as e:
            logger.error(f"Error fetching auction data from TreasuryDirect: {str(e)}")
            
            # Fallback to web scraping if API request fails
            logger.info(f"Trying web scraping as fallback...")
            auctions = self._scrape_treasury_direct_auctions(bond_type)
            
            if auctions:
                # Save to cache
                data_to_cache = {
                    'bond_type': bond_type,
                    'timestamp': datetime.now().isoformat(),
                    'auctions': auctions
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(data_to_cache, f, indent=2)
                
                logger.info(f"Saved scraped {bond_type} auction data to cache: {len(auctions)} auctions")
            
            return auctions
        
        except Exception as e:
            logger.error(f"Error parsing auction data: {str(e)}")
            return []
    
    def _scrape_treasury_direct_auctions(self, bond_type: str) -> List[Dict[str, Any]]:
        """
        Scrape historical auction data from TreasuryDirect website as fallback.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            
        Returns:
            List of auction data dictionaries with real historical data
        """
        logger.info(f"Attempting to scrape TreasuryDirect auctions for {bond_type}")
        auctions = []
        
        try:
            # Convert bond type to parameters needed for TreasuryDirect
            security_type = TREASURY_TYPE_MAP.get(bond_type)
            security_term = TREASURY_TERM_MAP.get(bond_type)
            
            if not security_type or not security_term:
                logger.warning(f"No TreasuryDirect mapping for {bond_type}")
                return []
            
            # Web scraping from TreasuryDirect auction archive page
            base_url = "https://www.treasurydirect.gov/auctions/announcements-data-results/"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # HTML page from TreasuryDirect with results
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for auction results table
            # This requires analyzing the structure of the TreasuryDirect website
            # Note: Website structure may change, requiring updates to this parsing logic
            
            # Find the table rows with auction data
            table = soup.find('table', {'class': 'auctions-table'})
            if not table:
                logger.warning("Couldn't find auction results table on TreasuryDirect")
                return []
                
            rows = table.find_all('tr')
            
            # Process each row
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) < 6:
                    continue
                
                # Extract relevant data from columns
                auction_security_type = cols[0].get_text().strip()
                auction_term = cols[1].get_text().strip()
                auction_date = cols[2].get_text().strip()
                cusip = cols[3].get_text().strip()
                high_yield = cols[5].get_text().strip().replace('%', '')
                
                # Check if this row matches our bond type
                is_match = False
                
                if security_type.lower() in auction_security_type.lower():
                    if bond_type == '20Y':
                        if '20' in auction_term or '19' in auction_term:
                            is_match = True
                    else:
                        term_digit = security_term.split('-')[0]
                        if term_digit in auction_term:
                            is_match = True
                
                if is_match:
                    # Parse date
                    try:
                        date_obj = datetime.strptime(auction_date, '%m/%d/%Y')
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        logger.warning(f"Invalid auction date format: {auction_date}")
                        continue
                    
                    # Create auction record
                    auction_record = {
                        'date': formatted_date,
                        'cusip': cusip,
                        'high_yield': high_yield
                    }
                    auctions.append(auction_record)
            
            # Fallback to the Treasury fiscal data API if web scraping didn't work
            if not auctions:
                logger.info("Web scraping didn't find auctions, trying Treasury fiscal data API...")
                auctions = self._fetch_treasury_fiscal_api_auctions(bond_type)
                
            return auctions
            
        except Exception as e:
            logger.error(f"Error scraping TreasuryDirect auction data: {str(e)}")
            return []
    
    def _fetch_treasury_fiscal_api_auctions(self, bond_type: str) -> List[Dict[str, Any]]:
        """
        Fetch auction data from Treasury Fiscal Data API as another fallback.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            
        Returns:
            List of auction data dictionaries
        """
        auctions = []
        
        try:
            # Convert bond type to parameters needed for Treasury API
            security_type = TREASURY_TYPE_MAP.get(bond_type)
            security_term = TREASURY_TERM_MAP.get(bond_type)
            
            if not security_type or not security_term:
                logger.warning(f"No Treasury API mapping for {bond_type}")
                return []
            
            # Treasury Fiscal Data API - auction results endpoint
            api_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/auction_data"
            
            # Build term parameter based on bond type
            term_param = ""
            
            # Specific handling for each bond type
            if bond_type == '20Y':
                term_param = "20-YEAR"
            elif bond_type == '30Y':
                term_param = "30-YEAR"
            elif bond_type == '10Y':
                term_param = "10-YEAR"
            elif bond_type == '7Y':
                term_param = "7-YEAR"
            elif bond_type == '5Y':
                term_param = "5-YEAR"
            elif bond_type == '2Y':
                term_param = "2-YEAR"
            elif bond_type == '1Y':
                term_param = "52-WEEK"
            elif bond_type == '6M':
                term_param = "26-WEEK"
            elif bond_type == '3M':
                term_param = "13-WEEK"
            elif bond_type == '1M':
                term_param = "4-WEEK"
                
            # Parameters for the API request
            params = {
                "security_term": term_param,
                "security_type": security_type.upper(),
                "sort": "-auction_date",  # Sort by auction date descending
                "format": "json",
                "page[size]": "20"  # Get 20 results per page
            }
            
            # Make request to API
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process the results
            api_auctions = data.get('data', [])
            
            for auction in api_auctions:
                try:
                    # Get auction date
                    auction_date = auction.get('auction_date')
                    if not auction_date:
                        continue
                    
                    # Get yield - format depends on bond type
                    if security_type.lower() == 'bill':
                        # For bills, use the discount rate
                        high_yield = auction.get('high_discount_rate')
                    else:
                        # For notes and bonds, use the high yield
                        high_yield = auction.get('high_yield')
                    
                    if not high_yield:
                        continue
                    
                    # Create auction record
                    auction_record = {
                        'date': auction_date,
                        'cusip': auction.get('cusip', ''),
                        'high_yield': str(high_yield)
                    }
                    
                    auctions.append(auction_record)
                    
                except Exception as e:
                    logger.warning(f"Error processing API auction data: {str(e)}")
                    continue
            
            return auctions
            
        except Exception as e:
            logger.error(f"Error fetching data from Treasury Fiscal API: {str(e)}")
            return []

    def get_fred_data(self, series_ids: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch economic data from FRED."""
        self.logger.info(f"Fetching FRED data for {series_ids} from {start_date} to {end_date}")
        try:
            fred_data = web.DataReader(series_ids, 'fred', start_date, end_date)
            fred_data.index.name = 'date' # Ensure index name is 'date'
            # Forward fill to handle missing values (common in daily FRED data)
            fred_data = fred_data.ffill()
            return fred_data
        except Exception as e:
            self.logger.error(f"Error fetching FRED data: {e}")
            return None

    def get_combined_data(self, bond_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get Treasury yield data combined with relevant FRED economic indicators."""
        try:
            # Fix: Handle default values properly for start_date and end_date
            if not end_date:
                end_date = True  # This will use current date in get_treasury_data
            
            treasury_df = self.get_treasury_data(bond_type, force_update=end_date)
            if treasury_df is None or treasury_df.empty:
                return None

            # Define relevant FRED series
            # Example: Fed Funds Rate, 10-Year Breakeven Inflation, VIX
            fred_series = ['FEDFUNDS', 'T10YIE', 'VIXCLS']
            # Determine date range for FRED data based on treasury data
            data_start_date = treasury_df['date'].min().strftime('%Y-%m-%d')
            data_end_date = treasury_df['date'].max().strftime('%Y-%m-%d')

            fred_df = self.get_fred_data(fred_series, data_start_date, data_end_date)

            if fred_df is None:
                self.logger.warning("Could not fetch FRED data, proceeding with Treasury data only.")
                return treasury_df

            # Ensure treasury_df date is index for merging
            treasury_df = treasury_df.set_index('date')

            # Merge dataframes
            combined_df = pd.merge(treasury_df, fred_df, left_index=True, right_index=True, how='left')

            # Forward fill FRED data again after merge
            combined_df[fred_series] = combined_df[fred_series].ffill()
            # Optional: Backward fill initial NaNs if any
            combined_df[fred_series] = combined_df[fred_series].bfill()

            combined_df = combined_df.reset_index() # Reset index to have 'date' column back
            self.logger.info(f"Combined data shape: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in get_combined_data: {str(e)}")
            return None

    def fetch_upcoming_treasury_auctions(self, force_update: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch upcoming Treasury auctions using the Treasury Fiscal Data API.
        
        Args:
            force_update: If True, bypass cache and fetch fresh data
            
        Returns:
            List of upcoming auction data dictionaries
        """
        cache_file = os.path.join(self.cache_dir, "upcoming_auctions.json")
        
        # Check cache first if not forcing update
        if not force_update and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                last_fetch = datetime.fromisoformat(cached_data['timestamp'])
                now = datetime.now()
                
                # Use cache if it's less than a day old
                if (now - last_fetch).days < 1 and 'auctions' in cached_data:
                    logger.info(f"Using cached upcoming auction data")
                    return cached_data['auctions']
            
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading upcoming auctions cache file: {str(e)}")
        
        logger.info(f"Fetching fresh upcoming auction data from Treasury API")
        
        try:
            # Fetch data from Treasury API
            response = requests.get(self.treasury_api_url)
            response.raise_for_status()
            data = response.json()
            
            auctions = data.get('data', [])
            
            # Save to cache
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'auctions': auctions
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved upcoming auction data to cache with {len(auctions)} entries")
            return auctions
            
        except requests.RequestException as e:
            logger.error(f"Error fetching upcoming auction data from Treasury API: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing upcoming auction data: {str(e)}")
            return []

    def get_next_auction_date(self, bond_type: str, force_update: bool = False) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get the next auction date for a specific bond type from Treasury API.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            
        Returns:
            Tuple of (next_auction_date_str, auction_details) or (None, None) if not available
        """
        try:
            today = datetime.now().date()
            
            # Get the appropriate security type and term for this bond
            security_type = TREASURY_TYPE_MAP.get(bond_type)
            security_term = TREASURY_TERM_MAP.get(bond_type)
            
            if not security_type or not security_term:
                logger.warning(f"No Treasury mapping for bond type: {bond_type}")
                return None, None
                
            logger.info(f"Looking for {bond_type} auctions (security type: {security_type}, term: {security_term})")
            
            # Fetch all upcoming auctions
            auctions = self.fetch_upcoming_treasury_auctions(force_update)
            if not auctions:
                logger.warning(f"No upcoming auction data available")
                return None, None
                
            logger.info(f"Retrieved {len(auctions)} total upcoming auctions")
            
            # Special case for 20-Year bonds - they might be listed with slightly different terms
            if bond_type == '20Y':
                # For 20Y, look for any bond with "20-Year", "20 Year", "19-Year", etc.
                matching_auctions = [
                    auction for auction in auctions
                    if auction.get('security_type') == 'Bond' and 
                    ('20' in auction.get('security_term', '') or '19' in auction.get('security_term', ''))
                ]
                if matching_auctions:
                    logger.info(f"Found {len(matching_auctions)} potential 20Y bond auctions")
            else:
                # For other bond types, match type and term more strictly
                type_key = security_type.lower()
                term_key = security_term.split('-')[0].lower()  # Get the numeric part (e.g., "5" from "5-Year")
                
                matching_auctions = [
                    auction for auction in auctions 
                    if auction.get('security_type', '').lower() == type_key and
                    term_key in auction.get('security_term', '').lower()
                ]
            
            # Debug output about matching auctions
            if matching_auctions:
                logger.info(f"Found {len(matching_auctions)} {bond_type} auctions matching criteria")
                sample_term = matching_auctions[0].get('security_term', 'unknown')
                logger.info(f"Sample matching term: {sample_term}")
            else:
                logger.info(f"No {bond_type} auctions found matching criteria (type: {security_type}, term: {security_term})")
            
            # Find the next auction date from matching auctions
            next_auction = None
            next_auction_date = None
            
            for auction in matching_auctions:
                auction_date_str = auction.get('auction_date')
                if not auction_date_str:
                    continue
                
                try:
                    auction_date = datetime.strptime(auction_date_str, '%Y-%m-%d').date()
                    
                    # Consider only future auctions or today
                    if auction_date >= today:
                        if next_auction_date is None or auction_date < next_auction_date:
                            next_auction_date = auction_date
                            next_auction = auction
                except ValueError:
                    logger.warning(f"Invalid auction date format: {auction_date_str}")
            
            if next_auction_date:
                logger.info(f"Next {bond_type} auction date: {next_auction_date.strftime('%Y-%m-%d')}")
                if next_auction.get('reopening') == 'Yes':
                    logger.info(f"This is a reopening of existing {bond_type} bond")
                return next_auction_date.strftime('%Y-%m-%d'), next_auction
            else:
                logger.info(f"No upcoming auctions found for {bond_type}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error determining next auction date for {bond_type}: {e}")
            return None, None
    
    def add_auction_data_to_prediction(self, prediction: Dict[str, Any], bond_type: str, 
                                      force_update: bool = False) -> Dict[str, Any]:
        """
        Add auction data to a prediction result.
        
        Args:
            prediction: Prediction dictionary to update
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            
        Returns:
            Updated prediction dictionary with auction data
        """
        try:
            next_auction_date, auction_details = self.get_next_auction_date(bond_type, force_update)
            prediction['next_auction_date'] = next_auction_date
            
            if next_auction_date and auction_details:
                prediction_date = datetime.strptime(prediction['prediction_date'], '%Y-%m-%d').date()
                next_date = datetime.strptime(next_auction_date, '%Y-%m-%d').date()
                days_to_auction = (next_date - prediction_date).days
                
                auction_info = {
                    'days_to_auction': days_to_auction,
                    'cusip': auction_details.get('cusip'),
                    'announcement_date': auction_details.get('announcemt_date'),
                    'issue_date': auction_details.get('issue_date'),
                    'offering_amount': auction_details.get('offering_amt'),
                    'reopening': auction_details.get('reopening') == 'Yes',
                    'security_term': auction_details.get('security_term')
                }
                
                prediction['auction_info'] = auction_info
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error adding auction data to prediction: {e}")
            return prediction

    def test_next_auction_date(self, bond_type: str) -> None:
        """
        Test function to check what auction dates the current implementation finds.
        
        Args:
            bond_type: Bond type to test (e.g., '20Y')
        """
        logger.info(f"Testing auction date finder for {bond_type}")
        
        # Force update to get fresh data
        next_date, auction_details = self.get_next_auction_date(bond_type, force_update=True)
        
        logger.info(f"Next auction date for {bond_type}: {next_date}")
        
        if auction_details:
            logger.info("Auction details:")
            for key, value in auction_details.items():
                logger.info(f"  {key}: {value}")
            
            # Special attention to security type and term
            logger.info(f"Security Type: {auction_details.get('security_type', 'N/A')}")
            logger.info(f"Security Term: {auction_details.get('security_term', 'N/A')}")
        else:
            logger.info(f"No auction details available for {bond_type}")
        
        # Log debug info about all auctions
        all_auctions = self.fetch_upcoming_treasury_auctions(force_update=True)
        potential_matches = [a for a in all_auctions if '20' in a.get('security_term', '')]
        
        if bond_type == '20Y' and potential_matches:
            logger.info(f"Found {len(potential_matches)} potential 20Y auctions in raw data:")
            for i, auction in enumerate(potential_matches[:5], 1):  # Show first 5
                logger.info(f"Potential {bond_type} auction {i}:")
                logger.info(f"  Type: {auction.get('security_type')}")
                logger.info(f"  Term: {auction.get('security_term')}")
                logger.info(f"  Date: {auction.get('auction_date')}")
                
    def backtest_yield_predictions(self, bond_type: str, num_auctions: int = 10) -> Dict[str, Any]:
        """
        Perform back-testing of yield predictions against historical auction data.
        
        Args:
            bond_type: Bond type (e.g., '10Y', '20Y')
            num_auctions: Number of past auctions to test against (default: 10)
            
        Returns:
            Dictionary with back-testing results including accuracy metrics
        """
        logger.info(f"Starting back-testing for {bond_type} with {num_auctions} historical auctions")
        
        try:
            # Get historical auction data
            historical_auctions = self.get_treasury_direct_auctions(bond_type, force_update=True)
            
            if not historical_auctions:
                logger.warning(f"No historical auction data available for {bond_type}")
                return {"success": False, "error": "No historical auction data available"}
                
            # Take only the specified number of most recent auctions
            test_auctions = historical_auctions[:num_auctions]
            logger.info(f"Found {len(test_auctions)} historical auctions for testing")
            
            results = []
            
            # For each historical auction, simulate making a prediction
            for auction in test_auctions:
                auction_date_str = auction.get('date')
                actual_yield = float(auction.get('high_yield', 0))
                
                # Convert string to datetime
                auction_date = datetime.strptime(auction_date_str, '%Y-%m-%d')
                
                # Simulate making prediction 7 days before auction
                prediction_date = auction_date - timedelta(days=7)
                prediction_date_str = prediction_date.strftime('%Y-%m-%d')
                
                # Get Treasury data up to the prediction date
                # This simulates only having data available up to the prediction date
                treasury_df = self.get_treasury_data(
                    bond_type, 
                    force_update=True,
                    end_date=prediction_date_str
                )
                
                if treasury_df.empty:
                    logger.warning(f"No treasury data available for prediction date {prediction_date_str}")
                    continue
                
                # Use the current yield on prediction date as a simple prediction
                # In a real implementation, this would call the predictor module
                latest_yield = treasury_df['value'].iloc[-1]
                
                # In a real implementation, we would use a more sophisticated prediction model
                # For now, we use a simple model: current yield + small adjustment
                # This is just a placeholder - the actual prediction would come from the predictor module
                predicted_yield = latest_yield + 0.05  # Simple adjustment
                
                # Calculate error
                error = predicted_yield - actual_yield
                percent_error = (error / actual_yield) * 100 if actual_yield != 0 else float('inf')
                
                result = {
                    'auction_date': auction_date_str,
                    'prediction_date': prediction_date_str,
                    'actual_yield': actual_yield,
                    'predicted_yield': predicted_yield,
                    'error': error,
                    'percent_error': percent_error
                }
                
                results.append(result)
                logger.info(f"Auction {auction_date_str}: Predicted {predicted_yield:.3f}%, Actual {actual_yield:.3f}%, Error {error:.3f}%")
            
            # Calculate overall metrics
            errors = [r['error'] for r in results]
            percent_errors = [r['percent_error'] for r in results]
            
            metrics = {
                'mean_absolute_error': sum(abs(e) for e in errors) / len(errors) if errors else 0,
                'root_mean_squared_error': (sum(e**2 for e in errors) / len(errors))**0.5 if errors else 0,
                'mean_absolute_percent_error': sum(abs(p) for p in percent_errors) / len(percent_errors) if percent_errors else 0,
                'num_predictions': len(results),
                'num_underestimates': sum(1 for e in errors if e < 0),
                'num_overestimates': sum(1 for e in errors if e > 0)
            }
            
            # Save results to cache
            cache_file = os.path.join(self.cache_dir, f"backtest_{bond_type}.json")
            backtest_data = {
                'bond_type': bond_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'predictions': results
            }
            
            with open(cache_file, 'w') as f:
                json.dump(backtest_data, f, indent=2)
            
            logger.info(f"Back-testing complete for {bond_type}")
            logger.info(f"Mean absolute error: {metrics['mean_absolute_error']:.4f}%")
            logger.info(f"Root mean squared error: {metrics['root_mean_squared_error']:.4f}%")
            logger.info(f"Mean absolute percent error: {metrics['mean_absolute_percent_error']:.2f}%")
            
            return {
                'success': True,
                'bond_type': bond_type,
                'metrics': metrics,
                'predictions': results
            }
            
        except Exception as e:
            logger.error(f"Error during back-testing for {bond_type}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def run_auction_date_test():
    """Simple helper function to test the next auction date functionality."""
    fetcher = DataFetcher()
    fetcher.test_next_auction_date('20Y')
    
if __name__ == "__main__":
    # This allows running the test directly from this file
    run_auction_date_test()