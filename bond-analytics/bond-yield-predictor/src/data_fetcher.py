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
    
    def get_treasury_data(self, bond_type: str, force_update: bool = False) -> pd.DataFrame:
        """
        Get Treasury yield data for a specific bond type.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            
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
                    return pd.DataFrame(cached_data['data'])
            
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading cache file for {bond_type}: {str(e)}")
        
        # Fetch fresh data from FRED
        logger.info(f"Fetching fresh data for {bond_type}")
        df = self._get_fred_data(series_id)
        
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
        Get historical auction data from cache or generate mock data if needed.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            force_update: If True, bypass cache and fetch fresh data
            
        Returns:
            List of auction data dictionaries
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
        
        logger.info(f"Fetching TreasuryDirect auction data for {bond_type}")
        
        try:
            # Map bond type to TreasuryDirect term
            treasury_term_map = {
                '1M': '4-week',
                '3M': '13-week',
                '6M': '26-week',
                '1Y': '52-week',
                '2Y': '2-year',
                '5Y': '5-year',
                '7Y': '7-year',
                '10Y': '10-year',
                '20Y': '20-year',
                '30Y': '30-year'
            }
            
            term = treasury_term_map.get(bond_type)
            if not term:
                logger.warning(f"No TreasuryDirect term mapping for {bond_type}")
                return []
            
            # For demonstration and testing purposes, we'll generate mock auction data
            # that resembles what might come from TreasuryDirect
            # In a production environment, this would be replaced with actual web scraping
            
            # Generate mock auctions (for historical testing)
            auctions = self._generate_mock_auctions(bond_type)
            
            # Save to cache
            data_to_cache = {
                'bond_type': bond_type,
                'timestamp': datetime.now().isoformat(),
                'auctions': auctions
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data_to_cache, f, indent=2)
            
            logger.info(f"Saved {bond_type} auction data to cache")
            return auctions
            
        except requests.RequestException as e:
            logger.error(f"Error fetching auction data from TreasuryDirect: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error parsing auction data: {str(e)}")
            return []
            
    def _generate_mock_auctions(self, bond_type: str) -> List[Dict[str, Any]]:
        """
        Generate mock auction data for testing when web scraping is not available.
        
        Args:
            bond_type: Bond type (e.g., '10Y')
            
        Returns:
            List of mock auction dictionaries
        """
        # Base yields for different bond types
        base_yields = {
            '1M': 3.8,
            '3M': 4.0,
            '6M': 4.2,
            '1Y': 4.3,
            '2Y': 4.4,
            '5Y': 4.3,
            '7Y': 4.3,
            '10Y': 4.2,
            '20Y': 4.4,
            '30Y': 4.5
        }
        
        # CUSIP prefixes by bond type
        cusip_prefixes = {
            '10Y': '912810',
            '20Y': '912810',
            '30Y': '912810',
            '7Y': '912828',
            '5Y': '912828',
            '2Y': '912828',
            '1Y': '912796',
            '6M': '912796',
            '3M': '912796',
            '1M': '912796'
        }
        
        # Generate auctions
        auctions = []
        
        # End date is today
        end_date = datetime.now()
        
        # Different bonds have different auction frequencies
        if bond_type in ['1M', '3M', '6M']:
            frequency_days = 7  # Weekly
            n_auctions = 60  # About a year of weekly auctions
        elif bond_type in ['1Y', '2Y', '5Y', '7Y']:
            frequency_days = 30  # Monthly
            n_auctions = 36  # 3 years of monthly auctions
        else:
            frequency_days = 90  # Quarterly
            n_auctions = 24  # 6 years of quarterly auctions
        
        # Generate auction dates (backwards from today)
        for i in range(n_auctions):
            # Auction date
            auction_date = end_date - timedelta(days=i * frequency_days)
            auction_date_str = auction_date.strftime("%Y-%m-%d")
            
            # Generate CUSIP
            cusip = f"{cusip_prefixes.get(bond_type, '912810')}{chr(65 + (i % 26))}{i % 10}"
            
            # Base yield with some variation
            base_yield = base_yields.get(bond_type, 4.0)
            
            # Add time trend (yields were generally lower in the past)
            time_factor = i / n_auctions  # 0 to 1
            historical_adjustment = -1.5 * time_factor  # -1.5% change over the full period
            
            # Add cyclical component
            cyclical = 0.3 * np.sin(i * 0.5)
            
            # Add randomness
            random_factor = np.random.normal(0, 0.1)
            
            # Calculate yield
            high_yield = base_yield + historical_adjustment + cyclical + random_factor
            high_yield = max(0.1, high_yield)  # Ensure positive yield
            
            auctions.append({
                'date': auction_date_str,
                'cusip': cusip,
                'high_yield': f"{high_yield:.3f}"
            })
        
        # Sort by date (newest first, which is how TreasuryDirect typically presents data)
        auctions.sort(key=lambda x: x['date'], reverse=True)
        
        return auctions

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