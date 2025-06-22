#!/usr/bin/env python3
"""
S&P 500 Low P/E or P/B Dividend Screener

This script identifies companies in the S&P 500 with:
- The lowest HISTORICAL P/E or P/B ratios for a specified year
- 5+ years of dividend payments (including the target year)
- Dividend yield above specified threshold in the target year

Then reports stock price gains for those companies in the year subsequent to the target year.

HISTORICAL P/E CALCULATION:
- Uses year-end stock price for the target year
- Uses net income and shares outstanding for the target year
- Calculates EPS = Net Income / Shares Outstanding
- Calculates P/E = Price / EPS

HISTORICAL P/B CALCULATION:
- Uses year-end stock price for the target year
- Uses book value per share for the target year
- Calculates P/B = Price / Book Value per Share

Usage: python value_price_screener.py 2019 [--count 20] [--metric pb] [--dividend-yield 2.0] [--min-dividend-years 5] [--verbose]
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse
import time  # Added for rate limiting
import requests
import json
from pathlib import Path
from io import StringIO
import warnings
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO as IOStringIO
import wcwidth  # Add this import at the top of your file

__version__ = '1.0.5'

# Default configuration values
DEFAULT_COUNT = 20
DEFAULT_METRIC = 'pb'  # 'pe' or 'pb'
DEFAULT_DIVIDEND_YIELD = 2.0
DEFAULT_MIN_DIVIDEND_YEARS = 5

def create_ticker_with_suppression(ticker_symbol):
    """
    Create a yfinance Ticker object while suppressing warnings and error messages.
    For historical analysis, we don't care if the ticker is currently delisted.
    Returns (ticker_object, should_process) tuple.
    """
    try:
        # Suppress all warnings and stdout/stderr during ticker creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Capture stdout and stderr to prevent yfinance messages
            stdout_capture = IOStringIO()
            stderr_capture = IOStringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Also suppress specific logging that yfinance uses
                import logging
                logging.getLogger('yfinance').setLevel(logging.CRITICAL)
                
                ticker = yf.Ticker(ticker_symbol)
                
                # For historical analysis, we always try to process the ticker
                # Even if it's delisted, historical data should still be available
                return ticker, True
                    
    except Exception:
        return None, False

def get_sp500_tickers_for_year(target_year):
    """
    Get S&P 500 companies for a specific year using comprehensive historical data.
    
    Uses the hanshof/sp500_constituents GitHub repository which provides
    daily historical S&P 500 membership data from 1996 to present.
    
    Implements local caching to ./data/cache/ directory.
    """
    
    # Set up cache directory and file paths
    cache_dir = Path("./data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "sp500_historical_constituents.csv"
    metadata_file = cache_dir / "sp500_cache_metadata.json"
    
    # GitHub URL for comprehensive historical data
    github_url = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv"
    
    def should_download_fresh_data():
        """Check if we need to download fresh data"""
        if not cache_file.exists():
            return True, "Cache file doesn't exist"
        
        if not metadata_file.exists():
            return True, "Metadata file doesn't exist"
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache contains data for the target year
            cached_years = metadata.get('available_years', [])
            if target_year not in cached_years:
                return True, f"Target year {target_year} not in cached data"
            
            # Check cache age (refresh if older than 7 days)
            cache_date = datetime.fromisoformat(metadata['last_updated'])
            days_old = (datetime.now() - cache_date).days
            if days_old > 7:
                return True, f"Cache is {days_old} days old"
            
            return False, "Using cached data"
            
        except Exception as e:
            return True, f"Error reading metadata: {e}"
    
    def download_and_cache_data():
        """Download historical data and cache it locally"""
        try:
            print(f"Downloading comprehensive S&P 500 historical data...")
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            
            # Save the CSV data
            with open(cache_file, 'w') as f:
                f.write(response.text)
            
            # Parse to get available years for metadata
            df = pd.read_csv(StringIO(response.text))
            df['date'] = pd.to_datetime(df['date'])
            available_years = sorted(df['date'].dt.year.unique().tolist())
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'source_url': github_url,
                'file_size_bytes': len(response.text),
                'total_records': len(df),
                'date_range': {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat()
                },
                'available_years': available_years
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Data cached successfully: {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return False
    
    def get_constituents_from_cache(year):
        """Get S&P 500 constituents for a specific year from cached data"""
        try:
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Find the last trading day of the target year
            target_date = datetime(year, 12, 31)
            
            # Get all dates on or before the target date
            available_dates = df[df['date'] <= target_date]['date']
            if available_dates.empty:
                print(f"❌ No historical data available for {year}")
                return []
            
            # Get the most recent date (closest to year-end)
            most_recent_date = available_dates.max()
            row = df[df['date'] == most_recent_date].iloc[0]
            
            # Parse the comma-separated tickers
            tickers = [ticker.strip() for ticker in row['tickers'].split(',') if ticker.strip()]
            
            # Clean ticker symbols for yfinance compatibility
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            print(f"✅ S&P 500 companies for {year}: {len(tickers)} companies")
            print(f"   Using data from: {most_recent_date.date()}")
            
            return tickers
            
        except Exception as e:
            print(f"❌ Error reading cached data: {e}")
            return []
    
    # Main logic
    try:
        need_download, reason = should_download_fresh_data()
        
        if need_download:
            print(f"📥 {reason}")
            if not download_and_cache_data():
                # If download fails, try to use existing cache if available
                if cache_file.exists():
                    print("⚠️  Using existing cached data due to download failure")
                else:
                    print("❌ No cached data available and download failed")
                    return []
        else:
            print(f"📋 {reason}")
        
        # Get constituents from cached data
        return get_constituents_from_cache(target_year)
        
    except Exception as e:
        print(f"❌ Error in get_sp500_tickers_for_year: {e}")
        return []

def get_year_end_price(stock, year):
    """Get year-end price with enhanced retry logic and multiple date fallbacks"""
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Suppress warnings during data retrieval
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try multiple date ranges to find year-end data
                date_ranges = [
                    (f"{year}-12-01", f"{year}-12-31"),  # December
                    (f"{year}-11-15", f"{year}-12-31"),  # Late November to December
                    (f"{year}-11-01", f"{year}-12-31"),  # November to December
                    (f"{year}-10-15", f"{year}-12-31"),  # Mid October to December
                    (f"{year}-01-01", f"{year}-12-31"),  # Full year (last resort)
                ]
                
                for start_date, end_date in date_ranges:
                    try:
                        # GET REAL CLOSING PRICE, NOT ADJUSTED
                        hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
                        if not hist.empty:
                            # Return the last available REAL close price in the range
                            return hist['Close'].iloc[-1]
                        time.sleep(1)  # Brief pause between attempts
                    except Exception:
                        continue
                
                return None
            
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                wait_time = min(60, (2 ** attempt) * 10)  # Exponential backoff, max 60s
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return None
    
    return None

def get_historical_eps(stock, year, ticker=None):
    """
    Get historical EPS for a specific year using yfinance with retry logic.
    Enhanced to work with delisted companies by focusing on historical financials.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            time.sleep(1 + attempt)  # Progressive delay
            
            # Suppress warnings during data retrieval
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Method 1: Try to get annual financials and calculate EPS
                try:
                    income_stmt = stock.income_stmt
                    if not income_stmt.empty:
                        # Look for the target year
                        target_col = None
                        for col in income_stmt.columns:
                            try:
                                col_year = pd.to_datetime(col).year
                                if col_year == year:
                                    target_col = col
                                    break
                            except:
                                continue
                        
                        if target_col is not None:
                            # Try different net income field names
                            net_income = None
                            for field in ['Net Income', 'Net Income Common Stockholders', 'NetIncome', 'Net Income Available To Common Shareholders']:
                                if field in income_stmt.index:
                                    net_income = income_stmt.at[field, target_col]
                                    break
                            
                            if net_income and not pd.isna(net_income):
                                # Try to get historical shares outstanding
                                shares = None
                                
                                # First try from cash flow statement for the same year
                                try:
                                    cash_flow = stock.cashflow
                                    if not cash_flow.empty:
                                        for cf_col in cash_flow.columns:
                                            try:
                                                cf_year = pd.to_datetime(cf_col).year
                                                if cf_year == year:
                                                    for shares_field in ['Shares Outstanding', 'Common Stock Shares Outstanding', 'SharesOutstanding', 'Weighted Average Shares Outstanding']:
                                                        if shares_field in cash_flow.index:
                                                            shares_val = cash_flow.at[shares_field, cf_col]
                                                            if shares_val and not pd.isna(shares_val) and shares_val > 0:
                                                                shares = shares_val
                                                                break
                                                    if shares:
                                                        break
                                            except:
                                                continue
                                except:
                                    pass
                                
                                # If no shares from cash flow, try from income statement
                                if not shares:
                                    try:
                                        for shares_field in ['Diluted Average Shares', 'Basic Average Shares', 'Weighted Average Shares Outstanding']:
                                            if shares_field in income_stmt.index:
                                                shares_val = income_stmt.at[shares_field, target_col]
                                                if shares_val and not pd.isna(shares_val) and shares_val > 0:
                                                    shares = shares_val
                                                    break
                                    except:
                                        pass
                                
                                # REMOVE THIS BUGGY FALLBACK - Don't mix historical and current data
                                # if not shares:
                                #     try:
                                #         info = stock.info
                                #         if info:
                                #             shares = info.get('sharesOutstanding')
                                #     except:
                                #         pass
                                
                                # Calculate EPS ONLY if we have historical shares
                                if shares and shares > 0:
                                    return net_income / shares
                                else:
                                    # No historical shares found - return None instead of using current shares
                                    return None
                except Exception as e:
                    if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    pass
                
                # Method 2: Try quarterly data and sum for the year
                try:
                    quarterly_income = stock.quarterly_income_stmt
                    if not quarterly_income.empty:
                        annual_income = 0
                        quarters_found = 0
                        quarterly_shares = []
                        
                        for col in quarterly_income.columns:
                            try:
                                col_year = pd.to_datetime(col).year
                                if col_year == year:
                                    # Get net income for this quarter
                                    for field in ['Net Income', 'Net Income Common Stockholders', 'NetIncome']:
                                        if field in quarterly_income.index:
                                            val = quarterly_income.at[field, col]
                                            if val and not pd.isna(val):
                                                annual_income += val
                                                quarters_found += 1
                                                
                                                # Try to get shares for this quarter
                                                try:
                                                    quarterly_cash = stock.quarterly_cashflow
                                                    if not quarterly_cash.empty and col in quarterly_cash.columns:
                                                        for shares_field in ['Shares Outstanding', 'Common Stock Shares Outstanding', 'SharesOutstanding']:
                                                            if shares_field in quarterly_cash.index:
                                                                shares_val = quarterly_cash.at[shares_field, col]
                                                                if shares_val and not pd.isna(shares_val) and shares_val > 0:
                                                                    quarterly_shares.append(shares_val)
                                                                    break
                                                except:
                                                    pass
                                                break
                            except:
                                continue
                        
                        if quarters_found >= 3:  # Need at least 3 quarters
                            # Use average shares ONLY if we have quarterly historical data
                            if quarterly_shares:
                                avg_shares = sum(quarterly_shares) / len(quarterly_shares)
                                return annual_income / avg_shares
                            else:
                                # No historical shares found - return None instead of using current shares
                                return None
                except Exception as e:
                    if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    pass
                
                # REMOVE Method 3: Don't fallback to current EPS - causes inconsistency
                # try:
                #     info = stock.info
                #     if info:
                #         eps = info.get('trailingEps') or info.get('forwardEps')
                #         if eps and eps > 0:
                #             return eps
                # except Exception as e:
                #     if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                #         time.sleep(10 * (attempt + 1))
                #         continue
                #     pass
            
            return None
            
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            break
    
    return None

def get_pb_for_year(stock, year, verbose=False):
    """
    Calculate HISTORICAL P/B ratio for a specific year.
    Uses year-end price and book value per share for that year.
    Returns (pb_ratio, error_message) tuple
    """
    try:
        # Get historical year-end price
        price = get_year_end_price(stock, year)
        if not price or price <= 0:
            error_msg = f"No price data for {year}" if not price else f"Invalid price ${price:.2f}"
            return None, error_msg
        
        # Get book value per share for the year
        book_value_per_share = get_historical_book_value_per_share(stock, year)
        if not book_value_per_share:
            return None, f"No book value data for {year}"
        elif book_value_per_share <= 0:
            return None, f"Negative book value ${book_value_per_share:.2f} for {year}"
        
        # Calculate P/B = Price / Book Value per Share
        pb_ratio = price / book_value_per_share
        
        # Sanity check - reject extreme values
        if pb_ratio <= 0:
            return None, f"Invalid P/B ratio {pb_ratio:.2f}"
        elif pb_ratio >= 1000:
            return None, f"Extreme P/B ratio {pb_ratio:.2f} (>1000)"
        
        if verbose:
            return pb_ratio, f"Success: P/B={pb_ratio:.2f} (Price=${price:.2f}, BVPS=${book_value_per_share:.2f})"
        else:
            return pb_ratio, None
        
    except Exception as e:
        return None, f"Error calculating P/B: {str(e)}"

def get_cik_from_ticker(ticker):
    """Get CIK (Central Index Key) from ticker symbol using SEC company tickers file"""
    cache_dir = Path("./data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cik_cache_file = cache_dir / "sec_cik_mapping.json"
    
    # Check if we have cached CIK mapping
    if cik_cache_file.exists():
        try:
            with open(cik_cache_file, 'r') as f:
                cik_mapping = json.load(f)
            if ticker.upper() in cik_mapping:
                return cik_mapping[ticker.upper()]
        except:
            pass
    
    # Download fresh CIK mapping from SEC
    try:
        headers = {
            'User-Agent': 'Financial Analytics Script (your-email@example.com)'  # SEC requires User-Agent
        }
        
        response = requests.get(
            'https://www.sec.gov/files/company_tickers.json',
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse the data and create ticker -> CIK mapping
        company_data = response.json()
        cik_mapping = {}
        
        for company_info in company_data.values():
            ticker_symbol = company_info.get('ticker', '').upper()
            cik = str(company_info.get('cik_str', '')).zfill(10)  # Pad with zeros to 10 digits
            if ticker_symbol and cik:
                cik_mapping[ticker_symbol] = cik
        
        # Cache the mapping
        with open(cik_cache_file, 'w') as f:
            json.dump(cik_mapping, f, indent=2)
        
        return cik_mapping.get(ticker.upper())
        
    except Exception as e:
        print(f"Error fetching CIK for {ticker}: {e}")
        return None

def get_edgar_balance_sheet_data(ticker, year):
    """Get balance sheet data from SEC EDGAR API for a specific year"""
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return None
    
    try:
        headers = {
            'User-Agent': 'Financial Analytics Script (your-email@example.com)'  # SEC requires User-Agent
        }
        
        # Get company facts (all historical financial data)
        response = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        company_facts = response.json()
        
        # Navigate to balance sheet data
        facts = company_facts.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Look for stockholders equity field
        stockholders_equity_data = None
        equity_fields = [
            'StockholdersEquity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
            'TotalStockholdersEquity'
        ]
        
        for field in equity_fields:
            if field in us_gaap:
                stockholders_equity_data = us_gaap[field]
                break
        
        if not stockholders_equity_data:
            return None
        
        # Look for shares outstanding data
        shares_data = None
        shares_fields = [
            'CommonStockSharesOutstanding',
            'CommonSharesOutstanding',
            'WeightedAverageNumberOfSharesOutstandingBasic'
        ]
        
        for field in shares_fields:
            if field in us_gaap:
                shares_data = us_gaap[field]
                break
        
        if not shares_data:
            return None
        
        # Find data for the target year
        target_equity = None
        target_shares = None
        
        # Look for annual data (10-K filings) for the target year
        for unit_data in stockholders_equity_data.get('units', {}).get('USD', []):
            filing_date = unit_data.get('end', '')
            form = unit_data.get('form', '')
            
            # Only use annual reports (10-K) and check if it's for our target year
            if form == '10-K' and filing_date.startswith(str(year)):
                target_equity = unit_data.get('val')
                break
        
        for unit_data in shares_data.get('units', {}).get('shares', []):
            filing_date = unit_data.get('end', '')
            form = unit_data.get('form', '')
            
            # Only use annual reports (10-K) and check if it's for our target year
            if form == '10-K' and filing_date.startswith(str(year)):
                target_shares = unit_data.get('val')
                break
        
        if target_equity and target_shares and target_shares > 0:
            book_value_per_share = target_equity / target_shares
            return book_value_per_share
        
        return None
        
    except Exception as e:
        print(f"Error fetching EDGAR data for {ticker}: {e}")
        return None

# Replace the get_historical_book_value_per_share function with this EDGAR-based version:

def get_edgar_stockholders_equity(ticker, year):
    """Get only stockholders equity from EDGAR (ignore shares data due to split issues)"""
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return None
    
    try:
        headers = {'User-Agent': 'Financial Analytics Script (your-email@example.com)'}
        
        response = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        company_facts = response.json()
        facts = company_facts.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Look for stockholders equity
        equity_fields = [
            'StockholdersEquity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
            'TotalStockholdersEquity'
        ]
        
        for field in equity_fields:
            if field in us_gaap:
                equity_data = us_gaap[field]
                
                # Find 10-K filing for target year
                for unit_data in equity_data.get('units', {}).get('USD', []):
                    filing_date = unit_data.get('end', '')
                    form = unit_data.get('form', '')
                    
                    if form == '10-K' and filing_date.startswith(str(year)):
                        return unit_data.get('val')
        
        return None
        
    except Exception as e:
        print(f"Error fetching EDGAR equity for {ticker}: {e}")
        return None

def get_stock_split_factor(ticker, from_date, to_date):
    """Get cumulative split factor using yfinance (fallback from Polygon.io)"""
    try:
        # Try Polygon.io first (will fail with 401 but worth trying)
        url = f"https://api.polygon.io/v3/reference/splits?ticker={ticker}&execution_date.gte={from_date}&execution_date.lte={to_date}&limit=1000"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            splits = data.get('results', [])
            
            split_factor = 1.0
            for split in splits:
                split_from = split.get('split_from', 1)
                split_to = split.get('split_to', 1)
                if split_from and split_to and split_from > 0:
                    ratio = split_to / split_from
                    split_factor *= ratio
            
            return split_factor
        
    except Exception:
        pass
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        
        from_year = int(from_date[:4])
        to_year = int(to_date[:4])
        
        return get_stock_split_factor_yfinance(stock, from_year, to_year)
        
    except Exception as e:
        print(f"Error getting split data for {ticker}: {e}")
        return 1.0

def get_stock_split_factor_yfinance(stock, from_year, to_year):
    """Calculate cumulative stock split factor using yfinance historical data"""
    try:
        # Get stock actions (splits, dividends) from yfinance
        actions = stock.actions
        if actions.empty:
            return 1.0
        
        # Create timezone-aware datetime objects to match yfinance index
        from_date = pd.Timestamp(f"{from_year}-12-31").tz_localize('America/New_York')
        to_date = pd.Timestamp(f"{to_year}-12-31").tz_localize('America/New_York')
        
        # Handle case where actions index might not be timezone-aware
        if actions.index.tz is None:
            # If actions index is not timezone-aware, convert our timestamps to naive
            from_date = pd.Timestamp(f"{from_year}-12-31")
            to_date = pd.Timestamp(f"{to_year}-12-31")
        elif actions.index.tz != from_date.tz:
            # Convert to match the timezone of the actions index
            from_date = from_date.tz_convert(actions.index.tz)
            to_date = to_date.tz_convert(actions.index.tz)
        
        # Stock splits show up as values != 0 in the 'Stock Splits' column
        splits_in_range = actions[(actions.index > from_date) & 
                                 (actions.index <= to_date) & 
                                 (actions['Stock Splits'] != 0)]
        
        if splits_in_range.empty:
            return 1.0
        
        # Calculate cumulative split factor
        split_factor = 1.0
        for split_ratio in splits_in_range['Stock Splits']:
            if split_ratio > 0:
                split_factor *= split_ratio
        
        return split_factor
        
    except Exception as e:
        print(f"Error calculating split factor: {e}")
        return 1.0

def get_edgar_shares_outstanding(ticker, year):
    """Get raw shares outstanding from EDGAR for a specific year with enhanced units correction"""
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return None
    
    try:
        headers = {'User-Agent': 'Financial Analytics Script (your-email@example.com)'}
        
        response = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        company_facts = response.json()
        facts = company_facts.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Look for shares outstanding
        shares_fields = [
            'CommonStockSharesOutstanding',
            'CommonSharesOutstanding',
            'WeightedAverageNumberOfSharesOutstandingBasic',
            'CommonStockSharesIssued'
        ]
        
        raw_shares = None
        
        for field in shares_fields:
            if field in us_gaap:
                shares_data = us_gaap[field]
                
                # Find 10-K filing for target year
                for unit_data in shares_data.get('units', {}).get('shares', []):
                    filing_date = unit_data.get('end', '')
                    form = unit_data.get('form', '')
                    
                    if form == '10-K' and filing_date.startswith(str(year)):
                        raw_shares = unit_data.get('val')
                        break
                
                if raw_shares:
                    break
        
        if not raw_shares:
            return None
        
        # ENHANCED UNITS CORRECTION: Multiple heuristics to detect wrong units
        
        # Heuristic 1: Any shares count under 1 million is suspicious for a public company
        if raw_shares < 1000000:
            equity = get_edgar_stockholders_equity(ticker, year)
            
            if equity and equity > 100000000:  # Equity > $100M (lowered threshold)
                if raw_shares < 1000:  # Likely in millions
                    corrected_shares = raw_shares * 1000000
                    print(f"   📊 EDGAR shares correction for {ticker}: {raw_shares:,.0f} → {corrected_shares:,.0f} (assumed millions)")
                    return corrected_shares
                elif raw_shares < 1000000:  # Likely in thousands
                    corrected_shares = raw_shares * 1000
                    print(f"   📊 EDGAR shares correction for {ticker}: {raw_shares:,.0f} → {corrected_shares:,.0f} (assumed thousands)")
                    return corrected_shares
        
        # Heuristic 2: Book value per share > $1000 is almost always wrong
        if raw_shares > 0:
            equity = get_edgar_stockholders_equity(ticker, year)
            if equity:
                potential_bvps = equity / raw_shares
                if potential_bvps > 1000:  # BVPS > $1000 is extremely suspicious
                    if raw_shares < 1000:  # Try millions first
                        corrected_shares = raw_shares * 1000000
                        corrected_bvps = equity / corrected_shares
                        if 1 <= corrected_bvps <= 500:  # Reasonable BVPS range
                            print(f"   📊 EDGAR shares correction for {ticker}: {raw_shares:,.0f} → {corrected_shares:,.0f} (BVPS fix: ${potential_bvps:.0f} → ${corrected_bvps:.2f})")
                            return corrected_shares
                    
                    if raw_shares < 1000000:  # Try thousands
                        corrected_shares = raw_shares * 1000
                        corrected_bvps = equity / corrected_shares
                        if 1 <= corrected_bvps <= 500:  # Reasonable BVPS range
                            print(f"   📊 EDGAR shares correction for {ticker}: {raw_shares:,.0f} → {corrected_shares:,.0f} (BVPS fix: ${potential_bvps:.0f} → ${corrected_bvps:.2f})")
                            return corrected_shares
        
        return raw_shares
        
    except Exception as e:
        print(f"Error fetching EDGAR shares for {ticker}: {e}")
        return None

def get_split_adjusted_edgar_bvps(ticker, year):
    """Get split-adjusted book value per share using EDGAR + Polygon splits"""
    
    # Step 1: Get raw data from EDGAR
    stockholders_equity = get_edgar_stockholders_equity(ticker, year)
    raw_shares = get_edgar_shares_outstanding(ticker, year)
    
    if not stockholders_equity or not raw_shares:
        return None
    
    # Step 2: Get split factor from target year to today
    from_date = f"{year}-12-31"
    to_date = datetime.now().strftime("%Y-%m-%d")
    split_factor = get_stock_split_factor(ticker, from_date, to_date)
    
    # Step 3: Apply split adjustment to shares
    split_adjusted_shares = raw_shares * split_factor
    
    # Step 4: Calculate split-adjusted BVPS
    book_value_per_share = stockholders_equity / split_adjusted_shares
    
    return book_value_per_share

def get_historical_book_value_per_share_corrected(stock, year):
    """
    Get historical book value per share using EDGAR equity data with split adjustments.
    This corrects for stock split inconsistencies between raw SEC data and adjusted price data.
    """
    ticker = stock.ticker
    
    # Use the split-adjusted EDGAR approach only
    split_adjusted_bvps = get_split_adjusted_edgar_bvps(ticker, year)
    return split_adjusted_bvps

def get_historical_book_value_per_share(stock, year):
    """
    Compatibility wrapper for get_historical_book_value_per_share_corrected.
    This function maintains backward compatibility while using the corrected approach.
    """
    return get_historical_book_value_per_share_corrected(stock, year)

def get_pe_for_year(stock, year, verbose=False):
    """
    Calculate HISTORICAL P/E ratio for a specific year.
    Uses year-end price and EPS for that year.
    Returns (pe_ratio, error_message) tuple
    """
    try:
        # Get historical year-end price
        price = get_year_end_price(stock, year)
        if not price or price <= 0:
            error_msg = f"No price data for {year}" if not price else f"Invalid price ${price:.2f}"
            return None, error_msg
        
        # Get historical EPS for the year
        eps = get_historical_eps(stock, year)
        if not eps:
            return None, f"No EPS data for {year}"
        elif eps <= 0:
            return None, f"Negative EPS ${eps:.2f} for {year}"
        
        # Calculate P/E = Price / EPS
        pe_ratio = price / eps
        
        # Sanity check - reject extreme values
        if pe_ratio <= 0:
            return None, f"Invalid P/E ratio {pe_ratio:.2f}"
        elif pe_ratio >= 1000:
            return None, f"Extreme P/E ratio {pe_ratio:.2f} (>1000)"
        
        if verbose:
            return pe_ratio, f"Success: P/E={pe_ratio:.2f} (Price=${price:.2f}, EPS=${eps:.2f})"
        else:
            return pe_ratio, None
        
    except Exception as e:
        return None, f"Error calculating P/E: {str(e)}"

def get_dividend_yield_for_year(stock, year):
    """Get dividend yield for a specific year with retry logic and warning suppression"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                price = get_year_end_price(stock, year)
                if price is None:
                    return None
                dividends = stock.dividends
                if dividends.empty:
                    return 0.0
                year_divs = dividends[(dividends.index.year == year)].sum()
                return (year_divs / price) * 100 if price > 0 else None
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                return None
    return None

def has_dividend_history(stock, year, min_years=5):
    """Check if stock has dividend history with retry logic and warning suppression"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                dividends = stock.dividends
                if dividends.empty:
                    return False
                years = set(dividends.index.year)
                needed = set(range(year - min_years + 1, year + 1))
                return needed.issubset(years)
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                return False
    return False

def get_price_gain(stock, year):
    """Calculate price gain for a year with warning suppression"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            price_start = get_year_start_price(stock, year)
            price_end = get_year_end_price(stock, year)
            if price_start and price_end and price_start != 0:
                gain = ((price_end - price_start) / price_start) * 100
            else:
                gain = None
    except Exception:
        pass
    return None

def get_year_start_price(stock, year):
    """Get year start price with warning suppression"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            start = f"{year}-01-01"
            end = f"{year}-01-15"
            # GET REAL CLOSING PRICE, NOT ADJUSTED
            hist = stock.history(start=start, end=end, auto_adjust=False)
            if not hist.empty:
                return hist['Close'].iloc[0]
    except Exception:
        pass
    return None

def get_alternative_data_source_suggestion():
    """Suggest alternative data sources for better reliability"""
    return """
ALTERNATIVE DATA SOURCES FOR BETTER RELIABILITY:

1. **Alpha Vantage API** (free tier available)
   - More reliable historical data
   - Better rate limiting
   - signup: https://www.alphavantage.co/

2. **Financial Modeling Prep API** (free tier available) 
   - Comprehensive financial statements
   - Historical price data
   - signup: https://financialmodelingprep.com/

3. **Quandl/Nasdaq Data Link** (paid)
   - High-quality financial data
   - Extensive historical coverage
   - Better for institutional use

4. **EDGAR SEC Filings** (free)
   - Official company filings
   - Most reliable for fundamental data
   - Requires more processing

Consider implementing fallback data sources for production use.
"""

def print_pretty_results_table(results, metric_name, target_year, dividend_yield, min_dividend_years):
    """Print a beautifully formatted results table with colors and clean borders"""
    
    # ANSI color codes for terminal styling
    class Colors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        
    # Check if terminal supports colors
    use_colors = sys.stdout.isatty()
    
    def colorize(text, color_code):
        if use_colors:
            return f"{color_code}{text}{Colors.ENDC}"
        return text
    
    def get_display_width(text):
        """Get the actual display width of text, accounting for ANSI codes and emojis"""
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        # Use wcwidth to get the correct display width for Unicode/emojis
        return sum(wcwidth.wcwidth(char) for char in clean_text)
    
    def pad_text(text, width, align='center'):
        """Pad text to width, accounting for ANSI codes and emojis"""
        display_width = get_display_width(text)
        padding_needed = width - display_width
        
        if align == 'center':
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            return ' ' * left_pad + text + ' ' * right_pad
        elif align == 'left':
            return text + ' ' * padding_needed
        else:  # right
            return ' ' * padding_needed + text
    
    if len(results) == 0:
        print(colorize("\n❌ No companies met all screening criteria.", Colors.FAIL))
        return
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Beautiful header with clean spacing
    title = f"{target_year+1} PERFORMANCE FOR TOP {len(results)} S&P 500 COMPANIES WITH LOWEST {metric_name} RATIOS IN {target_year}"
    subtitle = f"Meeting Dividend Quality Criteria (≥{dividend_yield}% yield, {min_dividend_years}+ years)"
    
    # Define table structure with fixed widths
    headers = ["Ticker", f"{metric_name}", "Div Years", "Div Yield (%)", f"{target_year+1} Start ($)", f"{target_year+1} End ($)", "YoY Gain (%)"]
    col_widths = [8, 8, 12, 15, 18, 16, 15]
    
    # Calculate total table width
    table_width = sum(col_widths) + len(col_widths) + 1  # +1 for each separator + 1 for final border
    
    # Calculate header box width to match table
    header_width = table_width - 2  # -2 for left and right borders
    
    # Header box (seamlessly connected to table)
    print(f"\n{colorize('┌' + '─' * header_width + '┐', Colors.OKBLUE)}")
    print(f"{colorize('│', Colors.OKBLUE)}{title.center(header_width)}{colorize('│', Colors.OKBLUE)}")
    print(f"{colorize('│', Colors.OKBLUE)}{subtitle.center(header_width)}{colorize('│', Colors.OKBLUE)}")
    
    # Connect header to table seamlessly
    print(f"{colorize('├', Colors.OKBLUE)}", end="")
    for i, width in enumerate(col_widths):
        print(colorize('─' * width, Colors.OKBLUE), end="")
        if i < len(col_widths) - 1:
            print(colorize('┬', Colors.OKBLUE), end="")
    print(f"{colorize('┤', Colors.OKBLUE)}")
    
    # Header row
    print(f"{colorize('│', Colors.OKBLUE)}", end="")
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        colored_header = colorize(header, Colors.BOLD + Colors.HEADER)
        padded_header = pad_text(colored_header, width)
        print(padded_header, end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
    print()
    
    # Header separator
    print(f"{colorize('├', Colors.OKBLUE)}", end="")
    for i, width in enumerate(col_widths):
        print(colorize('─' * width, Colors.OKBLUE), end="")
        if i < len(col_widths) - 1:
            print(colorize('┼', Colors.OKBLUE), end="")
    print(f"{colorize('┤', Colors.OKBLUE)}")
    
    # Data rows
    for _, row in df.iterrows():
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 1: Ticker (bold, centered)
        ticker = colorize(str(row['Ticker']), Colors.BOLD + Colors.OKGREEN)
        print(pad_text(ticker, col_widths[0]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 2: P/E or P/B ratio
        ratio = f"{row[metric_name]:.2f}"
        print(pad_text(ratio, col_widths[1]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 3: Dividend years (cyan)
        div_years = colorize(str(row.get('Div Years', 'N/A')), Colors.OKCYAN)
        print(pad_text(div_years, col_widths[2]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 4: Dividend yield (cyan)
        div_yield = colorize(f"{row['Div Yield (%)']:.2f}", Colors.OKCYAN)
        print(pad_text(div_yield, col_widths[3]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 5: Start price
        start_price = row[f'{target_year + 1} Start ($)']
        if start_price is not None:
            start_str = f"${start_price:.2f}"
        else:
            start_str = "N/A"
        print(pad_text(start_str, col_widths[4]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 6: End price
        end_price = row[f'{target_year + 1} End ($)']
        if end_price is not None:
            end_str = f"${end_price:.2f}"
        else:
            end_str = "N/A"
        print(pad_text(end_str, col_widths[5]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}", end="")
        
        # Column 7: YoY Gain with color (no emoji for cleaner spacing)
        gain = row['YoY Gain (%)']
        if gain is not None:
            if gain > 0:
                gain_str = colorize(f"+{gain:.2f}%", Colors.BOLD + Colors.OKGREEN)
            else:
                gain_str = colorize(f"{gain:.2f}%", Colors.BOLD + Colors.FAIL)
        else:
            gain_str = "N/A"
        
        print(pad_text(gain_str, col_widths[6]), end="")
        print(f"{colorize('│', Colors.OKBLUE)}")
    
    # Bottom border
    print(f"{colorize('└', Colors.OKBLUE)}", end="")
    for i, width in enumerate(col_widths):
        print(colorize('─' * width, Colors.OKBLUE), end="")
        if i < len(col_widths) - 1:
            print(colorize('┴', Colors.OKBLUE), end="")
    print(f"{colorize('┘', Colors.OKBLUE)}")
    
    # Summary statistics with enhanced formatting
    valid_gains = [r['YoY Gain (%)'] for r in results if r['YoY Gain (%)'] is not None]
    if valid_gains:
        avg_gain = sum(valid_gains) / len(valid_gains)
        
        print(f"\n{colorize('📊 PERFORMANCE SUMMARY', Colors.BOLD + Colors.HEADER)}")
        print(f"{colorize('─' * 50, Colors.OKBLUE)}")
        
        # Performance summary with colors
        if avg_gain > 0:
            gain_color = Colors.OKGREEN
        else:
            gain_color = Colors.FAIL
        
        print(f"{colorize('Average Return:', Colors.BOLD)} {colorize(f'{avg_gain:+.2f}%', Colors.BOLD + gain_color)}")
        print(f"{colorize('Companies with Data:', Colors.BOLD)} {colorize(f'{len(valid_gains)}/{len(results)}', Colors.OKCYAN)}")
        print(f"{colorize(f'{metric_name} Range:', Colors.BOLD)} {colorize(f'{results[0][metric_name]:.2f} - {results[-1][metric_name]:.2f}', Colors.OKCYAN)}")
        
        # Performance distribution
        positive_gains = [g for g in valid_gains if g > 0]
        negative_gains = [g for g in valid_gains if g <= 0]
        
        print(f"\n{colorize('🎯 PERFORMANCE DISTRIBUTION', Colors.BOLD + Colors.HEADER)}")
        print(f"{colorize('─' * 50, Colors.OKBLUE)}")
        print(f"{colorize('Winners:', Colors.BOLD)} {colorize(f'{len(positive_gains)} companies', Colors.OKGREEN)} {colorize(f'({len(positive_gains)/len(valid_gains)*100:.1f}%)', Colors.OKGREEN)}")
        print(f"{colorize('Losers:', Colors.BOLD)} {colorize(f'{len(negative_gains)} companies', Colors.FAIL)} {colorize(f'({len(negative_gains)/len(valid_gains)*100:.1f}%)', Colors.FAIL)}")
        
        if positive_gains:
            best_gain = max(valid_gains)
            print(f"{colorize('Best Performer:', Colors.BOLD)} {colorize(f'+{best_gain:.2f}%', Colors.BOLD + Colors.OKGREEN)}")
        
        if negative_gains:
            worst_gain = min(valid_gains)
            print(f"{colorize('Worst Performer:', Colors.BOLD)} {colorize(f'{worst_gain:.2f}%', Colors.BOLD + Colors.FAIL)}")

def print_progress_bar(current, total, prefix='Progress', suffix='Complete', length=50):
    """Print a progress bar with percentage"""
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()  # New line when complete

def main(target_year, count=DEFAULT_COUNT, metric=DEFAULT_METRIC, dividend_yield=DEFAULT_DIVIDEND_YIELD, min_dividend_years=DEFAULT_MIN_DIVIDEND_YEARS, verbose=False):
    # Suppress yfinance logging to reduce noise
    import logging
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    
    metric_name = "P/E" if metric == 'pe' else "P/B"
    print(f"\n🔍 S&P 500 Low {metric_name} Dividend Screener for {target_year}")
    print("=" * 70)
    print(f"📈 Step 1: Getting S&P 500 companies for {target_year}...")
    
    tickers = get_sp500_tickers_for_year(target_year)
    print(f"✅ Retrieved {len(tickers)} S&P 500 companies for {target_year}")
    
    # Sort alphabetically for consistent processing order
    tickers.sort()
    print(f"📋 Processing companies alphabetically: {tickers[0]} to {tickers[-1]}")
    print(f"🎯 Using {len(tickers)} S&P 500 companies for {target_year} analysis")
    
    print(f"\n📊 Step 2: Calculating historical {metric_name} ratios for {target_year}...")
    print("Screening criteria:")
    print(f"  📈 Historical {metric_name} ratio calculated for {target_year}")
    print(f"  💰 {min_dividend_years}+ years of dividends ({target_year-min_dividend_years+1} to {target_year})")
    print(f"  📊 Dividend yield ≥ {dividend_yield}% in {target_year}")
    print(f"  🏆 Select {count} companies with LOWEST {metric_name} ratios")
    print("-" * 70)
    
    # Step 1: Calculate ratios for ALL companies
    companies_with_ratio = []
    processed = 0
    errors_summary = {'no_price': 0, 'no_data': 0, 'negative_data': 0, 'extreme_ratio': 0, 'other': 0}
    
    for i, ticker in enumerate(tickers):
        processed += 1
        if verbose:
            print(f"[{processed}/{len(tickers)}] Processing {ticker}...")
        else:
            if processed % 10 == 0 or processed == len(tickers):
                success_count = len(companies_with_ratio)
                print_progress_bar(processed, len(tickers), 
                                 prefix=f'Processing ({success_count} valid {metric_name})', 
                                 suffix=f'{ticker}')
            elif processed == 1:
                print_progress_bar(0, len(tickers), 
                                 prefix=f'Processing S&P 500 companies', 
                                 suffix='Starting...')
                print()  # ADD THIS LINE - creates carriage return after initial progress bar
        
        # Rate limiting
        if i > 0 and i % 10 == 0:
            time.sleep(1)
        
        try:
            stock, should_process = create_ticker_with_suppression(ticker)
            if not should_process:
                errors_summary['other'] += 1
                if verbose:
                    print(f"  ❌ {ticker}: Failed to create ticker object")
                elif processed <= 20:
                    print(f"  ❌ {ticker}: Failed to create ticker object")
                continue
            
            # Calculate the appropriate ratio based on metric
            if metric == 'pe':
                ratio, error_msg = get_pe_for_year(stock, target_year, verbose=verbose)
                data_type = "EPS"
            else:  # pb
                ratio, error_msg = get_pb_for_year(stock, target_year, verbose=verbose)
                data_type = "book value"
            
            if ratio is not None and ratio > 0:
                companies_with_ratio.append({
                    'ticker': ticker,
                    'ratio': ratio,
                    'stock_obj': stock  # Keep for later use
                })
                if verbose:
                    print(f"  ✓ {ticker}: {metric_name} = {ratio:.2f}")
            else:
                # Categorize the error for summary
                if error_msg:
                    if "No price data" in error_msg or "Invalid price" in error_msg:
                        errors_summary['no_price'] += 1
                    elif f"No {data_type.lower()} data" in error_msg.lower() or f"No {data_type} data" in error_msg:
                        errors_summary['no_data'] += 1
                    elif "Negative" in error_msg:
                        errors_summary['negative_data'] += 1
                    elif "Extreme" in error_msg:
                        errors_summary['extreme_ratio'] += 1
                    else:
                        errors_summary['other'] += 1
                
                if verbose:
                    print(f"  ❌ {ticker}: {error_msg}")
                elif processed <= 20:  # Show first 20 errors even in non-verbose mode
                    print(f"  ❌ {ticker}: {error_msg}")
                
        except Exception as e:
            errors_summary['other'] += 1
            error_msg = f"Processing error: {str(e)}"
            if verbose:
                print(f"  ❌ {ticker}: {error_msg}")
            elif processed <= 20:
                print(f"  ❌ {ticker}: {error_msg}")
            continue
    
    print(f"\n✅ Step 3: Found {len(companies_with_ratio)} companies with valid {metric_name} ratios")
    
    # Show prettier error summary
    total_errors = sum(errors_summary.values())
    if total_errors > 0:
        print(f"\n📊 Data Quality Summary ({total_errors} companies failed):")
        print("─" * 50)
        if errors_summary['no_price'] > 0:
            print(f"  📉 No price data: {errors_summary['no_price']} companies")
        if errors_summary['no_data'] > 0:
            print(f"  📋 No {data_type.lower()} data: {errors_summary['no_data']} companies") 
        if errors_summary['negative_data'] > 0:
            print(f"  ⚠️  Negative {data_type.lower()}: {errors_summary['negative_data']} companies")
        if errors_summary['extreme_ratio'] > 0:
            print(f"  📈 Extreme P/B ratios: {errors_summary['extreme_ratio']} companies")
        if errors_summary['other'] > 0:
            print(f"  🔧 Other errors: {errors_summary['other']} companies")
    
    # Step 2: Sort by ratio (lowest first) and apply dividend filters
    companies_with_ratio.sort(key=lambda x: x['ratio'])
    
    print(f"\n🔍 Step 4: Applying dividend filters to find {count} lowest {metric_name} companies...")
    qualified_companies = []
    
    for company in companies_with_ratio:
        ticker = company['ticker']
        ratio = company['ratio']
        stock = company['stock_obj']
        
        if verbose:
            print(f"\nChecking {ticker} ({metric_name}: {ratio:.2f})...")
        
        # Check dividend history
        has_div_hist = has_dividend_history(stock, target_year, min_years=min_dividend_years)
        if not has_div_hist:
            if verbose:
                print(f"  {ticker}: Missing dividend history for {target_year-min_dividend_years+1}-{target_year}")
            continue
        
        # Check dividend yield
        div_yield = get_dividend_yield_for_year(stock, target_year)
        if div_yield is None:
            if verbose:
                print(f"  {ticker}: Could not calculate dividend yield for {target_year}")
            continue
        elif div_yield < dividend_yield:
            if verbose:
                print(f"  {ticker}: Dividend yield {div_yield:.2f}% < {dividend_yield}% requirement")
            continue
        
        # Company qualifies!
        qualified_companies.append({
            'Ticker': ticker,
            metric_name: ratio,
            'Div Years': min_dividend_years,  # Change from 'Consecutive Dividend Years' to match usage
            'Dividend Yield (%)': div_yield,
            'stock_obj': stock
        })
        
        if verbose:
            print(f"  ✓ {ticker}: QUALIFIED ({metric_name}: {ratio:.2f}, Yield: {div_yield:.2f}%)")
        
        # Stop when we have the requested count
        if len(qualified_companies) >= count:
            break
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n🎯 Step 5: Found {len(qualified_companies)} companies meeting all criteria")
    
    if len(qualified_companies) == 0:
        print("❌ No companies met all screening criteria.")
        return
    
    # Step 3: Calculate next year performance and collect price data
    print(f"\n📈 Step 6: Calculating {target_year + 1} performance...")
    
    for i, company in enumerate(qualified_companies):
        ticker = company['Ticker']
        stock = company['stock_obj']
        
        if verbose:
            print(f"  Getting {target_year + 1} performance for {ticker}...")
        
        # Get start and end prices for the next year
        price_start = get_year_start_price(stock, target_year + 1)
        price_end = get_year_end_price(stock, target_year + 1)
        
        # Calculate gain
        if price_start and price_end:
            gain = ((price_end - price_start) / price_start) * 100
        else:
            gain = None
        
        company[f'{target_year + 1} Start Price ($)'] = price_start
        company[f'{target_year + 1} End Price ($)'] = price_end
        company['YoY Gain (%)'] = gain
        
        # Rate limiting
        time.sleep(0.5)
    
    # Remove stock_obj before displaying
    results = []
    for company in qualified_companies:
        results.append({
            'Ticker': company['Ticker'],
            metric_name: company[metric_name],
            'Div Years': company['Div Years'],  # Now matches the field name set above
            'Div Yield (%)': company['Dividend Yield (%)'],
            f'{target_year + 1} Start ($)': company[f'{target_year + 1} Start Price ($)'],
            f'{target_year + 1} End ($)': company[f'{target_year + 1} End Price ($)'],
            'YoY Gain (%)': company['YoY Gain (%)']
        })
    
    # Display results with prettier formatting
    print_pretty_results_table(results, metric_name, target_year, dividend_yield, min_dividend_years)
    
    # Summary statistics (now handled in the pretty table function)
    valid_gains = [r['YoY Gain (%)'] for r in results if r['YoY Gain (%)'] is not None]
    
    print(f"\n📋 Methodology Summary:")
    print("─" * 50)
    print(f"  1️⃣  Analyzed all {len(tickers)} S&P 500 companies for {target_year}")
    print(f"  2️⃣  Calculated historical {metric_name} ratios using {target_year} data")
    print(f"  3️⃣  Sorted by {metric_name} ratio (lowest first)")
    print(f"  4️⃣  Applied dividend filters to find {count} lowest {metric_name} companies")
    print(f"  5️⃣  Calculated subsequent year ({target_year + 1}) performance")
    
    # Show data quality recommendations if there were significant errors
    if len(valid_gains) > 0:
        total_errors = sum([v for v in [
            sum(locals().get('errors_summary', {}).values()) if 'errors_summary' in locals() else 0
        ]])
        total_companies = len(tickers) if 'tickers' in locals() else len(results)
        
        if total_companies > 0 and total_errors > total_companies * 0.1:  # If >10% of tickers had errors
            print(f"\n⚠️  DATA QUALITY NOTICE:")
            print(f"   {total_errors} out of {total_companies} companies ({total_errors/total_companies*100:.1f}%) had data issues.")
            print(f"   This may indicate Yahoo Finance API limitations.")
            print(f"   For production use, consider alternative data sources.")
            print(f"   Run with --help to see data source recommendations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
S&P 500 Value Dividend Screener - Historical Analysis & Performance Tracking

OVERVIEW:
This tool implements a systematic value investing strategy by identifying historically undervalued
dividend-paying companies from the S&P 500 and measuring their subsequent performance. It enables
rigorous backtesting of value investing principles using accurate historical data.

CORE HYPOTHESIS:
Companies trading at low valuation multiples (P/E or P/B ratios) while maintaining consistent
dividend payments often outperform the market in following years, providing both capital
appreciation and income generation.

METHODOLOGY:
1. DATA SOURCING: Retrieves accurate S&P 500 membership for the target year (avoiding survivorship bias)
2. VALUATION SCREENING: Calculates historical P/E or P/B ratios using actual year-end financial data
3. DIVIDEND FILTERING: Applies dividend yield and payment history quality filters
4. SELECTION: Ranks and selects the N companies with the LOWEST valuation ratios
5. PERFORMANCE TRACKING: Measures total return (price + dividends) over the subsequent year

KEY FEATURES:
• Uses comprehensive historical S&P 500 constituent data (avoids missing delisted companies)
• Implements proper historical point-in-time analysis (no look-ahead bias)
• Supports both P/E and P/B ratio screening methodologies
• Configurable screening parameters for strategy optimization
• Detailed error handling and progress reporting

USE CASES:
• Backtest value investing strategies across different market cycles
• Compare effectiveness of P/E vs P/B screening approaches
• Optimize dividend yield and payment history requirements
• Research impact of portfolio size on risk-adjusted returns
• Validate academic studies on value premium with real data

EXAMPLE USAGE:
# Screen 2019 for top 15 low P/B stocks with 3%+ dividend yield:
python value_price_screener.py 2019 --count 15 --metric pb --dividend-yield 3.0

# Test low P/E strategy with stricter dividend history requirements:
python value_price_screener.py 2018 --metric pe --min-dividend-years 10

# Conservative high-yield value approach:
python value_price_screener.py 2020 --count 10 --dividend-yield 4.0 --min-dividend-years 15
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DATA SOURCES & RELIABILITY:
• S&P 500 Membership: Daily historical constituent data from hanshof/sp500_constituents (GitHub)
• Financial Metrics: Yahoo Finance API with comprehensive error handling and validation
• Dividend History: Multi-year dividend payment records and yield calculations
• Local Caching: Automatically caches data to reduce API calls and improve performance

⚠️  DATA QUALITY CONSIDERATIONS:
Yahoo Finance free API has known limitations including:
- Rate limiting that can cause processing delays
- Missing historical data for some companies (especially around corporate events)
- Inconsistent data quality for delisted or acquired companies

ALTERNATIVE DATA SOURCES FOR PRODUCTION USE:
1. Alpha Vantage API (free tier) - more reliable, better rate limits
2. Financial Modeling Prep API (free tier) - comprehensive financial data
3. Quandl/Nasdaq Data Link (paid) - institutional-grade data quality
4. EDGAR SEC filings (free) - official company data, requires processing

TECHNICAL NOTES:
• Supports analysis from 2009 onwards (requires 2+ years of historical data)
• Implements rate limiting to respect API usage guidelines  
• Handles corporate actions, splits, and data quality issues automatically
• Results include detailed performance attribution and statistical summaries

For implementation details and calculation methodologies, see the script header documentation.
        """
    )
    parser.add_argument("year", type=int, 
                        help="Target year for historical analysis (e.g., 2019 to analyze 2019 data and measure 2020 performance)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, 
                        help=f"Number of top-ranked companies to select for the portfolio (default: {DEFAULT_COUNT})")
    parser.add_argument("--metric", choices=['pe', 'pb'], default=DEFAULT_METRIC,
                        help=f"Valuation metric for screening: 'pe' for Price-to-Earnings ratio, 'pb' for Price-to-Book ratio (default: {DEFAULT_METRIC})")
    parser.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD,
                        help=f"Minimum dividend yield percentage required (e.g., 2.5 for 2.5%% yield) (default: {DEFAULT_DIVIDEND_YIELD})")
    parser.add_argument("--min-dividend-years", type=int, default=DEFAULT_MIN_DIVIDEND_YEARS,
                        help=f"Minimum consecutive years of dividend payments required (ensures dividend quality) (default: {DEFAULT_MIN_DIVIDEND_YEARS})")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable detailed progress output showing individual ticker processing and ratios")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s version {__version__}")
    
    args = parser.parse_args()
    
    # Validate year
    current_year = datetime.now().year
    if args.year < 2009 or args.year > current_year - 2:
        print(f"Error: Year must be between 2009 and {current_year - 2}")
        print("(Need at least 2 years of historical data)")
        exit(1)
    
    # Validate other parameters
    if args.count <= 0:
        print("Error: Count must be a positive integer")
        exit(1)
    
    if args.dividend_yield < 0:
        print("Error: Dividend yield must be non-negative")
        exit(1)
    
    if args.min_dividend_years <= 0:
        print("Error: Minimum dividend years must be a positive integer")
        exit(1)
    
    main(args.year, count=args.count, metric=args.metric, dividend_yield=args.dividend_yield, 
         min_dividend_years=args.min_dividend_years, verbose=args.verbose)