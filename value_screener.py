#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import datetime
from datetime import date, timedelta
import pandas_datareader.data as web
import json
import os
import argparse
from pathlib import Path
import logging
import time
import requests
from typing import Dict, Any, Optional, List
from alpha_vantage.fundamentaldata import FundamentalData

class AlphaVantageClient:
    def __init__(self, api_key):
        self.client = FundamentalData(key=api_key)
        
    def get_income_statement(self, ticker):
        """Get annual income statements"""
        try:
            income_stmt, _ = self.client.get_income_statement_annual(ticker)
            return pd.DataFrame(income_stmt)
        except Exception as e:
            logger.error(f"Error fetching income statement for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_balance_sheet(self, ticker):
        """Get annual balance sheets"""
        try:
            balance_sheet, _ = self.client.get_balance_sheet_annual(ticker)
            return pd.DataFrame(balance_sheet)
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_cash_flow(self, ticker):
        """Get annual cash flow statements"""
        try:
            cash_flow, _ = self.client.get_cash_flow_annual(ticker)
            return pd.DataFrame(cash_flow)
        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_overview(self, ticker):
        """Get company overview including key metrics"""
        try:
            overview, _ = self.client.get_company_overview(ticker)
            return overview
        except Exception as e:
            logger.error(f"Error fetching overview for {ticker}: {e}")
            return {}

    def get_market_cap(self, ticker):
        """Get market capitalization"""
        try:
            overview = self.get_overview(ticker)
            return float(overview.get('MarketCapitalization', 0))
        except Exception as e:
            logger.error(f"Error fetching market cap for {ticker}: {e}")
            return None

    def get_dividend_history(self, ticker):
        """Get dividend history using yfinance as Alpha Vantage doesn't provide historical dividends"""
        try:
            stock = yf.Ticker(ticker)
            return stock.dividends
        except Exception as e:
            logger.error(f"Error fetching dividend history for {ticker}: {e}")
            return None

class EdgarClient:
    def __init__(self):
        self.ticker_to_cik = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.headers = {
            'User-Agent': 'FinancialAnalytics/1.0 (asanchez@example.com)',  # Required by SEC
            'Accept': 'application/json',
        }

    def _wait_for_rate_limit(self):
        """Ensure we wait at least min_request_interval between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get the CIK for a given ticker symbol."""
        if ticker in self.ticker_to_cik:
            return self.ticker_to_cik[ticker]

        self._wait_for_rate_limit()
        
        try:
            response = requests.get(
                'https://www.sec.gov/files/company_tickers_exchange.json',
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Search through the data for the ticker
            for entry in data['data']:
                if entry[2] == ticker:  # Index 2 contains the ticker symbol
                    cik = str(entry[0]).zfill(10)  # Index 0 contains the CIK
                    self.ticker_to_cik[ticker] = cik
                    return cik
                    
            logging.warning(f"No CIK found for ticker {ticker}")
            return None
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting CIK for {ticker}: {str(e)}")
            return None

    def get_financial_data(self, cik: str, concept: str) -> Optional[Dict]:
        """Get financial data for a given CIK and concept."""
        if not cik:
            return None

        self._wait_for_rate_limit()
        
        try:
            url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting financial data for CIK {cik}, concept {concept}: {str(e)}")
            return None

    def get_financial_data_with_alternatives(self, cik: str, concepts: List[str]) -> Optional[Dict]:
        """Try multiple concept names and return the first one that works"""
        for concept in concepts:
            data = self.get_financial_data(cik, concept)
            if data and 'units' in data:
                return data
        return None

    def get_historical_eps(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical EPS data from SEC filings"""
        cik = self.get_cik(ticker)
        if not cik:
            return None
            
        try:
            # Get EPS data from SEC API
            eps_data = self.get_financial_data(cik, 'EarningsPerShareDiluted')
            if not eps_data or 'units' not in eps_data:
                return None
                
            # Extract EPS values from annual reports
            eps_values = []
            for unit_type in eps_data['units']:
                for entry in eps_data['units'][unit_type]:
                    if entry.get('form') == '10-K':  # Only use annual reports
                        eps_values.append({
                            'date': pd.to_datetime(entry['end']),
                            'eps': entry['val']
                        })
            
            if not eps_values:
                return None
                
            # Convert to DataFrame and sort by date
            eps_df = pd.DataFrame(eps_values)
            return eps_df.sort_values('date', ascending=False)
            
        except Exception as e:
            logger.error(f"Error getting historical EPS for {ticker}: {str(e)}")
            return None

    def get_sector_specific_metrics(self, cik: str, ticker: str) -> Dict[str, Any]:
        """Get sector-specific financial metrics"""
        # Try various sector-specific concept names
        metrics = {}
        
        # Financial sector specific metrics
        financial_metrics = [
            ('InterestAndDividendIncomeOperating', 'NetInterestIncome'),
            ('NoninterestExpense', 'OperatingExpenses'),
            ('ProvisionForLoanAndLeaseLosses', 'LoanLossProvision'),
            ('NonInterestIncome', 'NonInterestIncome'),
            ('InterestIncome', 'InterestIncome')
        ]
        
        for primary, fallback in financial_metrics:
            data = self.get_financial_data_with_alternatives(cik, [primary, fallback])
            if data:
                val = get_latest_annual_value(data)
                if val is not None:
                    metrics[primary] = val
        
        return metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Screening criteria settings
SETTINGS = {
    'PE_RATIO_MAX': 18,           # Default: 15
    'PE_PB_COMBO_MAX': 50,      # Default: 22.5
    'BALANCE_SHEET_RATIO_MIN': 2,  # Default: 2
    'POSITIVE_EARNINGS_YEARS': 8,  # Default: 8
    'FCF_YIELD_MIN': 5,           # Default: 8%
    'ROIC_MIN': 10,               # Default: 10%
    'DIVIDEND_HISTORY_YEARS': 5,  # Default: 10
    'EARNINGS_GROWTH_MIN': 33,     # Default: 33%
    'REQUEST_DELAY': 2,           # Delay between API requests in seconds
    'DATA_MAX_AGE_DAYS': 7,        # Maximum age of data before requiring refresh
    'ALPHA_VANTAGE_KEY': 'MBSVCBG83NNOZ197'  # Alpha Vantage API key
}

# Data storage settings
DATA_DIR = Path('./data/json')
DATA_FILE = DATA_DIR / 'stock_data.json'

def ensure_data_dir():
    """Ensure the data directory exists"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_sp500_tickers():
    """Get list of test tickers (temporary override)"""
    return ['PFE', 'AAPL', 'MRNA', 'MSFT', 'MS']

def load_stock_data() -> Dict[str, Any]:
    """Load stock data from JSON file"""
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
    return {}

def save_stock_data(data: Dict[str, Any]):
    """Save stock data to JSON file"""
    ensure_data_dir()
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving stock data: {e}")

def data_needs_update(data: Dict[str, Any], max_age_days: int = None) -> bool:
    """Check if data needs to be updated based on age"""
    if not data or "Last Updated" not in data:
        return True
        
    max_age = SETTINGS['DATA_MAX_AGE_DAYS']
    if max_age_days is not None:  # Override default if max_age_days is provided
        max_age = max_age_days
        
    try:
        last_updated = datetime.datetime.fromisoformat(data["Last Updated"])
        age = datetime.datetime.now() - last_updated
        return age.days >= max_age
    except (ValueError, TypeError):
        return True

def get_tickers_needing_update(stored_data: Dict[str, Any], tickers: List[str], max_age_days: int = None) -> List[str]:
    """Get list of tickers that need updating"""
    update_needed = []
    for ticker in tickers:
        if ticker not in stored_data or data_needs_update(stored_data.get(ticker, {}), max_age_days):
            update_needed.append(ticker)
    return update_needed

def validate_financial_data(data: Dict[str, Any], ticker: str) -> bool:
    """Validate that we have all required financial data"""
    required_fields = [
        ("P/E", "trailingPE"),
        ("P/B", "priceToBook"),
        ("Market Cap", "marketCap"),
        ("Revenue", "totalRevenue")
    ]
    
    missing = []
    for display_name, field in required_fields:
        if not data.get(field):
            missing.append(display_name)
    
    if missing:
        logger.warning(f"{ticker}: Missing required data: {', '.join(missing)}")
        return False
    return True

def safe_get(df, row_name, default=0):
    try:
        return df.loc[row_name].dropna().iloc[0]
    except:
        return default

def get_earnings_growth(ticker: str, edgar_client: Optional[EdgarClient] = None, years: int = 10) -> Optional[float]:
    """Calculate cumulative earnings growth using SEC EDGAR data if available"""
    try:
        if edgar_client:
            # Try to get historical EPS from SEC EDGAR
            eps_df = edgar_client.get_historical_eps(ticker)
            if eps_df is not None and len(eps_df) >= 5:
                eps_df = eps_df.sort_values('date', ascending=False)
                newest = eps_df.iloc[0]['eps']
                oldest = eps_df.iloc[-1]['eps']
                years_diff = (eps_df.iloc[0]['date'] - eps_df.iloc[-1]['date']).days / 365.25
                
                if oldest != 0 and not pd.isna(oldest) and not pd.isna(newest):
                    growth = (((newest / oldest) ** (1/years_diff)) - 1) * 100
                    logger.debug(f"Calculated {years_diff:.1f}-year earnings growth: {growth:.1f}%")
                    return growth
                    
        logger.debug(f"Insufficient earnings history for growth calculation")
        return None
    except Exception as e:
        logger.debug(f"Error calculating earnings growth: {e}")
        return None

def check_positive_earnings_streak(sec_data, years=8):
    """Check if earnings have been positive for the specified number of years"""
    try:
        if 'historicalEPS' in sec_data and len(sec_data['historicalEPS']) >= years:
            return all(float(eps['eps']) > 0 for eps in sec_data['historicalEPS'][:years])
        logger.debug(f"Insufficient earnings history for streak check")
        return False
    except Exception as e:
        logger.debug(f"Error checking earnings streak: {e}")
        return False

def check_dividend_history(stock, years=20):
    """Check if dividends have been paid uninterrupted for specified years"""
    try:
        dividends = stock.dividends
        if dividends.empty:
            logger.debug("No dividend history found")
            return False
            
        current_year = date.today().year
        earliest_year = dividends.index[0].year
        
        if current_year - earliest_year < years:
            logger.debug(f"Insufficient dividend history: {current_year - earliest_year} years")
            return False
            
        yearly_dividends = dividends.groupby(dividends.index.year).sum()
        recent_years = yearly_dividends[-years:]
        
        return len(recent_years) >= years and all(d > 0 for d in recent_years)
    except Exception as e:
        logger.debug(f"Error checking dividend history: {e}")
        return False

def calculate_roic(sec_data, _):
    """Calculate Return on Invested Capital"""
    try:
        # Get net income
        net_income = sec_data.get('netIncome')
        if not net_income:
            return None
            
        # Get total assets
        total_assets = sec_data.get('totalAssets')
        if not total_assets:
            return None
            
        # Get total liabilities
        total_liabilities = sec_data.get('totalLiabilities')
        if not total_liabilities:
            return None
        
        invested_capital = total_assets - total_liabilities
        if invested_capital != 0:
            roic = (net_income / invested_capital) * 100
            return roic
                
        logger.debug("Missing data for ROIC calculation")
        return None
    except Exception as e:
        logger.debug(f"Error calculating ROIC: {e}")
        return None

def validate_sec_data(sec_data: Dict) -> bool:
    """Validate that SEC data is recent and complete"""
    try:
        if not sec_data:
            return False
            
        # Check if we have all required fields
        required_fields = ['netIncome', 'totalAssets', 'totalLiabilities', 
                         'operatingCashFlow', 'capitalExpenditures']
        return all(field in sec_data for field in required_fields)
    except Exception as e:
        logger.error(f"Error validating SEC data: {e}")
        return False

def calculate_pe_ratio(stock, eps_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Calculate P/E ratio with negative values for negative earnings"""
    try:
        if eps_df is not None and not eps_df.empty:
            latest_eps = eps_df.iloc[0]['eps']
            if latest_eps == 0:
                return None  # Don't calculate P/E for zero earnings
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            return current_price / latest_eps  # Will be negative if eps is negative
        return None
    except Exception as e:
        logger.error(f"Error calculating P/E ratio: {e}")
        return None

def calculate_pe_pb_combo(pe: Optional[float], pb: Optional[float]) -> Optional[float]:
    """Calculate P/E×P/B combo with negative values for negative P/E"""
    try:
        if pe is not None and pb is not None:
            return pe * pb  # Will be negative if P/E is negative
        return None
    except Exception as e:
        logger.error(f"Error calculating P/E×P/B combo: {e}")
        return None

def calculate_fcf_yield(sec_data: Dict, market_cap: float) -> Optional[float]:
    """Calculate FCF yield with better error handling"""
    try:
        if not all(k in sec_data for k in ['operatingCashFlow', 'capitalExpenditures']):
            return None
            
        cfo = sec_data['operatingCashFlow']
        capex = abs(sec_data['capitalExpenditures']) if sec_data['capitalExpenditures'] else 0
        fcf = cfo - capex
        
        if market_cap <= 0:
            return None
            
        fcf_yield = (fcf / market_cap) * 100
        return fcf_yield if fcf_yield > -100 else None  # Cap extreme negative values
    except Exception as e:
        logger.error(f"Error calculating FCF yield: {e}")
        return None

def calculate_earnings_growth(eps_df: pd.DataFrame) -> Optional[float]:
    """Calculate earnings growth with better handling of data quality"""
    try:
        if eps_df is None or len(eps_df) < 2:
            return None
            
        eps_df = eps_df.sort_values('date', ascending=False)
        newest = eps_df.iloc[0]['eps']
        oldest = eps_df.iloc[-1]['eps']
        
        if newest <= 0 or oldest <= 0:
            return None  # Don't calculate growth with negative earnings
            
        years_diff = (eps_df.iloc[0]['date'] - eps_df.iloc[-1]['date']).days / 365.25
        if years_diff < 1:
            return None  # Need at least 1 year of data
            
        growth = (((newest / oldest) ** (1/years_diff)) - 1) * 100
        return growth if abs(growth) < 1000 else None  # Cap extreme growth rates
    except Exception as e:
        logger.error(f"Error calculating earnings growth: {e}")
        return None

def calculate_pb_ratio(sec_data: Dict, market_cap: float) -> Optional[float]:
    """Calculate P/B ratio with better error handling"""
    try:
        if not all(k in sec_data for k in ['totalAssets', 'totalLiabilities']):
            return None
            
        total_assets = float(sec_data['totalAssets'])
        total_liabilities = float(sec_data['totalLiabilities'])
        book_value = total_assets - total_liabilities
        
        if book_value <= 0 or market_cap <= 0:
            return None
            
        return market_cap / book_value
    except Exception as e:
        logger.error(f"Error calculating P/B ratio: {e}")
        return None

def calculate_balance_sheet_ratio(sec_data: Dict) -> Optional[float]:
    """Calculate balance sheet ratio with better error handling"""
    try:
        if not all(k in sec_data for k in ['totalAssets', 'totalLiabilities']):
            return None
            
        total_assets = float(sec_data['totalAssets'])
        total_liabilities = float(sec_data['totalLiabilities'])
        
        if total_liabilities <= 0:
            return None
            
        ratio = total_assets / total_liabilities
        logger.debug(f"Balance sheet ratio calculation: {total_assets} / {total_liabilities} = {ratio}")
        return ratio
    except Exception as e:
        logger.error(f"Error calculating balance sheet ratio: {e}")
        return None

def modern_graham_screen(ticker: str, use_local: bool = False) -> Optional[Dict[str, Any]]:
    """Screen a stock using Graham's criteria with SEC EDGAR data"""
    try:
        if use_local:
            data = load_stock_data()
            if ticker in data:
                stored_data = data[ticker]
                if stored_data and not any(stored_data.get(field) is None for field in 
                                         ["Balance Sheet Ratio", "FCF Yield (%)", "ROIC (%)", "10Y Earnings Growth (%)"]):
                    return stored_data
                logger.debug(f"No valid data found for {ticker}")
                return None

        # Initialize clients
        edgar_client = EdgarClient()
        av_client = AlphaVantageClient(SETTINGS['ALPHA_VANTAGE_KEY'])
        
        # Get data
        sec_data = edgar_client.get_financial_data(edgar_client.get_cik(ticker), 'NetIncomeLoss')
        if not validate_sec_data(sec_data):
            logger.warning(f"Incomplete or invalid SEC data for {ticker}")
            return None
            
        overview = av_client.get_overview(ticker)
        stock = yf.Ticker(ticker)
        
        # Calculate metrics
        eps_df = edgar_client.get_historical_eps(ticker)
        pe = calculate_pe_ratio(stock, eps_df)
        
        # Calculate market cap
        if 'sharesOutstanding' in sec_data:
            shares = sec_data['sharesOutstanding']
            price = stock.history(period='1d')['Close'].iloc[-1]
            market_cap = shares * price
        else:
            market_cap = float(overview.get('MarketCapitalization', 0)) or None

        # Calculate other metrics
        pb = calculate_pb_ratio(sec_data, market_cap)
        pe_pb_combo = calculate_pe_pb_combo(pe, pb)
        fcf_yield = calculate_fcf_yield(sec_data, market_cap)
        roic = calculate_roic(sec_data, sec_data)
        earnings_growth = calculate_earnings_growth(eps_df)
        has_positive_earnings = check_positive_earnings_streak(sec_data)
        has_dividend_history = check_dividend_history(stock, SETTINGS['DIVIDEND_HISTORY_YEARS'])

        # Store results
        stock_data = {
            "Ticker": ticker,
            "P/E": pe,
            "P/B": pb,
            "P/E×P/B": pe_pb_combo,
            "Balance Sheet Ratio": calculate_balance_sheet_ratio(sec_data),
            "FCF Yield (%)": fcf_yield,
            "ROIC (%)": roic,
            "10Y Earnings Growth (%)": earnings_growth,
            "Has 8Y+ Positive Earnings": has_positive_earnings,
            "Has Required Dividend History": has_dividend_history,
            "Market Cap ($B)": market_cap / 1e9 if market_cap else None,
            "Revenue ($B)": sec_data.get('revenue', 0) / 1e9 if sec_data.get('revenue') else None,
            "Last Updated": datetime.datetime.now().isoformat(),
            "Data Source": "SEC EDGAR" if sec_data else "Alpha Vantage",
            "cik": edgar_client.get_cik(ticker)  # Store the CIK
        }

        # Validate against criteria
        stock_data["Meets All Criteria"] = validate_against_criteria(stock_data)
        
        return stock_data

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None

def validate_against_criteria(data: Dict) -> bool:
    """Validate stock data against screening criteria"""
    try:
        return all([
            data.get("P/E") and data["P/E"] < SETTINGS['PE_RATIO_MAX'],
            data.get("P/E×P/B") and data["P/E×P/B"] <= SETTINGS['PE_PB_COMBO_MAX'],
            data.get("Balance Sheet Ratio") and data["Balance Sheet Ratio"] >= SETTINGS['BALANCE_SHEET_RATIO_MIN'],
            data.get("Has 8Y+ Positive Earnings", False),
            data.get("FCF Yield (%)") and data["FCF Yield (%)"] > SETTINGS['FCF_YIELD_MIN'],
            data.get("ROIC (%)") and data["ROIC (%)"] >= SETTINGS['ROIC_MIN'],
            data.get("Has Required Dividend History", False),
            data.get("10Y Earnings Growth (%)") and data["10Y Earnings Growth (%)"] >= SETTINGS['EARNINGS_GROWTH_MIN']
        ])
    except Exception as e:
        logger.error(f"Error validating criteria: {e}")
        return False

def screen_stock(ticker: str) -> Dict[str, Any]:
    """Screen a single stock for value metrics."""
    try:
        # Initialize clients
        edgar_client = EdgarClient()
        stock = yf.Ticker(ticker)
        
        # Get CIK
        cik = edgar_client.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return {}

        metrics = {}
        
        # 1. Net Income (historical data for growth and positive earnings check)
        net_income_data = edgar_client.get_financial_data_with_alternatives(cik, 
            ['NetIncomeLoss', 
             'NetIncome', 
             'ProfitLoss', 
             'IncomeLossFromContinuingOperations',
             'NetIncomeLossAvailableToCommonStockholdersBasic'])
        
        if net_income_data and 'units' in net_income_data:
            # Get historical net income values
            net_income_history = []
            for unit_type in net_income_data['units']:
                for entry in net_income_data['units'][unit_type]:
                    if entry.get('form') == '10-K':
                        net_income_history.append((entry['end'], entry['val']))
            
            # Sort by date
            net_income_history.sort(key=lambda x: x[0], reverse=True)
            
            if net_income_history:
                metrics['net_income'] = net_income_history[0][1]  # Latest
                
                # Count consecutive years of positive earnings from most recent
                positive_years = 0
                for _, val in net_income_history:
                    if val > 0:
                        positive_years += 1
                    else:
                        break  # Stop at first negative/zero earnings
                metrics['Consecutive Positive Earnings Years'] = positive_years
                
                # Calculate 10Y earnings growth if enough data
                if len(net_income_history) >= 10:
                    latest = net_income_history[0][1]
                    oldest = net_income_history[9][1]
                    if oldest > 0:  # Avoid division by zero
                        growth_rate = ((latest / oldest) ** (1/10) - 1) * 100
                        metrics['10Y Earnings Growth (%)'] = growth_rate

        # 2. Balance Sheet Data
        assets_data = edgar_client.get_financial_data_with_alternatives(cik, 
            ['Assets', 'TotalAssets', 'AssetsTotal'])
        liabilities_data = edgar_client.get_financial_data_with_alternatives(cik,
            ['Liabilities', 'TotalLiabilities', 'LiabilitiesTotal'])
        
        if assets_data and liabilities_data:
            latest_assets = get_latest_annual_value(assets_data)
            latest_liabilities = get_latest_annual_value(liabilities_data)
            
            if latest_assets and latest_liabilities:
                metrics['total_assets'] = latest_assets
                metrics['total_liabilities'] = latest_liabilities
                
                # Calculate Balance Sheet Ratio
                if latest_assets > 0:
                    metrics['Balance Sheet Ratio'] = latest_assets / latest_liabilities

        # 3. Free Cash Flow data for FCF Yield
        operating_cash_flow = edgar_client.get_financial_data_with_alternatives(cik,
            ['NetCashProvidedByUsedInOperatingActivities', 
             'OperatingCashFlow',
             'CashFlowFromOperations',
             'NetCashProvidedByOperatingActivities',
             'CashFlowsFromUsedInOperatingActivities'])
        
        capex = edgar_client.get_financial_data_with_alternatives(cik,
            ['PaymentsToAcquirePropertyPlantAndEquipment',
             'CapitalExpenditures',
             'PaymentsForPropertyPlantAndEquipment',
             'InvestmentsInPropertyPlantAndEquipment',
             'CapitalExpendituresIncurredButNotYetPaid',
             'PaymentsToAcquireProductiveAssets'])

        # For financial companies, try alternative cash flow metrics
        if not operating_cash_flow or not capex:
            sector_metrics = edgar_client.get_sector_specific_metrics(cik, ticker)
            if sector_metrics:
                # For financial companies, use change in cash and investments as proxy for FCF
                cash_and_investments = edgar_client.get_financial_data_with_alternatives(cik,
                    ['CashAndCashEquivalentsAtCarryingValue',
                     'CashAndShortTermInvestments',
                     'MarketableSecurities'])
                if cash_and_investments:
                    latest_cash = get_latest_annual_value(cash_and_investments)
                    if latest_cash is not None:
                        # Use change in cash position as proxy for FCF
                        fcf = latest_cash - (sector_metrics.get('NoninterestExpense', 0) or 0)
                        
                        try:
                            info = stock.info
                            market_cap = info.get('marketCap')
                            if market_cap and market_cap > 0:
                                metrics['FCF Yield (%)'] = (fcf / market_cap) * 100
                        except Exception as e:
                            logger.error(f"Error calculating FCF yield for {ticker}: {e}")
        else:
            latest_ocf = get_latest_annual_value(operating_cash_flow)
            latest_capex = get_latest_annual_value(capex)
            
            if latest_ocf and latest_capex:
                fcf = latest_ocf - abs(latest_capex)
                try:
                    info = stock.info
                    market_cap = info.get('marketCap')
                    if market_cap and market_cap > 0:
                        metrics['FCF Yield (%)'] = (fcf / market_cap) * 100
                        
                        # Also calculate P/E since we have market cap
                        if metrics.get('net_income') and metrics['net_income'] > 0:
                            metrics['P/E'] = market_cap / metrics['net_income']
                except Exception as e:
                    logger.error(f"Error getting market data for {ticker}: {e}")

        # 4. Book Value for P/B ratio
        if 'total_assets' in metrics and 'total_liabilities' in metrics:
            book_value = metrics['total_assets'] - metrics['total_liabilities']
            try:
                info = stock.info
                market_cap = info.get('marketCap')
                if market_cap and market_cap > 0 and book_value > 0:
                    metrics['P/B'] = market_cap / book_value
                    if 'P/E' in metrics:
                        metrics['P/E×P/B'] = metrics['P/E'] * metrics['P/B']
            except Exception as e:
                logger.error(f"Error calculating P/B ratio for {ticker}: {e}")

        # 5. Calculate ROIC
        try:
            # For financial companies, use Return on Assets (ROA) as a proxy for ROIC
            if 'total_assets' in metrics:
                operating_income = edgar_client.get_financial_data_with_alternatives(cik,
                    ['OperatingIncomeLoss', 
                     'OperatingIncome', 
                     'EBIT',
                     'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
                     'IncomeLossFromContinuingOperations',
                     'OperatingIncomeLossIncludingIncomeLossFromEquityMethodInvestments',
                     'InterestAndDividendIncomeOperating',  # Financial sector
                     'NetInterestIncome'])  # Financial sector
                
                if operating_income:
                    latest_operating_income = get_latest_annual_value(operating_income)
                    
                    # Estimate NOPAT (assuming 21% corporate tax rate)
                    if latest_operating_income:
                        nopat = latest_operating_income * (1 - 0.21)  # 21% corporate tax rate
                        
                        # For financial companies, use total assets as invested capital
                        if ticker in ['MS', 'GS', 'JPM', 'BAC', 'C', 'WFC']:  # Financial sector
                            invested_capital = metrics['total_assets']
                        else:
                            # Calculate Invested Capital
                            current_liabilities = edgar_client.get_financial_data_with_alternatives(cik,
                                ['LiabilitiesCurrent', 'CurrentLiabilities'])
                            if current_liabilities:
                                latest_current_liabilities = get_latest_annual_value(current_liabilities)
                                if latest_current_liabilities:
                                    # Invested Capital = Total Assets - Current Liabilities
                                    invested_capital = metrics['total_assets'] - latest_current_liabilities
                                else:
                                    invested_capital = metrics['total_assets'] - metrics['total_liabilities']
                            else:
                                invested_capital = metrics['total_assets'] - metrics['total_liabilities']
                        
                        if invested_capital > 0:
                            metrics['ROIC (%)'] = (nopat / invested_capital) * 100
                            logger.info(f"Calculated ROIC for {ticker}: {metrics['ROIC (%)']}%")
        except Exception as e:
            logger.error(f"Error calculating ROIC for {ticker}: {str(e)}")

        # 6. Dividend History
        try:
            dividends = stock.dividends
            if not dividends.empty:
                yearly_dividends = dividends.groupby(dividends.index.year).sum()
                # Count years with positive dividends
                positive_div_years = sum(1 for d in yearly_dividends if d > 0)
                metrics['Positive Dividend Years'] = positive_div_years
                
                # Also include consecutive years of dividends
                consecutive_years = 0
                for div in reversed(yearly_dividends.values):  # Start from most recent
                    if div > 0:
                        consecutive_years += 1
                    else:
                        break
                metrics['Consecutive Dividend Years'] = consecutive_years
            else:
                metrics['Positive Dividend Years'] = 0
                metrics['Consecutive Dividend Years'] = 0
        except Exception as e:
            logger.error(f"Error checking dividend history for {ticker}: {str(e)}")
            metrics['Positive Dividend Years'] = 0
            metrics['Consecutive Dividend Years'] = 0

        return metrics

    except Exception as e:
        logger.error(f"Error screening {ticker}: {str(e)}")
        return {}

def get_latest_annual_value(data: Dict) -> Optional[float]:
    """Extract the latest annual value from SEC API response."""
    try:
        values = []
        for unit_type in data['units']:  # Usually 'USD'
            for entry in data['units'][unit_type]:
                if entry.get('form') == '10-K':  # Only consider annual reports
                    values.append((entry['end'], entry['val']))
        
        if values:
            # Sort by date and get most recent value
            latest = sorted(values, key=lambda x: x[0], reverse=True)[0]
            return latest[1]
        return None
    except Exception as e:
        logger.error(f"Error getting latest value: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Value Stock Screener')
    parser.add_argument('--output', type=str, default='screener_results.json',
                       help='Output JSON file path')
    args = parser.parse_args()

    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    logger.info(f"Retrieved {len(tickers)} tickers from S&P 500")

    # Screen each stock
    results = {}
    for ticker in tickers:
        logger.info(f"Screening {ticker}...")
        metrics = screen_stock(ticker)
        if metrics:  # Only include stocks with data
            results[ticker] = metrics

    # Calculate derived metrics
    for ticker, metrics in results.items():
        try:
            # Remove the duplicate ROIC calculation here since it's now handled in screen_stock
            pass
        except Exception as e:
            logger.error(f"Error calculating metrics for {ticker}: {str(e)}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Screening complete. Processed {len(tickers)} stocks, found data for {len(results)} stocks.")
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
