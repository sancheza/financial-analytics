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

class EdgarClient:
    def __init__(self, email: str):
        """Initialize SEC EDGAR client with required user agent"""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Host": "data.sec.gov",
            "From": email
        }
        
    def _make_request(self, url: str) -> requests.Response:
        """Make request to SEC EDGAR API with proper headers and error handling"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            raise

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for ticker using mapping file"""
        try:
            # Hard-code known CIKs for testing
            cik_mapping = {
                'PFE': '0000078003',
                'AAPL': '0000320193',
                'MSFT': '0000789019',
                'GOOGL': '0001652044'
            }
            
            ticker = ticker.upper()
            if ticker in cik_mapping:
                return cik_mapping[ticker]
            
            # If not in mapping, try SEC API
            url = "https://www.sec.gov/include/ticker.txt"
            response = self._make_request(url)
            
            # Parse tab-delimited file
            for line in response.text.splitlines():
                if '\t' in line:
                    symbol, cik = line.strip().split('\t', 1)
                    if symbol.upper() == ticker:
                        return str(int(cik)).zfill(10)
            
            logger.warning(f"No CIK found for ticker {ticker}")
            return None
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"HTTP Status Code: {e.response.status_code}")
                logger.error(f"Response Text: {e.response.text[:500]}")
            return None
            
    def get_company_facts(self, cik: str) -> Optional[Dict]:
        """Get company financial facts from SEC EDGAR"""
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            response = self._make_request(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting company facts for CIK {cik}: {e}")
            return None

    def get_latest_value(self, facts: Dict, concept: str, unit: str = 'USD') -> Optional[float]:
        """Get the most recent value for a given concept from company facts"""
        try:
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            if concept in us_gaap and unit in us_gaap[concept].get('units', {}):
                values = us_gaap[concept]['units'][unit]
                # Filter for annual 10-K reports and get most recent
                annual_values = [v for v in values if v.get('form') == '10-K']
                if annual_values:
                    return sorted(annual_values, key=lambda x: x['end'])[-1]['val']
            return None
        except Exception as e:
            logger.error(f"Error getting latest value for {concept}: {e}")
            return None

    def get_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive financial data from SEC EDGAR"""
        data = {}
        cik = self.get_cik(ticker)
        if not cik:
            return data

        facts = self.get_company_facts(cik)
        if not facts:
            return data

        try:
            # Get latest income statement metrics
            net_income = self.get_latest_value(facts, 'NetIncomeLoss')
            revenue = self.get_latest_value(facts, 'Revenues')
            data['netIncome'] = net_income
            data['revenue'] = revenue

            # Balance sheet metrics
            total_assets = self.get_latest_value(facts, 'Assets')
            total_liabilities = self.get_latest_value(facts, 'Liabilities')
            data['totalAssets'] = total_assets
            data['totalLiabilities'] = total_liabilities

            # Cash flow metrics
            operating_cash_flow = self.get_latest_value(facts, 'NetCashProvidedByUsedInOperatingActivities')
            capex = self.get_latest_value(facts, 'PaymentsToAcquirePropertyPlantAndEquipment')
            data['operatingCashFlow'] = operating_cash_flow
            data['capitalExpenditures'] = capex

            # Market data and ratios
            shares_outstanding = self.get_latest_value(facts, 'CommonStockSharesOutstanding', 'shares')
            if shares_outstanding:
                data['sharesOutstanding'] = shares_outstanding

            # Get historical EPS data
            eps_df = self.get_historical_eps(ticker)
            if eps_df is not None and not eps_df.empty:
                data['historicalEPS'] = eps_df.to_dict('records')

            return data

        except Exception as e:
            logger.error(f"Error getting financial data for {ticker}: {e}")
            return data
            
    def get_historical_eps(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical EPS data from SEC filings"""
        cik = self.get_cik(ticker)
        if not cik:
            return None
            
        facts = self.get_company_facts(cik)
        if not facts:
            return None
            
        try:
            eps_tags = [
                'EarningsPerShareDiluted',
                'EarningsPerShareBasic',
                'IncomeLossFromContinuingOperationsPerDilutedShare'
            ]
            
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            eps_data = []
            
            for tag in eps_tags:
                if tag in us_gaap:
                    units = us_gaap[tag].get('units', {})
                    if 'USD/shares' in units:
                        values = units['USD/shares']
                        for v in values:
                            if 'form' in v and v['form'] == '10-K':
                                eps_data.append({
                                    'date': v['end'],
                                    'eps': v['val'],
                                    'tag': tag,
                                    'fp': 'FY'  # Annual data from 10-K
                                })
                                
            if eps_data:
                df = pd.DataFrame(eps_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                return df
                
            logger.error(f"No EPS data found for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error processing EPS data for {ticker}: {e}")
            return None

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
    'ALPHA_VANTAGE_KEY': os.getenv('ALPHA_VANTAGE_KEY', 'demo')  # Get API key from environment
}

# Data storage settings
DATA_DIR = Path('./data/json')
DATA_FILE = DATA_DIR / 'stock_data.json'

def ensure_data_dir():
    """Ensure the data directory exists"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_sp500_tickers():
    """Get list of S&P 500 tickers using pandas_datareader"""
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return sp500['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []

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

def check_positive_earnings_streak(income_stmt_df, years=8):
    """Check if earnings have been positive for the specified number of years"""
    try:
        if len(income_stmt_df) >= years:
            net_income = income_stmt_df['netIncome'].astype(float)
            return all(income > 0 for income in net_income[:years])
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

def calculate_roic(income_stmt_df, balance_sheet_df):
    """Calculate Return on Invested Capital"""
    try:
        if not income_stmt_df.empty and not balance_sheet_df.empty:
            net_income = float(income_stmt_df.iloc[0]['netIncome'])
            total_assets = float(balance_sheet_df.iloc[0]['totalAssets'])
            current_liabilities = float(balance_sheet_df.iloc[0]['totalCurrentLiabilities'])
            
            invested_capital = total_assets - current_liabilities
            if invested_capital != 0:
                roic = (net_income / invested_capital) * 100
                return roic
        logger.debug("Missing data for ROIC calculation")
        return None
    except Exception as e:
        logger.debug(f"Error calculating ROIC: {e}")
        return None

def modern_graham_screen(ticker: str, use_local: bool = False) -> Optional[Dict[str, Any]]:
    """Screen a stock using Graham's criteria with SEC EDGAR data"""
    if use_local:
        data = load_stock_data()
        if ticker in data:
            stored_data = data[ticker]
            if stored_data:
                if any(stored_data.get(field) is None for field in ["Balance Sheet Ratio", "FCF Yield (%)", "ROIC (%)", "10Y Earnings Growth (%)"]):
                    logger.debug(f"{ticker}: Some financial ratios need updating. Run with --update to refresh data.")
                return stored_data
            logger.debug(f"No valid data found for {ticker}")
            return None

    # Only proceed with API calls if not in local mode
    try:
        if not use_local:
            time.sleep(SETTINGS['REQUEST_DELAY'])
        
        # Initialize SEC EDGAR client
        edgar_client = EdgarClient("example@user.com")  # Replace with actual email
        
        # Get financial data from SEC EDGAR
        sec_data = edgar_client.get_financial_data(ticker)
        
        # If SEC data is incomplete, fall back to Alpha Vantage
        if not sec_data or not all(k in sec_data for k in ['netIncome', 'totalAssets', 'totalLiabilities']):
            logger.debug(f"Incomplete SEC data for {ticker}, falling back to Alpha Vantage")
            av_client = AlphaVantageClient(SETTINGS['ALPHA_VANTAGE_KEY'])
            overview = av_client.get_overview(ticker)
            income_stmt = av_client.get_income_statement(ticker)
            balance_sheet = av_client.get_balance_sheet(ticker)
            cash_flow = av_client.get_cash_flow(ticker)
        else:
            overview = {}
            income_stmt = pd.DataFrame([sec_data]) if 'netIncome' in sec_data else pd.DataFrame()
            balance_sheet = pd.DataFrame([{
                'totalAssets': sec_data.get('totalAssets'),
                'totalLiabilities': sec_data.get('totalLiabilities')
            }]) if 'totalAssets' in sec_data else pd.DataFrame()
            cash_flow = pd.DataFrame([{
                'operatingCashflow': sec_data.get('operatingCashFlow'),
                'capitalExpenditures': sec_data.get('capitalExpenditures')
            }]) if 'operatingCashFlow' in sec_data else pd.DataFrame()

        # Calculate P/E using EPS from SEC data
        eps_df = edgar_client.get_historical_eps(ticker)
        if eps_df is not None and not eps_df.empty:
            latest_eps = eps_df.iloc[0]['eps']
            # Get current stock price from yfinance
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            pe = current_price / latest_eps if latest_eps != 0 else None
        else:
            pe = float(overview.get('PERatio', 0)) or None

        # Get P/B from Alpha Vantage if needed
        pb = float(overview.get('PriceToBookRatio', 0)) or None
        
        # Calculate market cap using shares outstanding from SEC
        if 'sharesOutstanding' in sec_data:
            shares = sec_data['sharesOutstanding']
            stock = yf.Ticker(ticker)
            price = stock.history(period='1d')['Close'].iloc[-1]
            market_cap = shares * price
        else:
            market_cap = float(overview.get('MarketCapitalization', 0)) or None

        # Get revenue
        revenue = sec_data.get('revenue') or (float(income_stmt.iloc[0]['totalRevenue']) if not income_stmt.empty else None)

        # P/E × P/B
        pe_pb_combo = pe * pb if pe and pb else None

        # FCF Yield = (CFO - CapEx) / Market Cap
        if 'operatingCashFlow' in sec_data and 'capitalExpenditures' in sec_data and market_cap:
            cfo = sec_data['operatingCashFlow']
            capex = abs(sec_data['capitalExpenditures'])
            fcf = cfo - capex
            fcf_yield = (fcf / market_cap) * 100
        elif not cash_flow.empty and market_cap:
            cfo = float(cash_flow.iloc[0]['operatingCashflow'])
            capex = abs(float(cash_flow.iloc[0]['capitalExpenditures']))
            fcf = cfo - capex
            fcf_yield = (fcf / market_cap) * 100
        else:
            fcf_yield = None

        # Balance sheet ratio from SEC data
        if 'totalAssets' in sec_data and 'totalLiabilities' in sec_data:
            total_assets = sec_data['totalAssets']
            total_liabilities = sec_data['totalLiabilities']
            balance_sheet_ratio = (total_assets / total_liabilities) if total_liabilities != 0 else None
        elif not balance_sheet.empty:
            total_assets = float(balance_sheet.iloc[0]['totalAssets'])
            total_liabilities = float(balance_sheet.iloc[0]['totalLiabilities'])
            balance_sheet_ratio = (total_assets / total_liabilities) if total_liabilities != 0 else None
        else:
            balance_sheet_ratio = None

        # Additional criteria using SEC data when possible
        roic = calculate_roic(income_stmt, balance_sheet)
        earnings_growth = get_earnings_growth(ticker, edgar_client)
        has_positive_earnings = all(float(eps['eps']) > 0 for eps in sec_data.get('historicalEPS', [])[:SETTINGS['POSITIVE_EARNINGS_YEARS']]) if 'historicalEPS' in sec_data else check_positive_earnings_streak(income_stmt, SETTINGS['POSITIVE_EARNINGS_YEARS'])
        
        # Check dividend history using yfinance
        stock = yf.Ticker(ticker)
        has_dividend_history = check_dividend_history(stock, SETTINGS['DIVIDEND_HISTORY_YEARS'])

        # Store all the data
        stock_data = {
            "Ticker": ticker,
            "P/E": pe,
            "P/B": pb,
            "P/E×P/B": pe_pb_combo,
            "Balance Sheet Ratio": balance_sheet_ratio,
            "FCF Yield (%)": fcf_yield,
            "ROIC (%)": roic,
            "10Y Earnings Growth (%)": earnings_growth,
            "Has 8Y+ Positive Earnings": has_positive_earnings,
            "Has Required Dividend History": has_dividend_history,
            "Market Cap ($B)": market_cap / 1e9 if market_cap else None,
            "Revenue ($B)": revenue / 1e9 if revenue else None,
            "Last Updated": datetime.datetime.now().isoformat(),
            "Data Source": "SEC EDGAR" if sec_data else "Alpha Vantage"
        }

        # Log data validation results
        failed_criteria = []
        if pe and pe >= SETTINGS['PE_RATIO_MAX']:
            failed_criteria.append(f"P/E ratio {pe:.1f} >= {SETTINGS['PE_RATIO_MAX']}")
        if pe_pb_combo and pe_pb_combo > SETTINGS['PE_PB_COMBO_MAX']:
            failed_criteria.append(f"P/E×P/B {pe_pb_combo:.1f} > {SETTINGS['PE_PB_COMBO_MAX']}")
        if balance_sheet_ratio and balance_sheet_ratio < SETTINGS['BALANCE_SHEET_RATIO_MIN']:
            failed_criteria.append(f"Balance sheet ratio {balance_sheet_ratio:.1f} < {SETTINGS['BALANCE_SHEET_RATIO_MIN']}")
        if not has_positive_earnings:
            failed_criteria.append("Missing 8+ years of positive earnings")
        if fcf_yield and fcf_yield <= SETTINGS['FCF_YIELD_MIN']:
            failed_criteria.append(f"FCF yield {fcf_yield:.1f}% <= {SETTINGS['FCF_YIELD_MIN']}%")
        if roic and roic < SETTINGS['ROIC_MIN']:
            failed_criteria.append(f"ROIC {roic:.1f}% < {SETTINGS['ROIC_MIN']}%")
        if not has_dividend_history:
            failed_criteria.append(f"No required dividend history")
        if earnings_growth and earnings_growth < SETTINGS['EARNINGS_GROWTH_MIN']:
            failed_criteria.append(f"10Y earnings growth {earnings_growth:.1f}% < {SETTINGS['EARNINGS_GROWTH_MIN']}%")

        if failed_criteria:
            logger.debug(f"{ticker} failed criteria: {'; '.join(failed_criteria)}")

        # Check if stock meets all criteria
        meets_criteria = all([
            pe and pe < SETTINGS['PE_RATIO_MAX'],
            pe_pb_combo and pe_pb_combo <= SETTINGS['PE_PB_COMBO_MAX'],
            balance_sheet_ratio and balance_sheet_ratio >= SETTINGS['BALANCE_SHEET_RATIO_MIN'],
            has_positive_earnings,
            fcf_yield and fcf_yield > SETTINGS['FCF_YIELD_MIN'],
            roic and roic >= SETTINGS['ROIC_MIN'],
            has_dividend_history,
            earnings_growth and earnings_growth >= SETTINGS['EARNINGS_GROWTH_MIN']
        ])

        stock_data["Meets All Criteria"] = meets_criteria
        return stock_data

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Value Stock Screener')
    parser.add_argument('--local', action='store_true', 
                       help='Use only locally stored data without fetching missing values')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    parser.add_argument('--update', nargs='*', 
                       help='Update data for specific tickers or all if none specified')
    parser.add_argument('--max-age', type=int, 
                       help='Maximum age of data in days before requiring refresh')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Test with just PFE
    stored_data = load_stock_data()
    tickers = ['PFE']
    
    results = []
    all_data = stored_data.copy()
    
    print(f"Screening {len(tickers)} stocks...")
    
    # Process tickers
    for ticker in tickers:
        try:
            if args.local and ticker not in stored_data:
                logger.warning(f"No local data found for {ticker}")
                continue
                
            result = modern_graham_screen(ticker, args.local)
            if result:
                results.append(result)
                all_data[ticker] = result
            print(f"Processed {ticker}", end='\r')
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not args.local or args.update is not None:
        save_stock_data(all_data)
    
    # Create DataFrame and filter for stocks that meet all criteria
    df = pd.DataFrame(results)
    print("\nResults for PFE:")
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No data found for PFE")
    
    if args.debug or args.update is not None:
        print("\nData saved in:", DATA_FILE)

if __name__ == "__main__":
    main()
