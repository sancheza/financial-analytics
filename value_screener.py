#!/usr/bin/env python3

__version__ = "1.0.6"

import yfinance as yf
import pandas as pd
import datetime
from datetime import date, timedelta
import pandas_datareader.data as web
import json
import ijson  # Added for memory-efficient JSON parsing
import os
import argparse
from pathlib import Path
import logging
import time
import requests
import math  # Add math import for isnan/isinf functions
from typing import Dict, Any, Optional, List, Tuple
from alpha_vantage.fundamentaldata import FundamentalData
from concurrent.futures import ThreadPoolExecutor  # Added for parallel processing
import sys

# Screening criteria settings
SETTINGS = {
    "PE_RATIO_MAX": 18,  # Default: 15
    "PE_PB_COMBO_MAX": 150,  # Default: 22.5
    "BALANCE_SHEET_RATIO_MIN": 2,  # Default: 2
    "POSITIVE_EARNINGS_YEARS": 8,  # Default: 8
    "FCF_YIELD_MIN": 5,  # Default: 8%
    "ROIC_MIN": 10,  # Default: 10%
    "DIVIDEND_HISTORY_YEARS": 5,  # Default: 10
    "EARNINGS_GROWTH_MIN": 5,  # Default: 33%
    "REQUEST_DELAY": 5,  # Delay between API requests in seconds
    "DATA_MAX_AGE_DAYS": 7,  # Maximum age of data before requiring refresh
    "ALPHA_VANTAGE_KEY": os.environ.get(
        "ALPHA_VANTAGE_KEY", ""
    ),  # Get API key from environment variable
}

# Define data directory and file paths
DATA_DIR = Path("data/json")
DATA_FILE = DATA_DIR / "stock_data.json"

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING level (3)
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Force reconfiguration
)
logger = logging.getLogger(__name__)
# Ensure logging is properly configured for immediate output
logger.setLevel(logging.WARNING)  # Default to WARNING level
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)  # Default to WARNING level
logger.addHandler(handler)
# Ensure stdout is flushed after each write
import sys

sys.stdout.reconfigure(line_buffering=True)

# Define log level mapping for the --debug argument
LOG_LEVELS = {
    0: logging.CRITICAL,  # Only critical errors
    1: logging.ERROR,  # Error and critical
    2: logging.WARNING,  # Warning, error, and critical (default)
    3: logging.INFO,  # Info, warning, error, and critical
    4: logging.DEBUG,  # All messages including debug
}


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
            return float(overview.get("MarketCapitalization", 0))
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
            "User-Agent": "FinancialAnalytics/1.0 (asanchez@example.com)",  # Required by SEC
            "Accept": "application/json",
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
                "https://www.sec.gov/files/company_tickers_exchange.json",
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()

            # Search through the data for the ticker
            for entry in data["data"]:
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
            url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Error getting financial data for CIK {cik}, concept {concept}: {str(e)}"
            )
            return None

    def get_financial_data_with_alternatives(
        self, cik: str, concepts: List[str]
    ) -> Optional[Dict]:
        """Try multiple concept names and return the first one that works.
        Improved to handle multiple fallback options with minimal logging.
        """
        if not cik:
            return None

        result = None
        found_concept = None

        # Try each concept name in order until one works
        for concept in concepts:
            try:
                self._wait_for_rate_limit()
                url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
                response = requests.get(url, headers=self.headers)

                if response.status_code == 200:
                    data = response.json()
                    if "units" in data:
                        result = data
                        found_concept = concept
                        break
                # Don't log 404s as they're expected when trying alternative concepts
                elif response.status_code != 404:
                    logging.debug(f"{response.status_code} for {concept}")
            except Exception as e:
                # Only log non-HTTP errors at debug level
                logging.debug(f"Error for {concept}: {str(e)}")

        if result:
            if found_concept != concepts[0]:
                logging.debug(
                    f"Used alternative concept {found_concept} instead of {concepts[0]}"
                )
            return result
        else:
            # Only log one summary error instead of one per concept
            logging.debug(f"Could not find any concepts for CIK {cik}: {concepts}")
            return None

    def get_historical_eps(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical EPS data from SEC filings"""
        cik = self.get_cik(ticker)
        if not cik:
            return None

        try:
            # Get EPS data from SEC API - try multiple concept names
            eps_data = self.get_financial_data_with_alternatives(
                cik,
                [
                    "EarningsPerShareDiluted",
                    "EarningsPerShareBasic",
                    "EarningsPerShare",
                ],
            )

            if not eps_data or "units" not in eps_data:
                return None

            # Extract EPS values from annual reports
            eps_values = []
            for unit_type in eps_data["units"]:
                for entry in eps_data["units"][unit_type]:
                    if entry.get("form") == "10-K":  # Only use annual reports
                        eps_values.append(
                            {"date": pd.to_datetime(entry["end"]), "eps": entry["val"]}
                        )

            if not eps_values:
                return None

            # Convert to DataFrame and sort by date
            eps_df = pd.DataFrame(eps_values)
            return eps_df.sort_values("date", ascending=False)

        except Exception as e:
            logger.error(f"Error getting historical EPS for {ticker}: {str(e)}")
            return None

    def get_sector_specific_metrics(self, cik: str, ticker: str) -> Dict[str, Any]:
        """Get sector-specific financial metrics"""
        # Try various sector-specific concept names
        metrics = {}

        # Financial sector specific metrics
        financial_metrics = [
            ("InterestAndDividendIncomeOperating", "NetInterestIncome"),
            ("NoninterestExpense", "OperatingExpenses"),
            ("ProvisionForLoanAndLeaseLosses", "LoanLossProvision"),
            ("NonInterestIncome", "NonInterestIncome"),
            ("InterestIncome", "InterestIncome"),
        ]

        for primary, fallback in financial_metrics:
            data = self.get_financial_data_with_alternatives(cik, [primary, fallback])
            if data:
                val = get_latest_annual_value(data)
                if val is not None:
                    metrics[primary] = val

        return metrics


def ensure_data_dir():
    """Make sure the data directory exists"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_tickers():
    """Get list of S&P 500 tickers from Wikipedia."""
    try:
        # Fetch S&P 500 tickers from Wikipedia with proper headers
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        html = requests.get(url, headers=headers, timeout=10).text
        tables = pd.read_html(html)
        sp500_table = tables[0]  # First table contains S&P 500 companies
        tickers = sp500_table["Symbol"].tolist()

        # Clean ticker symbols
        tickers = [
            ticker.replace(".", "-") for ticker in tickers
        ]  # Replace dots with hyphens for Yahoo Finance

        # logger.info(f"Retrieved {len(tickers)} tickers from S&P 500")
        return tickers
    except Exception as e:
        logger.error(f"Error retrieving S&P 500 tickers: {e}")
        # Return a small default list in case of failure
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]


def get_cached_tickers(cached_data: Dict[str, Any]) -> List[str]:
    """Get list of tickers from cached data without any web requests.
    This is used in cache mode to avoid any connections to external services.
    """
    if not cached_data:
        return []

    # Extract only keys that are likely to be ticker symbols (not metadata keys)
    return [
        k
        for k in cached_data.keys()
        if k != "Last Updated" and isinstance(k, str) and len(k) < 6
    ]


def load_stock_data() -> Dict[str, Any]:
    """Load stock data from JSON file using memory-efficient parsing with ijson"""
    try:
        if DATA_FILE.exists():
            # First, check if the file is valid JSON and get the "Last Updated" field
            last_updated = None
            try:
                with open(DATA_FILE, "rb") as f:
                    # Just get the Last Updated field which is at the top level
                    for prefix, event, value in ijson.parse(f):
                        if prefix == "Last Updated" and event == "string":
                            last_updated = value
                            break
            except Exception as e:
                logger.warning(f"Could not parse Last Updated field: {e}")

            # Now load the full file into a pandas DataFrame first for better performance
            result = {"Last Updated": last_updated}
            ticker_data = {}

            # Use pandas to load the file which handles large JSON data more efficiently
            try:
                with open(DATA_FILE, "r") as f:
                    # Read the file as JSON
                    data = json.load(f)

                    # Skip the Last Updated key which we already captured
                    ticker_keys = [k for k in data.keys() if k != "Last Updated"]

                    # Use a DataFrame for more efficient data processing
                    if ticker_keys:
                        # Convert the nested dictionary to a more efficient format
                        df_data = []
                        for ticker in ticker_keys:
                            if isinstance(data[ticker], dict):
                                ticker_info = data[ticker].copy()
                                ticker_info["Ticker"] = ticker
                                df_data.append(ticker_info)

                        if df_data:
                            # Create a DataFrame with all ticker data
                            df = pd.DataFrame(df_data)

                            # Convert back to dictionary format but more efficiently
                            for _, row in df.iterrows():
                                ticker = row["Ticker"]
                                # Filter out the Ticker column and convert to dict
                                ticker_data[ticker] = row.drop("Ticker").to_dict()

            except Exception as e:
                logger.error(f"Error converting data to DataFrame: e")
                # Fall back to regular JSON loading if DataFrame approach fails
                with open(DATA_FILE, "r") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if key != "Last Updated" and isinstance(value, dict):
                            ticker_data[key] = value

            # Combine the results
            result.update(ticker_data)
            return result
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
    return {}


def save_stock_data(data: Dict[str, Any]):
    """Save stock data to JSON file with advanced type handling.
    This function preserves existing data while updating with new data.
    """
    ensure_data_dir()
    try:
        # First, load existing data if available
        existing_data = {}
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE, "r") as f:
                    existing_data = json.load(f)
                logger.info(
                    f"Loaded existing data with {len(existing_data) - 1} tickers"
                )  # -1 for Last Updated key
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")

        # Process data to ensure it's JSON serializable
        serializable_data = {}

        # First, add Last Updated field at the top level
        serializable_data["Last Updated"] = datetime.datetime.now().isoformat()

        # Copy all existing ticker data (except "Last Updated")
        for key, value in existing_data.items():
            if key != "Last Updated":
                serializable_data[key] = value

        # Handle each new ticker's data - this will update existing tickers and add new ones
        for ticker, stock_info in data.items():
            if not isinstance(stock_info, dict):
                continue  # Skip non-dictionary items

            cleaned_info = {}
            for key, value in stock_info.items():
                # Handle complex numbers
                if isinstance(value, complex):
                    cleaned_info[key] = str(value)
                # Handle NaN, Infinity
                elif isinstance(value, float) and (
                    math.isnan(value) or math.isinf(value)
                ):
                    cleaned_info[key] = None
                # Handle pandas DataFrame, Series or other objects
                elif hasattr(value, "to_dict"):
                    try:
                        cleaned_info[key] = value.to_dict()
                    except:
                        cleaned_info[key] = str(value)
                # Handle dates and datetimes
                elif isinstance(value, (datetime.date, datetime.datetime)):
                    cleaned_info[key] = value.isoformat()
                # Handle lists and arrays - check each element
                elif isinstance(value, (list, tuple, set)):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, (int, float, str, bool, dict)):
                            if isinstance(item, float) and (
                                math.isnan(item) or math.isinf(item)
                            ):
                                cleaned_list.append(None)
                            elif isinstance(item, complex):
                                cleaned_list.append(str(item))
                            else:
                                cleaned_list.append(item)
                        else:
                            cleaned_list.append(str(item))
                    cleaned_info[key] = cleaned_list
                # Convert any other non-serializable types to strings
                elif not isinstance(value, (int, float, str, bool, dict, type(None))):
                    cleaned_info[key] = str(value)
                else:
                    cleaned_info[key] = value
            serializable_data[ticker] = cleaned_info

        # Write with a clear error handling strategy - use a temporary file approach
        # to avoid corrupting the original file if serialization fails
        temp_file = DATA_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(serializable_data, f, indent=2, default=str)

        # If successful, rename the temp file to the actual file
        temp_file.replace(DATA_FILE)

        logger.info(
            f"Stock data saved successfully to {DATA_FILE} with {len(serializable_data) - 1} tickers"
        )

    except Exception as e:
        logger.error(f"Error saving stock data: {e}")

        # Try to create an empty but valid JSON file as fallback
        try:
            with open(DATA_FILE, "w") as f:
                json.dump({"Last Updated": datetime.datetime.now().isoformat()}, f)
            logger.warning("Created empty stock data file as fallback")
        except:
            logger.error("Failed to create even an empty stock data file")


def data_needs_update(data: Dict[str, Any], max_age_days: int = None) -> bool:
    """Check if data needs to be updated based on age"""
    if not data or "Last Updated" not in data:
        return True

    max_age = SETTINGS["DATA_MAX_AGE_DAYS"]
    if max_age_days is not None:  # Override default if max_age_days is provided
        max_age = max_age_days

    try:
        last_updated = datetime.datetime.fromisoformat(data["Last Updated"])
        age = datetime.datetime.now() - last_updated
        return age.days >= max_age
    except (ValueError, TypeError):
        return True


def get_tickers_needing_update(
    stored_data: Dict[str, Any], tickers: List[str], max_age_days: int = None
) -> List[str]:
    """Get list of tickers that need updating"""
    update_needed = []
    for ticker in tickers:
        if ticker not in stored_data or data_needs_update(
            stored_data.get(ticker, {}), max_age_days
        ):
            update_needed.append(ticker)
    return update_needed


def validate_financial_data(data: Dict[str, Any], ticker: str) -> bool:
    """Validate that we have all required financial data"""
    required_fields = [
        ("P/E", "trailingPE"),
        ("P/B", "priceToBook"),
        ("Market Cap", "marketCap"),
        ("Revenue", "totalRevenue"),
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


def get_earnings_growth(
    ticker: str, edgar_client: Optional[EdgarClient] = None, years: int = 10
) -> Optional[float]:
    """Calculate cumulative earnings growth using SEC EDGAR data if available"""
    try:
        if edgar_client:
            # Try to get historical EPS from SEC EDGAR
            eps_df = edgar_client.get_historical_eps(ticker)
            if eps_df is not None and len(eps_df) >= 5:
                eps_df = eps_df.sort_values("date", ascending=False)
                newest = eps_df.iloc[0]["eps"]
                oldest = eps_df.iloc[-1]["eps"]
                years_diff = (
                    eps_df.iloc[0]["date"] - eps_df.iloc[-1]["date"]
                ).days / 365.25

                if oldest != 0 and not pd.isna(oldest) and not pd.isna(newest):
                    growth = (((newest / oldest) ** (1 / years_diff)) - 1) * 100
                    return growth

        return None
    except Exception as e:
        logger.debug(f"Error calculating earnings growth: {e}")
        return None


def check_positive_earnings_streak(sec_data, years=SETTINGS["POSITIVE_EARNINGS_YEARS"]):
    """Check if earnings have been positive for the specified number of years"""
    try:
        if "historicalEPS" in sec_data and len(sec_data["historicalEPS"]) >= years:
            return all(
                float(eps["eps"]) > 0 for eps in sec_data["historicalEPS"][:years]
            )
        return False
    except Exception as e:
        logger.debug(f"Error checking earnings streak: {e}")
        return False


def check_dividend_history(stock, years=SETTINGS["DIVIDEND_HISTORY_YEARS"]):
    """Check if dividends have been paid uninterrupted for specified years AND are currently active"""
    try:
        dividends = stock.dividends
        # First check if dividends exist at all
        if dividends is None or len(dividends) == 0 or dividends.empty:
            return False, False  # No dividends at all, and not currently paying

        # Verify there's actual dividend data
        if dividends.sum() <= 0:
            return False, False

        current_year = date.today().year
        earliest_year = dividends.index[0].year

        # Group dividends by year to check for consistency
        yearly_dividends = dividends.groupby(dividends.index.year).sum()

        # Check if there are dividends in current or previous year (still active)
        currently_pays = (
            current_year in yearly_dividends.index
            or (current_year - 1) in yearly_dividends.index
        )

        # Now check historical consistency (separate from current payment status)
        if current_year - earliest_year < years:
            return False, currently_pays

        # Get the most recent years needed for our check
        recent_years = sorted(yearly_dividends.index)[-years:]

        # Make sure we have enough years with dividends
        if len(recent_years) < years:
            return False, currently_pays

        # Verify there are no gaps in the dividend history
        for i in range(1, len(recent_years)):
            if recent_years[i] != recent_years[i - 1] + 1:
                return False, currently_pays

        # All dividends must be positive
        all_positive = all(yearly_dividends[year] > 0 for year in recent_years)

        # Return both conditions separately so we can track them
        return all_positive, currently_pays
    except Exception as e:
        logger.debug(f"Error checking dividend history: {e}")
        return False, False


def calculate_roic(sec_data, _):
    """Calculate Return on Invested Capital"""
    try:
        # Get net income
        net_income = sec_data.get("netIncome")
        if not net_income:
            return None

        # Get total assets
        total_assets = sec_data.get("totalAssets")
        if not total_assets:
            return None

        # Get total liabilities
        total_liabilities = sec_data.get("totalLiabilities")
        if not total_liabilities:
            return None

        invested_capital = total_assets - total_liabilities
        if invested_capital != 0:
            roic = (net_income / invested_capital) * 100
            return roic

        return None
    except Exception as e:
        logger.debug(f"Error calculating ROIC: e")
        return None


def validate_sec_data(sec_data: Dict) -> bool:
    """Validate that SEC data is recent and complete"""
    try:
        if not sec_data:
            return False

        # Check if we have all required fields
        required_fields = [
            "netIncome",
            "totalAssets",
            "totalLiabilities",
            "operatingCashFlow",
            "capitalExpenditures",
        ]
        return all(field in sec_data for field in required_fields)
    except Exception as e:
        logger.error(f"Error validating SEC data: {e}")
        return False


def calculate_pe_ratio(stock, eps_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Calculate P/E ratio with negative values for negative earnings"""
    try:
        if eps_df is not None and not eps_df.empty:
            latest_eps = eps_df.iloc[0]["eps"]
            if latest_eps == 0:
                return None  # Don't calculate P/E for zero earnings
            current_price = stock.history(period="1d")["Close"].iloc[-1]
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
        if not all(k in sec_data for k in ["operatingCashFlow", "capitalExpenditures"]):
            return None

        cfo = sec_data["operatingCashFlow"]
        capex = (
            abs(sec_data["capitalExpenditures"])
            if sec_data["capitalExpenditures"]
            else 0
        )
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

        eps_df = eps_df.sort_values("date", ascending=False)
        newest = eps_df.iloc[0]["eps"]
        oldest = eps_df.iloc[-1]["eps"]

        if newest <= 0 or oldest <= 0:
            return None  # Don't calculate growth with negative earnings

        years_diff = (eps_df.iloc[0]["date"] - eps_df.iloc[-1]["date"]).days / 365.25
        if years_diff < 1:
            return None  # Need at least 1 year of data

        growth = (((newest / oldest) ** (1 / years_diff)) - 1) * 100
        return growth if abs(growth) < 1000 else None  # Cap extreme growth rates
    except Exception as e:
        logger.error(f"Error calculating earnings growth: {e}")
        return None


def calculate_pb_ratio(sec_data: Dict, market_cap: float) -> Optional[float]:
    """Calculate P/B ratio with better error handling"""
    try:
        if not all(k in sec_data for k in ["totalAssets", "totalLiabilities"]):
            return None

        total_assets = float(sec_data["totalAssets"])
        total_liabilities = float(sec_data["totalLiabilities"])
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
        if not all(k in sec_data for k in ["totalAssets", "totalLiabilities"]):
            return None

        total_assets = float(sec_data["totalAssets"])
        total_liabilities = float(sec_data["totalLiabilities"])

        if total_liabilities <= 0:
            return None

        ratio = total_assets / total_liabilities
        return ratio
    except Exception as e:
        logger.error(f"Error calculating balance sheet ratio: {e}")
        return None


def modern_value_screen(
    ticker: str, use_local: bool = False
) -> Optional[Dict[str, Any]]:
    """Screen a stock using Graham's criteria with SEC EDGAR data"""
    try:
        if use_local:
            data = load_stock_data()
            if ticker in data:
                stored_data = data[ticker]
                if stored_data and not any(
                    stored_data.get(field) is None
                    for field in [
                        "Balance Sheet Ratio",
                        "FCF Yield (%)",
                        "ROIC (%)",
                        "10Y Earnings Growth (%)",
                    ]
                ):
                    return stored_data
                return None

        # Initialize clients
        edgar_client = EdgarClient()
        av_client = AlphaVantageClient(SETTINGS["ALPHA_VANTAGE_KEY"])

        # Get data
        sec_data = edgar_client.get_financial_data(
            edgar_client.get_cik(ticker), "NetIncomeLoss"
        )
        if not validate_sec_data(sec_data):
            return None

        overview = av_client.get_overview(ticker)
        stock = yf.Ticker(ticker)

        # Calculate metrics
        eps_df = edgar_client.get_historical_eps(ticker)
        pe = calculate_pe_ratio(stock, eps_df)

        # Calculate market cap
        if "sharesOutstanding" in sec_data:
            shares = sec_data["sharesOutstanding"]
            price = stock.history(period="1d")["Close"].iloc[-1]
            market_cap = shares * price
        else:
            market_cap = float(overview.get("MarketCapitalization", 0)) or None

        # Calculate other metrics
        pb = calculate_pb_ratio(sec_data, market_cap)
        pe_pb_combo = calculate_pe_pb_combo(pe, pb)
        fcf_yield = calculate_fcf_yield(sec_data, market_cap)
        roic = calculate_roic(sec_data, sec_data)
        earnings_growth = calculate_earnings_growth(eps_df)
        has_positive_earnings = check_positive_earnings_streak(sec_data)
        has_dividend_history, currently_pays_dividends = check_dividend_history(
            stock, SETTINGS["DIVIDEND_HISTORY_YEARS"]
        )

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
            "Currently Pays Dividends": currently_pays_dividends,
            "Market Cap ($B)": market_cap / 1e9 if market_cap else None,
            "Revenue ($B)": sec_data.get("revenue", 0) / 1e9
            if sec_data.get("revenue")
            else None,
            "Last Updated": datetime.datetime.now().isoformat(),
            "Data Source": "SEC EDGAR" if sec_data else "Alpha Vantage",
            "cik": edgar_client.get_cik(ticker),  # Store the CIK
        }

        # Validate against criteria
        stock_data["Meets All Criteria"], stock_data["Met Criteria"] = (
            validate_against_criteria(stock_data)
        )

        return stock_data

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None


def validate_against_criteria(data: Dict, verbosity: int = 0) -> Tuple[bool, List[str]]:
    """Validate stock data against screening criteria and return which criteria were met"""
    try:
        # Function to safely compare numeric values, handling complex numbers
        def safe_compare(value, compare_fn, threshold):
            if value is None:
                return False
            if isinstance(value, complex):
                # For complex numbers, use only the real part for comparison
                try:
                    return compare_fn(value.real, threshold)
                except:
                    return False
            elif not isinstance(value, (int, float)):
                return False
            try:
                return compare_fn(value, threshold)
            except:
                return False

        criteria_results = [
            (
                safe_compare(
                    data.get("P/E"), lambda x, y: x < y, SETTINGS["PE_RATIO_MAX"]
                ),
                f"P/E ratio below {SETTINGS['PE_RATIO_MAX']}",
            ),
            (
                safe_compare(
                    data.get("P/E×P/B"),
                    lambda x, y: x <= y,
                    SETTINGS["PE_PB_COMBO_MAX"],
                ),
                f"P/E×P/B below {SETTINGS['PE_PB_COMBO_MAX']}",
            ),
            (
                safe_compare(
                    data.get("Balance Sheet Ratio"),
                    lambda x, y: x >= y,
                    SETTINGS["BALANCE_SHEET_RATIO_MIN"],
                ),
                f"Balance Sheet Ratio above {SETTINGS['BALANCE_SHEET_RATIO_MIN']}",
            ),
            (
                data.get("Consecutive Positive Earnings Years", 0)
                >= SETTINGS["POSITIVE_EARNINGS_YEARS"],
                f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings",
            ),
            (
                safe_compare(
                    data.get("FCF Yield (%)"),
                    lambda x, y: x > y,
                    SETTINGS["FCF_YIELD_MIN"],
                ),
                f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%",
            ),
            (
                safe_compare(
                    data.get("ROIC (%)"), lambda x, y: x >= y, SETTINGS["ROIC_MIN"]
                ),
                f"ROIC above {SETTINGS['ROIC_MIN']}%",
            ),
            # Modified dividend criterion - MUST satisfy BOTH history AND currently pays
            (
                data.get("Consecutive Dividend Years", 0)
                >= SETTINGS["DIVIDEND_HISTORY_YEARS"]
                and data.get("Currently Pays Dividends", False) == True,
                f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends",
            ),
            (
                safe_compare(
                    data.get("10Y Earnings Growth (%)"),
                    lambda x, y: x >= y,
                    SETTINGS["EARNINGS_GROWTH_MIN"],
                ),
                f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%",
            ),
        ]

        # Get list of met criteria and count
        met_criteria_list = []
        met_count = 0
        for met, desc in criteria_results:
            if met:
                met_criteria_list.append(desc)
                met_count += 1

        total_criteria = len(criteria_results)

        # Output based on verbosity level
        if verbosity >= 2:  # -vv mode: show detailed criteria with extra line spacing
            ticker = data.get("Ticker", "Unknown")
            print(f"\n{ticker} - Met {met_count}/{total_criteria} criteria:")
            # Print each criterion on a separate line with proper check/x mark
            for met, desc in criteria_results:
                if met:
                    print(f"✓ {desc}")
                else:
                    print(f"✗ {desc}")

        # Return whether all criteria are met and the list of met criteria
        return met_count == total_criteria, met_criteria_list
    except Exception as e:
        logger.error(f"Error validating criteria: {e}")
        return False, []


def screen_stock(ticker: str) -> Dict[str, Any]:
    """Screen a single stock for value metrics."""
    try:
        # Initialize clients
        edgar_client = EdgarClient()
        stock = yf.Ticker(ticker)
        metrics = {"Ticker": ticker}  # Add ticker to metrics right away

        # Get basic info from yfinance first as it's more reliable
        try:
            info = stock.info
            if info:
                metrics["Market Cap"] = info.get("marketCap")
                # Get current price reliably
                current_price = stock.history(period="1d")["Close"].iloc[-1]
                metrics["Current Price"] = current_price
        except Exception as e:
            logger.error(f"Error getting yfinance data for {ticker}: {e}")

        # Get CIK
        cik = edgar_client.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return {}

        # 1. Net Income and Earnings History
        # Start with the most reliable concepts
        net_income_data = edgar_client.get_financial_data_with_alternatives(
            cik, ["NetIncomeLoss", "NetIncome", "ProfitLoss"]
        )

        if net_income_data and "units" in net_income_data:
            net_income_history = []
            for unit_type in net_income_data["units"]:
                for entry in net_income_data["units"][unit_type]:
                    if entry.get("form") == "10-K":
                        net_income_history.append((entry["end"], entry["val"]))

            net_income_history.sort(key=lambda x: x[0], reverse=True)

            if net_income_history:
                metrics["net_income"] = net_income_history[0][1]  # Latest

                # Count consecutive years of positive earnings from most recent
                positive_years = 0
                for _, val in net_income_history:
                    if val > 0:
                        positive_years += 1
                    else:
                        break
                metrics["Consecutive Positive Earnings Years"] = positive_years

                # Calculate 10Y earnings growth if enough data
                if len(net_income_history) >= 10:
                    latest = net_income_history[0][1]
                    oldest = net_income_history[9][1]
                    if oldest > 0:
                        growth_rate = ((latest / oldest) ** (1 / 10) - 1) * 100
                        metrics["10Y Earnings Growth (%)"] = growth_rate

        # 2. Balance Sheet Data - these concepts work reliably
        assets_data = edgar_client.get_financial_data_with_alternatives(
            cik, ["Assets", "TotalAssets"]
        )
        liabilities_data = edgar_client.get_financial_data_with_alternatives(
            cik, ["Liabilities", "TotalLiabilities"]
        )

        if assets_data and liabilities_data:
            latest_assets = get_latest_annual_value(assets_data)
            latest_liabilities = get_latest_annual_value(liabilities_data)

            if latest_assets and latest_liabilities:
                metrics["total_assets"] = latest_assets
                metrics["total_liabilities"] = latest_liabilities

                if latest_assets > 0:
                    metrics["Balance Sheet Ratio"] = latest_assets / latest_liabilities

                # Calculate book value and P/B ratio
                book_value = latest_assets - latest_liabilities
                if book_value > 0 and metrics.get("Market Cap"):
                    metrics["P/B"] = metrics["Market Cap"] / book_value

        # 3. Calculate P/E using reliable market cap and net income
        if metrics.get("Market Cap") and metrics.get("net_income"):
            if metrics["net_income"] > 0:
                metrics["P/E"] = metrics["Market Cap"] / metrics["net_income"]

                # Calculate P/E×P/B if both are available
                if metrics.get("P/B"):
                    metrics["P/E×P/B"] = metrics["P/E"] * metrics["P/B"]

        # 4. Cash Flow and ROIC
        operating_cash_flow = edgar_client.get_financial_data_with_alternatives(
            cik,
            ["NetCashProvidedByUsedInOperatingActivities", "CashFlowFromOperations"],
        )

        if operating_cash_flow:
            latest_ocf = get_latest_annual_value(operating_cash_flow)
            if latest_ocf:
                # For FCF, use change in cash position for financial companies
                if ticker in ["MS", "GS", "JPM", "BAC", "C", "WFC"]:
                    cash_flow = latest_ocf
                else:
                    # For non-financials, subtract capex
                    capex_data = edgar_client.get_financial_data_with_alternatives(
                        cik,
                        [
                            "PaymentsToAcquirePropertyPlantAndEquipment",
                            "CapitalExpenditures",
                        ],
                    )
                    if capex_data:
                        latest_capex = get_latest_annual_value(capex_data)
                        if latest_capex:
                            cash_flow = latest_ocf - abs(latest_capex)
                        else:
                            cash_flow = latest_ocf
                    else:
                        cash_flow = latest_ocf

                # Calculate FCF Yield
                if (
                    cash_flow
                    and metrics.get("Market Cap")
                    and metrics["Market Cap"] > 0
                ):
                    metrics["FCF Yield (%)"] = (cash_flow / metrics["Market Cap"]) * 100

                # Calculate ROIC
                if metrics.get("total_assets"):
                    # For financials, use ROA as proxy for ROIC
                    if ticker in ["MS", "GS", "JPM", "BAC", "C", "WFC"]:
                        invested_capital = metrics["total_assets"]
                    else:
                        invested_capital = metrics["total_assets"] - metrics.get(
                            "total_liabilities", 0
                        )

                    if invested_capital > 0:
                        # Use net income for NOPAT as it's more reliable
                        if metrics.get("net_income"):
                            metrics["ROIC (%)"] = (
                                metrics["net_income"] / invested_capital
                            ) * 100

        # 5. Dividend History using yfinance (reliable source)
        try:
            dividends = stock.dividends
            if not dividends.empty:
                yearly_dividends = dividends.groupby(dividends.index.year).sum()
                metrics["Positive Dividend Years"] = sum(
                    1 for d in yearly_dividends if d > 0
                )

                consecutive_years = 0
                for div in reversed(yearly_dividends.values):
                    if div > 0:
                        consecutive_years += 1
                    else:
                        break
                metrics["Consecutive Dividend Years"] = consecutive_years

                # Add check for current or recent dividend payments
                current_year = date.today().year
                currently_pays = (
                    current_year in yearly_dividends.index
                    or (current_year - 1) in yearly_dividends.index
                )
                metrics["Currently Pays Dividends"] = currently_pays
            else:
                metrics["Positive Dividend Years"] = 0
                metrics["Consecutive Dividend Years"] = 0
                metrics["Currently Pays Dividends"] = False
        except Exception as e:
            logger.error(f"Error checking dividend history for {ticker}: {str(e)}")
            metrics["Positive Dividend Years"] = 0
            metrics["Consecutive Dividend Years"] = 0
            metrics["Currently Pays Dividends"] = False

        return metrics

    except Exception as e:
        logger.error(f"Error screening {ticker}: {str(e)}")
        return {}


def get_latest_annual_value(data: Dict) -> Optional[float]:
    """Extract the latest annual value from SEC API response."""
    try:
        values = []
        for unit_type in data["units"]:  # Usually 'USD'
            for entry in data["units"][unit_type]:
                if entry.get("form") == "10-K":  # Only consider annual reports
                    values.append((entry["end"], entry["val"]))

        if values:
            # Sort by date and get most recent value
            latest = sorted(values, key=lambda x: x[0], reverse=True)[0]
            return latest[1]
        return None
    except Exception as e:
        logger.error(f"Error getting latest value: {str(e)}")
        return None


def save_results_csv(results: Dict[str, Any], filepath: str):
    """Save screening results to CSV file"""
    try:
        # Convert the nested dictionary to a DataFrame
        data = []
        for ticker, metrics in results.items():
            metrics["Ticker"] = ticker
            data.append(metrics)

        df = pd.DataFrame(data)

        # Reorder columns to put Ticker first
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("Ticker")))
        df = df[cols]

        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: e")


def validate_cached_stock(
    ticker: str, cached_data: Dict[str, Any], verbosity: int = 0
) -> Dict[str, Any]:
    """Validate cached stock data without making external API calls.
    This function is used to evaluate cached data against criteria without
    fetching fresh data from APIs like Yahoo Finance.
    """
    # Create a clean copy of cached data and ensure ticker is included
    metrics = cached_data.copy() if isinstance(cached_data, dict) else {}
    metrics["Ticker"] = ticker

    # Define safe_compare here to ensure it's available within this function
    def safe_compare(value, compare_fn, threshold):
        if value is None:
            return False
        if isinstance(value, complex):
            # For complex numbers, use only the real part for comparison
            try:
                return compare_fn(value.real, threshold)
            except:
                return False
        elif not isinstance(value, (int, float)):
            return False
        try:
            return compare_fn(value, threshold)
        except:
            return False

    # Use our own validation logic here instead of calling validate_cached_criteria
    # This ensures we have complete control over the validation process
    try:
        # Define the exact 8 criteria to evaluate in a fixed order
        criteria_results = [
            (
                safe_compare(
                    metrics.get("P/E"), lambda x, y: x < y, SETTINGS["PE_RATIO_MAX"]
                ),
                f"P/E ratio below {SETTINGS['PE_RATIO_MAX']}",
            ),
            (
                safe_compare(
                    metrics.get("P/E×P/B"),
                    lambda x, y: x <= y,
                    SETTINGS["PE_PB_COMBO_MAX"],
                ),
                f"P/E×P/B below {SETTINGS['PE_PB_COMBO_MAX']}",
            ),
            (
                safe_compare(
                    metrics.get("Balance Sheet Ratio"),
                    lambda x, y: x >= y,
                    SETTINGS["BALANCE_SHEET_RATIO_MIN"],
                ),
                f"Balance Sheet Ratio above {SETTINGS['BALANCE_SHEET_RATIO_MIN']}",
            ),
            (
                metrics.get("Consecutive Positive Earnings Years", 0)
                >= SETTINGS["POSITIVE_EARNINGS_YEARS"],
                f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings",
            ),
            (
                safe_compare(
                    metrics.get("FCF Yield (%)"),
                    lambda x, y: x > y,
                    SETTINGS["FCF_YIELD_MIN"],
                ),
                f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%",
            ),
            (
                safe_compare(
                    metrics.get("ROIC (%)"), lambda x, y: x >= y, SETTINGS["ROIC_MIN"]
                ),
                f"ROIC above {SETTINGS['ROIC_MIN']}%",
            ),
            (
                metrics.get("Consecutive Dividend Years", 0)
                >= SETTINGS["DIVIDEND_HISTORY_YEARS"]
                and metrics.get("Currently Pays Dividends", False) == True,
                f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends",
            ),
            (
                safe_compare(
                    metrics.get("10Y Earnings Growth (%)"),
                    lambda x, y: x >= y,
                    SETTINGS["EARNINGS_GROWTH_MIN"],
                ),
                f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%",
            ),
        ]

        # Count met criteria
        met_criteria_list = []
        met_count = 0

        for met, desc in criteria_results:
            if met:
                met_criteria_list.append(desc)
                met_count += 1

        total_criteria = len(criteria_results)

        # Store results
        metrics["Meets All Criteria"] = met_count == total_criteria
        metrics["Met Criteria Count"] = met_count
        metrics["Met Criteria"] = met_criteria_list

        # Display output based on verbosity level
        if verbosity >= 1:
            if verbosity == 1:
                # Simple one-line summary
                print(f"{ticker}: {met_count} of {total_criteria} value criteria met")

            elif verbosity >= 2:
                # Detailed view showing every criterion
                print(f"\n{ticker} - Met {met_count}/{total_criteria} criteria:")

                # Print each criterion exactly once with appropriate mark
                for met, desc in criteria_results:
                    mark = "✓" if met else "✗"
                    print(f"{mark} {desc}")
    except Exception as e:
        logger.error(f"Error validating criteria for {ticker}: {e}")
        metrics["Meets All Criteria"] = False
        metrics["Met Criteria Count"] = 0
        metrics["Met Criteria"] = []

    return metrics


def analyze_criteria_stats(results: Dict[str, Any]):
    """Analyze how many stocks meet each individual criterion."""
    if not results:
        print("No results to analyze.")
        return

    # Define all criteria we want to check
    criteria = [
        (
            "P/E ratio below " + str(SETTINGS["PE_RATIO_MAX"]),
            lambda data: isinstance(data.get("P/E"), (int, float))
            and data["P/E"] < SETTINGS["PE_RATIO_MAX"],
        ),
        (
            "P/E×P/B below " + str(SETTINGS["PE_PB_COMBO_MAX"]),
            lambda data: isinstance(data.get("P/E×P/B"), (int, float))
            and data["P/E×P/B"] <= SETTINGS["PE_PB_COMBO_MAX"],
        ),
        (
            "Balance Sheet Ratio above " + str(SETTINGS["BALANCE_SHEET_RATIO_MIN"]),
            lambda data: isinstance(data.get("Balance Sheet Ratio"), (int, float))
            and data["Balance Sheet Ratio"] >= SETTINGS["BALANCE_SHEET_RATIO_MIN"],
        ),
        (
            f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings",
            lambda data: data.get("Consecutive Positive Earnings Years", 0)
            >= SETTINGS["POSITIVE_EARNINGS_YEARS"],
        ),
        (
            f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%",
            lambda data: isinstance(data.get("FCF Yield (%)"), (int, float))
            and data["FCF Yield (%)"] > SETTINGS["FCF_YIELD_MIN"],
        ),
        (
            f"ROIC above {SETTINGS['ROIC_MIN']}%",
            lambda data: isinstance(data.get("ROIC (%)"), (int, float))
            and data["ROIC (%)"] >= SETTINGS["ROIC_MIN"],
        ),
        (
            f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends",
            lambda data: data.get("Consecutive Dividend Years", 0)
            >= SETTINGS["DIVIDEND_HISTORY_YEARS"],
        ),
        (
            f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%",
            lambda data: isinstance(data.get("10Y Earnings Growth (%)"), (int, float))
            and data["10Y Earnings Growth (%)"] >= SETTINGS["EARNINGS_GROWTH_MIN"],
        ),
    ]

    # Count how many stocks meet each criterion
    total_stocks = len(results)
    criteria_counts = []

    criteria_counts.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("\n===== CRITERIA ANALYSIS =====")
    print(f"Total stocks analyzed: {total_stocks}")
    print("\nNumber of stocks meeting each criterion (sorted by most common):")
    print("=" * 80)
    print(f"{'CRITERION':<50} {'COUNT':>8} {'PERCENTAGE':>12}")
    print("-" * 80)
    for name, count, percentage in criteria_counts:
        print(f"{name:<50} {count:>8} {percentage:>11.1f}%")
    print("=" * 80)

    # Show which criteria are the most challenging (least frequently met)
    print("\nMost challenging criteria (least frequently met):")
    most_challenging = sorted(criteria_counts, key=lambda x: x[1])[:3]
    for i, (name, count, percentage) in enumerate(most_challenging):
        print(f"{i + 1}. {name}: {count} stocks ({percentage:.1f}%)")


def show_criteria_explanation():
    """Display detailed explanation of each screening criterion used in the value screener"""
    print("\n===== VALUE INVESTING CRITERIA EXPLANATION =====\n")

    criteria_explanations = [
        {
            "name": "Price to Earnings (P/E) Ratio",
            "setting": f"Current threshold: {SETTINGS['PE_RATIO_MAX']}",
            "description": """
The P/E ratio measures how expensive a stock is relative to its earnings. It represents how many 
years of current earnings it would take to equal the purchase price. 

A lower P/E ratio suggests a stock may be undervalued. Benjamin Graham typically recommended 
stocks with a P/E ratio under 15, though this varies by industry and market conditions.

Formula: Current Stock Price / Earnings Per Share (EPS)
            """,
        },
        {
            "name": "Price to Earnings × Price to Book (P/E×P/B) Ratio",
            "setting": f"Current threshold: {SETTINGS['PE_PB_COMBO_MAX']}",
            "description": """
This combined metric multiplies two valuation ratios to create a more comprehensive value 
assessment. Graham suggested that a reasonable stock should have P/E × P/B < 22.5 (i.e., 
P/E < 15 and P/B < 1.5).

This metric helps identify stocks that are reasonably priced both in terms of earnings 
and underlying assets, providing a margin of safety on both fronts.

Formula: (P/E Ratio) × (Price to Book Ratio)
            """,
        },
        {
            "name": "Balance Sheet Ratio (Assets to Liabilities)",
            "setting": f"Current threshold: {SETTINGS['BALANCE_SHEET_RATIO_MIN']}",
            "description": """
This ratio measures a company's financial strength by comparing total assets to total liabilities. 
A higher ratio indicates a stronger balance sheet with more assets than liabilities.

Graham recommended companies with at least twice as many assets as liabilities (ratio >= 2) 
to ensure financial stability and reduce the risk of bankruptcy.

Formula: Total Assets / Total Liabilities
            """,
        },
        {
            "name": "Positive Earnings History",
            "setting": f"Current threshold: {SETTINGS['POSITIVE_EARNINGS_YEARS']} years",
            "description": """
This criterion checks if a company has maintained positive earnings (profitability) for a 
sustained period, indicating business stability and consistent performance.

Graham preferred companies that had demonstrated consistent profitability across business 
cycles, typically looking for at least 10 years of positive earnings with no deficits.

Metric: Number of consecutive years with positive earnings
            """,
        },
        {
            "name": "Free Cash Flow (FCF) Yield",
            "setting": f"Current threshold: {SETTINGS['FCF_YIELD_MIN']}%",
            "description": """
FCF Yield measures how much cash a company generates relative to its market value, after 
accounting for capital expenditures needed to maintain the business.

A higher FCF yield indicates a company is generating significant cash relative to its price, 
potentially indicating undervaluation. It also suggests the company has funds available for 
dividends, buybacks, debt reduction, or reinvestment.

Formula: (Operating Cash Flow - Capital Expenditures) / Market Capitalization × 100%
            """,
        },
        {
            "name": "Return on Invested Capital (ROIC)",
            "setting": f"Current threshold: {SETTINGS['ROIC_MIN']}%",
            "description": """
ROIC measures how efficiently a company uses its capital to generate profits. It indicates 
management's ability to allocate capital effectively.

A higher ROIC suggests the company has a competitive advantage and can generate more profit 
per dollar of invested capital. Value investors typically look for companies with ROIC 
consistently above 10%.

Formula: Net Operating Profit After Tax / Invested Capital × 100%
            """,
        },
        {
            "name": "Dividend History",
            "setting": f"Current threshold: {SETTINGS['DIVIDEND_HISTORY_YEARS']} years",
            "description": """
This criterion checks whether a company has paid dividends consistently over time, indicating 
financial stability and shareholder-friendly management.

Graham favored companies that had paid dividends without interruption for many years, as this 
demonstrates financial health and a commitment to returning capital to shareholders.

Metric: Number of consecutive years paying dividends
            """,
        },
        {
            "name": "Earnings Growth",
            "setting": f"Current threshold: {SETTINGS['EARNINGS_GROWTH_MIN']}%",
            "description": """
This measures the compound annual growth rate of a company's earnings over a multi-year period, 
typically 10 years.

While Graham emphasized value over growth, he recognized the importance of some earnings growth 
to maintain the company's competitive position and increase intrinsic value over time.

Formula: ((Ending EPS / Beginning EPS) ^ (1/years) - 1) × 100%
            """,
        },
    ]

    for i, criterion in enumerate(criteria_explanations):
        print(f"{i + 1}. {criterion['name']} ({criterion['setting']})")
        print(criterion["description"].strip())
        print("-" * 80)

    print(
        "\nNOTE: These criteria are based on Benjamin Graham's value investing principles,"
    )
    print("      with adjustments for modern market conditions. The thresholds can be")
    print("      customized in the SETTINGS dictionary at the top of the script.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Value Stock Screener\n"
            "\n"
            "Overview:\n"
            "  This script screens stocks using value investing principles. "
            "It analyzes financial data from Alpha Vantage, SEC EDGAR, and Yahoo Finance to identify stocks "
            "that may be undervalued and that meet value criteria for safety, profitability, and growth.\n"
            "\n"
            "Performance & Efficiency:\n"
            "  For speed and reliability, the script uses locally-cached financial data whenever available. "
            "If valid cached data exists, it is loaded and used for screening, greatly reducing the need for repeated API calls, "
            "minimizing network delays, and avoiding rate limits. If cached data is missing or outdated, fresh data is fetched automatically.\n"
            "\n"
            "Scope:\n"
            "  By default, the script evaluates all S&P 500 tickers (NYSE and NASDAQ) as listed on Wikipedia. "
            "You can customize the universe by editing the code or cached data.\n"
            "\n"
            "Objectives:\n"
            "  - Identify stocks that meet all Graham value criteria (P/E, P/B, FCF yield, ROIC, etc.)\n"
            "  - Provide detailed explanations and statistics for each criterion\n"
            "  - Support cache-based and live screening for S&P 500 and custom tickers\n"
            "  - Save results in CSV and JSON formats for further analysis\n"
            "\n"
            "Usage:\n"
            "  python value_screener.py                            # Screen all S&P 500 stocks using cached data if available\n"
            "  python value_screener.py --test 10                  # Screen only 10 stocks (for testing)\n"
            "  python value_screener.py --forceupdate              # Force refresh of all stock data (ignore cache)\n"
            "  python value_screener.py --verbosity 1              # Show summary for each stock\n"
            "  python value_screener.py --verbosity 2              # Show detailed criteria for each stock\n"
            "  python value_screener.py --checkcriteria            # Analyze which criteria are most often met\n"
            "  python value_screener.py --showcriteria             # Display detailed explanation of each criterion\n"
            "\n"
            "Requirements:\n"
            "  - Alpha Vantage API key (set in .env or environment variable)\n"
            "  - Internet connection for live screening\n"
            "\n"
            "To set your API key, create a .env file in the project root with:\n"
            "  ALPHA_VANTAGE_KEY=your_api_key_here\n"
            "Or set it as an environment variable before running the script.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="screener_results.csv",
        help="Output file path for results (CSV format, default: screener_results.csv)",
    )
    parser.add_argument(
        "--forceupdate",
        action="store_true",
        help="Force update of all stock data instead of using cached data",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Run with a limited number of tickers (for testing). Example: --test 10",
    )
    parser.add_argument(
        "--verbosity",
        "-V",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity level (0: minimal output, 1: show summary for each stock, 2: show detailed criteria)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Value Stock Screener v{__version__}",
        help="Show the version number and exit",
    )
    parser.add_argument(
        "--checkcriteria",
        action="store_true",
        help="Analyze which criteria are most often met by stocks instead of showing passing stocks",
    )
    parser.add_argument(
        "--showcriteria",
        action="store_true",
        help="Display detailed explanation of each screening criterion and current settings",
    )
    args = parser.parse_args()

    # If the showcriteria argument is provided, show criteria explanation and exit
    if args.showcriteria:
        show_criteria_explanation()
        return

    results = {}
    stored_data = {}
    tickers_to_update = []
    cached_tickers = []

    # Check if we're using cached data
    use_cached_data = not args.forceupdate

    if use_cached_data:
        # Try to load existing data without making web requests
        stored_data = load_stock_data()

        if stored_data and isinstance(stored_data, dict):
            # Get tickers from cache without making web requests
            tickers = get_cached_tickers(stored_data)
            if tickers:
                print(f"Retrieved {len(tickers)} tickers from cached data")

                # Sort tickers alphabetically
                tickers.sort()
                print("Sorting tickers alphabetically")

                # Apply test parameter in both cache and web modes
                if args.test > 0:
                    tickers = tickers[: args.test]
                    print(f"TEST MODE: Limited to {len(tickers)} tickers")
                    logger.info(f"TEST MODE: Limited to {len(tickers)} tickers")

                print("Using cached stock data")
                # All tickers are from cache, so all can use cache
                cached_tickers = tickers

                print("All data is up to date")
            else:
                use_cached_data = False
        else:
            use_cached_data = False

    # If we can't use cached data, fetch from web
    if not use_cached_data:
        # Get S&P 500 tickers from web
        tickers = get_sp500_tickers()
        print(f"Retrieved {len(tickers)} tickers from S&P 500")

        # Sort tickers alphabetically
        tickers.sort()
        print("Sorting tickers alphabetically")

        # If test mode is active, limit the number of tickers
        if args.test > 0:
            tickers = tickers[: args.test]
            print(f"TEST MODE: Limited to {len(tickers)} tickers")
            logger.info(f"TEST MODE: Limited to {len(tickers)} tickers")

        if not args.forceupdate:
            # Check against stored data if not forcing update
            if stored_data and isinstance(stored_data, dict):
                for ticker in tickers:
                    if ticker in stored_data and isinstance(stored_data[ticker], dict):
                        cached_tickers.append(ticker)  # Use cache for this ticker
                    else:
                        tickers_to_update.append(
                            ticker
                        )  # Need to fetch data for this ticker
            else:
                # No cached data available
                tickers_to_update = tickers
        else:
            # Force update mode - process all tickers
            tickers_to_update = tickers

    # ===== MAJOR FIX: Process cached tickers WITHOUT threading to avoid output duplication =====
    if cached_tickers:
        print(f"Processing {len(cached_tickers)} cached tickers without API calls...")
        results_from_cache = {}

        # Process each ticker one by one for reliable output
        for ticker in cached_tickers:
            try:
                # Show progress
                if args.verbosity == 0:
                    print(f"Validating cached data for {ticker}...")

                # Use local validation function to avoid threading issues
                # This is a simplified version of validate_cached_stock that doesn't do its own output
                metrics = (
                    stored_data[ticker].copy()
                    if isinstance(stored_data[ticker], dict)
                    else {}
                )

                # Helper function for safe comparisons
                def safe_compare(value, compare_fn, threshold):
                    if value is None:
                        return False
                    if isinstance(value, complex):
                        try:
                            return compare_fn(value.real, threshold)
                        except:
                            return False
                    elif not isinstance(value, (int, float)):
                        return False
                    try:
                        return compare_fn(value, threshold)
                    except:
                        return False

                # Evaluate criteria
                criteria_results = [
                    (
                        safe_compare(
                            metrics.get("P/E"),
                            lambda x, y: x < y,
                            SETTINGS["PE_RATIO_MAX"],
                        ),
                        f"P/E ratio below {SETTINGS['PE_RATIO_MAX']}",
                    ),
                    (
                        safe_compare(
                            metrics.get("P/E×P/B"),
                            lambda x, y: x <= y,
                            SETTINGS["PE_PB_COMBO_MAX"],
                        ),
                        f"P/E×P/B below {SETTINGS['PE_PB_COMBO_MAX']}",
                    ),
                    (
                        safe_compare(
                            metrics.get("Balance Sheet Ratio"),
                            lambda x, y: x >= y,
                            SETTINGS["BALANCE_SHEET_RATIO_MIN"],
                        ),
                        f"Balance Sheet Ratio above {SETTINGS['BALANCE_SHEET_RATIO_MIN']}",
                    ),
                    (
                        metrics.get("Consecutive Positive Earnings Years", 0)
                        >= SETTINGS["POSITIVE_EARNINGS_YEARS"],
                        f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings",
                    ),
                    (
                        safe_compare(
                            metrics.get("FCF Yield (%)"),
                            lambda x, y: x > y,
                            SETTINGS["FCF_YIELD_MIN"],
                        ),
                        f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%",
                    ),
                    (
                        safe_compare(
                            metrics.get("ROIC (%)"),
                            lambda x, y: x >= y,
                            SETTINGS["ROIC_MIN"],
                        ),
                        f"ROIC above {SETTINGS['ROIC_MIN']}%",
                    ),
                    (
                        metrics.get("Consecutive Dividend Years", 0)
                        >= SETTINGS["DIVIDEND_HISTORY_YEARS"]
                        and metrics.get("Currently Pays Dividends", False) == True,
                        f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends",
                    ),
                    (
                        safe_compare(
                            metrics.get("10Y Earnings Growth (%)"),
                            lambda x, y: x >= y,
                            SETTINGS["EARNINGS_GROWTH_MIN"],
                        ),
                        f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%",
                    ),
                ]

                # Count met criteria
                met_criteria_list = []
                met_count = 0

                for met, desc in criteria_results:
                    if met:
                        met_criteria_list.append(desc)
                        met_count += 1

                total_criteria = len(criteria_results)

                # Store results in metrics
                metrics["Meets All Criteria"] = met_count == total_criteria
                metrics["Met Criteria Count"] = met_count
                metrics["Met Criteria"] = met_criteria_list

                # Display output based on verbosity level - now controlled at the loop level
                if args.verbosity >= 1:
                    if args.verbosity == 1:
                        print(
                            f"{ticker}: {met_count} of {total_criteria} value criteria met"
                        )
                    elif args.verbosity >= 2:
                        print(
                            f"\n{ticker} - Met {met_count}/{total_criteria} criteria:"
                        )
                        for met, desc in criteria_results:
                            mark = "✓" if met else "✗"
                            print(f"{mark} {desc}")

                # Add to results
                results_from_cache[ticker] = metrics
            except Exception as e:
                logger.error(f"Error processing cached ticker {ticker}: {str(e)}")

        # Merge results
        results.update(results_from_cache)

    # Process tickers that need updating (with API calls)
    if tickers_to_update:
        print(
            f"Fetching data for {len(tickers_to_update)} tickers that need updating..."
        )
        # Limit the number of workers to avoid overwhelming connection pools
        max_workers = 5  # Reduced from default to avoid connection pool warnings
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            # Submit jobs in smaller batches with delays between batches
            batch_size = max_workers
            for i in range(0, len(tickers_to_update), batch_size):
                batch = tickers_to_update[i : i + batch_size]
                print(
                    f"Processing batch {i // batch_size + 1}/{(len(tickers_to_update) + batch_size - 1) // batch_size}..."
                )

                # Submit this batch
                for ticker in batch:
                    # Using a modified process_new_ticker function that doesn't do its own output
                    def process_new_ticker(ticker):
                        if args.verbosity == 0:
                            print(f"Screening {ticker}...")
                            logger.info(f"Screening {ticker}...")

                        metrics = screen_stock(ticker)
                        if metrics:  # Only include stocks with data
                            # Calculate criteria but don't output
                            meets_criteria, met_criteria = validate_against_criteria(
                                metrics, 0
                            )  # Force verbosity=0
                            metrics["Meets All Criteria"] = meets_criteria
                            metrics["Met Criteria Count"] = len(met_criteria)
                            metrics["Met Criteria"] = met_criteria
                            return metrics
                        return None

                    futures[executor.submit(process_new_ticker, ticker)] = ticker

                # Wait for this batch to complete before submitting the next one
                results_from_api = {}
                for future in futures:
                    ticker = futures[future]
                    try:
                        metrics = future.result()
                        if metrics:
                            results_from_api[ticker] = metrics

                            # Now handle output here, after getting results
                            if args.verbosity >= 1:
                                criteria_list = metrics.get("Met Criteria", [])
                                total_criteria = 8  # Fixed number of criteria

                                if args.verbosity == 1:
                                    print(
                                        f"{ticker}: {len(criteria_list)} of {total_criteria} value criteria met"
                                    )
                                elif args.verbosity >= 2:
                                    # For detailed output, manually show each criterion
                                    print(
                                        f"\n{ticker} - Met {len(criteria_list)}/{total_criteria} criteria:"
                                    )

                                    # Define criteria to check
                                    criteria_to_check = [
                                        (
                                            f"P/E ratio below {SETTINGS['PE_RATIO_MAX']}",
                                            f"P/E ratio below {SETTINGS['PE_RATIO_MAX']}"
                                            in criteria_list,
                                        ),
                                        (
                                            f"P/E×P/B below {SETTINGS['PE_PB_COMBO_MAX']}",
                                            f"P/E×P/B below {SETTINGS['PE_PB_COMBO_MAX']}"
                                            in criteria_list,
                                        ),
                                        (
                                            f"Balance Sheet Ratio above {SETTINGS['BALANCE_SHEET_RATIO_MIN']}",
                                            f"Balance Sheet Ratio above {SETTINGS['BALANCE_SHEET_RATIO_MIN']}"
                                            in criteria_list,
                                        ),
                                        (
                                            f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings",
                                            f"At least {SETTINGS['POSITIVE_EARNINGS_YEARS']} years of positive earnings"
                                            in criteria_list,
                                        ),
                                        (
                                            f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%",
                                            f"FCF Yield above {SETTINGS['FCF_YIELD_MIN']}%"
                                            in criteria_list,
                                        ),
                                        (
                                            f"ROIC above {SETTINGS['ROIC_MIN']}%",
                                            f"ROIC above {SETTINGS['ROIC_MIN']}%"
                                            in criteria_list,
                                        ),
                                        (
                                            f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends",
                                            f"At least {SETTINGS['DIVIDEND_HISTORY_YEARS']} years of dividends"
                                            in criteria_list,
                                        ),
                                        (
                                            f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%",
                                            f"10Y Earnings Growth above {SETTINGS['EARNINGS_GROWTH_MIN']}%"
                                            in criteria_list,
                                        ),
                                    ]

                                    for desc, met in criteria_to_check:
                                        mark = "✓" if met else "✗"
                                        print(f"{mark} {desc}")
                    except Exception as e:
                        logger.error(f"Error processing ticker {ticker}: {str(e)}")

                # Merge results
                results.update(results_from_api)

                # Clear futures for the next batch
                futures = {}

                # Add a delay between batches to avoid overwhelming the connection pool
                if i + batch_size < len(tickers_to_update):
                    print(
                        f"Waiting {SETTINGS['REQUEST_DELAY']} seconds before next batch..."
                    )
                    time.sleep(SETTINGS["REQUEST_DELAY"])

    # Always process and save results
    if results:
        # Save both JSON and CSV formats - always do this regardless of mode
        save_stock_data(results)  # Save JSON for caching

        # Only save CSV in regular mode (not in checkcriteria mode)
        if not args.checkcriteria:
            save_results_csv(results, args.output)  # Save CSV for user

    if args.checkcriteria:
        # Special output mode that just counts how many stocks meet each criteria
        analyze_criteria_stats(results)
    elif args.showcriteria:
        # Show detailed explanation of each criterion
        show_criteria_explanation()
    else:
        print("\n===== RESULTS =====")  # Always show results header with direct print

        # Always display results based on verbosity level
        passing_stocks = []
        for ticker, metrics in results.items():
            if isinstance(metrics, dict) and metrics.get("Meets All Criteria", False):
                passing_stocks.append(ticker)

        # Sort passing stocks alphabetically for consistent output
        passing_stocks.sort()

        if args.verbosity == 0:  # Standard output mode
            msg = f"Screening complete. Found data for {len(results)} stocks."
            print(msg)
            logger.info(msg)

            msg = f"{len(passing_stocks)} stocks meet all Graham value criteria."
            print(msg)
            logger.info(msg)

            if passing_stocks:
                msg = f"Passing stocks: {', '.join(passing_stocks)}"
                print(msg)
                logger.info(msg)
        elif args.verbosity == 1:  # Simple summary mode
            msg = f"Screening complete. Found data for {len(results)} stocks."
            print(msg)

            # We've already displayed criteria for each stock during processing,
            # so just show a summary count of passing stocks
            print(f"{len(passing_stocks)} stocks meet all Graham value criteria.")
            if passing_stocks:
                print(f"Passing stocks: {', '.join(passing_stocks)}")
        elif args.verbosity >= 2:  # Detailed criteria mode
            msg = f"Screening complete. Found data for {len(results)} stocks."
            print(msg)
            logger.info(msg)

            # We've already displayed detailed criteria for each stock during processing,
            # so just show a summary count of passing stocks
            print(f"{len(passing_stocks)} stocks meet all Graham value criteria.")
            if passing_stocks:
                print(f"Passing stocks: {', '.join(passing_stocks)}")


if __name__ == "__main__":
    main()
