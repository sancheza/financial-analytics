# Finance scripts repo

This repo contains python scripts and libre office functions for performing financial analytics.

## API Keys and Environment Variables

This project uses API keys from various financial data providers. For security, all API keys should be stored in environment variables instead of in the code.

### Setting up environment variables

#### Using a .env file (recommended)

1. Create a file named `.env` in the project root directory
2. Add your API keys to this file in the format:
   ```
   ALPHA_VANTAGE_KEY=your_api_key_here
   # Add other API keys as needed
   ```
3. Make sure `.env` is listed in your `.gitignore` file so it won't be committed to the repository
4. Install the dotenv package with `pip install python-dotenv`
5. Add this code at the beginning of any script that needs to use these variables:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load variables from .env file
   ```

#### Using system environment variables

You can also set environment variables directly in your shell:

```bash
# For Unix/Linux/macOS
export ALPHA_VANTAGE_KEY=your_api_key_here

# For Windows (Command Prompt)
set ALPHA_VANTAGE_KEY=your_api_key_here

# For Windows (PowerShell)
$env:ALPHA_VANTAGE_KEY="your_api_key_here"
```

### Required API Keys

- **Alpha Vantage**: Required for `value_screener.py` and `test_alpha_vantage.py`. Get a free key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
- **Financial Modeling Prep (FMP)**: Required for `getBookValueFMP`. Get a free key at [FMP](https://financialmodelingprep.com).


## The Tools

### calculate_YTW

This script calculates a bond's Yield to Maturity (YTM) using the industry-standard Bond Equivalent Yield (BEY) method, which accounts for accrued interest and semi-annual coupon payments.

> **Dependencies**
> - Python 3.10 or later  
> - No third-party dependencies (only Python standard library modules are used)

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/calculate_YTW_demo.gif)


### getBookValueFMP

This is a Google Sheets function that populates book value in a worksheet for a set of tickers.

#### Usage
Create a Google worksheet with stock tickers in one column and a trigger cell used for activating the function in another. In the demo below, tickers are housed in column A and the trigger cell is N2. The trigger cell is useful if you are using a free API with limits. If not, feel free to remove it.

You will need to update the function with your API key. You can obtain it at the link listed below under Dependencies.

The function will populate the cell in which it is called as well as the two to the right of it. See cells I2 through K2 in the demo below.


> **Dependencies**
> - Google Sheets
> - Financial Modeling Prep (FMP) free API key [(get it here)](https://financialmodelingprep.com)

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/getBookValueFMP-demo.gif)


### get_dividend
This script retrieves the Forward Dividend Yield for a given stock ticker from Yahoo Finance. It can be run from the terminal or called by a Libre Office macro to populate relevant cells.

> **Dependencies**
> - Python 3.10 or later
> - requests
> - BeautifulSoup

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/get_dividend_demo.gif)

### value_price_screener
This tool performs value analytics by identifying historically undervalued dividend-paying companies from the S&P 500 and measuring their performance in the year following the target year. It enables rigorous backtesting of value investing principles using accurate historical data.

> **Dependencies**
> - Python 3.10 or later  
> - yfinance  
> - pandas  
> - requests  
> - wcwidth

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/value_price_screener_demo.gif)

### value_screener
This script screens stocks using value investing principles. It analyzes financial data from Alpha Vantage, SEC EDGAR, and Yahoo Finance to identify stocks that may be undervalued and that meet value criteria for safety, profitability, and growth.

By default, the script evaluates all S&P 500 tickers (NYSE and NASDAQ) as listed on Wikipedia. You can customize the universe by editing the code or cached data.

> **Dependencies**
> - Python 3.10 or later  
> - yfinance  
> - pandas  
> - pandas_datareader  
> - ijson  
> - requests  
> - alpha_vantage

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/value_screener_demo.gif)

### value_screener_peak_drawdown
This script analyzes the current S&P 500 constituents and identifies stocks that are down a specified percentage or more from their peak price over the past N years, with an optional minimum market capitalization filter.

> **Dependencies**
> - Python 3.10 or later  
> - yfinance  
> - pandas  
> - wcwidth

![](https://github.com/sancheza/Finance-scripts/blob/main/assets/value_screener_peak_drawdown_demo.gif)


### get_projected_fcf
This script fetches the projected Free Cash Flow (FCF) for a given stock ticker from GuruFocus. It uses Playwright to scrape the data from the GuruFocus website.

> **Dependencies**
> - Python 3.10 or later
> - playwright
> - beautifulsoup4

### get_earnings_date
This script retrieves the next earnings date for a given stock ticker from Yahoo Finance. It can be run from the terminal or called by other scripts.

> **Dependencies**
> - Python 3.10 or later
> - requests
> - beautifulsoup4

### get_ex_dividend
This script retrieves the Ex-Dividend date for a given stock ticker from Yahoo Finance. Similar to get_earnings_date, it can be run from the terminal or integrated into other workflows.

> **Dependencies**
> - Python 3.10 or later
> - requests
> - beautifulsoup4

### get_fair_value
This script fetches fair value estimates for a given stock ticker from GuruFocus. It uses Playwright to navigate to the GuruFocus page and extract the fair value data.

> **Dependencies**
> - Python 3.10 or later
> - playwright
> - beautifulsoup4

### get_forwardpe
This script retrieves the Forward P/E ratio for a given stock ticker using the yfinance library. It provides a reliable way to get forward valuation metrics.

> **Dependencies**
> - Python 3.10 or later
> - yfinance

### get_ticker_range
This script downloads historical price data for a list of tickers and calculates the 30-day low and high prices. It outputs a summary table with the range data.

> **Dependencies**
> - Python 3.10 or later
> - yfinance
> - pandas

### get_yearly_performance
This script retrieves stock performance data for a given ticker and year. It returns the start price, end price, dollar change, and percent change for the specified year.

> **Dependencies**
> - Python 3.10 or later
> - yfinance
> - pandas

### get_pb
This script retrieves the Price-to-Book (P/B) ratio for a given stock ticker using the yfinance library.

> **Dependencies**
> - Python 3.10 or later
> - yfinance


## Bond Analytics Tools

The following tools are located in the `bond-analytics/` directory and focus on US Treasury bond analysis.

### generate_auction_calendar
This script generates a Treasury auction calendar by fetching upcoming auction dates from the TreasuryDirect website. It provides a comprehensive schedule of upcoming Treasury bill, note, and bond auctions.

> **Dependencies**
> - Python 3.10 or later
> - requests
> - beautifulsoup4

### bond_market_analyzer
This script analyzes Treasury bonds from the secondary market. It parses input data (CUSIP, price, yield) and calculates various metrics including Yield to Maturity (YTM) using the Bond Equivalent Yield (BEY) method.

> **Dependencies**
> - Python 3.10 or later
> - Standard library only (uses calculate_YTW module)

### bond_alert
This script monitors US Treasury bond yields (10Y, 20Y, 30Y, and 10Y TIPS) using the FRED API. It checks if yields exceed configured thresholds and sends email alerts when thresholds are breached.

> **Dependencies**
> - Python 3.10 or later
> - requests
> - python-dotenv

### get_bond_yield
This script fetches current Treasury yields from the FRED (Federal Reserve Economic Data) API. It supports various maturities including Treasury Bills (4W, 8W, 13W, etc.) and Treasury Notes/Bonds (2Y, 5Y, 10Y, 20Y, 30Y).

> **Dependencies**
> - Python 3.10 or later
> - requests

### bond_return_calc
This script calculates the total return for Treasury bonds. It uses the Newton-Raphson method to compute yield and accounts for settlement dates, coupon payments, and semi-annual compounding.

> **Dependencies**
> - Python 3.10 or later
> - scipy
> - python-dateutil
> - tabulate
> - thefuzz

### bond_market_analyzer_viewer
This script reads treasury bond data from a JSON file and displays it in a well-formatted table. It's used to visualize data generated by other bond analytics tools.

> **Dependencies**
> - Python 3.10 or later
> - Standard library only


## LibreOffice Macros

The following macros are located in the `macros/` directory and integrate with LibreOffice Calc to populate financial data in spreadsheets.

### get_fair_value_macro
This LibreOffice Calc macro reads ticker symbols from column A and populates the corresponding "Fair Value" columns by calling the get_fair_value.py script. Numeric values are stored as numbers for calculations, while "N/A" entries are stored as text.

> **Dependencies**
> - LibreOffice Calc
> - Python 3.10 or later

### forward_pe_macro
This LibreOffice Calc macro reads ticker symbols from column A and populates the corresponding "Forward PE" column by calling the get_forwardpe.py script. It stores numeric values as numbers and "N/A" entries as text.

> **Dependencies**
> - LibreOffice Calc
> - Python 3.10 or later

### get_pb_macro
This LibreOffice Calc macro reads ticker symbols from column A and populates the corresponding "P/B" column by calling the get_pb.py script. It stores numeric values as numbers and "N/A" entries as text.

> **Dependencies**
> - LibreOffice Calc
> - Python 3.10 or later

### get_ex_dividend_macro
This LibreOffice Calc macro reads ticker symbols from column A and populates the corresponding "Ex-Dividend" column by calling the get_ex_dividend.py script.

> **Dependencies**
> - LibreOffice Calc
> - Python 3.10 or later

### get_dividend_macro
This LibreOffice Calc macro reads ticker symbols from column A and populates the corresponding "Dividend Yield" column by calling the get_dividend.py script. It stores numeric values as numbers and "N/A" entries as text.

> **Dependencies**
> - LibreOffice Calc
> - Python 3.10 or later

