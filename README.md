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

### getBookValue

This is a Google Sheets function that populates book value in a worksheet for a set of tickers.

#### Usage
Create a Google worksheet with stock tickers in one column and a trigger cell used for activating the function in another. In the demo below, tickers are housed in column A and the trigger cell is N2. The trigger cell is useful if you are using a free API with limits. If not, feel free to remove it.

You will need to update the function with your API key. You can obtain it at the link listed below under Dependencies.

The function will populate the cell in which it is called as well as the two to the right of it. See cells I2 through K2 in the demo below.


```Dependencies: Google Sheets, Financial Modeling Prep (FMP) free API key```[(get it here)](https://financialmodelingprep.com)

![](https://github.com/sancheza/Finance-scripts/blob/main/getBookValueFMP-demo.gif)


### get-dividends

This script obtains dividend yield for a list of stocks defined in ticker.txt and outputs in the file dividends.txt. This version leverages the Yahoo Finance API to pull dividend data.

```Dependencies: Python 3, yfinance module```

![](https://github.com/sancheza/Finance-scripts/blob/main/get-dividends-demo.gif)



