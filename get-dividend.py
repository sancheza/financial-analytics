# This scripts obtains dividend yield for a list of stocks defined in ticker.txt and outputs in the file dividends.txt. This version leverages the Yahoo Finance API to pull dividend data.

import yfinance as yf

# Function to get dividend yield using yfinance
def get_dividend_yield(ticker):
    try:
        # Download stock data using yfinance
        stock = yf.Ticker(ticker)
        
        # Fetch the dividend yield directly from the stock's info
        info = stock.info
        
        # Check if the dividend yield is available
        dividend_yield = info.get('dividendYield', None)
        
        if dividend_yield is None:
            return "No Dividend"
        
        # Return the dividend yield directly as a percentage
        return f"{dividend_yield:.2f}%"
    
    except Exception as e:
        # Handle cases where there's an issue fetching data (e.g., mutual funds, delisted)
        return f"Error for {ticker}: {str(e)}"

# Function to read tickers from a file and write results to another file
def write_dividends_to_file(input_file, output_file):
    with open(input_file, "r") as file:
        tickers = file.readlines()
    
    # Remove any extra whitespace or newlines from tickers
    tickers = [ticker.strip() for ticker in tickers]
    
    with open(output_file, "w") as file:
        for ticker in tickers:
            dividend = get_dividend_yield(ticker)
            file.write(f"{ticker},{dividend}\n")
            print(f"Written: {ticker}, {dividend}")  # For feedback in the terminal

# Specify the input file and output file
input_file = "tickers.txt"  # Tickers file in the local directory
output_file = "dividends.txt"  # Output file for dividends

# Run the function to write dividends to the text file
write_dividends_to_file(input_file, output_file)
