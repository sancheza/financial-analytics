#!/usr/bin/env python3

import yfinance as yf
import datetime
from datetime import date
import pandas as pd
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The fixed check_dividend_history function from our update
def check_dividend_history(stock, years=5):
    """Check if dividends have been paid uninterrupted for specified years AND are still active"""
    try:
        dividends = stock.dividends
        if dividends.empty:
            print("No dividend history found")
            return False
            
        current_year = date.today().year
        earliest_year = dividends.index[0].year
        
        # Check if we have enough history to evaluate
        if current_year - earliest_year < years:
            print(f"Not enough dividend history: {current_year - earliest_year} years, need {years}")
            return False
            
        # Group dividends by year to check for consistency
        yearly_dividends = dividends.groupby(dividends.index.year).sum()
        print(f"Yearly dividends: {yearly_dividends}")
        
        # Check if there are dividends in current or previous year (still active)
        if current_year not in yearly_dividends.index and (current_year - 1) not in yearly_dividends.index:
            print(f"No dividends in current year ({current_year}) or previous year ({current_year - 1})")
            return False
            
        # Get the most recent years needed for our check
        recent_years = sorted(yearly_dividends.index)[-years:]
        print(f"Recent years for dividend check: {recent_years}")
        
        # Make sure we have enough years with dividends
        if len(recent_years) < years:
            print(f"Not enough years with dividends: {len(recent_years)}, need {years}")
            return False
        
        # Verify there are no gaps in the dividend history
        for i in range(1, len(recent_years)):
            if recent_years[i] != recent_years[i-1] + 1:
                print(f"Gap in dividend history: {recent_years[i-1]} to {recent_years[i]}")
                return False
                
        # All dividends must be positive
        all_positive = all(yearly_dividends[year] > 0 for year in recent_years)
        if not all_positive:
            print("Some recent years have zero or negative dividends")
            return False
            
        print("All dividend checks passed")
        return True
    except Exception as e:
        print(f"Error checking dividend history: {e}")
        return False

def test_original_function():
    """Test the original dividend function from the codebase"""
    stock = yf.Ticker("APTV")
    dividends = stock.dividends
    
    try:
        if dividends.empty:
            print("Original function: No dividend history found")
            return False
            
        current_year = date.today().year
        earliest_year = dividends.index[0].year
        
        print(f"Original function: Current year: {current_year}, earliest dividend year: {earliest_year}")
        
        if current_year - earliest_year < 5:
            print("Original function: Not enough dividend history")
            return False
            
        yearly_dividends = dividends.groupby(dividends.index.year).sum()
        recent_years = yearly_dividends[-5:]
        
        result = len(recent_years) >= 5 and all(d > 0 for d in recent_years)
        print(f"Original function result: {result}")
        return result
    except Exception as e:
        print(f"Error in original function: {e}")
        return False

def check_actual_implementation():
    """Test how APTV is evaluated in the main code"""
    ticker = "APTV"
    print(f"\nChecking {ticker} with different implementations:")
    
    stock = yf.Ticker(ticker)
    
    # Print basic dividend info
    print(f"\n{ticker} dividend data:")
    dividends = stock.dividends
    if not dividends.empty:
        print(f"First dividend: {dividends.index[0].strftime('%Y-%m-%d')}")
        print(f"Latest dividend: {dividends.index[-1].strftime('%Y-%m-%d')}")
        yearly_dividends = dividends.groupby(dividends.index.year).sum()
        print(f"Yearly dividend totals: {yearly_dividends}")
        
        # Check if there are any dividends in current or previous year
        current_year = date.today().year
        if current_year in yearly_dividends.index:
            print(f"Has dividends in current year ({current_year})")
        else:
            print(f"No dividends in current year ({current_year})")
            
        if current_year - 1 in yearly_dividends.index:
            print(f"Has dividends in previous year ({current_year - 1})")
        else:
            print(f"No dividends in previous year ({current_year - 1})")
    else:
        print("No dividend data available")

    print("\n1. Testing with original function:")
    old_result = test_original_function()
    print(f"Original function passes: {old_result}")
    
    print("\n2. Testing with our updated function:")
    new_result = check_dividend_history(stock, 5)
    print(f"Updated function passes: {new_result}")
    
    print("\n3. Testing with even stricter function (just current year):")
    has_current_year_dividend = not dividends.empty and current_year in dividends.groupby(dividends.index.year).sum().index
    print(f"Current year dividend check passes: {has_current_year_dividend}")

if __name__ == "__main__":
    print(f"Current date: {datetime.datetime.now().date()}")
    check_actual_implementation()

