#!/usr/bin/env python3

import yfinance as yf
from datetime import date
import pandas as pd

def test_dividend_history():
    """Test the dividend history of APTV to understand why it's passing the criterion."""
    ticker = "APTV"
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    
    if dividends.empty:
        print(f"{ticker} has no dividend history at all.")
        return
    
    print(f"{ticker} dividend history:")
    print(f"Number of dividend records: {len(dividends)}")
    print(f"Earliest dividend date: {dividends.index[0].strftime('%Y-%m-%d')}")
    print(f"Most recent dividend date: {dividends.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate days since last dividend
    current_date = date.today()
    most_recent_date = dividends.index[-1].date()
    days_since_last_dividend = (current_date - most_recent_date).days
    print(f"Days since last dividend: {days_since_last_dividend}")
    
    # Group dividends by year
    yearly_dividends = dividends.groupby(dividends.index.year).sum()
    print("\nYearly dividend totals:")
    for year, amount in yearly_dividends.items():
        print(f"{year}: ${amount:.4f}")
    
    # Check if there are dividends in current or previous year
    current_year = current_date.year
    if current_year in yearly_dividends.index:
        print(f"\n✓ Has dividends in current year ({current_year})")
    else:
        print(f"\n✗ No dividends in current year ({current_year})")
    
    if (current_year - 1) in yearly_dividends.index:
        print(f"✓ Has dividends in previous year ({current_year - 1})")
    else:
        print(f"✗ No dividends in previous year ({current_year - 1})")
    
    # Test our fixed function logic
    years_required = 5
    current_year = date.today().year
    earliest_year = dividends.index[0].year
    
    # Check if we have enough history to evaluate
    if current_year - earliest_year < years_required:
        print(f"\n✗ Not enough dividend history ({current_year - earliest_year} years, need {years_required})")
    else:
        print(f"\n✓ Has at least {years_required} years of dividend history")
    
    # Check if there are dividends in current or previous year (still active)
    if current_year not in yearly_dividends.index and (current_year - 1) not in yearly_dividends.index:
        print(f"✗ No dividends in current or previous year - dividend appears to be suspended")
    else:
        print(f"✓ Has dividends in current or previous year")
    
    # Get the most recent years needed for our check
    recent_years = sorted(yearly_dividends.index)[-years_required:]
    
    # Make sure we have enough years with dividends
    if len(recent_years) < years_required:
        print(f"✗ Not enough years with dividends ({len(recent_years)} years, need {years_required})")
    else:
        print(f"✓ Has enough yearly dividend records ({len(recent_years)})")
        
        # Verify there are no gaps in the dividend history
        has_gaps = False
        for i in range(1, len(recent_years)):
            if recent_years[i] != recent_years[i-1] + 1:
                print(f"✗ Gap in dividend history: {recent_years[i-1]} to {recent_years[i]}")
                has_gaps = True
                break
        
        if not has_gaps:
            print(f"✓ No gaps in dividend history")
            
        # Check if all dividends are positive
        all_positive = all(yearly_dividends[year] > 0 for year in recent_years)
        if all_positive:
            print(f"✓ All dividends are positive")
        else:
            print(f"✗ Some dividends are zero or negative")
    
    # Final result based on our fixed function
    if (current_year not in yearly_dividends.index and (current_year - 1) not in yearly_dividends.index) or \
       len(recent_years) < years_required or has_gaps or not all_positive:
        print("\nFinal result: ✗ Does NOT meet dividend criterion with fixed function")
    else:
        print("\nFinal result: ✓ Meets dividend criterion with fixed function")

if __name__ == "__main__":
    test_dividend_history()