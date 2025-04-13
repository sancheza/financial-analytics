#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Back-testing script for bond yield predictions.
Tests the prediction algorithm against previous auction results.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader.data as web
from src.data_fetcher import DataFetcher
from src.utils import setup_logger
from src.predictor import YieldPredictor

# Configure logging
logger = setup_logger(__name__, "../data/log.txt")

def get_real_historical_data(bond_type, num_auctions=10):
    """
    Get real historical yield data for back-testing from FRED.
    
    Args:
        bond_type: Bond type to test (e.g., '20Y')
        num_auctions: Number of auctions to retrieve
        
    Returns:
        List of auction data dictionaries with real yield data
    """
    logger.info(f"Fetching real historical yield data for {bond_type} from FRED...")
    
    try:
        # Map bond types to FRED series IDs
        fred_series = {
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
        
        series_id = fred_series.get(bond_type)
        if not series_id:
            logger.error(f"No FRED series mapping for {bond_type}")
            return []
        
        # Get last 5 years of data
        end_date = datetime.now()
        start_date = datetime(end_date.year - 5, end_date.month, end_date.day)
        
        # Fetch data
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        logger.info(f"Retrieved {len(df)} days of yield data from FRED")
        
        # Reset index to get date as column
        df = df.reset_index()
        df.columns = ['date', 'yield']
        
        # Sort by date descending (newest first)
        df = df.sort_values('date', ascending=False)
        
        # Get monthly data points (approximating auction dates)
        # Group by year and month, then take first entry of each group
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        monthly_df = df.groupby('year_month').first().reset_index()
        logger.info(f"Extracted {len(monthly_df)} monthly data points")
        
        # Convert to list of dictionaries for the specified number of auctions
        auctions = []
        for _, row in monthly_df.head(num_auctions).iterrows():
            auction_record = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'high_yield': row['yield']
            }
            auctions.append(auction_record)
        
        logger.info(f"Successfully prepared {len(auctions)} real yield records for back-testing")
        return auctions
        
    except Exception as e:
        logger.error(f"Error fetching real historical data: {e}")
        return []

def run_backtest(bond_types=None, num_auctions=10, output_file=None, use_real_data=True):
    """
    Run back-testing on specified bond types.
    
    Args:
        bond_types: List of bond types to test, or None for all types
        num_auctions: Number of past auctions to test against
        output_file: Optional path to save consolidated results
        use_real_data: If True, use real data from FRED
    """
    if bond_types is None:
        # Default bond types to test
        bond_types = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    
    fetcher = DataFetcher()
    predictor = YieldPredictor()
    all_results = {}
    
    for bond_type in bond_types:
        logger.info(f"Back-testing {bond_type} bond yield predictions...")
        
        if use_real_data:
            # Get real historical auction data from FRED
            auctions = get_real_historical_data(bond_type, num_auctions)
        else:
            # Fall back to fetcher's method if real data isn't available
            results = fetcher.backtest_yield_predictions(bond_type, num_auctions)
            if not results.get('success'):
                logger.error(f"Back-testing failed for {bond_type}: {results.get('error', 'Unknown error')}")
                all_results[bond_type] = results
                continue
            auctions = results.get('predictions', [])
        
        if not auctions:
            logger.error(f"No auction data available for {bond_type}")
            all_results[bond_type] = {
                'success': False,
                'error': 'No auction data available'
            }
            continue
            
        logger.info(f"Testing against {len(auctions)} historical auction results")
        
        results = []
        errors = []
        percent_errors = []
        
        for auction in auctions:
            auction_date_str = auction.get('date')
            actual_yield = float(auction.get('high_yield'))
            
            # Convert string to datetime
            auction_date = datetime.strptime(auction_date_str, '%Y-%m-%d')
            
            # Simulate making prediction 7 days before auction
            prediction_date = auction_date - timedelta(days=7)
            prediction_date_str = prediction_date.strftime('%Y-%m-%d')
            
            # Get Treasury data up to the prediction date
            treasury_df = fetcher.get_treasury_data(
                bond_type, 
                force_update=False,
                end_date=prediction_date_str
            )
            
            if treasury_df.empty:
                logger.warning(f"No treasury data available for prediction date {prediction_date_str}")
                continue
            
            # Use a more sophisticated prediction approach
            # We'll use the predictor module instead of just the current yield + fixed adjustment
            prediction_result = predictor.predict_auction_yield(
                bond_type, 
                prediction_date=prediction_date_str,
                historical_mode=True
            )
            
            if not prediction_result.get('success'):
                logger.warning(f"Prediction failed for auction date {auction_date_str}")
                continue
                
            predicted_yield = prediction_result.get('predicted_yield')
            
            # Calculate error
            error = predicted_yield - actual_yield
            percent_error = (error / actual_yield) * 100 if actual_yield != 0 else float('inf')
            
            result = {
                'auction_date': auction_date_str,
                'prediction_date': prediction_date_str,
                'actual_yield': actual_yield,
                'predicted_yield': predicted_yield,
                'error': error,
                'percent_error': percent_error
            }
            
            results.append(result)
            errors.append(error)
            percent_errors.append(percent_error)
            
            logger.info(f"Auction {auction_date_str}: Predicted {predicted_yield:.3f}%, Actual {actual_yield:.3f}%, Error {error:.3f}%")
        
        # Calculate overall metrics
        metrics = {}
        if errors:
            metrics = {
                'mean_absolute_error': sum(abs(e) for e in errors) / len(errors),
                'root_mean_squared_error': (sum(e**2 for e in errors) / len(errors))**0.5,
                'mean_absolute_percent_error': sum(abs(p) for p in percent_errors) / len(percent_errors) if percent_errors else 0,
                'num_predictions': len(results),
                'num_underestimates': sum(1 for e in errors if e < 0),
                'num_overestimates': sum(1 for e in errors if e > 0)
            }
        
        # Save results to cache
        cache_file = os.path.join("../data/json", f"backtest_{bond_type}.json")
        backtest_data = {
            'bond_type': bond_type,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'FRED' if use_real_data else 'Internal',
            'metrics': metrics,
            'predictions': results
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(backtest_data, f, indent=2)
            logger.info(f"Saved back-testing results to {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving back-test results: {e}")
        
        all_results[bond_type] = {
            'success': True,
            'bond_type': bond_type,
            'metrics': metrics,
            'predictions': results
        }
        
        # Print summary
        if metrics:
            mae = metrics.get('mean_absolute_error', 0)
            rmse = metrics.get('root_mean_squared_error', 0)
            mape = metrics.get('mean_absolute_percent_error', 0)
            
            logger.info(f"Results for {bond_type}:")
            logger.info(f"  Mean Absolute Error: {mae:.4f}%")
            logger.info(f"  Root Mean Squared Error: {rmse:.4f}%")
            logger.info(f"  Mean Absolute Percent Error: {mape:.2f}%")
            logger.info(f"  Under/Over Estimates: {metrics.get('num_underestimates', 0)}/{metrics.get('num_overestimates', 0)}")
        else:
            logger.error(f"No valid prediction results for {bond_type}")
    
    # Save consolidated results
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved consolidated back-testing results to {output_file}")
        except Exception as e:
            logger.warning(f"Error saving consolidated results: {e}")
    
    # Overall summary
    successful_tests = sum(1 for r in all_results.values() if r.get('success', False))
    logger.info(f"Back-testing complete: {successful_tests} of {len(bond_types)} tests successful")
    
    # Calculate average errors across all bond types
    if successful_tests > 0:
        all_mae = [r.get('metrics', {}).get('mean_absolute_error', 0) 
                   for r in all_results.values() if r.get('success', False)]
        all_rmse = [r.get('metrics', {}).get('root_mean_squared_error', 0) 
                    for r in all_results.values() if r.get('success', False)]
        all_mape = [r.get('metrics', {}).get('mean_absolute_percent_error', 0) 
                    for r in all_results.values() if r.get('success', False)]
        
        avg_mae = sum(all_mae) / len(all_mae) if all_mae else 0
        avg_rmse = sum(all_rmse) / len(all_rmse) if all_rmse else 0
        avg_mape = sum(all_mape) / len(all_mape) if all_mape else 0
        
        logger.info(f"Average metrics across all bond types:")
        logger.info(f"  Average Mean Absolute Error: {avg_mae:.4f}%")
        logger.info(f"  Average Root Mean Squared Error: {avg_rmse:.4f}%")
        logger.info(f"  Average Mean Absolute Percent Error: {avg_mape:.2f}%")
    
    return all_results

def main():
    """Main function to run the back-testing script."""
    parser = argparse.ArgumentParser(description="Back-test bond yield predictions against historical auctions")
    parser.add_argument("--bond-types", "-b", nargs="+", help="Bond types to test (e.g., 10Y 20Y)")
    parser.add_argument("--num-auctions", "-n", type=int, default=10, help="Number of auctions to test")
    parser.add_argument("--output", "-o", help="Output file for consolidated results (JSON format)")
    parser.add_argument("--mock-data", "-m", action="store_true", help="Use mock data instead of real data")
    
    args = parser.parse_args()
    
    bond_types = args.bond_types
    num_auctions = args.num_auctions
    output_file = args.output if args.output else "../data/json/backtest_results.json"
    use_real_data = not args.mock_data
    
    run_backtest(bond_types, num_auctions, output_file, use_real_data)

if __name__ == "__main__":
    main()