#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proper back-testing script for Treasury bond yield predictions.

This script performs legitimate back-testing by:
1. Using the actual YieldPredictor implementation from the codebase
2. Using only data available before each auction date
3. Comparing predictions to actual auction results
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate

# Add bond-yield-predictor to path so we can import from it
sys.path.append('/Users/asanchez/dev/financial-analytics/bond-analytics/bond-yield-predictor')
from src.predictor import YieldPredictor
from src.data_fetcher import DataFetcher
from src.utils import setup_logger

# Configure logging
logger = setup_logger(__name__, "/Users/asanchez/dev/financial-analytics/bond-analytics/backtest_log.txt")

def load_auction_data(bond_type, data_fetcher):
    """
    Load real auction data for the given bond type.
    
    Args:
        bond_type: Bond type (e.g., '20Y')
        data_fetcher: DataFetcher instance
    
    Returns:
        List of auction data dictionaries with real auction results
    """
    logger.info(f"Loading auction data for {bond_type}")
    
    try:
        # First try to load from auction data file
        auctions_file = f"/Users/asanchez/dev/financial-analytics/bond-analytics/bond-yield-predictor/data/json/{bond_type}_auctions.json"
        if os.path.exists(auctions_file):
            with open(auctions_file, 'r') as f:
                auction_data = json.load(f)
                auctions = auction_data.get('auctions', [])
                if auctions:
                    logger.info(f"Loaded {len(auctions)} auctions from {auctions_file}")
                    # Sort by date (newest first)
                    auctions = sorted(auctions, key=lambda x: x.get('date', ''), reverse=True)
                    return auctions
        
        # Fallback to getting from data_fetcher
        logger.info(f"No auction file found, using data_fetcher to get auction data")
        result = data_fetcher.get_treasury_direct_auctions(bond_type)
        if result:
            logger.info(f"Retrieved {len(result)} auctions from data_fetcher")
            return result
        else:
            logger.warning(f"No auction data found for {bond_type}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading auction data: {str(e)}")
        return []

def run_backtest(bond_type, num_auctions=10, days_before=7):
    """
    Run proper back-testing on the YieldPredictor implementation.
    
    Args:
        bond_type: Bond type to test (e.g., '20Y')
        num_auctions: Number of auctions to test against
        days_before: Number of days before auction to make prediction
    
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Starting back-testing for {bond_type} bond ({num_auctions} auctions, {days_before} days before)")
    
    # Initialize the actual implementation classes
    data_fetcher = DataFetcher()
    predictor = YieldPredictor()
    
    # Get auction data
    auctions = load_auction_data(bond_type, data_fetcher)
    
    # Limit to requested number of auctions
    auctions = auctions[:num_auctions]
    
    if not auctions:
        logger.error(f"No auction data available for {bond_type}")
        return {
            'bond_type': bond_type,
            'success': False,
            'error': 'No auction data available'
        }
    
    # Load historical yield data
    treasury_data = data_fetcher.get_treasury_data(
        bond_type, 
        force_update=False
    )
    
    if treasury_data.empty:
        logger.error(f"No historical Treasury data available for {bond_type}")
        return {
            'bond_type': bond_type,
            'success': False, 
            'error': 'No historical Treasury data available'
        }
    
    # Store results
    results = []
    errors = []
    percent_errors = []
    
    print(f"\nTesting against {len(auctions)} historical {bond_type} auctions:")
    print(f"Making predictions {days_before} days before each auction date\n")
    
    # For each auction, make a prediction using data available days_before the auction
    for auction in auctions:
        try:
            # Get auction details
            auction_date_str = auction.get('date')
            if not auction_date_str:
                logger.warning(f"Auction missing date: {auction}")
                continue
                
            # Some auctions might have high_yield as a string
            try:
                actual_yield = float(auction.get('high_yield', 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid yield in auction: {auction}")
                continue
            
            # Parse auction date
            try:
                auction_date = datetime.strptime(auction_date_str, '%Y-%m-%d').date()
            except ValueError:
                # Try alternative format
                try:
                    auction_date = datetime.strptime(auction_date_str, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid auction date format: {auction_date_str}")
                    continue
            
            # Calculate prediction date (days_before the auction)
            prediction_date = auction_date - timedelta(days=days_before)
            prediction_date_str = prediction_date.strftime('%Y-%m-%d')
            
            print(f"Auction: {auction_date_str}, Prediction date: {prediction_date_str}")
            
            # Filter Treasury data to only use data available up to prediction date
            available_data = treasury_data[treasury_data['date'].dt.date <= prediction_date].copy()
            
            if len(available_data) < 30:  # Need sufficient history
                logger.warning(f"Insufficient historical data for {prediction_date_str}")
                print(f"  Insufficient historical data (only {len(available_data)} data points available)\n")
                continue
            
            # Use the actual predictor implementation to make a prediction
            logger.info(f"Making prediction for {bond_type} on {prediction_date_str} for auction {auction_date_str}")
            print(f"  Using actual YieldPredictor with {len(available_data)} data points")
            
            # Call the actual predict_yield method with historical mode
            prediction_result = predictor.predict_yield(
                available_data, 
                bond_type, 
                prediction_date_str
            )
            
            # Extract ensemble prediction (best prediction from all models)
            if 'predictions' in prediction_result and 'ensemble' in prediction_result['predictions']:
                predicted_yield = prediction_result['predictions']['ensemble']
            else:
                logger.warning(f"No ensemble prediction available for {prediction_date_str}")
                print(f"  No ensemble prediction available, using last known yield")
                predicted_yield = prediction_result.get('last_known_yield')
            
            # Calculate error
            error = predicted_yield - actual_yield
            percent_error = (error / actual_yield) * 100 if actual_yield != 0 else float('inf')
            
            # Store individual model predictions and their errors
            model_errors = {}
            for model, pred in prediction_result.get('predictions', {}).items():
                if model != 'ensemble' and pred is not None:
                    model_error = pred - actual_yield
                    model_errors[model] = {
                        'prediction': pred,
                        'error': model_error,
                        'percent_error': (model_error / actual_yield) * 100 if actual_yield != 0 else float('inf')
                    }
            
            result = {
                'auction_date': auction_date_str,
                'prediction_date': prediction_date_str,
                'actual_yield': actual_yield,
                'predicted_yield': predicted_yield,
                'error': error,
                'percent_error': percent_error,
                'model_errors': model_errors,
                'last_known_yield': prediction_result.get('last_known_yield'),
                'days_to_auction': days_before,
                'ensemble_method': prediction_result.get('ensemble_method')
            }
            
            results.append(result)
            errors.append(error)
            percent_errors.append(percent_error)
            
            print(f"  Last known yield: {result['last_known_yield']:.3f}%")
            print(f"  Predicted yield:  {predicted_yield:.3f}%")
            print(f"  Actual yield:     {actual_yield:.3f}%")
            print(f"  Error:            {error:.3f}%")
            print(f"  Ensemble method:  {result['ensemble_method'] or 'N/A'}")
            print(f"  Individual models:")
            
            for model, pred_data in model_errors.items():
                print(f"    {model:20}: {pred_data['prediction']:.3f}% (Error: {pred_data['error']:.3f}%)")
            
            print("")
        
        except Exception as e:
            logger.error(f"Error processing auction {auction.get('date', 'unknown')}: {str(e)}")
            print(f"  Error processing auction: {str(e)}\n")
    
    # Calculate overall metrics
    metrics = {}
    if errors:
        metrics = {
            'mean_absolute_error': sum(abs(e) for e in errors) / len(errors),
            'root_mean_squared_error': (sum(e**2 for e in errors) / len(errors))**0.5,
            'mean_absolute_percent_error': sum(abs(p) for p in percent_errors) / len(percent_errors) if percent_errors else 0,
            'num_predictions': len(results),
            'num_underestimates': sum(1 for e in errors if e < 0),
            'num_overestimates': sum(1 for e in errors if e > 0),
            'median_error': sorted(errors)[len(errors)//2] if errors else 0
        }
    
    # Calculate per-model metrics across all predictions
    model_metrics = {}
    for model in ['rolling_5d', 'rolling_10d', 'rolling_20d', 'arima', 
                 'exponential_smoothing', 'linear_regression', 'xgboost']:
        model_errors = []
        for result in results:
            if model in result.get('model_errors', {}):
                model_error = result['model_errors'][model]['error']
                model_errors.append(model_error)
        
        if model_errors:
            model_metrics[model] = {
                'mean_absolute_error': sum(abs(e) for e in model_errors) / len(model_errors),
                'num_predictions': len(model_errors),
                'num_underestimates': sum(1 for e in model_errors if e < 0),
                'num_overestimates': sum(1 for e in model_errors if e > 0)
            }
    
    # Prepare backtest results
    backtest_data = {
        'bond_type': bond_type,
        'timestamp': datetime.now().isoformat(),
        'days_before_auction': days_before,
        'auctions_tested': len(auctions),
        'valid_predictions': len(results),
        'success': True if results else False,
        'metrics': metrics,
        'model_metrics': model_metrics,
        'predictions': results
    }
    
    # Create a nice summary table
    if results:
        table_data = []
        for r in results:
            table_data.append([
                r['auction_date'],
                r['prediction_date'], 
                f"{r['predicted_yield']:.3f}%", 
                f"{r['actual_yield']:.3f}%", 
                f"{r['error']:.3f}%"
            ])
        
        headers = ["Auction Date", "Prediction Date", "Predicted Yield", "Actual Yield", "Error"]
        print("\n----- BACK-TESTING RESULTS -----")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\nSummary Metrics:")
        print(f"  Mean Absolute Error: {metrics.get('mean_absolute_error', 0):.4f}%")
        print(f"  Root Mean Squared Error: {metrics.get('root_mean_squared_error', 0):.4f}%")
        print(f"  Mean Absolute Percent Error: {metrics.get('mean_absolute_percent_error', 0):.2f}%")
        print(f"  Direction Bias: {metrics.get('num_underestimates', 0)} underestimates, "
              f"{metrics.get('num_overestimates', 0)} overestimates")
        
        print("\nModel-specific Mean Absolute Errors:")
        for model, model_metric in model_metrics.items():
            print(f"  {model:20}: {model_metric.get('mean_absolute_error', 0):.4f}%")
    
    # Save results to file
    output_file = f"/Users/asanchez/dev/financial-analytics/bond-analytics/real_backtest_{bond_type}.json"
    with open(output_file, 'w') as f:
        json.dump(backtest_data, f, indent=2)
    print(f"\nSaved back-testing results to {output_file}")
    
    return backtest_data

def main():
    """Parse command line arguments and run back-testing."""
    parser = argparse.ArgumentParser(description="Run proper back-testing on YieldPredictor implementation")
    parser.add_argument("--bond-type", "-b", default="20Y", help="Bond type to test (e.g., 20Y)")
    parser.add_argument("--num-auctions", "-n", type=int, default=10, help="Number of auctions to test")
    parser.add_argument("--days-before", "-d", type=int, default=7, 
                        help="Number of days before auction to make prediction")
    
    args = parser.parse_args()
    
    try:
        # Install tabulate if not present
        import importlib
        if importlib.util.find_spec("tabulate") is None:
            print("Installing required package: tabulate")
            import subprocess
            subprocess.check_call(["pip", "install", "tabulate"])
    except Exception as e:
        print(f"Warning: Could not install tabulate: {e}")
    
    # Run the back-test
    run_backtest(args.bond_type, args.num_auctions, args.days_before)

if __name__ == "__main__":
    main()