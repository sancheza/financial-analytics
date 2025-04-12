#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for the Bond Yield Predictor application.
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd
import json
import numpy as np
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from src.predictor import YieldPredictor
from src.utils import (
    setup_logger, 
    validate_bond_type, 
    get_supported_bond_types,
    format_prediction_output,
    clean_json_directory
)

# Configure logging
logger = setup_logger(__name__, "../data/log.txt")

# Script version
__version__ = "1.0.0"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict U.S. Treasury bond yields for upcoming auctions."
    )
    
    # Version information
    parser.add_argument(
        "-v", "--version", 
        action="version", 
        version=f"Bond Yield Predictor v{__version__}"
    )
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Predict bond yields for upcoming auctions"
    )
    predict_parser.add_argument(
        "--bondtype", 
        type=str, 
        help="Bond type to predict (e.g., 10Y). If omitted, predict all types."
    )
    predict_parser.add_argument(
        "--force-update", 
        action="store_true", 
        help="Bypass cache and fetch fresh data"
    )
    predict_parser.add_argument(
        "--show-prediction", 
        action="store_true", 
        help="Print prediction to console"
    )
    predict_parser.add_argument(
        "--date",
        type=str,
        help="Date to predict for (YYYY-MM-DD). If omitted, predict for next trading day."
    )
    predict_parser.add_argument(
        "--backtest",
        type=int,
        metavar="DAYS",
        help="Perform backtesting for the specified number of days"
    )
    
    # Auction Backtest command (new)
    auction_backtest_parser = subparsers.add_parser(
        "auction-backtest", 
        help="Backtest predictions against historical auction results"
    )
    auction_backtest_parser.add_argument(
        "--bondtype", 
        type=str, 
        help="Bond type to backtest (e.g., 10Y). If omitted, backtest all types."
    )
    auction_backtest_parser.add_argument(
        "--n-auctions",
        type=int,
        default=20,
        help="Number of historical auctions to test (default: 20)"
    )
    
    # Show command
    show_parser = subparsers.add_parser(
        "show", 
        help="Show existing predictions"
    )
    show_parser.add_argument(
        "--bondtype", 
        type=str, 
        help="Bond type to show (e.g., 10Y). If omitted, show all types."
    )
    
    # Update command
    update_parser = subparsers.add_parser(
        "update", 
        help="Update data without making predictions"
    )
    update_parser.add_argument(
        "--bondtype", 
        type=str, 
        help="Bond type to update (e.g., 10Y). If omitted, update all types."
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        "test", 
        help="Run tests for the predictor"
    )
    
    return parser.parse_args()


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if valid, False otherwise
    """
    # Load environment variables from .env file
    load_dotenv()
    
    required_vars = ['FRED_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with the required variables.")
        return False
    
    return True


def get_bond_types_to_process(bondtype_arg: Optional[str]) -> List[str]:
    """
    Determine which bond types to process based on the command-line argument.
    
    Args:
        bondtype_arg: Bond type from command line, or None
        
    Returns:
        List of bond types to process
    """
    if bondtype_arg:
        if validate_bond_type(bondtype_arg):
            return [bondtype_arg]
        else:
            logger.error(f"Invalid bond type: {bondtype_arg}")
            logger.info(f"Supported bond types: {', '.join(get_supported_bond_types())}")
            return []
    else:
        # Process all supported bond types
        return get_supported_bond_types()


def predict_bond_yields(bond_types: List[str], force_update: bool, 
                       show_prediction: bool, prediction_date: Optional[str] = None,
                       backtest_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Predict bond yields for the specified bond types.
    
    Args:
        bond_types: List of bond types to process
        force_update: Whether to bypass cache and fetch fresh data
        show_prediction: Whether to print predictions to console
        prediction_date: Date to predict for, or None for next trading day
        backtest_days: Number of days to use for backtesting, or None to skip
        
    Returns:
        Dictionary of results indexed by bond type
    """
    # Import datetime module directly in the function scope
    from datetime import datetime
    import traceback
    
    data_fetcher = DataFetcher()
    predictor = YieldPredictor()
    
    results = {}
    
    for bond_type in bond_types:
        logger.info(f"Processing {bond_type} bond")
        
        try:
            # Fetch yield data
            df = data_fetcher.get_treasury_data(bond_type, force_update)
            
            if df.empty:
                logger.error(f"No data available for {bond_type}")
                results[bond_type] = {'error': 'No data available'}
                continue
                
            # Fetch auction data (for future use)
            auctions = data_fetcher.get_treasury_direct_auctions(bond_type, force_update)
            logger.info(f"Found {len(auctions)} {bond_type} auctions")
            
            # Perform backtesting if requested
            if backtest_days:
                logger.info(f"Backtesting {bond_type} predictions for {backtest_days} days")
                backtest_result = predictor.backtest_predictions(df, bond_type, backtest_days)
                
                # Save backtest metrics
                results[f"{bond_type}_backtest"] = backtest_result
                
                if show_prediction and 'metrics' in backtest_result:
                    print(f"\n=== Backtesting Results for {bond_type} ===")
                    for model, metrics in backtest_result['metrics'].items():
                        print(f"\nModel: {model}")
                        for metric_name, value in metrics.items():
                            print(f"  {metric_name}: {value:.6f}")
                    
                    # Determine best model based on RMSE
                    best_model = min(
                        backtest_result['metrics'].items(), 
                        key=lambda x: x[1].get('rmse', float('inf'))
                    )[0]
                    print(f"\nBest model for {bond_type}: {best_model}")
            
            # Make prediction
            prediction = predictor.predict_yield(df, bond_type, prediction_date)
            results[bond_type] = prediction
            
            # Create visualization
            predictor.visualize_predictions(df, bond_type, prediction)
            
            if show_prediction:
                print(f"\n=== Prediction for {bond_type} ===")
                print(format_prediction_output(prediction))
                
        except Exception as e:
            logger.error(f"Error processing {bond_type}: {str(e)}")
            logger.error(traceback.format_exc())
            results[bond_type] = {'error': str(e)}
    
    return results


def show_existing_predictions(bond_types: List[str]) -> None:
    """
    Show existing predictions for the specified bond types.
    
    Args:
        bond_types: List of bond types to show
    """
    cache_dir = "../data/json"
    
    for bond_type in bond_types:
        metrics_file = os.path.join(cache_dir, f"metrics_{bond_type}.json")
        
        if not os.path.exists(metrics_file):
            print(f"No prediction found for {bond_type}")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                prediction_data = json.load(f)
                
            print(f"\n=== Prediction for {bond_type} ===")
            print(format_prediction_output(prediction_data))
                
        except Exception as e:
            logger.error(f"Error reading prediction for {bond_type}: {str(e)}")
            print(f"Error reading prediction for {bond_type}")


def update_data(bond_types: List[str]) -> None:
    """
    Update data for the specified bond types without making predictions.
    
    Args:
        bond_types: List of bond types to update
    """
    data_fetcher = DataFetcher()
    
    for bond_type in bond_types:
        logger.info(f"Updating data for {bond_type}")
        
        try:
            # Force update
            df = data_fetcher.get_treasury_data(bond_type, force_update=True)
            auctions = data_fetcher.get_treasury_direct_auctions(bond_type, force_update=True)
            
            print(f"Updated {bond_type} yield data: {len(df)} records")
            print(f"Updated {bond_type} auction data: {len(auctions)} auctions")
                
        except Exception as e:
            logger.error(f"Error updating {bond_type}: {str(e)}")
            print(f"Error updating {bond_type}")


def run_tests() -> None:
    """Run unit tests."""
    import unittest
    
    # Define test paths
    test_path = "../tests"
    
    # Run tests
    print("Running tests...")
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_path)
    
    test_runner = unittest.TextTestRunner()
    test_result = test_runner.run(test_suite)
    
    # Report results
    if test_result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(test_result.failures)} failures, {len(test_result.errors)} errors")
        sys.exit(1)


def backtest_against_auctions(bond_types: List[str], n_auctions: int = 20) -> Dict[str, Any]:
    """
    Backtest the prediction algorithm against real historical auction results.
    
    Args:
        bond_types: List of bond types to test (e.g., ['10Y', '20Y'])
        n_auctions: Number of historical auctions to test against
        
    Returns:
        Dictionary with results by bond type
    """
    data_fetcher = DataFetcher()
    predictor = YieldPredictor()
    
    results = {}
    
    for bond_type in bond_types:
        logger.info(f"Backtesting {bond_type} against historical auction results")
        
        try:
            # Fetch all historical yield data
            df = data_fetcher.get_treasury_data(bond_type)
            
            if df.empty:
                logger.error(f"No yield data available for {bond_type}")
                results[bond_type] = {'error': 'No yield data available'}
                continue
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Fetch historical auction data
            auctions = data_fetcher.get_treasury_direct_auctions(bond_type, force_update=True)
            
            if not auctions:
                logger.error(f"No auction data available for {bond_type}")
                results[bond_type] = {'error': 'No auction data available'}
                continue
                
            logger.info(f"Found {len(auctions)} {bond_type} auctions. Using the last {min(n_auctions, len(auctions))}.")
            
            # Limit to the specified number of most recent auctions
            auctions = sorted(auctions, key=lambda x: x.get('date', ''), reverse=True)[:n_auctions]
            
            auction_results = []
            
            # For each auction, predict the yield using data available before the auction
            for auction in auctions:
                auction_date_str = auction.get('date')
                if not auction_date_str:
                    continue
                
                try:
                    # Directly use pandas to_datetime without chaining with datetime module
                    auction_date = pd.to_datetime(auction_date_str)
                    
                    # Get high yield (the auction result)
                    high_yield_str = auction.get('high_yield', '').replace('%', '')
                    if not high_yield_str:
                        continue
                        
                    high_yield = float(high_yield_str)
                    
                    # Use data up to 1 day before the auction
                    pre_auction_date = auction_date - pd.Timedelta(days=1)
                    pre_auction_data = df[df['date'] <= pre_auction_date]
                    
                    if len(pre_auction_data) < 30:  # Need enough data for prediction
                        continue
                    
                    # Predict yield for the auction date
                    prediction = predictor.predict_yield(pre_auction_data, bond_type, auction_date_str)
                    
                    # Compare prediction with actual result
                    if 'predictions' in prediction and 'ensemble' in prediction['predictions']:
                        pred_yield = prediction['predictions']['ensemble']
                        error = pred_yield - high_yield
                        pct_error = (error / high_yield) * 100
                        
                        # Get applied adjustments if any
                        adjustments = prediction.get('adjustments', {})
                        
                        auction_result = {
                            'auction_date': auction_date_str,
                            'cusip': auction.get('cusip', 'N/A'),
                            'actual_yield': high_yield,
                            'predicted_yield': pred_yield,
                            'error': error,
                            'pct_error': pct_error,
                            'model_predictions': prediction['predictions'],
                            'adjustments': adjustments,
                            'ensemble_method': prediction.get('ensemble_method', 'unknown')
                        }
                        
                        auction_results.append(auction_result)
                
                except Exception as e:
                    logger.error(f"Error processing auction on {auction_date_str}: {str(e)}")
            
            # Calculate aggregate metrics
            if auction_results:
                errors = [r['error'] for r in auction_results]
                abs_errors = [abs(e) for e in errors]
                pct_errors = [r['pct_error'] for r in auction_results]
                abs_pct_errors = [abs(e) for e in pct_errors]
                
                # Use datetime.now() directly, don't use datetime.datetime
                from datetime import datetime
                
                metrics = {
                    'mae': np.mean(abs_errors),
                    'rmse': np.sqrt(np.mean(np.square(errors))),
                    'mape': np.mean(abs_pct_errors),
                    'bias': np.mean(errors),
                    'n_auctions': len(auction_results),
                    'max_error': max(abs_errors),
                    'min_error': min(abs_errors),
                    'auction_dates_covered': [r['auction_date'] for r in auction_results],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Find best performing model across all auctions
                model_errors = {}
                for model in ['rolling_5d', 'rolling_10d', 'rolling_20d', 
                             'arima', 'exponential_smoothing', 'linear_regression']:
                    model_errs = []
                    for r in auction_results:
                        if model in r['model_predictions']:
                            pred = r['model_predictions'][model]
                            actual = r['actual_yield']
                            model_errs.append(abs(pred - actual))
                    
                    if model_errs:
                        model_errors[model] = np.mean(model_errs)
                
                if model_errors:
                    best_model = min(model_errors.items(), key=lambda x: x[1])[0]
                    metrics['best_model'] = best_model
                    metrics['model_errors'] = model_errors
                
                results[bond_type] = {
                    'auction_results': auction_results,
                    'metrics': metrics
                }
                
                # Save results to file
                auction_backtest_file = os.path.join(predictor.cache_dir, f"auction_backtest_{bond_type}.json")
                try:
                    with open(auction_backtest_file, 'w') as f:
                        json.dump(results[bond_type], f, indent=2)
                        
                    logger.info(f"Saved {bond_type} auction backtest results to {auction_backtest_file}")
                except Exception as e:
                    logger.error(f"Error saving auction backtest results to file: {str(e)}")
            
            else:
                results[bond_type] = {'error': 'No valid auction results found for comparison'}
                
        except Exception as e:
            logger.error(f"Error backtesting {bond_type} against auctions: {str(e)}")
            results[bond_type] = {'error': str(e)}
    
    return results


def print_auction_backtest_results(results: Dict[str, Any]) -> None:
    """
    Print results from auction backtesting in a readable format.
    
    Args:
        results: Results from backtest_against_auctions
    """
    for bond_type, bond_results in results.items():
        print(f"\n{'='*60}")
        print(f"AUCTION BACKTEST RESULTS: {bond_type} TREASURY BOND")
        print(f"{'='*60}")
        
        if 'error' in bond_results:
            print(f"Error: {bond_results['error']}")
            continue
            
        metrics = bond_results.get('metrics', {})
        if not metrics:
            print("No metrics available.")
            continue
            
        n_auctions = metrics.get('n_auctions', 0)
        print(f"Number of auctions tested: {n_auctions}")
        print(f"Mean Absolute Error (MAE): {metrics.get('mae', 0):.4f}%")
        print(f"Root Mean Square Error (RMSE): {metrics.get('rmse', 0):.4f}%")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 0):.4f}%")
        print(f"Bias (average error): {metrics.get('bias', 0):.4f}% " + 
              ("(predictions tend to be high)" if metrics.get('bias', 0) > 0 else 
               "(predictions tend to be low)" if metrics.get('bias', 0) < 0 else ""))
        
        if 'best_model' in metrics:
            print(f"\nBest performing model: {metrics['best_model']}")
            
            # Print model performance comparison
            if 'model_errors' in metrics:
                print("\nModel Performance (MAE):")
                for model, error in sorted(metrics['model_errors'].items(), key=lambda x: x[1]):
                    print(f"  {model}: {error:.4f}%")
        
        print("\nIndividual Auction Results:")
        print(f"{'Auction Date':<15} {'CUSIP':<12} {'Actual':<8} {'Predicted':<8} {'Error':<8} {'% Error':<8}")
        print(f"{'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        auction_results = bond_results.get('auction_results', [])
        for result in auction_results:
            print(f"{result['auction_date']:<15} "
                  f"{result['cusip']:<12} "
                  f"{result['actual_yield']:<8.3f} "
                  f"{result['predicted_yield']:<8.3f} "
                  f"{result['error']:<+8.3f} "
                  f"{result['pct_error']:<+8.3f}%")


def predict_yield_command(args):
    """Run yield prediction for a specific bond."""
    logger.info(f"Processing {args.bondtype} bond")
    
    data_fetcher = DataFetcher(cache_dir=args.cache_dir)
    predictor = YieldPredictor(cache_dir=args.cache_dir)
    
    # Fetch data
    df = data_fetcher.get_treasury_data(args.bondtype, force_update=args.force_update)
    if df.empty:
        logger.error(f"No data available for {args.bondtype}")
        return
    
    # Get historical auction data for context
    auctions = data_fetcher.get_treasury_direct_auctions(args.bondtype)
    logger.info(f"Found {len(auctions)} {args.bondtype} auctions")
    
    # Predict yield
    prediction_date = args.date
    prediction = predictor.predict_yield(df, args.bondtype, prediction_date)
    
    # Add accurate auction data to the prediction
    prediction = data_fetcher.add_auction_data_to_prediction(prediction, args.bondtype, args.force_update)
    
    # Save prediction to file
    if not args.no_save:
        prediction_file = os.path.join(args.cache_dir, f"prediction_{args.bondtype}.json")
        try:
            with open(prediction_file, 'w') as f:
                json.dump(prediction, f, indent=2)
            logger.info(f"Saved prediction to {prediction_file}")
        except Exception as e:
            logger.error(f"Error saving prediction to {prediction_file}: {e}")
    
    # Show prediction if requested
    if args.show_prediction:
        output = format_prediction_output(prediction)
        
        print(f"\n=== Prediction for {args.bondtype} ===")
        print(output)
    
    return prediction


def main():
    """Main function."""
    args = parse_args()
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Clean up old JSON files
    clean_json_directory("../data/json", keep_days=30)
    
    if args.command == "predict":
        bond_types = get_bond_types_to_process(args.bondtype)
        if not bond_types:
            return 1
            
        predict_bond_yields(
            bond_types, 
            args.force_update, 
            args.show_prediction,
            args.date,
            args.backtest
        )
        
    elif args.command == "auction-backtest":
        bond_types = get_bond_types_to_process(args.bondtype)
        if not bond_types:
            return 1
            
        results = backtest_against_auctions(bond_types, args.n_auctions)
        print_auction_backtest_results(results)
    
    elif args.command == "show":
        bond_types = get_bond_types_to_process(args.bondtype)
        if not bond_types:
            return 1
            
        show_existing_predictions(bond_types)
        
    elif args.command == "update":
        bond_types = get_bond_types_to_process(args.bondtype)
        if not bond_types:
            return 1
            
        update_data(bond_types)
        
    elif args.command == "test":
        run_tests()
        
    else:
        # No command specified, show help
        print("Please specify a command. Use -h for help.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())