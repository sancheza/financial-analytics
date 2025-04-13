#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the Bond Yield Predictor.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Set


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Only add handlers if they don't exist to prevent duplicate logging
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def validate_bond_type(bond_type: str) -> bool:
    """
    Validate if the bond type is supported.
    
    Args:
        bond_type: Bond type string (e.g., '10Y')
        
    Returns:
        True if valid, False otherwise
    """
    valid_bond_types = {'1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '20Y', '30Y'}
    return bond_type in valid_bond_types


def get_supported_bond_types() -> List[str]:
    """
    Get a list of all supported bond types.
    
    Returns:
        List of bond type strings
    """
    return ['1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '20Y', '30Y']


def format_prediction_output(prediction_data: Dict[str, Any]) -> str:
    """
    Format prediction data for display in the console.
    
    Args:
        prediction_data: Dictionary of prediction data
        
    Returns:
        Formatted string for console output
    """
    if 'error' in prediction_data:
        return f"Error: {prediction_data['error']}"
    
    bond_type = prediction_data.get('bond_type', 'Unknown')
    prediction_date = prediction_data.get('prediction_date', 'Unknown')
    next_auction_date = prediction_data.get('next_auction_date', 'Unknown')
    last_yield = prediction_data.get('last_known_yield', 0.0)
    last_date = prediction_data.get('last_date', 'Unknown')
    
    output = []
    output.append(f"Prediction for {bond_type} Treasury Bond - {prediction_date}")
    
    # Add next auction date information
    if next_auction_date != 'Unknown':
        output.append(f"Next auction scheduled for: {next_auction_date}")
    
    # Yields are already in percentage form (e.g. 4.90 means 4.90%)
    output.append(f"Last known yield: {last_yield:.4f}% on {last_date}")
    output.append("")
    
    # Store ensemble prediction for the final prediction display
    ensemble_prediction = None
    
    # Add model predictions
    if 'predictions' in prediction_data:
        output.append("Individual Model Predictions:")
        for model, value in prediction_data['predictions'].items():
            if model == 'ensemble':
                ensemble_prediction = value
                continue
            output.append(f"  {model}: {value:.4f}%")
    
    # Highlight the final prediction (ensemble) separately
    if ensemble_prediction is not None:
        output.append(f"\nFINAL PREDICTION: {ensemble_prediction:.4f}%")
    
    # Add prediction confidence
    if 'prediction_confidence' in prediction_data:
        confidence = prediction_data['prediction_confidence']
        confidence_pct = confidence * 100
        output.append(f"Prediction Confidence: {confidence_pct:.2f}%")
    
    # Add backtest metrics for ensemble model
    if 'backtest_metrics' in prediction_data and 'ensemble' in prediction_data['backtest_metrics']:
        metrics = prediction_data['backtest_metrics']['ensemble']
        output.append("\nBacktest Metrics (Ensemble Model):")
        if 'mae' in metrics:
            output.append(f"  MAE: {metrics['mae']:.6f}")
        if 'rmse' in metrics:
            output.append(f"  RMSE: {metrics['rmse']:.6f}")
        if 'mape' in metrics:
            output.append(f"  MAPE: {metrics['mape']:.2f}%")
        if 'r2' in metrics:
            output.append(f"  R²: {metrics['r2']:.4f}")
    
    return '\n'.join(output)


def clean_json_directory(json_dir: str, keep_days: int = 7) -> None:
    """
    Clean up old JSON files in the cache directory.
    
    Args:
        json_dir: Directory containing JSON files
        keep_days: Number of days of data to keep
    """
    import time
    from datetime import datetime, timedelta
    
    if not os.path.exists(json_dir):
        return
    
    current_time = time.time()
    cutoff_time = current_time - (keep_days * 24 * 60 * 60)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning JSON directory: {json_dir}")
    
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(json_dir, filename)
        file_mod_time = os.path.getmtime(filepath)
        
        if file_mod_time < cutoff_time:
            try:
                os.remove(filepath)
                logger.info(f"Removed old file: {filename}")
            except OSError as e:
                logger.error(f"Error removing {filename}: {e}")


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def log_message(message, log_file='data/log.txt'):
    with open(log_file, 'a') as file:
        file.write(f"{message}\n")


def clear_log(log_file='data/log.txt'):
    with open(log_file, 'w') as file:
        file.write("")