#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the predictor module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json
import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import YieldPredictor


class TestYieldPredictor(unittest.TestCase):
    """Tests for the YieldPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test cache directory
        self.test_cache_dir = os.path.join(os.path.dirname(__file__), 'test_cache')
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # Initialize predictor with test cache directory
        self.predictor = YieldPredictor(cache_dir=self.test_cache_dir)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100)
        values = [3.5 + 0.01 * i + 0.1 * np.sin(i * 0.1) for i in range(100)]
        self.sample_data = pd.DataFrame({
            'date': dates,
            'value': values
        })

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test cache directory
        for file in os.listdir(self.test_cache_dir):
            os.remove(os.path.join(self.test_cache_dir, file))
        os.rmdir(self.test_cache_dir)

    def test_calculate_rolling_averages(self):
        """Test calculation of rolling averages."""
        df = self.predictor._calculate_rolling_averages(self.sample_data, [5, 10])
        
        # Assertions
        self.assertIn('rolling_5d', df.columns)
        self.assertIn('rolling_10d', df.columns)
        
        # First rolling_5d should be NaN (not enough data)
        self.assertTrue(pd.isna(df['rolling_5d'].iloc[0]))
        self.assertTrue(pd.isna(df['rolling_5d'].iloc[3]))
        
        # After enough data points, rolling average should be valid
        self.assertFalse(pd.isna(df['rolling_5d'].iloc[5]))
        self.assertFalse(pd.isna(df['rolling_10d'].iloc[10]))
        
        # Verify calculations for 5-day average
        indices = [10, 20, 30]
        for i in indices:
            with self.subTest(i=i):
                expected = self.sample_data['value'].iloc[i-5:i].mean()
                self.assertAlmostEqual(df['rolling_5d'].iloc[i], expected)

    def test_fit_arima_model(self):
        """Test ARIMA model fitting."""
        arima_model, arima_fitted = self.predictor._fit_arima_model(self.sample_data)
        
        # Assertions
        self.assertIsNotNone(arima_model)
        self.assertIsInstance(arima_fitted, pd.DataFrame)
        self.assertIn('fitted', arima_fitted.columns)
        
        # Forecast should produce values
        forecast = arima_model.forecast(steps=1)
        self.assertEqual(len(forecast), 1)

    def test_fit_exponential_smoothing(self):
        """Test Exponential Smoothing model fitting."""
        exp_model, exp_fitted = self.predictor._fit_exponential_smoothing(self.sample_data)
        
        # Assertions
        self.assertIsNotNone(exp_model)
        self.assertIsInstance(exp_fitted, pd.DataFrame)
        self.assertIn('fitted', exp_fitted.columns)
        
        # Forecast should produce values
        forecast = exp_model.forecast(1)
        self.assertEqual(len(forecast), 1)

    def test_fit_linear_regression(self):
        """Test Linear Regression model fitting."""
        lr_model, lr_fitted = self.predictor._fit_linear_regression(self.sample_data, forecast_days=30)
        
        # Assertions
        self.assertIsNotNone(lr_model)
        self.assertIsInstance(lr_fitted, pd.DataFrame)
        self.assertIn('fitted', lr_fitted.columns)
        self.assertIn('day_index', lr_fitted.columns)
        
        # Predict next value
        next_day_index = lr_fitted['day_index'].iloc[-1] + 1
        prediction = lr_model.predict([[next_day_index]])[0]
        self.assertIsInstance(prediction, float)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 3.1, 3.9, 5.2])
        
        metrics = self.predictor._calculate_metrics(actual, predicted)
        
        # Assertions
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('r2', metrics)
        
        # MAE should be average of absolute differences
        expected_mae = np.mean(np.abs(actual - predicted))
        self.assertAlmostEqual(metrics['mae'], expected_mae)
        
        # RMSE should be sqrt of mean squared error
        expected_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        self.assertAlmostEqual(metrics['rmse'], expected_rmse)

    def test_predict_yield(self):
        """Test yield prediction."""
        bond_type = '10Y'
        result = self.predictor.predict_yield(self.sample_data, bond_type)
        
        # Assertions
        self.assertEqual(result['bond_type'], bond_type)
        self.assertIn('prediction_date', result)
        self.assertIn('predictions', result)
        self.assertIn('backtest_metrics', result)
        
        # Should have predictions from different models
        predictions = result['predictions']
        self.assertIn('rolling_5d', predictions)
        self.assertIn('rolling_10d', predictions)
        self.assertIn('rolling_20d', predictions)
        self.assertIn('arima', predictions)
        self.assertIn('exponential_smoothing', predictions)
        self.assertIn('linear_regression', predictions)
        self.assertIn('ensemble', predictions)
        
        # Should have prediction confidence
        self.assertIn('prediction_confidence', result)
        self.assertGreaterEqual(result['prediction_confidence'], 0.0)
        self.assertLessEqual(result['prediction_confidence'], 1.0)
        
        # Check that metrics file was created
        metrics_file = os.path.join(self.test_cache_dir, f"metrics_{bond_type}.json")
        self.assertTrue(os.path.exists(metrics_file))

    def test_backtest_predictions(self):
        """Test backtesting functionality."""
        bond_type = '10Y'
        test_window = 30
        result = self.predictor.backtest_predictions(self.sample_data, bond_type, test_window_days=test_window)
        
        # Assertions
        self.assertEqual(result['bond_type'], bond_type)
        self.assertEqual(result['backtest_window_days'], test_window)
        self.assertIn('metrics', result)
        
        # Should have metrics for different models
        metrics = result['metrics']
        expected_models = {'rolling_5d', 'rolling_10d', 'rolling_20d', 'arima', 'exponential_smoothing', 'linear_regression'}
        for model in expected_models:
            with self.subTest(model=model):
                self.assertIn(model, metrics)
                self.assertIn('mae', metrics[model])
                self.assertIn('rmse', metrics[model])
                
        # Check that backtest file was created
        backtest_file = os.path.join(self.test_cache_dir, f"backtest_{bond_type}.json")
        self.assertTrue(os.path.exists(backtest_file))


if __name__ == '__main__':
    unittest.main()