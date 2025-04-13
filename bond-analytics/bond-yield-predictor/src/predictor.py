#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor module for Treasury bond yield predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime # Add this line
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from src.utils import setup_logger
import traceback # Add this import
from sklearn.ensemble import RandomForestRegressor # Example
import xgboost as xgb # Example

# Configure logging
logger = setup_logger(__name__, "../data/log.txt")

class YieldPredictor:
    """Class to predict Treasury bond yields."""
    
    def __init__(self, cache_dir: str = "../data/json"):
        """
        Initialize the YieldPredictor.
        
        Args:
            cache_dir: Directory for caching data and metrics
        """
        self.cache_dir = cache_dir
        self.logger = setup_logger(__name__, "../data/log.txt")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _calculate_rolling_averages(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Calculate rolling averages for the given data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            windows: List of window sizes for rolling averages
            
        Returns:
            DataFrame with rolling averages added as columns
        """
        df = data.copy()
        df = df.sort_values('date')
        
        for window in windows:
            df[f'rolling_{window}d'] = df['value'].rolling(window=window).mean()
            
        return df
    
    def _fit_arima_model(self, series: pd.Series, order: tuple = (5, 1, 0)) -> Optional[float]:
        """Fit ARIMA model and forecast."""
        try:
            # Ensure the series index is datetime and has frequency
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            series = series.asfreq(pd.infer_freq(series.index) or 'B') # Infer frequency or assume Business days
            series = series.fillna(method='ffill') # Fill any gaps after setting frequency

            if len(series) < sum(order) + 10: # Basic check for sufficient data
                 self.logger.warning(f"ARIMA: Not enough data points ({len(series)}) for order {order}")
                 return None

            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e:
            # Log the full traceback for detailed debugging
            self.logger.error(f"Error predicting with ARIMA: {e}\n{traceback.format_exc()}")
            return None # Return None explicitly on error

    def _fit_exponential_smoothing(self, series: pd.Series) -> Optional[float]:
        """Fit Exponential Smoothing model and forecast."""
        try:
            # Ensure the series index is datetime and has frequency
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            series = series.asfreq(pd.infer_freq(series.index) or 'B') # Infer frequency or assume Business days
            series = series.fillna(method='ffill') # Fill any gaps

            if len(series) < 10: # Basic check
                self.logger.warning(f"Exponential Smoothing: Not enough data points ({len(series)})")
                return None

            # Try with different configurations, starting simple
            model = ExponentialSmoothing(series, trend='add', seasonal=None) # Additive trend often works for yields
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e:
             # Log the full traceback for detailed debugging
            self.logger.error(f"Error predicting with Exponential Smoothing: {e}\n{traceback.format_exc()}")
            return None # Return None explicitly on error
    
    def _fit_linear_regression(self, df: pd.DataFrame, forecast_days: int = 1) -> Optional[float]:
        """Fit Linear Regression model using time index and potentially other features."""
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['day_index'] = (df['date'] - df['date'].min()).dt.days

            # Define features: time index + any available FRED columns
            feature_cols = ['day_index']
            fred_cols = ['FEDFUNDS', 'T10YIE', 'VIXCLS'] # Match columns from DataFetcher
            available_fred_cols = [col for col in fred_cols if col in df.columns and df[col].notna().any()]
            feature_cols.extend(available_fred_cols) # <<< Added closing parenthesis here

            if not available_fred_cols:
                 self.logger.warning("Linear Regression: No FRED features available.")

            # Drop rows with NaN in features or target
            df_train = df.dropna(subset=feature_cols + ['yield'])

            if len(df_train) < 5: # Need minimum data
                 self.logger.warning(f"Linear Regression: Not enough non-NaN data points ({len(df_train)})")
                 return None

            X = df_train[feature_cols]
            y = df_train['yield']

            model = LinearRegression()
            model.fit(X, y)

            # Prepare features for prediction (last day + forecast_days)
            last_day_index = df_train['day_index'].iloc[-1]
            predict_day_index = last_day_index + forecast_days
            predict_features = {'day_index': predict_day_index}
            for col in available_fred_cols:
                predict_features[col] = df_train[col].iloc[-1] # Use last known value for prediction

            predict_df = pd.DataFrame([predict_features])
            # Ensure columns match training columns order/names
            predict_df = predict_df.reindex(columns=X.columns, fill_value=0)

            prediction = model.predict(predict_df)
            return prediction[0]
        except Exception as e:
            self.logger.error(f"Error predicting with Linear Regression: {e}\n{traceback.format_exc()}")
            return None
    
    def _fit_xgboost(self, df: pd.DataFrame, forecast_days: int = 1) -> Optional[float]:
        """Fit XGBoost model."""
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['day_index'] = (df['date'] - df['date'].min()).dt.days
            # Add other engineered features if desired (e.g., lags, rolling stats)
            # df['yield_lag_1'] = df['yield'].shift(1)
            # df['yield_roll_5'] = df['yield'].rolling(window=5).mean()

            # Feature Engineering Examples
            df['yield_change_1d'] = df['yield'].diff() # 1-day change
            df['yield_change_5d'] = df['yield'].diff(5) # 5-day change
            df['yield_volatility_10d'] = df['yield'].rolling(window=10).std() # 10-day rolling std dev

            feature_cols = ['day_index']
            # Add FRED columns if available
            fred_cols = ['FEDFUNDS', 'T10YIE', 'VIXCLS']
            available_fred_cols = [col for col in fred_cols if col in df.columns and df[col].notna().any()]
            feature_cols.extend(available_fred_cols)

            # Add engineered features
            engineered_features = ['yield_change_1d', 'yield_change_5d', 'yield_volatility_10d']
            available_engineered = [col for col in engineered_features if col in df.columns and df[col].notna().any()]
            feature_cols.extend(available_engineered)

            df_train = df.dropna(subset=feature_cols + ['yield'])

            if len(df_train) < 20: # Need more data for complex models
                self.logger.warning(f"XGBoost: Not enough non-NaN data points ({len(df_train)})")
                return None

            X = df_train[feature_cols]
            y = df_train['yield']

            # Basic XGBoost Regressor - parameters can be tuned
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            model.fit(X, y)

            # Prepare features for prediction
            last_day_index = df_train['day_index'].iloc[-1]
            predict_day_index = last_day_index + forecast_days
            predict_features = {'day_index': predict_day_index}
            for col in available_fred_cols:
                predict_features[col] = df_train[col].iloc[-1] # Use last known value
            # Add last known values for other engineered features if used

            for col in available_engineered:
                # Use last known value, handle potential NaNs if window extends beyond data
                predict_features[col] = df_train[col].iloc[-1] if pd.notna(df_train[col].iloc[-1]) else 0

            predict_df = pd.DataFrame([predict_features])
            # Ensure columns match training columns order/names
            predict_df = predict_df.reindex(columns=X.columns, fill_value=0) # Or use appropriate fill value

            prediction = model.predict(predict_df)
            return prediction[0]
        except Exception as e:
            self.logger.error(f"Error predicting with XGBoost: {e}\n{traceback.format_exc()}")
            return None
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for model predictions.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Filter out NaN values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        if sum(mask) < 2:
            self.logger.warning("Not enough valid data points to calculate metrics")
            return metrics
            
        actual_valid = actual[mask]
        predicted_valid = predicted[mask]
        
        # Calculate metrics
        metrics['mae'] = mean_absolute_error(actual_valid, predicted_valid)
        metrics['rmse'] = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        non_zero_mask = actual_valid != 0
        if sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((actual_valid[non_zero_mask] - predicted_valid[non_zero_mask]) / actual_valid[non_zero_mask])) * 100
            metrics['mape'] = mape
        
        # Calculate R²
        metrics['r2'] = r2_score(actual_valid, predicted_valid)
        
        return metrics
    
    def _calculate_dynamic_ensemble(self, predictions: Dict[str, float], backtest_metrics: Dict[str, Dict[str, float]]) -> Optional[float]:
        """Calculate ensemble prediction using weights based on backtest performance (e.g., inverse MAE)."""
        weights = {}
        total_inverse_error = 0
        valid_predictions = 0

        for model, pred in predictions.items():
            if model == 'ensemble': continue # Skip previous ensemble calculation
            if pred is None: continue

            # Get MAE for this model from backtest metrics
            mae = backtest_metrics.get(model, {}).get('mae')

            if mae is not None and mae > 1e-6: # Use MAE if available and non-zero
                inverse_error = 1.0 / mae
                weights[model] = inverse_error
                total_inverse_error += inverse_error
                valid_predictions += 1
            else:
                # Fallback: If no MAE, give it a default small weight or equal weight
                weights[model] = 1.0 # Simple equal weight fallback
                total_inverse_error += 1.0
                valid_predictions +=1


        if valid_predictions == 0:
            self.logger.warning("Dynamic Ensemble: No valid predictions or metrics found.")
            return None
        if total_inverse_error == 0: # Avoid division by zero if all weights ended up zero
             self.logger.warning("Dynamic Ensemble: Total inverse error is zero, falling back to simple average.")
             return sum(p for p in predictions.values() if p is not None and p != predictions.get('ensemble')) / valid_predictions


        # Calculate weighted average
        ensemble_prediction = 0.0
        for model, pred in predictions.items():
             if model in weights:
                 normalized_weight = weights[model] / total_inverse_error
                 ensemble_prediction += normalized_weight * pred

        return ensemble_prediction

    def _get_recent_auction_bias(self, bond_type: str, lookback: int = 5) -> float:
        """Calculate average prediction error (bias) from recent auction backtests."""
        bias = 0.0
        # Ensure self.cache_dir is used correctly if defined in __init__
        # Assuming self.cache_dir points to '../data/json'
        backtest_file = os.path.join(self.cache_dir, f'auction_backtest_{bond_type}.json')
        if not os.path.exists(backtest_file):
            self.logger.warning(f"Auction backtest file not found: {backtest_file}. Cannot calculate adaptive bias.")
            return bias # Return 0 bias

        try:
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)

            results = backtest_data.get('results', [])
            if not results:
                self.logger.warning(f"No results found in {backtest_file}.")
                return bias

            # Sort results by date descending to get the most recent
            results.sort(key=lambda x: x.get('auction_date', ''), reverse=True)

            # Get errors from the last 'lookback' auctions
            recent_errors = [r.get('error') for r in results[:lookback] if r.get('error') is not None]

            if recent_errors:
                bias = sum(recent_errors) / len(recent_errors)
                self.logger.info(f"Calculated adaptive bias for {bond_type} from last {len(recent_errors)} auctions: {bias:.6f}")
            else:
                 self.logger.warning(f"No valid errors found in recent {lookback} auctions for {bond_type} in {backtest_file}.")

        except Exception as e:
            self.logger.error(f"Error reading or processing auction backtest file {backtest_file}: {e}\n{traceback.format_exc()}")

        return bias

    def _calculate_pre_auction_weighted_prediction(self, df: pd.DataFrame, days_to_auction: int) -> float:
        """
        Calculate a prediction weighted more heavily on recent days if close to an auction.
        This is based on the observation that yields immediately preceding an auction
        are highly predictive of the auction result.
        
        Args:
            df: DataFrame with historical yield data
            days_to_auction: Number of days until next auction (if known)
            
        Returns:
            Weighted yield prediction
        """
        if df.empty:
            return None
            
        # Sort by date and get the most recent points
        df = df.sort_values('date')
        
        # Use last 30 days by default, or fewer if not available
        recent_days = min(30, len(df))
        recent_df = df.iloc[-recent_days:]
        
        if recent_days < 3:
            # Not enough data for weighted calculation
            return float(df['yield'].iloc[-1])
            
        # Calculate weights based on recency - more recent = higher weight
        weights = np.linspace(0.5, 10, recent_days)  # Linear weights from 0.5 to 10
        
        # If we know days until auction, adjust weights accordingly
        if days_to_auction is not None and days_to_auction < 10:
            # Exponential boost for very near-term auction (within 10 days)
            recency_boost = np.exp(-np.arange(recent_days)/2)  # Exponential decay
            weights = weights * recency_boost
            
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Calculate weighted average
        weighted_pred = (recent_df['yield'] * weights).sum()
        
        return float(weighted_pred)

    def _calculate_auction_aware_ensemble(self, 
                                    predictions: Dict[str, float], 
                                    days_to_auction: int,
                                    backtest_metrics: Dict[str, Dict[str, float]]) -> Optional[float]:
        """
        Calculate an ensemble prediction with greater weight on pre-auction prediction when close to auction date.
        
        Args:
            predictions: Dictionary of model predictions
            days_to_auction: Number of days until next auction
            backtest_metrics: Dictionary of backtest metrics for each model
            
        Returns:
            Auction-aware ensemble prediction
        """
        if not predictions:
            return None
            
        # Base weights on inverse MAE from backtest metrics, just like dynamic ensemble
        weights = {}
        total_weight = 0
        
        # Start with inverse MAE weights like regular dynamic ensemble
        for model, pred in predictions.items():
            if model == 'ensemble':
                continue # Skip previous ensemble calculation
            
            # Get MAE for this model from backtest metrics
            mae = backtest_metrics.get(model, {}).get('mae')
            
            if mae is not None and mae > 1e-6:
                inverse_error = 1.0 / mae
                weights[model] = inverse_error
            else:
                # Fallback: If no MAE, give it a default weight 
                weights[model] = 1.0
        
        # Apply auction proximity adjustment for pre-auction weighted prediction
        if 'pre_auction_weighted' in predictions and days_to_auction is not None:
            # Calculate a boost factor based on closeness to auction
            # Scale from 1.0 (far from auction) to 5.0 (day before auction)
            if days_to_auction <= 1:
                auction_boost = 5.0  # Day before auction - highest weight
            elif days_to_auction <= 3:
                auction_boost = 3.0  # 2-3 days before auction
            elif days_to_auction <= 7:
                auction_boost = 2.0  # Within a week
            else:
                auction_boost = 1.0  # More than a week away
                
            # Apply the boost to the pre-auction weighted prediction
            weights['pre_auction_weighted'] *= auction_boost
            self.logger.info(f"Applied auction proximity boost: {auction_boost}x to pre-auction weighted prediction (days to auction: {days_to_auction})")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Fallback to equal weights if total_weight is zero
            n_models = len(weights)
            normalized_weights = {k: 1.0/n_models for k in weights.keys()}
        
        # Calculate weighted ensemble prediction
        ensemble_pred = 0.0
        for model, pred in predictions.items():
            if model in normalized_weights and pd.notna(pred):
                ensemble_pred += normalized_weights[model] * pred
                
        return ensemble_pred

    def predict_yield(self, df: pd.DataFrame, bond_type: str, prediction_date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict the yield for a Treasury bond using various methods and dynamic ensemble.

        Args:
            df: DataFrame with historical yield data (requires 'date' and 'yield' columns,
                plus any FRED columns if used by models like LinearRegression/XGBoost).
            bond_type: Bond type (e.g., '10Y').
            prediction_date_str: Date to predict yield for (YYYY-MM-DD format).
                                 If None, predicts the next business day.

        Returns:
            Dictionary with prediction results.
        """
        if df is None or df.empty:
            self.logger.error(f"No data provided for {bond_type} bond")
            return {'error': 'No data available'}

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Rename 'value' to 'yield' if necessary for consistency
        if 'value' in df.columns and 'yield' not in df.columns:
             df = df.rename(columns={'value': 'yield'})
        elif 'yield' not in df.columns:
             self.logger.error("Input DataFrame must have a 'yield' column.")
             return {'error': "Input DataFrame must have a 'yield' column."}

        # Determine prediction date if not provided
        if not prediction_date_str:
            last_date = df['date'].max()
            next_day = last_date + pd.Timedelta(days=1)
            # Simple check for weekend, adjust to Monday
            if next_day.weekday() == 5: # Saturday
                prediction_date_dt = next_day + pd.Timedelta(days=2)
            elif next_day.weekday() == 6: # Sunday
                prediction_date_dt = next_day + pd.Timedelta(days=1)
            else:
                prediction_date_dt = next_day
            prediction_date_str = prediction_date_dt.strftime("%Y-%m-%d")
        else:
            try:
                prediction_date_dt = pd.to_datetime(prediction_date_str)
            except ValueError:
                 self.logger.error(f"Invalid prediction_date_str format: {prediction_date_str}. Use YYYY-MM-DD.")
                 return {'error': f"Invalid prediction_date_str format: {prediction_date_str}. Use YYYY-MM-DD."}

        self.logger.info(f"Predicting {bond_type} yield for {prediction_date_str}")

        # Import datetime properly to avoid attribute error
        from datetime import datetime
        
        # Get next auction date for this bond type
        from src.data_fetcher import DataFetcher
        data_fetcher = DataFetcher(cache_dir=self.cache_dir)
        next_auction_date_str, auction_details = data_fetcher.get_next_auction_date(bond_type)
        
        # Calculate days until next auction if available
        days_to_auction = None
        if next_auction_date_str:
            try:
                next_auction_date = datetime.strptime(next_auction_date_str, '%Y-%m-%d').date()
                prediction_date = prediction_date_dt.date()
                days_to_auction = (next_auction_date - prediction_date).days
                if days_to_auction < 0:
                    # If prediction date is after next auction, set to None
                    days_to_auction = None
            except Exception as e:
                self.logger.warning(f"Error calculating days to auction: {e}")
        
        result = {
            'bond_type': bond_type,
            'prediction_date': prediction_date_str,
            'next_auction_date': next_auction_date_str,
            'days_to_auction': days_to_auction,
            'timestamp': datetime.now().isoformat(),
            'last_known_yield': float(df['yield'].iloc[-1]),
            'last_date': df['date'].iloc[-1].strftime("%Y-%m-%d"),
            'predictions': {},
            'confidence_intervals': {},
            'backtest_metrics_used': {}, # Store metrics used for ensemble weights
            'adjustments': {},
            'ensemble_method': 'none' # Will be updated later
        }

        # --- Load Backtest Metrics (for dynamic ensemble weights) ---
        metrics_file = os.path.join(self.cache_dir, f'metrics_{bond_type}.json')
        current_metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    # Assuming metrics file structure is { 'backtest_metrics': { 'model': {'mae': ...} } }
                    # Adjust path if structure is different (e.g., directly metrics at top level)
                    loaded_data = json.load(f)
                    # Check common structures
                    if 'backtest_metrics' in loaded_data:
                        current_metrics = loaded_data.get('backtest_metrics', {})
                    elif 'metrics' in loaded_data: # Handle structure from backtest_predictions output
                        current_metrics = loaded_data.get('metrics', {})
                    else:
                        current_metrics = loaded_data # Assume top level if keys not found

                    result['backtest_metrics_used'] = current_metrics # Store for reference
                    self.logger.info(f"Loaded backtest metrics from {metrics_file}")
            except Exception as e:
                self.logger.error(f"Error loading metrics file {metrics_file}: {e}")
        else:
            self.logger.warning(f"Metrics file not found: {metrics_file}. Dynamic ensemble will use fallback weights.")

        # --- Anomaly Detection - Filter outliers ---
        # Detect and adjust outlier points that could skew predictions
        df = self._filter_outliers(df, 'yield', z_threshold=3.0)

        # --- Model Fitting ---
        predictions = {}
        series = df.set_index('date')['yield'] # Time series for ARIMA/ES

        # 1. Rolling Averages (use last available value as prediction)
        for window in [5, 10, 20]:
            rolling_avg = df['yield'].rolling(window=window).mean().iloc[-1]
            if pd.notna(rolling_avg):
                predictions[f'rolling_{window}d'] = float(rolling_avg)

        # 2. ARIMA model
        # Use bond-specific ARIMA parameters if desired, or keep default
        arima_order = (7, 1, 2) if bond_type in ['20Y', '30Y'] else (5, 1, 0)
        arima_pred = self._fit_arima_model(series, order=arima_order)
        if arima_pred is not None:
            predictions['arima'] = float(arima_pred)

        # 3. Exponential Smoothing
        exp_pred = self._fit_exponential_smoothing(series)
        if exp_pred is not None:
            predictions['exponential_smoothing'] = float(exp_pred)

        # 4. Linear Regression (using features including FRED data if available)
        # Pass the full DataFrame df which might contain FRED columns
        lr_pred = self._fit_linear_regression(df, forecast_days=1) # Predict 1 day ahead
        if lr_pred is not None:
            predictions['linear_regression'] = float(lr_pred)

        # 5. XGBoost (using features including FRED data if available)
        xgb_pred = self._fit_xgboost(df, forecast_days=1) # Predict 1 day ahead
        if xgb_pred is not None:
            predictions['xgboost'] = float(xgb_pred)
            
        # 6. Add the pre-auction weighted prediction if within 14 days of auction
        if days_to_auction is not None and days_to_auction <= 14:
            pre_auction_pred = self._calculate_pre_auction_weighted_prediction(df, days_to_auction)
            if pre_auction_pred is not None:
                predictions['pre_auction_weighted'] = float(pre_auction_pred)
                # If very close to auction (3 days or less), give this prediction more weight
                if days_to_auction <= 3:
                    self.logger.info(f"Very close to auction ({days_to_auction} days). Pre-auction weighted prediction will be heavily weighted.")

        result['predictions'] = predictions # Store raw model predictions
        
        # Calculate confidence intervals for each prediction
        confidence_intervals = self._calculate_confidence_intervals(df, predictions)
        if confidence_intervals:
            result['confidence_intervals'] = confidence_intervals

        # --- Ensemble Calculation ---
        valid_preds_for_ensemble = {k: v for k, v in predictions.items() if v is not None and pd.notna(v)}

        if not valid_preds_for_ensemble:
            self.logger.error("All models failed to produce a valid prediction.")
            result['error'] = 'All models failed to produce a prediction.'
            return result # Return early if no models worked

        # Use auction-aware ensemble calculation if close to an auction
        if days_to_auction is not None and days_to_auction <= 14 and 'pre_auction_weighted' in predictions:
            # Use an auction-aware ensemble that puts more weight on the pre-auction prediction
            # when close to auction date
            ensemble_prediction = self._calculate_auction_aware_ensemble(
                valid_preds_for_ensemble, 
                days_to_auction,
                current_metrics
            )
            ensemble_method = 'auction_aware_ensemble'
        else:
            # Use regular dynamic ensemble when not near an auction
            ensemble_prediction = self._calculate_dynamic_ensemble(valid_preds_for_ensemble, current_metrics)
            ensemble_method = 'dynamic_inverse_mae'

        if ensemble_prediction is None:
            # Fallback to simple average if dynamic fails or no metrics were loaded
            self.logger.warning("Ensemble calculation failed. Falling back to simple average.")
            ensemble_prediction = sum(valid_preds_for_ensemble.values()) / len(valid_preds_for_ensemble)
            ensemble_method = 'simple_average_fallback'

        result['predictions']['ensemble'] = ensemble_prediction # Store the final ensemble prediction
        result['ensemble_method'] = ensemble_method

        # --- Bias Correction (Adaptive) ---
        final_ensemble_pred = result['predictions']['ensemble'] # Get the potentially updated ensemble value
        if final_ensemble_pred is not None:
            adaptive_bias = self._get_recent_auction_bias(bond_type, lookback=5) # Look at last 5 auctions

            if abs(adaptive_bias) > 1e-6: # Apply if bias is non-negligible
                corrected_pred = final_ensemble_pred - adaptive_bias # Subtract the average error
                self.logger.info(f"Applied adaptive bias correction: {adaptive_bias:.6f}. Prediction changed from {final_ensemble_pred:.4f} to {corrected_pred:.4f}")
                result['predictions']['ensemble'] = corrected_pred # Update the ensemble prediction
                result['adjustments']['adaptive_bias_correction'] = -adaptive_bias # Store the value added/subtracted
            else:
                self.logger.info("No significant adaptive bias detected or applied.")

        # --- Calculate Prediction Confidence ---
        # Provide a confidence score based on model agreement and historical accuracy
        confidence_score = self._calculate_prediction_confidence(
            predictions, 
            current_metrics, 
            days_to_auction
        )
        result['prediction_confidence'] = confidence_score

        return result
    
    def backtest_predictions(self, bond_data: pd.DataFrame, bond_type: str,
                            test_window_days: int = 30) -> Dict[str, Any]:
        """
        Backtest prediction models using historical data by simulating predictions day-by-day.

        Args:
            bond_data: DataFrame with historical yield data (must include 'date' and 'yield' columns).
            bond_type: Bond type (e.g., '10Y').
            test_window_days: Number of days in the past to use for the test set.

        Returns:
            Dictionary with backtest results including metrics for each model.
        """
        if bond_data is None or bond_data.empty:
            self.logger.error(f"Backtest Error ({bond_type}): Input data is empty.")
            return {'error': 'Input data is empty'}

        df = bond_data.copy()
        # Ensure correct columns exist
        if 'date' not in df.columns:
            self.logger.error(f"Backtest Error ({bond_type}): 'date' column missing.")
            return {'error': "'date' column missing."}
        if 'yield' not in df.columns:
            if 'value' in df.columns:
                df = df.rename(columns={'value': 'yield'})
                self.logger.warning(f"Backtest ({bond_type}): Renamed 'value' column to 'yield'.")
            else:
                self.logger.error(f"Backtest Error ({bond_type}): 'yield' or 'value' column missing.")
                return {'error': "'yield' or 'value' column missing."}

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if len(df) < test_window_days + 20: # Need sufficient history + test window
            self.logger.error(f"Backtest Error ({bond_type}): Not enough data ({len(df)} points) for backtesting with window {test_window_days}.")
            return {'error': f'Not enough data for backtesting (need at least {test_window_days + 20})'}

        # Split data
        train_df_initial = df.iloc[:-test_window_days]
        test_df = df.iloc[-test_window_days:]

        actuals = test_df['yield'].tolist()
        dates = test_df['date'].tolist()

        # Store predictions for each model over the test window
        model_predictions = {
            'rolling_5d': [], 'rolling_10d': [], 'rolling_20d': [],
            'arima': [], 'exponential_smoothing': [], 'linear_regression': [],
        }
        
        # Check if XGBoost is available and add it to the models
        try:
            import xgboost
            model_predictions['xgboost'] = []
            self.logger.info(f"Backtest ({bond_type}): XGBoost is available and will be included")
        except (ImportError, ModuleNotFoundError):
            self.logger.warning(f"Backtest ({bond_type}): XGBoost not available. Excluding from models.")
        
        all_models = list(model_predictions.keys())

        # Iterate through the test window day by day
        for i in range(test_window_days):
            try:
                current_train_df = pd.concat([train_df_initial, test_df.iloc[:i]], ignore_index=True)
                current_train_series = current_train_df.set_index('date')['yield']

                # 1. Rolling Averages (based on the last day of current_train_df)
                for window in [5, 10, 20]:
                    if len(current_train_df) >= window:
                        pred = current_train_df['yield'].rolling(window=window).mean().iloc[-1]
                        model_predictions[f'rolling_{window}d'].append(float(pred) if pd.notna(pred) else np.nan)
                    else:
                        model_predictions[f'rolling_{window}d'].append(np.nan)

                # 2. ARIMA - handle exceptions for each model individually
                try:
                    arima_order = (7, 1, 2) if bond_type in ['20Y', '30Y'] else (5, 1, 0)
                    pred_arima = self._fit_arima_model(current_train_series, order=arima_order)
                    model_predictions['arima'].append(float(pred_arima) if pred_arima is not None else np.nan)
                except Exception as e:
                    self.logger.warning(f"Backtest ({bond_type}): ARIMA model failed at step {i}: {str(e)}")
                    model_predictions['arima'].append(np.nan)

                # 3. Exponential Smoothing
                try:
                    pred_es = self._fit_exponential_smoothing(current_train_series)
                    model_predictions['exponential_smoothing'].append(float(pred_es) if pred_es is not None else np.nan)
                except Exception as e:
                    self.logger.warning(f"Backtest ({bond_type}): ES model failed at step {i}: {str(e)}")
                    model_predictions['exponential_smoothing'].append(np.nan)

                # 4. Linear Regression
                try:
                    pred_lr = self._fit_linear_regression(current_train_df, forecast_days=1)
                    model_predictions['linear_regression'].append(float(pred_lr) if pred_lr is not None else np.nan)
                except Exception as e:
                    self.logger.warning(f"Backtest ({bond_type}): Linear Regression failed at step {i}: {str(e)}")
                    model_predictions['linear_regression'].append(np.nan)

                # 5. XGBoost - only if available
                if 'xgboost' in model_predictions:
                    try:
                        pred_xgb = self._fit_xgboost(current_train_df, forecast_days=1)
                        model_predictions['xgboost'].append(float(pred_xgb) if pred_xgb is not None else np.nan)
                    except Exception as e:
                        self.logger.warning(f"Backtest ({bond_type}): XGBoost model failed at step {i}: {str(e)}")
                        model_predictions['xgboost'].append(np.nan)
            
            except Exception as e:
                self.logger.error(f"Backtest ({bond_type}): Error during iteration {i}: {str(e)}\n{traceback.format_exc()}")
                # Fill with NaN for this iteration for all models
                for model in all_models:
                    model_predictions[model].append(np.nan)

        # Calculate metrics for each model
        try:
            # Use datetime directly from imported module to avoid attribute errors
            from datetime import datetime
            timestamp_str = datetime.now().isoformat()
            
            results = {
                'bond_type': bond_type,
                'backtest_window_days': test_window_days,
                'timestamp': timestamp_str,
                'metrics': {}
            }
            actuals_np = np.array(actuals)

            for model_name in all_models:
                if model_name not in model_predictions or not model_predictions[model_name]:
                    self.logger.warning(f"Backtest ({bond_type}): No predictions available for {model_name}")
                    continue
                    
                preds_np = np.array(model_predictions[model_name])
                if len(preds_np) == len(actuals_np):
                    metrics = self._calculate_metrics(actuals_np, preds_np)
                    if metrics: # Only add if metrics could be calculated
                        results['metrics'][model_name] = metrics
                else:
                    self.logger.warning(f"Backtest ({bond_type}): Length mismatch for {model_name} predictions ({len(preds_np)}) vs actuals ({len(actuals_np)}). Skipping metrics.")

            # Calculate Ensemble Metrics (Simple Average for backtesting consistency)
            ensemble_preds = []
            for i in range(test_window_days):
                step_preds = [model_predictions[m][i] for m in all_models 
                              if i < len(model_predictions[m]) 
                              and model_predictions[m] 
                              and i < len(model_predictions[m]) 
                              and pd.notna(model_predictions[m][i])]
                if step_preds:
                    ensemble_preds.append(sum(step_preds) / len(step_preds))
                else:
                    ensemble_preds.append(np.nan)

            ensemble_preds_np = np.array(ensemble_preds)
            if len(ensemble_preds_np) == len(actuals_np):
                ensemble_metrics = self._calculate_metrics(actuals_np, ensemble_preds_np)
                if ensemble_metrics:
                    results['metrics']['ensemble_simple_avg'] = ensemble_metrics
            else:
                self.logger.warning(f"Backtest ({bond_type}): Length mismatch for ensemble predictions ({len(ensemble_preds_np)}) vs actuals ({len(actuals_np)}). Skipping metrics.")

            # Save backtest metrics results
            metrics_file = os.path.join(self.cache_dir, f"metrics_{bond_type}.json")
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Saved {bond_type} backtest metrics to {metrics_file}")
            except Exception as e:
                self.logger.error(f"Error saving backtest metrics to file {metrics_file}: {e}\n{traceback.format_exc()}")

            return results
            
        except Exception as e:
            self.logger.error(f"Backtest ({bond_type}): Fatal error during metrics calculation: {str(e)}\n{traceback.format_exc()}")
            return {
                'bond_type': bond_type,
                'error': f'Error calculating metrics: {str(e)}',
                'backtest_window_days': test_window_days
            }
    
    def visualize_predictions(self, bond_data: pd.DataFrame, bond_type: str, 
                             prediction_results: Dict[str, Any], output_dir: str = "../data/plots"):
        """
        Create visualizations of predictions and historical data.
        
        Args:
            bond_data: DataFrame with historical yield data
            bond_type: Bond type (e.g., '10Y')
            prediction_results: Dictionary of prediction results
            output_dir: Directory to save plots
        """
        if bond_data.empty:
            self.logger.error(f"No data available for {bond_type} visualization")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        df = bond_data.copy().sort_values('date')
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot historical data
            plt.plot(df['date'], df['value'], label='Historical Yield', color='blue')
            
            # Add predictions
            last_date = df['date'].max()
            prediction_date = pd.to_datetime(prediction_results.get('prediction_date', None))
            
            if prediction_date is not None and 'predictions' in prediction_results:
                # Create a point for each model's prediction
                for model, value in prediction_results['predictions'].items():
                    if pd.notna(value):
                        plt.scatter(prediction_date, value, label=f'{model} prediction', s=50)
                
                # Connect last known point to ensemble prediction with dashed line
                if 'ensemble' in prediction_results['predictions']:
                    ensemble_value = prediction_results['predictions']['ensemble']
                    if pd.notna(ensemble_value):
                        plt.plot(
                            [last_date, prediction_date], 
                            [df['value'].iloc[-1], ensemble_value], 
                            'k--', label='Ensemble prediction'
                        )
            
            # Add labels and title
            plt.title(f'{bond_type} Treasury Bond Yield - Historical Data and Predictions')
            plt.xlabel('Date')
            plt.ylabel('Yield (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save plot
            plot_file = os.path.join(output_dir, f"{bond_type}_prediction.png")
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Saved visualization to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")

    def _filter_outliers(self, df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Filter outliers from a dataframe column using z-score.
        
        Args:
            df: DataFrame containing data
            column: Column name to filter outliers from
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers replaced with interpolated values
        """
        result_df = df.copy()
        
        if column not in result_df.columns:
            self.logger.warning(f"Column {column} not found in dataframe, cannot filter outliers")
            return result_df
        
        # Calculate z-scores
        values = result_df[column].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:  # Avoid division by zero
            return result_df
            
        z_scores = np.abs((values - mean_val) / std_val)
        
        # Identify outliers
        outlier_indices = np.where(z_scores > z_threshold)[0]
        
        if len(outlier_indices) > 0:
            # Replace outliers with interpolated values
            result_df.loc[result_df.index[outlier_indices], column] = np.nan
            
            # Interpolate missing values
            result_df[column] = result_df[column].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Identified and replaced {len(outlier_indices)} outliers in {column}")
            
        return result_df

    def _calculate_confidence_intervals(self, df: pd.DataFrame, predictions: Dict[str, float], 
                                      confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for predictions based on historical volatility.
        
        Args:
            df: DataFrame with historical yield data
            predictions: Dictionary of model predictions
            confidence_level: Confidence level (default: 95%)
            
        Returns:
            Dictionary of confidence intervals for each prediction
        """
        from scipy import stats
        
        confidence_intervals = {}
        
        if df.empty or 'yield' not in df.columns:
            return confidence_intervals
            
        try:
            # Calculate historical volatility (standard deviation of daily changes)
            daily_changes = df['yield'].diff().dropna()
            
            if len(daily_changes) < 10:  # Need enough data points
                return confidence_intervals
                
            # Use standard error of the historical changes
            std_dev = daily_changes.std()
            
            # Calculate z-score for the desired confidence level
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Calculate margin of error
            margin_error = z_score * std_dev
            
            # Calculate confidence intervals for each model prediction
            for model, pred in predictions.items():
                if pd.isna(pred):
                    continue
                    
                lower_bound = pred - margin_error
                upper_bound = pred + margin_error
                
                confidence_intervals[model] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'margin_error': float(margin_error)
                }
                
            return confidence_intervals
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {e}")
            return {}

    def _calculate_prediction_confidence(self, predictions: Dict[str, float], 
                                       metrics: Dict[str, Dict[str, float]],
                                       days_to_auction: Optional[int] = None) -> float:
        """
        Calculate a confidence score for the prediction based on model agreement and performance.
        
        Args:
            predictions: Dictionary of model predictions
            metrics: Dictionary of model performance metrics
            days_to_auction: Number of days until next auction
            
        Returns:
            Confidence score between 0 and 1
        """
        if not predictions or len(predictions) < 2:
            return 0.5  # Default medium confidence with insufficient data
            
        try:
            # Filter out NaN predictions
            valid_preds = {k: v for k, v in predictions.items() 
                          if v is not None and pd.notna(v) and k != 'ensemble'}
            
            if not valid_preds or len(valid_preds) < 2:
                return 0.5
                
            # 1. Model agreement factor (how close predictions are to each other)
            values = list(valid_preds.values())
            mean_pred = np.mean(values)
            relative_spread = np.std(values) / mean_pred if mean_pred != 0 else 1.0
            
            # Convert spread to agreement score (lower spread -> higher agreement)
            # Cap the relative spread at 0.2 (20% of mean)
            # A spread of 0 -> perfect agreement -> score of 1
            # A spread of 0.2 or more -> poor agreement -> score of 0
            agreement_score = max(0, 1 - (relative_spread / 0.2))
            
            # 2. Historical accuracy factor (based on backtest metrics)
            accuracy_scores = []
            for model in valid_preds:
                if model in metrics and 'mae' in metrics[model]:
                    # Convert MAE to a score where lower MAE -> higher score
                    # Normalize against a "good" MAE of 0.1 and a "bad" MAE of 0.5
                    mae = metrics[model]['mae']
                    model_score = max(0, 1 - (mae / 0.5))
                    accuracy_scores.append(model_score)
            
            accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 0.5
            
            # 3. Auction proximity factor
            # Higher confidence when closer to auction (if date is known)
            auction_factor = 1.0
            if days_to_auction is not None:
                if days_to_auction <= 3:  # Very close to auction
                    auction_factor = 1.2  # Boost confidence
                elif days_to_auction <= 7:  # Within a week
                    auction_factor = 1.1  # Slight boost
            
            # Combine factors (weighted average with optional auction boost)
            # 60% weight on model agreement, 40% on historical accuracy
            base_confidence = (0.6 * agreement_score) + (0.4 * accuracy_score)
            
            # Apply auction factor (limit to 0-1 range)
            confidence = min(1.0, base_confidence * auction_factor)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5  # Default medium confidence on error

    def _calculate_auction_proximity_boost(self, days_to_auction: Optional[int]) -> float:
        """
        Calculate boost factor based on proximity to auction.
        
        Args:
            days_to_auction: Number of days until the next auction
            
        Returns:
            Boost factor (multiplier)
        """
        if days_to_auction is None or days_to_auction > 14:
            return 1.0  # No boost if no auction or too far away
            
        # Much more aggressive boosting for imminent auctions
        # For auctions less than 5 days away, apply exponential boosting
        if days_to_auction <= 5:
            # Exponential increase as auction approaches
            # 5 days: 2.5x boost
            # 4 days: 3.5x boost
            # 3 days: 5.0x boost
            # 2 days: 7.0x boost
            # 1 day:  10.0x boost
            # Auction day: 15.0x boost
            return 15.0 * (1.0 - 0.35 * days_to_auction)
        elif days_to_auction <= 10:
            # Linear decrease from 2.0 to 1.0 as days increase from 5 to 14
            return 2.5 - ((days_to_auction - 5) * 0.2)
        else:  # 11-14 days
            # Minor boost for auctions 11-14 days away
            return 1.2 - ((days_to_auction - 10) * 0.05)
            
    def _calculate_auction_aware_ensemble(self, 
                                        predictions: Dict[str, float],
                                        days_to_auction: Optional[int],
                                        current_metrics: Dict[str, Dict[str, float]] = None) -> float:
        """
        Calculate ensemble prediction with auction awareness.
        
        Args:
            predictions: Dictionary of model name to prediction
            days_to_auction: Number of days until the next auction
            current_metrics: Dictionary of backtest metrics for each model
            
        Returns:
            Weighted ensemble prediction
        """
        # Default weights favor balanced approach
        weights = {
            'rolling_5d': 0.15,
            'rolling_10d': 0.10,
            'rolling_20d': 0.10,
            'arima': 0.25,
            'exponential_smoothing': 0.25,
            'linear_regression': 0.05,
            'xgboost': 0.10,
        }
        
        # Adjust weights based on auction proximity
        if days_to_auction is not None:
            if days_to_auction <= 4:
                # EXTREMELY aggressive for auctions within 4 days
                # For imminent auctions, make dramatic adjustments to yield higher predictions
                if days_to_auction <= 2:
                    # Extreme shift for auctions within 48 hours
                    # Almost exclusively favor most recent trends and market expectations
                    weights = {
                        'rolling_5d': 0.50,      # Recent movements dominate
                        'rolling_10d': 0.00,
                        'rolling_20d': 0.00,
                        'arima': 0.10,           # Minimal influence from models
                        'exponential_smoothing': 0.05,
                        'linear_regression': 0.00,
                        'xgboost': 0.00,
                        'pre_auction_weighted': 1.5    # Pre-auction model gets extremely aggressive weight
                    }
                    self.logger.info(f"Using extremely aggressive auction weights (days to auction: {days_to_auction})")
                else:  # 3-4 days
                    # Very strong adjustments for auctions 3-4 days away
                    weights = {
                        'rolling_5d': 0.40,
                        'rolling_10d': 0.05,
                        'rolling_20d': 0.00,
                        'arima': 0.15,
                        'exponential_smoothing': 0.10,
                        'linear_regression': 0.00,
                        'xgboost': 0.00,
                        'pre_auction_weighted': 1.2   # Pre-auction model gets aggressive weight
                    }
                    self.logger.info(f"Using very aggressive auction weights (days to auction: {days_to_auction})")
            elif days_to_auction <= 7:
                # Medium-high proximity: emphasize recent trends more
                weights = {
                    'rolling_5d': 0.30,
                    'rolling_10d': 0.10,
                    'rolling_20d': 0.05,
                    'arima': 0.25,
                    'exponential_smoothing': 0.15,
                    'linear_regression': 0.05,
                    'xgboost': 0.10,
                    'pre_auction_weighted': 0.70   # Significant weight but not dominant
                }
            elif days_to_auction <= 14:
                # Further out: more balanced approach with moderate pre-auction influence
                weights = {
                    'rolling_5d': 0.20,
                    'rolling_10d': 0.15,
                    'rolling_20d': 0.10,
                    'arima': 0.25,
                    'exponential_smoothing': 0.20,
                    'linear_regression': 0.05,
                    'xgboost': 0.05,
                    'pre_auction_weighted': 0.40   # Moderate weight
                }
                
        # Filter predictions to only include models we have
        available_models = {k: v for k, v in predictions.items() if k in weights}
        available_weights = {k: weights[k] for k in available_models.keys()}
        
        # Normalize weights
        weight_sum = sum(available_weights.values())
        if weight_sum > 0:
            normalized_weights = {k: v/weight_sum for k, v in available_weights.items()}
        else:
            # Fallback to equal weights if no valid weights
            normalized_weights = {k: 1.0/len(available_models) for k in available_models.keys()}
        
        # Calculate weighted average
        weighted_sum = sum(predictions[model] * normalized_weights[model] 
                          for model in available_models)
        
        return weighted_sum
        
    def _calculate_pre_auction_weighted_prediction(self, df: pd.DataFrame, days_to_auction: int) -> float:
        """
        Calculate prediction with pre-auction weighting that prioritizes recent movements.
        This method is specifically designed for the period immediately before auctions.
        
        Args:
            df: DataFrame with historical yield data (includes 'date' and 'yield' columns)
            days_to_auction: Number of days until the next auction
            
        Returns:
            Pre-auction weighted prediction
        """
        if days_to_auction is None or days_to_auction > 14:
            return None  # Only activate for auctions within 14 days
        
        if df.empty or 'yield' not in df.columns:
            self.logger.error("Cannot calculate pre-auction prediction: DataFrame is empty or missing 'yield' column")
            return None
            
        # Ensure we're working with a properly sorted DataFrame
        df = df.sort_values('date')
        
        # Get increasingly recent data segments
        last_5d = df['yield'].iloc[-5:].values if len(df) >= 5 else None
        last_3d = df['yield'].iloc[-3:].values if len(df) >= 3 else None
        last_2d = df['yield'].iloc[-2:].values if len(df) >= 2 else None
        last_1d = df['yield'].iloc[-1:].values if len(df) >= 1 else None
        
        # Base calculation on trend directionality
        if last_5d is not None and len(last_5d) >= 5:
            # Calculate day-over-day changes
            changes_5d = np.diff(last_5d)
            changes_3d = np.diff(last_3d)
            changes_2d = np.diff(last_2d)
            
            # Determine if there's a consistent trend in the last days
            trend_5d = np.mean(changes_5d)
            trend_3d = np.mean(changes_3d)
            trend_2d = np.mean(changes_2d)
            
            # For auctions ≤4 days away, amplify recent trends dramatically
            # The closer to auction, the more we emphasize very recent movements
            if days_to_auction <= 2:
                # Extremely aggressive for 1-2 days before auction
                weight_5d = 0.1   # Minimal weight to older data
                weight_3d = 0.2
                weight_2d = 0.3
                weight_1d = 0.4   # Last day gets highest weight
                
                # Calculate amplification factor - much higher for imminent auctions
                # Typical pre-auction movements are 3-5 times larger than normal
                if days_to_auction == 1:
                    amplification = 6.0  # Day before auction
                else:
                    amplification = 4.5  # Two days before
                
                # Weighted average of trends with amplification
                projected_change = ((trend_5d * weight_5d) + 
                                   (trend_3d * weight_3d) + 
                                   (trend_2d * weight_2d)) * amplification
                                   
                # Apply a strong directional bias based on the most recent day's movement
                # This captures market sentiment right before the auction
                last_day_change = 0
                if len(last_2d) >= 2:
                    last_day_change = last_2d[1] - last_2d[0]
                    
                if abs(last_day_change) > 0.01:  # If meaningful movement on last day
                    # Give even more weight to the last day's direction for imminent auctions
                    projected_change = (projected_change * (1 - weight_1d)) + (last_day_change * weight_1d * amplification)
                
            elif days_to_auction <= 4:
                # Very aggressive for 3-4 days away
                weight_5d = 0.20
                weight_3d = 0.30
                weight_2d = 0.30
                weight_1d = 0.20
                
                # Calculate amplification factor
                amplification = 3.0
                
                # Weighted average of trends
                projected_change = ((trend_5d * weight_5d) + 
                                   (trend_3d * weight_3d) + 
                                   (trend_2d * weight_2d)) * amplification
                
                # Apply directional bias from last day
                if len(last_2d) >= 2:
                    last_day_change = last_2d[1] - last_2d[0]
                    if abs(last_day_change) > 0.01:  # If meaningful movement
                        projected_change = (projected_change * (1 - weight_1d)) + (last_day_change * weight_1d * amplification)
                
            else:  # 5-14 days
                # Moderate pre-auction adjustment
                weight_5d = 0.4
                weight_3d = 0.3
                weight_2d = 0.3
                
                # Amplification decreases as we get further from auction
                amplification = max(1.0, 2.0 - ((days_to_auction - 5) * 0.1))
                
                # Weighted average of trends
                projected_change = ((trend_5d * weight_5d) + 
                                   (trend_3d * weight_3d) + 
                                   (trend_2d * weight_2d)) * amplification
            
            # Apply projected change to last value
            last_value = df['yield'].iloc[-1]
            prediction = last_value + projected_change
            
            self.logger.info(f"Pre-auction weighted prediction: {prediction:.4f} (last value: {last_value:.4f}, projected change: {projected_change:.4f})")
            
            return prediction
            
        # Fallback if not enough data
        return df['yield'].iloc[-1]

    def predict_auction_yield(self, bond_type: str, prediction_date: Optional[str] = None,
                            historical_mode: bool = False) -> Dict[str, Any]:
        """
        Predict the yield for the next auction of the given bond type.
        
        Args:
            bond_type: Bond type to predict (e.g., '10Y')
            prediction_date: Date for prediction (default: today)
            historical_mode: If True, don't include auction data
            
        Returns:
            Dictionary with prediction results
        """
        if not prediction_date:
            prediction_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Predicting yield for {bond_type} bond as of {prediction_date}")
        
        # Get historical bond data
        bond_data = self.data_fetcher.get_treasury_data(bond_type)
        if bond_data.empty:
            logger.error(f"No historical data available for {bond_type}")
            return {'success': False, 'error': 'No historical data available'}
            
        # Filter data to only use what would be available on prediction date
        pred_date = datetime.strptime(prediction_date, '%Y-%m-%d').date()
        bond_data = bond_data[bond_data['date'].dt.date <= pred_date]
        
        if len(bond_data) < 30:  # Need sufficient history
            logger.warning(f"Insufficient historical data for {bond_type} (only {len(bond_data)} points)")
            return {'success': False, 'error': 'Insufficient historical data'}
            
        # Get most recent yield
        latest_yield = bond_data['value'].iloc[-1]
        latest_date = bond_data['date'].iloc[-1]
        
        # Calculate yield trend over past month
        month_ago = pred_date - timedelta(days=30)
        month_data = bond_data[bond_data['date'].dt.date >= month_ago]
        
        if len(month_data) > 5:  # Need sufficient recent data
            first_yield = month_data['value'].iloc[0]
            last_yield = month_data['value'].iloc[-1]
            monthly_change = last_yield - first_yield
        else:
            monthly_change = 0
            
        # Calculate volatility (standard deviation over past month)
        if len(month_data) > 5:
            volatility = month_data['value'].std()
        else:
            volatility = bond_data['value'].std() * 0.5  # Reduced weight
        
        # Apply the prediction model logic here
        # In a production system, this would be a real model (regression, ML, etc.)
        # For now, use a simple heuristic based on recent trends
        
        # Basic model - current yield + trend adjustment + volatility factor
        trend_factor = monthly_change * 0.5  # Assume trend continues but dampened
        random_factor = np.random.normal(0, volatility * 0.3)  # Add some randomness based on volatility
        
        predicted_yield = latest_yield + trend_factor + random_factor
        
        # Create prediction result
        prediction = {
            'bond_type': bond_type,
            'prediction_date': prediction_date,
            'predicted_yield': round(predicted_yield, 3),
            'current_yield': round(latest_yield, 3),
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'monthly_change': round(monthly_change, 3),
            'volatility': round(volatility, 3),
            'trend_factor': round(trend_factor, 3),
            'success': True
        }
        
        # Add auction data if not in historical mode
        if not historical_mode:
            prediction = self.data_fetcher.add_auction_data_to_prediction(prediction, bond_type)
        
        return prediction

    def evaluate_prediction_accuracy(self, bond_type: str, num_auctions: int = 10) -> Dict[str, Any]:
        """
        Evaluate the accuracy of the prediction model on historical data.
        
        Args:
            bond_type: Bond type to evaluate
            num_auctions: Number of past auctions to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating prediction accuracy for {bond_type} with {num_auctions} auctions")
        
        # Use the data fetcher's backtest functionality
        backtest_results = self.data_fetcher.backtest_yield_predictions(bond_type, num_auctions)
        
        # Add interpretation of results
        if backtest_results.get('success'):
            metrics = backtest_results.get('metrics', {})
            mae = metrics.get('mean_absolute_error', 0)
            
            # Interpret the accuracy
            if mae < 0.1:
                interpretation = "Excellent accuracy (MAE < 0.1%)"
            elif mae < 0.2:
                interpretation = "Good accuracy (MAE < 0.2%)"
            elif mae < 0.3:
                interpretation = "Moderate accuracy (MAE < 0.3%)"
            else:
                interpretation = "Poor accuracy (MAE >= 0.3%)"
                
            backtest_results['interpretation'] = interpretation
            
        return backtest_results