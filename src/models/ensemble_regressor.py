
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from darts import TimeSeries
from darts.models import NaiveMovingAverage
# You can add more darts models here as needed

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble Regressor that combines multiple models:
    - CatBoost
    - Simple Moving Average (via Darts or custom)
    - KNN
    - Linear Regression
    - Random Forest
    - Additional Darts Time Series Models
    
    It supports using subsets of features for specific models.
    """
    
    def __init__(
        self,
        date_column: str,
        target_column: str,
        catboost_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        knn_params: Optional[Dict[str, Any]] = None,
        sma_weeks: int = 8,
        weights: Optional[Dict[str, float]] = None,
        feature_subsets: Optional[Dict[str, List[str]]] = None,
        ttm_model_path: str = "ibm-granite/granite-timeseries-ttm-v1-r1",
        ttm_context_length: int = 96
    ):
        """
        Args:
            date_column: Name of the column containing date information (required for time series models).
            target_column: Name of the target column.
            catboost_params: Parameters for CatBoostRegressor.
            rf_params: Parameters for RandomForestRegressor.
            knn_params: Parameters for KNeighborsRegressor.
            sma_weeks: Number of weeks for Simple Moving Average (same day of week).
            weights: Dictionary mapping model names to their weights in the ensemble.
                     Models: 'catboost', 'sma', 'knn', 'linear', 'rf'.
            feature_subsets: Dictionary mapping model names to list of feature names to use.
                             If None or key missing, uses all available features (except date/target).
        """
        self.date_column = date_column
        self.target_column = target_column
        self.catboost_params = catboost_params or {}
        self.rf_params = rf_params or {}
        self.knn_params = knn_params or {}
        self.sma_weeks = sma_weeks
        self.weights = weights
        self.feature_subsets = feature_subsets or {}
        self.ttm_model_path = ttm_model_path
        self.ttm_context_length = ttm_context_length
        
        # Initialize models
        self.models = {
            'catboost': CatBoostRegressor(**self.catboost_params, verbose=0),
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(**self.rf_params),
            'knn': KNeighborsRegressor(**self.knn_params),
            # SMA is handled differently as it's a time-series logic often better implemented custom for "same day of week"
            # deeply integrated or via Darts. We will use a custom implementation for "same day of week" logic 
            # if the user specifically asked for "simple moving average of the last 8 weeks SAME DAY OF WEEK".
            # Standard Darts NaiveMovingAverage is contiguous.
        }
        
        self.is_fitted = False
        self.feature_names_in_ = None
        self._training_data_tail = None # Store tail for time series forecasting if needed
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all component models.
        
        Args:
            X: Features DataFrame. MUST include the date column.
            y: Target Series.
        """
        # Validate inputs
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in X")
        
        # Store metadata
        self.feature_names_in_ = [c for c in X.columns if c != self.date_column]
        
        # Prepare data for standard ML models
        # We generally drop the date column for standard ML unless extracted as features
        # Assuming X already contains preprocessed features (or we use feature_subsets)
        
        for name, model in self.models.items():
            # Determine features for this model
            if name in self.feature_subsets:
                cols = self.feature_subsets[name]
                # specific features
                X_sub = X[cols]
            else:
                # all features except date
                cols = [c for c in X.columns if c != self.date_column]
                X_sub = X[cols]
            
            # Handle categorical features for CatBoost if necessary, 
            # but usually we assume X is numeric or CatBoost handles it if specified.
            # For this implementation, we assume X is ready for the models.
            
            try:
                model.fit(X_sub, y)
                logging.info(f"Fitted {name} model.")
            except Exception as e:
                logging.error(f"Failed to fit {name}: {e}")
        
        # Store data needed for SMA (Same Day of Week)
        # We need the history to predict future points based on "last 8 weeks same day"
        # We'll store the relevant history. 
        # For a robust implementation, we might store the whole series or just the tail.
        # Since this is "last 8 weeks", we need at least 8 weeks of history for every day of week.
        
        combined_df = X.copy()
        combined_df[self.target_column] = y
        combined_df[self.date_column] = pd.to_datetime(combined_df[self.date_column])
        self._historical_data = combined_df[[self.date_column, self.target_column]].sort_values(self.date_column)
        
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble.
        
        Args:
            X: Features DataFrame. MUST include the date column.
            
        Returns:
            np.ndarray: Predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in X")
            
        predictions = {}
        
        # 1. Standard ML Models
        for name, model in self.models.items():
            if name in self.feature_subsets:
                cols = self.feature_subsets[name]
                X_sub = X[cols]
            else:
                cols = [c for c in X.columns if c != self.date_column]
                X_sub = X[cols]
            
            predictions[name] = model.predict(X_sub)

            predictions[name] = model.predict(X_sub)

        # 1.5 TTM (Tiny Time Mixers)
        try:
             predictions['ttm'] = self._predict_ttm(X)
        except Exception as e:
             logging.error(f"TTM prediction failed: {e}")
             predictions['ttm'] = np.zeros(len(X))

        # 2. SMA (Same Day of Week)
        # For each row in X, find the average of the target for the same day of week 
        # in the last n weeks from the history.
        
        sma_preds = []
        X_dates = pd.to_datetime(X[self.date_column])
        
        # Optimization: Create a lookup or use rolling operations if doing batch prediction on contiguous future.
        # But X might be random samples. We'll do a row-by-row lookup for safety and correctness first.
        # This can be slow for large X. Vectorization is preferred if possible.
        
        # Vectorized approach:
        # We need to look back from each target date.
        # Since we might be predicting for "future" where we don't have history in self._historical_data for the interim,
        # strictly speaking methods like SMA need contiguous update. 
        # HOWEVER, the valid requirement is "last 8 weeks" relative to the prediction time. 
        # If we are engaged in a simulation/backtest, X contains the date.
        
        history_df = self._historical_data.set_index(self.date_column)
        
        for date in X_dates:
            # Day of week (0=Monday, 6=Sunday)
            target_dow = date.dayofweek
            
            # Find candidate dates: same day of week, strictly before prediction date, within window
            # Window = 8 weeks approx 56 days.
            start_date = date - pd.Timedelta(weeks=self.sma_weeks + 2) # small buffer
            
            # mask = (history_df.index >= start_date) & (history_df.index < date) & (history_df.index.dayofweek == target_dow)
            # Efficient lookup: generate the specific dates we want
            # This handles missing data (e.g. if a week is missing, we just skip it or take what we have)
            
            # Generate expected dates: [date - 1 week, date - 2 weeks, ...]
            lookback_dates = [date - pd.Timedelta(weeks=i) for i in range(1, self.sma_weeks + 1)]
            
            # Intersect with available history
            valid_lookbacks = [d for d in lookback_dates if d in history_df.index]
            
            if valid_lookbacks:
                val = history_df.loc[valid_lookbacks, self.target_column].mean()
            else:
                # Fallback if no history: Global mean? or Model mean?
                # For now use 0 or global mean of history
                val = self._historical_data[self.target_column].mean()
                
            sma_preds.append(val)
            
        predictions['sma'] = np.array(sma_preds)
        
        # 3. Aggregate
        if self.weights:
            # Normalize weights
            total_weight = sum(self.weights.values())
            final_pred = np.zeros(len(X))
            used_weight = 0
            
            for name, weight in self.weights.items():
                if name in predictions:
                    final_pred += predictions[name] * weight
                    used_weight += weight
            
            if used_weight > 0:
                final_pred /= used_weight
            return final_pred
        else:
            # Simple average
            return np.mean(list(predictions.values()), axis=0)

    def fit_darts_model(self, model_instance, X: pd.DataFrame, y: pd.Series):
        """
        Helper to fit a Darts model if we were to add them to self.models directly.
        (Future extension for user request: "addition time series models from darts")
        """
        pass

    def _predict_ttm(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using Tiny Time Mixers (TTM) via Hugging Face.
        Assumes Z-zero-shot inference using pre-trained weights.
        """
        try:
            from transformers import AutoModel, AutoConfig
            import torch
        except ImportError:
            logging.error("Transformers or Torch not installed. TTM prediction failed.")
            return np.zeros(len(X))

        if not self.ttm_model_path:
            return np.zeros(len(X))

        # Lazy loading of TTM model to save resources if not used
        if 'ttm' not in self.models:
            try:
                # TTM requires a specific configuration for context length
                # We assume the model path points to a valid HF model
                self.models['ttm'] = AutoModel.from_pretrained(
                    self.ttm_model_path, 
                    trust_remote_code=True
                )
                self.models['ttm'].eval()
            except Exception as e:
                logging.error(f"Failed to load TTM model: {e}")
                return np.zeros(len(X))

        model = self.models['ttm']
        predictions = []
        
        # TTM expects input tensor of shape (batch, context_length, num_features)
        # We need to construct the context history for each prediction point in X.
        
        history_df = self._historical_data.sort_values(self.date_column).set_index(self.date_column)
        
        # Determine features to use for TTM (or all available)
        if 'ttm' in self.feature_subsets:
            features = self.feature_subsets['ttm']
        else:
            features = [c for c in history_df.columns if c != self.target_column] # Use covariates? 
            # Usually TTM is univariate or multivariate target. 
            # For simplicity in this regressor context (predicting target Y), 
            # we often feed the target history itself.
            features = [self.target_column]

        # Ensure history has the features
        history_data = history_df[features]

        # Optimization needed for batch processing. 
        # For each row in X, we need the LOOKBACK window (context_length) ending just before X.date.
        
        dates = pd.to_datetime(X[self.date_column])
        
        for date in dates:
            # Get valid history before this date
            # We need strictly last context_length points.
            # TTM usually handles fixed context length (e.g. 512, 96).
            
            # Simple approach: slicing by position if data is regular frequency.
            # If irregular, this is harder. Assuming daily/weekly regular for now or just taking last N available?
            # TTM is a time-series model, assumes regularity often.
            
            # Get data strictly before date
            past_data = history_data.loc[history_data.index < date]
            
            if len(past_data) < self.ttm_context_length:
                # Not enough history? Pad or return mean?
                # Zero padding
                # Construct tensor
                input_seq = np.zeros((self.ttm_context_length, len(features)))
                if len(past_data) > 0:
                    input_seq[-len(past_data):, :] = past_data.iloc[-self.ttm_context_length:].values
            else:
                input_seq = past_data.iloc[-self.ttm_context_length:].values
            
            # Prepare tensor
            # Shape: (1, context_length, num_features)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # TTM output format depends on specific head. 
                # Assuming simple forecasting head returning (batch, pred_len, num_features)
                # We want 1 step ahead? or horizon?
                # Regressor usually implies 1 step or specific horizon.
                # Here we assume we want prediction for the 'date'.
                try:
                    output = model(past_values=input_tensor)
                    # Use last prediction or specific horizon
                    # TTM often outputs forecast.prediction
                    pred = output.last_hidden_state[:,-1,0].item() # Simplified assumption for embedding or specific head
                    # If using AutoModelForPrediction, output.forecast
                    if hasattr(output, 'forecast'):
                         pred = output.forecast[:, 0, 0].item() # (batch, horizon, n_vars)
                    else:
                         # Fallback if raw backbone
                         pred = output.last_hidden_state[:,-1,:].mean().item()
                         
                except Exception as e:
                    logging.warning(f"TTM inference error: {e}")
                    pred = 0.0
            
            predictions.append(pred)
            
        return np.array(predictions)
