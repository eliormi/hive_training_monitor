
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
from unittest.mock import MagicMock

# Mock catboost before import
class MockCatBoostRegressor:
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.ones(len(X)) * 0.5 # Dummy prediction

mock_cb = MagicMock()
mock_cb.CatBoostRegressor = MockCatBoostRegressor
sys.modules['catboost'] = mock_cb

# Mock transformers
class MockAutoModel:
    def from_pretrained(self, *args, **kwargs):
        mock_model = MagicMock()
        mock_output = MagicMock()
        # Mock last_hidden_state (batch, seq, hidden) -> (1, 96, 1) or similar
        mock_output.last_hidden_state = torch.ones(1, 96, 1) 
        mock_output.forecast = torch.ones(1, 96, 1) * 0.5 
        mock_model.return_value = mock_output
        mock_model.eval.return_value = None
        return mock_model

mock_transformers = MagicMock()
mock_transformers.AutoModel = MockAutoModel()
sys.modules['transformers'] = mock_transformers

import torch # Should be available via requirements, but if not we might need to mock it too if environment is strict.
# Assuming torch is installed now.

from src.models.ensemble_regressor import EnsembleRegressor

class TestEnsembleRegressor(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='W-MON')
        self.df = pd.DataFrame({
            'date': dates,
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        
        # Add some seasonality to target for SMA testing
        self.df['target'] = self.df['feature1'] * 2 + self.df['feature2'] + np.sin(np.arange(100))
        
    def test_fit_predict(self):
        model = EnsembleRegressor(
            date_column='date',
            target_column='target',
            sma_weeks=8
        )
        
        # Fit
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        model.fit(X, y)
        
        # Predict
        preds = model.predict(X)
        
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.isfinite(preds).all())
        
    def test_feature_subsets(self):
        model = EnsembleRegressor(
            date_column='date',
            target_column='target',
            feature_subsets={
                'catboost': ['feature1'],
                'rf': ['feature2']
            }
        )
        
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_weights(self):
        model = EnsembleRegressor(
            date_column='date',
            target_column='target',
            weights={'catboost': 0.8, 'sma': 0.2}
        )
        
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_with_ttm(self):
        model = EnsembleRegressor(
            date_column='date',
            target_column='target',
            ttm_model_path='mock/path'
        )
        
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        model.fit(X, y)
        
        # Test predict calls TTM
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        
        # Check if ttm model was loaded (we can check internal dictionary)
        self.assertIn('ttm', model.models)

if __name__ == '__main__':
    unittest.main()
