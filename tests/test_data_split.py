"""
NOTE: This test file is skipped. It was written for DataSplitter (src.data_split),
which belongs to a separate project and does not exist in hive_training_monitor.
If DataSplitter is ported into this repo, remove the pytest.skip below.
"""
import pytest
pytest.skip("src.data_split not present in this repo", allow_module_level=True)

"""
Tests for DataSplitter.split_data_by_time method.

These tests verify:
1. No voy_id leakage between train/val/test splits
2. Temporal ordering: val and test contain the LATEST voyages
3. Edge cases: empty dataframes, small datasets, invalid parameters
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the external Cols and SplittedData before importing DataSplitter
@dataclass
class MockSplittedData:
    """Mock for the external SplittedData dataclass."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class MockCols:
    """Mock for the external Cols class with column name constants."""
    VOY_ID = 'voy_id'
    DEPARTURE_DATE = 'departure_date'
    CREATED_DATE = 'created_date'
    LINE = 'line'
    LEG = 'leg'
    ORIGIN = 'origin'


# Patch the imports before importing the module under test
sys.modules['src.data.data_names'] = MagicMock()
sys.modules['src.data.data_names'].Cols = MockCols

sys.modules['src.data.data_classes'] = MagicMock()
sys.modules['src.data.data_classes'].SplittedData = MockSplittedData

# Mock mlflow to avoid import errors
sys.modules['mlflow'] = MagicMock()

from src.data_split import DataSplitter


class TestSplitDataByTime:
    """Test suite for DataSplitter.split_data_by_time method."""

    @pytest.fixture
    def splitter(self):
        """Create a DataSplitter instance."""
        return DataSplitter()

    @pytest.fixture
    def sample_df(self):
        """
        Create a sample DataFrame with 10 voyages, each having multiple rows.
        Voyages are ordered chronologically by departure_date.
        """
        data = []
        base_date = pd.Timestamp('2024-01-01')
        
        for voy_idx in range(10):
            voy_id = f'VOY_{voy_idx:03d}'
            departure_date = base_date + pd.Timedelta(days=voy_idx * 7)
            
            # Each voyage has 3-5 rows (multiple segments/legs)
            num_rows = np.random.randint(3, 6)
            for row_idx in range(num_rows):
                data.append({
                    'voy_id': voy_id,
                    'departure_date': departure_date,
                    'created_date': departure_date - pd.Timedelta(days=30 - row_idx),
                    'feature1': np.random.rand(),
                    'feature2': np.random.rand(),
                })
        
        return pd.DataFrame(data)

    def test_no_voy_id_leakage(self, splitter, sample_df):
        """
        CRITICAL TEST: Ensure no voy_id appears in more than one split.
        This is the primary correctness criterion.
        """
        result = splitter.split_data_by_time(sample_df, val_size=0.2, test_size=0.2)
        
        train_voy_ids = set(result.train['voy_id'].unique())
        val_voy_ids = set(result.val['voy_id'].unique())
        test_voy_ids = set(result.test['voy_id'].unique())
        
        # No intersection between any two sets
        assert train_voy_ids.isdisjoint(val_voy_ids), "Leakage detected: voy_ids in both train and val"
        assert train_voy_ids.isdisjoint(test_voy_ids), "Leakage detected: voy_ids in both train and test"
        assert val_voy_ids.isdisjoint(test_voy_ids), "Leakage detected: voy_ids in both val and test"
        
        # All original voy_ids are accounted for
        all_split_voy_ids = train_voy_ids | val_voy_ids | test_voy_ids
        original_voy_ids = set(sample_df['voy_id'].unique())
        assert all_split_voy_ids == original_voy_ids, "Some voy_ids were lost in splitting"

    def test_temporal_ordering(self, splitter, sample_df):
        """
        Ensure val and test contain the LATEST voyages chronologically.
        Train max date < Val min date <= Val max date < Test min date
        """
        result = splitter.split_data_by_time(sample_df, val_size=0.2, test_size=0.2)
        
        train_max_date = result.train['departure_date'].max()
        val_min_date = result.val['departure_date'].min()
        val_max_date = result.val['departure_date'].max()
        test_min_date = result.test['departure_date'].min()
        
        # Train should end before val starts
        assert train_max_date <= val_min_date, \
            f"Train max ({train_max_date}) should be <= Val min ({val_min_date})"
        
        # Val should end before test starts
        assert val_max_date <= test_min_date, \
            f"Val max ({val_max_date}) should be <= Test min ({test_min_date})"

    def test_split_sizes_approximate(self, splitter, sample_df):
        """
        Verify the split sizes are approximately correct.
        Allow for rounding since we split at voyage level.
        """
        result = splitter.split_data_by_time(sample_df, val_size=0.2, test_size=0.2)
        
        total_voyages = sample_df['voy_id'].nunique()
        train_voyages = result.train['voy_id'].nunique()
        val_voyages = result.val['voy_id'].nunique()
        test_voyages = result.test['voy_id'].nunique()
        
        # Approximate check (within 1 voyage due to rounding)
        expected_train = int(total_voyages * 0.6)
        expected_val = int(total_voyages * 0.2)
        expected_test = int(total_voyages * 0.2)
        
        assert abs(train_voyages - expected_train) <= 1
        assert abs(val_voyages - expected_val) <= 1
        assert abs(test_voyages - expected_test) <= 1

    def test_all_rows_preserved(self, splitter, sample_df):
        """Ensure no rows are lost during splitting."""
        result = splitter.split_data_by_time(sample_df, val_size=0.15, test_size=0.15)
        
        total_rows = len(sample_df)
        split_rows = len(result.train) + len(result.val) + len(result.test)
        
        assert split_rows == total_rows, \
            f"Row count mismatch: original={total_rows}, after split={split_rows}"

    def test_empty_dataframe(self, splitter):
        """Handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=['voy_id', 'departure_date', 'created_date'])
        result = splitter.split_data_by_time(empty_df, val_size=0.15, test_size=0.15)
        
        assert len(result.train) == 0
        assert len(result.val) == 0
        assert len(result.test) == 0

    def test_single_voyage(self, splitter):
        """Handle single voyage case - should all go to train."""
        single_df = pd.DataFrame({
            'voy_id': ['VOY_001'] * 5,
            'departure_date': [pd.Timestamp('2024-01-01')] * 5,
            'created_date': pd.date_range('2023-12-01', periods=5, freq='D'),
        })
        
        result = splitter.split_data_by_time(single_df, val_size=0.2, test_size=0.2)
        
        # With only 1 voyage, it should go to train (num_voyages <= 2 means no val/test)
        assert len(result.train) == 5
        assert len(result.val) == 0
        assert len(result.test) == 0

    def test_two_voyages(self, splitter):
        """Handle two voyage case - edge case for splitting."""
        two_df = pd.DataFrame({
            'voy_id': ['VOY_001'] * 3 + ['VOY_002'] * 3,
            'departure_date': [pd.Timestamp('2024-01-01')] * 3 + [pd.Timestamp('2024-01-08')] * 3,
            'created_date': pd.date_range('2023-12-01', periods=6, freq='D'),
        })
        
        result = splitter.split_data_by_time(two_df, val_size=0.2, test_size=0.2)
        
        # With only 2 voyages, both should go to train (num_voyages <= 2)
        assert len(result.train) == 6
        assert len(result.val) == 0
        assert len(result.test) == 0

    def test_invalid_sizes_negative(self, splitter, sample_df):
        """Raise error for negative sizes."""
        with pytest.raises(ValueError, match="non-negative"):
            splitter.split_data_by_time(sample_df, val_size=-0.1, test_size=0.1)
        
        with pytest.raises(ValueError, match="non-negative"):
            splitter.split_data_by_time(sample_df, val_size=0.1, test_size=-0.1)

    def test_invalid_sizes_too_large(self, splitter, sample_df):
        """Raise error when val_size + test_size >= 1.0."""
        with pytest.raises(ValueError, match="less than 1.0"):
            splitter.split_data_by_time(sample_df, val_size=0.5, test_size=0.5)
        
        with pytest.raises(ValueError, match="less than 1.0"):
            splitter.split_data_by_time(sample_df, val_size=0.6, test_size=0.5)

    def test_voyages_with_same_departure_date(self, splitter):
        """
        Handle multiple voyages with the same departure date.
        The split should still be deterministic and leak-free.
        """
        # 5 voyages all departing on the same date
        same_date_df = pd.DataFrame({
            'voy_id': ['VOY_001'] * 2 + ['VOY_002'] * 2 + ['VOY_003'] * 2 + ['VOY_004'] * 2 + ['VOY_005'] * 2,
            'departure_date': [pd.Timestamp('2024-01-01')] * 10,
            'created_date': pd.date_range('2023-12-01', periods=10, freq='D'),
        })
        
        result = splitter.split_data_by_time(same_date_df, val_size=0.2, test_size=0.2)
        
        # No leakage check
        train_voy_ids = set(result.train['voy_id'].unique())
        val_voy_ids = set(result.val['voy_id'].unique())
        test_voy_ids = set(result.test['voy_id'].unique())
        
        assert train_voy_ids.isdisjoint(val_voy_ids)
        assert train_voy_ids.isdisjoint(test_voy_ids)
        assert val_voy_ids.isdisjoint(test_voy_ids)

    def test_large_dataset(self, splitter):
        """Performance test with larger dataset."""
        data = []
        base_date = pd.Timestamp('2020-01-01')
        
        for voy_idx in range(1000):
            voy_id = f'VOY_{voy_idx:05d}'
            departure_date = base_date + pd.Timedelta(days=voy_idx)
            
            for row_idx in range(5):
                data.append({
                    'voy_id': voy_id,
                    'departure_date': departure_date,
                    'created_date': departure_date - pd.Timedelta(days=30 - row_idx),
                    'feature1': np.random.rand(),
                })
        
        large_df = pd.DataFrame(data)
        result = splitter.split_data_by_time(large_df, val_size=0.15, test_size=0.15)
        
        # Basic sanity checks
        assert len(result.train) + len(result.val) + len(result.test) == len(large_df)
        
        # No leakage
        train_voy_ids = set(result.train['voy_id'].unique())
        val_voy_ids = set(result.val['voy_id'].unique())
        test_voy_ids = set(result.test['voy_id'].unique())
        
        assert train_voy_ids.isdisjoint(val_voy_ids)
        assert train_voy_ids.isdisjoint(test_voy_ids)
        assert val_voy_ids.isdisjoint(test_voy_ids)


class TestTimeSeriesCVExpanding:
    """Test suite for DataSplitter.time_series_cv_expanding method."""

    @pytest.fixture
    def splitter(self):
        return DataSplitter()

    @pytest.fixture
    def cv_dataset(self):
        """
        Create a dataset spanning ~2 years with 100 voyages.
        This ensures enough data for validation window and test set.
        """
        data = []
        base_date = pd.Timestamp('2022-01-01')
        
        for voy_idx in range(100):
            voy_id = f'VOY_{voy_idx:03d}'
            # Spread voyages over ~730 days (2 years)
            departure_date = base_date + pd.Timedelta(days=voy_idx * 7)
            
            for row_idx in range(3):
                data.append({
                    'voy_id': voy_id,
                    'departure_date': departure_date,
                    'created_date': departure_date - pd.Timedelta(days=30 - row_idx),
                    'feature1': np.random.rand(),
                })
        
        return pd.DataFrame(data)

    def test_returns_correct_number_of_folds(self, splitter, cv_dataset):
        """Should return exactly n_splits folds."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        assert len(folds) == 4

    def test_no_voy_id_leakage_within_fold(self, splitter, cv_dataset):
        """No voyage should appear in multiple splits within any fold."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        for i, fold in enumerate(folds):
            train_voy_ids = set(fold.train['voy_id'].unique())
            val_voy_ids = set(fold.val['voy_id'].unique())
            test_voy_ids = set(fold.test['voy_id'].unique())
            
            assert train_voy_ids.isdisjoint(val_voy_ids), f"Fold {i}: train-val leakage"
            assert train_voy_ids.isdisjoint(test_voy_ids), f"Fold {i}: train-test leakage"
            assert val_voy_ids.isdisjoint(test_voy_ids), f"Fold {i}: val-test leakage"

    def test_test_set_same_across_folds(self, splitter, cv_dataset):
        """Test set should be identical across all folds."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        test_voy_ids_0 = set(folds[0].test['voy_id'].unique())
        
        for i, fold in enumerate(folds[1:], start=1):
            test_voy_ids_i = set(fold.test['voy_id'].unique())
            assert test_voy_ids_0 == test_voy_ids_i, f"Fold {i} has different test set"

    def test_expanding_train_grows(self, splitter, cv_dataset):
        """In expanding mode, training set should grow with each fold."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        train_sizes = [fold.train['voy_id'].nunique() for fold in folds]
        
        # Each subsequent fold should have >= training voyages
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], \
                f"Training should expand: fold {i} ({train_sizes[i]}) < fold {i-1} ({train_sizes[i-1]})"

    def test_temporal_ordering_across_folds(self, splitter, cv_dataset):
        """Val periods should be chronologically ordered across folds."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        val_max_dates = [fold.val['departure_date'].max() for fold in folds]
        
        # Each fold's val max should be <= next fold's val max
        for i in range(len(val_max_dates) - 1):
            assert val_max_dates[i] <= val_max_dates[i+1], \
                f"Val periods not chronologically ordered: fold {i} > fold {i+1}"

    def test_val_before_test(self, splitter, cv_dataset):
        """All validation data should be before test data."""
        folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        for i, fold in enumerate(folds):
            val_max = fold.val['departure_date'].max()
            test_min = fold.test['departure_date'].min()
            assert val_max <= test_min, f"Fold {i}: val ends after test starts"

    def test_invalid_n_splits(self, splitter, cv_dataset):
        """Should reject n_splits < 2."""
        with pytest.raises(ValueError, match="at least 2"):
            splitter.time_series_cv_expanding(cv_dataset, n_splits=1)

    def test_invalid_test_size(self, splitter, cv_dataset):
        """Should reject invalid test_size."""
        with pytest.raises(ValueError):
            splitter.time_series_cv_expanding(cv_dataset, test_size=0)
        with pytest.raises(ValueError):
            splitter.time_series_cv_expanding(cv_dataset, test_size=1.0)


class TestTimeSeriesCVPooled:
    """Test suite for DataSplitter.time_series_cv_pooled method."""

    @pytest.fixture
    def splitter(self):
        return DataSplitter()

    @pytest.fixture
    def cv_dataset(self):
        """Same dataset as expanding tests."""
        data = []
        base_date = pd.Timestamp('2022-01-01')
        
        for voy_idx in range(100):
            voy_id = f'VOY_{voy_idx:03d}'
            departure_date = base_date + pd.Timedelta(days=voy_idx * 7)
            
            for row_idx in range(3):
                data.append({
                    'voy_id': voy_id,
                    'departure_date': departure_date,
                    'created_date': departure_date - pd.Timedelta(days=30 - row_idx),
                    'feature1': np.random.rand(),
                })
        
        return pd.DataFrame(data)

    def test_returns_correct_number_of_folds(self, splitter, cv_dataset):
        """Should return exactly n_splits folds."""
        folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        assert len(folds) == 4

    def test_no_voy_id_leakage_within_fold(self, splitter, cv_dataset):
        """No voyage should appear in multiple splits within any fold."""
        folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        for i, fold in enumerate(folds):
            train_voy_ids = set(fold.train['voy_id'].unique())
            val_voy_ids = set(fold.val['voy_id'].unique())
            test_voy_ids = set(fold.test['voy_id'].unique())
            
            assert train_voy_ids.isdisjoint(val_voy_ids), f"Fold {i}: train-val leakage"
            assert train_voy_ids.isdisjoint(test_voy_ids), f"Fold {i}: train-test leakage"
            assert val_voy_ids.isdisjoint(test_voy_ids), f"Fold {i}: val-test leakage"

    def test_pooled_train_roughly_constant(self, splitter, cv_dataset):
        """
        In pooled mode, training set size should be roughly constant across folds.
        (Training pool + n_splits-1 periods, only the excluded period changes)
        """
        folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        train_sizes = [fold.train['voy_id'].nunique() for fold in folds]
        
        # All training sizes should be similar (within 20% of each other)
        max_size = max(train_sizes)
        min_size = min(train_sizes)
        
        # Allow some variance due to unequal period sizes
        assert min_size >= max_size * 0.7, \
            f"Pooled training sizes vary too much: {train_sizes}"

    def test_pooled_has_more_training_than_expanding(self, splitter, cv_dataset):
        """
        Pooled mode should have >= training data than expanding mode at fold 1.
        (Because pooled includes future periods, expanding doesn't)
        """
        expanding_folds = splitter.time_series_cv_expanding(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        pooled_folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        # First fold: expanding has only pool, pooled has pool + Q2+Q3+Q4
        exp_train_size = expanding_folds[0].train['voy_id'].nunique()
        pool_train_size = pooled_folds[0].train['voy_id'].nunique()
        
        assert pool_train_size >= exp_train_size, \
            f"Pooled ({pool_train_size}) should have >= expanding ({exp_train_size}) at fold 0"

    def test_val_sets_disjoint_across_folds(self, splitter, cv_dataset):
        """Validation sets should be disjoint across different folds."""
        folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        all_val_voy_ids = []
        for fold in folds:
            all_val_voy_ids.append(set(fold.val['voy_id'].unique()))
        
        # Check all pairs for disjoint
        for i in range(len(all_val_voy_ids)):
            for j in range(i + 1, len(all_val_voy_ids)):
                intersection = all_val_voy_ids[i] & all_val_voy_ids[j]
                assert len(intersection) == 0, \
                    f"Fold {i} and {j} share val voyages: {intersection}"

    def test_all_val_window_voyages_covered(self, splitter, cv_dataset):
        """Union of all val sets should equal all voyages in the validation window."""
        folds = splitter.time_series_cv_pooled(
            cv_dataset, test_size=0.15, val_window_days=180, n_splits=4
        )
        
        # Collect all val voyages
        all_val_voy_ids = set()
        for fold in folds:
            all_val_voy_ids.update(fold.val['voy_id'].unique())
        
        # These should be all voyages in the validation window
        # We can verify by checking that val voyages >= n_splits (minimum 1 per fold)
        assert len(all_val_voy_ids) >= 4, "Should have at least 1 voyage per fold in val window"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

