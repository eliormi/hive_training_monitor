"""Tests for BaselineManager: save, load, metadata, backup."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.drift.baseline_manager import BaselineManager
from src.drift.config import DriftConfig


@pytest.fixture
def config(tmp_path: object) -> DriftConfig:
    """Config pointing to a temp directory for baselines."""
    return DriftConfig(baseline_dir=str(tmp_path / "baselines"))


@pytest.fixture
def manager(config: DriftConfig) -> BaselineManager:
    return BaselineManager(config)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "feat_a": np.random.normal(0, 1, size=100),
        "feat_b": np.random.normal(5, 2, size=100),
        "cat_x": np.random.choice(["A", "B", "C"], size=100),
    })


# ------------------------------------------------------------------
# Save and load
# ------------------------------------------------------------------


def test_save_and_load(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Round-trip save/load should preserve the DataFrame."""
    path = manager.save_baseline(sample_df, baseline_type="last_train", label="test run")

    loaded = manager.load_baseline("last_train")

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_save_both_types(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Both baseline types should coexist independently."""
    manager.save_baseline(sample_df, baseline_type="last_train")
    manager.save_baseline(sample_df, baseline_type="yearly")

    assert manager.has_baseline("last_train")
    assert manager.has_baseline("yearly")

    loaded_train = manager.load_baseline("last_train")
    loaded_yearly = manager.load_baseline("yearly")

    assert loaded_train is not None
    assert loaded_yearly is not None
    pd.testing.assert_frame_equal(loaded_train, sample_df)
    pd.testing.assert_frame_equal(loaded_yearly, sample_df)


def test_load_nonexistent(manager: BaselineManager) -> None:
    """Loading a baseline that doesn't exist should return None."""
    result = manager.load_baseline("last_train")
    assert result is None


def test_has_baseline(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """has_baseline should reflect whether a baseline is saved."""
    assert not manager.has_baseline("last_train")

    manager.save_baseline(sample_df, baseline_type="last_train")

    assert manager.has_baseline("last_train")


def test_invalid_baseline_type(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Invalid baseline type should raise ValueError."""
    with pytest.raises(ValueError, match="baseline_type must be one of"):
        manager.save_baseline(sample_df, baseline_type="invalid")

    with pytest.raises(ValueError, match="baseline_type must be one of"):
        manager.load_baseline("invalid")


# ------------------------------------------------------------------
# Metadata tracking
# ------------------------------------------------------------------


def test_metadata_tracking(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Metadata JSON should be updated with label, row count, and date."""
    manager.save_baseline(sample_df, baseline_type="last_train", label="v1.0")

    # Read metadata directly
    with open(manager.meta_path, "r") as f:
        meta = json.load(f)

    # Check active entry
    active = meta["active"]["last_train"]
    assert active["label"] == "v1.0"
    assert active["n_rows"] == len(sample_df)
    assert "date" in active
    assert "created_at" in active
    assert "filename" in active

    # Check history
    assert len(meta["history"]) == 1
    assert meta["history"][0]["label"] == "v1.0"
    assert meta["history"][0]["baseline_type"] == "last_train"


def test_metadata_history_grows(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Each save should append to the history list."""
    manager.save_baseline(sample_df, baseline_type="last_train", label="first")
    manager.save_baseline(sample_df, baseline_type="last_train", label="second")

    history = manager.list_baselines()
    assert len(history) == 2
    assert history[0]["label"] == "first"
    assert history[1]["label"] == "second"


def test_get_active_baselines(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """get_active_baselines should return currently active entries."""
    manager.save_baseline(sample_df, baseline_type="last_train", label="lt")
    manager.save_baseline(sample_df, baseline_type="yearly", label="yr")

    active = manager.get_active_baselines()
    assert "last_train" in active
    assert "yearly" in active
    assert active["last_train"]["label"] == "lt"
    assert active["yearly"]["label"] == "yr"


# ------------------------------------------------------------------
# Backup creation
# ------------------------------------------------------------------


def test_backup_creation(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """.bak files should be created when overwriting existing baselines."""
    # First save
    path = manager.save_baseline(sample_df, baseline_type="last_train", label="v1")

    # Second save (should create .bak of first)
    manager.save_baseline(sample_df, baseline_type="last_train", label="v2")

    from pathlib import Path
    bak_path = Path(path + ".bak")
    assert bak_path.exists(), f"Expected .bak file at {bak_path}"


def test_metadata_backup(manager: BaselineManager, sample_df: pd.DataFrame) -> None:
    """Metadata file should also get .bak backup on update."""
    manager.save_baseline(sample_df, baseline_type="last_train", label="v1")
    # At this point, meta file exists

    manager.save_baseline(sample_df, baseline_type="last_train", label="v2")
    # Should have created meta .bak

    from pathlib import Path
    meta_bak = Path(str(manager.meta_path) + ".bak")
    assert meta_bak.exists(), f"Expected metadata .bak file at {meta_bak}"
