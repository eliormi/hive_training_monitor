"""Integration tests for DriftRunner: full pipeline with synthetic data."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.drift.config import DriftConfig
from src.drift.detectors import DriftResult
from src.drift.runner import DriftRunner


@pytest.fixture
def config(tmp_path: object) -> DriftConfig:
    """Config with small feature lists and temp baseline dir."""
    return DriftConfig(
        numeric_features=["feat_a", "feat_b"],
        categorical_features=["cat_x"],
        label_column="target",
        baseline_dir=str(tmp_path / "baselines"),
    )


@pytest.fixture
def runner(config: DriftConfig) -> DriftRunner:
    return DriftRunner(config)


def _make_df(
    n: int = 500,
    mean_a: float = 0.0,
    mean_b: float = 5.0,
    target_mean: float = 0.0,
    cat_probs: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper to create synthetic DataFrames for testing."""
    np.random.seed(seed)
    if cat_probs is None:
        cat_probs = [0.5, 0.3, 0.2]
    categories = np.random.choice(["A", "B", "C"], size=n, p=cat_probs)
    return pd.DataFrame({
        "feat_a": np.random.normal(mean_a, 1, size=n),
        "feat_b": np.random.normal(mean_b, 1, size=n),
        "cat_x": categories,
        "target": np.random.normal(target_mean, 1, size=n),
        "confirmed_at_end": np.random.normal(10, 1, size=n),
        "cumsum_fresh_teu": np.random.normal(10, 1, size=n),
    })


# ------------------------------------------------------------------
# Full pipeline: no drift
# ------------------------------------------------------------------


def test_full_pipeline_no_drift(runner: DriftRunner) -> None:
    """Identical train and test data should produce HEALTHY overall status."""
    baseline_df = _make_df(seed=42)
    current_df = _make_df(seed=42)

    # Save as both baseline types
    runner.baseline_manager.save_baseline(baseline_df, baseline_type="last_train")
    runner.baseline_manager.save_baseline(baseline_df, baseline_type="yearly")

    report = runner.run(current_df=current_df)

    assert report.overall_status == "HEALTHY"

    # All unified verdicts should be HEALTHY
    for feat, verdict in report.unified_verdicts.items():
        assert verdict == "HEALTHY", f"Feature {feat} has verdict {verdict}, expected HEALTHY"


# ------------------------------------------------------------------
# Full pipeline: with drift
# ------------------------------------------------------------------


def test_full_pipeline_with_drift(runner: DriftRunner) -> None:
    """Shifted current data should produce drift detection."""
    baseline_df = _make_df(seed=42, mean_a=0.0)
    current_df = _make_df(seed=99, mean_a=10.0, target_mean=10.0)  # large shift

    runner.baseline_manager.save_baseline(baseline_df, baseline_type="last_train")
    runner.baseline_manager.save_baseline(baseline_df, baseline_type="yearly")

    report = runner.run(current_df=current_df)

    # Should detect some drift
    assert report.overall_status in ("WARNING", "CRITICAL")

    # At least some features should show drift
    drifted_features = {
        feat for feat, verdict in report.unified_verdicts.items()
        if verdict == "DRIFT"
    }
    assert len(drifted_features) > 0, "Expected at least one DRIFT verdict"


# ------------------------------------------------------------------
# Unified verdict: SEASONAL
# ------------------------------------------------------------------


def test_unified_verdict_seasonal(runner: DriftRunner) -> None:
    """Drift vs train only (not yearly) should produce SEASONAL verdict."""
    baseline_train = _make_df(seed=42, mean_a=0.0)
    baseline_yearly = _make_df(seed=99, mean_a=10.0)  # yearly matches current
    current_df = _make_df(seed=99, mean_a=10.0)        # same as yearly

    runner.baseline_manager.save_baseline(baseline_train, baseline_type="last_train")
    runner.baseline_manager.save_baseline(baseline_yearly, baseline_type="yearly")

    report = runner.run(current_df=current_df)

    # feat_a should show drift vs train but NOT vs yearly -> SEASONAL
    seasonal_features = {
        feat for feat, verdict in report.unified_verdicts.items()
        if verdict == "SEASONAL"
    }
    assert "feat_a" in seasonal_features, (
        f"Expected feat_a to have SEASONAL verdict. "
        f"Verdicts: {report.unified_verdicts}"
    )


# ------------------------------------------------------------------
# Unified verdict: static method
# ------------------------------------------------------------------


def test_compute_unified_verdict_all_cases() -> None:
    """Test all 4 verdict outcomes from the static method."""
    # Feature in both train and yearly drift
    train_results = [
        DriftResult(feature="A", drift_type="data_drift", test_name="KS",
                    statistic=0.5, p_value=0.001, severity="WARNING"),
        DriftResult(feature="B", drift_type="data_drift", test_name="KS",
                    statistic=0.5, p_value=0.001, severity="WARNING"),
        DriftResult(feature="C", drift_type="data_drift", test_name="KS",
                    statistic=0.1, p_value=0.5, severity="HEALTHY"),
        DriftResult(feature="D", drift_type="data_drift", test_name="KS",
                    statistic=0.1, p_value=0.5, severity="HEALTHY"),
    ]
    yearly_results = [
        DriftResult(feature="A", drift_type="data_drift", test_name="KS",
                    statistic=0.5, p_value=0.001, severity="WARNING"),
        DriftResult(feature="B", drift_type="data_drift", test_name="KS",
                    statistic=0.1, p_value=0.5, severity="HEALTHY"),
        DriftResult(feature="C", drift_type="data_drift", test_name="KS",
                    statistic=0.5, p_value=0.001, severity="WARNING"),
        DriftResult(feature="D", drift_type="data_drift", test_name="KS",
                    statistic=0.1, p_value=0.5, severity="HEALTHY"),
    ]

    verdicts = DriftRunner.compute_unified_verdict(train_results, yearly_results)

    assert verdicts["A"] == "DRIFT"      # drift in both
    assert verdicts["B"] == "SEASONAL"   # drift only in train
    assert verdicts["C"] == "YEAR-SHIFT" # drift only in yearly
    assert verdicts["D"] == "HEALTHY"    # no drift in either


# ------------------------------------------------------------------
# Only train baseline
# ------------------------------------------------------------------


def test_run_with_only_train_baseline(runner: DriftRunner) -> None:
    """Pipeline should work with only a train baseline (no yearly)."""
    baseline_df = _make_df(seed=42)
    current_df = _make_df(seed=42)

    runner.baseline_manager.save_baseline(baseline_df, baseline_type="last_train")
    # No yearly baseline

    report = runner.run(current_df=current_df)

    assert report.overall_status == "HEALTHY"
    assert len(report.results_vs_train) > 0
    assert len(report.results_vs_yearly) == 0


# ------------------------------------------------------------------
# Report structure
# ------------------------------------------------------------------


def test_report_serialization(runner: DriftRunner, tmp_path: object) -> None:
    """Report should serialize to JSON and save to file."""
    baseline_df = _make_df(seed=42)
    current_df = _make_df(seed=42)

    runner.baseline_manager.save_baseline(baseline_df, baseline_type="last_train")
    runner.baseline_manager.save_baseline(baseline_df, baseline_type="yearly")

    report = runner.run(current_df=current_df)

    # JSON serialization
    json_str = report.to_json()
    assert isinstance(json_str, str)
    assert "overall_status" in json_str

    # File save
    saved_path = report.save(str(tmp_path / "reports"))
    from pathlib import Path
    assert Path(saved_path).exists()
