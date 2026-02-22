"""Tests for drift detectors: KS, PSI, chi-square, NaN rate, label drift."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.drift.config import DriftConfig
from src.drift.detectors import DriftDetector, DriftResult


@pytest.fixture
def config() -> DriftConfig:
    """Minimal config with small feature lists for testing."""
    return DriftConfig(
        numeric_features=["feat_a", "feat_b"],
        categorical_features=["cat_x"],
        label_column="target",
    )


@pytest.fixture
def detector(config: DriftConfig) -> DriftDetector:
    return DriftDetector(config)


# ------------------------------------------------------------------
# KS tests (numeric data drift)
# ------------------------------------------------------------------


def test_ks_no_drift(detector: DriftDetector) -> None:
    """Identical numeric distributions should produce no drift (HEALTHY)."""
    np.random.seed(42)
    data = np.random.normal(0, 1, size=500)
    baseline = pd.DataFrame({"feat_a": data, "feat_b": data})
    current = pd.DataFrame({"feat_a": data, "feat_b": data})

    results = detector._numeric_data_drift(baseline, current)

    assert len(results) == 2
    for r in results:
        assert r.severity == "HEALTHY"
        assert r.drift_type == "data_drift"
        assert r.test_name == "KS + PSI"
        assert not r.is_drifted


def test_ks_drift(detector: DriftDetector) -> None:
    """Shifted numeric distribution (mean shift) should be detected as drift."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        "feat_a": np.random.normal(0, 1, size=1000),
        "feat_b": np.random.normal(0, 1, size=1000),
    })
    current = pd.DataFrame({
        "feat_a": np.random.normal(5, 1, size=1000),  # large shift
        "feat_b": np.random.normal(0, 1, size=1000),  # no shift
    })

    results = detector._numeric_data_drift(baseline, current)

    feat_a_result = [r for r in results if r.feature == "feat_a"][0]
    feat_b_result = [r for r in results if r.feature == "feat_b"][0]

    # feat_a should show drift (large mean shift)
    assert feat_a_result.is_drifted
    assert feat_a_result.severity in ("WARNING", "CRITICAL")
    assert feat_a_result.p_value < 0.05

    # feat_b should be healthy (same distribution)
    assert feat_b_result.severity == "HEALTHY"


# ------------------------------------------------------------------
# PSI calculation
# ------------------------------------------------------------------


def test_psi_calculation() -> None:
    """PSI of identical distributions should be near zero; shifted should be large."""
    np.random.seed(42)
    same_a = np.random.normal(0, 1, size=5000)
    same_b = np.random.normal(0, 1, size=5000)

    psi_same = DriftDetector.compute_psi(same_a, same_b, n_bins=10)
    assert psi_same < 0.1, f"PSI for same distribution should be < 0.1, got {psi_same}"

    shifted = np.random.normal(3, 1, size=5000)
    psi_shifted = DriftDetector.compute_psi(same_a, shifted, n_bins=10)
    assert psi_shifted > 0.2, f"PSI for shifted distribution should be > 0.2, got {psi_shifted}"


def test_psi_non_negative() -> None:
    """PSI should always be non-negative."""
    np.random.seed(123)
    a = np.random.uniform(0, 10, size=500)
    b = np.random.uniform(2, 12, size=500)
    psi = DriftDetector.compute_psi(a, b, n_bins=10)
    assert psi >= 0


# ------------------------------------------------------------------
# Chi-square tests (categorical data drift)
# ------------------------------------------------------------------


def test_chi2_no_drift(detector: DriftDetector) -> None:
    """Same categorical distribution should produce no drift."""
    categories = ["A"] * 200 + ["B"] * 200 + ["C"] * 100
    baseline = pd.DataFrame({"cat_x": categories})
    current = pd.DataFrame({"cat_x": categories})

    results = detector._categorical_data_drift(baseline, current)

    assert len(results) == 1
    r = results[0]
    assert r.severity == "HEALTHY"
    assert r.test_name == "Chi-Square"
    assert not r.is_drifted


def test_chi2_drift(detector: DriftDetector) -> None:
    """Different categorical proportions should trigger drift."""
    baseline = pd.DataFrame({"cat_x": ["A"] * 400 + ["B"] * 100})
    current = pd.DataFrame({"cat_x": ["A"] * 100 + ["B"] * 400})

    results = detector._categorical_data_drift(baseline, current)

    assert len(results) == 1
    r = results[0]
    assert r.severity == "WARNING"
    assert r.is_drifted
    assert r.p_value < 0.05


# ------------------------------------------------------------------
# NaN rate drift (data quality)
# ------------------------------------------------------------------


def test_nan_rate_drift(detector: DriftDetector) -> None:
    """Significantly different NaN rates should trigger data quality drift."""
    np.random.seed(42)
    n = 500

    # Baseline: 0% NaN on feat_a
    baseline = pd.DataFrame({
        "feat_a": np.random.normal(0, 1, size=n),
        "feat_b": np.random.normal(0, 1, size=n),
        "cat_x": ["A"] * n,
    })

    # Current: 20% NaN on feat_a (exceeds 5% threshold)
    feat_a_current = np.random.normal(0, 1, size=n).astype(float)
    nan_mask = np.random.choice(n, size=int(n * 0.20), replace=False)
    feat_a_current[nan_mask] = np.nan

    current = pd.DataFrame({
        "feat_a": feat_a_current,
        "feat_b": np.random.normal(0, 1, size=n),
        "cat_x": ["A"] * n,
    })

    results = detector._nan_rate_drift(baseline, current)

    # Should detect NaN rate drift on feat_a
    feat_a_results = [r for r in results if r.feature == "feat_a"]
    assert len(feat_a_results) >= 1
    r = feat_a_results[0]
    assert r.is_drifted
    assert r.drift_type == "data_quality"
    assert r.test_name == "NaN rate change"


def test_nan_rate_no_drift(detector: DriftDetector) -> None:
    """Identical NaN rates should not trigger drift."""
    np.random.seed(42)
    n = 500
    data = np.random.normal(0, 1, size=n)
    df = pd.DataFrame({
        "feat_a": data,
        "feat_b": data,
        "cat_x": ["A"] * n,
    })

    results = detector._nan_rate_drift(df, df)
    assert len(results) == 0


# ------------------------------------------------------------------
# Label drift
# ------------------------------------------------------------------


def test_label_drift(detector: DriftDetector) -> None:
    """Shifted target distribution should trigger label drift."""
    np.random.seed(42)
    baseline = pd.DataFrame({"target": np.random.normal(0, 1, size=1000)})
    current = pd.DataFrame({"target": np.random.normal(5, 1, size=1000)})

    results = detector.detect_label_drift(baseline, current)

    assert len(results) == 1
    r = results[0]
    assert r.drift_type == "label_drift"
    assert r.is_drifted
    assert r.severity in ("WARNING", "CRITICAL")


def test_label_no_drift(detector: DriftDetector) -> None:
    """Identical target distribution should not trigger label drift."""
    np.random.seed(42)
    data = np.random.normal(0, 1, size=1000)
    baseline = pd.DataFrame({"target": data})
    current = pd.DataFrame({"target": data})

    results = detector.detect_label_drift(baseline, current)

    assert len(results) == 1
    r = results[0]
    assert r.severity == "HEALTHY"
    assert not r.is_drifted


# ------------------------------------------------------------------
# Concept drift
# ------------------------------------------------------------------


def test_concept_drift_detected(detector: DriftDetector) -> None:
    """Large increase in residuals should trigger concept drift."""
    np.random.seed(42)
    n = 500
    baseline = pd.DataFrame({
        "actual": np.random.normal(10, 1, size=n),
        "predicted": np.random.normal(10, 1, size=n),
    })
    current = pd.DataFrame({
        "actual": np.random.normal(10, 1, size=n),
        "predicted": np.random.normal(15, 1, size=n),  # large error
    })

    results = detector.detect_concept_drift(
        baseline, current, actual_col="actual", predicted_col="predicted"
    )

    # Should have MAE comparison and residual KS test
    assert len(results) == 2
    mae_result = [r for r in results if r.test_name == "MAE comparison"][0]
    assert mae_result.is_drifted


# ------------------------------------------------------------------
# Prediction drift
# ------------------------------------------------------------------


def test_prediction_drift(detector: DriftDetector) -> None:
    """Shifted prediction distribution should trigger prediction drift."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        "pred_col": np.random.normal(0, 1, size=1000)
    })
    current = pd.DataFrame({
        "pred_col": np.random.normal(5, 1, size=1000)
    })

    results = detector.detect_prediction_drift(
        baseline, current, prediction_col="pred_col"
    )

    assert len(results) == 1
    assert results[0].is_drifted


# ------------------------------------------------------------------
# DriftResult properties
# ------------------------------------------------------------------


def test_drift_result_is_drifted() -> None:
    """DriftResult.is_drifted should correctly reflect severity."""
    healthy = DriftResult(
        feature="x", drift_type="data_drift", test_name="KS",
        statistic=0.1, p_value=0.5, severity="HEALTHY",
    )
    warning = DriftResult(
        feature="x", drift_type="data_drift", test_name="KS",
        statistic=0.1, p_value=0.01, severity="WARNING",
    )
    critical = DriftResult(
        feature="x", drift_type="data_drift", test_name="KS",
        statistic=0.5, p_value=0.001, severity="CRITICAL",
    )
    assert not healthy.is_drifted
    assert warning.is_drifted
    assert critical.is_drifted
