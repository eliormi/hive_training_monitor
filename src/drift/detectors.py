"""Statistical drift detection: 5 drift types with dual-threshold decision logic."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.drift.config import DriftConfig


@dataclass
class DriftResult:
    """Result of a single drift test on one feature or metric."""

    feature: str
    drift_type: str
    test_name: str
    statistic: float
    p_value: float
    severity: str  # "HEALTHY", "WARNING", "CRITICAL"
    detail: Dict = field(default_factory=dict)

    @property
    def is_drifted(self) -> bool:
        return self.severity in ("WARNING", "CRITICAL")


class DriftDetector:
    """Runs statistical drift tests comparing current data against a baseline.

    Supports five drift types:
    1. Data drift (covariate shift) -- KS + PSI for numerics, chi2 for categoricals
    2. Label drift (target shift)
    3. Concept drift (accuracy degradation)
    4. Prediction drift (model output shift)
    5. Data quality drift (NaN rates, schema, cardinality)
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # PSI computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_psi(
        baseline: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Population Stability Index between two distributions.

        Bins both arrays into ``n_bins`` equal-width bins derived from the
        baseline range, then computes sum( (P - Q) * ln(P / Q) ).

        Small epsilon is added to avoid division by zero.
        """
        eps = 1e-4

        # Determine bin edges from baseline
        min_val = min(np.min(baseline), np.min(current))
        max_val = max(np.max(baseline), np.max(current))
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
        current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

        # Convert to proportions
        baseline_pct = baseline_counts / baseline_counts.sum() + eps
        current_pct = current_counts / current_counts.sum() + eps

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    # ------------------------------------------------------------------
    # Severity helpers
    # ------------------------------------------------------------------

    def _numeric_severity(self, p_value: float, psi: float) -> str:
        """Determine severity from KS p-value AND PSI (dual threshold)."""
        significant = p_value < self.config.ks_alpha
        if significant and psi >= self.config.psi_critical:
            return "CRITICAL"
        if significant and psi >= self.config.psi_warning:
            return "WARNING"
        return "HEALTHY"

    def _categorical_severity(self, p_value: float) -> str:
        """Determine severity from chi-square p-value."""
        if p_value < self.config.chi2_alpha:
            return "WARNING"
        return "HEALTHY"

    # ------------------------------------------------------------------
    # 1. Data Drift
    # ------------------------------------------------------------------

    def detect_data_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Detect covariate shift on numeric and categorical features."""
        results: List[DriftResult] = []
        results.extend(self._numeric_data_drift(baseline_df, current_df))
        results.extend(self._categorical_data_drift(baseline_df, current_df))
        return results

    def _numeric_data_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        results: List[DriftResult] = []
        for feat in self.config.numeric_features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            bl = baseline_df[feat].dropna().values
            cur = current_df[feat].dropna().values

            if len(bl) < 2 or len(cur) < 2:
                continue

            ks_stat, ks_p = ks_2samp(bl, cur)
            psi = self.compute_psi(bl, cur, n_bins=self.config.psi_n_bins)
            severity = self._numeric_severity(ks_p, psi)

            results.append(
                DriftResult(
                    feature=feat,
                    drift_type="data_drift",
                    test_name="KS + PSI",
                    statistic=round(ks_stat, 6),
                    p_value=round(ks_p, 6),
                    severity=severity,
                    detail={"psi": round(psi, 6), "ks_statistic": round(ks_stat, 6)},
                )
            )
        return results

    def _categorical_data_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        results: List[DriftResult] = []
        for feat in self.config.categorical_features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            bl_counts = baseline_df[feat].value_counts()
            cur_counts = current_df[feat].value_counts()

            # Align categories
            all_cats = sorted(set(bl_counts.index) | set(cur_counts.index))
            bl_aligned = pd.Series([bl_counts.get(c, 0) for c in all_cats], index=all_cats)
            cur_aligned = pd.Series([cur_counts.get(c, 0) for c in all_cats], index=all_cats)

            contingency = pd.DataFrame({"baseline": bl_aligned, "current": cur_aligned})

            # Skip if all zeros in either column
            if contingency["baseline"].sum() == 0 or contingency["current"].sum() == 0:
                continue

            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency.values.T)
            except ValueError:
                continue

            severity = self._categorical_severity(p_value)

            results.append(
                DriftResult(
                    feature=feat,
                    drift_type="data_drift",
                    test_name="Chi-Square",
                    statistic=round(chi2, 6),
                    p_value=round(p_value, 6),
                    severity=severity,
                    detail={"chi2": round(chi2, 6), "dof": dof},
                )
            )
        return results

    # ------------------------------------------------------------------
    # 2. Label Drift
    # ------------------------------------------------------------------

    def detect_label_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Detect shift in the label (target) distribution."""
        label = self.config.label_column
        if label not in baseline_df.columns or label not in current_df.columns:
            return []

        bl = baseline_df[label].dropna().values
        cur = current_df[label].dropna().values

        if len(bl) < 2 or len(cur) < 2:
            return []

        ks_stat, ks_p = ks_2samp(bl, cur)
        psi = self.compute_psi(bl, cur, n_bins=self.config.psi_n_bins)
        severity = self._numeric_severity(ks_p, psi)

        return [
            DriftResult(
                feature=label,
                drift_type="label_drift",
                test_name="KS + PSI",
                statistic=round(ks_stat, 6),
                p_value=round(ks_p, 6),
                severity=severity,
                detail={"psi": round(psi, 6), "ks_statistic": round(ks_stat, 6)},
            )
        ]

    # ------------------------------------------------------------------
    # 3. Concept Drift
    # ------------------------------------------------------------------

    def detect_concept_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        actual_col: str = "confirmed_at_end",
        predicted_col: str = "cumsum_fresh_teu",
    ) -> List[DriftResult]:
        """Detect accuracy degradation by comparing residual distributions.

        Computes residuals (actual - predicted) for both baseline and current,
        then checks for MAE increase and residual distribution shift.
        """
        results: List[DriftResult] = []

        for col in [actual_col, predicted_col]:
            if col not in baseline_df.columns or col not in current_df.columns:
                return results

        bl_residuals = (baseline_df[actual_col] - baseline_df[predicted_col]).dropna().values
        cur_residuals = (current_df[actual_col] - current_df[predicted_col]).dropna().values

        if len(bl_residuals) < 2 or len(cur_residuals) < 2:
            return results

        bl_mae = float(np.mean(np.abs(bl_residuals)))
        cur_mae = float(np.mean(np.abs(cur_residuals)))

        if bl_mae > 0:
            mae_change_pct = (cur_mae - bl_mae) / bl_mae
        else:
            mae_change_pct = 0.0

        # MAE-based severity
        if mae_change_pct >= self.config.performance_degradation_critical_pct:
            severity = "CRITICAL"
        elif mae_change_pct >= self.config.performance_degradation_warning_pct:
            severity = "WARNING"
        else:
            severity = "HEALTHY"

        results.append(
            DriftResult(
                feature=f"{actual_col} vs {predicted_col}",
                drift_type="concept_drift",
                test_name="MAE comparison",
                statistic=round(mae_change_pct, 6),
                p_value=0.0,  # not a p-value-based test
                severity=severity,
                detail={
                    "baseline_mae": round(bl_mae, 4),
                    "current_mae": round(cur_mae, 4),
                    "mae_change_pct": round(mae_change_pct, 4),
                },
            )
        )

        # KS test on residual distributions
        ks_stat, ks_p = ks_2samp(bl_residuals, cur_residuals)
        psi = self.compute_psi(bl_residuals, cur_residuals, n_bins=self.config.psi_n_bins)
        resid_severity = self._numeric_severity(ks_p, psi)

        results.append(
            DriftResult(
                feature="residuals",
                drift_type="concept_drift",
                test_name="KS + PSI on residuals",
                statistic=round(ks_stat, 6),
                p_value=round(ks_p, 6),
                severity=resid_severity,
                detail={"psi": round(psi, 6), "ks_statistic": round(ks_stat, 6)},
            )
        )

        return results

    # ------------------------------------------------------------------
    # 4. Prediction Drift
    # ------------------------------------------------------------------

    def detect_prediction_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        prediction_col: str = "norm_confirmed_remaining_end",
    ) -> List[DriftResult]:
        """Detect shift in model prediction output distribution."""
        if prediction_col not in baseline_df.columns or prediction_col not in current_df.columns:
            return []

        bl = baseline_df[prediction_col].dropna().values
        cur = current_df[prediction_col].dropna().values

        if len(bl) < 2 or len(cur) < 2:
            return []

        ks_stat, ks_p = ks_2samp(bl, cur)
        psi = self.compute_psi(bl, cur, n_bins=self.config.psi_n_bins)
        severity = self._numeric_severity(ks_p, psi)

        return [
            DriftResult(
                feature=prediction_col,
                drift_type="prediction_drift",
                test_name="KS + PSI",
                statistic=round(ks_stat, 6),
                p_value=round(ks_p, 6),
                severity=severity,
                detail={"psi": round(psi, 6), "ks_statistic": round(ks_stat, 6)},
            )
        ]

    # ------------------------------------------------------------------
    # 5. Data Quality Drift
    # ------------------------------------------------------------------

    def detect_data_quality_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Detect data quality issues: NaN rate changes, schema, cardinality."""
        results: List[DriftResult] = []
        results.extend(self._nan_rate_drift(baseline_df, current_df))
        results.extend(self._schema_drift(baseline_df, current_df))
        results.extend(self._cardinality_drift(baseline_df, current_df))
        return results

    def _nan_rate_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Check for NaN rate changes exceeding the threshold."""
        results: List[DriftResult] = []
        all_features = self.config.get_all_features()

        for feat in all_features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            bl_nan_rate = baseline_df[feat].isna().mean()
            cur_nan_rate = current_df[feat].isna().mean()
            change = abs(cur_nan_rate - bl_nan_rate)

            if change >= self.config.nan_rate_change_threshold:
                severity = "CRITICAL" if change >= 2 * self.config.nan_rate_change_threshold else "WARNING"
                results.append(
                    DriftResult(
                        feature=feat,
                        drift_type="data_quality",
                        test_name="NaN rate change",
                        statistic=round(change, 6),
                        p_value=0.0,
                        severity=severity,
                        detail={
                            "baseline_nan_rate": round(float(bl_nan_rate), 4),
                            "current_nan_rate": round(float(cur_nan_rate), 4),
                            "change_pp": round(float(change * 100), 2),
                        },
                    )
                )
        return results

    def _schema_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Check for missing columns or dtype changes."""
        results: List[DriftResult] = []
        all_features = self.config.get_all_features()

        for feat in all_features:
            if feat in baseline_df.columns and feat not in current_df.columns:
                results.append(
                    DriftResult(
                        feature=feat,
                        drift_type="data_quality",
                        test_name="Schema: missing column",
                        statistic=1.0,
                        p_value=0.0,
                        severity="CRITICAL",
                        detail={"issue": f"Column '{feat}' missing from current data"},
                    )
                )
            elif feat in baseline_df.columns and feat in current_df.columns:
                bl_dtype = str(baseline_df[feat].dtype)
                cur_dtype = str(current_df[feat].dtype)
                if bl_dtype != cur_dtype:
                    results.append(
                        DriftResult(
                            feature=feat,
                            drift_type="data_quality",
                            test_name="Schema: dtype change",
                            statistic=1.0,
                            p_value=0.0,
                            severity="WARNING",
                            detail={
                                "baseline_dtype": bl_dtype,
                                "current_dtype": cur_dtype,
                            },
                        )
                    )
        return results

    def _cardinality_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Check for >20% new categories in categorical features."""
        results: List[DriftResult] = []

        for feat in self.config.categorical_features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            bl_cats = set(baseline_df[feat].dropna().unique())
            cur_cats = set(current_df[feat].dropna().unique())

            if len(bl_cats) == 0:
                continue

            new_cats = cur_cats - bl_cats
            new_ratio = len(new_cats) / len(bl_cats)

            if new_ratio > self.config.cardinality_change_threshold:
                severity = "CRITICAL" if new_ratio > 2 * self.config.cardinality_change_threshold else "WARNING"
                results.append(
                    DriftResult(
                        feature=feat,
                        drift_type="data_quality",
                        test_name="Cardinality change",
                        statistic=round(new_ratio, 4),
                        p_value=0.0,
                        severity=severity,
                        detail={
                            "baseline_n_categories": len(bl_cats),
                            "current_n_categories": len(cur_cats),
                            "new_categories_count": len(new_cats),
                            "new_categories_ratio": round(new_ratio, 4),
                        },
                    )
                )
        return results
