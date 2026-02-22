"""Drift detection orchestrator: load data, run detectors, produce report."""

from typing import Dict, List, Optional

import pandas as pd

from src.drift.action_mapper import ActionMapper
from src.drift.baseline_manager import BaselineManager
from src.drift.config import DriftConfig
from src.drift.detectors import DriftDetector, DriftResult
from src.drift.report import DriftReport


class DriftRunner:
    """Orchestrates the full drift detection pipeline.

    1. Loads the parquet data and splits by data_type.
    2. Compares current data against both baselines (last_train, yearly).
    3. Computes unified verdicts per feature.
    4. Maps results to recommended actions.
    5. Produces a DriftReport.
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config = config or DriftConfig()
        self.baseline_manager = BaselineManager(self.config)
        self.detector = DriftDetector(self.config)
        self.action_mapper = ActionMapper(self.config)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load the observations parquet file."""
        path = path or self.config.data_path
        return pd.read_parquet(path)

    def get_train_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to training rows only."""
        return df[df[self.config.data_type_column] == self.config.train_value].copy()

    # ------------------------------------------------------------------
    # Baseline creation
    # ------------------------------------------------------------------

    def create_baseline(
        self,
        df: Optional[pd.DataFrame] = None,
        label: str = "",
        baseline_type: str = "last_train",
    ) -> str:
        """Create and save a baseline from the training partition.

        Args:
            df: Optional pre-loaded DataFrame. If None, loads from config.data_path.
            label: Human-readable label for the baseline.
            baseline_type: 'last_train' or 'yearly'.

        Returns:
            Path to the saved baseline file.
        """
        if df is None:
            df = self.load_data()
        train_df = self.get_train_partition(df)
        return self.baseline_manager.save_baseline(train_df, baseline_type=baseline_type, label=label)

    # ------------------------------------------------------------------
    # Run all detectors against one baseline
    # ------------------------------------------------------------------

    def _run_detectors(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> List[DriftResult]:
        """Run all five drift detectors against a single baseline."""
        results: List[DriftResult] = []
        results.extend(self.detector.detect_data_drift(baseline_df, current_df))
        results.extend(self.detector.detect_label_drift(baseline_df, current_df))
        results.extend(self.detector.detect_concept_drift(baseline_df, current_df))
        results.extend(self.detector.detect_prediction_drift(baseline_df, current_df))
        results.extend(self.detector.detect_data_quality_drift(baseline_df, current_df))
        return results

    # ------------------------------------------------------------------
    # Unified verdict
    # ------------------------------------------------------------------

    @staticmethod
    def compute_unified_verdict(
        train_results: List[DriftResult],
        yearly_results: List[DriftResult],
    ) -> Dict[str, str]:
        """Compute unified verdict per feature from both baseline comparisons.

        Logic:
        - No drift vs both        -> HEALTHY
        - Drift vs train only     -> SEASONAL
        - Drift vs yearly only    -> YEAR-SHIFT
        - Drift vs both           -> DRIFT
        """
        train_drifted = {r.feature for r in train_results if r.is_drifted}
        yearly_drifted = {r.feature for r in yearly_results if r.is_drifted}

        all_features = {r.feature for r in train_results} | {r.feature for r in yearly_results}
        verdicts: Dict[str, str] = {}

        for feat in sorted(all_features):
            in_train = feat in train_drifted
            in_yearly = feat in yearly_drifted

            if in_train and in_yearly:
                verdicts[feat] = "DRIFT"
            elif in_train and not in_yearly:
                verdicts[feat] = "SEASONAL"
            elif not in_train and in_yearly:
                verdicts[feat] = "YEAR-SHIFT"
            else:
                verdicts[feat] = "HEALTHY"

        return verdicts

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        current_df: Optional[pd.DataFrame] = None,
        current_partition: str = "test",
    ) -> DriftReport:
        """Run the full drift detection pipeline.

        Args:
            current_df: Optional pre-loaded current data. If None, loads from
                config.data_path and filters to ``current_partition``.
            current_partition: Which data_type partition to treat as "current"
                (default "test").

        Returns:
            A populated DriftReport.
        """
        # Load current data
        if current_df is None:
            full_df = self.load_data()
            current_df = full_df[full_df[self.config.data_type_column] == current_partition].copy()

        # Load baselines
        train_baseline = self.baseline_manager.load_baseline("last_train")
        yearly_baseline = self.baseline_manager.load_baseline("yearly")

        report = DriftReport()

        # Run against last_train baseline
        if train_baseline is not None:
            report.results_vs_train = self._run_detectors(train_baseline, current_df)

        # Run against yearly baseline
        if yearly_baseline is not None:
            report.results_vs_yearly = self._run_detectors(yearly_baseline, current_df)

        # Unified verdicts (only if both baselines exist)
        if train_baseline is not None and yearly_baseline is not None:
            report.unified_verdicts = self.compute_unified_verdict(
                report.results_vs_train, report.results_vs_yearly
            )
        elif train_baseline is not None:
            # Only train baseline: report raw results as verdicts
            report.unified_verdicts = {
                r.feature: ("DRIFT" if r.is_drifted else "HEALTHY")
                for r in report.results_vs_train
            }

        # Actions from all results
        all_results = report.results_vs_train + report.results_vs_yearly
        report.actions = self.action_mapper.map_actions(all_results)
        report.overall_status = self.action_mapper.get_overall_status(all_results)

        report.compute_summary_stats()
        return report
