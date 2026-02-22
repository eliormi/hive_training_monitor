"""Drift detection configuration with thresholds, feature lists, and action mappings."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Default numeric features for drift monitoring
DEFAULT_NUMERIC_FEATURES: List[str] = [
    "cumsum_fresh_teu",
    "zim_teu_capacity",
    "confirmed_at_end",
    "confirmed_remaining_end",
    "norm_confirmed_remaining_end",
    "forecast_teu",
    "allocation_teu",
    "fresh_teu",
    "norm_cumsum_fresh_teu",
    "norm_cumsum_fresh_confirmed_without_canceled",
    "norm_cumsum_fresh_unconfirmed_teu",
    "norm_cumsum_rollover_confirmed_teu",
    "norm_cumsum_fresh_non_rollover_confirmed",
    "norm_cumsum_canceled_fresh_teu",
    "norm_cumsum_rollover_canceled_teu",
    "norm_cumsum_rollover_unconfirmed_teu",
    "norm_cumsum_current_rollover_confirmed_teu",
    "norm_cumsum_current_rollover_canceled_teu",
    "norm_cumsum_current_rollover_unconfirmed_teu",
    "gate_in",
    "gate_out_at_end",
    "gate_in_at_end",
    "norm_gate_in_at_end",
    "gate_in_remaining_end",
    "norm_gate_in_remaining_end",
    "days_from_departure",
    "norm_allocation",
    "norm_confirmed",
    "norm_confirmed_sum_origins",
    "norm_allocation_sum_origins",
    "norm_rollover_teu",
    "ratio_origin_confirmed",
    "ratio_allocation_confirmed",
    "norm_empty_picked_up",
    "norm_gate_in",
    "norm_gross_weight",
]

# Default categorical features for drift monitoring
DEFAULT_CATEGORICAL_FEATURES: List[str] = [
    "line",
    "trade",
    "vessel",
    "leg",
    "location_origin",
    "port",
    "leg_type",
    "cn_holiday_name",
    "cn_holiday_duration_category",
    "us_holiday_name",
    "country",
]

# Default label column
DEFAULT_LABEL_COLUMN: str = "norm_confirmed_remaining_end"

# Default action mappings: (drift_type, severity) -> (action, owner)
DEFAULT_ACTION_MAPPINGS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("data_drift", "CRITICAL"): {
        "action": "Investigate feature distributions and consider retraining",
        "owner": "ML Engineer",
    },
    ("data_drift", "WARNING"): {
        "action": "Monitor feature distributions closely",
        "owner": "ML Engineer",
    },
    ("label_drift", "CRITICAL"): {
        "action": "Retrain model with updated label distribution",
        "owner": "ML Engineer",
    },
    ("label_drift", "WARNING"): {
        "action": "Monitor label distribution trend",
        "owner": "ML Engineer",
    },
    ("concept_drift", "CRITICAL"): {
        "action": "Immediate model retraining required",
        "owner": "ML Engineer",
    },
    ("concept_drift", "WARNING"): {
        "action": "Schedule model retraining",
        "owner": "ML Engineer",
    },
    ("prediction_drift", "CRITICAL"): {
        "action": "Investigate model output shift and retrain",
        "owner": "ML Engineer",
    },
    ("prediction_drift", "WARNING"): {
        "action": "Monitor model prediction distributions",
        "owner": "ML Engineer",
    },
    ("data_quality", "CRITICAL"): {
        "action": "Fix data pipeline issues before retraining",
        "owner": "Data Engineer",
    },
    ("data_quality", "WARNING"): {
        "action": "Investigate data quality changes",
        "owner": "Data Engineer",
    },
}


@dataclass
class DriftConfig:
    """Configuration for drift detection pipeline.

    Holds all thresholds, feature lists, and action mappings needed
    by detectors, the action mapper, and the runner.
    """

    # Statistical test thresholds
    ks_alpha: float = 0.05
    chi2_alpha: float = 0.05

    # PSI thresholds
    psi_warning: float = 0.1
    psi_critical: float = 0.2

    # Performance degradation thresholds (concept drift)
    performance_degradation_warning_pct: float = 0.10
    performance_degradation_critical_pct: float = 0.25

    # Data quality thresholds
    nan_rate_change_threshold: float = 0.05
    cardinality_change_threshold: float = 0.20

    # PSI bin count
    psi_n_bins: int = 10

    # Feature lists
    numeric_features: List[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC_FEATURES))
    categorical_features: List[str] = field(default_factory=lambda: list(DEFAULT_CATEGORICAL_FEATURES))

    # Label column
    label_column: str = DEFAULT_LABEL_COLUMN

    # Data type column and train value
    data_type_column: str = "data_type"
    train_value: str = "train"

    # Parquet data path
    data_path: str = "data/synthetic_observations.parquet"

    # Baseline storage directory
    baseline_dir: str = "data/baselines"

    # Action mappings
    action_mappings: Dict[Tuple[str, str], Dict[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_ACTION_MAPPINGS)
    )

    def get_all_features(self) -> List[str]:
        """Return the combined list of numeric and categorical features."""
        return self.numeric_features + self.categorical_features
