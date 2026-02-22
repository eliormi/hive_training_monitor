"""Config-driven mapping from drift results to recommended actions."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.drift.config import DriftConfig
from src.drift.detectors import DriftResult


@dataclass
class RecommendedAction:
    """A single recommended action derived from drift results."""

    drift_type: str
    severity: str  # "WARNING" or "CRITICAL"
    action: str
    owner: str
    affected_features: List[str]


class ActionMapper:
    """Maps drift detection results to prioritized recommended actions.

    Groups multiple affected features under one action, deduplicates,
    and returns actions sorted by severity (CRITICAL first).
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    def map_actions(self, results: List[DriftResult]) -> List[RecommendedAction]:
        """Map a list of DriftResults to a list of RecommendedActions.

        Groups results by (drift_type, severity), looks up the action
        mapping from config, and aggregates affected features.

        Args:
            results: Drift detection results from all detectors.

        Returns:
            Sorted list of RecommendedActions (CRITICAL first, then WARNING).
        """
        # Group drifted results by (drift_type, severity)
        groups: Dict[tuple, List[str]] = {}
        for r in results:
            if not r.is_drifted:
                continue
            key = (r.drift_type, r.severity)
            groups.setdefault(key, []).append(r.feature)

        actions: List[RecommendedAction] = []
        for (drift_type, severity), features in groups.items():
            mapping = self.config.action_mappings.get((drift_type, severity))
            if mapping is not None:
                action_text = mapping["action"]
                owner = mapping["owner"]
            else:
                action_text = f"Investigate {drift_type} ({severity})"
                owner = "ML Engineer"

            actions.append(
                RecommendedAction(
                    drift_type=drift_type,
                    severity=severity,
                    action=action_text,
                    owner=owner,
                    affected_features=sorted(set(features)),
                )
            )

        # Sort: CRITICAL first, then WARNING, then alphabetically by drift_type
        severity_order = {"CRITICAL": 0, "WARNING": 1}
        actions.sort(key=lambda a: (severity_order.get(a.severity, 2), a.drift_type))
        return actions

    def get_overall_status(self, results: List[DriftResult]) -> str:
        """Determine overall pipeline status from all drift results.

        Returns:
            "HEALTHY", "WARNING", or "CRITICAL".
        """
        severities = {r.severity for r in results}
        if "CRITICAL" in severities:
            return "CRITICAL"
        if "WARNING" in severities:
            return "WARNING"
        return "HEALTHY"
