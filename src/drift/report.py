"""Drift report: dataclass with JSON serialization and summary printing."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.drift.action_mapper import RecommendedAction
from src.drift.detectors import DriftResult


@dataclass
class DriftReport:
    """Full drift detection report with results, verdicts, and actions.

    Attributes:
        timestamp: When the report was generated.
        overall_status: HEALTHY, WARNING, or CRITICAL.
        results_vs_train: Drift results compared to last_train baseline.
        results_vs_yearly: Drift results compared to yearly baseline.
        unified_verdicts: Per-feature unified verdict (HEALTHY/SEASONAL/YEAR-SHIFT/DRIFT).
        actions: Prioritized recommended actions.
        summary_stats: Aggregate counts by drift type and severity.
    """

    timestamp: str = ""
    overall_status: str = "HEALTHY"
    results_vs_train: List[DriftResult] = field(default_factory=list)
    results_vs_yearly: List[DriftResult] = field(default_factory=list)
    unified_verdicts: Dict[str, str] = field(default_factory=dict)
    actions: List[RecommendedAction] = field(default_factory=list)
    summary_stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def compute_summary_stats(self) -> None:
        """Populate summary_stats from unified verdicts."""
        counts: Dict[str, int] = {"HEALTHY": 0, "SEASONAL": 0, "YEAR-SHIFT": 0, "DRIFT": 0}
        for verdict in self.unified_verdicts.values():
            counts[verdict] = counts.get(verdict, 0) + 1
        self.summary_stats = counts

    def to_dict(self) -> Dict:
        """Serialize the report to a plain dictionary (JSON-safe)."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "results_vs_train": [asdict(r) for r in self.results_vs_train],
            "results_vs_yearly": [asdict(r) for r in self.results_vs_yearly],
            "unified_verdicts": self.unified_verdicts,
            "actions": [asdict(a) for a in self.actions],
            "summary_stats": self.summary_stats,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, directory: str) -> str:
        """Save the report as a JSON file in the given directory.

        Filename: drift_report_{timestamp}.json (colons replaced for filesystem safety).

        Returns:
            The path to the saved file.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_ts = self.timestamp.replace(":", "-")
        filepath = dir_path / f"drift_report_{safe_ts}.json"
        filepath.write_text(self.to_json())
        return str(filepath)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"=== Drift Report ({self.timestamp}) ===")
        print(f"Overall Status: {self.overall_status}")
        print()

        if self.summary_stats:
            print("Unified Verdict Summary:")
            for verdict, count in self.summary_stats.items():
                if count > 0:
                    print(f"  {verdict}: {count} feature(s)")
            print()

        # Print drifted features from unified verdicts
        drifted = {k: v for k, v in self.unified_verdicts.items() if v != "HEALTHY"}
        if drifted:
            print("Features with drift signals:")
            for feat, verdict in sorted(drifted.items()):
                print(f"  {feat}: {verdict}")
            print()

        if self.actions:
            print("Recommended Actions:")
            for i, a in enumerate(self.actions, 1):
                print(f"  {i}. [{a.severity}] {a.action}")
                print(f"     Owner: {a.owner}")
                print(f"     Affected: {', '.join(a.affected_features)}")
            print()

        if not drifted and not self.actions:
            print("All features are stable. No action required.")
