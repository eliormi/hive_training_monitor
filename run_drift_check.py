"""CLI entry point for drift detection.

Usage:
    python run_drift_check.py --create-baseline --label "Feb 2025 retrain"
    python run_drift_check.py                    # check against latest baseline
    python run_drift_check.py --verbose          # detailed output
    python run_drift_check.py --baseline yearly  # specify baseline type
"""

import argparse
import sys
from typing import List

from src.drift.config import DriftConfig
from src.drift.runner import DriftRunner


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run drift detection on training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_drift_check.py --create-baseline --label 'initial'\n"
            "  python run_drift_check.py\n"
            "  python run_drift_check.py --verbose\n"
        ),
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create a new baseline snapshot from current training data.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Human-readable label for the baseline (used with --create-baseline).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["last_train", "yearly"],
        default="last_train",
        help="Baseline type to create (default: last_train).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed drift report.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    """Run drift detection or create a baseline.

    Returns:
        Exit code: 0 = HEALTHY, 1 = WARNING or CRITICAL.
    """
    args = parse_args(argv)
    config = DriftConfig()
    runner = DriftRunner(config)

    if args.create_baseline:
        print(f"Loading data from {config.data_path}...")
        df = runner.load_data()
        path = runner.create_baseline(df=df, label=args.label, baseline_type=args.baseline)
        print(f"Baseline saved to {path}")
        if args.label:
            print(f"Label: {args.label}")
        return 0

    # Check if baseline exists
    if not runner.baseline_manager.has_baseline("last_train"):
        print("ERROR: No baseline found. Run with --create-baseline first.")
        return 1

    print(f"Loading data from {config.data_path}...")
    report = runner.run()

    if args.verbose:
        report.print_summary()
    else:
        print(f"Overall Status: {report.overall_status}")
        drifted = {k: v for k, v in report.unified_verdicts.items() if v != "HEALTHY"}
        if drifted:
            print(f"Drifted features: {len(drifted)}")
            for feat, verdict in sorted(drifted.items()):
                print(f"  {feat}: {verdict}")
        else:
            print("All features are stable.")

    if report.overall_status == "HEALTHY":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
