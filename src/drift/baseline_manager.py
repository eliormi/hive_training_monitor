"""Baseline snapshot management: save, load, and version baseline data."""

import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.drift.config import DriftConfig


class BaselineManager:
    """Manages baseline snapshots for drift detection.

    Supports two baseline types:
    - last_train: previous month's training data partition
    - yearly: same calendar month from the previous year

    Baselines are stored as pickle files with a JSON metadata index.
    """

    BASELINE_TYPES = ("last_train", "yearly")

    def __init__(self, config: DriftConfig) -> None:
        self.config = config
        self.baseline_dir = Path(config.baseline_dir)
        self.meta_path = self.baseline_dir / "baseline_meta.json"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create baseline directory if it does not exist."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def _baseline_filename(self, date_str: str, baseline_type: str) -> str:
        """Return the pickle filename for a baseline."""
        return f"baseline_{date_str}_{baseline_type}.pkl"

    def _make_bak(self, path: Path) -> None:
        """Create a .bak copy of the file if it exists (project convention)."""
        if path.exists():
            bak_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(str(path), str(bak_path))

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> Dict:
        """Load the metadata JSON file, or return an empty structure."""
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return {"active": {}, "history": []}

    def _save_meta(self, meta: Dict) -> None:
        """Save the metadata JSON file (with .bak copy first)."""
        self._make_bak(self.meta_path)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_baseline(
        self,
        df: pd.DataFrame,
        baseline_type: str,
        label: str = "",
        date: Optional[datetime] = None,
    ) -> str:
        """Save a DataFrame as a baseline snapshot.

        Args:
            df: The baseline data (typically the training partition).
            baseline_type: One of 'last_train' or 'yearly'.
            label: Optional human-readable label (e.g. "Feb 2025 retrain").
            date: Override date for the snapshot; defaults to now.

        Returns:
            The path to the saved pickle file.
        """
        if baseline_type not in self.BASELINE_TYPES:
            raise ValueError(
                f"baseline_type must be one of {self.BASELINE_TYPES}, got '{baseline_type}'"
            )

        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        filename = self._baseline_filename(date_str, baseline_type)
        filepath = self.baseline_dir / filename

        # .bak before overwriting
        self._make_bak(filepath)

        with open(filepath, "wb") as f:
            pickle.dump(df, f)

        # Update metadata
        meta = self._load_meta()
        meta["active"][baseline_type] = {
            "filename": filename,
            "date": date_str,
            "label": label,
            "n_rows": len(df),
            "created_at": datetime.now().isoformat(),
        }
        meta["history"].append(
            {
                "filename": filename,
                "baseline_type": baseline_type,
                "date": date_str,
                "label": label,
                "n_rows": len(df),
                "created_at": datetime.now().isoformat(),
            }
        )
        self._save_meta(meta)

        return str(filepath)

    def load_baseline(self, baseline_type: str) -> Optional[pd.DataFrame]:
        """Load the currently active baseline for the given type.

        Args:
            baseline_type: One of 'last_train' or 'yearly'.

        Returns:
            The baseline DataFrame, or None if no baseline is stored.
        """
        if baseline_type not in self.BASELINE_TYPES:
            raise ValueError(
                f"baseline_type must be one of {self.BASELINE_TYPES}, got '{baseline_type}'"
            )

        meta = self._load_meta()
        active = meta.get("active", {}).get(baseline_type)
        if active is None:
            return None

        filepath = self.baseline_dir / active["filename"]
        if not filepath.exists():
            return None

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def load_baseline_by_date(self, date_str: str, baseline_type: str) -> Optional[pd.DataFrame]:
        """Load a specific baseline by date string and type.

        Args:
            date_str: Date in YYYY-MM-DD format.
            baseline_type: One of 'last_train' or 'yearly'.

        Returns:
            The baseline DataFrame, or None if not found.
        """
        filename = self._baseline_filename(date_str, baseline_type)
        filepath = self.baseline_dir / filename
        if not filepath.exists():
            return None

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def list_baselines(self) -> List[Dict]:
        """Return the history of all saved baselines."""
        meta = self._load_meta()
        return meta.get("history", [])

    def get_active_baselines(self) -> Dict:
        """Return the currently active baselines."""
        meta = self._load_meta()
        return meta.get("active", {})

    def has_baseline(self, baseline_type: str) -> bool:
        """Check whether an active baseline exists for the given type."""
        meta = self._load_meta()
        active = meta.get("active", {}).get(baseline_type)
        if active is None:
            return False
        filepath = self.baseline_dir / active["filename"]
        return filepath.exists()
