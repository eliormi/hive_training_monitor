"""
Synthetic data generator for trip_scores and origin_line_leg_scores.

Synthesizes new data that preserves the statistical distributions of the
originals — including NaN patterns, inter-column correlations, categorical
proportions, and structural constraints (paired DSG_model/Agent rows).

WHY: The original data is sensitive. Synthetic data allows safe sharing,
dashboard demos, and CI testing without exposing production data.
"""

import shutil
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
TRIP_SCORES_FILE = DATA_DIR / "trip_scores.pkl"
ORIGIN_SCORES_FILE = DATA_DIR / "origin_line_leg_scores.pkl"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _backup(path: Path) -> None:
    """Create a .bak copy before overwriting — per project rules."""
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, bak)
        print(f"  ✓ Backup created: {bak.name}")


def _kde_sample(
    series: pd.Series,
    n: int,
    rng: np.random.Generator,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """Draw *n* samples from a 1-D KDE fitted on *series* (NaN-safe)."""
    clean = series.dropna().values.astype(float)
    if len(clean) < 5:
        # Too few points for KDE — fall back to empirical resampling
        return rng.choice(clean, size=n, replace=True)

    try:
        kde = gaussian_kde(clean, bw_method="silverman")
        samples = kde.resample(n, seed=rng).flatten()
    except np.linalg.LinAlgError:
        # Singular matrix — constant column
        samples = np.full(n, clean[0])

    if clip_min is not None or clip_max is not None:
        samples = np.clip(samples, clip_min, clip_max)
    return samples


def _multivariate_kde_sample(
    df_subset: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    clip_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Draw *n* samples from a multivariate KDE fitted on *df_subset*.

    WHY multivariate: confirmed_at_end and mean_forecast_w1..w6 are
    correlated at ~0.97. Independent KDE would destroy that structure.
    """
    clean = df_subset.dropna().values.astype(float)
    cols = list(df_subset.columns)

    if clean.shape[0] < clean.shape[1] + 5:
        # Not enough data — fall back to row resampling
        idx = rng.choice(clean.shape[0], size=n, replace=True)
        result = pd.DataFrame(clean[idx], columns=cols)
    else:
        try:
            kde = gaussian_kde(clean.T, bw_method="silverman")
            samples = kde.resample(n, seed=rng)  # shape: (d, n)
            result = pd.DataFrame(samples.T, columns=cols)
        except np.linalg.LinAlgError:
            idx = rng.choice(clean.shape[0], size=n, replace=True)
            result = pd.DataFrame(clean[idx], columns=cols)

    if clip_bounds:
        for col, (lo, hi) in clip_bounds.items():
            if col in result.columns:
                result[col] = result[col].clip(lo, hi)

    return result


def _sample_categorical(
    series: pd.Series, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample from empirical categorical frequencies."""
    vc = series.value_counts(normalize=True)
    return rng.choice(vc.index, size=n, p=vc.values, replace=True)


def _inject_nans(
    arr: np.ndarray, nan_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Randomly set values to NaN at the given rate."""
    if nan_rate <= 0:
        return arr
    mask = rng.random(len(arr)) < nan_rate
    arr = arr.astype(float)
    arr[mask] = np.nan
    return arr


def _generate_synthetic_ids(
    n_trips: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate unique synthetic voy_id and trip_id values.

    WHY synthetic IDs: avoids leaking real voyage identifiers while keeping
    the format recognizable for downstream code that parses ID structure.
    """
    # Random 3-letter line codes
    lines = ["".join(rng.choice(list(string.ascii_uppercase), size=3)) for _ in range(50)]
    directions = ["E", "W", "N", "S"]

    voy_ids: List[str] = []
    trip_ids: List[str] = []

    for i in range(n_trips):
        line = lines[i % len(lines)]
        seq = f"{i // len(lines) + 100}"
        direction = directions[i % len(directions)]
        loc_code = "".join(rng.choice(list(string.ascii_uppercase), size=2 + rng.integers(1, 4)))
        voy = f"{line}-SYN-{seq}-{direction}"
        trip = f"{voy}-{loc_code}"
        voy_ids.append(voy)
        trip_ids.append(trip)

    return np.array(voy_ids), np.array(trip_ids)


# ---------------------------------------------------------------------------
# Trip Scores Synthesis
# ---------------------------------------------------------------------------
def synthesize_trip_scores(
    original: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Synthesize a new trip_scores DataFrame matching original distributions."""
    n_total = len(original)
    n_trips = n_total // 2  # Each trip has DSG_model + Agent rows

    print(f"\n{'='*60}")
    print(f"Synthesizing trip_scores: {n_total} rows ({n_trips} trips × 2 models)")
    print(f"{'='*60}")

    # --- 1. IDs (paired: each trip_id appears for DSG_model and Agent) ---
    voy_ids, trip_ids = _generate_synthetic_ids(n_trips, rng)
    # Duplicate for both model types
    syn_voy = np.concatenate([voy_ids, voy_ids])
    syn_trip = np.concatenate([trip_ids, trip_ids])
    syn_model = np.array(["DSG_model"] * n_trips + ["Agent"] * n_trips)

    synth = pd.DataFrame({
        "voy_id": syn_voy,
        "trip_id": syn_trip,
        "model_type": syn_model,
    })

    # --- 2. High-correlation numeric group (multivariate KDE) ---
    corr_cols = [
        "confirmed_at_end",
        "mean_forecast_w1", "mean_forecast_w2", "mean_forecast_w3",
        "mean_forecast_w4", "mean_forecast_w5", "mean_forecast_w6",
    ]
    # Separate synthesis for each model type to preserve per-model distributions
    for model_type in ["DSG_model", "Agent"]:
        mask_orig = original["model_type"] == model_type
        mask_syn = synth["model_type"] == model_type
        sub_orig = original.loc[mask_orig, corr_cols]

        clip_bounds = {c: (sub_orig[c].min(), sub_orig[c].max()) for c in corr_cols}
        sampled = _multivariate_kde_sample(sub_orig, n_trips, rng, clip_bounds)

        # confirmed_at_end is shared between both model types for the same trip
        # We set confirmed_at_end from DSG pass only; Agent pass reuses it
        if model_type == "DSG_model":
            confirmed_values = sampled["confirmed_at_end"].values
        else:
            sampled["confirmed_at_end"] = confirmed_values  # type: ignore[possibly-undefined]

        for col in corr_cols:
            nan_rate = sub_orig[col].isna().mean()
            vals = sampled[col].values.copy()
            vals = _inject_nans(vals, nan_rate, rng)
            synth.loc[mask_syn, col] = vals

    # --- 3. Score columns (independent KDE) ---
    score_cols = [
        "trip_score_4_weeks", "trip_score_6_weeks",
        "weekly_score_w1", "weekly_score_w2", "weekly_score_w3",
        "weekly_score_w4", "weekly_score_w5", "weekly_score_w6",
        "stability_score", "weeks_6_to_5_avg",
    ]
    for model_type in ["DSG_model", "Agent"]:
        mask_orig = original["model_type"] == model_type
        mask_syn = synth["model_type"] == model_type

        for col in score_cols:
            series = original.loc[mask_orig, col]
            nan_rate = series.isna().mean()
            vals = _kde_sample(
                series, n_trips, rng,
                clip_min=series.min(), clip_max=series.max(),
            )
            vals = _inject_nans(vals, nan_rate, rng)
            synth.loc[mask_syn, col] = vals

    # --- 4. Count columns (empirical integer resampling) ---
    count_cols = [
        "n_samples_w1", "n_samples_w2", "n_samples_w3",
        "n_samples_w4", "n_samples_w5", "n_samples_w6",
    ]
    for col in count_cols:
        nan_rate = original[col].isna().mean()
        vals = _kde_sample(
            original[col], n_total, rng,
            clip_min=original[col].min(), clip_max=original[col].max(),
        )
        vals = np.round(vals).astype(float)
        vals = _inject_nans(vals, nan_rate, rng)
        synth[col] = vals

    # --- 5. Boolean columns ---
    for col in ["complete_6", "complete_4", "valid_trips_mask"]:
        true_rate = original[col].mean()
        synth[col] = rng.random(n_total) < true_rate

    # --- 6. Categorical string columns ---
    for col in ["data_type"]:
        synth[col] = _sample_categorical(original[col], n_total, rng)

    # Structural string cols — sampled per-trip then duplicated
    for col in ["line", "leg", "location_origin", "trade"]:
        per_trip = _sample_categorical(
            original.loc[original["model_type"] == "DSG_model", col], n_trips, rng
        )
        synth[col] = np.concatenate([per_trip, per_trip])

    # Derived string composites
    synth["line_leg"] = synth["line"] + "_" + synth["leg"]
    synth["origin_line_leg"] = (
        synth["location_origin"] + "_" + synth["line"] + "_" + synth["leg"]
    )

    # --- 7. Category bin columns (derived from confirmed_at_end) ---
    # Extract original bin edges
    for col in ["confirmed_at_end_grouped", "zim_teu_capacity_grouped", "allocation_at_end_grouped"]:
        cats = original[col].cat.categories
        synth[col] = pd.Categorical(
            _sample_categorical(original[col], n_total, rng),
            categories=cats,
            ordered=True,
        )

    # --- 8. Ensure column order matches original ---
    synth = synth[original.columns]

    print(f"  ✓ Synthesized {len(synth)} rows")
    return synth


# ---------------------------------------------------------------------------
# Origin-Line-Leg Scores Synthesis
# ---------------------------------------------------------------------------
def synthesize_origin_scores(
    original: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Synthesize a new origin_line_leg_scores DataFrame."""
    n = len(original)

    print(f"\n{'='*60}")
    print(f"Synthesizing origin_line_leg_scores: {n} rows")
    print(f"{'='*60}")

    synth = pd.DataFrame(index=range(n))

    # --- 1. origin_line_leg (unique synthetic labels) ---
    synth["origin_line_leg"] = [f"SYN_{i:03d}" for i in range(n)]

    # --- 2. Boolean flag columns ---
    bool_cols = [
        "final_decision_filter", "score_or_effect_ind",
        "trip_count_gt_min", "flag_big_diff_total",
        "flag_big_diff_w5", "flag_big_diff_w6", "flag_generally_good",
    ]
    for col in bool_cols:
        true_rate = original[col].mean()
        synth[col] = rng.random(n) < true_rate

    # --- 3. Core numeric columns (KDE) ---
    # These are the "base" columns from which diffs are derived
    base_numeric_cols = [
        "confirmed_at_end_sum", "trips_count", "avg_confirmed",
    ]
    for col in base_numeric_cols:
        vals = _kde_sample(
            original[col], n, rng,
            clip_min=original[col].min(), clip_max=original[col].max(),
        )
        if col == "trips_count":
            vals = np.round(vals).astype(int)
        synth[col] = vals

    # --- 4. Score pairs (DSG_model mean, Agent mean → diff derived) ---
    score_pairs = [
        ("trip_score_6_weeks_mean_DSG_model", "trip_score_6_weeks_mean_Agent", "diff_trip_score_6_weeks"),
        ("weeks_6_to_5_avg_mean_DSG_model", "weeks_6_to_5_avg_mean_Agent", "diff_weeks_6_to_5"),
        ("weekly_score_w6_mean_DSG_model", "weekly_score_w6_mean_Agent", "diff_w6"),
        ("weekly_score_w5_mean_DSG_model", "weekly_score_w5_mean_Agent", "diff_w5"),
        ("stability_score_mean_DSG_model", "stability_score_mean_Agent", "diff_stability"),
    ]
    for dsg_col, agent_col, diff_col in score_pairs:
        # Sample DSG and Agent from bivariate KDE to preserve their correlation
        pair_df = original[[dsg_col, agent_col]]
        sampled = _multivariate_kde_sample(
            pair_df, n, rng,
            clip_bounds={
                dsg_col: (original[dsg_col].min(), original[dsg_col].max()),
                agent_col: (original[agent_col].min(), original[agent_col].max()),
            },
        )
        synth[dsg_col] = sampled[dsg_col].values
        synth[agent_col] = sampled[agent_col].values
        # WHY derived: diff = DSG - Agent — preserves the semantic relationship
        synth[diff_col] = synth[dsg_col] - synth[agent_col]

    # --- 5. Ensure column order matches original ---
    synth = synth[original.columns]
    # Preserve original index style
    synth.index = original.index

    print(f"  ✓ Synthesized {len(synth)} rows")
    return synth


# ---------------------------------------------------------------------------
# Comparison Report
# ---------------------------------------------------------------------------
def print_comparison(
    original: pd.DataFrame, synthetic: pd.DataFrame, label: str
) -> None:
    """Print a side-by-side distribution comparison."""
    print(f"\n{'='*60}")
    print(f"Distribution Comparison: {label}")
    print(f"{'='*60}")

    numeric_cols = original.select_dtypes(include=[np.number]).columns

    rows = []
    for col in numeric_cols:
        orig_s = original[col]
        syn_s = synthetic[col]
        rows.append({
            "column": col,
            "orig_mean": f"{orig_s.mean():.2f}",
            "syn_mean": f"{syn_s.mean():.2f}",
            "orig_std": f"{orig_s.std():.2f}",
            "syn_std": f"{syn_s.std():.2f}",
            "orig_min": f"{orig_s.min():.2f}",
            "syn_min": f"{syn_s.min():.2f}",
            "orig_max": f"{orig_s.max():.2f}",
            "syn_max": f"{syn_s.max():.2f}",
            "orig_nan": int(orig_s.isna().sum()),
            "syn_nan": int(syn_s.isna().sum()),
        })

    report = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(report.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Load originals
    print("Loading original data...")
    trip_orig = pd.read_pickle(TRIP_SCORES_FILE)
    origin_orig = pd.read_pickle(ORIGIN_SCORES_FILE)
    print(f"  trip_scores: {trip_orig.shape}")
    print(f"  origin_line_leg_scores: {origin_orig.shape}")

    # Create backups
    print("\nCreating backups...")
    _backup(TRIP_SCORES_FILE)
    _backup(ORIGIN_SCORES_FILE)

    # Synthesize
    trip_synth = synthesize_trip_scores(trip_orig, rng)
    origin_synth = synthesize_origin_scores(origin_orig, rng)

    # Save
    print("\nSaving synthetic data...")
    trip_synth.to_pickle(TRIP_SCORES_FILE)
    origin_synth.to_pickle(ORIGIN_SCORES_FILE)
    print(f"  ✓ Saved {TRIP_SCORES_FILE}")
    print(f"  ✓ Saved {ORIGIN_SCORES_FILE}")

    # Comparison report
    print_comparison(trip_orig, trip_synth, "trip_scores")
    print_comparison(origin_orig, origin_synth, "origin_line_leg_scores")

    print("\n✅ Done! Synthetic data generated and saved.")


if __name__ == "__main__":
    main()
