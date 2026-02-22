# Drift Detection Pipeline — Implementation Plan

## Context

The model is retrained monthly on shipping/logistics trip data. Currently there's no way to detect when incoming data has shifted away from what the model was trained on, when the target variable distribution changes, or when model accuracy degrades. This pipeline adds automated drift detection with statistical rigor, using three seasonal baselines to distinguish genuine drift from seasonal patterns.

## Answers to the User's Core Questions

1. **Baseline strategy**: Three rolling baselines — (1) last training data, (2) same-month-last-year data, (3) unified verdict from both. Stored as pickle snapshots in `data/baselines/`.
2. **Frequency**: Baseline created at each monthly retrain. Drift checks can run at any time against the stored baselines.
3. **Statistical decisions**: KS test + PSI for numerics, chi-square for categoricals, MAE/RMSE comparison for concept drift. Dual threshold (statistical significance AND practical significance via PSI) to avoid false alarms on large datasets.
4. **Action mapping**: A config-driven dictionary mapping each drift scenario + severity to a recommended action and owner.

## New Files to Create

```
src/drift/
├── __init__.py
├── config.py              # All thresholds, feature lists, action mappings
├── baseline_manager.py    # Save/load/version baseline snapshots (3 types)
├── detectors.py           # Statistical drift detection (5 drift types)
├── action_mapper.py       # Drift results → recommended actions
├── report.py              # DriftReport dataclass, JSON serialization
└── runner.py              # Orchestrator: load data, run detectors, produce report

tests/
├── test_drift_detectors.py
├── test_drift_baseline.py
└── test_drift_runner.py

run_drift_check.py             # CLI entry point
src/dashboard/views/drift.py   # Streamlit dashboard tab
```

## Files to Modify

- `dashboard_app.py` — add "Drift Monitor" tab to ACT 3 (line 106)

## Existing Code to Reuse

- `scipy.stats.chi2_contingency` — already used in `src/segment_analyzer.py:94`
- `scipy.stats.ks_2samp` — imported but commented out at `src/segment_analyzer.py:118`
- `pd.qcut` with `n_bins=5, duplicates='drop'` — used in `src/segment_analyzer.py:125`
- `src/dashboard/data_loader.py` — `load_trip_data()`, `get_filtered_data()`, `get_valid_trips()`
- `src/dashboard/metrics.py` — `calculate_global_metrics()` for concept drift performance comparison

## Design Details

### 1. Three-Baseline System (`baseline_manager.py`)

Each baseline is a pickle of the training partition + metadata JSON:

| Baseline | What it stores | Purpose |
|----------|---------------|---------|
| **Last Train** | Previous month's `data_type=='train'` partition | Detect month-over-month drift |
| **Last Year** | Same calendar month from previous year | Detect deviation from seasonal norms |
| **Unified** | Not a separate file — a combined verdict | If drifted vs last train BUT stable vs last year → seasonal, not drift. If drifted vs both → real drift |

Storage: `data/baselines/baseline_{YYYY-MM-DD}_{type}.pkl` (type = `last_train` or `yearly`)

Metadata: `data/baselines/baseline_meta.json` tracks active baselines and history. `.bak` copies made before overwriting (per project convention from `synthesize_data.py`).

### 2. Five Drift Types (`detectors.py`)

**Data drift** (covariate shift):
- Numeric features → KS test (`scipy.stats.ks_2samp`) + PSI
- Categorical features → chi-square (`scipy.stats.chi2_contingency`)
- Boolean features → chi-square on 2x2 table
- Decision: p-value < 0.05 AND PSI ≥ 0.1 = warning, PSI ≥ 0.2 = critical

**Label drift** (target shift):
- KS test + PSI on `trip_score_6_weeks` (filtered to single `model_type`)
- Same dual-threshold decision rule

**Concept drift** (accuracy degradation):
- Compare MAE/RMSE between baseline and current using `confirmed_at_end` as ground truth
- KS test on residual distributions (actual - predicted)
- MAE increase ≥ 10% = warning, ≥ 25% = critical

**Prediction drift** (model output shift):
- KS test + PSI on model predictions (`trip_score_6_weeks` where `model_type == 'DSG_model'`)

**Data quality drift**:
- NaN rate changes per feature (threshold: 5 percentage point change)
- Schema violations (missing columns, dtype changes)
- Cardinality changes in categoricals (>20% new categories)

### 3. Unified Verdict Logic

For each feature/metric, the runner compares against both baselines and produces:

| vs Last Train | vs Last Year | Unified Verdict |
|---------------|-------------|-----------------|
| No drift | No drift | **HEALTHY** — stable |
| Drift | No drift | **SEASONAL** — expected seasonal change, not actionable |
| No drift | Drift | **YEAR-SHIFT** — diverging from historical pattern, monitor |
| Drift | Drift | **DRIFT** — genuine drift, action required |

This is the key differentiator: seasonal shipping patterns (e.g., holiday peaks) won't trigger false alarms because the same-month-last-year baseline provides context.

### 4. Configuration (`config.py`)

A single `DriftConfig` dataclass with all thresholds, feature lists (numeric/categorical/boolean), and action mappings. Serializable to JSON. No external config files (matches project convention). Key defaults:
- `ks_alpha = 0.05`, `chi2_alpha = 0.05`
- `psi_warning = 0.1`, `psi_critical = 0.2`
- `performance_degradation_warning_pct = 0.10`, `performance_degradation_critical_pct = 0.25`
- `nan_rate_change_threshold = 0.05`
- Feature lists derived from the known 39 columns in `trip_scores.pkl`

### 5. Action Mapping (`action_mapper.py`)

Config-driven dictionary: `{scenario_key: {severity, action, owner}}`. The mapper aggregates drift results, groups multiple affected features under one action, deduplicates, and returns a prioritized action list sorted by severity (CRITICAL first). Overall status: HEALTHY / WARNING / CRITICAL.

### 6. Report (`report.py`)

`DriftReport` dataclass serializable to JSON. Saved to `data/baselines/drift_report_{timestamp}.json` for historical tracking. Contains: all results per drift type, per-baseline verdicts, unified verdicts, recommended actions, summary stats. Also has `print_summary()` for CLI output.

### 7. CLI (`run_drift_check.py`)

```
python run_drift_check.py --create-baseline --label "Feb 2025 retrain"
python run_drift_check.py                          # check against latest baselines
python run_drift_check.py --baseline 2025-01-15    # specific baseline date
python run_drift_check.py --verbose                # detailed output
```

Exit code 0 = HEALTHY, 1 = WARNING/CRITICAL.

### 8. Dashboard Integration (`src/dashboard/views/drift.py`)

New tab in ACT 3 of `dashboard_app.py` (line 106). Shows: overall status badge, per-feature drift table with test statistics, unified verdict column, and action recommendations.

## Implementation Order

1. `src/drift/__init__.py` + `config.py` — no dependencies
2. `baseline_manager.py` — depends on config
3. `detectors.py` — core statistical engine, depends on config
4. `tests/test_drift_detectors.py` — alongside detectors
5. `action_mapper.py` — depends on detector result types
6. `report.py` — depends on detector + action types
7. `runner.py` — orchestrator, depends on all above
8. `tests/test_drift_baseline.py` + `tests/test_drift_runner.py`
9. `run_drift_check.py` — CLI wrapper
10. `src/dashboard/views/drift.py` + modify `dashboard_app.py`

## Verification

1. **Unit tests**: `./.venv/bin/pytest tests/test_drift_detectors.py tests/test_drift_baseline.py tests/test_drift_runner.py -v`
2. **Smoke test**: Create baseline from current data → run drift check (should be HEALTHY) → modify data slightly → run again (should detect drift)
3. **All existing tests still pass**: `./.venv/bin/pytest tests/ -v`
4. **Dashboard**: `streamlit run dashboard_app.py` → navigate to Drift Monitor tab
5. **CLI exit codes**: Verify `run_drift_check.py` returns 0 for healthy, 1 for drift
