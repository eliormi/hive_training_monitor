# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project virtual environment:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
./.venv/bin/pip install -r requirements.txt
```

## Commands

Run the Streamlit dashboard:
```bash
source .venv/bin/activate && streamlit run dashboard_app.py
```

Run all tests:
```bash
./.venv/bin/pytest tests/
```

Run a single test file:
```bash
./.venv/bin/pytest tests/test_ensemble_regressor.py -v
./.venv/bin/pytest tests/test_data_split.py -v
```

## Development Rules

- All new functions must have type hints.
- Create a `.bak` copy before major file modifications.

## Architecture

This is a model performance analysis platform comparing a DSG ML model against an Agent baseline for trip-based predictions. It identifies performance drivers and failure modes across data segments.

### Data Flow

```
data/trip_scores.pkl + data/origin_line_leg_scores.pkl
    ↓
src/dashboard/data_loader.py  (loads, caches, filters)
    ↓
src/dashboard/metrics.py      (lift, OLL success rate, segment lift)
    ↓
src/dashboard/views/          (executive, segments, overfitting, graveyard)
    ↓
dashboard_app.py              (Streamlit entry point, 4-act layout)
```

### Key Modules

**`src/segment_analyzer.py` — `SegmentErrorAnalyzer`**
Identifies which feature segments are over-represented in a subset (e.g., error cases). Uses chi-square tests for categoricals and quantile binning (5 bins) for numerics. Returns lift, subset share, p-values, and counts.

**`src/models/ensemble_regressor.py` — `EnsembleRegressor`**
Scikit-learn-compatible regressor combining CatBoost, Linear Regression, Random Forest, KNN, SMA (same-day-of-week moving average over 8-week window), and TTM (Tiny Time Mixers, lazy-loaded from Hugging Face). Supports per-model feature subsets and customizable ensemble weights.

**`src/dashboard/data_loader.py`**
Loads and caches pickle files. Key filters:
- Valid trips: `confirmed_at_end > 0` AND `allocation_at_end_grouped > 50`
- Graveyard trips: inverse of valid, with death reasons
- `data_type` column splits rows into `train`/`val`/`test`

**`src/dashboard/metrics.py`**
- `calculate_global_metrics(df, target, model_col)` → `(agent_mean, model_mean, lift)`
- `calculate_oll_success_rate(df_oll)` → `(ratio, count, total)` — % of Origin-Line-Legs where Model > Agent
- `get_segment_lift(df, segment_col, target, model_col)` → ranked DataFrame

**`dashboard_app.py`**
Four-act Streamlit layout:
1. Executive summary (KPI cards, Plotly charts)
2. Scrollytelling narrative (sticky graphics, `src/dashboard/components/story_module.py`)
3. Deep dive — overfitting tab + segment lift tab (trade/line breakdown)
4. Graveyard — analysis of invalid/low-quality trips

**`synthesize_data.py`**
Generates synthetic datasets preserving original statistical distributions (multivariate KDE, NaN patterns, inter-column correlations). Used to enable safe testing without production data.

### Data Schema (from code inference)

`trip_scores.pkl` columns include: `data_type`, `confirmed_at_end`, `allocation_at_end_grouped`, `trip_score_6_weeks`, `model_type` (values: `Agent` / `DSG_model`), `trade`, `line`.

`origin_line_leg_scores.pkl` columns include: `diff_trip_score_6_weeks`, `origin`, `line`, `leg`.
