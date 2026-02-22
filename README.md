# Hive Training Monitor

> Model performance analysis platform comparing a DSG ML model against an Agent baseline for trip-based predictions.

## Overview

- Compares **DSG model** vs **Agent baseline** across trip-level and origin-line-leg-level metrics
- Identifies performance drivers, failure modes, and overfitting signals across data segments
- Interactive Streamlit dashboard with executive KPIs, narrative insights, and deep-dive analysis
- Synthetic data generation for safe sharing and CI testing

## Quick Start

### Prerequisites
- Python 3.13+
- pip

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run dashboard_app.py
```

### Run Tests
```bash
pytest tests/
```

## Architecture

### Data Flow
```
data/trip_scores.pkl + data/origin_line_leg_scores.pkl
    |
    v
src/dashboard/data_loader.py  (loads, caches, filters)
    |
    v
src/dashboard/metrics.py      (lift, OLL success rate, segment lift)
    |
    v
src/dashboard/views/          (executive, segments, overfitting, graveyard)
    |
    v
dashboard_app.py              (Streamlit entry point, 4-act layout)
```

### Key Modules

| Module | Description |
|--------|-------------|
| `dashboard_app.py` | Streamlit entry point — four-act layout |
| `src/models/ensemble_regressor.py` | EnsembleRegressor: CatBoost + RF + KNN + Linear + SMA + TTM |
| `src/segment_analyzer.py` | SegmentErrorAnalyzer: chi-square tests, quantile binning |
| `src/dashboard/data_loader.py` | Load/filter/cache pickle files, valid/graveyard trip splits |
| `src/dashboard/metrics.py` | Global lift, OLL success rate, segment lift calculations |
| `src/dashboard/views/executive.py` | KPI cards: lift, OLL success %, Agent vs Model comparison |
| `src/dashboard/views/overfitting.py` | Generalization check: Train/Val/Test lift comparison |
| `src/dashboard/views/segments.py` | Segment lift breakdown by Trade or Line |
| `src/dashboard/views/graveyard.py` | Invalid trip analysis: causes, zombie segments |
| `src/dashboard/components/story_module.py` | Scrollytelling narrative with chart + story cards |
| `synthesize_data.py` | Synthetic data generator (multivariate KDE, NaN patterns) |

## Dashboard Views

The dashboard follows a **four-act narrative structure**:

1. **Executive Summary** — KPI cards showing test-set lift, OLL success rate, Agent vs Model comparison
2. **Flight Recorder** — Scrollytelling narrative guiding stakeholders through key findings
3. **Deep Dive** — Overfitting detection (Train/Val/Test lift) + Segment lift (Trade/Line breakdown)
4. **Graveyard** — Analysis of invalid/excluded trips with cause breakdown

## Models

### EnsembleRegressor

Scikit-learn-compatible regressor combining 6 models:

| Model | Type | Notes |
|-------|------|-------|
| CatBoost | Gradient boosting | Default params or custom |
| Random Forest | Bagging | Configurable via `rf_params` |
| KNN | Instance-based | Configurable via `knn_params` |
| Linear Regression | Linear | Standard sklearn |
| SMA | Moving average | Same-day-of-week, 8-week window |
| TTM | Transformer | Tiny Time Mixers, lazy-loaded from HuggingFace |

Supports per-model feature subsets and customizable ensemble weights.

### SegmentErrorAnalyzer

Identifies feature segments over-represented in a subset (e.g., high-error cases):
- Chi-square tests for categorical columns
- Quantile binning (5 bins) for numeric columns
- Returns lift, subset share, p-values, and counts

## Data

### Source Files

| File | Description |
|------|-------------|
| `data/trip_scores.pkl` | Trip-level scores with dual rows per trip (Agent + DSG_model) |
| `data/origin_line_leg_scores.pkl` | Aggregated origin-line-leg scores with model diffs |

Data files are gitignored and not committed to the repository.

### Synthetic Data

`synthesize_data.py` generates synthetic datasets that preserve:
- Multivariate distributions (KDE)
- NaN patterns and rates
- Inter-column correlations
- Categorical proportions
- Paired Agent/DSG_model row structure

Run: `python synthesize_data.py`

### Train/Val/Test Split

The `data_type` column splits rows into `train`, `val`, and `test`. Valid trips require:
- `confirmed_at_end > 0`
- `allocation_at_end_grouped >= 50` (left boundary of interval bin)

## Development

### Rules
- All new functions must have **type hints**
- Create a **`.bak` copy** before major file modifications
- Adhere to **SOLID principles** for class and module designs

See [`CLAUDE.md`](CLAUDE.md) for full developer guidelines and Claude Code configuration.

## Testing

```bash
# All tests
pytest tests/

# Specific test files
pytest tests/test_ensemble_regressor.py -v
pytest tests/test_data_split.py -v

# Frontend QA (headless Streamlit AppTest — all views, widgets, view switching)
pytest tests/test_frontend.py -v

# Drift detection tests
pytest tests/test_drift_detectors.py tests/test_drift_baseline.py tests/test_drift_runner.py -v
```

| Test File | Covers | Tests |
|-----------|--------|-------|
| `test_frontend.py` | Headless dashboard QA: all views, selectbox, radio, expander | 16 |
| `test_drift_detectors.py` | Drift detection: KS, PSI, chi-square, NaN rate, label drift | 13 |
| `test_drift_baseline.py` | Baseline save/load, metadata, versioning, backups | 10 |
| `test_drift_runner.py` | End-to-end drift pipeline, unified verdicts | 6 |
| `test_ensemble_regressor.py` | EnsembleRegressor fit/predict/weights/TTM | 4 |
| `test_data_split.py` | DataSplitter temporal splits, CV modes, leakage (skipped) | 20+ |
| `repro_segment_analysis.py` | SegmentErrorAnalyzer verification script | 1 |

## Project Structure

```
hive_training_monitor/
├── dashboard_app.py                    Streamlit 4-act dashboard entry point
├── synthesize_data.py                  Synthetic data generator
├── requirements.txt                    Python dependencies
├── CLAUDE.md                           Developer guidelines
├── src/
│   ├── segment_analyzer.py             SegmentErrorAnalyzer
│   ├── models/
│   │   └── ensemble_regressor.py       EnsembleRegressor (6 models)
│   └── dashboard/
│       ├── data_loader.py              Load/filter/cache data
│       ├── metrics.py                  Lift, OLL success, segment metrics
│       ├── styles_premium.css          Dark-mode Glassmorphism CSS
│       ├── components/
│       │   └── story_module.py         Scrollytelling narrative
│       └── views/
│           ├── executive.py            KPI cards
│           ├── overfitting.py          Generalization check
│           ├── segments.py             Segment lift breakdown
│           └── graveyard.py            Invalid trip analysis
├── tests/
│   ├── test_ensemble_regressor.py      EnsembleRegressor tests
│   ├── test_data_split.py              DataSplitter tests
│   └── repro_segment_analysis.py       Segment analyzer verification
└── data/                               (gitignored)
    ├── trip_scores.pkl
    └── origin_line_leg_scores.pkl
```
