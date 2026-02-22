# IMPLANTATION PLAN: Synthetic Time-Series Data Generation

## Goal Description
Generate a robust synthetic dataset representing container shipping bookings over time. The dataset will contain approximately 3 years of data with 30-40k unique trips. Each trip will be tracked for 42 days prior to departure (snapshots), resulting in a time-series structure where `(trip_id, created_date)` serves as the primary key.

The data will feature realistic distributions, correlations, and specifically defined logical relationships between columns (e.g., `days_from_departure`, `confirmed_remaining_end`).

## User Review Required
> [!IMPORTANT]
> **Data Volume**: Generaring 30k-40k trips * 42 snapshots = ~1.2 - 1.6 million rows. This fits comfortably in memory (Pandas), but we will observe memory usage.

> [!NOTE]
> **Schema Definition**: The provided schema contains ~100 columns. I will implement generators for all, grouping them by logic (Static trip info, Dynamic daily stats, Lags/Windows, Target variables).

## Proposed Changes

### [NEW] `generate_synthetic_observations.py`

This script will handle the end-to-end generation process.

#### 1. Entity & Trip Generation
- **Time Range**: Generate trips with `departure_date` spanning 3 years.
- **Base Entities**: Pre-define a set of `Lines`, `Vessels`, `Trades`, `services` to create realistic `trip_id` components.
- **Trip ID Construction**: `line-vessel-voyage-leg-origin`.
- **Voyage ID Construction**: `line-vessel-voyage-leg`.

#### 2. Time-Series Expansion (The "42 Dates" Logic)
- For each trip, generate 42 `created_date` values: $created\_date = departure\_date - n \text{ days}$, where $n \in \{0, 1, ..., 41\}$.
- **Calculated Columns**:
    - `days_from_departure`: $departure\_date - created\_date$
    - `departure_year`, `departure_month`, `departure_week` derived from `departure_date`.
    - `snapshot_days`: same as `days_from_departure` (or inverse, will clarify).

#### 3. Feature Generation (Simulated Dynamics)
- **Booking Curves**: Simulate `cumsum_fresh_teu` using a trend (e.g., sigmoid or exponential ramp-up as departure approaches) + noise.
- **Capacity**: `zim_teu_capacity` sampled from a distribution.
- **Target Variable**:
    - `Conf_at_end`: The final value of `cumsum_fresh_teu` (at $days\_from\_departure = 0$).
    - `confirmed_remaining_end`: $Conf\_at\_end - cumsum\_fresh\_teu$.
    - `norm_confirmed_remaining_end`: $confirmed\_remaining\_end / zim\_teu\_capacity$ (**LABEL**).

#### 4. Data Splitting
- **Logic**: Time-based split on `departure_date`.
- **Ratios**: Test (10%), Val (10%), Train (80%).
- **Verification**: Ensure no leakage (trips in test should have departure dates strictly after train/val).

## Verification Plan

### Automated Verification
The script will include a `verify_data()` function to assert:
1.  **Shape**: Total rows $\approx$ Num Trips $\times$ 42.
2.  **Keys**: `trip_id` + `created_date` is unique.
3.  **Consistency**:
    - `days_from_departure` matches date diff.
    - `Conf_at_end` is consistent for all rows of a trip.
    - `norm_confirmed_remaining_end` matches the formula.
4.  **Splits**:
    - Check sizes (approx 10% for val/test).
    - Check date overlap (Test `departure_date` > Val > Train).

### Manual Verification
- Inspect the first few rows of the generated DataFrame.
- Plot a few "booking curves" (`cumsum_fresh_teu` vs `days_from_departure`) to ensure they look like realistic booking accumulations (increasing over time).
