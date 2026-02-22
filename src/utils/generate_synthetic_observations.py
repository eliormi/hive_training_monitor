import pandas as pd
import numpy as np
from datetime import timedelta, date
import random
import string

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIPS = 35000  # Target ~30-40k unique trip_ids
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
SNAPSHOTS_PER_TRIP = 42

OUTPUT_FILE = "synthetic_observations.parquet"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_base_ids(n: int) -> pd.DataFrame:
    """Generate base identifiers and static trip properties."""
    print("Generating base trip IDs...")
    
    # Pre-defined entities
    lines = ["ZIM", "MSC", "MSK", "CMA", "COS"]
    trades = ["TP", "TA", "ASIA", "LATAM"]
    vessels = [f"VESS_{i:03d}" for i in range(1, 201)]
    ports = ["NYC", "LAX", "SHA", "SIN", "ROT", "HAM", "SAV", "VAN"]
    origins = ["SHA", "NGB", "YTN", "SIN", "PUS"]
    legs = ["E", "W", "N", "S"]
    
    data = []
    
    # Generate random departure dates uniformly across the period
    date_range_days = (END_DATE - START_DATE).days
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Track unique IDs to prevent collisions
    seen_trip_ids = set()
    rows_generated = 0
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    while rows_generated < n:
        line = rng.choice(lines)
        trade = rng.choice(trades)
        vessel = rng.choice(vessels)
        voyage_num = rng.integers(1, 999)
        leg = rng.choice(legs)
        origin = rng.choice(origins)
        port = rng.choice(ports)
        
        # ID construction
        voy_id = f"{line}-{vessel}-{voyage_num:03d}-{leg}"
        trip_id = f"{voy_id}-{origin}"
        
        if trip_id in seen_trip_ids:
            continue
            
        seen_trip_ids.add(trip_id)
        
        # Date
        dep_date = START_DATE + timedelta(days=int(rng.integers(0, date_range_days)))
        
        # Capacity
        capacity = rng.choice([4000, 6000, 8000, 10000, 12000, 14000, 20000])
        
        data.append({
            "line": line,
            "trade": trade,
            "vessel": vessel,
            "voyage": f"{voyage_num:03d}",
            "leg": leg,
            "location_origin": origin,
            "port": port,
            "departure_date": dep_date,
            "voy_id": voy_id,
            "trip_id": trip_id,
            "zim_teu_capacity": float(capacity),
            "weight_capac": float(capacity * 12.5), # approx weight conversion
            "leg_type": "deep_sea" if trade in ["TP", "TA"] else "short_sea",
            
            # Additional static placeholders
            "line_port": f"{line}_{port}",
            "line_origin": f"{line}_{origin}",
            "line_port_leg": f"{line}_{port}_{leg}",
            "line_origin_leg": f"{line}_{origin}_{leg}",
            "line_leg": f"{line}_{leg}",
        })
        rows_generated += 1

        
    df = pd.DataFrame(data)
    
    # Sort by departure date to help with lag features
    df = df.sort_values("departure_date").reset_index(drop=True)
    return df

def expand_observations(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Expand each trip into 42 daily observations."""
    print(f"Expanding {len(trips_df)} trips into time series...")
    
    # Create a list of days backward from 0 to 41
    days_back = np.arange(SNAPSHOTS_PER_TRIP)
    
    # Repeat trips
    expanded = trips_df.loc[trips_df.index.repeat(SNAPSHOTS_PER_TRIP)].reset_index(drop=True)
    
    # Assign days_from_departure
    expanded["days_from_departure"] = np.tile(days_back, len(trips_df))
    expanded["snapshot_days"] = expanded["days_from_departure"]  # Duplicate column as per schema
    
    # Calculate created_date
    expanded["created_date"] = expanded["departure_date"] - pd.to_timedelta(expanded["days_from_departure"], unit="D")
    
    # Derived date parts
    expanded["departure_year"] = expanded["departure_date"].dt.year.astype(float)
    expanded["departure_month"] = expanded["departure_date"].dt.month.astype(float)
    expanded["departure_week"] = expanded["departure_date"].dt.isocalendar().week.astype(str)
    
    return expanded

def simulate_cumsum_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate booking curves (cumsum_fresh_teu) that grow over time 
    and calculate target variables.
    """
    print("Simulating booking dynamics and features...")
    
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = len(df)
    
    # We need to process by trip. 
    # To be fast, we can use vectorized operations assuming strict ordering (repeated blocks).
    # Since we expanded using repeat/tile, rows for a trip are contiguous block of 42.
    # Order: [row0 (day0), row1 (day1)...] wait, tile gave 0,1,2..41. 
    # Usually booking curves grow as days_from_departure DECREASES (dates approach departure).
    # So day 41 (far out) should have low bookings, day 0 (departure) high.
    
    # Let's verify 'days_from_departure'. 0 is departure. 41 is start of monitoring.
    days = df["days_from_departure"].values
    capacities = df["zim_teu_capacity"].values
    
    # Base curve: Sigmoid-like shape dependent on days (inverted, since 41->0)
    # x goes from 41 down to 0. 
    # Normalize time: t = (42 - days) / 42  => goes 0 to 1
    t = (42 - days) / 42.0
    
    # Random max load factor for each trip (broadcasted)
    # This simulates if a trip ends up full or empty
    # We need a random factor per trip, expanded to rows.
    # Since rows are sorted by trip blocks (from repeat), we can generate trip-level randoms.
    n_trips = len(df) // SNAPSHOTS_PER_TRIP
    trip_final_load_factor = rng.uniform(0.4, 1.1, size=n_trips) # 40% to 110% utilization
    trip_final_loads = np.repeat(trip_final_load_factor, SNAPSHOTS_PER_TRIP)
    
    # Noise per step
    noise = rng.normal(0, 0.02, size=n_rows) # 2% noise
    
    # Curve shape: t^2 (slow start, fast finish) or sigmoid. Let's use power.
    curve_shape = np.power(t, 2.5) 
    
    # Current utilization ratio
    current_util_ratio = curve_shape * trip_final_loads + noise
    current_util_ratio = np.clip(current_util_ratio, 0, None)
    
    # Calculate cumsum_fresh_teu
    df["cumsum_fresh_teu"] = current_util_ratio * capacities
    
    # Add other CUMSUM columns (correlated but distinct)
    for col in [
        "CUMSUM_FRESH_CONFIRMED_WITHOUT_CANCELED", 
        "CUMSUM_FRESH_UNCONFIRMED_TEU", 
        "CUMSUM_ROLLOVER_CONFIRMED_TEU",
        "CUMSUM_FRESH_NON_ROLLOVER_CONFIRMED"
    ]:
         # Just slight variations of the main curve for synthetic purposes
         df[col] = df["cumsum_fresh_teu"] * rng.uniform(0.9, 1.0, size=n_rows)
         
    # Canceled/Rollover (smaller values)
    for col in [
        "CUMSUM_CANCELED_FRESH_TEU", 
        "CUMSUM_ROLLOVER_CANCELED_TEU", 
        "CUMSUM_ROLLOVER_UNCONFIRMED_TEU",
        "CUMSUM_CURRENT_ROLLOVER_CONFIRMED_TEU",
        "CUMSUM_CURRENT_ROLLOVER_CANCELED_TEU",
        "CUMSUM_CURRENT_ROLLOVER_UNCONFIRMED_TEU"
    ]:
        df[col] = df["cumsum_fresh_teu"] * rng.uniform(0.0, 0.1, size=n_rows)

    # NORM columns
    for col in df.columns:
        if "CUMSUM" in col or col == "cumsum_fresh_teu":
            norm_col_name = "norm_" + col.lower()
            if "norm" not in col.lower(): # Avoid double naming if existing schema logic differs
                # The schema asks for 'norm_cumsum_fresh_teu', etc.
                pass
            df[f"norm_{col.lower()}"] = df[col] / df["zim_teu_capacity"]

    # --- Target Calculation ---
    # We need 'confirmed_at_end' which is the value of cumsum_fresh_teu at days_from_departure=0
    # Since days=0 is the last row of each trip block (or first depending on sort?), 
    # we can group transform.
    # Group by trip_id is slow. 
    # Optimization: We know blocks are 42 size. 
    # Indices where days_from_departure == 0 are trip ends.
    
    # Ensure sorted by trip and days descending or ascending
    # Let's sort to be safe: trip_id, days_from_departure
    df = df.sort_values(["trip_id", "days_from_departure"], ascending=[True, True])
    # Now each block is Day 0, Day 1... Day 41. 
    # Wait, day 0 is departure. Usually we want time series 41->0. 
    # But schema says days_from_departure. 
    # Let's grab the value where days_from_departure == 0.
    
    # Create a Series indexed by trip_id with the final value
    final_values = df.loc[df["days_from_departure"] == 0].set_index("trip_id")["cumsum_fresh_teu"]
    final_values.name = "Conf_at_end" # Temporary helper
    
    # Map back to all rows
    df["confirmed_at_end"] = df["trip_id"].map(final_values)
    
    # Calculate Targets
    df["confirmed_remaining_end"] = df["confirmed_at_end"] - df["cumsum_fresh_teu"]
    df["norm_confirmed_remaining_end"] = df["confirmed_remaining_end"] / df["zim_teu_capacity"]
    
    # Gate In/Out logic (similar to bookings but lagged)
    # Gate in usually happens closer to departure.
    df["gate_in"] = df["cumsum_fresh_teu"] * 0.9 # Simplified
    df["gate_out_at_end"] = df["confirmed_at_end"] * 0.95
    df["gate_in_at_end"] = df["confirmed_at_end"] * 0.98
    
    # Norm Gate End
    df["norm_gate_in_at_end"] = df["gate_in_at_end"] / df["zim_teu_capacity"]
    df["gate_in_remaining_end"] = df["gate_in_at_end"] - df["gate_in"]
    df["norm_gate_in_remaining_end"] = df["gate_in_remaining_end"] / df["zim_teu_capacity"]
    
    # Other dailies (noisy)
    df["fresh_teu"] = np.abs(rng.normal(10, 5, size=n_rows)) # Daily delta
    df["forecast_teu"] = df["cumsum_fresh_teu"] * rng.uniform(0.9, 1.2, size=n_rows)
    df["allocation_teu"] = df["zim_teu_capacity"] * rng.uniform(0.8, 1.0, size=n_rows)
    
    return df

def add_lags_and_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features based on previous trips."""
    print("Calculating historical window features (Lags)...")
    
    # This is complex to do vectorized for 1.4M rows.
    # Simplified approach: Group by Line/Service and use expanding windows.
    # To keep it efficient for synthetic data, we will fill with random consistent noise
    # or simple rolling averages on the whole dataset masquerading as history,
    # because strict previous-trip lookup is heavy.
    
    # However, to be "good", let's do a simple sort validation.
    # Just fill these with gaussian noise centered around meaningful means for now,
    # or use rolling mean on the 'trip_id' sequence if sorted by date.
    
    rng = np.random.default_rng(RANDOM_SEED)
    k = len(df)
    
    df["time_diff_from_last_voyage"] = rng.exponential(7, size=k) # Avg 7 days between voyages
    df["num_voyages_last_30_days"] = rng.choice([1, 2, 3, 4], size=k)
    
    # Past stats
    df["past_confirmed_remaining_avg_mean_3"] = rng.normal(500, 100, size=k)
    df["past_confirmed_remaining_median_3"] = df["past_confirmed_remaining_avg_mean_3"]
    df["past_confirmed_remaining_avg_mean_5"] = rng.normal(500, 100, size=k)
    
    return df

def split_train_val_test(df: pd.DataFrame) -> pd.DataFrame:
    """Split data by departure_date time."""
    print("Splitting into Train, Val, Test...")
    
    # Get unique dates and sort
    dates = df["departure_date"].unique()
    dates = np.sort(dates)
    
    n_dates = len(dates)
    n_test = int(n_dates * 0.10)
    n_val = int(n_dates * 0.10)
    n_train = n_dates - n_test - n_val
    
    # Cutoffs
    # Time based split: Train -> Val -> Test (chronological)
    train_end = dates[n_train]
    val_end = dates[n_train + n_val]
    
    conditions = [
        (df["departure_date"] < train_end),
        (df["departure_date"] >= train_end) & (df["departure_date"] < val_end),
        (df["departure_date"] >= val_end)
    ]
    choices = ["train", "val", "test"]
    
    df["data_type"] = np.select(conditions, choices, default="train")
    
    print("Split counts:")
    print(df["data_type"].value_counts(normalize=True))
    
    return df

def fill_missing_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns from the request exist, filling missing with 0/NaN/Category."""
    print("Filling remaining schema columns...")
    
    # List of columns to ensure exist (from prompt)
    required_float = [
        "snapshot_days","date_left_area","allocation_teu","confirmed_teu","forecast_teu",
        "source_fresh_teu","rollover_teu","gate_in","allocation_weight","gross_weight",
        "empty_picked_up","days_from_area_departure","weight_capac",
        "FRESH_CANCELED_TEU","FRESH_UNCONFIRMED_TEU","ROLLOVER_CONFIRMED_TEU",
        "ROLLOVER_CANCELED_TEU","ROLLOVER_UNCONFIRMED_TEU",
        "CURRENT_ROLLOVER_CANCELED_TEU","CURRENT_ROLLOVER_UNCONFIRMED_TEU",
        "CURRENT_ROLLOVER_CONFIRMED_TEU","FRESH_CONFIRMED_WITHOUT_CANCELED",
        "FRESH_NON_ROLLOVER_CONFIRMED",
        "gate_out_remaining_end","norm_gate_out_remaining_end",
        "days_from_start","confirmed_sum_origins","allocation_sum_origins",
        "empty_pick_up_confirmed_ratio","norm_allocation","norm_confirmed",
        "norm_confirmed_sum_origins","norm_allocation_sum_origins",
        "norm_rollover_teu","ratio_origin_confirmed","ratio_allocation_confirmed",
        "norm_empty_picked_up","norm_gate_in","norm_gross_weight",
        "port_order","before_norm_cumsum_fresh_teu","after_norm_cumsum_fresh_teu",
        "fresh_teu_ramp_up_before", "gate_in_ramp_up_before", "gate_out_ramp_up_before",
        "current_gate_in_gate_out_ratio","current_gate_out_fresh_ratio","current_gate_in_fresh_ratio"
    ]
    
    required_cats = [
        "cn_holiday_name","cn_holiday_duration_category", "us_holiday_name", "country"
    ]
    
    for col in required_float:
        if col not in df.columns:
            df[col] = np.random.uniform(0, 100, size=len(df)) # Placeholder random float
            
    for col in required_cats:
        if col not in df.columns:
            df[col] = "None"
            
    return df

def main():
    # 1. Generate core trips
    trips = generate_base_ids(N_TRIPS)
    
    # 2. Expand to time series
    obs = expand_observations(trips)
    
    # 3. Simulate Logic & Targets
    obs = simulate_cumsum_curves(obs)
    
    # 4. Add Lags/History
    obs = add_lags_and_windows(obs)
    
    # 5. Fill remaining schema cols with placeholders/defaults
    obs = fill_missing_schema_columns(obs)
    
    # 6. Split
    obs = split_train_val_test(obs)
    
    # 7. Convert types
    # Ensure object/category types match request for string cols
    for col in ["trip_id", "voy_id", "line", "trade", "vessel", "voyage", "leg", "port", "data_type"]:
        obs[col] = obs[col].astype(str)
        
    # 8. Save
    print(f"Final shape: {obs.shape}")
    print(f"Saving to {OUTPUT_FILE}...")
    obs.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
