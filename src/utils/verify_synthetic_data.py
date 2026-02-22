import pandas as pd
import numpy as np

FILE = "synthetic_observations.parquet"

def verify():
    print(f"Reading {FILE}...")
    df = pd.read_parquet(FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    
    # 1. Key Uniqueness
    print("\n--- Checking Keys ---")
    if df.duplicated(subset=["trip_id", "created_date"]).any():
        print("❌ FAILD: Duplicate keys found for (trip_id, created_date)!")
    else:
        print("✅ Keys are unique.")

    # 2. Date Logic
    print("\n--- Checking Date Logic ---")
    calc_days = (df["departure_date"] - df["created_date"]).dt.days
    # Allow 1 day tolerance for float/int mismatch if any, but should be exact
    diff = (df["days_from_departure"] - calc_days).abs()
    if (diff > 0.001).any():
        print(f"❌ FAILED: days_from_departure mismatch! Max diff: {diff.max()}")
    else:
        print("✅ days_from_departure logic holds.")
        
    # 3. Connection Logic (Trip ID components)
    print("\n--- Checking Trip ID Logic ---")
    # trip_id is line-vessel-voyage-leg-origin
    # voy_id is line-vessel-voyage-leg
    # Check if trip_id starts with voy_id
    starts_with = df.apply(lambda x: x["trip_id"].startswith(x["voy_id"]), axis=1)
    if not starts_with.all():
         print("❌ FAILED: trip_id does not start with voy_id for some rows.")
    else:
         print("✅ trip_id / voy_id consistency holds.")

    # 4. Target Logic
    print("\n--- Checking Target Logic ---")
    # norm_confirmed_remaining_end = (confirmed_at_end - cumsum_fresh_teu) / capacity
    # confirmed_remaining_end = confirmed_at_end - cumsum_fresh_teu
    
    # Check conf_remaining logic
    rem_diff = (df["confirmed_remaining_end"] - (df["confirmed_at_end"] - df["cumsum_fresh_teu"])).abs()
    if (rem_diff > 0.001).any():
        print(f"❌ FAILED: confirmed_remaining_end logic mismatch! Max diff: {rem_diff.max()}")
    else:
        print("✅ confirmed_remaining_end logic holds.")
        
    # Check norm logic
    norm_diff = (df["norm_confirmed_remaining_end"] - (df["confirmed_remaining_end"] / df["zim_teu_capacity"])).abs()
    # Handle division by zero if any (capacity shouldn't be 0)
    if (norm_diff > 0.001).any():
        print(f"❌ FAILED: norm_confirmed_remaining_end logic mismatch! Max diff: {norm_diff.max()}")
    else:
        print("✅ norm_confirmed_remaining_end logic holds.")
        
    # 5. Splits
    print("\n--- Checking Splits ---")
    # Train < Val < Test by departure_date
    train_dates = df[df["data_type"] == "train"]["departure_date"]
    val_dates = df[df["data_type"] == "val"]["departure_date"]
    test_dates = df[df["data_type"] == "test"]["departure_date"]
    
    print(f"Train max: {train_dates.max()}")
    print(f"Val min: {val_dates.min()}")
    print(f"Val max: {val_dates.max()}")
    print(f"Test min: {test_dates.min()}")
    
    if train_dates.max() < val_dates.min() and val_dates.max() < test_dates.min():
        print("✅ Splits are correctly ordered by time.")
    else:
        print("❌ FAILED: Leakage in splits!")
        
    # 6. Schema coverage
    print("\n--- Checking Schema ---")
    # Provide a few key columns check
    required = ["trip_id", "created_date", "cumsum_fresh_teu", "norm_confirmed_remaining_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ FAILED: Missing columns: {missing}")
    else:
        print("✅ Critical schema columns present.")
        
    print("\nValidation Complete.")

if __name__ == "__main__":
    verify()
