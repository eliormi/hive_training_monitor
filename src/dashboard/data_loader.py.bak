import streamlit as st
import pandas as pd
import os

TRIP_FILE = 'data/trip_scores.pkl'
OLL_FILE = 'data/origin_line_leg_scores.pkl'

@st.cache_data
def load_trip_data():
    """Loads and caches the trip scores dataset."""
    if not os.path.exists(TRIP_FILE):
        st.error(f"File not found: {TRIP_FILE}")
        return pd.DataFrame()
    return pd.read_pickle(TRIP_FILE)

@st.cache_data
def load_oll_data():
    """Loads and caches the Origin-Line-Leg scores dataset."""
    if not os.path.exists(OLL_FILE):
        st.error(f"File not found: {OLL_FILE}")
        return pd.DataFrame()
    return pd.read_pickle(OLL_FILE)

def get_filtered_data(df, data_type_filter='test'):
    """Filters data by data_type (train, val, test)."""
    if data_type_filter == 'all':
        return df
    return df[df['data_type'] == data_type_filter].copy()

def get_valid_trips(df):
    """
    Filters for trips with:
    1. confirmed_at_end > 0
    2. allocation_at_end_grouped > 50 (min value of bin > 50)
    """
    # 1. Confirmed Check
    mask_confirmed = df['confirmed_at_end'] > 0
    
    # 2. Allocation Check (Interval Category)
    # We assume the index is an Interval. safely allow for potential numeric fallback
    def is_big_allocation(val):
        try:
            # Check if it's an interval
            if hasattr(val, 'left'):
                return val.left >= 50
            # Or if it's a string representation "(50, ...]"
            if isinstance(val, str):
                params = val.strip('()[]').split(',')
                return float(params[0]) >= 50
            return False
        except:
            return False
            
    mask_alloc = df['allocation_at_end_grouped'].apply(is_big_allocation)
    
    return df[mask_confirmed & mask_alloc].copy()

def get_graveyard_trips(df):
    """
    Returns trips that failed the validity check:
    1. Confirmed == 0
    OR
    2. Allocation <= 50
    """
    # Inverse logic of get_valid_trips
    
    # Re-use the logic for consistency
    mask_confirmed = df['confirmed_at_end'] > 0
    
    def is_big_allocation(val):
        try:
            if hasattr(val, 'left'):
                return val.left >= 50
            if isinstance(val, str):
                params = val.strip('()[]').split(',')
                return float(params[0]) >= 50
            return False
        except:
            return False
            
    mask_alloc = df['allocation_at_end_grouped'].apply(is_big_allocation)
    
    # Graveyard = NOT (Confirmed + Alloc)
    # De Morgan's: NOT (A & B) = (NOT A) | (NOT B)
    mask_graveyard = (~mask_confirmed) | (~mask_alloc)
    
    df_grave = df[mask_graveyard].copy()
    
    # Add 'Reason' column for analysis
    def get_reason(row):
        if row['confirmed_at_end'] <= 0:
            return 'Zero Bookings'
        return 'Low Allocation' # implied since it failed the AND check but passed confirmed
        
    df_grave['death_reason'] = df_grave.apply(get_reason, axis=1)
    
    return df_grave
