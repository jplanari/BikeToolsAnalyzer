# src/data/fit.py
import fitparse
import pandas as pd
import numpy as np
import io
import streamlit as st

@st.cache_data(show_spinner=False)
def parse_fit(file_bytes):
    """
    Parses a FIT file object and returns a standardized Pandas DataFrame 
    compatible with the rest of the BikeToolsAnalyzer pipeline.
    """
    # fitparse expects a file path or a file-like object opened in binary mode
    file_obj = io.BytesIO(file_bytes)

    try:
        fitfile = fitparse.FitFile(file_obj)
    except Exception as e:
        print(f"Error opening FIT file: {e}")
        return pd.DataFrame()

    records = []

    # Iterate over all messages of type "record" (points in the track)
    for record in fitfile.get_messages("record"):
        row = {}
        
        # Extract data fields from the record
        # fitparse returns values with units, we extract the raw values
        for data in record:
            row[data.name] = data.value

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # --- STANDARDIZATION ---
    # We need to rename columns to match the app's standard schema (derived from gpx.py)
    
    # Standard mapping based on common FIT fields
    col_mapping = {
        'timestamp': 'time',
        'position_lat': 'lat',
        'position_long': 'lon',
        'altitude': 'ele',
        'heart_rate': 'hr',
        'cadence': 'cadence',
        'power': 'power',
        'speed': 'speed',        # usually in m/s
        'temperature': 'temp_c',
        'distance': 'dist_m'     # usually cumulative distance in meters
    }
    
    df.rename(columns=col_mapping, inplace=True)

    df['speed_kmh'] = df['speed'] * 3.6 if 'speed' in df.columns else np.nan

    # --- DATA CONVERSIONS ---
    
    # 1. Convert Coordinates (Semicircles to Degrees)
    # FIT stores lat/lon as semicircles. Conversion: degrees = semicircles * (180 / 2^31)
    semicircle_converson = 180 / (2**31)
    
    if 'lat' in df.columns:
        df['lat'] = df['lat'] * semicircle_converson
    if 'lon' in df.columns:
        df['lon'] = df['lon'] * semicircle_converson

    # 2. Ensure Time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # 3. Handle Missing standard columns (fill with None/NaN if missing so downstream logic doesn't break)
    required_cols = ['lat', 'lon', 'ele', 'power', 'hr', 'cadence']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 4. Clean Speed (Optional: Convert m/s to km/h if your app prefers it, 
    # but your aerodyn.py uses m/s, so we keep it or ensure we know what it is.
    # GPX parser usually calculates speed, here we have it natively.)
    
    # 5. Sort by time just in case
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)

    return df



@st.cache_data(show_spinner=False)
def get_fit_laps(file_bytes):
    """
    Parses a FIT file and extracts user-defined laps.
    Includes robust field mapping and pointer reset.
    """
    # 1. RESET POINTER (Critical!)
    # If parse_fit() ran before this, the pointer is at the end of the file.
    file_obj = io.BytesIO(file_bytes)

    try:
        fitfile = fitparse.FitFile(file_obj)
    except Exception as e:
        print(f"FIT Parse Error: {e}")
        return pd.DataFrame()

    laps_data = []

    # 2. Iterate over 'lap' messages
    # Note: Some files might use 'session' if there are no manual laps, 
    # but we specifically want 'lap' messages here.
    for lap in fitfile.get_messages("lap"):
        data = {}
        for record_data in lap:
            # We capture all raw values first
            data[record_data.name] = record_data.value
        laps_data.append(data)

    if not laps_data:
        print("Debug: No 'lap' messages found in FIT file.")
        return pd.DataFrame()

    df_laps = pd.DataFrame(laps_data)
    
    # Debug: Print found columns to console (check your terminal)
    print(f"Debug: Found Lap Columns: {df_laps.columns.tolist()}")

    # 3. Robust Column Mapping
    # FIT fields can vary. We map common variations to our display names.
    # format: 'Display Name': ['possible_fit_field_1', 'possible_fit_field_2']
    field_map = {
        'Duration': ['total_timer_time', 'total_elapsed_time'],
        'Distance (km)': ['total_distance'],
        'Avg Power (W)': ['avg_power'],
        'Max Power (W)': ['max_power'],
        'Avg HR': ['avg_heart_rate'],
        'Max HR': ['max_heart_rate'],
        'Avg Cadence (rpm)': ['avg_cadence'],
        'Avg Speed (km/h)': ['avg_speed'] # usually m/s
    }
    
    final_df = pd.DataFrame()

    for display_name, candidates in field_map.items():
        for col in candidates:
            if col in df_laps.columns:
                final_df[display_name] = df_laps[col]
                break # Found a match, move to next field

    # 4. Clean and Format Data
    if 'Avg Speed (km/h)' in final_df.columns:
        # Convert m/s to km/h
        final_df['Avg Speed (km/h)'] = (final_df['Avg Speed (km/h)'] * 3.6).round(1)
        final_df.rename(columns={'Avg Speed (km/h)': 'Speed (km/h)'}, inplace=True)
        
    if 'Distance (km)' in final_df.columns:
        final_df['Distance (km)'] = final_df['Distance (km)'].round(0)/1000 # kilometers

    if 'Duration' in final_df.columns:
        # Format seconds to MM:SS
        def fmt_time(s):
            m, s = divmod(int(s or 0), 60)
            return f"{m:02d}:{s:02d}"
        final_df['Duration'] = final_df['Duration'].apply(fmt_time)

    # 5. Drop empty columns (if a sensor was missing)
    final_df = final_df.dropna(axis=1, how='all')
    
    return final_df

