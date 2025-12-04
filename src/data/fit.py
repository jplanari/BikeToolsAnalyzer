# src/data/fit.py
import fitparse
import pandas as pd
import numpy as np

def parse_fit(file_obj):
    """
    Parses a FIT file object and returns a standardized Pandas DataFrame 
    compatible with the rest of the BikeToolsAnalyzer pipeline.
    """
    # fitparse expects a file path or a file-like object opened in binary mode
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
