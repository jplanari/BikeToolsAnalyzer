import gpxpy
import pandas as pd
from geographiclib.geodesic import Geodesic
import numpy as np
import re

def parse_gpx(file_path):
    """
    Parses GPX file and return DataFrame with time, lat, lon, ele, hr, power, etc.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                # 1. Handle Timezone (Fix for matplotlib crash)
                time_naive = p.time.replace(tzinfo=None) if p.time else None
                
                row = {
                        'time': time_naive,
                        'lat': p.latitude,
                        'lon': p.longitude,
                        'ele': p.elevation,
                }

                # Initialize optional fields
                hr = None
                power = None
                cadence = None

                # 2. Robust Parsing of Extensions (Fix for Strava Power)
                if p.extensions:
                    for ext in p.extensions:
                        # CASE A: Strava often puts 'power' as a direct extension tag
                        # We check if 'power' is in the tag name (ignoring namespaces like {http...}power)
                        if 'power' in ext.tag:
                            try:
                                power = int(ext.text)
                            except (ValueError, TypeError):
                                pass

                        # CASE B: Garmin TrackPointExtension (contains HR, Cadence, etc. as children)
                        # We iterate over the children of the extension
                        for child in ext:
                            tag = child.tag.lower()
                            
                            # Heart Rate
                            if 'hr' in tag or 'heartrate' in tag:
                                try:
                                    hr = int(child.text)
                                except (ValueError, TypeError):
                                    pass
                            
                            # Cadence (usually 'cad' or 'cadence')
                            if 'cad' in tag:
                                try:
                                    cadence = int(child.text)
                                except (ValueError, TypeError):
                                    pass
                            
                            # Power (sometimes nested in other formats)
                            if 'power' in tag:
                                try:
                                    power = int(child.text)
                                except (ValueError, TypeError):
                                    pass
                
                row['hr'] = hr
                row['power'] = power
                row['cadence'] = cadence
                rows.append(row)

    df = pd.DataFrame(rows)
    
    # 3. Enforce Numeric Types (Fix for Pandas FutureWarning & Interpolation)
    numeric_cols = ['lat', 'lon', 'ele', 'hr', 'power', 'cadence']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure time is sorted and drop rows without time
    if not df.empty and 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
        
    return df

def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculates geodesic distance between two points.
    Returns 0.0 if any coordinate is NaN.
    """
    # Safety check for NaNs to prevent breaking the cumulative sum
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
        return 0.0
        
    geod = Geodesic.WGS84
    g = geod.Inverse(lat1, lon1, lat2, lon2)
    return g['s12']

def compute_distance_and_ascent(df):
    """
    Computes cumulative distance and ascent.
    Robust against NaNs to prevent 'cutting' the profile.
    """
    if df.empty:
        return df, 0, 0

    N = len(df)
    dists = np.zeros(N)
    climbs = np.zeros(N)

    lats = df['lat'].values
    lons = df['lon'].values
    eles = df['ele'].values
    
    for i in range(1, N):
        # Calculate distance
        d = haversine_dist(lats[i-1], lons[i-1], lats[i], lons[i])
        dists[i] = d
        
        # Calculate ascent (only if elevation data exists)
        e_prev = eles[i-1]
        e_curr = eles[i]
        
        if not np.isnan(e_curr) and not np.isnan(e_prev):
            diff = e_curr - e_prev
            if diff > 0:
                climbs[i] = diff

    total_dist = np.nansum(dists)
    total_climb = np.nansum(climbs)
    
    # cumsum can propagate NaNs, so we fill NaNs with 0 before summing
    df['dist_m'] = np.cumsum(np.nan_to_num(dists))
    df['segment_dist'] = dists
    df['segment_climb'] = climbs
    
    return df, total_dist, total_climb

def compute_speed(df):
    """
    Computes speed in m/s from distance and time.
    """
    if df.empty or 'dist_m' not in df.columns:
        return df

    df = df.copy()
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    df['dist_diff'] = df['dist_m'].diff().fillna(0)

    # Speed = distance / time
    df['speed'] = df.apply(
        lambda row: row['dist_diff'] / row['time_diff'] if row['time_diff'] > 0 else 0,
        axis=1
    )

    df['speed_kmh'] = df['speed'] * 3.6  # Convert m/s to km/h

    return df

def compute_grade(df):
    """
    Computes grade (%) from elevation and distance.
    """
    if df.empty or 'ele' not in df.columns or 'dist_m' not in df.columns:
        return df

    df = df.copy()
    df['ele_diff'] = df['ele'].diff().fillna(0)
    df['grade'] = df.apply(
        lambda row: (row['ele_diff'] / row['segment_dist'] * 100) if row['segment_dist'] > 0 else 0,
        axis=1
    )

    # Smooth grade with rolling mean (window=5s)
    df['grade_smooth'] = df['grade'].rolling(window=5, min_periods=1, center=True).mean()

    return df

def calculate_bearing(df):

    if 'lat' not in df.columns or 'lon' not in df.columns:
        return pd.Series(0, index=df.index)

    lat1 = np.radians(df['lat'])
    lon1 = np.radians(df['lon'])
    lat2 = np.radians(df['lat'].shift(-1).fillna(method='ffill'))
    lon2 = np.radians(df['lon'].shift(-1).fillna(method='ffill'))

    d_lon = lon2 - lon1
    
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    y = np.sin(d_lon) * np.cos(lat2)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360

    #Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg

def resample_to_seconds(df):
    """
    Resamples data to 1-second intervals.
    Handles gaps (pauses) by holding position (distance=0) and zeroing power.
    """
    if df.empty:
        return df
        
    df = df.copy()
    df = df.set_index('time')
    
    # 1. Resample to 1s grid
    df_resampled = df.resample('1s').asfreq()
    
    # 2. Interpolate short gaps (GPS dropouts < 30s)
    #    We interpolate Lat/Lon/Ele/HR so movement looks smooth
    cols_to_interp = ['lat', 'lon', 'ele', 'hr']
    # Only interpolate columns that actually exist
    cols_to_interp = [c for c in cols_to_interp if c in df_resampled.columns]
    
    df_resampled[cols_to_interp] = df_resampled[cols_to_interp].interpolate(
        method='time', limit=30
    )

    # 3. Handle Long Gaps (> 30s, e.g., lunch break)
    #    If we didn't interpolate (limit reached), we assume the rider STOPPED.
    #    Position (lat/lon/ele) -> Forward Fill (stay at last known point)
    df_resampled['lat'] = df_resampled['lat'].ffill()
    df_resampled['lon'] = df_resampled['lon'].ffill()
    df_resampled['ele'] = df_resampled['ele'].ffill()
    
    #    Power/Cadence -> Fill with 0 (not pedaling)
    if 'power' in df_resampled.columns:
        df_resampled['power'] = df_resampled['power'].fillna(0)
    if 'cadence' in df_resampled.columns:
        df_resampled['cadence'] = df_resampled['cadence'].fillna(0)
        
    #    HR -> Forward Fill (heart still beating, assume last value)
    if 'hr' in df_resampled.columns:
        df_resampled['hr'] = df_resampled['hr'].ffill()

    df_resampled = df_resampled.reset_index()
    return df_resampled

