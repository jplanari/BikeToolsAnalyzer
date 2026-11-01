import gpxpy
import pandas as pd
from geographiclib.geodesic import Geodesic
import numpy as np
import re
import streamlit as st

@st.cache_data(show_spinner=False)
def parse_gpx(file_content):
    """
    Parses GPX file and return DataFrame with time, lat, lon, ele, hr, power, etc.
    """

    try:
        if isinstance(file_content, bytes):
            gpx_xml = file_content.decode('utf-8-sig')
        elif isinstance(file_content, str):
            gpx_xml = file_content
        else:
            gpx_xml = file_content.read()
            if isinstance(gpx_xml, bytes):
                gpx_xml = gpx_xml.decode('utf-8-sig')
        gpx = gpxpy.parse(gpx_xml)
    except Exception as e:
        raise ValueError(f"Failed to parse GPX file: {e}")
        return pd.DataFrame()

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
    # Indoor rides (Zwift, trainers) and GPS dropouts have no position: use the distance the device recorded
    native = pd.to_numeric(df['dist_m'], errors='coerce').to_numpy(dtype=float) if 'dist_m' in df.columns else None

    for i in range(1, N):
        # Calculate distance
        d = haversine_dist(lats[i-1], lons[i-1], lats[i], lons[i])
        if native is not None and (np.isnan(lats[i-1]) or np.isnan(lons[i-1]) or np.isnan(lats[i]) or np.isnan(lons[i])):
            step = native[i] - native[i-1]
            d = step if np.isfinite(step) and step >= 0 else 0.0
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

BEARING_WINDOW_S = 5


def calculate_bearing(df, window=BEARING_WINDOW_S):
    """
    Direction of travel (degrees, 0 = north) from the position `window` samples behind to the one
    `window` samples ahead. One-second steps of 5-8 m are within GPS noise, so a single step gives
    a bearing that wanders by tens of degrees.
    """
    if 'lat' not in df.columns or 'lon' not in df.columns:
        return pd.Series(0, index=df.index)

    n = len(df)
    idx = np.arange(n)
    back = np.clip(idx - window, 0, max(n - 1, 0))
    fwd = np.clip(idx + window, 0, max(n - 1, 0))
    lat, lon = df['lat'].to_numpy(dtype=float), df['lon'].to_numpy(dtype=float)
    lat1, lon1 = np.radians(lat[back]), np.radians(lon[back])
    lat2, lon2 = np.radians(lat[fwd]), np.radians(lon[fwd])

    d_lon = lon2 - lon1
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    y = np.sin(d_lon) * np.cos(lat2)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360

    #Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360

    return pd.Series(bearing_deg, index=df.index)

# Recording gaps up to this long are treated as "still riding" (smart
# recording, brief sensor dropouts): power, cadence and speed are carried
# across. Longer gaps are pauses: power and cadence 0, position held.
MAX_HOLD_GAP_S = 10

def resample_to_seconds(df):
    """
    Resamples data to 1-second intervals.
    - Gaps <= MAX_HOLD_GAP_S: all channels interpolated (devices with smart
      recording only log every few seconds; zero-filling those seconds used
      to cut average power by up to two thirds).
    - Longer gaps (pauses): position held, power/cadence 0, speed 0.
    - Lat/lon/ele/HR are interpolated over gaps up to 30 s, as before.
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.dropna(subset=['time']).drop_duplicates(subset='time').set_index('time').sort_index()
    df['_orig'] = 1.0

    # 1. Resample to 1s grid
    df_resampled = df.resample('1s').asfreq()

    # Length of the recording gap each inserted second belongs to
    t = df_resampled.index.to_series()
    prev_t = t.where(df_resampled['_orig'].notna()).ffill()
    next_t = t.where(df_resampled['_orig'].notna()).bfill()
    gap_s = (next_t - prev_t).dt.total_seconds()
    inserted = df_resampled['_orig'].isna()
    short_gap = inserted & (gap_s <= MAX_HOLD_GAP_S)
    long_gap = inserted & ~short_gap

    # 2. Interpolate position/HR over gaps up to 30 s (GPS dropouts)
    cols_to_interp = [c for c in ['lat', 'lon', 'ele', 'hr'] if c in df_resampled.columns]
    df_resampled[cols_to_interp] = df_resampled[cols_to_interp].interpolate(method='time', limit=30)

    # 3. Effort channels: carry across short gaps only
    for col in ['power', 'cadence', 'speed']:
        if col in df_resampled.columns:
            filled = pd.to_numeric(df_resampled[col], errors='coerce').interpolate(
                method='time', limit_area='inside')
            df_resampled[col] = filled.where(~long_gap)

    # 4. Long gaps: the rider stopped
    for col in ['lat', 'lon', 'ele']:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].ffill()
    for col in ['power', 'cadence']:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].fillna(0)
    if 'speed' in df_resampled.columns:
        df_resampled.loc[long_gap, 'speed'] = 0.0
    if 'hr' in df_resampled.columns:
        df_resampled['hr'] = df_resampled['hr'].ffill()

    # Recorded distance: linear across short gaps, held while stopped
    if 'dist_m' in df_resampled.columns:
        d = pd.to_numeric(df_resampled['dist_m'], errors='coerce')
        df_resampled['dist_m'] = d.interpolate(method='time', limit_area='inside').where(~long_gap, d.ffill())

    df_resampled = df_resampled.drop(columns='_orig').reset_index()
    return df_resampled
