import numpy as np
import pandas as pd

def estimate_hr_threshold(hr_series):
    """
    Estimates LTHR based on 95% of the Peak 20-minute Heart Rate.
    Returns: (estimated_lthr, peak_20min_hr)
    """
    hr = hr_series.dropna()
    if len(hr) == 0:
        return None, None

    # We need 20 minutes (1200 seconds)
    window_size = 1200
    if len(hr) < window_size:
        return None, None

    # Use rolling average (Pandas is usually faster/easier than np.convolve for Series)
    rolling_20min = hr.rolling(window=window_size, min_periods=window_size).mean()
    peak_20min_hr = rolling_20min.max()

    if pd.isna(peak_20min_hr):
        return None, None

    # Standard Friel calculation: 95% of 20m max
    estimated_lthr = peak_20min_hr * 0.95
    
    return estimated_lthr, peak_20min_hr

def hr_effort_zones(threshold_hr):
    """
    Returns Friel cycling heart rate zones based on LTHR, as
    {name: (low, high)} with contiguous half-open bounds [low, high).
    (Previously there were gaps, e.g. 0.89-0.90*LTHR, so those samples
    were never counted.)
    """
    upper = [
        ("Z1 (Recovery)", 0.81),
        ("Z2 (Aerobic)", 0.90),
        ("Z3 (Tempo)", 0.94),
        ("Z4 (Threshold)", 1.00),
        ("Z5 (VO2 Max)", 1.06),
        ("Z6 (Anaerobic)", np.inf),
    ]
    zones = {}
    low = 0.0
    for name, up in upper:
        zones[name] = (threshold_hr * low, threshold_hr * up)
        low = up
    return zones

def time_in_hr_zones(hr_series, thr):
    """
    Calculates time spent in each HR zone.
    Returns a dictionary: {'Zone Name': seconds}
    """
    from src.analysis.power import time_in_zones
    return time_in_zones(hr_series, hr_effort_zones(thr))
