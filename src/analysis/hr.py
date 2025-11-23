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
    Returns heart rate zones based on LTHR.
    """
    # Friel Cycling Zones (approximate)
    zones = {
            "Z1 (Recovery)": (0, threshold_hr * 0.81),
            "Z2 (Aerobic)":  (threshold_hr * 0.81, threshold_hr * 0.89),
            "Z3 (Tempo)":    (threshold_hr * 0.90, threshold_hr * 0.93),
            "Z4 (Threshold)":(threshold_hr * 0.94, threshold_hr * 0.99),
            "Z5 (VO2 Max)":  (threshold_hr * 1.00, threshold_hr * 1.06),
            "Z6 (Anaerobic)":(threshold_hr * 1.06, 250)
            }
    return zones

def time_in_hr_zones(hr_series, thr):
    """
    Calculates time spent in each HR zone.
    Returns a dictionary: {'Zone Name': seconds}
    """
    zones = hr_effort_zones(thr)
    counts = {z: 0 for z in zones}
    
    hr = hr_series.dropna()
    if len(hr) == 0:
        return counts

    # Categorize HR data into zones
    for val in hr:
        for zone, (low, high) in zones.items():
            if low <= val <= high:
                counts[zone] += 1
                break
                
    return counts
