import numpy as np
from scipy.ndimage import uniform_filter1d
import pandas as pd
from scipy.stats import linregress

def NP(power_series):
    """
    Compute Normalized Power (NP).
    """
    p = power_series.dropna().to_numpy()
    if len(p) == 0:
        return np.nan

    if len(p) < 30:
        av30 = uniform_filter1d(p, size=max(1, len(p)))
    else:
        av30 = uniform_filter1d(p, size=30)
    
    av30 = av30[~np.isnan(av30)]
    if len(av30) == 0:
        return np.nan

    npower = (np.mean(av30 ** 4)) ** 0.25
    return npower

def IF(norm_power, ftp):
    """
    Compute Intensity Factor (IF).
    """
    if ftp == 0 or ftp is None:
        return 0.0
    return norm_power / ftp

def TSS(norm_power, duration_seconds, ftp):

    if ftp is None or ftp == 0:
        return 0.0
    if_val = IF(norm_power, ftp)
    return (duration_seconds * norm_power * if_val) / (ftp * 3600) * 100

def power_curve(power_series, duration_s=None):
    """
    Compute the max-average power for a list of durations.
    """
    if duration_s is None:
        duration_s = [5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200]
    
    p = power_series.dropna().to_numpy()
    n = len(p)
    out = {}
    
    for d in duration_s:
        if d <= 0 or n == 0 or d > n:
            out[d] = np.nan
            continue
        
        window = np.ones(d) / d
        means = np.convolve(p, window, mode='valid')
        out[d] = float(np.nanmax(means))
        
    return out

def coggan_zones(ftp):
    """
    Returns Coggan power zones based on FTP as {zone: (low, high)} in watts.
    Bounds are contiguous and half-open [low, high): every sample falls in
    exactly one zone (previously values between e.g. 0.55 and 0.56*FTP were
    not counted at all).
    """
    # Coggan upper bounds as fraction of FTP (Z7 is open-ended)
    upper = [0.55, 0.75, 0.90, 1.05, 1.20, 1.50, np.inf]
    zones = {}
    low = 0.0
    for z, up in enumerate(upper, start=1):
        zones[z] = (ftp * low, ftp * up)
        low = up
    return zones

def time_in_zones(series, zones):
    """
    Seconds spent in each zone (assumes 1 Hz samples).
    'zones' must be a dict {zone_id: (low, high)} with half-open bounds
    [low, high); NaNs are ignored and values below the first bound are
    counted in the first zone.
    """
    counts = {z: 0 for z in zones}
    if series is None or len(series) == 0:
        return counts

    vals = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return counts

    keys = list(zones.keys())
    # Inner edges = lower bounds of zones 2..n, so digitize gives the zone index
    edges = np.array([zones[k][0] for k in keys[1:]], dtype=float)
    idx = np.digitize(vals, edges, right=False)
    binned = np.bincount(idx, minlength=len(keys))
    for i, k in enumerate(keys):
        counts[k] = int(binned[i])
    return counts

def calculate_w_prime_balance(power_series, ftp, w_prime_cap_j=20000, tau=546):
    """
    Calculate W' Balance over time.
    """
    if ftp is None or ftp == 0 or power_series.empty:
        return pd.Series(dtype=float)

    p = power_series.fillna(0).to_numpy()
    w_bal = np.zeros_like(p, dtype=float)

    w_exp = 0.0
    decay = np.exp(-1 / tau)

    for i, watts in enumerate(p):
        # Integral model:
        # W_exp(t) = W_exp(t-1) * exp(-1/tau) + max(0, P(t) - CP)

        excess_power = max(0, watts - ftp)
        w_exp = w_exp * decay + excess_power

        w_bal[i] = w_prime_cap_j - w_exp

    return pd.Series(w_bal, index=power_series.index)

def calculate_ride_kJ(power_series):
    """ Assume series sampled a 1Hz """
    if power_series.empty:
        return 0.0

    total_kJ = power_series.fillna(0).sum() / 1000.0
    return total_kJ


def estimate_critical_power(power_curve_dict):
    """
    Estimate Critical Power (CP) using linear regression on the power-duration curve.
    power_curve_dict: dict of {duration_s: power_watts}
    Returns estimated CP in watts.
    """
    if not power_curve_dict:
        return None

    # 1. Convert to arrays
    # Filter for valid physiological window for CP model (usually 3m to 20m)
    # Short durations (<180s) are dominated by anaerobic capacity (AC)
    # Long durations (>1200s) are dominated by VO2max/Efficiency
    min_sec = 180   # 3 minutes
    max_sec = 1200  # 20 minutes
    
    times = []
    work = []
    
    for t, p in power_curve_dict.items():
        if min_sec <= t <= max_sec:
            times.append(t)
            work.append(p * t) # Joules
            
    # We need at least 2 points to draw a line, but 3+ is better
    if len(times) < 2:
        return None
        
    # 2. Linear Regression (Work vs Time)
    # Slope = CP, Intercept = W'
    slope, intercept, r_value, p_value, std_err = linregress(times, work)
    
    results = {
        'cp': slope,               # Critical Power in Watts
        'w_prime': intercept,      # W' in Joules
        'r_squared': r_value**2    # Goodness of fit (close to 1.0 is best)
    }
    
    return results   

def detect_breakthrough(w_prime_balance_series):
    """
    Checks if W' Balance went negative.
    Returns the magnitude of the breakthrough (kJ) or 0 if none.
    """
    min_w_bal = w_prime_balance_series.min()
    
    if min_w_bal < 0:
        # A negative balance of -5000 J means the rider had 5kJ more capacity
        # than we thought, OR their CP is higher.
        return abs(min_w_bal)
    return 0.0

def suggest_new_cp(breakthrough_kj, current_cp, current_w_prime, duration_above_cp_sec):
    """
    Rough estimator to suggest a new CP based on the breakthrough.
    If we assume W' is correct, how much higher must CP be to avoid this negative dip?
    
    Math: Work_done = (CP_old * t) + W_prime_old + Breakthrough
          Work_done = (CP_new * t) + W_prime_old
          
    Therefore: CP_new = CP_old + (Breakthrough / t)
    """
    if duration_above_cp_sec == 0:
        return current_cp
        
    cp_increase = breakthrough_kj / duration_above_cp_sec
    return current_cp + cp_increase
