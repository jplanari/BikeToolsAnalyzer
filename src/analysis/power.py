import numpy as np
from scipy.ndimage import uniform_filter1d
import pandas as pd

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
    """ Returns Coggan power zones based on FTP. """
    # Coggan v3 approximate boundaries ratios
    zones = {
            1: (0, 0.55),
            2: (0.56, 0.75),
            3: (0.76, 0.90),
            4: (0.91, 1.05),
            5: (1.06, 1.20),
            6: (1.21, 1.50),
            7: (1.51, 50.0) # High cap
    }
    # Multiply the ratios by FTP
    return {z: (ftp * low, ftp * high) for z, (low, high) in zones.items()}

def time_in_zones(series, zones):
    """
    Calculate time spent in each zone.
    IMPORTANT: 'zones' must be a DICTIONARY of {zone_id: (low_bound, high_bound)}, 
    not an FTP number.
    """
    # FIX: Do NOT call coggan_zones here. We assume 'zones' is already the dictionary.
    counts = {z: 0 for z in zones}
    
    if series is None or len(series) == 0:
        return counts

    # Drop NaNs
    clean_series = series.dropna()

    # Iterate and count
    for val in clean_series:
        for z, (low, high) in zones.items():
            if low <= val <= high:
                counts[z] += 1
                break
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

