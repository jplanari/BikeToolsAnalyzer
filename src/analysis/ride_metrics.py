"""
Per-ride quality and load metrics (pure functions, 1 Hz input).

- VI (variability index) = NP / average power
- EF (efficiency factor)  = NP / average heart rate (moving samples)
- Aerobic decoupling (Pa:HR) = drift of power/HR between the first and
  second half of the moving time, in percent (positive = HR drifted up)
- hrTSS: TSS estimate from heart rate for rides without power
"""
import numpy as np
import pandas as pd

from src.analysis.power import NP, TSS, IF, coggan_zones, time_in_zones
from src.analysis.hr import hr_effort_zones

# Decoupling needs a steady block long enough for HR drift to show
MIN_DECOUPLING_S = 20 * 60


def _clean(series):
    if series is None:
        return np.array([], dtype=float)
    return pd.Series(series).to_numpy(dtype=float)


def variability_index(power_series):
    p = _clean(power_series)
    p = p[~np.isnan(p)]
    if len(p) == 0 or p.mean() <= 0:
        return None
    return float(NP(pd.Series(p)) / p.mean())


def efficiency_factor(norm_power, hr_series, moving=None):
    hr = _clean(hr_series)
    if moving is not None:
        hr = hr[_clean(moving).astype(bool)]
    hr = hr[~np.isnan(hr) & (hr > 0)]
    if not norm_power or len(hr) == 0:
        return None
    return float(norm_power / hr.mean())


def aerobic_decoupling(power_series, hr_series, moving=None):
    """
    Pa:HR decoupling in percent: (ratio_1st_half - ratio_2nd_half) / ratio_1st_half,
    where ratio = mean power / mean HR over each half of the moving samples
    that have both power and HR. None if under 20 minutes of usable data.
    """
    p, hr = _clean(power_series), _clean(hr_series)
    if len(p) == 0 or len(p) != len(hr):
        return None
    mask = ~np.isnan(p) & ~np.isnan(hr) & (hr > 0)
    if moving is not None:
        mask &= _clean(moving).astype(bool)
    p, hr = p[mask], hr[mask]
    if len(p) < MIN_DECOUPLING_S:
        return None

    half = len(p) // 2
    r1 = p[:half].mean() / hr[:half].mean()
    r2 = p[half:].mean() / hr[half:].mean()
    if r1 <= 0:
        return None
    return float((r1 - r2) / r1 * 100)


def hr_tss(hr_series, lthr, moving=None):
    """
    Heart-rate TSS approximation: hours * (avg HR / LTHR)^2 * 100 over the
    moving time, i.e. one hour at LTHR = 100, same form as power TSS with
    IF = avgHR / LTHR. Less precise than power TSS (HR lags efforts and
    drifts), so rides keep a tss_source flag.
    """
    hr = _clean(hr_series)
    if moving is not None:
        hr = hr[_clean(moving).astype(bool)]
    hr = hr[~np.isnan(hr) & (hr > 0)]
    if not lthr or len(hr) == 0:
        return None
    hours = len(hr) / 3600.0
    return float(hours * (hr.mean() / lthr) ** 2 * 100)


def zone_seconds(df, ftp, lthr):
    """{'power': {zone: s}, 'hr': {zone: s}} for the channels the ride has."""
    out = {}
    if ftp and 'power' in df.columns and df['power'].notna().any():
        out['power'] = {f"Z{z}": s for z, s in time_in_zones(df['power'], coggan_zones(ftp)).items()}
    if lthr and 'hr' in df.columns and df['hr'].notna().any():
        out['hr'] = time_in_zones(df['hr'], hr_effort_zones(lthr))
    return out


def training_stress(norm_power, duration_s, ftp, hr_series, lthr, moving=None):
    """
    Returns (tss, intensity_factor, source). Power TSS when there is power
    and an FTP; otherwise hrTSS when there is HR and an LTHR; else (0, 0, None).
    """
    if norm_power and ftp:
        return TSS(norm_power, duration_s, ftp), IF(norm_power, ftp), 'power'
    hrt = hr_tss(hr_series, lthr, moving)
    if hrt is not None:
        # IF stays a power concept (stored if_factor = NP / FTP), so 0 here
        return hrt, 0.0, 'hr'
    return 0.0, 0.0, None
