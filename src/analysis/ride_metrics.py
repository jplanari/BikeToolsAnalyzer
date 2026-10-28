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


# --- Joint distributions (for cross-ride correlation analytics) --------------
# Stored per ride as sparse 2D histograms {(x_bin, y_bin): seconds}. Bins are
# identified by their lower edge. Histograms of different rides simply add,
# so any period can be analysed without re-reading the ride files.

POWER_BIN_W = 10
HR_BIN_BPM = 5
CAD_BIN_RPM = 5
# HR lags changes in power by tens of seconds: pairing raw 1 s samples blurs
# the relationship, so both channels are smoothed before pairing.
PAIR_SMOOTH_S = 30


def _hist2d(x, y, x_bin, y_bin):
    xb = (np.floor(x / x_bin) * x_bin).astype(int)
    yb = (np.floor(y / y_bin) * y_bin).astype(int)
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([xb, yb])).groupby(level=[0, 1]).sum()
    return {(int(a), int(b)): int(n) for (a, b), n in counts.items()}


def joint_histograms(df, moving=None):
    """
    {'p_hr': {(power_bin, hr_bin): s}, 'p_cad': {(power_bin, cadence_bin): s}}
    - p_hr: 30 s smoothed power and HR, moving and pedalling (raw power > 0,
      smoothed power >= 10 W)
    - p_cad: raw 1 s samples while pedalling (power > 0 and cadence > 0)
    """
    out = {}
    n = len(df)
    mov = np.ones(n, dtype=bool) if moving is None else _clean(moving).astype(bool)
    has = lambda c: c in df.columns and df[c].notna().any()

    if has('power') and has('hr'):
        ps = df['power'].rolling(PAIR_SMOOTH_S, min_periods=PAIR_SMOOTH_S // 2).mean().to_numpy()
        hs = df['hr'].rolling(PAIR_SMOOTH_S, min_periods=PAIR_SMOOTH_S // 2).mean().to_numpy()
        raw = _clean(df['power'])
        # Pedalling seconds only: coasting (raw 0 W, HR still elevated) and the
        # fade-out of the 30 s average would skew the low-power end.
        m = (mov & ~np.isnan(ps) & ~np.isnan(hs) & (hs > 0)
             & (np.nan_to_num(raw) > 0) & (ps >= POWER_BIN_W))
        if m.any():
            out['p_hr'] = _hist2d(ps[m], hs[m], POWER_BIN_W, HR_BIN_BPM)

    if has('power') and has('cadence'):
        p, c = _clean(df['power']), _clean(df['cadence'])
        m = mov & ~np.isnan(p) & ~np.isnan(c) & (p > 0) & (c > 0)
        if m.any():
            out['p_cad'] = _hist2d(p[m], c[m], POWER_BIN_W, CAD_BIN_RPM)
    return out


# --- Heart-rate recovery -----------------------------------------------------

def hr_recovery(power_series, hr_series, ftp, effort_s=60, recovery_s=60,
                effort_frac=1.05, easy_frac=0.5):
    """
    HR recovery after hard efforts: for each effort (60 s averaging >= 105 %
    FTP) followed by an easy minute (< 50 % FTP), HRR60 = HR at the end of
    the effort - HR 60 s later. Higher = faster recovery; a falling trend can
    signal fatigue. Returns (median_hrr_bpm, n_efforts) or (None, 0).
    """
    p, hr = _clean(power_series), _clean(hr_series)
    if not ftp or len(p) == 0 or len(p) != len(hr):
        return None, 0
    p_s = pd.Series(np.nan_to_num(p))
    hard = p_s.rolling(effort_s, min_periods=effort_s).mean().to_numpy()
    nxt = p_s[::-1].rolling(recovery_s, min_periods=recovery_s).mean()[::-1].shift(-1).to_numpy()

    n = len(p)
    ok = np.zeros(n, dtype=bool)
    ok[:n - recovery_s] = ((hard[:n - recovery_s] >= effort_frac * ftp)
                           & (nxt[:n - recovery_s] < easy_frac * ftp))

    drops, t = [], 0
    while t < n - recovery_s:
        if not ok[t]:
            t += 1
            continue
        # The condition starts holding a few seconds before the effort ends;
        # the true end is where (last minute - next minute) power peaks.
        run_end = t
        while run_end + 1 < n - recovery_s and ok[run_end + 1]:
            run_end += 1
        gap = hard[t:run_end + 1] - nxt[t:run_end + 1]
        end = t + int(np.argmax(gap))
        if not np.isnan(hr[end]) and not np.isnan(hr[end + recovery_s]):
            drops.append(hr[end] - hr[end + recovery_s])
        t = max(run_end + 1, end + recovery_s)  # efforts must not overlap
    if not drops:
        return None, 0
    return float(np.median(drops)), len(drops)
