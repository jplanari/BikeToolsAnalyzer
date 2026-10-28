"""
Power profile across rides, from the per-ride mean-maximal power records
(DataFrame with columns duration, watts, date, ride_id; see
db.get_power_records).
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analysis.power import estimate_critical_power

PROFILE_DURATIONS = {'5 s': 5, '1 min': 60, '5 min': 300, '20 min': 1200}


def best_curve(records, start=None, end=None):
    """
    Best power per duration within [start, end] (inclusive dates; None =
    open). Returns DataFrame(duration, watts, date, ride_id), one row per
    duration, with the date and ride where each best was set.
    """
    cols = ['duration', 'watts', 'date', 'ride_id']
    if records is None or records.empty:
        return pd.DataFrame(columns=cols)
    r = records.dropna(subset=['watts'])
    if start is not None:
        r = r[r['date'] >= pd.Timestamp(start)]
    if end is not None:
        r = r[r['date'] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if r.empty:
        return pd.DataFrame(columns=cols)
    return _best_of_sorted(_sort_for_best(r))


def _sort_for_best(r):
    # Best first within each duration; ties go to the earliest ride (the record was set first)
    return r.sort_values(['duration', 'watts', 'date'], ascending=[True, False, True])


def _best_of_sorted(r):
    cols = ['duration', 'watts', 'date', 'ride_id']
    return r.drop_duplicates('duration', keep='first')[cols].reset_index(drop=True)


def window_curves(records, today=None):
    """Best curves for the standard comparison windows."""
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    windows = {
        'Last 42 days': today - pd.Timedelta(days=41),
        'Last 90 days': today - pd.Timedelta(days=89),
        f'Season {today.year}': pd.Timestamp(year=today.year, month=1, day=1),
        'All time': None,
    }
    return {label: best_curve(records, start, today) for label, start in windows.items()}


def as_dict(curve):
    """best_curve() -> {duration: watts}, the format estimate_critical_power uses."""
    if curve is None or curve.empty:
        return {}
    return {int(d): float(w) for d, w in zip(curve['duration'], curve['watts'])}


def _morton(t, cp, w_prime, pmax):
    # Morton's 3-parameter model: P(t) = CP + W' / (t + W' / (Pmax - CP))
    return cp + w_prime / (t + w_prime / (pmax - cp))


def fit_cp3(curve_dict, min_s=60, max_s=1800):
    """
    Three-parameter critical power model (Morton 1996) fitted to a
    mean-maximal power curve between min_s and max_s. Unlike the 2-parameter
    model it also estimates the maximal instantaneous power (Pmax), and it
    fits short efforts better.
    Returns dict(cp, w_prime, pmax, rmse) or None (needs >= 4 points).
    """
    pts = sorted((d, w) for d, w in (curve_dict or {}).items()
                 if min_s <= d <= max_s and w and np.isfinite(w))
    if len(pts) < 4:
        return None
    t = np.array([d for d, _ in pts], dtype=float)
    p = np.array([w for _, w in pts], dtype=float)

    two = estimate_critical_power(curve_dict) or {}
    cp0 = float(np.clip(two.get('cp', p.min() * 0.9), 50, p.min() - 1))
    w0 = float(np.clip(two.get('w_prime', 20000), 2000, 60000))
    pmax0 = float(max(curve_dict.get(1, 0) or 0, p.max() * 1.5, cp0 + 100))
    try:
        (cp, w_prime, pmax), _ = curve_fit(
            _morton, t, p, p0=[cp0, w0, pmax0],
            bounds=([20, 500, cp0 + 1], [p.min(), 100000, 5000]), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    rmse = float(np.sqrt(np.mean((_morton(t, cp, w_prime, pmax) - p) ** 2)))
    return {'cp': float(cp), 'w_prime': float(w_prime), 'pmax': float(pmax), 'rmse': rmse}


def cp_history(records, window_days=90, freq='W', today=None):
    """
    How CP / W' / FTP estimates evolved: at the end of each week (or month),
    the best curve of the preceding `window_days` is modelled.
    - cp, w_prime: 2-parameter CP model (3-20 min efforts)
    - ftp_20min:   95 % of best 20-minute power
    Weeks without enough maximal efforts give NaN.
    """
    cols = ['cp', 'w_prime', 'ftp_20min']
    if records is None or records.empty:
        return pd.DataFrame(columns=cols)
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    first = records['date'].min().normalize()
    step = 'W-SUN' if freq == 'W' else 'ME'
    ends = pd.date_range(first, today, freq=step)
    if len(ends) == 0 or ends[-1] < today:
        ends = ends.append(pd.DatetimeIndex([today]))
    rows = {}
    ranked = _sort_for_best(records.dropna(subset=['watts']))     # sort once, filter per window
    for end in ends:
        start = end - pd.Timedelta(days=window_days - 1)
        in_window = ranked[(ranked['date'] >= start) & (ranked['date'] < end + pd.Timedelta(days=1))]
        curve = as_dict(_best_of_sorted(in_window)) if not in_window.empty else {}
        cp = estimate_critical_power(curve) if curve else None
        rows[end] = {
            'cp': cp['cp'] if cp else np.nan,
            'w_prime': cp['w_prime'] if cp else np.nan,
            'ftp_20min': 0.95 * curve[1200] if 1200 in curve else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient='index', columns=cols)
    out.index.name = 'date'
    return out


MIN_CHANGE_PCT = 3.0
MIN_CHANGE_W = 5


def suggest_threshold(records, current_ftp, today=None, window_days=90):
    """
    Does your recent riding suggest a different FTP? Uses the rolling `window_days` best curve:
    FTP estimate = 95 % of the best 20-minute power, and CP from the 3-20 min efforts.
    Returns None without a 20-minute effort in the window or when the estimate is within
    max(3 %, 5 W) of `current_ftp`; otherwise
      {'ftp', 'change_w', 'change_pct', 'direction' ('up'/'down'), 'best_20min_w',
       'best_20min_date', 'cp', 'w_prime', 'window_days'}.
    A 'down' result only means no recent effort backs your FTP up; an easy period does not
    lower your fitness, so callers should word it softly.
    """
    if records is None or records.empty or not current_ftp:
        return None
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    best = best_curve(records, today - pd.Timedelta(days=window_days - 1), today)
    row = best[best['duration'] == 1200]
    if row.empty:
        return None
    best20 = float(row['watts'].iloc[0])
    est = 0.95 * best20
    diff = est - current_ftp
    if abs(diff) < max(current_ftp * MIN_CHANGE_PCT / 100, MIN_CHANGE_W):
        return None
    cp = estimate_critical_power(as_dict(best))
    return {'ftp': int(round(est)), 'change_w': int(round(diff)), 'change_pct': diff / current_ftp * 100,
            'direction': 'up' if diff > 0 else 'down', 'best_20min_w': best20,
            'best_20min_date': pd.Timestamp(row['date'].iloc[0]),
            'cp': cp['cp'] if cp else None, 'w_prime': cp['w_prime'] if cp else None,
            'window_days': window_days}


def power_profile(curve_dict, weight_kg, ftp):
    """
    W/kg at 5 s, 1 min, 5 min and 20 min, and each as a multiple of FTP
    (e.g. 5-min power at 1.2x FTP). Describes the shape of your power curve
    without population categories.
    """
    rows = []
    for label, d in PROFILE_DURATIONS.items():
        w = (curve_dict or {}).get(d)
        rows.append({
            'duration': label,
            'watts': w,
            'w_per_kg': (w / weight_kg) if w and weight_kg else None,
            'x_ftp': (w / ftp) if w and ftp else None,
        })
    return pd.DataFrame(rows)
