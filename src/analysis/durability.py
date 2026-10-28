"""
Durability: how much of your fresh power you can still produce after a given
amount of work. Per ride we store the mean-maximal power of the part of the
ride AFTER 1000 / 2000 / 3000 kJ; across rides it is compared with the fresh
(whole-ride) best of the same period.
"""
import numpy as np
import pandas as pd

from src.analysis.power import power_curve

KJ_THRESHOLDS = (1000, 2000, 3000)
DURABILITY_DURATIONS = (5, 60, 300, 1200, 3600)


def fatigued_curves(power_series, thresholds_kj=KJ_THRESHOLDS, durations=DURABILITY_DURATIONS):
    """
    {kj: {duration: watts}} for the remainder of the ride after `kj` of work
    (1 Hz samples). Thresholds the ride never reaches are omitted.
    """
    p = pd.Series(power_series).fillna(0).to_numpy(dtype=float)
    if len(p) == 0:
        return {}
    cum_kj = np.cumsum(p) / 1000.0
    out = {}
    for kj in thresholds_kj:
        idx = int(np.searchsorted(cum_kj, kj))   # first second with >= kj done
        if idx >= len(p) - 1:
            continue
        curve = power_curve(pd.Series(p[idx + 1:]), list(durations))
        if curve:
            out[int(kj)] = curve
    return out


def durability_table(fresh_records, kj_records, start=None, end=None):
    """
    Per threshold and duration: fresh best, best after the threshold, and
    retention (%) in the period. Both inputs are DataFrames with duration,
    watts, date (kj_records also has kj).
    """
    cols = ['kj', 'duration', 'fresh', 'fatigued', 'retention_pct']
    if kj_records is None or kj_records.empty or fresh_records is None or fresh_records.empty:
        return pd.DataFrame(columns=cols)

    def window(df):
        if start is not None:
            df = df[df['date'] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df['date'] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        return df

    fresh = window(fresh_records).groupby('duration')['watts'].max()
    tired = window(kj_records).groupby(['kj', 'duration'])['watts'].max()
    rows = []
    for (kj, d), w in tired.items():
        f = fresh.get(d)
        if f:
            rows.append({'kj': int(kj), 'duration': int(d), 'fresh': float(f), 'fatigued': float(w),
                         'retention_pct': float(w / f * 100)})
    return pd.DataFrame(rows, columns=cols)
