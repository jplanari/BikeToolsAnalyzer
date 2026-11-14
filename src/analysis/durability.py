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


# --- durability normalised to body mass and to intensity -------------------------------------------------------------
# The systematic review of Sanchez-Jimenez et al. (2025) finds that absolute kJ ignore both the rider's size and the
# intensity: work above CP cut power by about 10-20 % after 2.5-15 kJ/kg, work below CP usually by under 5 %. So the
# same best-power-after-X question is also asked with X in kJ per kg of body mass, total and above CP.
TOTAL_KJ_PER_KG = (15.0, 30.0, 45.0)
ABOVE_CP_KJ_PER_KG = (2.5, 5.0, 7.5)
BASES = ('total_kjkg', 'above_cp_kjkg')


def _after(p, cumulative, thresholds, durations):
    out = {}
    for x in thresholds:
        idx = int(np.searchsorted(cumulative, x))
        if idx >= len(p) - 1:
            continue
        curve = power_curve(pd.Series(p[idx + 1:]), list(durations))
        if curve:
            out[float(x)] = curve
    return out


def work_curves(power_series, weight_kg, cp=None, durations=DURABILITY_DURATIONS):
    """
    {basis: {threshold: {duration: watts}}}: best power in the rest of the ride after `threshold` kJ/kg of work in
    total ('total_kjkg') and above CP ('above_cp_kjkg'). The second needs a CP; neither is computed without a weight.
    """
    if not weight_kg or weight_kg <= 0:
        return {}
    p = pd.Series(power_series).fillna(0).to_numpy(dtype=float)
    if len(p) == 0:
        return {}
    out = {}
    total = np.cumsum(p) / 1000.0 / weight_kg
    got = _after(p, total, TOTAL_KJ_PER_KG, durations)
    if got:
        out['total_kjkg'] = got
    if cp and cp > 0:
        above = np.cumsum(np.clip(p - cp, 0, None)) / 1000.0 / weight_kg
        got = _after(p, above, ABOVE_CP_KJ_PER_KG, durations)
        if got:
            out['above_cp_kjkg'] = got
    return out


def work_table(fresh_records, work_records, basis, start=None, end=None):
    """
    Per threshold (kJ/kg) and duration for one basis: fresh best, best after the threshold, retention (%) in the
    period. fresh_records: duration, watts, date; work_records: basis, threshold, duration, watts, date.
    """
    cols = ['threshold', 'duration', 'fresh', 'fatigued', 'retention_pct']
    if work_records is None or work_records.empty or fresh_records is None or fresh_records.empty:
        return pd.DataFrame(columns=cols)

    def window(df):
        if start is not None:
            df = df[df['date'] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df['date'] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        return df

    fresh = window(fresh_records).groupby('duration')['watts'].max()
    tired = window(work_records[work_records['basis'] == basis]).groupby(['threshold', 'duration'])['watts'].max()
    rows = [{'threshold': float(x), 'duration': int(d), 'fresh': float(fresh[d]), 'fatigued': float(w),
             'retention_pct': float(w / fresh[d] * 100)}
            for (x, d), w in tired.items() if fresh.get(d)]
    return pd.DataFrame(rows, columns=cols).sort_values(['threshold', 'duration']).reset_index(drop=True) if rows \
        else pd.DataFrame(columns=cols)
