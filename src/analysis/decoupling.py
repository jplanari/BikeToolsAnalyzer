"""
Aerobic decoupling (Friel's method): how much the power you produce per heart beat drops from the first half of a
ride to the second. A small drop (under about 5 %) means the aerobic base holds up over a long steady ride; a large one
means fatigue or dehydration is catching up. Only riding time counts (power above zero), and the result is only
meaningful for steady, long efforts: it is reported with its length so the page can say how much to trust it.
"""
import numpy as np
import pandas as pd

MIN_SECONDS = 20 * 60          # shorter rides give no useful drift at all
RELIABLE_SECONDS = 60 * 60     # Friel's own guide: one hour or more of steady riding
GOOD_PCT = 5.0                 # below this the aerobic base is said to hold up


def decoupling(df):
    """{'pct', 'minutes', 'first', 'second', 'reliable'} or None when the ride has no heart rate or is too short.
    'first' and 'second' are the power-per-beat (W per bpm) of each half; 'pct' is the drop from first to second.
    A negative pct (the second half was more efficient) is returned as it is: it usually means warm-up or a late
    climb, not a real gain."""
    if df is None or df.empty or not {'power', 'hr', 'time'} <= set(df.columns):
        return None
    d = pd.DataFrame({'t': (pd.to_datetime(df['time']) - pd.to_datetime(df['time'].iloc[0])).dt.total_seconds(),
                      'p': pd.to_numeric(df['power'], errors='coerce'),
                      'h': pd.to_numeric(df['hr'], errors='coerce')}).dropna()
    d = d[(d['p'] > 0) & (d['h'] > 0)]
    if len(d) < 2:
        return None
    total = float(d['t'].max() - d['t'].min())
    if total < MIN_SECONDS:
        return None
    mid = d['t'].min() + total / 2.0
    first, second = d[d['t'] <= mid], d[d['t'] > mid]
    if first.empty or second.empty:
        return None
    ef1 = float(first['p'].mean() / first['h'].mean())
    ef2 = float(second['p'].mean() / second['h'].mean())
    if not (np.isfinite(ef1) and np.isfinite(ef2)) or ef1 <= 0:
        return None
    return {'pct': (ef1 - ef2) / ef1 * 100.0, 'minutes': total / 60.0, 'first': ef1, 'second': ef2,
            'reliable': total >= RELIABLE_SECONDS}
