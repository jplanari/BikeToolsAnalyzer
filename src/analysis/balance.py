"""
Left/right leg power balance, and the pedal metrics that come with the same sensors.

Dual-sided power meters (pedals, dual crank arms) record how the power is split between the legs. A FIT file
stores it as `left_right_balance`: bit 7 says the percentage in the lower 7 bits belongs to the right pedal
(when it is clear, to the left). Everything here works on the RIGHT leg's share in percent, so 50 is an even
split, above 50 the right leg does more, below 50 the left one does.

Per ride the app keeps a few numbers (ride_metrics, below) and two joint histograms (see
ride_metrics.joint_histograms): power x balance ('p_bal') and cadence x balance ('c_bal'). Histograms of many
rides add up, so every period can be analysed without re-reading the files, like power x HR and power x cadence.

Torque effectiveness and pedal smoothness (percent, per leg) are only kept as ride averages.
"""
import numpy as np
import pandas as pd

from src.analysis.power import coggan_zones
from src.analysis.ride_metrics import POWER_BIN_W, CAD_BIN_RPM, BAL_BIN_PCT
from src.i18n import mark

RIGHT_FLAG = 0x80
BALANCED_TOLERANCE_PCT = 2.0         # within 48-52 counts as even
NOTICEABLE_PCT = 4.0                 # beyond 46-54 is worth a look
MIN_DRIFT_S = 20 * 60                # pedalling time needed to compare the start and the end of a ride

# Rough guide words, shown next to the numbers (a rule of thumb, not a diagnosis)
VERDICTS = {
    'even': mark('Even'),
    'slight': mark('Slightly uneven'),
    'uneven': mark('Uneven'),
}

PEDAL_COLUMNS = ('te_l', 'te_r', 'ps_l', 'ps_r')       # torque effectiveness and pedal smoothness, left / right
RIDE_KEYS = ('bal_right_pct', 'bal_first_pct', 'bal_last_pct', 'bal_s') + PEDAL_COLUMNS


# --- reading the sensor -------------------------------------------------------------------------------------------

def right_percent(raw):
    """The right leg's share (0-100) from raw FIT `left_right_balance` values; NaN where there is none.
    A value of 0 (or 128, flag only) is what many devices write when the sensor is absent: treated as missing."""
    v = pd.to_numeric(pd.Series(raw), errors='coerce')
    v = v.where(v == np.floor(v))                                # only whole numbers are raw FIT values
    pct = v % 128
    right = pct.where(v >= RIGHT_FLAG, 100 - pct)
    return right.where((pct > 0) & (pct <= 100)).astype(float)


def percent_channel(raw):
    """A per-leg pedal metric in percent (0-100); anything else is a bad reading."""
    v = pd.to_numeric(pd.Series(raw), errors='coerce')
    return v.where((v > 0) & (v <= 100)).astype(float)


def has_balance(df):
    return df is not None and 'lr_balance' in df.columns and df['lr_balance'].notna().any()


def verdict(right_pct):
    """Key into VERDICTS for a right-leg share, or None."""
    if right_pct is None or pd.isna(right_pct):
        return None
    gap = abs(right_pct - 50.0)
    return 'even' if gap <= BALANCED_TOLERANCE_PCT else 'slight' if gap <= NOTICEABLE_PCT else 'uneven'


def leg_watts(avg_power, right_pct):
    """(left W, right W) of an average power for a right-leg share, or (None, None)."""
    if avg_power is None or right_pct is None or pd.isna(avg_power) or pd.isna(right_pct):
        return None, None
    return float(avg_power * (100 - right_pct) / 100), float(avg_power * right_pct / 100)


def split_text(right_pct, digits=0):
    """'48 / 52' (left / right) for a right-leg share."""
    if right_pct is None or pd.isna(right_pct):
        return None
    return f"{100 - right_pct:.{digits}f} / {right_pct:.{digits}f}"


# --- one ride -----------------------------------------------------------------------------------------------------

def _pedalling(df, moving=None):
    p = pd.to_numeric(df['power'], errors='coerce') if 'power' in df.columns else pd.Series(np.nan, index=df.index)
    m = df['lr_balance'].notna() & (p > 0)
    if moving is not None:
        m &= pd.Series(np.asarray(moving, dtype=bool), index=df.index)
    return m, p


def _weighted(bal, power):
    total = power.sum()
    return float((bal * power).sum() / total) if total > 0 else None


def ride_metrics(df, moving=None):
    """The per-ride numbers kept in the database (RIDE_KEYS); every value None when the ride has no balance.
    bal_right_pct is power-weighted (the share of the work each leg did); first / last are the same over the
    first and last third of the pedalling time, so a drift shows fatigue building on one side."""
    out = {k: None for k in RIDE_KEYS}
    if not has_balance(df) or 'power' not in df.columns:
        return out
    m, p = _pedalling(df, moving)
    if not m.any():
        return out
    bal, pw = df.loc[m, 'lr_balance'], p[m]
    out['bal_s'] = int(m.sum())
    out['bal_right_pct'] = _weighted(bal, pw)
    if m.sum() >= MIN_DRIFT_S:
        third = int(m.sum()) // 3
        out['bal_first_pct'] = _weighted(bal.iloc[:third], pw.iloc[:third])
        out['bal_last_pct'] = _weighted(bal.iloc[-third:], pw.iloc[-third:])
    for col in PEDAL_COLUMNS:
        if col in df.columns:
            v = df.loc[m, col].dropna()
            out[col] = float(v.mean()) if len(v) >= 60 else None
    return out


def ride_series(df, moving=None, window_s=30):
    """Right-leg share smoothed for the ride page (NaN while not pedalling)."""
    m, _ = _pedalling(df, moving)
    return df['lr_balance'].where(m).rolling(window_s, min_periods=max(5, window_s // 3)).mean()


# --- a period, from stored histograms -----------------------------------------------------------------------------
# Cells: DataFrame(x, y, seconds [, date, ride_id]); x = power (W) or cadence (rpm) bin, y = right-leg % bin.

def _agg(cells):
    if cells is None or cells.empty:
        return pd.DataFrame(columns=['x', 'y', 'seconds'])
    return cells.groupby(['x', 'y'], as_index=False)['seconds'].sum()


def right_share(cells, power_weighted=True):
    """Mean right-leg share of the period. Power-weighted (default) is the share of the work; time-weighted is how
    the sensor's own average usually reads. None below 10 minutes of data."""
    h = _agg(cells)
    if h.empty or h['seconds'].sum() < 600:
        return None
    w = h['seconds'].astype(float)
    if power_weighted:
        w = w * (h['x'] + POWER_BIN_W / 2)
    if w.sum() <= 0:
        return None
    return float(np.average(h['y'] + BAL_BIN_PCT / 2, weights=w))


def mean_power(cells):
    """Average power (W) over the cells' pedalling time, or None."""
    h = _agg(cells)
    if h.empty or h['seconds'].sum() <= 0:
        return None
    return float(np.average(h['x'] + POWER_BIN_W / 2, weights=h['seconds']))


def spread(cells):
    """Time-weighted standard deviation of the balance, in points: how much it wanders."""
    h = _agg(cells)
    if h.empty or h['seconds'].sum() < 600:
        return None
    y = (h['y'] + BAL_BIN_PCT / 2).to_numpy(dtype=float)
    w = h['seconds'].to_numpy(dtype=float)
    m = np.average(y, weights=w)
    return float(np.sqrt(np.average((y - m) ** 2, weights=w)))


def time_even(cells, tolerance=BALANCED_TOLERANCE_PCT):
    """Share of the time the right leg was within 50 ± tolerance."""
    h = _agg(cells)
    if h.empty:
        return None
    yc = h['y'] + BAL_BIN_PCT / 2
    return float(h.loc[(yc - 50).abs() <= tolerance, 'seconds'].sum() / h['seconds'].sum())


def by_power(cells, min_seconds=60):
    """Mean right-leg share at each power: DataFrame(power, mean, seconds)."""
    h = _agg(cells)
    if h.empty:
        return pd.DataFrame(columns=['power', 'mean', 'seconds'])
    h = h.assign(wy=(h['y'] + BAL_BIN_PCT / 2) * h['seconds'])
    g = h.groupby('x').agg(seconds=('seconds', 'sum'), wy=('wy', 'sum')).reset_index()
    g = g[g['seconds'] >= min_seconds]
    return pd.DataFrame({'power': g['x'] + POWER_BIN_W / 2, 'mean': g['wy'] / g['seconds'],
                         'seconds': g['seconds']}).reset_index(drop=True)


def by_cadence(cells, min_seconds=60):
    """Mean right-leg share at each cadence: DataFrame(cadence, mean, seconds)."""
    h = _agg(cells)
    if h.empty:
        return pd.DataFrame(columns=['cadence', 'mean', 'seconds'])
    h = h.assign(wy=(h['y'] + BAL_BIN_PCT / 2) * h['seconds'])
    g = h.groupby('x').agg(seconds=('seconds', 'sum'), wy=('wy', 'sum')).reset_index()
    g = g[g['seconds'] >= min_seconds]
    return pd.DataFrame({'cadence': g['x'] + CAD_BIN_RPM / 2, 'mean': g['wy'] / g['seconds'],
                         'seconds': g['seconds']}).reset_index(drop=True)


def by_zone(cells, ftp, min_seconds=60):
    """Right-leg share (time-weighted) and share of the time in each Coggan power zone, from 'p_bal' cells."""
    h = _agg(cells)
    if h.empty or not ftp:
        return pd.DataFrame(columns=['zone', 'right', 'share'])
    xc = h['x'] + POWER_BIN_W / 2
    yc = h['y'] + BAL_BIN_PCT / 2
    rows = []
    for z, (lo, hi) in coggan_zones(ftp).items():
        m = (xc >= lo) & (xc < hi)
        sec = h.loc[m, 'seconds'].sum()
        if sec >= min_seconds:
            rows.append({'zone': f"Z{z}", 'right': float(np.average(yc[m], weights=h.loc[m, 'seconds'])),
                         'share': sec / h['seconds'].sum()})
    return pd.DataFrame(rows, columns=['zone', 'right', 'share'])


def intensity_slope(cells):
    """Points of right-leg share gained per +100 W (least squares on power-bin means), or None.
    Positive = the right leg takes more as the effort gets harder; negative = the left one does."""
    c = by_power(cells, min_seconds=120)
    if len(c) < 4:
        return None
    w = c['seconds'].to_numpy(dtype=float)
    x, y = c['power'].to_numpy(dtype=float), c['mean'].to_numpy(dtype=float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.average((x - mx) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / sxx * 100) if sxx > 0 else None


# --- many rides ---------------------------------------------------------------------------------------------------

def ride_drift(stats):
    """Per-ride change of the right-leg share from the first to the last third of the ride (points), as a Series
    indexed by date: positive = the right leg took over as the ride went on."""
    if stats is None or stats.empty or 'bal_last_pct' not in stats.columns:
        return pd.Series(dtype=float)
    s = stats.dropna(subset=['bal_first_pct', 'bal_last_pct'])
    return (s['bal_last_pct'] - s['bal_first_pct']).set_axis(s['date'])


def ride_share(stats):
    """Per-ride right-leg share (power-weighted), indexed by date."""
    if stats is None or stats.empty or 'bal_right_pct' not in stats.columns:
        return pd.Series(dtype=float)
    s = stats.dropna(subset=['bal_right_pct'])
    return s['bal_right_pct'].set_axis(s['date'])


def pedal_table(stats):
    """Mean torque effectiveness / pedal smoothness per leg over the rides that have them:
    {'te': (left, right), 'ps': (left, right)}, each entry None when never recorded."""
    out = {'te': (None, None), 'ps': (None, None)}
    if stats is None or stats.empty:
        return out
    for key, cols in (('te', ('te_l', 'te_r')), ('ps', ('ps_l', 'ps_r'))):
        if all(c in stats.columns for c in cols):
            vals = tuple(float(stats[c].mean()) if stats[c].notna().any() else None for c in cols)
            out[key] = vals
    return out
