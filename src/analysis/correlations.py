"""
Cross-ride analytics from stored joint histograms (see ride_metrics.joint_histograms).

All functions take a DataFrame of histogram cells with columns x, y, seconds
(optionally date / ride_id). x = power bin lower edge (W); y = HR (bpm) or
cadence (rpm) bin lower edge. Cells from many rides are simply summed.
"""
import numpy as np
import pandas as pd

from src.analysis.power import coggan_zones
from src.analysis.ride_metrics import POWER_BIN_W, HR_BIN_BPM, CAD_BIN_RPM

QUADRANT_NAMES = {
    'QI': 'High force · High cadence',
    'QII': 'High force · Low cadence',
    'QIII': 'Low force · Low cadence',
    'QIV': 'Low force · High cadence',
}


def _centers(h, y_bin):
    x = h['x'].to_numpy(dtype=float) + POWER_BIN_W / 2
    y = h['y'].to_numpy(dtype=float) + y_bin / 2
    w = h['seconds'].to_numpy(dtype=float)
    return x, y, w


def aggregate(h):
    """Sums cells across rides -> DataFrame(x, y, seconds)."""
    if h is None or h.empty:
        return pd.DataFrame(columns=['x', 'y', 'seconds'])
    return h.groupby(['x', 'y'], as_index=False)['seconds'].sum()


def linear_fit(h, y_bin=HR_BIN_BPM):
    """
    Time-weighted Pearson r and least-squares line y = a + b*x on bin centres.
    Returns dict(r, slope, intercept, hours) or None with < 10 min of data.
    """
    h = aggregate(h)
    if h.empty or h['seconds'].sum() < 600:
        return None
    x, y, w = _centers(h, y_bin)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.average((x - mx) ** 2, weights=w)
    syy = np.average((y - my) ** 2, weights=w)
    sxy = np.average((x - mx) * (y - my), weights=w)
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    return {'r': float(sxy / np.sqrt(sxx * syy)), 'slope': float(slope),
            'intercept': float(my - slope * mx), 'hours': float(w.sum() / 3600)}


def mean_y_by_power(h, y_bin=HR_BIN_BPM, min_seconds=60):
    """Average HR (or cadence) at each power bin with enough data."""
    h = aggregate(h)
    if h.empty:
        return pd.DataFrame(columns=['power', 'mean', 'seconds'])
    h = h.assign(yc=h['y'] + y_bin / 2, wy=lambda d: d['yc'] * d['seconds'])
    g = h.groupby('x').agg(seconds=('seconds', 'sum'), wy=('wy', 'sum')).reset_index()
    g = g[g['seconds'] >= min_seconds]
    return pd.DataFrame({'power': g['x'] + POWER_BIN_W / 2, 'mean': g['wy'] / g['seconds'],
                         'seconds': g['seconds']}).reset_index(drop=True)


def y_at_power(h, power_ref, y_bin=HR_BIN_BPM, half_width=15, min_seconds=300):
    """Average HR (or cadence) while riding within ±half_width W of power_ref."""
    h = aggregate(h)
    if h.empty:
        return None
    xc = h['x'] + POWER_BIN_W / 2
    sel = h[(xc - power_ref).abs() <= half_width]
    if sel['seconds'].sum() < min_seconds:
        return None
    return float(np.average(sel['y'] + y_bin / 2, weights=sel['seconds']))


def weighted_median_y(h, y_bin=CAD_BIN_RPM):
    """Median of y over time (e.g. preferred cadence while pedalling), interpolated within bins."""
    h = aggregate(h)
    if h.empty:
        return None
    g = h.groupby('y')['seconds'].sum().sort_index()
    cum = g.cumsum().to_numpy()
    half = cum[-1] / 2
    i = int(np.searchsorted(cum, half))
    # Interpolate inside the median bin (assumes uniform spread within it):
    # bin-centre medians would only move in 5 rpm steps.
    before = cum[i - 1] if i > 0 else 0.0
    return float(g.index[i] + (half - before) / g.iloc[i] * y_bin)


def weekly(h, fn):
    """Applies fn to each ISO week's cells -> Series indexed by week start."""
    if h is None or h.empty:
        return pd.Series(dtype=float)
    h = h.dropna(subset=['date'])
    week = h['date'].dt.to_period('W-SUN').dt.start_time
    out = {wk: fn(grp) for wk, grp in h.groupby(week)}
    return pd.Series(out, dtype=float).dropna().sort_index()


def cadence_by_zone(h_pcad, ftp):
    """Mean cadence and time share for each Coggan power zone."""
    h = aggregate(h_pcad)
    if h.empty or not ftp:
        return pd.DataFrame(columns=['zone', 'cadence', 'share'])
    xc = h['x'] + POWER_BIN_W / 2
    yc = h['y'] + CAD_BIN_RPM / 2
    rows = []
    for z, (lo, hi) in coggan_zones(ftp).items():
        m = (xc >= lo) & (xc < hi)
        sec = h.loc[m, 'seconds'].sum()
        if sec:
            rows.append({'zone': f"Z{z}", 'cadence': float(np.average(yc[m], weights=h.loc[m, 'seconds'])),
                         'share': sec / h['seconds'].sum()})
    return pd.DataFrame(rows)


def pedal_force_velocity(power, cadence_rpm, crank_m):
    """Coggan's AEPF (N) and CPV (m/s) from power and cadence."""
    cpv = np.asarray(cadence_rpm, dtype=float) * crank_m * 2 * np.pi / 60
    with np.errstate(divide='ignore', invalid='ignore'):
        aepf = np.asarray(power, dtype=float) / cpv
    return aepf, cpv


def quadrants(h_pcad, ftp, ref_cadence):
    """
    Quadrant analysis (Coggan): splits pedalling time by force (above/below
    the force needed to make FTP at ref_cadence) and by cadence (above/below
    ref_cadence). Crank length cancels out of the force test:
        AEPF > AEPF_ftp  <=>  power / cadence > FTP / ref_cadence
    Returns {quadrant: share of time} or None.
    """
    h = aggregate(h_pcad)
    if h.empty or not ftp or not ref_cadence:
        return None
    xc = h['x'] + POWER_BIN_W / 2
    yc = h['y'] + CAD_BIN_RPM / 2
    high_force = (xc / yc) > (ftp / ref_cadence)
    high_cad = yc > ref_cadence
    total = h['seconds'].sum()
    share = lambda m: float(h.loc[m, 'seconds'].sum() / total)
    return {'QI': share(high_force & high_cad), 'QII': share(high_force & ~high_cad),
            'QIII': share(~high_force & ~high_cad), 'QIV': share(~high_force & high_cad)}


def endurance_rides(stats, min_minutes=60, max_vi=1.10):
    """Rides comparable for EF/decoupling trends: long and steady."""
    if stats is None or stats.empty:
        return stats
    m = ((stats['moving_s'].fillna(0) >= min_minutes * 60)
         & (stats['vi'].fillna(99) <= max_vi) & stats['ef'].notna())
    return stats[m]
