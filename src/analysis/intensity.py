"""
Training intensity distribution (TID) in three zones and the
polarization index (Treff et al., Front. Physiol. 2019, doi:10.3389/fphys.2019.00707).

Three-zone mapping of the stored 7 power zones (Coggan) / 6 HR zones (Friel):
    low  = Z1-Z2  (below ~75 % FTP / ~90 % LTHR, roughly below the 1st threshold)
    mid  = Z3-Z4  (tempo to threshold)
    high = Z5+    (above FTP / LTHR)
"""
import numpy as np
import pandas as pd

LOW, MID, HIGH = ('Z1', 'Z2'), ('Z3', 'Z4'), ('Z5', 'Z6', 'Z7')


def zone_group(zone_name):
    """'Z3' or 'Z3 (Tempo)' -> 'mid'."""
    z = str(zone_name).split(' ')[0]
    if z in LOW:
        return 'low'
    if z in MID:
        return 'mid'
    if z in HIGH:
        return 'high'
    return None


def three_zone(zone_seconds):
    """DataFrame(zone, seconds) or {zone: seconds} -> {'low','mid','high': seconds}."""
    if isinstance(zone_seconds, dict):
        items = zone_seconds.items()
    else:
        items = zip(zone_seconds['zone'], zone_seconds['seconds'])
    out = {'low': 0.0, 'mid': 0.0, 'high': 0.0}
    for z, s in items:
        g = zone_group(z)
        if g:
            out[g] += float(s)
    return out


def polarization_index(low, mid, high):
    """
    PI = log10(Z1 / Z2 x Z3 x 100), zones as fractions of total time.
    > 2.00 = polarized. Per the paper: undefined if Z3 > Z1 (returns None),
    0 if Z3 = 0. When Z2 = 0 the paper's Eq. 2 replaces Z2 by 0.01 and
    takes 0.01 from Z3 (our reading of the published equation).
    """
    total = low + mid + high
    if total <= 0:
        return None
    z1, z2, z3 = low / total, mid / total, high / total
    if z3 > z1:
        return None
    if z3 == 0:
        return 0.0
    if z2 == 0:
        val = z1 / 0.01 * (z3 - 0.01) * 100
    else:
        val = z1 / z2 * z3 * 100
    return float(np.log10(val)) if val > 0 else 0.0


def distribution(zone_rows, freq='W'):
    """
    zone_rows: DataFrame(date, zone, seconds) for one zone kind.
    Returns per period: hours, low/mid/high shares and the polarization index.
    """
    cols = ['hours', 'low', 'mid', 'high', 'pi']
    if zone_rows is None or zone_rows.empty:
        return pd.DataFrame(columns=cols)
    df = zone_rows.dropna(subset=['date']).copy()
    df['group'] = df['zone'].map(zone_group)
    df = df.dropna(subset=['group'])
    period = 'W-SUN' if freq == 'W' else 'M'
    df['period'] = df['date'].dt.to_period(period).dt.start_time
    sec = df.pivot_table(index='period', columns='group', values='seconds', aggfunc='sum', fill_value=0)
    sec = sec.reindex(columns=['low', 'mid', 'high'], fill_value=0)
    total = sec.sum(axis=1)
    out = sec.div(total.where(total > 0), axis=0)
    out['hours'] = total / 3600
    out['pi'] = [polarization_index(r.low, r.mid, r.high) for r in sec.itertuples()]
    out.index.name = 'period'
    return out[cols]
