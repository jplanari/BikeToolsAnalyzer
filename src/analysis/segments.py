"""
Analysis of any stretch of a ride (a custom selection, a lap, a detected climb): the same numbers as for the
whole ride, computed on that slice. Pure functions on the 1 Hz ride DataFrame; positions are row numbers
(iloc), both ends included.
"""
import numpy as np
import pandas as pd

from src.analysis.power import NP, IF
from src.data.pipeline import moving_mask

MIN_NP_SAMPLES = 30          # NP uses a 30 s rolling average: with less data it is not defined
ELE_WINDOW = 15              # samples; same smoothing the climb detector uses


def _clean_series(df, col):
    return df[col].dropna() if col in df.columns else pd.Series(dtype=float)


def span_by_distance(df, start_m, end_m):
    """Row positions (first, last) of the samples between two distances from the start (metres)."""
    d = df['dist_m'].cummax().to_numpy()
    i0 = int(np.searchsorted(d, start_m, side='left'))
    i1 = int(np.searchsorted(d, end_m, side='right')) - 1
    return _clip(df, i0, i1)


def selection_range(selection, lo, hi, step=0.01):
    """
    The (start, end) a rider drew on the elevation profile, from a Plotly selection ({'box': [{'x': [x0, x1]}],
    'points': [{'x': ...}]}), clipped to [lo, hi] and rounded to `step`. The box is used when there is one, else
    the extent of the selected points. None when nothing usable was selected or it is narrower than two steps.
    """
    if not selection:
        return None
    xs = []
    for box in (selection.get('box') or []):
        xs += [x for x in (box.get('x') or []) if x is not None]
        if xs:
            break
    if not xs:
        xs = [p['x'] for p in (selection.get('points') or []) if p.get('x') is not None]
    if not xs:
        return None
    a, b = max(lo, min(xs)), min(hi, max(xs))
    a, b = round(round(a / step) * step, 6), round(round(b / step) * step, 6)
    return (a, b) if b - a >= 2 * step else None


def span_by_time(df, start, end):
    """Row positions of the samples between two timestamps."""
    t = pd.to_datetime(df['time']).to_numpy()
    i0 = int(np.searchsorted(t, np.datetime64(pd.Timestamp(start)), side='left'))
    i1 = int(np.searchsorted(t, np.datetime64(pd.Timestamp(end)), side='right')) - 1
    return _clip(df, i0, i1)


def _clip(df, i0, i1):
    n = len(df)
    i0, i1 = max(0, min(i0, n - 1)), max(0, min(i1, n - 1))
    return (i0, max(i0, i1))


def elevation_change(ele):
    """(ascent, descent) in metres from a smoothed elevation series, so GPS/barometer noise is not counted."""
    e = pd.Series(ele).dropna()
    if len(e) < 2:
        return 0.0, 0.0
    d = e.rolling(ELE_WINDOW, min_periods=1, center=True).mean().diff().dropna()
    return float(d[d > 0].sum()), float(-d[d < 0].sum())


def segment_stats(df, i0, i1, ftp=None, weight=None):
    """
    Numbers for rows i0..i1 of the ride. Keys that cannot be computed (no power channel, no HR...) are None.
    Average speed, power, HR and cadence are over moving samples (cadence: while pedalling), like the ride summary.
    """
    seg = df.iloc[i0:i1 + 1]
    out = {'start_idx': i0, 'end_idx': i1, 'samples': len(seg)}
    if seg.empty:
        return out
    t = pd.to_datetime(seg['time']) if 'time' in seg.columns else None
    elapsed = (t.iloc[-1] - t.iloc[0]).total_seconds() + 1.0 if t is not None else float(len(seg))
    moving = moving_mask(seg)
    moving_s = int(moving.sum())
    out.update(elapsed_s=elapsed, moving_s=moving_s)
    if t is not None:
        out['start_time'], out['end_time'] = t.iloc[0], t.iloc[-1]

    if 'dist_m' in seg.columns and seg['dist_m'].notna().any():
        dist = float(seg['dist_m'].iloc[-1] - seg['dist_m'].iloc[0])
        out.update(dist_km=dist / 1000.0, start_km=float(seg['dist_m'].iloc[0]) / 1000.0,
                   end_km=float(seg['dist_m'].iloc[-1]) / 1000.0)
    else:
        dist = None
    ele_col = 'ele' if 'ele' in seg.columns else None
    if ele_col and seg[ele_col].notna().any():
        asc, desc = elevation_change(seg[ele_col])
        out.update(ascent_m=asc, descent_m=desc)
        if dist and dist > 0:
            smooth = seg[ele_col].rolling(ELE_WINDOW, min_periods=1, center=True).mean()
            out['avg_grade_pct'] = float((smooth.iloc[-1] - smooth.iloc[0]) / dist * 100.0)
        out['vam_m_h'] = asc / elapsed * 3600.0 if elapsed > 0 else None

    if 'speed' in seg.columns and moving_s > 0:
        out['avg_speed_kph'] = float(seg.loc[moving, 'speed'].mean() * 3.6)
        out['max_speed_kph'] = float(seg['speed'].max() * 3.6)

    power = _clean_series(seg, 'power')
    if len(power):
        avg = float(seg.loc[moving, 'power'].mean()) if moving_s > 0 else float(power.mean())
        out.update(avg_power=avg, max_power=float(power.max()), kj=float(seg['power'].fillna(0).sum() / 1000.0))
        if weight:
            out['w_per_kg'] = avg / weight
        if len(power) >= MIN_NP_SAMPLES:
            npower = float(NP(seg['power']))
            out['norm_power'] = npower
            out['vi'] = npower / float(power.mean()) if power.mean() > 0 else None
            if ftp:
                out['if'] = float(IF(npower, ftp))
                out['tss'] = elapsed / 3600.0 * out['if'] ** 2 * 100.0
    hr = _clean_series(seg, 'hr')
    hr = hr[hr > 0]
    if len(hr):
        out.update(avg_hr=float(hr.mean()), max_hr=float(hr.max()))
        if out.get('norm_power'):
            out['ef'] = out['norm_power'] / out['avg_hr']
    cad = _clean_series(seg, 'cadence')
    cad = cad[cad > 0]
    if len(cad):
        out['avg_cadence'] = float(cad.mean())
        out['pedalling_share'] = float(len(cad) / len(seg))
    return out


def as_climb(df, i0, i1):
    """A detected-climb–shaped dict for the selection, so the climb profile plot and the performance estimator
    work on it unchanged."""
    s = segment_stats(df, i0, i1)
    length = (s.get('dist_km') or 0.0) * 1000.0
    return {'start_idx': df.index[i0], 'end_idx': df.index[i1], 'length_m': length,
            'elev_gain_m': s.get('ascent_m', 0.0), 'avg_gradient_pct': s.get('avg_grade_pct', 0.0),
            'vam_mph': s.get('vam_m_h', 0.0) or 0.0, 'type': 'Selected segment'}


def lap_spans(df, spans):
    """
    spans: DataFrame with 'start' and 'elapsed_s' per lap (see data.fit.get_fit_lap_spans). Returns
    [(label, i0, i1)] for the laps that overlap the ride's samples.
    """
    out = []
    if spans is None or len(spans) == 0 or 'time' not in df.columns:
        return out
    first, last = pd.to_datetime(df['time'].iloc[0]), pd.to_datetime(df['time'].iloc[-1])
    for n, row in enumerate(spans.itertuples(index=False), 1):
        start = pd.Timestamp(row.start)
        end = start + pd.Timedelta(seconds=float(row.elapsed_s))
        if pd.isna(start) or end < first or start > last:
            continue
        i0, i1 = span_by_time(df, start, end - pd.Timedelta(seconds=1))
        out.append((f"Lap {n}", i0, i1))
    return out


def climb_spans(df, climbs):
    """[(label, i0, i1, climb)] for detected climbs (their start_idx/end_idx are index labels of df)."""
    out = []
    for n, c in enumerate(climbs or [], 1):
        try:
            i0, i1 = int(df.index.get_loc(c['start_idx'])), int(df.index.get_loc(c['end_idx']))
        except KeyError:
            continue
        out.append((f"Climb {n}", i0, max(i0, i1), c))
    return out


def stats_table(df, spans, ftp=None, weight=None):
    """One row per span (label, i0, i1), for comparing laps."""
    rows = []
    for label, i0, i1 in spans:
        s = segment_stats(df, i0, i1, ftp, weight)
        rows.append({'Segment': label, 'Start (km)': s.get('start_km'), 'Distance (km)': s.get('dist_km'),
                     'Time': s.get('elapsed_s'), 'Ascent (m)': s.get('ascent_m'),
                     'Grade (%)': s.get('avg_grade_pct'), 'Speed (km/h)': s.get('avg_speed_kph'),
                     'Avg power (W)': s.get('avg_power'), 'NP (W)': s.get('norm_power'),
                     'Avg HR': s.get('avg_hr'), 'Cadence': s.get('avg_cadence')})
    return pd.DataFrame(rows)
