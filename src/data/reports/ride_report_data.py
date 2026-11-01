"""
Everything the PDF report of ONE analysed ride shows, gathered in one place (no Streamlit, no plotting).
The numbers are those of the ride page: the same summary dict, the same climbs, the same physics functions.
"""
import numpy as np
import pandas as pd

from src.physics import aerodyn as ae

MAX_POINTS = 900                      # points per time-series chart: a 6-hour ride at 1 Hz has 21,600


def _downsample(df, columns, max_points=MAX_POINTS):
    """Mean per bin of consecutive samples, so a spike is not lost the way plain striding would."""
    step = max(1, int(np.ceil(len(df) / max_points)))
    keep = [c for c in columns if c in df.columns]
    out = df[keep].groupby(np.arange(len(df)) // step).mean(numeric_only=True)
    return out.reset_index(drop=True), step


def _series(df):
    d = pd.DataFrame(index=df.index)
    t = pd.to_datetime(df['time']) if 'time' in df.columns else pd.Series(pd.RangeIndex(len(df)), index=df.index)
    d['t_min'] = (t - t.iloc[0]).dt.total_seconds() / 60.0 if 'time' in df.columns else np.arange(len(df)) / 60.0
    for col, src in (('dist_km', 'dist_m'), ('ele', 'ele'), ('speed_kmh', 'speed_kmh'), ('hr', 'hr'),
                     ('cadence', 'cadence'), ('w_prime_balance', 'w_prime_balance')):
        if src in df.columns:
            d[col] = df[src] / (1000.0 if src == 'dist_m' else 1.0)
    if 'speed_kmh' not in d and 'speed' in df.columns:
        d['speed_kmh'] = df['speed'] * 3.6
    if 'power' in df.columns:
        d['power'] = df['power'].rolling(30, min_periods=1, center=True).mean()      # 30 s average
    out, step = _downsample(d, d.columns)
    out.attrs['step_s'] = step
    return out


def _climb_rows(df, climbs):
    rows = []
    for i, c in enumerate(climbs or [], 1):
        seg = df.loc[c['start_idx']:c['end_idx']]
        if seg.empty:
            continue
        secs = (pd.to_datetime(seg['time']).iloc[-1] - pd.to_datetime(seg['time']).iloc[0]).total_seconds() \
            if 'time' in seg.columns else np.nan
        rows.append({
            'n': i, 'type': c.get('type', ''),
            'start_km': float(seg['dist_m'].iloc[0] / 1000) if 'dist_m' in seg.columns else np.nan,
            'length_km': c['length_m'] / 1000, 'gradient': c['avg_gradient_pct'], 'ascent_m': c['elev_gain_m'],
            'vam': c.get('vam_mph', np.nan), 'seconds': secs,
            'power': float(seg['power'].mean()) if 'power' in seg.columns else np.nan,
            'hr': float(seg['hr'].mean()) if 'hr' in seg.columns else np.nan,
        })
    return rows


def _aero(df, rider_mass):
    """CdA fit and the mean power budget (W) on the basis of calculate_CdA. None without speed and power."""
    if 'power' not in df.columns or 'speed' not in df.columns or not df['power'].notna().any():
        return None
    fit = ae.estimate_cda(df, rider_mass)
    _, p_aero = ae.calculate_CdA(df, rider_mass)
    comps = ae.calculate_power_components(df, rider_mass)
    if not len(comps) or p_aero.empty:
        return {'cda': fit['cda'] if fit else None, 'seconds': fit['seconds'] if fit else 0, 'budget': None}
    p_grav, p_roll, p_accel = comps
    pedal = float(df['power'].mean())
    budget = {'Aerodynamic drag': float(p_aero.mean()), 'Rolling resistance': float(p_roll.mean()),
              'Climbing / descending': float(p_grav.mean()), 'Acceleration': float(p_accel.mean()),
              'Drivetrain loss': pedal * ae.DRIVETRAIN_LOSS}
    if not all(np.isfinite(v) for v in budget.values()):
        budget = None
    return {'cda': fit['cda'] if fit else None, 'seconds': fit['seconds'] if fit else 0,
            'budget': budget, 'pedal_w': pedal}


def _thermal(thermal):
    if thermal is None or thermal.empty or 'core_temp' not in thermal.columns:
        return None
    t = pd.to_datetime(thermal['time']) if 'time' in thermal.columns else None
    mins = (t - t.iloc[0]).dt.total_seconds() / 60.0 if t is not None else pd.Series(np.arange(len(thermal)) / 60.0)
    d = pd.DataFrame({'t_min': mins.to_numpy(), 'core': thermal['core_temp'].to_numpy(),
                      'skin': thermal['skin_temp'].to_numpy()})
    series, _ = _downsample(d, d.columns)
    core = thermal['core_temp']
    return {'series': series, 'max_core': float(core.max()), 'end_core': float(core.iloc[-1]),
            'minutes_above_385': float((core > 38.5).sum() / 60.0), 'minutes_above_39': float((core > 39.0).sum() / 60.0)}


def collect_ride_report(df, summary, settings, name, total_dist_m, total_ascent_m, climbs=None,
                        best_curve=None, recent_curve=None, laps=None, thermal=None, user_name=None):
    """
    df: the analysed ride; summary: summarize_ride(); settings: dict with ftp, lthr, weight, height;
    climbs: detect_climbs() output; best_curve / recent_curve: {seconds: watts} of the rider's history;
    laps: DataFrame; thermal: calculate_thermal_profile() result (optional, it needs a clothing choice).
    """
    weight = settings.get('weight') or 70
    start = pd.to_datetime(df['time'].iloc[0]) if 'time' in df.columns and len(df) else None
    weather = None
    if 'temp_c' in df.columns and df['temp_c'].notna().any():
        weather = {'temp_c': float(df['temp_c'].mean()),
                   'wind_kmh': float(df['wind_speed'].mean() * 3.6) if 'wind_speed' in df.columns else None,
                   'rh': float(df['relative_humidity'].mean()) if 'relative_humidity' in df.columns else None}
    has_power = 'power' in df.columns and df['power'].notna().any()
    moving = summary.get('is_moving')
    return {
        'name': name, 'user': user_name, 'start': start,
        'ftp': settings.get('ftp'), 'lthr': settings.get('lthr'), 'weight': weight,
        'summary': {k: summary.get(k) for k in (
            'duration_s', 'moving_time_s', 'avg_speed_kph', 'avg_power', 'avg_hr', 'norm_power', 'if', 'tss',
            'tss_source', 'kj', 'vi', 'ef', 'decoupling_pct', 'hrr60', 'cleaned')},
        'distance_km': total_dist_m / 1000.0, 'ascent_m': float(total_ascent_m),
        'max_power': float(df['power'].max()) if has_power else None,
        'max_hr': float(df['hr'].max()) if 'hr' in df.columns and df['hr'].notna().any() else None,
        'weather': weather,
        'series': _series(df),
        'route': (_downsample(df, ['lat', 'lon'], 700)[0] if {'lat', 'lon'} <= set(df.columns) and
                  df[['lat', 'lon']].notna().all(axis=1).any() else None),
        'power_curve': dict(summary.get('power_curve') or {}), 'best_curve': best_curve or {},
        'recent_curve': recent_curve or {},
        'zones': summary.get('zones') or {},
        'climbs': _climb_rows(df, climbs),
        'aero': _aero(df, weight),
        'thermal': _thermal(thermal),
        'laps': laps if laps is not None and len(laps) else None,
        'moving_share': float(moving.mean()) if moving is not None and len(moving) else None,
    }
