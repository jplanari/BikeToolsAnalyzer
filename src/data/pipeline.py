"""
Headless ride pipeline shared by the dashboard, the Strava history import
and the history repair: parse a stored/uploaded ride into the standard 1 Hz
DataFrame and compute the summary metrics that get saved.
"""
import numpy as np
import pandas as pd

from src.data.gpx import (parse_gpx, compute_distance_and_ascent, resample_to_seconds,
                          compute_speed, compute_grade)
from src.data.fit import parse_fit
from src.data.strava import streams_to_dataframe
from src.analysis.power import NP, power_curve, calculate_ride_kJ
from src.analysis import ride_metrics as rm

MOVING_THRESHOLD_MS = 2.0


def ride_kind(filename):
    name = (filename or "").lower()
    if name.endswith(".fit"):
        return "fit"
    if name.endswith(".json"):
        return "strava"
    return "gpx"


def prepare_ride(filename, file_bytes):
    """
    Returns (df, total_dist_m, total_ascent_m). Raises ValueError if the file
    contains no timestamped samples.
    """
    kind = ride_kind(filename)
    if kind == "fit":
        df = parse_fit(file_bytes)
    elif kind == "strava":
        df = streams_to_dataframe(file_bytes)
    else:
        df = parse_gpx(file_bytes)

    if df is None or df.empty or 'time' not in df.columns:
        raise ValueError("no timestamped samples")

    df, total_dist, total_ascent = compute_distance_and_ascent(df)
    df = resample_to_seconds(df)
    df, total_dist, total_ascent = compute_distance_and_ascent(df)
    if kind != "fit":  # FIT files carry a native speed channel
        df = compute_speed(df)
    df = compute_grade(df)
    return df, total_dist, total_ascent


def summarize_ride(df, ftp, lthr=None):
    """Summary metrics shown on the ride page and saved to the database."""
    has_power = 'power' in df.columns and df['power'].notna().any()
    has_hr = 'hr' in df.columns and df['hr'].notna().any()
    hr = df['hr'] if has_hr else None

    speed = df['speed'] if 'speed' in df.columns else pd.Series(np.nan, index=df.index)
    is_moving = speed >= MOVING_THRESHOLD_MS
    moving_time_s = int(is_moving.sum())
    duration_s = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()

    avg_speed_kph = speed[is_moving].mean() * 3.6 if moving_time_s > 0 else 0
    avg_power = df.loc[is_moving, 'power'].mean() if has_power and moving_time_s > 0 else 0
    avg_hr = df.loc[is_moving, 'hr'].mean() if has_hr and moving_time_s > 0 else 0
    norm_power = NP(df['power']) if has_power else 0
    norm_power = 0 if pd.isna(norm_power) else norm_power

    tss, if_val, tss_source = rm.training_stress(norm_power, duration_s, ftp, hr, lthr, is_moving)

    return {
        'duration_s': duration_s,
        'moving_time_s': moving_time_s,
        'is_moving': is_moving,
        'avg_speed_kph': avg_speed_kph,
        'avg_power': avg_power,
        'avg_hr': avg_hr,
        'norm_power': norm_power,
        'if': if_val,
        'tss': tss,
        'tss_source': tss_source,
        'kj': calculate_ride_kJ(df['power']) if has_power else 0,
        'power_curve': power_curve(df['power']) if has_power else {},
        'vi': rm.variability_index(df['power']) if has_power else None,
        'ef': rm.efficiency_factor(norm_power, hr, is_moving) if has_power and has_hr else None,
        'decoupling_pct': (rm.aerobic_decoupling(df['power'], hr, is_moving)
                           if has_power and has_hr else None),
        'zones': rm.zone_seconds(df, ftp, lthr),
        'ftp': ftp,
        'lthr': lthr,
    }


def metrics_for_storage(summary):
    """The per-ride columns that build_ride_metadata() adds to the metadata."""
    keys = ['duration_s', 'moving_time_s', 'kj', 'tss_source', 'vi', 'ef',
            'decoupling_pct', 'zones']
    out = {k: summary.get(k) for k in keys}
    out['ftp_at_ride'] = summary.get('ftp') or None
    out['lthr_at_ride'] = summary.get('lthr') or None
    return out
