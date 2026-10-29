"""The per-ride PDF report: built from a normal ride and from rides missing channels."""
import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import summarize_ride
from src.data.ride_report_data import collect_ride_report
from src.viz.ride_report_pdf import build_ride_report_pdf

SETTINGS = {'ftp': 250, 'lthr': 170, 'weight': 70, 'height': 180}


def ride(n=1800, **drop):
    """drop: channels the file lacks; the parsers leave them as all-NaN columns or drop them (GPX has no power)."""
    t = np.arange(n)
    ele = 200 + 60 * np.sin(2 * np.pi * t / 900)
    df = pd.DataFrame({
        'time': pd.date_range('2026-10-01 08:00', periods=n, freq='1s'),
        'lat': 41 + t * 1e-5, 'lon': 2 + 0.001 * np.sin(t / 300), 'ele': ele, 'dist_m': t * 7.0,
        'speed': 7.0, 'speed_kmh': 25.2, 'power': 200 + 60 * np.sin(2 * np.pi * t / 240),
        'hr': 140 + 15 * np.sin(2 * np.pi * t / 600), 'cadence': 88.0,
        'grade_smooth': np.gradient(ele) / 7.0 * 100})
    return df.drop(columns=list(drop))


def blank(df, *cols):
    df = df.copy()
    df[list(cols)] = np.nan
    return df


def pages(pdf):
    return pdf.count(b'/Type /Page\n') or pdf.count(b'/Type /Page')


def build(df, **kw):
    s = summarize_ride(df, 250, 170)
    data = collect_ride_report(df, s, SETTINGS, 'test', float(df['dist_m'].iloc[-1]) if 'dist_m' in df else 0.0,
                               100.0, user_name='u', **kw)
    return build_ride_report_pdf(data)


def test_full_ride_makes_a_pdf():
    pdf = build(ride(), best_curve={1: 800, 300: 320}, recent_curve={1: 700, 300: 300})
    assert pdf.startswith(b'%PDF') and len(pdf) > 10_000


@pytest.mark.parametrize('missing', [('power',), ('hr',), ('lat', 'lon'), ('cadence',)])
def test_missing_channels_do_not_break_the_report(missing):
    assert build(ride(**{c: 1 for c in missing})).startswith(b'%PDF')


def test_all_sensor_channels_empty():
    assert build(blank(ride(), 'power', 'hr', 'cadence', 'lat', 'lon')).startswith(b'%PDF')


def test_laps_and_thermal_are_optional_extras():
    from src.physics.thermal import calculate_thermal_profile
    df = ride()
    laps = pd.DataFrame({'Duration': ['10:00'], 'Avg Power (W)': [210.0]})
    th = calculate_thermal_profile(df, 70, 180, ['Summer Jersey'], ftp=250)
    assert len(build(df, laps=laps, thermal=th)) > len(build(df))
