"""The overlay / stacked time-series figure. Skipped where plotly is not installed (CI has it)."""
import os

import numpy as np
import pandas as pd
import pytest

try:
    import plotly.graph_objects as go
    _real = type(go).__name__ == "module"
except ImportError:
    _real = False
if not _real:
    if os.environ.get("CI"):
        raise RuntimeError("plotly must be installed in CI to validate the figures")
    pytest.skip("plotly not installed", allow_module_level=True)

from src.viz import timeseries as ts


def ride(n=1200, **drop):
    t = np.arange(n)
    df = pd.DataFrame({
        'time': pd.date_range('2026-10-01 08:00', periods=n, freq='1s'), 'dist_m': t * 8.0,
        'ele': 100 + 20 * np.sin(t / 200), 'power': 200 + 50 * np.sin(t / 30), 'hr': 140.0 + t * 0.01,
        'speed_kmh': 28.8, 'cadence': 90.0, 'grade_smooth': np.gradient(100 + 20 * np.sin(t / 200)) / 8 * 100,
        'w_prime_balance': 20000.0})
    return df.drop(columns=list(drop))


def test_available_channels_follow_the_data():
    assert ts.available(ride()) == ["Power", "Heart rate", "Speed", "Cadence", "Gradient", "W′ balance"]
    df = ride()
    df['hr'] = np.nan
    assert "Heart rate" not in ts.available(df) and ts.available(ride(power=1, hr=1))[0] == "Speed"


def test_one_row_per_channel_under_the_elevation_base():
    fig = ts.build_figure(ride(), ["Power", "Heart rate", "Speed"])
    assert [t.name for t in fig.data] == ["Elevation", "Power", "Heart rate", "Speed"]
    assert fig.data[0].fill == "tozeroy" and len({t.yaxis for t in fig.data}) == 4
    assert fig.layout.xaxis4.title.text == "Distance (km)"               # shared axis, labelled once at the bottom
    assert max(fig.data[1].x) == pytest.approx(1199 * 8 / 1000)


def test_time_axis_only_when_there_is_no_distance():
    fig = ts.build_figure(ride(dist_m=1), ["Power"])
    assert fig.layout.xaxis2.title.text == "Time (min)" and max(fig.data[1].x) == pytest.approx(1199 / 60)


def test_smoothing_is_a_fixed_five_second_average():
    assert ts.SMOOTHING_S == 5
    df = ride()
    df['power'] = np.where(np.arange(len(df)) % 2, 400.0, 0.0)          # alternating every second
    y = ts.build_figure(df, ["Power"]).data[1].y
    assert np.std(y[10:-10]) < 100                                       # the 5 s average removes most of the +-200 W swing


def test_long_rides_are_thinned_and_odd_input_is_tolerated():
    big = ride(n=30000)
    assert len(ts.build_figure(big, ["Power"]).data[1].x) <= ts.MAX_POINTS
    assert len(ts.build_figure(ride(), ["Power", "nonsense"]).data) == 2          # unknown channel ignored
    no_ele = ts.build_figure(ride(ele=1), ["Power"])
    assert [t.name for t in no_ele.data] == ["Power"]
    assert len(ts.build_figure(ride(), []).data) == 1                              # only the base
