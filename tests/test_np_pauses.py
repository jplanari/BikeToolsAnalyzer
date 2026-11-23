"""Average power, NP and VI over the same seconds: a stop inside a segment or ride must not pull NP below the average."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import segments as sg
from src.data.processing.pipeline import power_mask, summarize_ride


def _climb_with_stop(n_ride=383, n_stop=120, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(409 + 60 * np.sin(np.arange(n_ride) / 20) + rng.normal(0, 25, n_ride), 0, None)
    half = n_ride // 2
    power = np.r_[p[:half], np.zeros(n_stop), p[half:]]               # a pause in the middle: speed 0, power 0
    speed = np.r_[np.full(half, 6.6), np.zeros(n_stop), np.full(n_ride - half, 6.6)]
    n = len(power)
    return pd.DataFrame({'time': pd.date_range("2026-11-20T09:00:00Z", periods=n, freq="1s"), 'power': power,
                         'speed': speed, 'dist_m': np.cumsum(speed), 'hr': np.full(n, 172.0)})


def test_power_mask_drops_stops_but_keeps_coasting_and_indoor_seconds():
    df = pd.DataFrame({'power': [300, 0, 0, 0, 250], 'speed': [8, 8, 0, 0.5, 1.0]})
    assert list(power_mask(df)) == [True, True, False, False, True]       # coasting at 8 m/s stays, slow pedalling stays
    assert power_mask(df.drop(columns='speed')).all()                      # indoor: everything counts


def test_segment_np_is_not_below_average_power_after_a_stop():
    df = _climb_with_stop()
    s = sg.segment_stats(df, 0, len(df) - 1, ftp=385)
    assert s['norm_power'] >= s['avg_power']
    assert 1.0 <= s['vi'] < 1.05
    assert s['avg_power'] == pytest.approx(df.loc[df['speed'] > 0, 'power'].mean())
    assert s['tss'] == pytest.approx(383 / 3600 * (s['norm_power'] / 385) ** 2 * 100)


def test_ride_np_is_not_below_average_power_after_a_stop():
    s = summarize_ride(_climb_with_stop(), 385, 170)
    assert s['norm_power'] >= s['avg_power'] and s['vi'] >= 1.0
