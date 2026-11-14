"""Aerobic decoupling (Friel) and the year-by-year power curves."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import decoupling as dc
from src.analysis import power_profile as pp


def ride(seconds, power=200.0, hr_start=130.0, hr_end=130.0, start="2026-06-01 08:00"):
    """A steady ride whose heart rate climbs linearly from hr_start to hr_end (drift)."""
    t = pd.date_range(start, periods=seconds, freq="s")
    hr = np.linspace(hr_start, hr_end, seconds)
    return pd.DataFrame({'time': t, 'power': np.full(seconds, power), 'hr': hr})


def test_no_drift_gives_about_zero():
    d = dc.decoupling(ride(2 * 3600))
    assert d['pct'] == pytest.approx(0.0, abs=0.1)
    assert d['reliable'] and d['minutes'] == pytest.approx(120.0, abs=0.1)


def test_heart_rate_drift_shows_as_positive_decoupling():
    # the same power, but the heart rate goes from 130 to 150 bpm: efficiency falls about 13 %
    d = dc.decoupling(ride(2 * 3600, hr_start=130, hr_end=150))
    assert d['pct'] == pytest.approx(7.0, abs=1.0)
    assert d['second'] < d['first']


def test_a_negative_result_is_returned_not_hidden():
    d = dc.decoupling(ride(2 * 3600, hr_start=150, hr_end=130))
    assert d['pct'] < 0


def test_coasting_does_not_count():
    df = ride(2 * 3600, hr_start=130, hr_end=130)
    df.loc[df.index[:600], 'power'] = 0.0                  # ten minutes of freewheeling at the start
    assert dc.decoupling(df)['pct'] == pytest.approx(0.0, abs=0.2)


def test_short_or_unusable_rides_give_nothing():
    assert dc.decoupling(ride(15 * 60)) is None             # under 20 minutes
    assert dc.decoupling(ride(40 * 60)) is not None and not dc.decoupling(ride(40 * 60))['reliable']
    assert dc.decoupling(ride(2 * 3600).drop(columns='hr')) is None
    assert dc.decoupling(pd.DataFrame()) is None
    assert dc.decoupling(ride(2 * 3600).assign(hr=0.0)) is None


def _records(rows):
    return pd.DataFrame(rows, columns=['duration', 'watts', 'date', 'ride_id'])


def test_season_curves_keep_each_years_own_best():
    recs = _records([
        (300, 300.0, pd.Timestamp("2024-05-01"), 1),
        (300, 280.0, pd.Timestamp("2025-05-01"), 2),
        (300, 320.0, pd.Timestamp("2026-03-01"), 3),
        (1200, 250.0, pd.Timestamp("2026-03-01"), 3),
    ])
    seasons = pp.season_curves(recs, today=pd.Timestamp("2026-10-01"), years=4)
    assert list(seasons) == [2026, 2025, 2024]              # newest first
    assert seasons[2026].set_index('duration').loc[300, 'watts'] == 320.0
    assert seasons[2025].set_index('duration').loc[300, 'watts'] == 280.0
    assert 1200 not in seasons[2025].index                  # a 2025 curve only has 2025's efforts


def test_season_curves_ignore_future_years_and_empty_input():
    recs = _records([(300, 300.0, pd.Timestamp("2027-01-05"), 1), (300, 290.0, pd.Timestamp("2026-02-01"), 2)])
    assert list(pp.season_curves(recs, today=pd.Timestamp("2026-10-01"))) == [2026]
    assert pp.season_curves(_records([]), today=pd.Timestamp("2026-10-01")) == {}
    assert pp.season_curves(None) == {}
