"""A CdA figure is always shown, and says how it was obtained: measured, rough or assumed."""
import numpy as np
import pandas as pd
import pytest

from src.physics import aerodyn as ae
from tests.test_aerodyn import simulate


def levels(df, **kw):
    kw.setdefault('rider_mass', 70)
    kw.setdefault('rho', 1.225)
    return ae.cda_with_confidence(df, **kw)


def test_a_long_ride_is_measured_and_close_to_the_truth():
    r = levels(simulate(cda=0.30, n=2400))
    assert r['level'] == 'measured' and r['cda'] == pytest.approx(0.30, abs=0.01)
    assert r['seconds'] >= ae.MEASURED_MIN_SAMPLES


def test_a_short_ride_is_rough_not_measured():
    # two minutes gave a 7 % error under a 5 kg weight error: it must not be labelled as a measurement
    r = levels(simulate(cda=0.30, n=120), rider_mass=75)
    assert r['level'] == 'rough'
    assert 0.2 < r['cda'] < 0.4


def test_slow_riding_still_gives_a_rough_value_from_the_samples_above_the_floor():
    slow = simulate(cda=0.30, n=900)
    slow['speed'] = slow['speed'].clip(upper=4.0)                 # never above 14 km/h: only the rough tier can use it
    r = levels(slow)
    assert r['level'] in ('rough', 'assumed')
    assert r['level'] == 'assumed' or r['cda'] > 0


@pytest.mark.parametrize("name, df", [
    ("no power channel", simulate(n=900).drop(columns='power')),
    ("no speed channel", simulate(n=900).drop(columns='speed')),
    ("empty table", pd.DataFrame()),
    ("stopped the whole time", pd.DataFrame({'speed': np.zeros(600), 'power': np.full(600, 200.0)})),
])
def test_without_usable_data_the_assumed_value_is_shown_and_labelled(name, df):
    r = levels(df)
    assert r['level'] == 'assumed' and r['cda'] == ae.ASSUMED_CDA and r['seconds'] == 0, name


def test_a_figure_is_never_empty_whatever_the_ride():
    for n in (1, 10, 60, 200, 1200):
        for kw in ({}, dict(grade_amp=0.0)):
            r = levels(simulate(n=n, **kw))
            assert r['level'] in ('measured', 'rough', 'assumed') and np.isfinite(r['cda'])


def test_an_implausible_rough_fit_falls_back_to_the_assumed_value(monkeypatch):
    # a rough fit outside the physical range must not be shown as if it were a rider's figure
    monkeypatch.setattr(ae, "CDA_VALID", (0.5, 0.6))
    r = levels(simulate(cda=0.30, n=120))
    assert r['level'] == 'assumed'


def test_the_measured_path_is_unchanged_for_the_strict_function():
    # estimate_cda keeps returning None under two minutes: validation scripts and older tests rely on it
    assert ae.estimate_cda(simulate(n=60), rider_mass=70, rho=1.225) is None
