import numpy as np
import pandas as pd
import pytest

from src.analysis import ride_metrics as rm
from src.analysis.power import power_curve, MMP_DURATIONS, estimate_critical_power


def s(values):
    return pd.Series(np.asarray(values, dtype=float))


# --- Power curve -------------------------------------------------------------

def test_power_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    p = s(rng.uniform(100, 400, 2000))
    curve = power_curve(p)
    for d in (1, 5, 60, 300, 1200):
        brute = max(p.iloc[i:i + d].mean() for i in range(len(p) - d + 1))
        assert curve[d] == pytest.approx(brute)


def test_power_curve_skips_durations_longer_than_ride():
    curve = power_curve(s([200] * 600))
    assert max(curve) == 600 and 900 not in curve
    assert all(v == pytest.approx(200) for v in curve.values())
    assert power_curve(s([])) == {}


def test_dense_curve_feeds_cp_model():
    # Synthetic rider: CP 250 W, W' 20 kJ -> P(t) = CP + W'/t
    curve = {d: 250 + 20000 / d for d in MMP_DURATIONS}
    res = estimate_critical_power(curve)
    assert res['cp'] == pytest.approx(250, abs=0.5)
    assert res['w_prime'] == pytest.approx(20000, rel=0.01)


# --- VI / EF / decoupling ----------------------------------------------------

def test_variability_index():
    assert rm.variability_index(s([200] * 3600)) == pytest.approx(1.0)
    surges = s(([400] * 30 + [100] * 30) * 60)
    assert rm.variability_index(surges) > 1.15
    assert rm.variability_index(s([0] * 100)) is None


def test_efficiency_factor_uses_moving_samples():
    hr = s([140] * 50 + [90] * 50)
    moving = pd.Series([True] * 50 + [False] * 50)
    assert rm.efficiency_factor(210, hr, moving) == pytest.approx(1.5)
    assert rm.efficiency_factor(0, hr) is None
    assert rm.efficiency_factor(210, s([np.nan] * 10)) is None


def test_decoupling_zero_when_steady():
    assert rm.aerobic_decoupling(s([200] * 3600), s([140] * 3600)) == pytest.approx(0.0)


def test_decoupling_detects_hr_drift():
    # Same power; HR 140 in the first hour, 147 in the second -> ~4.8 % decoupling
    p, hr = s([200] * 7200), s([140] * 3600 + [147] * 3600)
    assert rm.aerobic_decoupling(p, hr) == pytest.approx((1 - 140 / 147) * 100)


def test_decoupling_ignores_coasting():
    # Steady 200 W / 140 bpm when pedalling; the 2nd half coasts 20 % of the time
    # (HR 125) vs 5 % in the 1st. Counting the zeros would fake ~14 % drift.
    def half(coast):
        n = 3600
        k = int(n * coast)
        return [0] * k + [200] * (n - k), [125] * k + [140] * (n - k)
    (p1, h1), (p2, h2) = half(0.05), half(0.20)
    assert rm.aerobic_decoupling(s(p1 + p2), s(h1 + h2)) == pytest.approx(0.0)


def test_decoupling_needs_20_minutes():
    assert rm.aerobic_decoupling(s([200] * 1000), s([140] * 1000)) is None


# --- TSS ---------------------------------------------------------------------

def test_hr_tss_one_hour_at_lthr_is_100():
    assert rm.hr_tss(s([170] * 3600), 170) == pytest.approx(100)
    assert rm.hr_tss(s([136] * 3600), 170) == pytest.approx(64)  # (0.8)^2
    assert rm.hr_tss(s([140] * 10), None) is None


def test_training_stress_prefers_power_then_hr():
    hr = s([170] * 3600)
    tss, if_val, src = rm.training_stress(250, 3600, 250, hr, 170)
    assert (round(tss), round(if_val, 2), src) == (100, 1.0, 'power')

    tss, if_val, src = rm.training_stress(0, 3600, 250, hr, 170)
    assert (round(tss), if_val, src) == (100, 0.0, 'hr')

    assert rm.training_stress(0, 3600, 250, None, 170) == (0.0, 0.0, None)


def test_zone_seconds():
    df = pd.DataFrame({'power': [100.0] * 60 + [290.0] * 30, 'hr': [150.0] * 90})
    z = rm.zone_seconds(df, ftp=250, lthr=170)
    assert z['power']['Z1'] == 60 and z['power']['Z5'] == 30
    assert sum(z['power'].values()) == 90 and sum(z['hr'].values()) == 90
    assert rm.zone_seconds(df.drop(columns='hr'), 250, 170).keys() == {'power'}
