"""Durability per kg, differential W' balance, TRIMP and the omni-domain power-duration model."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import durability as du
from src.analysis import power as pw
from src.analysis import power_profile as pp
from src.analysis import trimp


# --- TRIMP ---------------------------------------------------------------------------------------------------------

def test_edwards_weights_minutes_by_zone():
    hr = pd.Series(np.full(3600, 150.0))                 # 75 % of 200: zone 3 for an hour
    assert trimp.edwards(hr, max_hr=200) == pytest.approx(180.0)
    mixed = pd.Series(np.r_[np.full(1800, 110.0), np.full(1800, 185.0)])   # 55 % then 92.5 %
    assert trimp.edwards(mixed, max_hr=200) == pytest.approx(30 * 1 + 30 * 5)
    assert trimp.edwards(pd.Series(np.full(600, 80.0)), max_hr=200) == 0.0   # below 50 %: no load


def test_banister_follows_the_published_formula():
    hr = pd.Series(np.full(3600, 125.0))                 # halfway between rest 50 and max 200
    expected = 60 * 0.5 * 0.64 * np.exp(1.92 * 0.5)
    assert trimp.banister(hr, rest_hr=50, max_hr=200) == pytest.approx(expected, rel=1e-9)
    women = trimp.banister(hr, 50, 200, constants=trimp.BANISTER_WOMEN)
    assert women == pytest.approx(60 * 0.5 * 0.86 * np.exp(1.67 * 0.5), rel=1e-9)
    assert women / trimp.banister(hr, 50, 200) == pytest.approx(1.344 * np.exp(-0.25 * 0.5), rel=0.01)


def test_trimp_needs_the_profile_and_ignores_stops():
    hr = pd.Series(np.full(600, 150.0))
    assert trimp.edwards(hr, max_hr=None) is None
    assert trimp.banister(hr, rest_hr=None, max_hr=190) is None
    assert trimp.banister(hr, rest_hr=190, max_hr=180) is None              # rest above max: nonsense
    moving = pd.Series([True] * 300 + [False] * 300)
    assert trimp.edwards(hr, 200, moving) == pytest.approx(trimp.edwards(hr[:300], 200))


# --- differential W' balance ---------------------------------------------------------------------------------------

def test_differential_balance_spends_above_cp_and_recovers_below():
    p = pd.Series(np.r_[np.full(60, 400.0), np.full(600, 100.0)])
    bal = pw.w_prime_balance_differential(p, cp=300, w_prime_cap_j=20000)
    assert bal.iloc[59] == pytest.approx(20000 - 60 * 100)                   # 100 W over CP for a minute
    assert bal.iloc[59] < bal.iloc[200] < bal.iloc[-1] <= 20000              # then recovers towards full
    assert pw.w_prime_balance_differential(pd.Series(np.full(300, 250.0)), 300, 20000).min() == 20000


def test_differential_recovers_faster_at_lower_power():
    hard = np.full(120, 450.0)
    easy = pw.w_prime_balance_differential(pd.Series(np.r_[hard, np.full(300, 50.0)]), 300, 20000).iloc[-1]
    tempo = pw.w_prime_balance_differential(pd.Series(np.r_[hard, np.full(300, 250.0)]), 300, 20000).iloc[-1]
    assert easy > tempo


def test_differential_balance_without_cp_is_empty():
    assert pw.w_prime_balance_differential(pd.Series([300.0] * 10), cp=0).empty


# --- durability per kg -------------------------------------------------------------------------------------------

def test_work_curves_use_total_and_above_cp_work_per_kg():
    p = pd.Series(np.full(3 * 3600, 200.0))              # 2160 kJ: 30.9 kJ/kg for 70 kg
    out = du.work_curves(p, weight_kg=70, cp=180)
    assert set(out['total_kjkg']) == {15.0, 30.0}       # 45 kJ/kg never reached
    assert set(out['above_cp_kjkg']) == {2.5}           # 20 W over CP for 3 h = 3.1 kJ/kg
    assert out['total_kjkg'][15.0][300] == pytest.approx(200.0)
    assert 'above_cp_kjkg' not in du.work_curves(p, weight_kg=70, cp=None)
    assert du.work_curves(p, weight_kg=None) == {}


def test_work_table_compares_fresh_and_tired_best():
    fresh = pd.DataFrame({'duration': [300], 'watts': [300.0], 'date': [pd.Timestamp('2026-06-01')]})
    tired = pd.DataFrame({'basis': ['total_kjkg', 'above_cp_kjkg'], 'threshold': [15.0, 2.5], 'duration': [300, 300],
                          'watts': [270.0, 240.0], 'date': [pd.Timestamp('2026-06-01')] * 2})
    t = du.work_table(fresh, tired, 'above_cp_kjkg')
    assert list(t['threshold']) == [2.5] and t['retention_pct'].iloc[0] == pytest.approx(80.0)
    assert du.work_table(fresh, tired.iloc[:0], 'total_kjkg').empty


# --- omni-domain power-duration model ------------------------------------------------------------------------------

DURS = [1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200]


def test_ompd_recovers_its_own_parameters():
    true = dict(cp=280.0, w_prime=22000.0, pmax=1100.0, a=25.0)
    curve = {d: float(pp._ompd([d], **true)[0]) for d in DURS}
    fit = pp.fit_ompd(curve)
    assert fit['long_end'] is True
    assert fit['cp'] == pytest.approx(280, rel=0.02) and fit['a'] == pytest.approx(25, rel=0.1)
    assert fit['rmse'] < 2.0


def test_ompd_without_long_efforts_does_not_invent_a_decline():
    true = dict(cp=250.0, w_prime=18000.0, pmax=900.0, a=0.0)
    curve = {d: float(pp._ompd([d], **true)[0]) for d in DURS if d <= 1800}
    fit = pp.fit_ompd(curve)
    assert fit['long_end'] is False and fit['a'] == 0.0
    assert fit['cp'] == pytest.approx(250, rel=0.03)


def test_ompd_needs_enough_points():
    assert pp.fit_ompd({1: 900.0, 60: 500.0, 300: 350.0}) is None
    assert pp.fit_ompd(None) is None
    assert pp.ompd_curve(None, DURS) == {}
