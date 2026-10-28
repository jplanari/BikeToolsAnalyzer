import numpy as np
import pandas as pd
import pytest

from src.analysis.training_load import compute_pmc


def test_empty_and_invalid_history():
    assert compute_pmc([]).empty
    assert compute_pmc([{'date': None, 'tss': 50}, {'date': '2026-01-01', 'tss': None}]).empty


def test_pmc_extends_to_today_and_decays():
    history = [{'date': '2026-10-01T08:00:00+00:00', 'tss': 100}]
    pmc = compute_pmc(history, until='2026-10-28')

    assert pmc.index[0] == pd.Timestamp('2026-10-01')
    assert pmc.index[-1] == pd.Timestamp('2026-10-28')
    assert len(pmc) == 28
    # Loads decay on rest days instead of freezing at the last ride
    assert pmc['CTL'].iloc[-1] < pmc['CTL'].iloc[0]
    assert pmc['ATL'].iloc[-1] < pmc['ATL'].iloc[0]


def test_pmc_standard_constants_from_zero():
    pmc = compute_pmc([{'date': '2026-10-01', 'tss': 100}], until='2026-10-02')
    k_ctl = 1 - np.exp(-1 / 42)
    k_atl = 1 - np.exp(-1 / 7)
    # Day 1 starts from zero load (old code set CTL = ATL = 100 on day 1)
    assert pmc['CTL'].iloc[0] == pytest.approx(100 * k_ctl)
    assert pmc['ATL'].iloc[0] == pytest.approx(100 * k_atl)
    # Day 2 is a rest day: pure exponential decay
    assert pmc['CTL'].iloc[1] == pytest.approx(100 * k_ctl * (1 - k_ctl))
    # TSB is yesterday's CTL - ATL
    assert pmc['TSB'].iloc[0] == 0
    assert pmc['TSB'].iloc[1] == pytest.approx(100 * k_ctl - 100 * k_atl)


def test_steady_load_converges():
    days = pd.date_range('2025-01-01', periods=400, freq='D')
    pmc = compute_pmc([{'date': d, 'tss': 60} for d in days], until=days[-1])
    assert pmc['CTL'].iloc[-1] == pytest.approx(60, abs=0.5)
    assert pmc['ATL'].iloc[-1] == pytest.approx(60, abs=0.01)


def test_multiple_rides_same_day_are_summed_and_timezones_mixed():
    history = [
        {'date': '2026-10-01T07:00:00+00:00', 'tss': 40},
        {'date': '2026-10-01 18:00:00', 'tss': 60},   # legacy naive format
    ]
    pmc = compute_pmc(history, until='2026-10-01')
    assert len(pmc) == 1
    assert pmc['tss'].iloc[0] == 100
