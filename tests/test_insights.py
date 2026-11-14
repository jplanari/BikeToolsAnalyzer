"""How you're evolving: findings computed from stored data with fixed rules."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import insights as ins
from src.analysis import training_load as tl

TODAY = pd.Timestamp("2026-11-14")


def _stats(rows):
    cols = ['date', 'moving_s', 'tss', 'vi', 'ef', 'decoupling_pct', 'hrr60_bpm']
    return pd.DataFrame(rows, columns=cols)


def _ride(day, tss=80, moving_h=2.0, ef=1.60, dec=3.0, hrr=30.0, vi=1.05):
    return (TODAY - pd.Timedelta(days=day), moving_h * 3600, tss, vi, ef, dec, hrr)


def _pmc(daily_tss):
    """daily_tss: {days_ago: tss}"""
    hist = [{'date': TODAY - pd.Timedelta(days=d), 'tss': v} for d, v in daily_tss.items()]
    return tl.compute_pmc(hist, today=TODAY)


def _by_key(found):
    return {f.key: f for f in found}


def test_volume_compares_with_the_period_before_and_flags_a_big_jump():
    stats = _stats([_ride(d, tss=150) for d in (1, 3, 5)] + [_ride(d, tss=60) for d in (8, 10)])
    f = _by_key(ins.evolution(TODAY, 7, stats=stats))['volume']
    assert f.value == 450 and f.previous == 120 and f.level == 'caution'
    empty = _stats([_ride(d) for d in (8, 10)])
    assert "No rides" in _by_key(ins.evolution(TODAY, 7, stats=empty))['volume'].text
    calm = _stats([_ride(d, tss=100) for d in (1, 3, 8, 10)])
    assert _by_key(ins.evolution(TODAY, 7, stats=calm))['volume'].level == 'neutral'


def test_fitness_ramp_and_form_follow_the_load():
    building = _pmc({d: 60 for d in range(120, 28, -1)} | {d: 160 for d in range(28, 0, -1)})
    f = _by_key(ins.evolution(TODAY, 28, pmc=building))
    assert f['fitness'].extra['per_week'] > ins.RAMP_AGGRESSIVE and f['fitness'].level == 'warning'
    assert f['form'].value < -10
    steady = _pmc({d: 80 for d in range(200, 0, -1)})
    f = _by_key(ins.evolution(TODAY, 28, pmc=steady))
    assert f['fitness'].level == 'neutral' and abs(f['fitness'].extra['per_week']) < ins.RAMP_BUILDING


def test_monotony_is_flagged_only_when_every_day_is_alike():
    alike = _pmc({d: 100 + (d % 2) for d in range(60, 0, -1)})
    assert 'monotony' in _by_key(ins.evolution(TODAY, 7, pmc=alike))
    varied = _pmc({d: (200 if d % 3 == 0 else 20) for d in range(60, 0, -1)})
    assert 'monotony' not in _by_key(ins.evolution(TODAY, 7, pmc=varied))


def _records(rows):
    return pd.DataFrame(rows, columns=['duration', 'watts', 'date', 'ride_id'])


def test_new_bests_and_critical_power_trend():
    old, new = TODAY - pd.Timedelta(days=40), TODAY - pd.Timedelta(days=2)
    rows = [(d, w, old, 1) for d, w in ((180, 360), (300, 330), (600, 300), (1200, 280))]
    rows += [(d, w * 1.06, new, 2) for d, w in ((180, 360), (300, 330), (600, 300), (1200, 280))]
    f = _by_key(ins.evolution(TODAY, 7, records=_records(rows)))
    assert f['bests'].level == 'good' and f['bests'].extra['bests'][300][0] == pytest.approx(330 * 1.06)
    assert f['cp'].level == 'good' and f['cp'].value > f['cp'].previous


def test_aerobic_trend_needs_comparable_rides():
    better = _stats([_ride(d, ef=1.70, dec=3.0) for d in (2, 5)] + [_ride(d, ef=1.60, dec=4.0) for d in (9, 12)])
    f = _by_key(ins.evolution(TODAY, 7, stats=better))
    assert f['ef'].level == 'good' and f['decoupling'].level == 'good'
    drifting = _stats([_ride(d, dec=8.0) for d in (2, 5)])
    assert _by_key(ins.evolution(TODAY, 7, stats=drifting))['decoupling'].level == 'caution'
    short = _stats([_ride(d, moving_h=0.5) for d in (2, 5, 9, 12)])     # under an hour: not comparable
    assert 'ef' not in _by_key(ins.evolution(TODAY, 7, stats=short))


def test_slower_heart_rate_recovery_is_flagged():
    stats = _stats([_ride(d, hrr=22) for d in (1, 3, 5)] + [_ride(d, hrr=30) for d in (8, 10, 12)])
    f = _by_key(ins.evolution(TODAY, 7, stats=stats))['hrr']
    assert f.level == 'caution' and f.value == 22 and f.previous == 30


def test_intensity_distribution_and_ordering():
    zones = pd.DataFrame({'date': [TODAY - pd.Timedelta(days=1)] * 3, 'zone': ['Z2', 'Z3', 'Z5'],
                          'seconds': [8000, 1000, 1000]})
    found = ins.evolution(TODAY, 7, stats=_stats([_ride(1, tss=300), _ride(9, tss=50)]), zone_rows=zones)
    f = _by_key(found)
    assert f['intensity'].extra['low'] == pytest.approx(0.8)
    assert [x.level for x in found] == sorted((x.level for x in found), key=ins.LEVELS.index)


def test_every_finding_cites_a_reference_the_app_lists():
    from src.references import REFERENCES
    keys = {r['key'] for r in REFERENCES}
    stats = _stats([_ride(d, hrr=22) for d in (1, 3, 5)] + [_ride(d, hrr=30) for d in (8, 10, 12)])
    pmc = _pmc({d: 100 + (d % 2) for d in range(90, 0, -1)})
    found = ins.evolution(TODAY, 7, pmc=pmc, stats=stats)
    assert found and all(f.ref in keys for f in found)


def test_no_data_gives_an_honest_headline():
    assert ins.evolution(TODAY, 7) == []
    assert "Not enough" in ins.headline([])
