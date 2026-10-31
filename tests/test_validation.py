"""The validation tooling (validation/): the simulation recovers the truth and the analysis finds planted biases."""
import numpy as np
import pandas as pd
import pytest

from validation import sim, cda_error_budget as eb, cda_consistency as cc
from tests.test_db import temp_db  # noqa: F401 (fixture)


def _rng():
    return np.random.default_rng(3)


def test_error_free_observation_recovers_the_true_cda_in_every_scenario():
    for name, sc in sim.SCENARIOS.items():
        err = sim.cda_error(sim.simulate(sc), sim.Errors(), _rng())
        assert err is not None and abs(err) < 0.01, name


def test_signs_of_the_main_errors():
    truth = sim.simulate(sim.SCENARIOS['hilly, still air'])
    e = lambda **k: sim.cda_error(truth, sim.Errors(**k), _rng())
    assert e(power_gain=0.02) > 0                      # meter reads high: more drag is blamed
    assert e(crr_rel=0.2) < 0                          # more rolling resistance assumed leaves less for the air
    assert e(rho_rel=0.03) < 0                         # denser air assumed: same drag from a smaller CdA
    assert e(grade_bias_pct=0.3) < 0 < e(grade_bias_pct=-0.3)


def test_ignoring_the_wind_biases_a_windy_out_and_back_low():
    truth = sim.simulate(sim.SCENARIOS['hilly out-and-back, 4 m/s wind'])
    assert sim.cda_error(truth, sim.Errors(wind_missing=True), _rng()) < -0.1


def test_error_budget_runs_and_reports(tmp_path):
    rng = _rng()
    oat = eb.one_at_a_time(rng)
    mc = eb.monte_carlo(4, rng)
    text = eb.report(oat, mc, 4)
    assert "all together" in text and "wind_dir_deg" in text
    eb.write_outputs(oat, mc, 4, str(tmp_path))
    assert (tmp_path / "cda_error_budget.md").exists() and (tmp_path / "cda_error_budget_montecarlo.csv").exists()


def test_ride_metrics_on_a_simulated_ride_with_wind():
    truth = sim.simulate(sim.SCENARIOS['hilly out-and-back, 4 m/s wind'])
    df, kw = sim.observe(truth, sim.Errors(), _rng())
    m = cc.ride_metrics(df, kw['rider_mass'], kw['bike_mass'], kw['crr'])
    assert m['cda'] == pytest.approx(0.30, abs=0.01)
    assert m['cda_still_air'] < m['cda'] - 0.03        # the wind matters for this ride
    assert m['has_wind'] and m['head_ms'] == pytest.approx(0, abs=0.5)
    assert m['cda_head'] == pytest.approx(0.30, abs=0.02) and m['cda_tail'] == pytest.approx(0.30, abs=0.02)


def test_ride_metrics_without_enough_data_is_none():
    assert cc.ride_metrics(pd.DataFrame({'speed': [8.0] * 10, 'power': [200.0] * 10}), 70, 8, 0.005) is None


def _rides(n, trend=0.0, ratio=1.0, seed=0):
    rng = np.random.default_rng(seed)
    head = rng.uniform(-3, 3, n)
    cda = 0.30 + trend * head + rng.normal(0, 0.005, n)
    return pd.DataFrame({'cda': cda, 'cda_still_air': cda * 1.1, 'has_wind': True, 'bike': 'Road', 'head_ms': head,
                         'cross_ms': rng.uniform(0, 3, n), 'speed_ms': rng.uniform(7, 11, n),
                         'grade_pct': rng.uniform(-1, 2, n), 'temp_c': rng.uniform(5, 30, n),
                         'cda_head': cda * ratio, 'cda_tail': cda})


def test_a_planted_wind_bias_is_flagged_and_a_clean_set_is_not():
    clean = cc.analyse(_rides(30))
    assert not any(flag for *_, flag in clean['corr'].values())
    assert clean['head_tail'][1] == pytest.approx(1.0, abs=0.01)
    biased = cc.analyse(_rides(30, trend=0.02, ratio=1.15))
    assert biased['corr']["head wind (m/s)"][3] is True
    assert biased['head_tail'][1] == pytest.approx(1.15, abs=0.01) and biased['head_tail'][4] < 0.01
    assert biased['wind_effect'][1] == pytest.approx(-9.1, abs=0.5)


def test_report_text_and_few_rides():
    res = cc.analyse(_rides(5))
    text = cc.report("Ana", _rides(5), res, "")
    assert "Needs at least" in text and "CdA self-consistency: Ana" in text


def test_load_rides_from_the_database_end_to_end(temp_db, tmp_path):
    from src.data import db, strava as sv
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    from tests.test_strava import make_streams, activity
    db.save_user("Rider")
    payload = sv.build_ride_payload(activity(7, "2026-09-07T07:00:00Z"), make_streams(n=1800))
    df, dist, ele = prepare_ride("strava_7.json", payload)
    summ = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, summ['avg_speed_kph'], summ['norm_power'], summ['tss'], summ['if'],
                                  filename="strava_7.json", metrics=metrics_for_storage(summ))
    assert db.save_ride("Rider", "strava_7.json", payload, meta, summ['power_curve'])[0]
    rides, skipped = cc.load_rides("Rider")
    assert isinstance(rides, pd.DataFrame)
    assert len(rides) == 1 or "too few usable samples" in skipped or "no power or speed" in skipped
    if len(rides):
        assert {'cda', 'cda_still_air', 'bike'} <= set(rides.columns)
        cc.write_outputs("Rider", rides, cc.analyse(rides), skipped, str(tmp_path))
        assert (tmp_path / "cda_consistency_Rider.md").exists()


# --- wind height factor sweep -------------------------------------------------------------

def _rides_with_true_factor(true_factor, n=10, seed=0):
    """Simulated rides whose real wind effect is `true_factor` x the 10 m wind (the app assumes 0.7)."""
    from src.physics import aerodyn as ae
    rng = np.random.default_rng(seed)
    saved, out = ae.WIND_HEIGHT_FACTOR, []
    try:
        ae.WIND_HEIGHT_FACTOR = true_factor
        for i in range(n):
            sc = dict(terrain='hilly' if i % 2 else 'flat', route='out_and_back',
                      wind10=float(rng.uniform(3, 6)), wind_from=float(rng.uniform(0, 360)))
            truth = sim.simulate(sc)
            df, kw = sim.observe(truth, sim.Errors(), rng)
            out.append(dict(df=df, rider_mass=kw['rider_mass'], bike_mass=kw['bike_mass'], crr=kw['crr'],
                            ride_id=i, date=f"2026-0{1 + i // 4}-{10 + i:02d}", name=f"r{i}", bike="Road"))
    finally:
        ae.WIND_HEIGHT_FACTOR = saved
    return out


def test_sweep_finds_the_true_wind_height_factor_and_restores_the_setting(tmp_path):
    from src.physics import aerodyn as ae
    from validation import wind_factor_sweep as ws
    before = ae.WIND_HEIGHT_FACTOR
    rides = _rides_with_true_factor(0.4)
    factors = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    long_df = ws.sweep(rides, factors)
    assert ae.WIND_HEIGHT_FACTOR == before
    res = ws.analyse(long_df, factors)
    assert res['crossing'] == pytest.approx(0.4, abs=0.06)
    lo, hi, _ = res['crossing_ci']
    assert lo - 0.01 <= 0.4 <= hi + 0.01 and hi < 0.7                  # the data say the current 0.70 is too high
    for name, v in res['held_out'].items():
        assert v is not None and v[1] == pytest.approx(1.0, abs=0.1) and v[2] < 0.9      # 0.70 would be biased low
    text = ws.report("Ana", res, "")
    assert "do not support the current value" in text
    ws.write_outputs("Ana", long_df, res, "", str(tmp_path))
    assert (tmp_path / "wind_factor_sweep_Ana.md").exists() and (tmp_path / "wind_factor_sweep_Ana.csv").exists()


def test_sweep_says_so_when_the_current_factor_is_right():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.7)
    factors = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    res = ws.analyse(ws.sweep(rides, factors), factors)
    assert res['crossing'] == pytest.approx(0.7, abs=0.06)
    assert "no evidence to change it" in ws.report("Ana", res, "")


def test_crossing_helper():
    from validation import wind_factor_sweep as ws
    assert ws.crossing([0.2, 0.4, 0.6], [0.5, 0.8, 1.25]) == pytest.approx(0.5, abs=0.01)
    assert ws.crossing([0.2, 0.4], [0.5, 0.6]) is None


def test_a_crossing_below_the_range_is_flagged_not_reported_as_located():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.1)
    factors = (0.2, 0.3, 0.4, 0.5, 0.7)
    res = ws.analyse(ws.sweep(rides, factors), factors)
    assert res['edge']
    text = ws.report("Ana", res, "")
    assert "at or below the lowest factor tested (0.20)" in text and "bootstrap 90 % interval" not in text
    assert ws.crossing_label(factors, [0.9, 0.8, 0.7, 0.6, 0.5]) == "below 0.20"
    assert ws.crossing_label(factors, [1.5, 1.4, 1.3, 1.2, 1.1]) == "above 0.70"


def test_a_located_crossing_is_not_flagged():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.5)
    factors = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    res = ws.analyse(ws.sweep(rides, factors), factors)
    assert not res['edge'] and res['crossing'] == pytest.approx(0.5, abs=0.06)


def test_rides_are_grouped_by_how_windy_they_were_and_the_sets_are_reported():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.5, n=26, seed=2)
    factors = (0.3, 0.5, 0.7)
    long_df = ws.sweep(rides, factors)
    res = ws.analyse(long_df, factors)
    assert [g['name'] for g in res['strata']] == ["calmest third", "middle third", "windiest third"]
    assert sum(g['rides'] for g in res['strata']) == res['n_rides']
    assert res['strata'][0]['wind_hi'] <= res['strata'][2]['wind_lo'] + 1e-9
    assert set(res['sets']) == {'alternate rides, set A', 'alternate rides, set B', 'earlier rides', 'later rides'}
    text = ws.report("Ana", res, "")
    assert "windiest third" in text and "Reaches 1 at" in text
