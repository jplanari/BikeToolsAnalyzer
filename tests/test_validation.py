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
