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


# --- constant extra power term ------------------------------------------------------------

def test_joint_fit_recovers_cda_and_a_power_offset_that_the_plain_fit_absorbs():
    truth = sim.simulate(sim.SCENARIOS['hilly out-and-back, 4 m/s wind'])
    df, kw = sim.observe(truth, sim.Errors(power_offset_w=15.0), _rng())
    m = cc.ride_metrics(df, kw['rider_mass'], kw['bike_mass'], kw['crr'], with_still_air=False)
    assert m['cda'] > 0.305                                         # the offset is blamed on the air
    assert m['cda_joint'] == pytest.approx(0.30, abs=0.012)
    assert m['delta_w'] == pytest.approx(15 * (1 - 0.03), abs=3.0)
    assert m['delta_crr'] > 0.001                                  # worth about a fifth of a typical Crr
    clean_df, kw = sim.observe(truth, sim.Errors(), _rng())
    clean = cc.ride_metrics(clean_df, kw['rider_mass'], kw['bike_mass'], kw['crr'], with_still_air=False)
    assert abs(clean['delta_w']) < 3.0 and clean['cda_joint'] == pytest.approx(0.30, abs=0.01)


def test_joint_block_of_the_analysis_and_the_report():
    rides = _rides(30)
    rng = np.random.default_rng(5)
    rides['delta_w'] = 12 + rng.normal(0, 3, 30)
    rides['delta_crr'] = rides['delta_w'] / 2000.0
    rides['cda_joint'] = rides['cda'] - 0.01
    rides['cda_head_adj'], rides['cda_tail_adj'] = rides['cda'], rides['cda']
    res = cc.analyse(rides)
    j = res['joint']
    assert j['delta_med'] == pytest.approx(12, abs=2) and j['frac_positive'] > 0.95 and j['delta_p'] < 0.001
    assert j['ratio_adj'][1] == pytest.approx(1.0)
    text = cc.report("Ana", rides, res, "")
    assert "With a constant extra power term allowed" in text and "delta: median +1" in text


def test_sweep_reports_the_ratio_after_removing_the_constant_term():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.5)
    factors = (0.3, 0.5, 0.7)
    long_df = ws.sweep(rides, factors)
    assert {'ratio_adj', 'delta_w'} <= set(long_df.columns)
    res = ws.analyse(long_df, factors)
    assert {'median_ratio_adj', 'delta_med'} <= set(res['table'].columns)
    assert "constant extra power term" in ws.report("Ana", res, "")


# --- solo and group rides -----------------------------------------------------------------

def test_athlete_count_is_stored_only_when_strava_gave_it():
    import json
    from src.data import strava as sv
    from tests.test_strava import make_streams, activity
    with_n = json.loads(sv.build_ride_payload(activity(1, athlete_count=6), make_streams(n=60)))
    without = json.loads(sv.build_ride_payload(activity(1), make_streams(n=60)))
    assert with_n['activity']['athlete_count'] == 6 and 'athlete_count' not in without['activity']
    assert sv.build_ride_payload(activity(1), make_streams(n=60)) == sv.build_ride_payload(activity(1), make_streams(n=60))


def test_ride_type_from_the_stored_payload():
    import json
    from validation import ride_types as rt
    pay = lambda n: json.dumps({"activity": {"athlete_count": n}, "streams": {}}).encode()
    assert rt.type_from_content(pay(1)) == "solo" and rt.type_from_content(pay(7)) == "group"
    assert rt.type_from_content(pay(None)) is None and rt.type_from_content(pay(0)) is None
    assert rt.type_from_content(b"not json") is None and rt.type_from_content(b"<gpx/>") is None


def test_tags_file_is_read_applied_and_bad_lines_are_named(tmp_path):
    from validation import ride_types as rt
    f = tmp_path / "tags.csv"
    f.write_text("ride_id,date,name,type\n1,2026-01-01,a.fit,Solo\n2,2026-01-02,b.fit,GROUP\n3,2026-01-03,c.fit,\n"
                 ",,d.fit,group\n")
    tags = rt.read_tags(str(f))
    assert tags == {1: "solo", 2: "group", "d.fit": "group"}
    prepared = [dict(ride_id=1, name="a.fit", ride_type="group"), dict(ride_id=3, name="c.fit", ride_type="solo"),
                dict(ride_id=4, name="d.fit"), dict(ride_id=5, name="e.fit")]
    rt.apply_tags(prepared, tags)
    assert [p['ride_type'] for p in prepared] == ["solo", "solo", "group", "unknown"]     # file wins, then the import
    f.write_text("ride_id,type\n1,tandem\n")
    with pytest.raises(ValueError, match="line 2"):
        rt.read_tags(str(f))
    f.write_text("foo,bar\n1,solo\n")
    with pytest.raises(ValueError, match="header"):
        rt.read_tags(str(f))


def test_template_lists_every_ride_with_the_known_type_filled_in(tmp_path):
    from validation import ride_types as rt
    prepared = [dict(ride_id=1, date="2026-03-01T07:00:00Z", name="s1.json", ride_type="group"),
                dict(ride_id=2, date=None, name="s2.json", ride_type=None)]
    path = tmp_path / "t.csv"
    assert rt.write_template(prepared, str(path)) == 2
    back = rt.read_tags(str(path))
    assert back == {1: "group"}                      # the unfilled row stays unlabelled
    assert rt.summary(prepared) == {"solo": 0, "group": 1, "unknown": 1}


def _typed_rides():
    """Solo rides whose real wind effect is the app's 0.7; group rides sheltered down to 0.3 (drafting)."""
    solo = _rides_with_true_factor(0.7, n=10, seed=1)
    group = _rides_with_true_factor(0.3, n=10, seed=2)
    for p in solo:
        p['ride_type'] = "solo"
    for p in group:
        p['ride_id'] += 100
        p['ride_type'] = "group"
    return solo + group


def test_sweep_separates_sheltered_group_rides_from_solo_rides(tmp_path):
    from validation import wind_factor_sweep as ws
    factors = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    long_df = ws.sweep(_typed_rides(), factors)
    res = ws.analyse(long_df, factors)
    solo, group = res['types']['solo'], res['types']['group']
    assert solo['crossing'] == pytest.approx(0.7, abs=0.1) and group['crossing'] == pytest.approx(0.3, abs=0.1)
    text = ws.report("Ana", res, "")
    assert "## Solo and group rides" in text and "do not overlap" in text and "drafting explains part of the gap" in text
    assert res['types']['_unlabelled'] == 0


def test_sweep_without_labels_says_how_to_get_them():
    from validation import wind_factor_sweep as ws
    rides = _rides_with_true_factor(0.5, n=12, seed=4)
    factors = (0.3, 0.5, 0.7)
    res = ws.analyse(ws.sweep(rides, factors), factors)
    assert not [k for k in res['types'] if k in ("solo", "group")]
    assert "--tags" in ws.report("Ana", res, "")


def test_consistency_splits_by_ride_type_and_the_unlabelled_are_counted():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(24):
        kind = "solo" if i < 10 else ("group" if i < 20 else "unknown")
        ratio = 0.95 if kind == "solo" else 0.6
        cda = 0.35 + rng.normal(0, 0.01)
        rows.append(dict(cda=cda, cda_head=cda * ratio, cda_tail=cda, bike="Road", ride_type=kind,
                         has_wind=True, cda_still_air=cda, head_ms=1.0 + i * 0.1, cross_ms=0.5, speed_ms=8 + 0.1 * i,
                         grade_pct=0.5 + rng.normal(0, 0.1), temp_c=15 + rng.normal(0, 1)))
    rides = pd.DataFrame(rows)
    res = cc.analyse(rides)
    assert res['by_type']['solo']['ratio'] == pytest.approx(0.95) and res['by_type']['group']['ratio'] == pytest.approx(0.6)
    assert res['by_type']['p_ratio'] < 0.01 and res['unlabelled'] == 4
    text = cc.report("Ana", rides, res, "")
    assert "## Solo and group rides" in text and "| solo | 10 |" in text and "without a label (left out above): 4" in text


def test_tags_flow_through_the_database_loader_and_the_cli_template(temp_db, tmp_path, capsys):
    from src.data import db, strava as sv
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    from tests.test_strava import make_streams, activity
    db.save_user("Rider")
    payload = sv.build_ride_payload(activity(7, "2026-09-07T07:00:00Z", athlete_count=5), make_streams(n=1800))
    df, dist, ele = prepare_ride("strava_7.json", payload)
    summ = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, summ['avg_speed_kph'], summ['norm_power'], summ['tss'], summ['if'],
                                  filename="strava_7.json", metrics=metrics_for_storage(summ))
    assert db.save_ride("Rider", "strava_7.json", payload, meta, summ['power_curve'])[0]
    cc.main(["--user", "Rider", "--tags-template", str(tmp_path / "t.csv")])
    assert "1 group" in capsys.readouterr().out
    assert (tmp_path / "t.csv").read_text().strip().splitlines()[1].endswith(",group")
    rides, _ = cc.load_rides("Rider")
    if len(rides):
        assert rides['ride_type'].iloc[0] == "group"
        rides, _ = cc.load_rides("Rider", tags={"strava_7.json": "solo"})
        assert rides['ride_type'].iloc[0] == "solo"
    with pytest.raises(SystemExit, match="tags file"):
        cc.read_tags_or_exit(str(tmp_path / "missing.csv"))


# --- wind direction and timing sweep ------------------------------------------------------

def _rides_with_direction_error(err_deg, n=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sc = dict(terrain='hilly' if i % 2 else 'flat', route='out_and_back',
                  wind10=float(rng.uniform(3, 6)), wind_from=float(rng.uniform(0, 360)))
        truth = sim.simulate(sc)
        df, kw = sim.observe(truth, sim.Errors(wind_dir_deg=err_deg), rng)
        out.append(dict(df=df, rider_mass=kw['rider_mass'], bike_mass=kw['bike_mass'], crr=kw['crr'],
                        ride_id=i, date=f"2026-0{1 + i // 4}-{10 + i:02d}", name=f"r{i}", bike="Road"))
    return out


def test_shifting_the_wind_rotates_it_and_moves_it_in_time():
    from validation import wind_direction_sweep as wd
    t = pd.date_range("2026-06-01 10:00", periods=7200, freq="s")
    ramp = np.linspace(2.0, 8.0, len(t))                                  # wind speed grows over the two hours
    df = pd.DataFrame(dict(time=t, wind_speed=ramp, wind_deg=90.0))
    rot = wd.shifted(df, rotation_deg=30)
    assert np.allclose(rot['wind_deg'], 120.0) and np.allclose(rot['wind_speed'], ramp)
    later = wd.shifted(df, lag_h=1)                                       # read one hour later: stronger
    assert later['wind_speed'].iloc[0] == pytest.approx(ramp[3600], rel=1e-3)
    assert later['wind_speed'].iloc[-1] == pytest.approx(ramp[-1])        # clamped at the end of the ride
    assert (wd.shifted(df)['wind_speed'] == ramp).all() and df['wind_deg'].eq(90.0).all()      # the input is not touched


def test_sweep_finds_a_planted_direction_error_and_ignores_a_clean_set(tmp_path):
    from validation import wind_direction_sweep as wd
    kw = dict(factors=(0.7,), rotations=(-60, -40, -20, 0, 20, 40), lags=(0,))
    bad = _rides_with_direction_error(40.0)
    res = wd.analyse(wd.sweep(bad, **kw), (0.7,))['results'][(0.7, 'rotation')]
    assert res['improves'] and res['best']['offset'] == pytest.approx(-40, abs=20)
    assert abs(res['best']['median'] - 1.0) < abs(res['zero']['median'] - 1.0)
    clean = wd.analyse(wd.sweep(_rides_with_direction_error(0.0, seed=3), **kw), (0.7,))['results'][(0.7, 'rotation')]
    assert clean['best']['offset'] == 0 and not clean['improves']
    long_df = wd.sweep(bad, **kw)
    res_all = wd.analyse(long_df, (0.7,))
    text = wd.report("Ana", res_all, "")
    assert "Direction offset" in text and "Wind height factor 0.70" in text
    wd.write_outputs("Ana", long_df, res_all, "", str(tmp_path))
    assert (tmp_path / "wind_direction_sweep_Ana.md").exists()


def test_lag_sweep_runs_and_restores_the_factor():
    from validation import wind_direction_sweep as wd
    before = ae_factor()
    rides = _rides_with_direction_error(0.0, n=9, seed=5)
    res = wd.analyse(wd.sweep(rides, factors=(0.7,), rotations=(0,), lags=(-1, 0, 1)), (0.7,))
    t = res['results'][(0.7, 'lag')]['table']
    assert set(t['offset']) == {-1.0, 0.0, 1.0}
    assert ae_factor() == before


def ae_factor():
    from src.physics import aerodyn as ae
    return ae.WIND_HEIGHT_FACTOR


def test_random_wind_direction_error_makes_a_correct_factor_look_too_low():
    """Why the sweep cannot identify the factor: a ride-to-ride scatter of the model wind direction, which a constant
    offset cannot undo, moves the crossing well below the true 0.7 and the ratio at 0.7 below 1."""
    from src.physics import aerodyn as ae
    from validation import wind_factor_sweep as ws
    factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def rides(sd, n=24, seed=0):
        rng = np.random.default_rng(seed)
        saved, out = ae.WIND_HEIGHT_FACTOR, []
        try:
            ae.WIND_HEIGHT_FACTOR = 0.7
            for i in range(n):
                sc = dict(terrain='hilly' if i % 2 else 'flat', route='out_and_back',
                          wind10=float(rng.uniform(2, 6)), wind_from=float(rng.uniform(0, 360)))
                df, kw = sim.observe(sim.simulate(sc), sim.Errors(wind_dir_deg=float(rng.normal(0, sd))), rng)
                out.append(dict(df=df, rider_mass=kw['rider_mass'], bike_mass=kw['bike_mass'], crr=kw['crr'],
                                ride_id=i, date=f"2026-01-{i % 28 + 1:02d}", name=f"r{i}", bike="Road"))
        finally:
            ae.WIND_HEIGHT_FACTOR = saved
        return out

    exact = ws.analyse(ws.sweep(rides(0), factors), factors)
    noisy = ws.analyse(ws.sweep(rides(90), factors), factors)
    assert exact['crossing'] == pytest.approx(0.7, abs=0.06)
    assert noisy['crossing'] is not None and noisy['crossing'] < 0.45
    t = noisy['table']
    assert t[t['factor'] == 0.7]['median_ratio'].iloc[0] < 0.9
