import numpy as np
import pandas as pd
import pytest

from src.analysis import ride_metrics as rm
from src.analysis import correlations as co
from src.data import db
from src.data import strava as sv
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


def cells(hist):
    return pd.DataFrame([(x, y, s) for (x, y), s in hist.items()], columns=['x', 'y', 'seconds'])


@pytest.fixture
def synthetic_ride():
    """2 h of 1-minute steady blocks with HR = 60 + 0.4 P and cadence = 70 + 0.08 P."""
    rng = np.random.default_rng(1)
    p = pd.Series(np.repeat(rng.uniform(120, 320, 120), 60))
    return pd.DataFrame({'power': p, 'hr': 60 + 0.4 * p, 'cadence': 70 + 0.08 * p})


# --- Per-ride histograms -------------------------------------------------------

def test_histograms_skip_coasting_and_stops():
    df = pd.DataFrame({'power': [200.0] * 60 + [0.0] * 60, 'hr': [140.0] * 120,
                       'cadence': [90.0] * 60 + [0.0] * 60})
    moving = pd.Series([True] * 90 + [False] * 30)
    h = rm.joint_histograms(df, moving)
    assert sum(h['p_cad'].values()) == 60                  # pedalling only
    assert set(h['p_cad']) == {(200, 90)}
    assert all(x > 0 for x, _ in h['p_hr'])                 # no zero-power cells
    assert rm.joint_histograms(df.drop(columns=['hr', 'cadence'])) == {}


def test_histograms_add_up_across_rides(synthetic_ride):
    first, second = synthetic_ride.iloc[:3600], synthetic_ride.iloc[3600:].reset_index(drop=True)
    parts = pd.concat([cells(rm.joint_histograms(first)['p_cad']),
                       cells(rm.joint_histograms(second)['p_cad'])])
    whole = cells(rm.joint_histograms(synthetic_ride)['p_cad'])
    pd.testing.assert_frame_equal(co.aggregate(parts).sort_values(['x', 'y']).reset_index(drop=True),
                                  co.aggregate(whole).sort_values(['x', 'y']).reset_index(drop=True))


# --- Power vs HR ----------------------------------------------------------------

def test_linear_fit_recovers_relationship(synthetic_ride):
    fit = co.linear_fit(cells(rm.joint_histograms(synthetic_ride)['p_hr']))
    assert fit['r'] > 0.99
    assert fit['slope'] == pytest.approx(0.4, abs=0.01)     # 4 bpm per 10 W
    assert fit['intercept'] == pytest.approx(60, abs=2)
    assert co.linear_fit(cells({(200, 140): 300})) is None  # < 10 min


def test_hr_at_power_and_curve(synthetic_ride):
    h = cells(rm.joint_histograms(synthetic_ride)['p_hr'])
    assert co.y_at_power(h, 200) == pytest.approx(140, abs=2)
    curve = co.mean_y_by_power(h)
    assert np.allclose(curve['mean'], 60 + 0.4 * curve['power'], atol=3)
    assert co.y_at_power(h, 600) is None                    # never ridden there


def test_weekly_groups_by_iso_week():
    h = pd.DataFrame({'date': pd.to_datetime(['2026-10-05', '2026-10-11', '2026-10-12']),
                      'x': [200, 200, 200], 'y': [140, 150, 130], 'seconds': [600, 600, 600]})
    w = co.weekly(h, lambda g: co.y_at_power(g, 205, min_seconds=60))
    assert list(w.index) == [pd.Timestamp('2026-10-05'), pd.Timestamp('2026-10-12')]
    assert list(w.values) == [147.5, 132.5]                 # Mon-Sun weeks


# --- Power vs cadence -------------------------------------------------------------

def test_weighted_median_cadence():
    h = pd.DataFrame({'x': [200, 200, 200], 'y': [80, 85, 95], 'seconds': [100, 300, 50]})
    # half of 450 s falls 125 s into the 300 s bin starting at 85 rpm
    assert co.weighted_median_y(h) == pytest.approx(85 + 125 / 300 * 5)
    # a continuous median is recovered within a fraction of a bin
    rng = np.random.default_rng(0)
    cad = rng.normal(86.3, 6, 20_000)
    hist = rm.joint_histograms(pd.DataFrame({'power': np.full(cad.size, 200.0), 'cadence': cad}))['p_cad']
    assert co.weighted_median_y(cells(hist)) == pytest.approx(np.median(cad), abs=0.3)


def test_quadrants_split_by_force_and_cadence():
    # FTP 250 at ref 90 rpm: force line at power/cadence = 2.78 W/rpm
    h = pd.DataFrame({'x': [350, 300, 100, 150], 'y': [100, 60, 60, 100],
                      'seconds': [100, 200, 300, 400]})
    q = co.quadrants(h, ftp=250, ref_cadence=90)
    assert q == pytest.approx({'QI': 0.1, 'QII': 0.2, 'QIII': 0.3, 'QIV': 0.4})


def test_cadence_by_zone(synthetic_ride):
    t = co.cadence_by_zone(cells(rm.joint_histograms(synthetic_ride)['p_cad']), 250)
    assert list(t['cadence']) == sorted(t['cadence'])        # cadence rises with zone
    assert t['share'].sum() == pytest.approx(1.0)


def test_pedal_force_velocity():
    aepf, cpv = co.pedal_force_velocity([250], [90], 0.1725)
    assert cpv[0] == pytest.approx(1.6258, abs=1e-3)
    assert aepf[0] == pytest.approx(153.8, abs=0.1)


# --- HR recovery -----------------------------------------------------------------

def test_hr_recovery_finds_effort_ends():
    p, hr = [], []
    for _ in range(3):
        p += [300] * 60 + [100] * 120
        hr += list(np.linspace(140, 170, 60)) + list(np.linspace(170, 145, 61)[1:]) + [145] * 60
    assert rm.hr_recovery(pd.Series(p, dtype=float), pd.Series(hr), 250) == (25.0, 3)


def test_hr_recovery_ignores_efforts_without_easy_minute():
    p = pd.Series([300.0] * 60 + [250.0] * 120)
    assert rm.hr_recovery(p, pd.Series([150.0] * 180), 250) == (None, 0)
    assert rm.hr_recovery(pd.Series([200.0] * 600), pd.Series([140.0] * 600), None) == (None, 0)


def test_endurance_filter():
    stats = pd.DataFrame({'moving_s': [4000, 1800, 5000], 'vi': [1.03, 1.02, 1.25], 'ef': [1.4, 1.5, 1.3]})
    assert list(co.endurance_rides(stats).index) == [0]


# --- Storage round trip -------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def _save_strava(aid, start, **kw):
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    payload = sv.build_ride_payload(activity(aid, start), make_streams(n=1800, **kw))
    df, dist, ele = prepare_ride(sv.ride_filename(aid), payload)
    summ = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, summ['avg_speed_kph'], summ['norm_power'],
                                  summ['tss'], summ['if'], filename=sv.ride_filename(aid),
                                  metrics=metrics_for_storage(summ))
    assert db.save_ride("Rider", sv.ride_filename(aid), payload, meta, summ['power_curve'])[0]


def test_histograms_stored_and_filtered_by_date(temp_db):
    _save_strava(1, "2026-09-01T07:00:00Z", power=200.0, hr=140.0)
    _save_strava(2, "2026-10-01T07:00:00Z", power=200.0, hr=130.0)

    sep = db.get_hist("Rider", 'p_hr', "2026-09-01", "2026-09-30")
    octo = db.get_hist("Rider", 'p_hr', "2026-10-01", "2026-10-01")   # end date inclusive
    assert set(sep['y']) == {140} and set(octo['y']) == {130}
    assert co.y_at_power(octo, 205) == 132.5
    assert len(db.get_hist("Rider", 'p_cad')) > 0

    stats = db.get_ride_stats("Rider", "2026-09-15", None)
    assert list(stats['ride_id']) == [2] and stats['ef'].iloc[0] == pytest.approx(200 / 130)
    assert db.get_hist("Nobody", 'p_hr').empty


def test_recompute_replaces_histograms(temp_db):
    from src.data.history import recompute_history
    _save_strava(3, "2026-09-03T07:00:00Z")
    before = len(db.get_hist("Rider", 'p_hr'))
    recompute_history("Rider", 250, 170, all_rides=True)
    assert len(db.get_hist("Rider", 'p_hr')) == before      # replaced, not duplicated


def test_crank_length_setting(temp_db):
    assert db.get_user_data("Rider")['crank_mm'] == 172.5
    db.update_user_crank("Rider", 170)
    assert db.get_user_data("Rider")['crank_mm'] == 170
