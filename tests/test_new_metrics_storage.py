"""Max / resting heart rate in the profile, and TRIMP and per-kg durability stored with each ride."""
import numpy as np
import pandas as pd

from src.data import db
from src.data.processing.pipeline import metrics_for_storage, summarize_ride
from tests.test_db import temp_db  # noqa: F401 (fixture)


def test_max_and_resting_hr_are_validated(temp_db):
    db.save_user("Rider", 250, 170, 70.0, 175.0)
    assert db.set_user_hr("Rider", 190, 48) is True
    u = db.get_user_data("Rider")
    assert (u['max_hr'], u['rest_hr']) == (190, 48)
    for bad in ((300, 48), (190, 10), (150, 160), ("abc", 50)):
        assert db.set_user_hr("Rider", *bad) is False
    assert db.get_user_data("Rider")['max_hr'] == 190                   # a refused change changes nothing
    assert db.set_user_hr("Rider", None, None) is True                   # clearing is allowed
    assert db.get_user_data("Rider")['max_hr'] is None


def _ride(n=3 * 3600):
    t = pd.date_range("2026-10-01T08:00:00Z", periods=n, freq="1s")
    return pd.DataFrame({'time': t, 'power': np.full(n, 230.0), 'hr': np.full(n, 150.0),
                         'speed': np.full(n, 9.0), 'dist_m': np.arange(n) * 9.0})


def _save(df, rider):
    s = summarize_ride(df, 250, 170, rider)
    meta = db.build_ride_metadata(df, 97000, 0, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                  filename="r.fit", metrics=metrics_for_storage(s))
    ok, msg = db.save_ride("Rider", "r.fit", b"x" * 100, meta, s['power_curve'])
    assert ok, msg
    return s


def test_trimp_and_per_kg_durability_are_stored_and_replaced_on_update(temp_db):
    db.save_user("Rider", 250, 170, 70.0, 175.0)
    db.set_user_hr("Rider", 190, 50)
    rider = db.get_user_data("Rider")
    s = _save(_ride(), rider)
    assert s['trimp_edwards'] and s['trimp_banister']
    ride = db.get_user_rides("Rider")[0]
    stats = db.get_ride_stats("Rider")
    work = db.get_work_records("Rider")
    assert set(work['basis']) == {'total_kjkg'}                           # 230 W is below FTP: no work above CP
    assert set(work['threshold']) == {15.0, 30.0}                         # 2484 kJ = 35.5 kJ/kg
    # the profile changes and the ride is recomputed: old records are replaced, not added to
    db.set_user_hr("Rider", None, None)
    s2 = summarize_ride(_ride(), 200, 170, db.get_user_data("Rider"))
    meta = db.build_ride_metadata(_ride(), 97000, 0, s2['avg_speed_kph'], s2['norm_power'], s2['tss'], s2['if'],
                                  filename="r.fit", metrics=metrics_for_storage(s2))
    db.update_ride_metrics("Rider", ride['id'], meta, s2['power_curve'])
    assert s2['trimp_edwards'] is None                                    # no max HR any more
    work2 = db.get_work_records("Rider")
    assert set(work2['basis']) == {'total_kjkg', 'above_cp_kjkg'}         # FTP 200 now: 30 W above it for 3 h
    assert len(work2[work2['basis'] == 'total_kjkg']) == len(work[work['basis'] == 'total_kjkg'])
    assert len(stats) == 1


def test_without_a_profile_nothing_new_is_computed(temp_db):
    s = summarize_ride(_ride(3600), 250, 170)
    assert s['trimp_edwards'] is None and s['trimp_banister'] is None and s['fatigued_work'] == {}


def test_another_riders_records_are_not_returned(temp_db):
    db.save_user("Rider", 250, 170, 70.0, 175.0)
    db.save_user("Other", 250, 170, 70.0, 175.0)
    _save(_ride(), db.get_user_data("Rider"))
    assert db.get_work_records("Other").empty
