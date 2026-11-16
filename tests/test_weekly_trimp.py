"""TRIMP summed per week or month in the Volume chart."""
import numpy as np
import pandas as pd

from src.analysis import training_load as tl
from src.data import db
from tests.test_db import temp_db  # noqa: F401 (fixture)
from tests.test_new_metrics_storage import _save, _ride


def test_volume_sums_trimp_per_week_and_skips_rides_without_one():
    stats = pd.DataFrame({
        'date': pd.to_datetime(["2026-11-02", "2026-11-04", "2026-11-10"]),   # Mon + Wed, then next Tue
        'moving_s': [3600, 3600, 3600], 'distance_km': [30, 30, 30], 'elevation_m': [100, 100, 100],
        'kj': [700, 700, 700], 'tss': [60, 60, 60],
        'trimp_edwards': [120.0, np.nan, 90.0], 'trimp_banister': [70.0, 50.0, np.nan]})
    vol = tl.volume(stats, 'W')
    assert list(vol['trimp_edwards']) == [120.0, 90.0]
    assert list(vol['trimp_banister']) == [120.0, 0.0]


def test_volume_without_trimp_columns_still_works():
    stats = pd.DataFrame({'date': pd.to_datetime(["2026-11-02"]), 'moving_s': [3600], 'distance_km': [30],
                          'elevation_m': [0], 'kj': [700], 'tss': [60]})
    assert tl.volume(stats, 'W')['trimp_edwards'].iloc[0] == 0


def test_ride_stats_carry_the_stored_trimp(temp_db):
    db.save_user("Rider", 250, 170, 70.0, 175.0)
    db.set_user_hr("Rider", 190, 50)
    s = _save(_ride(3600), db.get_user_data("Rider"))
    stats = db.get_ride_stats("Rider")
    assert stats['trimp_edwards'].iloc[0] == s['trimp_edwards']
    assert stats['trimp_banister'].iloc[0] == s['trimp_banister']
