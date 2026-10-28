"""Regressions for smart recording, enhanced FIT channels and moving detection."""
import numpy as np
import pandas as pd
import pytest

from src.data import db, fit, pipeline
from src.data import strava as sv
from src.data.gpx import resample_to_seconds
from tests.test_strava import activity, real_distances  # noqa: F401 (fixture)


def _ride(times, **cols):
    base = {'lat': 41.0, 'lon': 2.0, 'ele': 100.0, 'hr': 140.0, 'power': 200.0, 'cadence': 90.0}
    base.update(cols)
    return pd.DataFrame({'time': times, **base})


def test_smart_recording_keeps_power():
    # A sample every 3 s at a steady 200 W used to average 67 W (gaps zero-filled)
    r = resample_to_seconds(_ride(pd.date_range('2026-10-01', periods=200, freq='3s'), speed=8.0))
    assert len(r) == 598
    assert r['power'].mean() == pytest.approx(200)
    assert (r['cadence'] == 90).all() and (r['speed'] == 8).all()


def test_pauses_are_still_stops():
    times = list(pd.date_range('2026-10-01 00:00:00', periods=100, freq='1s')) + \
            list(pd.date_range('2026-10-01 00:02:40', periods=100, freq='1s'))   # 60 s pause
    r = resample_to_seconds(_ride(times, speed=8.0))
    paused = r[(r['time'] > '2026-10-01 00:01:39') & (r['time'] < '2026-10-01 00:02:40')]
    assert len(paused) == 60
    assert (paused['power'] == 0).all() and (paused['speed'] == 0).all()
    assert paused['lat'].notna().all()                     # position held


def test_gap_boundary():
    times = pd.to_datetime(['2026-10-01 00:00:00', '2026-10-01 00:00:10',    # 10 s: held
                            '2026-10-01 00:00:21'])                         # 11 s: stop
    r = resample_to_seconds(_ride(times))
    assert (r['power'].iloc[:11] == 200).all()
    assert (r['power'].iloc[11:21] == 0).all() and r['power'].iloc[21] == 200


def test_moving_mask_without_speed():
    df = pd.DataFrame({'power': [0, 200, 200, 0], 'cadence': [0, 0, 90, 85]})
    assert pipeline.moving_mask(df).tolist() == [False, True, True, True]
    df['speed'] = [0.0, 0.0, 0.0, 0.0]                     # trainer FIT with speed 0
    assert pipeline.moving_mask(df).tolist() == [False, True, True, True]
    df['speed'] = [1.0, 5.0, np.nan, 3.0]
    assert pipeline.moving_mask(df).tolist() == [False, True, False, True]


def test_fit_with_only_enhanced_channels(monkeypatch):
    """The reported bug: analytics empty because FIT rides had no plain 'speed'."""
    class Field:
        def __init__(self, name, value): self.name, self.value = name, value

    start = pd.Timestamp('2026-10-01 07:00:00')
    records = [[Field('timestamp', start + pd.Timedelta(seconds=i)),
                Field('position_lat', int((41 + i * 8 / 111_000) * 2**31 / 180)),
                Field('position_long', int(2 * 2**31 / 180)),
                Field('enhanced_altitude', 100.0), Field('enhanced_speed', 8.0),
                Field('heart_rate', 140), Field('cadence', 90), Field('power', 200)]
               for i in range(1800)]

    class FakeFit:
        def __init__(self, *_): pass
        def get_messages(self, kind): return records if kind == 'record' else []

    monkeypatch.setattr(fit.fitparse, 'FitFile', FakeFit, raising=False)
    df, dist, ele = pipeline.prepare_ride('ride.fit', b'fit-bytes')
    assert df['speed'].notna().all() and df['ele'].notna().all()

    s = pipeline.summarize_ride(df, 250, 170)
    assert s['moving_time_s'] == 1800
    assert sum(s['hist']['p_hr'].values()) > 1700
    assert sum(s['hist']['p_cad'].values()) == 1800


def test_smart_recorded_strava_ride_metrics():
    t = list(range(0, 3600, 3))
    streams = {"time": {"data": t},
               "latlng": {"data": [[41 + i * 8 / 111_000, 2.0] for i in t]},
               "altitude": {"data": [100.0] * len(t)}, "heartrate": {"data": [140.0] * len(t)},
               "watts": {"data": [200.0] * len(t)}, "cadence": {"data": [90.0] * len(t)}}
    payload = sv.build_ride_payload(activity(1), streams)
    df, _, _ = pipeline.prepare_ride('strava_1.json', payload)
    s = pipeline.summarize_ride(df, 250, 170)
    assert s['norm_power'] == pytest.approx(200, abs=0.5)
    assert s['avg_power'] == pytest.approx(200, abs=0.5)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def test_period_coverage_explains_empty_analytics(temp_db):
    df = pd.DataFrame({'time': pd.date_range('2026-10-01', periods=60, freq='1s')})
    db.save_ride("Rider", "old.gpx", b"a", db.build_ride_metadata(df, 1000, 0, 20, 0, 0, 0), {})
    cov = db.period_coverage("Rider", "2026-09-01", "2026-10-31")
    assert cov == {'rides': 1, 'needs_update': 1, 'p_hr': 0, 'p_cad': 0}
    assert db.period_coverage("Rider", "2026-01-01", "2026-01-31")['rides'] == 0
