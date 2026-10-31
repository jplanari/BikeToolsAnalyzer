import sqlite3

import pandas as pd
import pytest

from src.data import db
from src.data import strava as sv
from src.data.history import recompute_history, ftp_used_for_ride
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_FILE", str(path))
    db.init_db()
    db.save_user("Rider", 300, 170, 70.0, 180.0)   # current FTP 300
    return path


def _legacy_ride(aid, meta, n=3600, **stream_kw):
    """A ride as stored before Phase 1: file kept, no new metrics."""
    payload = sv.build_ride_payload(activity(aid, "2026-09-01T07:00:00Z"), make_streams(n=n, **stream_kw))
    ok, msg = db.save_ride("Rider", sv.ride_filename(aid), payload, meta, {60: 1.0})
    assert ok, msg


def _row(path, filename):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    r = dict(conn.execute("SELECT * FROM rides WHERE filename = ?", (filename,)).fetchone())
    conn.close()
    return r


def test_ftp_used_for_ride():
    assert ftp_used_for_ride({'ftp_at_ride': 260, 'norm_power': 200, 'if_factor': 0.5}, 300) == 260
    assert ftp_used_for_ride({'norm_power': 200, 'if_factor': 0.8}, 300) == 250
    assert ftp_used_for_ride({'norm_power': 0, 'if_factor': 0}, 300) == 300


def test_recompute_backfills_metrics_and_keeps_original_ftp(temp_db):
    # Saved when FTP was 250: NP 200, IF 0.8 (and, pre-Phase-0, no date)
    _legacy_ride(1, {'np': 200, 'if': 0.8, 'tss': 64})
    assert db.count_rides_needing_update("Rider") == 1

    updated, failed = recompute_history("Rider", current_ftp=300, current_lthr=170)
    assert (updated, failed) == (1, [])
    assert db.count_rides_needing_update("Rider") == 0

    r = _row(temp_db, "strava_1.json")
    assert r['date_time'] == "2026-09-01T07:00:00+00:00"          # date recovered
    assert r['ftp_at_ride'] == 250                            # not the current 300
    assert r['tss'] == 64 and r['if_factor'] == pytest.approx(0.8)
    assert r['tss_source'] == 'power' and r['metrics_version'] == db.METRICS_VERSION
    assert r['duration_s'] == 3599 and r['moving_s'] == 3599  # 1st sample has no speed yet
    assert r['vi'] == pytest.approx(1.0) and r['ef'] == pytest.approx(200 / 140)
    assert r['decoupling_pct'] == pytest.approx(0.0, abs=1e-9)
    assert r['kj'] == pytest.approx(720)

    curve = db.get_user_best_power("Rider")
    assert 60 in curve and curve[60] == pytest.approx(200)    # placeholder record replaced
    assert 3600 in curve and 5400 not in curve
    zones = db.get_ride_zones([r['id']], 'power')
    # zone time counts moving seconds only (= moving_s)
    assert zones == {'Z1': 0, 'Z2': 0, 'Z3': 3599, 'Z4': 0, 'Z5': 0, 'Z6': 0, 'Z7': 0}


def test_recompute_gives_hr_tss_to_rides_without_power(temp_db):
    s = make_streams(n=3600, hr=170.0)
    del s["watts"]
    payload = sv.build_ride_payload(activity(2, "2026-09-02T07:00:00Z"), s)
    db.save_ride("Rider", "strava_2.json", payload, {'tss': 0}, {})

    recompute_history("Rider", current_ftp=300, current_lthr=170)
    r = _row(temp_db, "strava_2.json")
    assert r['tss_source'] == 'hr' and r['tss'] == 100      # 1 h at LTHR
    assert r['norm_power'] == 0 and r['vi'] is None
    assert db.get_user_tss_history("Rider")[0]['tss'] == 100  # counts in the PMC


def test_recompute_reports_unreadable_files(temp_db):
    db.save_ride("Rider", "broken.json", b"not json", {'tss': 5}, {})
    updated, failed = recompute_history("Rider", 300, 170)
    assert updated == 0 and failed and failed[0].startswith("broken.json")
    assert db.count_rides_needing_update("Rider") == 1


def test_new_rides_are_saved_up_to_date(temp_db):
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    payload = sv.build_ride_payload(activity(3), make_streams(n=1800))
    df, dist, ele = prepare_ride("strava_3.json", payload)
    summ = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, summ['avg_speed_kph'], summ['norm_power'],
                                  summ['tss'], summ['if'], filename="strava_3.json",
                                  metrics=metrics_for_storage(summ))
    assert db.save_ride("Rider", "strava_3.json", payload, meta, summ['power_curve'])[0]
    assert db.count_rides_needing_update("Rider") == 0
    r = _row(temp_db, "strava_3.json")
    assert (r['ftp_at_ride'], r['lthr_at_ride'], r['tss_source']) == (250, 170, 'power')
    assert sum(db.get_ride_zones([r['id']], 'hr').values()) == r['moving_s'] == 1799


def test_recompute_backs_up_the_database_first(temp_db):
    import glob
    _legacy_ride(1, {'np': 200, 'if': 0.8, 'tss': 64})
    recompute_history("Rider", current_ftp=300, current_lthr=170)
    backups = glob.glob(str(temp_db.parent / "backups" / "*before-recompute.db"))
    assert len(backups) == 1
    old = sqlite3.connect(backups[0])
    assert old.execute("SELECT metrics_version FROM rides").fetchone()[0] == 0   # state before the update
    # nothing to update -> no new backup
    recompute_history("Rider", current_ftp=300, current_lthr=170)
    assert len(glob.glob(str(temp_db.parent / "backups" / "*.db"))) == 1
