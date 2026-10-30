"""FTP / LTHR left empty follow the rider's data; values set by hand never move."""
import pandas as pd
import pytest

from src.data import db
from src.data import thresholds as th

TODAY = pd.Timestamp("2026-10-30")


def records(*rows):
    return pd.DataFrame([{'duration': 1200, 'watts': w, 'date': pd.Timestamp(d), 'ride_id': i}
                         for i, (d, w) in enumerate(rows)])


def test_ftp_is_95_percent_of_the_best_recent_20_minutes():
    recs = records(("2026-10-20", 300), ("2026-10-25", 280), ("2026-03-01", 400))
    assert th.estimate_ftp(recs, TODAY) == 285                       # the old 400 W is outside the 90 days


def test_ftp_falls_back_to_older_rides_and_needs_a_20_minute_effort():
    assert th.estimate_ftp(records(("2026-03-01", 400)), TODAY) == 380
    short = pd.DataFrame([{'duration': 300, 'watts': 400.0, 'date': TODAY, 'ride_id': 1}])
    assert th.estimate_ftp(short, TODAY) is None and th.estimate_ftp(None, TODAY) is None


def test_lthr_is_95_percent_of_the_best_20_minute_hr_and_ignores_artefacts():
    rows = [("2026-10-20", 160.0), ("2026-10-22", 172.0), ("2026-10-24", 0.0), ("2026-10-25", 400.0)]
    assert th.estimate_lthr(rows, TODAY) == 163                       # 0.95 * 172; 0 and 400 bpm are not real
    assert th.estimate_lthr([("2026-10-20", 0.0)], TODAY) is None and th.estimate_lthr([], TODAY) is None


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "t.db"))
    db.init_db()


def _ride(user, filename, date, watts, hr20):
    meta = {'date': date, 'dist': 30, 'ele': 100, 'speed': 28, 'np': watts, 'avg_p': watts, 'tss': 60,
            'if': 0.8, 'avg_hr': 140, 'hr20_bpm': hr20, 'metrics_version': db.METRICS_VERSION}
    assert db.save_ride(user, filename, filename.encode(), meta, {1200: float(watts)})[0]


def test_empty_thresholds_are_flagged_and_estimated(fresh_db):
    db.save_user("Auto", None, None, 70.0, 175.0)
    u = db.get_user_data("Auto")
    assert (u['ftp'], u['lthr'], u['ftp_auto'], u['lthr_auto']) == (200, 170, 1, 1)
    _ride("Auto", "a.fit", "2026-10-25T07:00:00", 300, 165.0)
    assert th.refresh_auto_thresholds("Auto", TODAY) == {'ftp': 285, 'lthr': 157}
    u = db.get_user_data("Auto")
    assert (u['ftp'], u['lthr'], u['ftp_auto'], u['lthr_auto']) == (285, 157, 1, 1)      # still following the data
    assert th.refresh_auto_thresholds("Auto", TODAY) == {}


def test_values_set_by_hand_never_move(fresh_db):
    db.save_user("Hand", 250, 170, 70.0, 175.0)
    _ride("Hand", "b.fit", "2026-10-25T07:00:00", 330, 180.0)
    assert th.refresh_auto_thresholds("Hand", TODAY) == {}
    assert db.get_user_data("Hand")['ftp'] == 250


def test_mixed_and_switching_back_and_forth(fresh_db):
    db.save_user("Mix", 250, None, 70.0, 175.0)                      # FTP given, LTHR left to the data
    _ride("Mix", "c.fit", "2026-10-25T07:00:00", 330, 170.0)
    assert th.refresh_auto_thresholds("Mix", TODAY) == {'lthr': 162}
    db.set_user_thresholds("Mix", lthr=150)                           # by hand: stops following
    assert th.refresh_auto_thresholds("Mix", TODAY) == {} and db.get_user_data("Mix")['lthr'] == 150
    db.set_user_thresholds("Mix", auto_ftp=True)                      # hand FTP back to the data
    assert th.refresh_auto_thresholds("Mix", TODAY) == {'ftp': 314}
    db.update_user_ftp("Mix", 260)                                    # accepting a suggestion is a manual choice
    assert db.get_user_data("Mix")['ftp_auto'] == 0


def test_rides_without_a_stored_hr_peak_are_listed_for_backfill(fresh_db):
    db.save_user("Old", None, None, 70.0, 175.0)
    _ride("Old", "d.fit", "2026-10-25T07:00:00", 200, None)
    assert [f for _, f in db.get_rides_without_hr20("Old")] == ["d.fit"]
    rid = db.get_rides_without_hr20("Old")[0][0]
    db.set_ride_hr20(rid, 0.0)
    assert db.get_rides_without_hr20("Old") == [] and db.get_hr20("Old") == []
