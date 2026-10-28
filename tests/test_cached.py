import numpy as np
import pandas as pd
import pytest

from src.data import db, cached


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def test_parse_dates_mixed_formats_and_gaps():
    s = pd.Series(["2026-10-01T07:00:00", "2026-10-01T09:00:00+02:00", None, "garbage",
                   "2026-10-01T07:00:00", "2026-10-02"], index=[5, 6, 7, 8, 9, 10])
    out = db._parse_dates(s)
    assert list(out.index) == [5, 6, 7, 8, 9, 10]
    assert str(out.dtype) == "datetime64[ns]"
    assert out[5] == pd.Timestamp("2026-10-01 07:00") and out[6] == pd.Timestamp("2026-10-01 07:00")
    assert out[9] == out[5] and out[10] == pd.Timestamp("2026-10-02")
    assert out[[7, 8]].isna().all()


def test_parse_dates_empty():
    out = db._parse_dates(pd.Series([], dtype=object))
    assert len(out) == 0 and str(out.dtype) == "datetime64[ns]"
    assert db._parse_dates(pd.Series([None, None])).isna().all()


def test_indexes_created(temp_db):
    import sqlite3
    names = {r[0] for r in sqlite3.connect(db.DB_FILE).execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"idx_rides_user_date", "idx_power_records_ride"} <= names
    db.init_db()                                   # idempotent


def test_data_stamp_changes_on_write(temp_db):
    before = db.data_stamp()
    db.save_user("Other", 200, 160, 65.0)
    assert db.data_stamp() != before


def test_cached_reads_match_db_and_see_new_data(temp_db):
    assert cached.get_user_data("Rider") == db.get_user_data("Rider")
    assert cached.get_user_data("Nobody") is None
    db.save_user("Other", 200, 160, 65.0)
    assert cached.get_user_data("Other")["ftp"] == 200
    pd.testing.assert_frame_equal(cached.get_power_records("Rider"), db.get_power_records("Rider"))
    pmc = cached.compute_pmc([{"date": "2026-10-01", "tss": 50}], today=pd.Timestamp("2026-10-03"))
    assert len(pmc) == 3
