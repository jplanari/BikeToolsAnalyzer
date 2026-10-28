import sqlite3

import pytest

from src.data import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def _ride(name="a.gpx", content=b"payload"):
    meta = {'date': '2026-10-01T07:00:00', 'dist': 30, 'ele': 300, 'speed': 25, 'np': 200, 'avg_p': 180,
            'tss': 60, 'if': 0.8, 'avg_hr': 140, 'metrics_version': db.METRICS_VERSION,
            'zones': {'power': {'Z1': 60}}, 'hist': {'p_hr': {(150, 140): 30}},
            'fatigued': {1000: {60: 200.0}}}
    assert db.save_ride("Rider", name, content, meta, {60: 300.0})[0]


def _count(table):
    return sqlite3.connect(db.DB_FILE).execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_connections_enforce_foreign_keys(temp_db):
    assert db._connect().execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_deleting_a_user_removes_everything_they_own(temp_db):
    _ride()
    db.delete_user("Rider")
    for table in ("rides", "power_records", "ride_zones", "ride_hist", "power_records_kj"):
        assert _count(table) == 0, table


def test_child_rows_need_a_real_ride(temp_db):
    conn = db._connect()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ride_zones (ride_id, kind, zone, seconds) VALUES (999, 'power', 'Z1', 5)")


def test_backup_is_a_consistent_copy_and_old_ones_are_pruned(temp_db, tmp_path, monkeypatch):
    _ride()
    stamps = iter(f"20261028-00000{i}" for i in range(6))
    monkeypatch.setattr(db.time, "strftime", lambda fmt: next(stamps))
    paths = [db.backup_db("test", keep=3) for _ in range(5)]
    assert all(paths)
    copy = sqlite3.connect(paths[-1])
    assert copy.execute("SELECT COUNT(*) FROM rides").fetchone()[0] == 1
    assert copy.execute("SELECT COUNT(*) FROM ride_hist").fetchone()[0] == 1
    import glob, os
    left = sorted(glob.glob(os.path.join(os.path.dirname(db.DB_FILE), "backups", "*.db")))
    assert [os.path.basename(p) for p in left] == [os.path.basename(p) for p in paths[-3:]]


def test_backup_without_a_database_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "missing.db"))
    assert db.backup_db() is None
