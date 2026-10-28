import os
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


# --- Ride files ----------------------------------------------------------------------

def test_new_rides_store_a_compressed_file_and_leave_the_row_small(temp_db):
    content = b'{"watts": [200, 200, 200]}' * 500
    _ride("big.json", content)
    conn = sqlite3.connect(db.DB_FILE)
    assert conn.execute("SELECT file_content FROM rides").fetchone()[0] is None
    codec, blob = conn.execute("SELECT codec, content FROM ride_files").fetchone()
    assert codec == "zlib" and len(blob) < len(content) / 5
    rid = conn.execute("SELECT id FROM rides").fetchone()[0]
    assert db.get_ride_file(rid) == ("big.json", content)


def test_incompressible_files_are_stored_raw(temp_db):
    import os
    content = os.urandom(2000)
    _ride("noise.fit", content)
    rid = sqlite3.connect(db.DB_FILE).execute("SELECT id FROM rides").fetchone()[0]
    assert db.get_ride_file(rid)[1] == content
    assert sqlite3.connect(db.DB_FILE).execute("SELECT codec FROM ride_files").fetchone()[0] == "raw"


def test_old_database_is_migrated_once_with_a_backup(tmp_path, monkeypatch):
    import glob, os
    path = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_FILE", str(path))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0)
    payloads = {f"r{i}.json": (b'{"w": [1, 2, 3]}' * 400 + bytes([i])) for i in range(60)}
    conn = sqlite3.connect(path)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    for name, body in payloads.items():                # how rides were stored before ride_files
        conn.execute("INSERT INTO rides (user_id, filename, file_hash, file_content, date_time) "
                     "VALUES (?, ?, ?, ?, '2026-10-01T07:00:00')", (uid, name, name, body))
    conn.commit()
    conn.close()
    size_before = os.path.getsize(path)

    db.init_db()                                       # next app start migrates

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM rides WHERE file_content IS NOT NULL").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ride_files").fetchone()[0] == 60
    for rid, name in conn.execute("SELECT id, filename FROM rides"):
        assert db.get_ride_file(rid) == (name, payloads[name])
    assert os.path.getsize(path) < size_before / 4      # compressed and vacuumed
    backups = glob.glob(str(tmp_path / "backups" / "*before-file-migration.db"))
    assert len(backups) == 1
    assert sqlite3.connect(backups[0]).execute(
        "SELECT COUNT(*) FROM rides WHERE file_content IS NOT NULL").fetchone()[0] == 60

    db.init_db()                                       # nothing left to do: no second backup
    assert len(glob.glob(str(tmp_path / "backups" / "*.db"))) == 1


def test_unmigrated_row_still_readable(temp_db):
    conn = sqlite3.connect(db.DB_FILE)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    conn.execute("INSERT INTO rides (user_id, filename, file_hash, file_content) VALUES (?, 'x.gpx', 'h', ?)",
                 (uid, b"legacy"))
    conn.commit()
    rid = conn.execute("SELECT id FROM rides").fetchone()[0]
    assert db.get_ride_file(rid) == ("x.gpx", b"legacy")


# --- Locked databases ----------------------------------------------------------------

def _legacy_db(tmp_path, monkeypatch, n=40):
    path = tmp_path / "locked.db"
    monkeypatch.setattr(db, "DB_FILE", str(path))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0)
    conn = sqlite3.connect(path)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    for i in range(n):
        conn.execute("INSERT INTO rides (user_id, filename, file_hash, file_content, date_time) "
                     "VALUES (?, ?, ?, ?, '2026-10-01T07:00:00')", (uid, f"r{i}.json", f"h{i}", b'{"w": 1}' * 300))
    conn.commit()
    conn.close()
    return path


def test_concurrent_app_starts_migrate_once_without_errors(tmp_path, monkeypatch):
    import threading
    path = _legacy_db(tmp_path, monkeypatch)
    errors = []

    def start():
        try:
            db.init_db()
        except Exception as e:                        # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=start) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM ride_files").fetchone()[0] == 40
    assert conn.execute("SELECT COUNT(*) FROM rides WHERE file_content IS NOT NULL").fetchone()[0] == 0


def test_locked_database_postpones_the_migration_instead_of_crashing(tmp_path, monkeypatch):
    path = _legacy_db(tmp_path, monkeypatch)
    monkeypatch.setattr(db, "BUSY_TIMEOUT_S", 0.2)
    other = sqlite3.connect(path, isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")                  # another process holds the file
    with pytest.warns(UserWarning, match="postponed"):
        assert db._move_ride_files() == 0
    other.execute("ROLLBACK")
    other.close()
    assert db._move_ride_files() == 40                # next run picks up where it was


def test_database_location_can_be_set_with_an_environment_variable(tmp_path):
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import src.data.db as d; print(d.DB_FILE)"],
                         env={**os.environ, "BIKETOOLS_DB": str(tmp_path / "elsewhere.db")},
                         capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.stdout.strip() == str(tmp_path / "elsewhere.db"), out.stderr
