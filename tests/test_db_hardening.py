import sqlite3

import pytest

from src.data import db


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = tmp_path / "t.db"
    monkeypatch.setattr(db, "DB_FILE", str(p))
    db._initialised.discard(str(p))
    return p


def _scalar(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def test_new_database_uses_wal_and_current_schema_version(path):
    db.init_db()
    assert _scalar(path, "PRAGMA journal_mode") == "wal"
    assert _scalar(path, "PRAGMA user_version") == db.MIGRATIONS[-1][0]


def test_init_db_runs_once_per_database_file(path, monkeypatch):
    calls = []
    real = db._init_db
    monkeypatch.setattr(db, "_init_db", lambda: calls.append(1) or real())
    db.init_db(); db.init_db(); db.init_db()
    assert len(calls) == 1
    db.init_db(force=True)
    assert len(calls) == 2


def test_init_db_redoes_the_work_when_the_file_is_gone(path):
    import os
    db.init_db()
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(str(path) + ext):
            os.remove(str(path) + ext)
    db.init_db()
    assert _scalar(path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'users'") == 1


def test_connect_commits_rolls_back_and_closes(path):
    db.init_db()
    with db.connect() as conn:
        conn.execute("INSERT INTO users (name) VALUES ('a')")
    assert _scalar(path, "SELECT COUNT(*) FROM users") == 1
    with pytest.raises(RuntimeError):
        with db.connect() as conn:
            conn.execute("INSERT INTO users (name) VALUES ('b')")
            raise RuntimeError("boom")
    assert _scalar(path, "SELECT COUNT(*) FROM users") == 1
    with pytest.raises(sqlite3.ProgrammingError):          # closed on exit
        conn.execute("SELECT 1")


def test_add_column_ignores_existing_but_not_real_errors(path):
    db.init_db()
    conn = sqlite3.connect(path)
    c = conn.cursor()
    db._add_column(c, "users", "ftp", "INTEGER")           # exists: no error
    db._add_column(c, "users", "extra_col", "TEXT")
    assert "extra_col" in {r[1] for r in c.execute("PRAGMA table_info(users)")}
    with pytest.raises(sqlite3.OperationalError):          # a missing table is a real error, not swallowed
        db._add_column(c, "no_such_table", "x", "TEXT")
    conn.close()


def test_two_riders_can_save_the_same_file_but_not_twice(path):
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    db.save_user("Ben", 250, 170, 70.0)
    f = b"the same club ride"
    assert db.save_ride("Ana", "club.gpx", f, {'date': "2026-10-01T07:00:00+00:00"}, {})[0]
    assert db.save_ride("Ben", "club.gpx", f, {'date': "2026-10-01T07:00:00+00:00"}, {})[0]
    ok, msg = db.save_ride("Ana", "club-again.gpx", f, {'date': "2026-10-01T07:00:00+00:00"}, {})
    assert not ok and "already exists" in msg


def _legacy_global_hash_db(path):
    """A database as shipped before migration 1: file_hash UNIQUE across all users, version 0."""
    db.init_db()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA legacy_alter_table = ON")         # keep the children pointing at the name "rides"
    conn.executescript("""
        DROP INDEX IF EXISTS idx_rides_user_hash;
        ALTER TABLE rides RENAME TO rides_old;
        CREATE TABLE rides (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, filename TEXT,
            file_hash TEXT UNIQUE, file_content BLOB, date_time TEXT, distance_km REAL, elevation_m REAL,
            avg_speed_kph REAL, norm_power INTEGER, avg_power INTEGER, tss INTEGER, if_factor REAL, avg_hr INTEGER,
            strava_activity_id INTEGER, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
        DROP TABLE rides_old;
        CREATE INDEX idx_rides_user_date ON rides(user_id, date_time, tss);
        PRAGMA user_version = 0;
    """)
    conn.commit()
    conn.close()


def test_migration_rebuilds_rides_keeping_rows_children_and_indexes(path):
    _legacy_global_hash_db(path)
    db.save_user("Ana", 250, 170, 70.0)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    for i in range(3):
        conn.execute("INSERT INTO rides (id, user_id, filename, file_hash, date_time, tss) VALUES (?, ?, ?, ?, ?, ?)",
                     (10 + i, uid, f"r{i}", f"h{i}", "2026-10-01T07:00:00", 50 + i))
    conn.execute("INSERT INTO ride_files (ride_id, codec, content) VALUES (12, 'raw', x'00')")
    conn.execute("DELETE FROM rides WHERE id = 12")                 # the high-water mark must survive
    conn.execute("INSERT INTO rides (id, user_id, filename, file_hash) VALUES (12, ?, 'r2', 'h2')", (uid,))
    conn.commit()
    conn.close()
    assert "file_hash TEXT UNIQUE" in _scalar(path, "SELECT sql FROM sqlite_master WHERE name = 'rides'")

    db.init_db(force=True)

    assert _scalar(path, "PRAGMA user_version") == db.MIGRATIONS[-1][0]
    assert "UNIQUE" not in _scalar(path, "SELECT sql FROM sqlite_master WHERE name = 'rides'")
    assert _scalar(path, "SELECT COUNT(*) FROM rides") == 3
    assert _scalar(path, "SELECT tss FROM rides WHERE id = 11") == 51
    assert _scalar(path, "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('idx_rides_user_date', 'idx_rides_user_hash')") == 2
    assert _scalar(path, "SELECT COUNT(*) FROM pragma_foreign_key_check") == 0
    db.save_user("Ben", 250, 170, 70.0)
    ok, msg = db.save_ride("Ben", "x", b"fresh", {}, {})
    assert ok, msg
    assert _scalar(path, "SELECT MAX(id) FROM rides") >= 13        # ids keep counting up
    # deleting a rider still cascades to the rebuilt table
    db.delete_user("Ana")
    assert _scalar(path, "SELECT COUNT(*) FROM rides WHERE user_id = %d" % uid) == 0


def test_failed_migration_leaves_the_database_untouched(path, monkeypatch):
    _legacy_global_hash_db(path)
    def boom(conn):
        conn.execute("DROP TABLE rides")
        raise RuntimeError("half way")
    monkeypatch.setattr(db, "MIGRATIONS", [(1, boom)])
    with pytest.raises(RuntimeError):
        db.init_db(force=True)
    assert _scalar(path, "PRAGMA user_version") == 0
    assert _scalar(path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'rides'") == 1


# --- dates: UTC storage, the rider's own day ---------------------------------------------------

def _ride(db_, name, start, f):
    return db_.save_ride(name, f"{f}.gpx", f.encode(), {'date': start, 'tss': 50, 'dist': 10.0}, {60: 100.0})


def test_dates_are_stored_as_utc_instants(path):
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    _ride(db, "Ana", "2026-10-31T23:30:00+01:00", "a")               # an aware time in another offset
    _ride(db, "Ana", "2026-11-01T08:00:00", "b")                     # a naive time is UTC
    stored = {_scalar(path, f"SELECT date_time FROM rides WHERE filename = '{n}.gpx'") for n in "ab"}
    assert stored == {"2026-10-31T22:30:00+00:00", "2026-11-01T08:00:00+00:00"}
    assert _scalar(path, "SELECT date_time FROM power_records LIMIT 1").endswith("+00:00")


def test_period_filter_uses_the_riders_local_day(path):
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    db.set_user_timezone("Ana", "Europe/Madrid")
    _ride(db, "Ana", "2026-10-25T23:30:00+00:00", "late")           # 00:30 on 26 Oct in Madrid (CET, UTC+1)
    _ride(db, "Ana", "2026-10-26T12:00:00+00:00", "noon")
    sunday = db.get_ride_stats("Ana", "2026-10-19", "2026-10-25")
    monday = db.get_ride_stats("Ana", "2026-10-26", "2026-10-26")
    assert len(sunday) == 0 and len(monday) == 2                     # the late ride is Monday for her
    assert monday['date'].dt.day.tolist() == [26, 26]
    assert db.period_coverage("Ana", "2026-10-26", "2026-10-26")['rides'] == 2
    history = db.get_user_tss_history("Ana")
    assert history[0]['date'] == "2026-10-26T00:30:00"                # training-load days are hers too
    assert db.get_user_rides("Ana")[-1]['start_local'] == "2026-10-26T00:30:00"


def test_other_zones_and_the_computer_default(path):
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    _ride(db, "Ana", "2026-10-25T22:30:00+00:00", "x")
    assert len(db.get_ride_stats("Ana", "2026-10-25", "2026-10-25")) == 1          # BIKETOOLS_TZ=UTC in tests
    db.set_user_timezone("Ana", "Pacific/Auckland")                                  # already the 26th there
    assert len(db.get_ride_stats("Ana", "2026-10-25", "2026-10-25")) == 0
    db.set_user_timezone("Ana", "Not/AZone")                                         # unknown: back to the default
    assert db.get_user_data("Ana")['timezone'] is None


def test_migration_normalizes_old_date_formats(path):
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    _ride(db, "Ana", "2026-10-01T07:00:00+00:00", "old")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE rides SET date_time = '2026-10-01T09:00:00+02:00'")     # as an older version stored it
    conn.execute("UPDATE power_records SET date_time = '2026-10-01T07:00:00'")
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
    db.init_db(force=True)
    assert _scalar(path, "SELECT date_time FROM rides") == "2026-10-01T07:00:00+00:00"
    assert _scalar(path, "SELECT date_time FROM power_records") == "2026-10-01T07:00:00+00:00"


def test_new_indexes_exist(path):
    db.init_db()
    names = {r for r in (sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type = 'index'"))}
    names = {n[0] for n in names}
    assert {"idx_power_records_user_dur", "idx_power_records_kj_user", "idx_planned_sessions_user_date",
            "idx_rides_user_hash"} <= names


def test_delete_user_takes_a_backup_first(path):
    import glob
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    db.delete_user("Ana")
    assert len(glob.glob(str(path.parent / "backups" / "*before-delete-user.db"))) == 1
    assert db.get_user_data("Ana") is None
