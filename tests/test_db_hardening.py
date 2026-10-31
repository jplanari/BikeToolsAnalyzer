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

    assert _scalar(path, "PRAGMA user_version") == 1
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
