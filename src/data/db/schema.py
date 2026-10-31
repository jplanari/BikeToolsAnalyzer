"""Tables, migrations and init_db."""
from src.analysis.dates import to_utc_iso
import os
import re
import sqlite3
import threading
from . import core
from .auth import encrypt_plaintext_secrets
from .core import _add_column, _enable_wal, backup_db, connect
from .rides import _move_ride_files

# --- numbered migrations -------------------------------------------------------------------
# The schema version lives in PRAGMA user_version. Each migration runs once, in order, in its own transaction
# (with a backup first), and the version moves up only when it succeeded: a failure stops the app start with
# the real error instead of being swallowed. Add new migrations at the END; never edit one that has shipped.

def _migration_per_user_file_hash(conn):
    """
    rides.file_hash was UNIQUE across all users, so a second rider could not save a file another rider had
    (a club ride, say). Duplicates are now detected per rider: UNIQUE(user_id, file_hash). SQLite cannot drop
    a column constraint, so the table is rebuilt the way the SQLite documentation prescribes.
    """
    import re
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rides'").fetchone()
    if row and 'file_hash TEXT UNIQUE' in row[0]:
        indexes = [r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'rides' AND sql IS NOT NULL")]
        cols = ", ".join(f'"{r[1]}"' for r in conn.execute("PRAGMA table_info(rides)"))
        seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'rides'").fetchone()
        new_sql = re.sub(r'^CREATE TABLE\s+"?rides"?', 'CREATE TABLE rides_new', row[0].replace(
            'file_hash TEXT UNIQUE', 'file_hash TEXT'), count=1)
        conn.execute(new_sql)
        conn.execute(f"INSERT INTO rides_new ({cols}) SELECT {cols} FROM rides")
        conn.execute("DROP TABLE rides")
        conn.execute("ALTER TABLE rides_new RENAME TO rides")
        for sql in indexes:
            conn.execute(sql)
        if seq:
            conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'rides'", (seq[0],))
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_rides_user_hash ON rides(user_id, file_hash)")


def _migration_query_indexes(conn):
    """Indexes for the per-user queries that had none: best power per duration, planned sessions by date."""
    for name, table, cols in (
            ("idx_power_records_user_dur", "power_records", "user_id, duration_sec, watts"),
            ("idx_power_records_kj_user", "power_records_kj", "user_id, kj, duration_sec, watts"),
            ("idx_planned_sessions_user_date", "planned_sessions", "user_id, date")):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({cols})")


def _migration_utc_dates(conn):
    """
    Ride start times were stored in mixed forms (with an offset, or naive). They are all rewritten as UTC
    instants ('2026-10-01T07:00:00+00:00'), so text comparison orders them and the day filters can use the
    rider's local day. A naive time was always read as UTC, so nothing moves.
    """
    for table in ("rides", "power_records", "power_records_kj"):
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
            continue
        for (old,) in conn.execute(f"SELECT DISTINCT date_time FROM {table} WHERE date_time IS NOT NULL").fetchall():
            new = to_utc_iso(old)
            if new != old:
                conn.execute(f"UPDATE {table} SET date_time = ? WHERE date_time = ?", (new, old))


MIGRATIONS = [
    (1, _migration_per_user_file_hash),
    (2, _migration_query_indexes),
    (3, _migration_utc_dates),
]


def schema_version():
    conn = sqlite3.connect(core.DB_FILE, timeout=core.BUSY_TIMEOUT_S)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def run_migrations():
    """Applies the migrations newer than the database's user_version. Returns the versions applied."""
    done = []
    current = schema_version()
    pending = [(v, fn) for v, fn in MIGRATIONS if v > current]
    if not pending:
        return done
    if _has_rides():
        backup_db("before-migration")
    for version, fn in pending:
        # Rebuilding a table needs foreign keys off (changing it is impossible inside a transaction), so
        # this connection is not the usual one and manages its own transaction.
        conn = sqlite3.connect(core.DB_FILE, timeout=core.BUSY_TIMEOUT_S, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN IMMEDIATE")
            try:
                fn(conn)
                broken = conn.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise sqlite3.IntegrityError(f"migration {version} left {len(broken)} broken foreign keys")
                conn.execute(f"PRAGMA user_version = {int(version)}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        done.append(version)
    return done


def _has_rides():
    conn = sqlite3.connect(core.DB_FILE, timeout=core.BUSY_TIMEOUT_S)
    try:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'rides'").fetchone() is not None and \
            conn.execute("SELECT 1 FROM rides LIMIT 1").fetchone() is not None
    finally:
        conn.close()


def migrate_db():
    """
    Adds new columns (cp, w_prime) to the users table if they don't exist.
    Run this once safely; it catches errors if columns already exist.
    """
    with connect() as conn:
        c = conn.cursor()
    
        _add_column(c, 'users', 'cp', 'INTEGER DEFAULT 0')
        _add_column(c, 'users', 'timezone', 'TEXT')       # IANA name; NULL = this computer's zone

        _add_column(c, 'users', 'w_prime', 'REAL DEFAULT 20000')

        # 1 = FTP / LTHR were left empty and follow the rider's data (see src/data/thresholds.py)
        for col in ("ftp_auto", "lthr_auto"):
            _add_column(c, 'users', col, 'INTEGER DEFAULT 0')

        _add_column(c, 'rides', 'strava_activity_id', 'INTEGER')
        # Phase 1 per-ride metrics. metrics_version marks rides whose stored
        # metrics predate the current calculation (0 = legacy) so they can be
        # recomputed from their stored files.
        for col, typ in [("duration_s", "INTEGER"), ("moving_s", "INTEGER"), ("kj", "REAL"),
                         ("ftp_at_ride", "INTEGER"), ("lthr_at_ride", "INTEGER"),
                         ("vi", "REAL"), ("ef", "REAL"), ("decoupling_pct", "REAL"),
                         ("tss_source", "TEXT"), ("metrics_version", "INTEGER DEFAULT 0"),
                         ("hrr60_bpm", "REAL"), ("hrr_efforts", "INTEGER"),
                         ("hr20_bpm", "REAL")]:          # best 20-min mean HR; NULL = not computed yet, 0 = no HR data
            _add_column(c, 'rides', col, typ)
        # the bike a ride was done on: NULL = not chosen (the rider's default bike), 0 = the generic bike
        _add_column(c, 'rides', 'bike_id', 'INTEGER')
        c.execute('''
            CREATE TABLE IF NOT EXISTS bikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                frame_name TEXT, frame_kg REAL,
                wheelset_name TEXT, wheelset_kg REAL,
                groupset_name TEXT, groupset_kg REAL,
                tyres_name TEXT, tyres_kg REAL,
                extras_kg REAL,              -- pedals, bar, saddle, bottles...: whatever the four parts leave out
                crr REAL,                    -- rolling resistance of the tyres on the road
                is_default INTEGER DEFAULT 0,
                UNIQUE (user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS ride_zones (
                ride_id INTEGER NOT NULL,
                kind TEXT NOT NULL,          -- 'power' or 'hr'
                zone TEXT NOT NULL,
                seconds INTEGER NOT NULL,
                PRIMARY KEY (ride_id, kind, zone),
                FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
            )
        ''')

        # Joint distributions per ride (sparse 2D histograms, seconds per cell)
        c.execute('''
            CREATE TABLE IF NOT EXISTS ride_hist (
                ride_id INTEGER NOT NULL,
                pair TEXT NOT NULL,          -- 'p_hr' (power x HR) or 'p_cad' (power x cadence)
                x INTEGER NOT NULL,          -- lower edge of the power bin (W)
                y INTEGER NOT NULL,          -- lower edge of the HR (bpm) / cadence (rpm) bin
                seconds INTEGER NOT NULL,
                PRIMARY KEY (ride_id, pair, x, y),
                FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
            )
        ''')
        # Planned sessions (TSS) for the PMC forecast on the Training page
        c.execute('''
            CREATE TABLE IF NOT EXISTS planned_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,          -- YYYY-MM-DD
                tss REAL NOT NULL,
                note TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        # Mean-maximal power of the part of each ride after `kj` of work (durability)
        c.execute('''
            CREATE TABLE IF NOT EXISTS power_records_kj (
                ride_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                kj INTEGER NOT NULL,
                duration_sec INTEGER NOT NULL,
                watts REAL NOT NULL,
                date_time TEXT,
                PRIMARY KEY (ride_id, kj, duration_sec),
                FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
            )
        ''')
        _add_column(c, 'users', 'crank_mm', 'REAL DEFAULT 172.5')
        _add_column(c, 'planned_sessions', 'blocks', 'TEXT')

        # One copy of each Strava activity per rider (NULLs, i.e. uploads, are exempt)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_rides_user_strava
                     ON rides(user_id, strava_activity_id)""")

        # rides.file_content (the original ride file) sits in the middle of each row, so a
        # query that needs a column stored after it has to read through the whole file. These
        # covering indexes answer the per-rerun queries (TSS history, "needs update" count,
        # period filters) from the index alone.
        c.execute("""CREATE INDEX IF NOT EXISTS idx_rides_user_date
                     ON rides(user_id, date_time, tss, metrics_version)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_power_records_ride ON power_records(ride_id)")

        # Original ride files live in their own table, compressed, so ride rows stay small.
        c.execute('''
            CREATE TABLE IF NOT EXISTS ride_files (
                ride_id INTEGER PRIMARY KEY,
                codec TEXT NOT NULL,         -- 'zlib' or 'raw'
                content BLOB NOT NULL,
                FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS strava_auth (
                user_id INTEGER PRIMARY KEY,
                athlete_id INTEGER,
                athlete_name TEXT,
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS wahoo_auth (
                user_id INTEGER PRIMARY KEY,
                wahoo_user_id INTEGER,
                wahoo_name TEXT,
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS app_credentials (
                provider TEXT PRIMARY KEY,           -- 'strava' | 'wahoo' | 'polar'
                client_id TEXT NOT NULL,
                client_secret TEXT NOT NULL,
                redirect_uri TEXT,
                timezone TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS smtp_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),     -- one mail account for the whole app
                host TEXT NOT NULL,
                port INTEGER NOT NULL DEFAULT 587,
                security TEXT NOT NULL DEFAULT 'starttls',  -- 'starttls' | 'ssl' | 'none'
                username TEXT,
                password TEXT,
                sender TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS weekly_email (
                user_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                recipient TEXT NOT NULL,
                last_week TEXT,                  -- Monday (ISO date) of the last week sent or being sent
                last_status TEXT,
                last_attempt TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS polar_auth (
                user_id INTEGER PRIMARY KEY,
                polar_user_id TEXT,
                access_token TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS wahoo_sent (
                session_id INTEGER PRIMARY KEY,      -- planned_sessions.id
                plan_id INTEGER NOT NULL,
                workout_id INTEGER NOT NULL,
                starts TEXT NOT NULL,                -- UTC ISO sent to Wahoo
                sent_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES planned_sessions(id) ON DELETE CASCADE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
        ''')


_init_lock = threading.Lock()
_initialised = set()


def init_db(force=False):
    """
    Creates the tables, applies the numbered migrations and switches on WAL. Streamlit runs the script (and so
    this call) on every interaction, so after the first success per database file it returns at once;
    force=True does the work again.
    """
    with _init_lock:
        if not force and core.DB_FILE in _initialised and os.path.exists(core.DB_FILE):
            return
        if _init_db():
            _initialised.add(core.DB_FILE)           # not when the file migration was postponed: retry next run


def _init_db():
    """Initializes the database with users, rides, and power_records tables."""
    _enable_wal()
    with connect() as conn:
        c = conn.cursor()
    
        # Users Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                ftp INTEGER DEFAULT 200,
                lthr INTEGER DEFAULT 170,
                weight REAL DEFAULT 70.0,
                height REAL DEFAULT 175.0,
                cp INTEGER DEFAULT 0,
                w_prime REAL DEFAULT 20000
            )
        ''')

        # Rides Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS rides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                file_hash TEXT,
                file_content BLOB,
                date_time TEXT,
                distance_km REAL,
                elevation_m REAL,
                avg_speed_kph REAL,
                norm_power INTEGER,
                avg_power INTEGER,
                tss INTEGER,
                if_factor REAL,
                avg_hr INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        # Power Records Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS power_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ride_id INTEGER,
                duration_sec INTEGER,
                watts REAL,
                date_time TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
            )
        ''')
    
    
    # Run migration to ensure existing databases get the new columns
    migrate_db()
    run_migrations()
    encrypt_plaintext_secrets()
    return _move_ride_files() is not None
