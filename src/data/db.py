import os
import sqlite3
import threading
import warnings
import pandas as pd
import hashlib
import re
import secrets
import time
from contextlib import contextmanager

from src.analysis.dates import parse_utc

# BIKETOOLS_DB lets you keep the database somewhere else, e.g. outside a OneDrive/Dropbox folder:
# those sync tools (and Windows drives mounted in WSL) lock files unpredictably, which SQLite hates.
DB_FILE = os.environ.get("BIKETOOLS_DB", "bikethools.db")
BUSY_TIMEOUT_S = 60          # wait this long for another connection's lock instead of failing at once


def _connect():
    """
    Every connection goes through here. SQLite only enforces foreign keys (and so
    ON DELETE CASCADE) when asked to, per connection; without this, deleting a user
    left their rides, records and histograms behind. synchronous=NORMAL is the safe, faster
    setting for WAL (a crash can lose the last commit, never corrupt the file).
    """
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def connect(row_factory=None):
    """
    `with connect() as conn:` commits when the block ends, rolls back when it raises, and always closes,
    so an early return or an exception cannot leak a connection or leave a half-written change.
    """
    conn = _connect()
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _add_column(c, table, column, definition):
    """ALTER TABLE ... ADD COLUMN when the column is missing. Any other error is a real one and propagates."""
    existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _enable_wal():
    """
    Write-ahead logging: readers no longer block the writer (the background weekly e-mail writes while the
    page reads). It needs shared memory, which network and some synced/mounted drives lack: when SQLite
    stays on its default journal there, that is fine, the app just works as before.
    """
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
    try:
        return conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


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


MIGRATIONS = [
    (1, _migration_per_user_file_hash),
]


def schema_version():
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
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
        conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S, isolation_level=None)
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
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
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
        if not force and DB_FILE in _initialised and os.path.exists(DB_FILE):
            return
        if _init_db():
            _initialised.add(DB_FILE)           # not when the file migration was postponed: retry next run


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
    return _move_ride_files() is not None

DEFAULT_FTP, DEFAULT_LTHR = 200, 170
DEFAULT_W_PRIME = 20000          # joules, when the rider has not given theirs

def save_user(name, ftp=None, lthr=None, weight=70.0, height=175.0, w_prime=None):
    """
    ftp / lthr None: stored as the defaults but flagged to be estimated from the rider's rides.
    w_prime (joules) None: DEFAULT_W_PRIME; a ride that uses more than it raises the figure (see the Power section).
    """
    try:
        with connect() as conn:
            conn.execute("""INSERT INTO users (name, ftp, lthr, weight, height, ftp_auto, lthr_auto, w_prime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (name, DEFAULT_FTP if ftp is None else ftp, DEFAULT_LTHR if lthr is None else lthr,
                          weight, height, int(ftp is None), int(lthr is None),
                          DEFAULT_W_PRIME if not w_prime else float(w_prime)))
        return True, "User saved successfully!"
    except sqlite3.IntegrityError:
        return False, "User already exists."
    except Exception as e:
        return False, f"Database Error: {e}"

def set_user_thresholds(user_name, ftp=None, lthr=None, auto_ftp=None, auto_lthr=None):
    """
    Sets FTP and/or LTHR by hand (which stops them following the data), or switches them back to
    "estimate from my rides" with auto_ftp / auto_lthr=True (the value stays until the next estimate).
    """
    with connect() as conn:
        if ftp is not None:
            conn.execute("UPDATE users SET ftp = ?, ftp_auto = 0 WHERE name = ?", (int(round(ftp)), user_name))
        if lthr is not None:
            conn.execute("UPDATE users SET lthr = ?, lthr_auto = 0 WHERE name = ?", (int(round(lthr)), user_name))
        if auto_ftp:
            conn.execute("UPDATE users SET ftp_auto = 1 WHERE name = ?", (user_name,))
        if auto_lthr:
            conn.execute("UPDATE users SET lthr_auto = 1 WHERE name = ?", (user_name,))

def set_user_w_prime(user_name, w_prime):
    """Sets the rider's W' (joules)."""
    with connect() as conn:
        conn.execute("UPDATE users SET w_prime = ? WHERE name = ?", (float(w_prime), user_name))

def set_auto_thresholds(user_name, ftp=None, lthr=None):
    """Stores an estimate without turning the 'estimated' flag off."""
    with connect() as conn:
        if ftp is not None:
            conn.execute("UPDATE users SET ftp = ? WHERE name = ?", (int(round(ftp)), user_name))
        if lthr is not None:
            conn.execute("UPDATE users SET lthr = ? WHERE name = ?", (int(round(lthr)), user_name))

def get_hr20(user_name, start=None):
    """Best-20-minute mean HR of each ride (> 0 only), newest rides last: list of (date, bpm)."""
    user = get_user_data(user_name)
    if not user:
        return []
    where, params = _date_filter(start, None)
    with connect() as conn:
        rows = conn.execute(f"""SELECT date_time, hr20_bpm FROM rides r WHERE user_id = ? AND hr20_bpm > 0{where}
                                ORDER BY date_time""", [user['id'], *params]).fetchall()
    return [(r[0], r[1]) for r in rows]

def get_rides_without_hr20(user_name, limit=25):
    """(id, filename) of the newest rides whose best-20-min HR was never computed."""
    user = get_user_data(user_name)
    if not user:
        return []
    with connect() as conn:
        rows = conn.execute("""SELECT id, filename FROM rides WHERE user_id = ? AND hr20_bpm IS NULL
                               ORDER BY date_time DESC LIMIT ?""", (user['id'], limit)).fetchall()
    return [(r[0], r[1]) for r in rows]

def set_ride_hr20(ride_id, bpm):
    with connect() as conn:
        conn.execute("UPDATE rides SET hr20_bpm = ? WHERE id = ?", (float(bpm or 0.0), ride_id))

def update_user_ftp(user_name, ftp):
    """Sets the FTP only. Rides already saved keep the FTP they were scored with."""
    with connect() as conn:
        conn.execute("UPDATE users SET ftp = ?, ftp_auto = 0 WHERE name = ?", (int(round(ftp)), user_name))

def update_user_physics(user_name, cp, w_prime, update_ftp=False):
    """
    Updates the user's physiological metrics.
    Optionally updates FTP to match CP.
    """
    with connect() as conn:
        c = conn.cursor()
    
        if update_ftp:
            c.execute('''
                UPDATE users 
                SET cp = ?, w_prime = ?, ftp = ?, ftp_auto = 0 
                WHERE name = ?
            ''', (cp, w_prime, cp, user_name))
        else:
            c.execute('''
                UPDATE users 
                SET cp = ?, w_prime = ? 
                WHERE name = ?
            ''', (cp, w_prime, user_name))
        

BACKUP_KEEP = 3

def backup_db(reason="manual", keep=None):
    """
    Consistent copy of the database to backups/ next to it (SQLite's backup API, safe
    while the app is running). Keeps the newest `keep` backups. Returns the path, or
    None if there is no database yet or the copy failed; callers go on either way,
    since a failed backup should not block the work it was protecting.
    """
    import os, glob
    if not os.path.exists(DB_FILE):
        return None
    folder = os.path.join(os.path.dirname(os.path.abspath(DB_FILE)), "backups")
    stem = os.path.splitext(os.path.basename(DB_FILE))[0]
    dest = os.path.join(folder, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.db")
    try:
        os.makedirs(folder, exist_ok=True)
        src, dst = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S), sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except (OSError, sqlite3.Error):
        if os.path.exists(dest):
            os.remove(dest)
        return None
    old = sorted(glob.glob(os.path.join(folder, f"{stem}-*.db")))[:-(keep or BACKUP_KEEP)]
    for path in old:
        try:
            os.remove(path)
        except OSError:
            pass
    return dest

def data_stamp():
    """
    Changes whenever the database file is written. UI caches key on it, so cached reads
    stay valid between interactions and refresh themselves after any save or update.
    """
    import os
    stamp = [DB_FILE]
    for path in (DB_FILE, DB_FILE + "-wal"):
        try:
            st = os.stat(path)
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)

def get_all_users():
    with connect() as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM users")
        users = [row[0] for row in c.fetchall()]
    return users

def get_user_data(name):
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = c.fetchone()
    if row:
        return dict(row)
    return None

def delete_user(name):
    with connect() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE name = ?", (name,))

def strava_id_from_filename(filename):
    """'strava_123.json' -> 123 (rides imported through the Strava API)."""
    m = re.fullmatch(r"strava_(\d+)\.json", (filename or "").lower())
    return int(m.group(1)) if m else None

def _iso_timestamp(ts):
    """Ride start time as an ISO-8601 string (UTC when timezone-aware)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()

# Bump when the per-ride calculations change; older rides get offered a recompute
# 2: joint power/HR/cadence histograms, HR recovery
# 3: gap-aware resampling (smart recording), enhanced FIT channels, moving fallback
# 4: power after 1/2/3 MJ of work (durability); zone times on moving time only
# 5: decoupling ignores coasting (0 W) seconds
# 6: impossible sensor readings (power spikes, HR dropouts) removed before scoring
METRICS_VERSION = 6

def build_ride_metadata(df, dist_m, ele_m, speed_kph, np_val, tss, if_val, filename=None,
                        metrics=None):
    """
    Builds the metadata dict that save_ride() expects. Keeping both in this
    module guarantees the keys match (they did not before: date, distance
    and elevation were silently stored as NULL / 0).

    metrics: optional dict from pipeline.metrics_for_storage(summary). Rides
    saved without it are marked as needing a recompute.
    """
    def safe_mean(col):
        if col not in df.columns or df[col].empty:
            return 0
        m = df[col].mean()
        return int(m) if pd.notna(m) else 0

    return {
        'date': _iso_timestamp(df['time'].iloc[0]),
        'dist': dist_m / 1000,
        'ele': ele_m,
        'speed': speed_kph,
        'np': int(np_val) if pd.notna(np_val) else 0,
        'avg_p': safe_mean('power'),
        'tss': int(round(tss)) if pd.notna(tss) else 0,
        'if': if_val,
        'avg_hr': safe_mean('hr'),
        'strava_activity_id': strava_id_from_filename(filename),
        **_metrics_fields(metrics),
    }

def _pack(content):
    """(codec, blob) for storage: zlib when it helps, the raw bytes otherwise."""
    import zlib
    packed = zlib.compress(content, 6)
    return ("zlib", packed) if len(packed) < len(content) else ("raw", content)

def _unpack(codec, blob):
    import zlib
    return zlib.decompress(blob) if codec == "zlib" else bytes(blob)

_migration_lock = threading.Lock()

def _move_ride_files(batch=10):
    """
    Runs the file migration, one thread at a time (Streamlit may run the script for several
    browser sessions at once, and init_db() runs on every one). If the database is locked by
    another process or a slow sync, the migration is postponed to the next run instead of
    stopping the app from loading.
    """
    with _migration_lock:
        try:
            return _do_move_ride_files(batch)
        except sqlite3.OperationalError as e:
            warnings.warn(f"Ride-file migration postponed ({e}); it will be retried on the next run.")
            return None

def _do_move_ride_files(batch):
    """
    One-time migration: rides saved before ride_files existed keep their original file in
    rides.file_content. Compress them into ride_files and empty that column (the column
    stays; NULL costs nothing and old code reading it just finds nothing to recompute).
    Resumable and idempotent: a backup is taken first, then it works in small batches.
    """
    conn = _connect()
    try:
        pending = conn.execute("SELECT 1 FROM rides WHERE file_content IS NOT NULL LIMIT 1").fetchone()
        if not pending:
            return 0
        conn.close()
        backup_db("before-file-migration")
        conn = _connect()
        moved = 0
        while True:
            rows = conn.execute("SELECT id, file_content FROM rides WHERE file_content IS NOT NULL LIMIT ?",
                                (batch,)).fetchall()
            if not rows:
                break
            for ride_id, content in rows:
                codec, blob = _pack(bytes(content))
                conn.execute("INSERT OR REPLACE INTO ride_files (ride_id, codec, content) VALUES (?, ?, ?)",
                             (ride_id, codec, blob))
                conn.execute("UPDATE rides SET file_content = NULL WHERE id = ?", (ride_id,))
            conn.commit()
            moved += len(rows)
        conn.close()
        try:                                   # give the freed space back to the disk
            vac = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
            vac.execute("VACUUM")
            vac.close()
        except sqlite3.Error:
            pass
        return moved
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass

def _opt_float(x):
    return None if x is None or pd.isna(x) else float(x)

def _opt_int(x):
    return None if x is None or pd.isna(x) else int(round(x))

def _metrics_fields(metrics):
    if not metrics:
        return {'metrics_version': 0}
    return {
        'duration_s': _opt_int(metrics.get('duration_s')),
        'moving_s': _opt_int(metrics.get('moving_time_s')),
        'kj': _opt_float(metrics.get('kj')),
        'ftp_at_ride': _opt_int(metrics.get('ftp_at_ride')),
        'lthr_at_ride': _opt_int(metrics.get('lthr_at_ride')),
        'vi': _opt_float(metrics.get('vi')),
        'ef': _opt_float(metrics.get('ef')),
        'decoupling_pct': _opt_float(metrics.get('decoupling_pct')),
        'tss_source': metrics.get('tss_source'),
        'zones': metrics.get('zones') or {},
        'hist': metrics.get('hist') or {},
        'fatigued': metrics.get('fatigued') or {},
        'hrr60_bpm': _opt_float(metrics.get('hrr60')),
        'hr20_bpm': _opt_float(metrics.get('hr20')),
        'hrr_efforts': _opt_int(metrics.get('hrr_efforts')),
        'metrics_version': METRICS_VERSION,
    }

_METRIC_COLUMNS = ['duration_s', 'moving_s', 'kj', 'ftp_at_ride', 'lthr_at_ride', 'vi', 'ef',
                   'decoupling_pct', 'tss_source', 'hrr60_bpm', 'hrr_efforts', 'hr20_bpm', 'metrics_version']

def _insert_children(c, user_id, ride_id, metadata, power_curve_dict):
    """Power records and zone times for one ride (caller deletes old ones)."""
    if power_curve_dict:
        c.executemany('''
            INSERT INTO power_records (user_id, ride_id, duration_sec, watts, date_time)
            VALUES (?, ?, ?, ?, ?)
        ''', [(user_id, ride_id, int(d), float(w), metadata.get('date'))
              for d, w in power_curve_dict.items() if w is not None and pd.notna(w)])
    zones = metadata.get('zones') or {}
    c.executemany("INSERT INTO ride_zones (ride_id, kind, zone, seconds) VALUES (?, ?, ?, ?)",
                  [(ride_id, kind, str(z), int(sec))
                   for kind, zz in zones.items() for z, sec in zz.items()])
    fatigued = metadata.get('fatigued') or {}
    c.executemany("""INSERT INTO power_records_kj (ride_id, user_id, kj, duration_sec, watts, date_time)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  [(ride_id, user_id, int(kj), int(d), float(w), metadata.get('date'))
                   for kj, curve in fatigued.items() for d, w in curve.items()
                   if w is not None and pd.notna(w)])
    hist = metadata.get('hist') or {}
    c.executemany("INSERT INTO ride_hist (ride_id, pair, x, y, seconds) VALUES (?, ?, ?, ?, ?)",
                  [(ride_id, pair, int(x), int(y), int(sec))
                   for pair, cells in hist.items() for (x, y), sec in cells.items()])

def save_ride(user_name, filename, file_content, metadata, power_curve_dict):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."

    file_hash = hashlib.md5(file_content).hexdigest()
    
    try:
        with connect() as conn:
            c = conn.cursor()
            # 1. Insert Ride Metadata
            cols = ['user_id', 'filename', 'file_hash', 'date_time',
                    'distance_km', 'elevation_m', 'avg_speed_kph', 'norm_power', 'avg_power',
                    'tss', 'if_factor', 'avg_hr', 'strava_activity_id'] + _METRIC_COLUMNS
            vals = [user['id'], filename, file_hash, metadata.get('date'),
                    metadata.get('dist', 0), metadata.get('ele', 0), metadata.get('speed', 0),
                    metadata.get('np', 0), metadata.get('avg_p', 0), metadata.get('tss', 0),
                    metadata.get('if', 0), metadata.get('avg_hr', 0),
                    metadata.get('strava_activity_id')] + [
                    metadata.get(k, 0 if k == 'metrics_version' else None) for k in _METRIC_COLUMNS]
            c.execute(f"INSERT INTO rides ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
            ride_id = c.lastrowid
            codec, blob = _pack(file_content)
            c.execute("INSERT INTO ride_files (ride_id, codec, content) VALUES (?, ?, ?)", (ride_id, codec, blob))

            # 2. Power records (curve) and time in zones
            _insert_children(c, user['id'], ride_id, metadata, power_curve_dict)
            
        return True, "Ride saved successfully!"
        
    except sqlite3.IntegrityError:
        return False, "Ride already exists (duplicate file)."
        
    except Exception as e:
        return False, f"Database Error: {e}"
        

def get_user_rides(user_name):
    user = get_user_data(user_name)
    if not user:
        return []
    
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''
            SELECT id, filename, date_time, distance_km, elevation_m, norm_power, tss,
                   moving_s, tss_source, vi, ef, decoupling_pct, kj, metrics_version
            FROM rides 
            WHERE user_id = ? 
            ORDER BY date_time DESC
        ''', (user['id'],))
        rows = c.fetchall()
    return [dict(row) for row in rows]

def get_ride_file(ride_id):
    """(filename, original file bytes) of a stored ride, or (None, None)."""
    with connect() as conn:
        row = conn.execute("""SELECT r.filename, f.codec, f.content, r.file_content
                              FROM rides r LEFT JOIN ride_files f ON f.ride_id = r.id
                              WHERE r.id = ?""", (ride_id,)).fetchone()
    if not row:
        return None, None
    filename, codec, blob, legacy = row
    if blob is not None:
        return filename, _unpack(codec, blob)
    return filename, (bytes(legacy) if legacy is not None else None)

def get_user_best_power(user_name):
    user = get_user_data(user_name)
    if not user:
        return {}
    
    with connect() as conn:
        c = conn.cursor()
    
        # Query: Find the maximum watts for each duration across ALL rides for this user
        c.execute('''
            SELECT duration_sec, MAX(watts) 
            FROM power_records 
            WHERE user_id = ? 
            GROUP BY duration_sec
            ORDER BY duration_sec ASC
        ''', (user['id'],))
    
        rows = c.fetchall()
    
    # Convert to dict {duration: watts}
    return {row[0]: row[1] for row in rows}

def get_user_tss_history(user_name):
    """
    Fetches date and TSS for all rides, mimicking get_user_rides logic.
    Does NOT filter by TSS value in SQL to ensure we get raw data first.
    """
    user = get_user_data(user_name)
    if not user:
        return []
    
    with connect() as conn:
        c = conn.cursor()
    
        # Just select the columns we need
        c.execute('''
            SELECT date_time, tss 
            FROM rides 
            WHERE user_id = ? 
            ORDER BY date_time ASC
        ''', (user['id'],))
    
        rows = c.fetchall()
    
    # Return raw list of dicts
    return [{'date': row[0], 'tss': row[1]} for row in rows]


# --- Recompute history ------------------------------------------------------
# Rides whose stored metrics are older than METRICS_VERSION (or that were
# saved without a date by the pre-Phase-0 bug) are rebuilt from the original
# file kept in ride_files (get_ride_file). See src/data/history.py.

_NEEDS_UPDATE_SQL = "(metrics_version IS NULL OR metrics_version < ? OR date_time IS NULL OR date_time = '')"

def count_rides_needing_update(user_name):
    user = get_user_data(user_name)
    if not user:
        return 0
    with connect() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM rides WHERE user_id = ? AND {_NEEDS_UPDATE_SQL}",
                         (user['id'], METRICS_VERSION)).fetchone()[0]
    return n

def get_rides_for_recompute(user_name, all_rides=False):
    """Rows needed to rebuild a ride's metrics (all rides, or only outdated ones)."""
    user = get_user_data(user_name)
    if not user:
        return []
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        where = "user_id = ?" + ("" if all_rides else f" AND {_NEEDS_UPDATE_SQL}")
        params = (user['id'],) if all_rides else (user['id'], METRICS_VERSION)
        rows = conn.execute(f"""SELECT id, filename, date_time, norm_power, if_factor,
                                       ftp_at_ride, lthr_at_ride
                                FROM rides WHERE {where} ORDER BY date_time""", params).fetchall()
    return [dict(r) for r in rows]

def update_ride_metrics(ride_id, metadata, power_curve_dict):
    """Overwrites a ride's computed fields, power records and zone times."""
    with connect() as conn:
        c = conn.cursor()
        user_id = c.execute("SELECT user_id FROM rides WHERE id = ?", (ride_id,)).fetchone()[0]
        cols = ['date_time', 'distance_km', 'elevation_m', 'avg_speed_kph', 'norm_power',
                'avg_power', 'tss', 'if_factor', 'avg_hr'] + _METRIC_COLUMNS
        vals = [metadata.get('date'), metadata.get('dist', 0), metadata.get('ele', 0),
                metadata.get('speed', 0), metadata.get('np', 0), metadata.get('avg_p', 0),
                metadata.get('tss', 0), metadata.get('if', 0), metadata.get('avg_hr', 0)] + [
                metadata.get(k) for k in _METRIC_COLUMNS]
        c.execute(f"UPDATE rides SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?", vals + [ride_id])
        c.execute("DELETE FROM power_records WHERE ride_id = ?", (ride_id,))
        c.execute("DELETE FROM ride_zones WHERE ride_id = ?", (ride_id,))
        c.execute("DELETE FROM ride_hist WHERE ride_id = ?", (ride_id,))
        c.execute("DELETE FROM power_records_kj WHERE ride_id = ?", (ride_id,))
        _insert_children(c, user_id, ride_id, metadata, power_curve_dict)

def get_ride_zones(ride_ids, kind):
    """{zone: total seconds} summed over the given rides."""
    if not ride_ids:
        return {}
    with connect() as conn:
        q = ",".join("?" * len(ride_ids))
        rows = conn.execute(f"""SELECT zone, SUM(seconds) FROM ride_zones
                                WHERE kind = ? AND ride_id IN ({q}) GROUP BY zone ORDER BY zone""",
                            (kind, *ride_ids)).fetchall()
    return {z: s for z, s in rows}


# --- Strava authorization --------------------------------------------------
# Tokens are stored per rider profile in the local database (plain text, like
# the rest of the local data). Strava may rotate the refresh token on every
# refresh, so save_strava_auth() must be called whenever it changes.

def save_strava_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO strava_auth (user_id, athlete_id, athlete_name, access_token, refresh_token, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                athlete_id = COALESCE(excluded.athlete_id, strava_auth.athlete_id),
                athlete_name = COALESCE(excluded.athlete_name, strava_auth.athlete_name),
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at
        ''', (user['id'], auth.get('athlete_id'), auth.get('athlete_name'),
              auth['access_token'], auth['refresh_token'], int(auth['expires_at'])))
    return True

def get_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM strava_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return dict(row) if row else None

def delete_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM strava_auth WHERE user_id = ?", (user['id'],))

# --- Wahoo authorization (same storage rules; Wahoo refresh tokens are single-use) ---

def save_wahoo_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO wahoo_auth (user_id, wahoo_user_id, wahoo_name, access_token, refresh_token, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                wahoo_user_id = COALESCE(excluded.wahoo_user_id, wahoo_auth.wahoo_user_id),
                wahoo_name = COALESCE(excluded.wahoo_name, wahoo_auth.wahoo_name),
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at
        ''', (user['id'], auth.get('wahoo_user_id'), auth.get('wahoo_name'),
              auth['access_token'], auth['refresh_token'], int(auth['expires_at'])))
    return True

def get_wahoo_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM wahoo_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return dict(row) if row else None

def delete_wahoo_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM wahoo_auth WHERE user_id = ?", (user['id'],))

# --- API app credentials entered in the app (instead of .streamlit/secrets.toml) ---

def save_app_credentials(provider, client_id, client_secret, redirect_uri=None, timezone=None):
    with connect() as conn:
        conn.execute('''
            INSERT INTO app_credentials (provider, client_id, client_secret, redirect_uri, timezone)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET client_id = excluded.client_id, client_secret = excluded.client_secret,
                redirect_uri = excluded.redirect_uri, timezone = excluded.timezone
        ''', (provider, client_id.strip(), client_secret.strip(), redirect_uri, timezone))

def get_app_credentials(provider):
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM app_credentials WHERE provider = ?", (provider,)).fetchone()
    return dict(row) if row else None

def delete_app_credentials(provider):
    with connect() as conn:
        conn.execute("DELETE FROM app_credentials WHERE provider = ?", (provider,))

# --- Polar authorization (tokens do not expire; stored like the other providers') ---

def save_polar_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO polar_auth (user_id, polar_user_id, access_token) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET polar_user_id = excluded.polar_user_id, access_token = excluded.access_token
        ''', (user['id'], auth.get('polar_user_id'), auth['access_token']))
    return True

def get_polar_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM polar_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return dict(row) if row else None

def delete_polar_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM polar_auth WHERE user_id = ?", (user['id'],))

def polar_id_from_filename(filename):
    """'polar_aB3x9.fit' -> 'aB3x9' (rides imported through Polar AccessLink)."""
    m = re.fullmatch(r"polar_([A-Za-z0-9]+)\.fit", filename or "")
    return m.group(1) if m else None

def get_imported_polar_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT filename FROM rides WHERE user_id = ? AND filename LIKE 'polar\\_%.fit' ESCAPE '\\'",
                            (user['id'],)).fetchall()
    return {i for i in (polar_id_from_filename(r[0]) for r in rows) if i is not None}

def get_wahoo_sent(user_name):
    """{planned session id: {plan_id, workout_id, starts, sent_at}} for sessions sent to Wahoo."""
    user = get_user_data(user_name)
    if not user:
        return {}
    with connect() as conn:
        rows = conn.execute("""SELECT w.session_id, w.plan_id, w.workout_id, w.starts, w.sent_at
                               FROM wahoo_sent w JOIN planned_sessions p ON p.id = w.session_id
                               WHERE p.user_id = ?""", (user['id'],)).fetchall()
    return {r[0]: {'plan_id': r[1], 'workout_id': r[2], 'starts': r[3], 'sent_at': r[4]} for r in rows}

def save_wahoo_sent(session_id, plan_id, workout_id, starts, sent_at):
    with connect() as conn:
        conn.execute("""INSERT INTO wahoo_sent (session_id, plan_id, workout_id, starts, sent_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET plan_id = excluded.plan_id,
                            workout_id = excluded.workout_id, starts = excluded.starts, sent_at = excluded.sent_at""",
                     (int(session_id), int(plan_id), int(workout_id), starts, sent_at))

def delete_wahoo_sent(session_id):
    with connect() as conn:
        conn.execute("DELETE FROM wahoo_sent WHERE session_id = ?", (int(session_id),))

def wahoo_id_from_filename(filename):
    """'wahoo_123.fit' -> 123 (rides imported through the Wahoo Cloud API)."""
    m = re.fullmatch(r"wahoo_(\d+)\.fit", (filename or "").lower())
    return int(m.group(1)) if m else None

def get_imported_wahoo_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT filename FROM rides WHERE user_id = ? AND filename LIKE 'wahoo\\_%.fit' ESCAPE '\\'",
                            (user['id'],)).fetchall()
    return {i for i in (wahoo_id_from_filename(r[0]) for r in rows) if i is not None}

def latest_wahoo_ride_date(user_name):
    """Start (naive UTC) of the newest ride imported from Wahoo, or None."""
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        row = conn.execute("SELECT MAX(date_time) FROM rides WHERE user_id = ? AND date_time IS NOT NULL "
                           "AND filename LIKE 'wahoo\\_%.fit' ESCAPE '\\'", (user['id'],)).fetchone()
    if not row or not row[0]:
        return None
    ts = pd.to_datetime(row[0], errors='coerce', utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)


OAUTH_STATE_TTL_S = 15 * 60

def create_oauth_state(user_name):
    """
    One-time CSRF token for the OAuth round trip. Stored in the DB rather
    than st.session_state because Strava redirects back into a new session.
    """
    user = get_user_data(user_name)
    if not user:
        return None
    state = secrets.token_urlsafe(24)
    now = int(time.time())
    with connect() as conn:
        conn.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - OAUTH_STATE_TTL_S,))
        conn.execute("INSERT INTO oauth_states (state, user_id, created_at) VALUES (?, ?, ?)",
                     (state, user['id'], now))
    return state

def consume_oauth_state(state):
    """Returns the user name the state was issued for (once), else None."""
    now = int(time.time())
    with connect() as conn:
        row = conn.execute('''SELECT u.name, s.created_at FROM oauth_states s
                              JOIN users u ON u.id = s.user_id WHERE s.state = ?''', (state,)).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
    if not row or now - row[1] > OAUTH_STATE_TTL_S:
        return None
    return row[0]

def get_imported_strava_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT strava_activity_id FROM rides WHERE user_id = ? "
                            "AND strava_activity_id IS NOT NULL", (user['id'],)).fetchall()
    return {r[0] for r in rows}

def latest_strava_ride_date(user_name):
    """Start (naive UTC) of the newest ride imported from Strava, or None."""
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        row = conn.execute("SELECT MAX(date_time) FROM rides WHERE user_id = ? AND strava_activity_id IS NOT NULL "
                           "AND date_time IS NOT NULL", (user['id'],)).fetchone()
    if not row or not row[0]:
        return None
    ts = pd.to_datetime(row[0], errors='coerce', utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)

def get_ride_start_times(user_name):
    """Start times (UTC, naive) of all dated rides, for cross-source dedup."""
    user = get_user_data(user_name)
    if not user:
        return []
    with connect() as conn:
        rows = conn.execute("SELECT date_time FROM rides WHERE user_id = ? AND date_time IS NOT NULL",
                            (user['id'],)).fetchall()
    out = []
    for (d,) in rows:
        ts = pd.to_datetime(d, errors='coerce', utc=True)
        if pd.notna(ts):
            out.append(ts.tz_convert(None))
    return out


# --- Deeper analytics queries ------------------------------------------------

def _parse_dates(series):
    """Stored ISO strings (with or without offset) -> naive UTC datetimes."""
    return parse_utc(series)

def _date_filter(start, end):
    sql, params = "", []
    if start is not None:
        sql += " AND r.date_time >= ?"
        params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        sql += " AND r.date_time < ?"
        params.append((pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return sql, params

def get_hist(user_name, pair, start=None, end=None):
    """
    Per-ride histogram cells for a period: DataFrame(ride_id, date, x, y, seconds).
    start/end are inclusive dates (None = open-ended).
    """
    cols = ['ride_id', 'date', 'x', 'y', 'seconds']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT h.ride_id, r.date_time AS date, h.x, h.y, h.seconds
                                   FROM ride_hist h JOIN rides r ON r.id = h.ride_id
                                   WHERE r.user_id = ? AND h.pair = ?{where}""",
                               conn, params=[user['id'], pair, *params])
    df['date'] = _parse_dates(df['date'])
    return df

def get_ride_stats(user_name, start=None, end=None):
    """Per-ride stored metrics for trend charts (one row per dated ride)."""
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame()
    where, params = _date_filter(start, end)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT r.id AS ride_id, r.date_time AS date, r.duration_s, r.moving_s,
                                          r.norm_power, r.avg_hr, r.vi, r.ef, r.decoupling_pct,
                                          r.hrr60_bpm, r.hrr_efforts, r.tss, r.tss_source,
                                          r.distance_km, r.elevation_m, r.kj
                                   FROM rides r WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}
                                   ORDER BY r.date_time""", conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'])
    return df

def update_user_crank(user_name, crank_mm):
    with connect() as conn:
        conn.execute("UPDATE users SET crank_mm = ? WHERE name = ?", (float(crank_mm), user_name))

def period_coverage(user_name, start=None, end=None):
    """Why analytics may be empty: rides in the period and what they contain."""
    out = {'rides': 0, 'needs_update': 0, 'p_hr': 0, 'p_cad': 0}
    user = get_user_data(user_name)
    if not user:
        return out
    where, params = _date_filter(start, end)
    with connect() as conn:
        out['rides'], out['needs_update'] = conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(CASE WHEN r.metrics_version IS NULL
                                                   OR r.metrics_version < ? THEN 1 ELSE 0 END), 0)
                FROM rides r WHERE r.user_id = ?{where}""",
            [METRICS_VERSION, user['id'], *params]).fetchone()
        for pair in ('p_hr', 'p_cad'):
            out[pair] = conn.execute(
                f"""SELECT COUNT(DISTINCT h.ride_id) FROM ride_hist h JOIN rides r ON r.id = h.ride_id
                    WHERE r.user_id = ? AND h.pair = ?{where}""", [user['id'], pair, *params]).fetchone()[0]
    return out

def get_power_records(user_name, start=None, end=None):
    """Per-ride mean-maximal power: DataFrame(duration, watts, date, ride_id)."""
    cols = ['duration', 'watts', 'date', 'ride_id']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT p.duration_sec AS duration, p.watts, r.date_time AS date, r.id AS ride_id
                                   FROM power_records p JOIN rides r ON r.id = p.ride_id
                                   WHERE r.user_id = ? AND r.date_time IS NOT NULL
                                     AND p.watts IS NOT NULL{where}""",
                               conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_kj_records(user_name, start=None, end=None):
    """Power after X kJ per ride: DataFrame(kj, duration, watts, date, ride_id)."""
    cols = ['kj', 'duration', 'watts', 'date', 'ride_id']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT k.kj, k.duration_sec AS duration, k.watts, r.date_time AS date,
                                          r.id AS ride_id
                                   FROM power_records_kj k JOIN rides r ON r.id = k.ride_id
                                   WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}""",
                               conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_zone_rows(user_name, kind, start=None, end=None):
    """Time in zones per ride: DataFrame(ride_id, date, zone, seconds); kind 'power' or 'hr'."""
    cols = ['ride_id', 'date', 'zone', 'seconds']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT z.ride_id, r.date_time AS date, z.zone, z.seconds
                                   FROM ride_zones z JOIN rides r ON r.id = z.ride_id
                                   WHERE r.user_id = ? AND z.kind = ? AND r.date_time IS NOT NULL{where}""",
                               conn, params=[user['id'], kind, *params])
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_planned_sessions(user_name, after=None, structured=None):
    """
    Planned sessions as DataFrame(id, date, tss, note, blocks), optionally only after a date.
    structured: None = all, False = quick TSS entries only, True = interval-built sessions only.
    blocks is the JSON text of the interval blocks (None for quick entries).
    """
    cols = ['id', 'date', 'tss', 'note', 'blocks']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    sql = "SELECT id, date, tss, note, blocks FROM planned_sessions WHERE user_id = ?"
    params = [user['id']]
    if after is not None:
        sql += " AND date > ?"
        params.append(pd.Timestamp(after).strftime("%Y-%m-%d"))
    if structured is not None:
        sql += " AND blocks IS NOT NULL" if structured else " AND blocks IS NULL"
    with connect() as conn:
        df = pd.read_sql_query(sql + " ORDER BY date, id", conn, params=params)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[cols]

def add_structured_session(user_name, date, blocks, tss, note=None):
    """Stores one interval-built session (blocks: list of dicts, scored to `tss`). Returns its id."""
    import json
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        cur = conn.execute("INSERT INTO planned_sessions (user_id, date, tss, note, blocks) VALUES (?, ?, ?, ?, ?)",
                           (user['id'], pd.Timestamp(date).strftime("%Y-%m-%d"), float(tss),
                            note or None, json.dumps(blocks)))
        new_id = cur.lastrowid
    return new_id

def delete_planned_session(user_name, session_id):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        cur = conn.execute("DELETE FROM planned_sessions WHERE id = ? AND user_id = ?", (int(session_id), user['id']))
    return cur.rowcount > 0

def replace_planned_sessions(user_name, sessions, after):
    """
    Replaces the user's quick-entry planned sessions dated after `after` (e.g. today)
    with `sessions` (rows with date, tss, note). Past plans and interval-built sessions
    (see add_structured_session) are kept.
    Rows without a date or with a negative/missing TSS are ignored.
    Returns the number of sessions stored.
    """
    user = get_user_data(user_name)
    if not user:
        return 0
    cutoff = pd.Timestamp(after).normalize()
    rows = []
    for r in (sessions.to_dict('records') if hasattr(sessions, 'to_dict') else sessions):
        d = pd.to_datetime(r.get('date'), errors='coerce')
        t = pd.to_numeric(r.get('tss'), errors='coerce')
        if pd.isna(d) or pd.isna(t) or t < 0 or d.normalize() <= cutoff:
            continue
        note = r.get('note')
        rows.append((user['id'], d.strftime("%Y-%m-%d"), float(t),
                     None if note is None or (isinstance(note, float) and pd.isna(note)) else str(note)))
    with connect() as conn:
        conn.execute("DELETE FROM planned_sessions WHERE user_id = ? AND date > ? AND blocks IS NULL",
                     (user['id'], cutoff.strftime("%Y-%m-%d")))
        conn.executemany("INSERT INTO planned_sessions (user_id, date, tss, note) VALUES (?, ?, ?, ?)", rows)
    return len(rows)
