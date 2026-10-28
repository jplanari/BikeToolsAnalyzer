import sqlite3
import pandas as pd
import hashlib
import re
import secrets
import time

DB_FILE = "bikethools.db"


def _connect():
    """
    Every connection goes through here. SQLite only enforces foreign keys (and so
    ON DELETE CASCADE) when asked to, per connection; without this, deleting a user
    left their rides, records and histograms behind.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def migrate_db():
    """
    Adds new columns (cp, w_prime) to the users table if they don't exist.
    Run this once safely; it catches errors if columns already exist.
    """
    conn = _connect()
    c = conn.cursor()
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN cp INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column likely exists already

    try:
        c.execute("ALTER TABLE users ADD COLUMN w_prime REAL DEFAULT 20000")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE rides ADD COLUMN strava_activity_id INTEGER")
    except sqlite3.OperationalError:
        pass
    # Phase 1 per-ride metrics. metrics_version marks rides whose stored
    # metrics predate the current calculation (0 = legacy) so they can be
    # recomputed from their stored files.
    for col, typ in [("duration_s", "INTEGER"), ("moving_s", "INTEGER"), ("kj", "REAL"),
                     ("ftp_at_ride", "INTEGER"), ("lthr_at_ride", "INTEGER"),
                     ("vi", "REAL"), ("ef", "REAL"), ("decoupling_pct", "REAL"),
                     ("tss_source", "TEXT"), ("metrics_version", "INTEGER DEFAULT 0"),
                     ("hrr60_bpm", "REAL"), ("hrr_efforts", "INTEGER")]:
        try:
            c.execute(f"ALTER TABLE rides ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
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
    try:
        c.execute("ALTER TABLE users ADD COLUMN crank_mm REAL DEFAULT 172.5")
    except sqlite3.OperationalError:
        pass

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
        CREATE TABLE IF NOT EXISTS oauth_states (
            state TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def init_db():
    """Initializes the database with users, rides, and power_records tables."""
    conn = _connect()
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
            file_hash TEXT UNIQUE,
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
    
    conn.commit()
    conn.close()
    
    # Run migration to ensure existing databases get the new columns
    migrate_db()

def save_user(name, ftp, lthr, weight, height=175.0):
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, ftp, lthr, weight, height) VALUES (?, ?, ?, ?, ?)", 
                  (name, ftp, lthr, weight, height))
        conn.commit()
        return True, "User saved successfully!"
    except sqlite3.IntegrityError:
        return False, "User already exists."
    except Exception as e:
        return False, f"Database Error: {e}"
    finally:
        conn.close()

def update_user_physics(user_name, cp, w_prime, update_ftp=False):
    """
    Updates the user's physiological metrics.
    Optionally updates FTP to match CP.
    """
    conn = _connect()
    c = conn.cursor()
    
    if update_ftp:
        c.execute('''
            UPDATE users 
            SET cp = ?, w_prime = ?, ftp = ? 
            WHERE name = ?
        ''', (cp, w_prime, cp, user_name))
    else:
        c.execute('''
            UPDATE users 
            SET cp = ?, w_prime = ? 
            WHERE name = ?
        ''', (cp, w_prime, user_name))
        
    conn.commit()
    conn.close()

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
        src, dst = sqlite3.connect(DB_FILE), sqlite3.connect(dest)
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
    stamp = []
    for path in (DB_FILE, DB_FILE + "-wal"):
        try:
            st = os.stat(path)
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)

def get_all_users():
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def get_user_data(name):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_user(name):
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()
    conn.close()

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
METRICS_VERSION = 5

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
        'hrr_efforts': _opt_int(metrics.get('hrr_efforts')),
        'metrics_version': METRICS_VERSION,
    }

_METRIC_COLUMNS = ['duration_s', 'moving_s', 'kj', 'ftp_at_ride', 'lthr_at_ride', 'vi', 'ef',
                   'decoupling_pct', 'tss_source', 'hrr60_bpm', 'hrr_efforts', 'metrics_version']

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
    
    conn = _connect()
    c = conn.cursor()
    
    try:
        # 1. Insert Ride Metadata
        cols = ['user_id', 'filename', 'file_hash', 'file_content', 'date_time',
                'distance_km', 'elevation_m', 'avg_speed_kph', 'norm_power', 'avg_power',
                'tss', 'if_factor', 'avg_hr', 'strava_activity_id'] + _METRIC_COLUMNS
        vals = [user['id'], filename, file_hash, file_content, metadata.get('date'),
                metadata.get('dist', 0), metadata.get('ele', 0), metadata.get('speed', 0),
                metadata.get('np', 0), metadata.get('avg_p', 0), metadata.get('tss', 0),
                metadata.get('if', 0), metadata.get('avg_hr', 0),
                metadata.get('strava_activity_id')] + [
                metadata.get(k, 0 if k == 'metrics_version' else None) for k in _METRIC_COLUMNS]
        c.execute(f"INSERT INTO rides ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        ride_id = c.lastrowid

        # 2. Power records (curve) and time in zones
        _insert_children(c, user['id'], ride_id, metadata, power_curve_dict)
            
        conn.commit()
        return True, "Ride saved successfully!"
        
    except sqlite3.IntegrityError:
        return False, "Ride already exists (duplicate file)."
        
    except Exception as e:
        return False, f"Database Error: {e}"
        
    finally:
        conn.close()

def get_user_rides(user_name):
    user = get_user_data(user_name)
    if not user:
        return []
    
    conn = _connect()
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
    conn.close()
    return [dict(row) for row in rows]

def get_ride_file(ride_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT filename, file_content FROM rides WHERE id = ?", (ride_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def get_user_best_power(user_name):
    user = get_user_data(user_name)
    if not user:
        return {}
    
    conn = _connect()
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
    conn.close()
    
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
    
    conn = _connect()
    c = conn.cursor()
    
    # Just select the columns we need
    c.execute('''
        SELECT date_time, tss 
        FROM rides 
        WHERE user_id = ? 
        ORDER BY date_time ASC
    ''', (user['id'],))
    
    rows = c.fetchall()
    conn.close()
    
    # Return raw list of dicts
    return [{'date': row[0], 'tss': row[1]} for row in rows]


# --- Recompute history ------------------------------------------------------
# Rides whose stored metrics are older than METRICS_VERSION (or that were
# saved without a date by the pre-Phase-0 bug) are rebuilt from the original
# file kept in rides.file_content. See src/data/history.py.

_NEEDS_UPDATE_SQL = "(metrics_version IS NULL OR metrics_version < ? OR date_time IS NULL OR date_time = '')"

def count_rides_needing_update(user_name):
    user = get_user_data(user_name)
    if not user:
        return 0
    conn = _connect()
    n = conn.execute(f"SELECT COUNT(*) FROM rides WHERE user_id = ? AND {_NEEDS_UPDATE_SQL}",
                     (user['id'], METRICS_VERSION)).fetchone()[0]
    conn.close()
    return n

def get_rides_for_recompute(user_name, all_rides=False):
    """Rows needed to rebuild a ride's metrics (all rides, or only outdated ones)."""
    user = get_user_data(user_name)
    if not user:
        return []
    conn = _connect()
    conn.row_factory = sqlite3.Row
    where = "user_id = ?" + ("" if all_rides else f" AND {_NEEDS_UPDATE_SQL}")
    params = (user['id'],) if all_rides else (user['id'], METRICS_VERSION)
    rows = conn.execute(f"""SELECT id, filename, file_content, date_time, norm_power, if_factor,
                                   ftp_at_ride, lthr_at_ride
                            FROM rides WHERE {where} ORDER BY date_time""", params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_ride_metrics(ride_id, metadata, power_curve_dict):
    """Overwrites a ride's computed fields, power records and zone times."""
    conn = _connect()
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
    conn.commit()
    conn.close()

def get_ride_zones(ride_ids, kind):
    """{zone: total seconds} summed over the given rides."""
    if not ride_ids:
        return {}
    conn = _connect()
    q = ",".join("?" * len(ride_ids))
    rows = conn.execute(f"""SELECT zone, SUM(seconds) FROM ride_zones
                            WHERE kind = ? AND ride_id IN ({q}) GROUP BY zone ORDER BY zone""",
                        (kind, *ride_ids)).fetchall()
    conn.close()
    return {z: s for z, s in rows}


# --- Strava authorization --------------------------------------------------
# Tokens are stored per rider profile in the local database (plain text, like
# the rest of the local data). Strava may rotate the refresh token on every
# refresh, so save_strava_auth() must be called whenever it changes.

def save_strava_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    conn = _connect()
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
    conn.commit()
    conn.close()
    return True

def get_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM strava_auth WHERE user_id = ?", (user['id'],)).fetchone()
    conn.close()
    return dict(row) if row else None

def delete_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    conn = _connect()
    conn.execute("DELETE FROM strava_auth WHERE user_id = ?", (user['id'],))
    conn.commit()
    conn.close()

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
    conn = _connect()
    conn.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - OAUTH_STATE_TTL_S,))
    conn.execute("INSERT INTO oauth_states (state, user_id, created_at) VALUES (?, ?, ?)",
                 (state, user['id'], now))
    conn.commit()
    conn.close()
    return state

def consume_oauth_state(state):
    """Returns the user name the state was issued for (once), else None."""
    now = int(time.time())
    conn = _connect()
    row = conn.execute('''SELECT u.name, s.created_at FROM oauth_states s
                          JOIN users u ON u.id = s.user_id WHERE s.state = ?''', (state,)).fetchone()
    conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
    conn.commit()
    conn.close()
    if not row or now - row[1] > OAUTH_STATE_TTL_S:
        return None
    return row[0]

def get_imported_strava_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    conn = _connect()
    rows = conn.execute("SELECT strava_activity_id FROM rides WHERE user_id = ? "
                        "AND strava_activity_id IS NOT NULL", (user['id'],)).fetchall()
    conn.close()
    return {r[0] for r in rows}

def get_ride_start_times(user_name):
    """Start times (UTC, naive) of all dated rides, for cross-source dedup."""
    user = get_user_data(user_name)
    if not user:
        return []
    conn = _connect()
    rows = conn.execute("SELECT date_time FROM rides WHERE user_id = ? AND date_time IS NOT NULL",
                        (user['id'],)).fetchall()
    conn.close()
    out = []
    for (d,) in rows:
        ts = pd.to_datetime(d, errors='coerce', utc=True)
        if pd.notna(ts):
            out.append(ts.tz_convert(None))
    return out


# --- Deeper analytics queries ------------------------------------------------

def _parse_dates(series):
    """
    Stored ISO strings (with or without offset) -> naive UTC datetimes.
    A history has few distinct dates but many rows (one per histogram cell or power
    record), so each distinct string is parsed once and mapped back.
    """
    uniq = pd.unique(series.dropna())
    if len(uniq) == 0:
        return pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    try:
        parsed = pd.to_datetime(pd.Series(uniq), errors='coerce', utc=True, format='ISO8601')
    except (ValueError, TypeError):      # unusual stored format: slower, forgiving path
        parsed = pd.Series([pd.to_datetime(d, errors='coerce', utc=True) for d in uniq])
    parsed = pd.Series(pd.DatetimeIndex(parsed).tz_convert(None))
    out = series.map(dict(zip(uniq, parsed)))
    return pd.to_datetime(out).astype('datetime64[ns]')

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
    conn = _connect()
    df = pd.read_sql_query(f"""SELECT h.ride_id, r.date_time AS date, h.x, h.y, h.seconds
                               FROM ride_hist h JOIN rides r ON r.id = h.ride_id
                               WHERE r.user_id = ? AND h.pair = ?{where}""",
                           conn, params=[user['id'], pair, *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df

def get_ride_stats(user_name, start=None, end=None):
    """Per-ride stored metrics for trend charts (one row per dated ride)."""
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame()
    where, params = _date_filter(start, end)
    conn = _connect()
    df = pd.read_sql_query(f"""SELECT r.id AS ride_id, r.date_time AS date, r.duration_s, r.moving_s,
                                      r.norm_power, r.avg_hr, r.vi, r.ef, r.decoupling_pct,
                                      r.hrr60_bpm, r.hrr_efforts, r.tss, r.tss_source,
                                      r.distance_km, r.elevation_m, r.kj
                               FROM rides r WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}
                               ORDER BY r.date_time""", conn, params=[user['id'], *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df

def update_user_crank(user_name, crank_mm):
    conn = _connect()
    conn.execute("UPDATE users SET crank_mm = ? WHERE name = ?", (float(crank_mm), user_name))
    conn.commit()
    conn.close()

def period_coverage(user_name, start=None, end=None):
    """Why analytics may be empty: rides in the period and what they contain."""
    out = {'rides': 0, 'needs_update': 0, 'p_hr': 0, 'p_cad': 0}
    user = get_user_data(user_name)
    if not user:
        return out
    where, params = _date_filter(start, end)
    conn = _connect()
    out['rides'], out['needs_update'] = conn.execute(
        f"""SELECT COUNT(*), COALESCE(SUM(CASE WHEN r.metrics_version IS NULL
                                               OR r.metrics_version < ? THEN 1 ELSE 0 END), 0)
            FROM rides r WHERE r.user_id = ?{where}""",
        [METRICS_VERSION, user['id'], *params]).fetchone()
    for pair in ('p_hr', 'p_cad'):
        out[pair] = conn.execute(
            f"""SELECT COUNT(DISTINCT h.ride_id) FROM ride_hist h JOIN rides r ON r.id = h.ride_id
                WHERE r.user_id = ? AND h.pair = ?{where}""", [user['id'], pair, *params]).fetchone()[0]
    conn.close()
    return out

def get_power_records(user_name, start=None, end=None):
    """Per-ride mean-maximal power: DataFrame(duration, watts, date, ride_id)."""
    cols = ['duration', 'watts', 'date', 'ride_id']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    conn = _connect()
    df = pd.read_sql_query(f"""SELECT p.duration_sec AS duration, p.watts, r.date_time AS date, r.id AS ride_id
                               FROM power_records p JOIN rides r ON r.id = p.ride_id
                               WHERE r.user_id = ? AND r.date_time IS NOT NULL
                                 AND p.watts IS NOT NULL{where}""",
                           conn, params=[user['id'], *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_kj_records(user_name, start=None, end=None):
    """Power after X kJ per ride: DataFrame(kj, duration, watts, date, ride_id)."""
    cols = ['kj', 'duration', 'watts', 'date', 'ride_id']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    conn = _connect()
    df = pd.read_sql_query(f"""SELECT k.kj, k.duration_sec AS duration, k.watts, r.date_time AS date,
                                      r.id AS ride_id
                               FROM power_records_kj k JOIN rides r ON r.id = k.ride_id
                               WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}""",
                           conn, params=[user['id'], *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_zone_rows(user_name, kind, start=None, end=None):
    """Time in zones per ride: DataFrame(ride_id, date, zone, seconds); kind 'power' or 'hr'."""
    cols = ['ride_id', 'date', 'zone', 'seconds']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end)
    conn = _connect()
    df = pd.read_sql_query(f"""SELECT z.ride_id, r.date_time AS date, z.zone, z.seconds
                               FROM ride_zones z JOIN rides r ON r.id = z.ride_id
                               WHERE r.user_id = ? AND z.kind = ? AND r.date_time IS NOT NULL{where}""",
                           conn, params=[user['id'], kind, *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df[cols]

def get_planned_sessions(user_name, after=None):
    """Planned sessions (date, tss, note), optionally only after a date."""
    cols = ['date', 'tss', 'note']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    sql = "SELECT date, tss, note FROM planned_sessions WHERE user_id = ?"
    params = [user['id']]
    if after is not None:
        sql += " AND date > ?"
        params.append(pd.Timestamp(after).strftime("%Y-%m-%d"))
    conn = _connect()
    df = pd.read_sql_query(sql + " ORDER BY date", conn, params=params)
    conn.close()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[cols]

def replace_planned_sessions(user_name, sessions, after):
    """
    Replaces the user's planned sessions dated after `after` (e.g. today)
    with `sessions` (rows with date, tss, note). Past plans are kept.
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
    conn = _connect()
    conn.execute("DELETE FROM planned_sessions WHERE user_id = ? AND date > ?",
                 (user['id'], cutoff.strftime("%Y-%m-%d")))
    conn.executemany("INSERT INTO planned_sessions (user_id, date, tss, note) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return len(rows)
