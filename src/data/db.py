import sqlite3
import pandas as pd
import hashlib
import re
import secrets
import time

DB_FILE = "bikethools.db"

def migrate_db():
    """
    Adds new columns (cp, w_prime) to the users table if they don't exist.
    Run this once safely; it catches errors if columns already exist.
    """
    conn = sqlite3.connect(DB_FILE)
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
    try:
        c.execute("ALTER TABLE users ADD COLUMN crank_mm REAL DEFAULT 172.5")
    except sqlite3.OperationalError:
        pass

    # One copy of each Strava activity per rider (NULLs, i.e. uploads, are exempt)
    c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_rides_user_strava
                 ON rides(user_id, strava_activity_id)""")

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
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def get_user_data(name):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_user(name):
    conn = sqlite3.connect(DB_FILE)
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
METRICS_VERSION = 2  # 2: joint power/HR/cadence histograms, HR recovery

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
    hist = metadata.get('hist') or {}
    c.executemany("INSERT INTO ride_hist (ride_id, pair, x, y, seconds) VALUES (?, ?, ?, ?, ?)",
                  [(ride_id, pair, int(x), int(y), int(sec))
                   for pair, cells in hist.items() for (x, y), sec in cells.items()])

def save_ride(user_name, filename, file_content, metadata, power_curve_dict):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."

    file_hash = hashlib.md5(file_content).hexdigest()
    
    conn = sqlite3.connect(DB_FILE)
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
    
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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
    
    conn = sqlite3.connect(DB_FILE)
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
    
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    n = conn.execute(f"SELECT COUNT(*) FROM rides WHERE user_id = ? AND {_NEEDS_UPDATE_SQL}",
                     (user['id'], METRICS_VERSION)).fetchone()[0]
    conn.close()
    return n

def get_rides_for_recompute(user_name, all_rides=False):
    """Rows needed to rebuild a ride's metrics (all rides, or only outdated ones)."""
    user = get_user_data(user_name)
    if not user:
        return []
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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
    _insert_children(c, user_id, ride_id, metadata, power_curve_dict)
    conn.commit()
    conn.close()

def get_ride_zones(ride_ids, kind):
    """{zone: total seconds} summed over the given rides."""
    if not ride_ids:
        return {}
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM strava_auth WHERE user_id = ?", (user['id'],)).fetchone()
    conn.close()
    return dict(row) if row else None

def delete_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - OAUTH_STATE_TTL_S,))
    conn.execute("INSERT INTO oauth_states (state, user_id, created_at) VALUES (?, ?, ?)",
                 (state, user['id'], now))
    conn.commit()
    conn.close()
    return state

def consume_oauth_state(state):
    """Returns the user name the state was issued for (once), else None."""
    now = int(time.time())
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT strava_activity_id FROM rides WHERE user_id = ? "
                        "AND strava_activity_id IS NOT NULL", (user['id'],)).fetchall()
    conn.close()
    return {r[0] for r in rows}

def get_ride_start_times(user_name):
    """Start times (UTC, naive) of all dated rides, for cross-source dedup."""
    user = get_user_data(user_name)
    if not user:
        return []
    conn = sqlite3.connect(DB_FILE)
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
    """Stored ISO strings (with or without offset) -> naive UTC datetimes."""
    parsed = [pd.to_datetime(d, errors='coerce', utc=True) for d in series]
    return pd.Series(pd.DatetimeIndex(parsed, dtype='datetime64[ns, UTC]').tz_convert(None),
                     index=series.index)

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
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"""SELECT r.id AS ride_id, r.date_time AS date, r.duration_s, r.moving_s,
                                      r.norm_power, r.avg_hr, r.vi, r.ef, r.decoupling_pct,
                                      r.hrr60_bpm, r.hrr_efforts, r.tss, r.tss_source
                               FROM rides r WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}
                               ORDER BY r.date_time""", conn, params=[user['id'], *params])
    conn.close()
    df['date'] = _parse_dates(df['date'])
    return df

def update_user_crank(user_name, crank_mm):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE users SET crank_mm = ? WHERE name = ?", (float(crank_mm), user_name))
    conn.commit()
    conn.close()
