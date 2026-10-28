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

def build_ride_metadata(df, dist_m, ele_m, speed_kph, np_val, tss, if_val, filename=None):
    """
    Builds the metadata dict that save_ride() expects. Keeping both in this
    module guarantees the keys match (they did not before: date, distance
    and elevation were silently stored as NULL / 0).
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
        'tss': int(tss) if pd.notna(tss) else 0,
        'if': if_val,
        'avg_hr': safe_mean('hr'),
        'strava_activity_id': strava_id_from_filename(filename),
    }

def save_ride(user_name, filename, file_content, metadata, power_curve_dict):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."

    file_hash = hashlib.md5(file_content).hexdigest()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # 1. Insert Ride Metadata
        c.execute('''
            INSERT INTO rides (
                user_id, filename, file_hash, file_content, date_time, 
                distance_km, elevation_m, avg_speed_kph, 
                norm_power, avg_power, tss, if_factor, avg_hr, strava_activity_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['id'], filename, file_hash, file_content, metadata.get('date'),
            metadata.get('dist', 0), metadata.get('ele', 0), metadata.get('speed', 0),
            metadata.get('np', 0), metadata.get('avg_p', 0), metadata.get('tss', 0),
            metadata.get('if', 0), metadata.get('avg_hr', 0),
            metadata.get('strava_activity_id')
        ))
        
        ride_id = c.lastrowid
        
        # 2. Insert Power Records (Curve)
        if power_curve_dict:
            records = []
            for duration, watts in power_curve_dict.items():
                records.append((user['id'], ride_id, duration, watts, metadata.get('date')))
            
            c.executemany('''
                INSERT INTO power_records (user_id, ride_id, duration_sec, watts, date_time)
                VALUES (?, ?, ?, ?, ?)
            ''', records)
            
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
        SELECT id, filename, date_time, distance_km, elevation_m, norm_power, tss 
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


# --- History repair --------------------------------------------------------
# Rides saved before the metadata-key fix have date_time = NULL and
# distance_km / elevation_m = 0. The original file is stored in
# rides.file_content, so these fields can be rebuilt.

# Keyed on the missing date only: an indoor ride legitimately has 0 km.
_MISSING_META_SQL = "(date_time IS NULL OR date_time = '')"

def count_rides_missing_metadata(user_name):
    user = get_user_data(user_name)
    if not user:
        return 0
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"SELECT COUNT(*) FROM rides WHERE user_id = ? AND {_MISSING_META_SQL}", (user['id'],))
    n = c.fetchone()[0]
    conn.close()
    return n

def _ride_metadata_from_file(filename, file_bytes):
    """
    Re-parses a stored ride with the same pipeline the dashboard uses and
    returns (iso_start_time, distance_km, elevation_m). Distance/elevation
    are None when the file has no GPS track (e.g. indoor rides).
    """
    from src.data.pipeline import prepare_ride  # lazy: parsers import streamlit

    df, total_dist, total_ascent = prepare_ride(filename, file_bytes)
    date_iso = _iso_timestamp(df['time'].dropna().iloc[0])
    if 'lat' not in df.columns or df['lat'].isna().all():
        return date_iso, None, None
    return date_iso, total_dist / 1000.0, float(total_ascent)

def repair_ride_metadata(user_name):
    """
    Fills in date_time (and distance/elevation when missing) for rides that
    were saved without them, and copies the date to their power records.
    Returns (n_repaired, [filenames that could not be parsed]).
    """
    user = get_user_data(user_name)
    if not user:
        return 0, []

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"""SELECT id, filename, file_content, date_time, distance_km, elevation_m
                  FROM rides WHERE user_id = ? AND {_MISSING_META_SQL}""", (user['id'],))
    rows = c.fetchall()

    repaired, failed = 0, []
    for ride_id, filename, blob, old_date, old_dist, old_ele in rows:
        try:
            date_iso, dist_km, ele_m = _ride_metadata_from_file(filename or "", blob)
        except Exception:
            failed.append(filename or f"ride {ride_id}")
            continue

        new_date = old_date or date_iso
        new_dist = dist_km if (not old_dist and dist_km is not None) else old_dist
        new_ele = ele_m if (not old_ele and ele_m is not None) else old_ele
        c.execute("UPDATE rides SET date_time = ?, distance_km = ?, elevation_m = ? WHERE id = ?",
                  (new_date, new_dist, new_ele, ride_id))
        c.execute("UPDATE power_records SET date_time = ? WHERE ride_id = ? AND date_time IS NULL",
                  (new_date, ride_id))
        repaired += 1

    conn.commit()
    conn.close()
    return repaired, failed


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
