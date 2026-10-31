"""Stored rides: saving, reading, per-ride metrics and recompute."""
from src.analysis.dates import to_local_naive
from src.analysis.dates import to_utc_iso
import hashlib
import pandas as pd
import re
import sqlite3
import threading
import warnings
from . import core
from .core import _connect, _date_filter, backup_db, connect
from .users import get_user_data

def get_hr20(user_name, start=None):
    """Best-20-minute mean HR of each ride (> 0 only), newest rides last: list of (date, bpm)."""
    user = get_user_data(user_name)
    if not user:
        return []
    where, params = _date_filter(start, None, user)
    with connect() as conn:
        rows = conn.execute(f"""SELECT date_time, hr20_bpm FROM rides r WHERE user_id = ? AND hr20_bpm > 0{where}
                                ORDER BY date_time""", [user['id'], *params]).fetchall()
    days = to_local_naive(pd.Series([r[0] for r in rows], dtype=object), user.get('timezone'))
    return [(None if pd.isna(d) else d.isoformat(), r[1]) for d, r in zip(days, rows)]

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

def strava_id_from_filename(filename):
    """'strava_123.json' -> 123 (rides imported through the Strava API)."""
    m = re.fullmatch(r"strava_(\d+)\.json", (filename or "").lower())
    return int(m.group(1)) if m else None

def _iso_timestamp(ts):
    """Ride start time as a UTC ISO-8601 string (a naive time is taken as UTC), the one format stored."""
    return to_utc_iso(ts)

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
            vac = sqlite3.connect(core.DB_FILE, timeout=core.BUSY_TIMEOUT_S)
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
        ''', [(user_id, ride_id, int(d), float(w), to_utc_iso(metadata.get('date')))
              for d, w in power_curve_dict.items() if w is not None and pd.notna(w)])
    zones = metadata.get('zones') or {}
    c.executemany("INSERT INTO ride_zones (ride_id, kind, zone, seconds) VALUES (?, ?, ?, ?)",
                  [(ride_id, kind, str(z), int(sec))
                   for kind, zz in zones.items() for z, sec in zz.items()])
    fatigued = metadata.get('fatigued') or {}
    c.executemany("""INSERT INTO power_records_kj (ride_id, user_id, kj, duration_sec, watts, date_time)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  [(ride_id, user_id, int(kj), int(d), float(w), to_utc_iso(metadata.get('date')))
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
                    'tss', 'if_factor', 'avg_hr', 'strava_activity_id', 'bike_id'] + _METRIC_COLUMNS
            vals = [user['id'], filename, file_hash, to_utc_iso(metadata.get('date')),
                    metadata.get('dist', 0), metadata.get('ele', 0), metadata.get('speed', 0),
                    metadata.get('np', 0), metadata.get('avg_p', 0), metadata.get('tss', 0),
                    metadata.get('if', 0), metadata.get('avg_hr', 0),
                    metadata.get('strava_activity_id'), metadata.get('bike_id')] + [
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
    out = [dict(row) for row in rows]
    local = to_local_naive(pd.Series([r['date_time'] for r in out], dtype=object), user.get('timezone'))
    for r, t in zip(out, local):
        r['start_local'] = None if pd.isna(t) else t.isoformat()        # the rider's own clock, naive
    return out

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
    
    # Dates on the rider's own clock (naive, so the training-load day buckets are their days)
    days = to_local_naive(pd.Series([r[0] for r in rows], dtype=object), user.get('timezone'))
    return [{'date': (None if pd.isna(d) else d.isoformat()), 'tss': row[1]} for d, row in zip(days, rows)]


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
        vals = [to_utc_iso(metadata.get('date')), metadata.get('dist', 0), metadata.get('ele', 0),
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
