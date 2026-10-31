"""Cross-ride queries behind the analytics pages."""
import pandas as pd
from .core import _date_filter, _parse_dates, connect
from .rides import METRICS_VERSION
from .users import get_user_data

def get_hist(user_name, pair, start=None, end=None):
    """
    Per-ride histogram cells for a period: DataFrame(ride_id, date, x, y, seconds).
    start/end are inclusive dates (None = open-ended).
    """
    cols = ['ride_id', 'date', 'x', 'y', 'seconds']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end, user)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT h.ride_id, r.date_time AS date, h.x, h.y, h.seconds
                                   FROM ride_hist h JOIN rides r ON r.id = h.ride_id
                                   WHERE r.user_id = ? AND h.pair = ?{where}""",
                               conn, params=[user['id'], pair, *params])
    df['date'] = _parse_dates(df['date'], user)
    return df

def get_ride_stats(user_name, start=None, end=None):
    """Per-ride stored metrics for trend charts (one row per dated ride)."""
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame()
    where, params = _date_filter(start, end, user)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT r.id AS ride_id, r.date_time AS date, r.duration_s, r.moving_s,
                                          r.norm_power, r.avg_hr, r.vi, r.ef, r.decoupling_pct,
                                          r.hrr60_bpm, r.hrr_efforts, r.tss, r.tss_source,
                                          r.distance_km, r.elevation_m, r.kj
                                   FROM rides r WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}
                                   ORDER BY r.date_time""", conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'], user)
    return df

def period_coverage(user_name, start=None, end=None):
    """Why analytics may be empty: rides in the period and what they contain."""
    out = {'rides': 0, 'needs_update': 0, 'p_hr': 0, 'p_cad': 0}
    user = get_user_data(user_name)
    if not user:
        return out
    where, params = _date_filter(start, end, user)
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
    where, params = _date_filter(start, end, user)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT p.duration_sec AS duration, p.watts, r.date_time AS date, r.id AS ride_id
                                   FROM power_records p JOIN rides r ON r.id = p.ride_id
                                   WHERE r.user_id = ? AND r.date_time IS NOT NULL
                                     AND p.watts IS NOT NULL{where}""",
                               conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'], user)
    return df[cols]

def get_kj_records(user_name, start=None, end=None):
    """Power after X kJ per ride: DataFrame(kj, duration, watts, date, ride_id)."""
    cols = ['kj', 'duration', 'watts', 'date', 'ride_id']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end, user)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT k.kj, k.duration_sec AS duration, k.watts, r.date_time AS date,
                                          r.id AS ride_id
                                   FROM power_records_kj k JOIN rides r ON r.id = k.ride_id
                                   WHERE r.user_id = ? AND r.date_time IS NOT NULL{where}""",
                               conn, params=[user['id'], *params])
    df['date'] = _parse_dates(df['date'], user)
    return df[cols]

def get_zone_rows(user_name, kind, start=None, end=None):
    """Time in zones per ride: DataFrame(ride_id, date, zone, seconds); kind 'power' or 'hr'."""
    cols = ['ride_id', 'date', 'zone', 'seconds']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    where, params = _date_filter(start, end, user)
    with connect() as conn:
        df = pd.read_sql_query(f"""SELECT z.ride_id, r.date_time AS date, z.zone, z.seconds
                                   FROM ride_zones z JOIN rides r ON r.id = z.ride_id
                                   WHERE r.user_id = ? AND z.kind = ? AND r.date_time IS NOT NULL{where}""",
                               conn, params=[user['id'], kind, *params])
    df['date'] = _parse_dates(df['date'], user)
    return df[cols]
