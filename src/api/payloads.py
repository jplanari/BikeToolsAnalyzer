"""What the phone app shows, as plain JSON-ready dicts. Each function takes the rider's name (the API has already
checked the token) and reads the same stored data and analysis functions as the desktop pages, so the two never
disagree. Text that depends on the language (status labels) comes back in English: the phone app translates it with
the same catalogs, keyed by the English text."""
import functools
import math

import numpy as np
import pandas as pd

from src.analysis import nutrition as nut
from src.analysis import power_profile as pp
from src.analysis import training_load as tl
from src.analysis.labels import acwr_status, form_status, monotony_status, ramp_status
from src.analysis.power import estimate_critical_power
from src.analysis.ride_list import collapse_duplicates
from src.data import db

MAX_PAGE = 100
SERIES_POINTS = 400
TRACK_POINTS = 300
CURVE_DURATIONS = (1, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 5400, 7200)
LOAD_DAYS = 180
VOLUME_WEEKS = 12


def _num(v, digits=None):
    """A float for JSON: None for missing/NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits) if digits is not None else f


def _int(v):
    f = _num(v)
    return None if f is None else int(round(f))


def _today(user):
    tz = user.get('timezone')
    try:
        if tz:
            return pd.Timestamp.now(tz=tz).tz_localize(None).normalize()
    except Exception:
        pass
    return pd.Timestamp.today().normalize()


def _day(ts):
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _source(filename):
    name = (filename or "").lower()
    return 'strava' if name.startswith("strava_") else 'wahoo' if name.startswith("wahoo_") else 'file'


def _stem(filename):
    return (filename or "ride").rsplit(".", 1)[0]


def me(user_name):
    u = db.get_user_data(user_name)
    if not u:
        return None
    return {'name': u['name'], 'ftp': _int(u.get('ftp')), 'lthr': _int(u.get('lthr')), 'weight_kg': _num(u.get('weight'), 1),
            'cp': _int(u.get('cp')) or None, 'language': u.get('language'), 'timezone': u.get('timezone')}


# --- Rides ------------------------------------------------------------------------------------------------------

def rides(user_name, limit=30, offset=0):
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    items = collapse_duplicates(db.get_user_rides(user_name))
    items.sort(key=lambda r: str(r.get('date_time') or ''), reverse=True)
    out = []
    for r in items[offset:offset + limit]:
        out.append({'id': r['id'], 'source': _source(r['filename']), 'name': _stem(r['filename']),
                    'when': r.get('start_local'), 'distance_km': _num(r['distance_km'], 1),
                    'elevation_m': _int(r['elevation_m']), 'moving_s': _int(r['moving_s']),
                    'np': _int(r['norm_power']), 'tss': _int(r['tss']), 'kj': _int(r['kj'])})
    return {'total': len(items), 'offset': offset, 'limit': limit, 'rides': out}


def _ride_row(user_name, ride_id):
    user = db.get_user_data(user_name)
    if not user:
        return None, None
    with db.connect() as conn:
        cur = conn.execute("SELECT * FROM rides WHERE id = ? AND user_id = ?", (int(ride_id), user['id']))
        row = cur.fetchone()
        if not row:
            return user, None
        return user, dict(zip([c[0] for c in cur.description], row))


def _zones(ride_id, kind):
    z = db.get_ride_zones([ride_id], kind)
    return [{'zone': str(k), 'seconds': _int(v)} for k, v in sorted(z.items())] if z else []


def _ride_curve(ride_id):
    with db.connect() as conn:
        rows = conn.execute("SELECT duration_sec, watts FROM power_records WHERE ride_id = ? ORDER BY duration_sec",
                            (ride_id,)).fetchall()
    keep = set(CURVE_DURATIONS)
    return [{'seconds': int(d), 'watts': _int(w)} for d, w in rows if int(d) in keep and w is not None]


def _energy(row, weight):
    kj = _num(row.get('kj'))
    if not kj:
        return None
    carb = _num(row.get('carb_g'))
    kcal = nut.kcal_from_kj(kj)
    moving = _num(row.get('moving_s'))
    if_ = _num(row.get('if_factor'))
    out = {'kj': _int(kj), 'kcal': _int(kcal), 'carb_g': _int(carb), 'fat_g': _int(nut.fat_grams(kj, carb)),
           'carb_pct': _int(carb * nut.CARB_KCAL_PER_G / kcal * 100) if carb is not None and kcal else None}
    eaten, fluid = _num(row.get('carb_eaten_g')), _num(row.get('fluid_ml'))
    used = nut.store_used(carb, weight, eaten or 0.0)
    out['store_used_pct'] = _int(used * 100) if used is not None else None
    out['load'] = nut.LOADS.get(nut.load(used)) if used is not None else None
    if moving:
        low, high = nut.target_total(moving, if_)
        rlow, rhigh = nut.target_rate(moving, if_)
        out['guide'] = {'total_low_g': _int(low), 'total_high_g': _int(high), 'rate_low': _int(rlow), 'rate_high': _int(rhigh)}
    if eaten is not None:
        r = nut.ride_intake({'moving_time_s': moving, 'carb_g': carb, 'if': if_}, eaten, fluid, weight)
        out['intake'] = {'eaten_g': _int(eaten), 'fluid_ml': _int(fluid), 'carb_rate': _int(r['carb_rate']),
                         'fluid_rate': _int(r['fluid_rate']),
                         'coverage_pct': _int(r['coverage'] * 100) if r['coverage'] is not None else None,
                         'status': nut.INTAKES.get(r['status'])}
    return out


def ride(user_name, ride_id):
    """Everything the ride page shows, from the stored metrics (no file is opened). None if it is not this rider's."""
    user, row = _ride_row(user_name, ride_id)
    if not row:
        return None
    rid = row['id']
    return {
        'id': rid, 'source': _source(row['filename']), 'name': _stem(row['filename']),
        'when': next((r['start_local'] for r in db.get_user_rides(user_name) if r['id'] == rid), None),
        'distance_km': _num(row.get('distance_km'), 1), 'elevation_m': _int(row.get('elevation_m')),
        'duration_s': _int(row.get('duration_s')), 'moving_s': _int(row.get('moving_s')),
        'avg_speed_kph': _num(row.get('avg_speed_kph'), 1),
        'avg_power': _int(row.get('avg_power')), 'np': _int(row.get('norm_power')),
        'if': _num(row.get('if_factor'), 2), 'tss': _int(row.get('tss')), 'tss_source': row.get('tss_source'),
        'vi': _num(row.get('vi'), 2), 'ef': _num(row.get('ef'), 2), 'decoupling_pct': _num(row.get('decoupling_pct'), 1),
        'avg_hr': _int(row.get('avg_hr')), 'kj': _int(row.get('kj')),
        'ftp': _int(row.get('ftp_at_ride')) or _int(user.get('ftp')),
        'zones': {'power': _zones(rid, 'power'), 'hr': _zones(rid, 'hr')},
        'power_curve': _ride_curve(rid),
        'energy': _energy(row, _num(user.get('weight'))),
    }


@functools.lru_cache(maxsize=16)
def _prepared(db_file, ride_id, stamp):
    """Parsed ride, kept for a few requests (the chart and the map ask separately). `stamp` changes when the
    database is written, so an updated ride is read again."""
    from src.data.processing.pipeline import prepare_ride
    name, data = db.get_ride_file(ride_id)
    if not data:
        return None
    df, _, _ = prepare_ride(name, data)
    return df


def _downsample(values, n):
    """Mean of n equal buckets (None where a bucket has no data)."""
    s = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if len(s) <= n:
        return [_num(v, 1) for v in s]
    edges = np.linspace(0, len(s), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = s[a:b]
        chunk = chunk[~np.isnan(chunk)]
        out.append(_num(chunk.mean(), 1) if len(chunk) else None)
    return out


def series(user_name, ride_id):
    """The ride as a few hundred points per channel, for the phone's charts and map. None if not this rider's;
    {'error': ...} if the stored file cannot be read."""
    user, row = _ride_row(user_name, ride_id)
    if not row:
        return None
    try:
        df = _prepared(db.DB_FILE, row['id'], db.data_stamp())
    except Exception as e:                      # a corrupt file must not take the server down
        return {'error': str(e)[:200]}
    if df is None or df.empty:
        return {'error': "no file stored"}
    n = len(df)
    out = {'seconds': n, 'points': min(n, SERIES_POINTS)}
    chan = {'power': 'power', 'hr': 'hr', 'elevation': 'ele', 'cadence': 'cadence'}
    for key, col in chan.items():
        out[key] = _downsample(df[col], SERIES_POINTS) if col in df.columns and df[col].notna().any() else None
    if 'speed' in df.columns and df['speed'].notna().any():
        out['speed_kph'] = [None if v is None else _num(v * 3.6, 1) for v in _downsample(df['speed'], SERIES_POINTS)]
    else:
        out['speed_kph'] = None
    track = None
    if 'lat' in df.columns and 'lon' in df.columns and df['lat'].notna().any():
        pts = df[['lat', 'lon']].dropna()
        idx = np.linspace(0, len(pts) - 1, min(len(pts), TRACK_POINTS)).astype(int)
        track = [[round(float(pts['lat'].iloc[i]), 5), round(float(pts['lon'].iloc[i]), 5)] for i in idx]
    out['track'] = track
    return out


# --- Training load ----------------------------------------------------------------------------------------------

def training_load(user_name, days=LOAD_DAYS):
    user = db.get_user_data(user_name)
    if not user:
        return None
    today = _today(user)
    pmc = tl.compute_pmc(db.get_user_tss_history(user_name), today=today)
    if pmc.empty:
        return {'empty': True}
    ramp, ratio, mono = tl.ramp_rate(pmc), tl.acwr(pmc), tl.monotony_strain(pmc)
    now = pmc.loc[:today].iloc[-1]
    ramp_now, acwr_now = ramp.loc[:today].iloc[-1], ratio.loc[:today].iloc[-1]
    mono_now = mono['monotony'].loc[:today].iloc[-1]
    view = pmc.loc[today - pd.Timedelta(days=max(1, int(days)) - 1):today]
    stats = db.get_ride_stats(user_name, today - pd.Timedelta(weeks=VOLUME_WEEKS), today)
    vol = tl.volume(stats, 'W')
    return {
        'today': _day(today),
        'now': {'ctl': _num(now['CTL'], 1), 'atl': _num(now['ATL'], 1), 'tsb': _num(now['TSB'], 1),
                'ramp': _num(ramp_now, 1), 'acwr': _num(acwr_now, 2), 'monotony': _num(mono_now, 2)},
        'status': {'form': form_status(now['TSB']), 'ramp': ramp_status(ramp_now), 'acwr': acwr_status(acwr_now),
                   'monotony': monotony_status(mono_now)},
        'days': [{'d': _day(i), 'tss': _int(r['tss']), 'ctl': _num(r['CTL'], 1), 'atl': _num(r['ATL'], 1),
                  'tsb': _num(r['TSB'], 1)} for i, r in view.iterrows()],
        'weeks': [{'d': _day(i), 'rides': _int(r['rides']), 'hours': _num(r['hours'], 1), 'km': _int(r['km']),
                   'tss': _int(r['tss'])} for i, r in vol.iterrows()],
    }


# --- Power profile ----------------------------------------------------------------------------------------------

def _curve_rows(curve, weight):
    keep = set(CURVE_DURATIONS)
    rows = []
    for _, r in curve.iterrows():
        if int(r['duration']) not in keep:
            continue
        w = _num(r['watts'])
        rows.append({'seconds': int(r['duration']), 'watts': _int(w), 'w_per_kg': _num(w / weight, 2) if w and weight else None,
                     'date': _day(r['date']), 'ride_id': _int(r['ride_id'])})
    return rows


def power(user_name):
    user = db.get_user_data(user_name)
    if not user:
        return None
    recs = db.get_power_records(user_name)
    if recs.empty:
        return {'empty': True}
    today = _today(user)
    weight, ftp = _num(user.get('weight')), _num(user.get('ftp'))
    curves = pp.window_curves(recs, today)
    recent = pp.as_dict(curves['Last 90 days'])
    allt = pp.as_dict(curves['All time'])
    two = estimate_critical_power(recent) if recent else None
    three = pp.fit_cp3(recent) if recent else None
    prof90, profall = pp.power_profile(recent, weight, ftp), pp.power_profile(allt, weight, ftp)
    profile = []
    for (_, a), (_, b) in zip(prof90.iterrows(), profall.iterrows()):
        profile.append({'label': a['duration'], 'watts': _int(a['watts']), 'w_per_kg': _num(a['w_per_kg'], 2),
                        'x_ftp': _num(a['x_ftp'], 2), 'all_time_w_per_kg': _num(b['w_per_kg'], 2)})
    return {
        'today': _day(today), 'ftp': _int(ftp), 'weight_kg': weight,
        'windows': [{'label': label, 'points': _curve_rows(c, weight)} for label, c in curves.items()],
        'cp': {'two': {'cp': _int(two['cp']), 'w_prime_kj': _num(two['w_prime'] / 1000, 1)} if two else None,
               'three': {'cp': _int(three['cp']), 'w_prime_kj': _num(three['w_prime'] / 1000, 1),
                         'pmax': _int(three['pmax'])} if three else None},
        'profile': profile,
        'records': _curve_rows(curves['All time'], weight),
    }
