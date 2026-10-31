"""
Weather of a ride: hourly temperature, pressure, humidity and wind from Open-Meteo, interpolated onto the ride.

Design:
  * `fetch_hourly` asks the service (timeout, status check, retries) and returns plain data; failures raise
    WeatherError, so nothing is cached for them.
  * The fetched hourly data is a small JSON-able payload that is stored with the ride (db.save_ride_weather):
    reopening a ride, or building the weekly report, needs no network and gives the same figures.
  * `apply_weather` is pure: payload + ride -> the ride with wind / temperature / pressure columns.
  * A long ride is sampled in stretches of SEGMENT_KM, each with its own position, not at one centroid.
  * Wind is interpolated as u/v components (interpolating a direction through north goes the wrong way).
  * The service: the free Open-Meteo API is for non-commercial use. Set OPEN_METEO_API_KEY (paid plan) and the
    customer endpoints are used; see the positioning document.
"""
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

SEGMENT_KM = 25.0                 # one weather sample position per this much riding
POSITION_DECIMALS = 2             # positions are rounded (~1 km) before they leave the machine and for the cache
TIMEOUT_S = 10
RETRIES = 3
ARCHIVE_DELAY_DAYS = 6            # the ERA5 archive lags by ~5 days: newer rides come from the forecast service
HOURLY = ["temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m"]

_FREE = {"archive": "https://archive-api.open-meteo.com/v1/archive",
         "forecast": "https://api.open-meteo.com/v1/forecast"}
_PAID = {"archive": "https://customer-archive-api.open-meteo.com/v1/archive",
         "forecast": "https://customer-api.open-meteo.com/v1/forecast"}


class WeatherError(Exception):
    """The weather service could not be reached or answered with something unusable."""


# ---------------------------------------------------------------------------------------------- positions
def _naive_utc(times):
    t = pd.to_datetime(pd.Series(times))
    if getattr(t.dt, 'tz', None) is not None:
        t = t.dt.tz_convert('UTC').dt.tz_localize(None)
    return t


def sample_points(df, segment_km=SEGMENT_KM):
    """
    Stretches of the ride that get their own weather: [{lat, lon, start, end}] with positional sample
    indexes start..end (end exclusive). Only samples with a valid position count; [] when there are none
    (an indoor ride, a file with no GPS).
    """
    if df.empty or not {'lat', 'lon', 'time'} <= set(df.columns):
        return []
    lat = pd.to_numeric(df['lat'], errors='coerce').to_numpy(dtype=float)
    lon = pd.to_numeric(df['lon'], errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180) & ((lat != 0) | (lon != 0))
    if not ok.any():
        return []
    # distance along the valid positions (haversine, metres)
    la, lo = np.radians(lat[ok]), np.radians(lon[ok])
    step = np.zeros(len(la))
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    step[1:] = 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    step[step > 2000] = 0.0                       # a GPS jump is not riding
    along = np.cumsum(step)
    n_seg = max(1, int(np.ceil(along[-1] / (segment_km * 1000.0)))) if along[-1] > 0 else 1
    seg_of_valid = np.minimum((along / (segment_km * 1000.0)).astype(int), n_seg - 1)
    idx_valid = np.flatnonzero(ok)
    points = []
    for s in range(n_seg):
        members = idx_valid[seg_of_valid == s]
        if not len(members):
            continue
        points.append({'lat': round(float(np.median(lat[members])), POSITION_DECIMALS),
                       'lon': round(float(np.median(lon[members])), POSITION_DECIMALS),
                       'start': int(members[0]), 'end': int(members[-1]) + 1})
    # stretches cover the whole ride: each one starts where the previous ended
    for i, p in enumerate(points):
        p['start'] = 0 if i == 0 else points[i - 1]['end']
    points[-1]['end'] = len(df)
    return points


# ---------------------------------------------------------------------------------------------- service
def _endpoint(kind):
    key = os.environ.get("OPEN_METEO_API_KEY")
    return (_PAID if key else _FREE)[kind], ({"apikey": key} if key else {})


def _get_json(url, params):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_S)
            if r.status_code == 429 or r.status_code >= 500:
                last = WeatherError(f"weather service answered {r.status_code}")
            elif r.status_code >= 400:
                reason = ""
                try:
                    reason = r.json().get('reason', '')
                except Exception:
                    pass
                raise WeatherError(f"weather service refused the request ({r.status_code}) {reason}".strip())
            else:
                return r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            last = WeatherError(f"weather service not reachable: {e}")
        if attempt < RETRIES - 1:
            time.sleep(0.5 * 2 ** attempt)
    raise last


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_hourly(lat, lon, start_date, end_date):
    """
    Hourly weather at a position for whole days (YYYY-MM-DD, UTC): {'elevation', 'rows': [[iso, temp_c,
    pressure_hpa, wind_ms, wind_from_deg, rh_pct], ...]}. Raises WeatherError (and so is never cached) on failure.
    """
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    recent = pd.Timestamp(end_date) > today - pd.Timedelta(days=ARCHIVE_DELAY_DAYS)
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "wind_speed_unit": "ms", "timezone": "UTC"}
    url, extra = _endpoint("forecast" if recent else "archive")
    if recent:
        days_back = int((today - pd.Timestamp(start_date)).days) + 1
        if days_back > 92:
            raise WeatherError("the forecast service only covers the last 92 days and the archive has no data yet")
        params.update(past_days=max(days_back, 1), forecast_days=1)
    else:
        params.update(start_date=start_date, end_date=end_date)
    params.update(extra)
    data = _get_json(url, params)
    hourly = data.get('hourly') if isinstance(data, dict) else None
    if not hourly or 'time' not in hourly:
        raise WeatherError("no weather data for this place and time")
    unit = str((data.get('hourly_units') or {}).get('wind_speed_10m', 'm/s')).lower()
    to_ms = {'km/h': 1 / 3.6, 'mp/h': 0.44704, 'mph': 0.44704, 'kn': 0.514444}.get(unit, 1.0)

    def col(name):
        return [None if v is None else float(v) for v in hourly.get(name, [None] * len(hourly['time']))]
    wind = [None if v is None else v * to_ms for v in col('wind_speed_10m')]
    rows = [[t, a, p, w, d, h] for t, a, p, w, d, h in zip(
        hourly['time'], col('temperature_2m'), col('surface_pressure'), wind, col('wind_direction_10m'),
        col('relative_humidity_2m'))]
    return {'elevation': data.get('elevation'), 'rows': rows}


def fetch_ride_payload(df):
    """The weather of a whole ride, ready to store: {'source', 'points': [{lat, lon, start, end, elevation,
    rows}], 'override'}. None when the ride has no GPS; raises WeatherError when the service fails."""
    points = sample_points(df)
    if not points:
        return None
    t = _naive_utc(df['time'])
    out = []
    for p in points:
        seg = t.iloc[p['start']:p['end']].dropna()
        if seg.empty:
            continue
        # the hour before and after, so that interpolation has both neighbours
        first = (seg.min() - pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        last = (seg.max() + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        h = fetch_hourly(p['lat'], p['lon'], first, last)
        out.append({**p, 'elevation': h['elevation'], 'rows': h['rows']})
    if not out:
        return None
    return {'source': 'open-meteo', 'points': out, 'override': None}


# ---------------------------------------------------------------------------------------------- applying
def wind_to_uv(speed, deg_from):
    """Velocity of the air (east u, north v) from a speed and the direction it blows FROM."""
    r = np.radians(np.asarray(deg_from, dtype=float))
    s = np.asarray(speed, dtype=float)
    return -s * np.sin(r), -s * np.cos(r)


def uv_to_wind(u, v):
    """(speed, direction it blows from in degrees) of an air velocity."""
    return np.hypot(u, v), (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def pressure_at(p_hpa, grid_elevation, elevation):
    """Station pressure moved from the weather model's elevation to the rider's (standard atmosphere)."""
    if grid_elevation is None:
        return p_hpa
    t0, lapse, expo = 288.15, 0.0065, 5.255
    return p_hpa * ((1 - lapse * np.asarray(elevation, dtype=float) / t0) /
                    (1 - lapse * float(grid_elevation) / t0)) ** expo


MAX_GAP = pd.Timedelta(hours=3)


def _drop_across_gaps(series, valid):
    """NaN where the observations on both sides of a time are more than MAX_GAP apart."""
    if len(valid) == 0:
        return series * np.nan
    t = series.index.to_numpy()
    v = valid.to_numpy()
    nxt = np.searchsorted(v, t, side='left')
    prev_i = np.clip(np.searchsorted(v, t, side='right') - 1, 0, len(v) - 1)
    next_i = np.clip(nxt, 0, len(v) - 1)
    span = pd.to_timedelta(v[next_i] - v[prev_i])
    return series.where(span <= MAX_GAP)


def _interpolate(rows, times):
    """Weather columns at `times` (naive UTC Series) from hourly rows: wind through u/v, gaps stay NaN."""
    w = pd.DataFrame(rows, columns=['time', 'temp_c', 'pressure', 'speed', 'deg', 'rh'])
    w['time'] = pd.to_datetime(w['time'])
    w = w.dropna(subset=['time']).drop_duplicates('time').set_index('time').sort_index()
    w['u'], w['v'] = wind_to_uv(w['speed'].astype(float), w['deg'].astype(float))
    idx = w.index.union(pd.DatetimeIndex(times))
    grid = w[['temp_c', 'pressure', 'u', 'v', 'rh']].astype(float).reindex(idx).interpolate(
        method='time', limit_area='inside')
    grid = grid.reindex(pd.DatetimeIndex(times))
    for c in grid.columns:                      # no values across a hole in the service's data: leave it missing
        grid[c] = _drop_across_gaps(grid[c], w[c].dropna().index)
    speed, deg = uv_to_wind(grid['u'].to_numpy(), grid['v'].to_numpy())
    return pd.DataFrame({'wind_speed': speed, 'wind_deg': deg, 'temp_c': grid['temp_c'].to_numpy(),
                         'pressure': grid['pressure'].to_numpy(), 'relative_humidity': grid['rh'].to_numpy() / 100.0})


def apply_weather(df, payload):
    """
    The ride with wind_speed (m/s), wind_deg (blowing from), temp_c (°C), pressure (hPa at the rider's altitude)
    and relative_humidity (fraction) columns. A recorded device temperature is kept. The payload's manual
    override, if any, replaces the wind. df unchanged when there is nothing to apply.
    """
    if payload is None or df.empty or 'time' not in df.columns:
        return df
    df = df.copy()
    times = _naive_utc(df['time']).reset_index(drop=True)
    cols = {c: np.full(len(df), np.nan) for c in ('wind_speed', 'wind_deg', 'temp_c', 'pressure', 'relative_humidity')}
    for p in payload.get('points', []):
        sl = slice(p['start'], min(p['end'], len(df)))
        part = times.iloc[sl]
        if part.empty or not p.get('rows'):
            continue
        got = _interpolate(p['rows'], part)
        if 'ele' in df.columns:
            ele = pd.to_numeric(df['ele'], errors='coerce').iloc[sl].ffill().bfill()
            if ele.notna().all():
                got['pressure'] = pressure_at(got['pressure'].to_numpy(), p.get('elevation'), ele.to_numpy())
        for c in cols:
            cols[c][sl] = got[c].to_numpy()
    for c, v in cols.items():
        if c == 'temp_c' and 'temp_c' in df.columns and df['temp_c'].notna().any():
            continue                                   # the head unit measured it: better than a model
        df[c] = v
    ov = payload.get('override')
    if ov:
        df['wind_speed'] = float(ov.get('wind_ms', 0.0))
        df['wind_deg'] = float(ov.get('wind_deg', 0.0))
    df.attrs['weather_source'] = 'manual wind' if ov else payload.get('source', 'open-meteo')
    return df


def weather_summary(df):
    """Short text for the ride page ('12.3 °C, wind ~14 km/h'), or None when the ride has no weather."""
    if 'wind_speed' not in df.columns or not df['wind_speed'].notna().any():
        return None
    parts = []
    if 'temp_c' in df.columns and df['temp_c'].notna().any():
        parts.append(f"{df['temp_c'].mean():.1f}°C")
    parts.append(f"wind ~{df['wind_speed'].mean() * 3.6:.1f} km/h")
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------- entry points
def load_or_fetch(df, stored=None, override=None):
    """
    (df with weather, payload, message). Uses the stored payload when there is one (no network), otherwise asks
    the service; the message says why a ride has no weather (None when all went well). `override` is the manual
    wind {'wind_ms', 'wind_deg'} (or None for automatic) and is kept in the returned payload.
    """
    payload, message = stored, None
    if payload is None:
        try:
            payload = fetch_ride_payload(df)
            if payload is None:
                message = "No GPS position in this file: weather not available."
        except WeatherError as e:
            message = f"Weather not available: {e}"
        except Exception as e:                          # nothing in here may stop the ride from opening
            message = f"Weather not available: {e}"
    if override is not None:
        payload = dict(payload or {'source': 'manual', 'points': []}, override=override)
    elif payload is not None and payload.get('override'):
        payload = dict(payload, override=None)
    return apply_weather(df, payload), payload, message
