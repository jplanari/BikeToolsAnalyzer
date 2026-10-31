import numpy as np
import pandas as pd
import pytest
import requests

from src.data import db, weather
from src.physics.aerodyn import get_air_density, _density, _wind_components


def _ride(n=3600, lat=41.4, lon=2.17, start="2026-06-01 08:00:00", dlat=0.0):
    t = pd.date_range(start, periods=n, freq="s")
    return pd.DataFrame({'time': t, 'lat': lat + np.arange(n) * dlat, 'lon': np.full(n, lon),
                         'ele': np.full(n, 100.0), 'speed': np.full(n, 8.0)})


def _rows(day="2026-06-01", speed=5.0, deg=0.0, temp=20.0, p=1000.0, rh=50.0):
    return [[f"{day}T{h:02d}:00", temp, p, speed, deg, rh] for h in range(24)]


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _body(day="2026-06-01"):
    t = [f"{day}T{h:02d}:00" for h in range(24)]
    return {'elevation': 12.0, 'hourly_units': {'wind_speed_10m': 'm/s'},
            'hourly': {'time': t, 'temperature_2m': [15.0] * 24, 'surface_pressure': [1010.0] * 24,
                       'wind_speed_10m': [4.0] * 24, 'wind_direction_10m': [90.0] * 24,
                       'relative_humidity_2m': [60.0] * 24}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)


# 1. rides without GPS, timeouts, status, retries
def test_ride_without_gps_has_no_sample_points_and_does_not_crash():
    assert weather.sample_points(pd.DataFrame({'time': pd.date_range("2026-06-01", periods=10, freq="s")})) == []
    df = _ride(10)
    df['lat'] = np.nan
    assert weather.sample_points(df) == []
    out, payload, msg = weather.load_or_fetch(df)
    assert payload is None and "No GPS" in msg and 'wind_speed' not in out.columns


def test_zero_zero_and_partial_nan_positions_are_ignored():
    df = _ride(100)
    df.loc[:49, ['lat', 'lon']] = np.nan
    df.loc[50:59, ['lat', 'lon']] = 0.0
    pts = weather.sample_points(df)
    assert len(pts) == 1 and pts[0]['lat'] == 41.4 and pts[0]['start'] == 0 and pts[0]['end'] == 100


def test_request_has_timeout_and_retries_server_errors(monkeypatch):
    calls = []

    def fake(url, params=None, timeout=None):
        calls.append(timeout)
        return _Resp(503) if len(calls) < 3 else _Resp(200, _body())
    monkeypatch.setattr(weather.requests, "get", fake)
    assert weather._get_json("u", {})['elevation'] == 12.0
    assert len(calls) == 3 and all(t == weather.TIMEOUT_S for t in calls)


def test_client_error_is_not_retried_and_failures_raise(monkeypatch):
    n = []
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: (n.append(1), _Resp(400, {'reason': 'bad'}))[1])
    with pytest.raises(weather.WeatherError, match="bad"):
        weather._get_json("u", {})
    assert len(n) == 1

    def boom(*a, **k):
        raise requests.exceptions.ConnectTimeout("slow")
    monkeypatch.setattr(weather.requests, "get", boom)
    with pytest.raises(weather.WeatherError):
        weather._get_json("u", {})


def test_unreadable_answer_is_a_message_not_a_crash(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: _Resp(200, None))
    out, payload, msg = weather.load_or_fetch(_ride(60))
    assert payload is None and msg.startswith("Weather not available") and 'wind_speed' not in out.columns


# 2. wind interpolation through u/v
def test_wind_interpolates_through_north_not_south():
    rows = [["2026-06-01T08:00", 20, 1000, 5.0, 350.0, 50], ["2026-06-01T09:00", 20, 1000, 5.0, 10.0, 50]]
    got = weather._interpolate(rows, pd.Series(pd.to_datetime(["2026-06-01 08:30"])))
    d = float(got['wind_deg'].iloc[0])
    assert min(d, 360 - d) < 1.0                       # ~0°, not 180°
    assert 4.9 < float(got['wind_speed'].iloc[0]) < 5.0  # speed dips slightly between the two directions


def test_missing_hours_stay_missing_instead_of_carrying_stale_values():
    rows = _rows()
    for r in rows[10:14]:
        r[3] = r[4] = None
    got = weather._interpolate(rows, pd.Series(pd.to_datetime(["2026-06-01 08:30", "2026-06-01 12:00"])))
    assert got['wind_speed'].iloc[0] == pytest.approx(5.0)
    assert np.isnan(got['wind_speed'].iloc[1])


# 3. archive vs forecast, nothing cached for failures
def test_old_rides_use_archive_recent_use_forecast(monkeypatch):
    seen = []
    monkeypatch.setattr(weather, "_get_json", lambda url, params: (seen.append((url, params)), _body())[1])
    weather.fetch_hourly(41.4, 2.17, "2020-01-01", "2020-01-01")
    assert "archive" in seen[-1][0] and seen[-1][1]['timezone'] == "UTC"
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    weather.fetch_hourly(41.4, 2.17, (today - pd.Timedelta(days=2)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    assert "archive" not in seen[-1][0] and seen[-1][1]['past_days'] == 3


def test_forecast_service_cannot_reach_beyond_92_days_for_unarchived_dates(monkeypatch):
    monkeypatch.setattr(weather, "_get_json", lambda *a: _body())
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    with pytest.raises(weather.WeatherError):
        weather.fetch_hourly(1, 2, (today - pd.Timedelta(days=200)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))


def test_paid_key_switches_to_customer_endpoints(monkeypatch):
    monkeypatch.setenv("OPEN_METEO_API_KEY", "k")
    url, extra = weather._endpoint("archive")
    assert "customer" in url and extra == {"apikey": "k"}
    monkeypatch.delenv("OPEN_METEO_API_KEY")
    assert "customer" not in weather._endpoint("forecast")[0]


# 5. long rides sampled along the route, measured pressure
def test_long_ride_gets_one_sample_per_stretch():
    df = _ride(n=6000, dlat=0.0001)                    # ~11 m per sample: ~66 km north
    pts = weather.sample_points(df)
    assert len(pts) == 3
    assert pts[0]['start'] == 0 and pts[-1]['end'] == 6000
    assert all(a['end'] == b['start'] for a, b in zip(pts, pts[1:]))
    assert pts[0]['lat'] < pts[1]['lat'] < pts[2]['lat']


def test_positions_are_rounded_before_they_are_sent(monkeypatch):
    sent = []
    monkeypatch.setattr(weather, "fetch_hourly", lambda lat, lon, a, b: (sent.append((lat, lon)),
                                                                          {'elevation': 0, 'rows': _rows()})[1])
    weather.fetch_ride_payload(_ride(60, lat=41.123456, lon=2.987654))
    assert sent == [(41.12, 2.99)]


def test_density_uses_the_measured_pressure():
    std = get_air_density(100.0, 15.0, 0.0)
    low = get_air_density(100.0, 15.0, 0.0, pressure_pa=98000.0)
    assert low < std * 0.98
    df = pd.DataFrame({'temp_c': [15.0] * 3, 'pressure': [980.0] * 3})
    assert _density(df, 100.0).iloc[0] == pytest.approx(low)


def test_pressure_moves_to_the_riders_altitude():
    assert weather.pressure_at(1000.0, 0.0, 500.0) < 950.0
    assert weather.pressure_at(1000.0, None, 500.0) == 1000.0


# apply, device temperature, manual override
def test_apply_weather_sets_columns_and_keeps_device_temperature():
    df = _ride(120)
    payload = {'source': 'open-meteo', 'override': None,
               'points': [{'lat': 41.4, 'lon': 2.17, 'start': 0, 'end': 120, 'elevation': 100.0, 'rows': _rows()}]}
    out = weather.apply_weather(df, payload)
    assert out['wind_speed'].iloc[0] == pytest.approx(5.0) and out['wind_deg'].iloc[0] == pytest.approx(0.0)
    assert out['temp_c'].iloc[0] == 20.0 and out['relative_humidity'].iloc[0] == pytest.approx(0.5)
    assert out['pressure'].iloc[0] == pytest.approx(1000.0)
    df['temp_c'] = 31.0
    assert (weather.apply_weather(df, payload)['temp_c'] == 31.0).all()
    assert 'wind_speed' not in df.columns               # the input is not modified


def test_manual_override_replaces_the_wind_and_no_wind_is_zero():
    df = _ride(60)
    df['bearing'] = 0.0
    payload = {'points': [], 'override': {'wind_ms': 6.0, 'wind_deg': 0.0}}
    out = weather.apply_weather(df, payload)
    assert (out['wind_speed'] == 6.0).all() and out.attrs['weather_source'] == 'manual wind'
    head, _ = _wind_components(out)
    assert head.iloc[0] > 0                              # from the north, riding north: head wind
    out2, p2, _ = weather.load_or_fetch(df, None, {'wind_ms': 0.0, 'wind_deg': 0.0})
    assert (out2['wind_speed'] == 0).all() and p2['override'] is not None


def test_summary_text_has_no_nan():
    df = _ride(60)
    assert weather.weather_summary(df) is None
    df['wind_speed'] = np.nan
    assert weather.weather_summary(df) is None
    df['wind_speed'], df['temp_c'] = 3.0, 12.0
    assert weather.weather_summary(df) == "12.0°C, wind ~10.8 km/h"


# 4. stored with the ride, reproducible offline
@pytest.fixture
def path(tmp_path, monkeypatch):
    p = tmp_path / "w.db"
    monkeypatch.setattr(db, "DB_FILE", str(p))
    db._initialised.discard(str(p))
    db.init_db()
    return p


def test_ride_weather_roundtrip_and_cascade(path):
    db.save_user("ann", 70, 250, 180)
    with db.connect() as conn:
        uid = conn.execute("SELECT id FROM users WHERE name='ann'").fetchone()[0]
        conn.execute("INSERT INTO rides (user_id, filename, file_hash) VALUES (?, 'a.gpx', 'h')", (uid,))
        rid = conn.execute("SELECT id FROM rides").fetchone()[0]
    assert db.get_ride_weather(rid) is None
    payload = {'source': 'open-meteo', 'override': None, 'points': [{'start': 0, 'end': 5, 'rows': _rows()}]}
    db.save_ride_weather(rid, payload)
    assert db.get_ride_weather(rid) == payload
    db.save_ride_weather(rid, {**payload, 'override': {'wind_ms': 1.0, 'wind_deg': 90.0}})
    assert db.get_ride_weather(rid)['override']['wind_ms'] == 1.0
    with db.connect() as conn:
        conn.execute("DELETE FROM rides WHERE id = ?", (rid,))
    assert db.get_ride_weather(rid) is None


def test_stored_weather_needs_no_network(monkeypatch):
    def offline(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(weather.requests, "get", offline)
    payload = {'source': 'open-meteo', 'override': None,
               'points': [{'start': 0, 'end': 60, 'elevation': 100.0, 'rows': _rows()}]}
    out, p, msg = weather.load_or_fetch(_ride(60), payload)
    assert msg is None and out['wind_speed'].iloc[0] == pytest.approx(5.0)


def test_weekly_report_ride_gets_stored_weather(path, monkeypatch):
    from src.data import weekly_report
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
    db.save_user("ann", 70, 250, 180)
    with db.connect() as conn:
        uid = conn.execute("SELECT id FROM users WHERE name='ann'").fetchone()[0]
        conn.execute("INSERT INTO rides (user_id, filename, file_hash) VALUES (?, 'a.gpx', 'h')", (uid,))
        rid = conn.execute("SELECT id FROM rides").fetchone()[0]
    db.save_ride_weather(rid, {'source': 'open-meteo', 'override': {'wind_ms': 7.0, 'wind_deg': 180.0},
                               'points': [{'start': 0, 'end': 60, 'elevation': 100.0, 'rows': _rows()}]})
    out = weekly_report._with_weather(_ride(60), rid)
    assert (out['wind_speed'] == 7.0).all() and 'bearing' in out.columns
    # no stored weather and no network: the ride is still reported, without wind
    assert 'wind_speed' not in weekly_report._with_weather(_ride(60), rid + 99).columns
