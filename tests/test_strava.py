import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import db
from src.data import strava as sv
from src.data import strava_import as si


CFG = sv.StravaConfig(client_id="1", client_secret="s", redirect_uri="http://localhost:8501/")


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def make_streams(n=1800, power=200.0, hr=140.0, with_gps=True):
    t = list(range(n))
    s = {
        "time": {"data": t},
        "distance": {"data": [i * 8.0 for i in t]},
        "altitude": {"data": [100 + i * 0.01 for i in t]},
        "heartrate": {"data": [hr] * n},
        "watts": {"data": [power] * n},
        "cadence": {"data": [90] * n},
        "velocity_smooth": {"data": [8.0] * n},
    }
    if with_gps:
        # ~8 m/s heading north (1 deg lat ≈ 111 km)
        s["latlng"] = {"data": [[41.0 + i * 8.0 / 111_000, 2.0] for i in t]}
    return s


def activity(aid, start="2026-10-01T08:00:00Z", sport="Ride", **kw):
    a = {"id": aid, "name": f"Ride {aid}", "sport_type": sport, "type": sport,
         "start_date": start, "manual": False, "distance": 14_000}
    a.update(kw)
    return a


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    return tmp_path / "test.db"


@pytest.fixture
def connected(temp_db):
    db.save_strava_auth("Rider", {"athlete_id": 7, "athlete_name": "Pep",
                                  "access_token": "A", "refresh_token": "R",
                                  "expires_at": 4_000_000_000})


# --- OAuth -------------------------------------------------------------------

def test_authorize_url_requests_activity_scope():
    url = sv.authorize_url(CFG, "xyz")
    assert url.startswith("https://www.strava.com/oauth/authorize?")
    assert "scope=read%2Cactivity%3Aread_all" in url
    assert "state=xyz" in url and "response_type=code" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8501%2F" in url


def test_granted_scope_check():
    assert sv.granted_scope_ok("read,activity:read_all")
    assert sv.granted_scope_ok("read,activity:read")
    assert not sv.granted_scope_ok("read")
    assert not sv.granted_scope_ok(None)


def test_exchange_code(monkeypatch):
    sent = {}

    def fake_post(url, data=None, timeout=None, **kw):
        sent.update(data)
        return FakeResponse(payload={"access_token": "A", "refresh_token": "R", "expires_at": 123,
                                     "athlete": {"id": 7, "firstname": "Pep", "lastname": "P"}})
    monkeypatch.setattr(sv.requests, "post", fake_post)
    auth = sv.exchange_code(CFG, "code123")
    assert sent["grant_type"] == "authorization_code" and sent["code"] == "code123"
    assert auth == {"athlete_id": 7, "athlete_name": "Pep P", "access_token": "A",
                    "refresh_token": "R", "expires_at": 123}


def test_exchange_code_rejected(monkeypatch):
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResponse(400, {"message": "Bad"}))
    with pytest.raises(sv.StravaAuthError):
        sv.exchange_code(CFG, "bad")


def test_refresh_not_needed_makes_no_request(monkeypatch):
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: pytest.fail("should not refresh"))
    auth = {"access_token": "A", "refresh_token": "R", "expires_at": 10_000}
    assert sv.refresh_if_needed(CFG, auth, now=10_000 - 3600) == (auth, False)


def test_refresh_rotates_token_and_is_persisted(connected, monkeypatch):
    # Expire the stored token, then refresh through the import helper
    db.save_strava_auth("Rider", {"access_token": "A", "refresh_token": "R", "expires_at": 0})
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResponse(
        payload={"access_token": "A2", "refresh_token": "R2", "expires_at": 4_000_000_000}))
    auth = si.get_valid_auth(CFG, "Rider")
    assert (auth["access_token"], auth["refresh_token"]) == ("A2", "R2")
    stored = db.get_strava_auth("Rider")
    assert stored["refresh_token"] == "R2"       # rotated token saved
    assert stored["athlete_name"] == "Pep"       # kept across refresh


def test_oauth_state_is_single_use(temp_db):
    state = db.create_oauth_state("Rider")
    assert db.consume_oauth_state(state) == "Rider"
    assert db.consume_oauth_state(state) is None
    assert db.consume_oauth_state("forged") is None


def test_rate_usage_headers():
    u = sv.RateUsage()
    u.update({"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "50,300",
              "X-ReadRateLimit-Limit": "100,1000", "X-ReadRateLimit-Usage": "97,400"})
    assert (u.limit_15min, u.used_15min, u.used_daily) == (100, 97, 400)
    assert u.remaining() == 3


def test_api_errors(monkeypatch):
    for status, exc in [(401, sv.StravaAuthError), (429, sv.StravaRateLimitError), (500, sv.StravaError)]:
        monkeypatch.setattr(sv.requests, "get", lambda *a, _s=status, **k: FakeResponse(_s))
        with pytest.raises(exc):
            sv.api_get("/athlete", "A")


# --- Streams -> DataFrame ------------------------------------------------------

def test_payload_roundtrip_to_dataframe():
    payload = sv.build_ride_payload(activity(1), make_streams(n=10))
    assert payload == sv.build_ride_payload(activity(1), make_streams(n=10))  # deterministic

    df = sv.streams_to_dataframe(payload)
    assert list(df.columns[:7]) == ["time", "lat", "lon", "ele", "hr", "power", "cadence"]
    assert df["time"].iloc[0] == pd.Timestamp("2026-10-01 08:00:00")  # naive UTC
    assert df["time"].iloc[-1] == pd.Timestamp("2026-10-01 08:00:09")
    assert df["power"].tolist() == [200.0] * 10
    assert df["lat"].iloc[0] == pytest.approx(41.0)


def test_indoor_ride_without_gps_or_hr():
    s = make_streams(n=5, with_gps=False)
    del s["heartrate"]
    df = sv.streams_to_dataframe(sv.build_ride_payload(activity(2, trainer=True), s))
    assert df["lat"].isna().all() and df["hr"].isna().all()
    assert df["power"].notna().all()


def test_filename_and_strava_id():
    assert sv.ride_filename(123) == "strava_123.json"
    assert db.strava_id_from_filename("strava_123.json") == 123
    assert db.strava_id_from_filename("morning.gpx") is None


# --- History import ------------------------------------------------------------

@pytest.fixture(autouse=True)
def real_distances(monkeypatch):
    """When geographiclib is stubbed (local runs without it), use plain haversine."""
    from unittest.mock import MagicMock
    from src.data import gpx
    if isinstance(gpx.Geodesic, MagicMock):
        def hav(lat1, lon1, lat2, lon2):
            if np.isnan([lat1, lon1, lat2, lon2]).any():
                return 0.0
            p1, p2 = np.radians(lat1), np.radians(lat2)
            dp, dl = p2 - p1, np.radians(lon2 - lon1)
            a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
            return 2 * 6_371_000 * np.arcsin(np.sqrt(a))
        monkeypatch.setattr(gpx, "haversine_dist", hav)


@pytest.fixture
def fake_strava(monkeypatch):
    calls = {"streams": []}
    acts = [
        activity(10, "2026-09-01T07:00:00Z"),
        activity(11, "2026-09-03T07:00:00Z", sport="Run"),            # not cycling
        activity(12, "2026-09-05T07:00:00Z", manual=True),            # no streams
        activity(13, "2026-09-07T07:00:00Z", sport="VirtualRide"),
        activity(14, "2026-09-09T07:00:30Z"),                         # matches an upload
    ]

    def fake_list(token, after=None, before=None, page=1, per_page=200, usage=None):
        return acts if page == 1 else []

    def fake_streams(token, aid, usage=None):
        calls["streams"].append(aid)
        if usage is not None:
            usage.update({"X-ReadRateLimit-Limit": "100,1000",
                          "X-ReadRateLimit-Usage": f"{len(calls['streams'])},{len(calls['streams'])}"})
        return make_streams(n=1800)

    monkeypatch.setattr(sv, "list_activities", fake_list)
    monkeypatch.setattr(sv, "get_streams", fake_streams)
    return calls


def _upload_ride(start):
    df = pd.DataFrame({"time": pd.date_range(start, periods=60, freq="1s")})
    meta = db.build_ride_metadata(df, 10_000, 50, 25, 200, 40, 0.8, filename="upload.gpx")
    ok, msg = db.save_ride("Rider", "upload.gpx", b"gpx-bytes", meta, {})
    assert ok, msg


def test_import_history(connected, fake_strava):
    _upload_ride("2026-09-09 07:00:00")
    res = si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)

    assert (res.found, res.imported, res.duplicates_of_uploads) == (3, 2, 1)
    assert res.failed == [] and res.stopped_reason == ""
    assert fake_strava["streams"] == [10, 13]
    assert db.get_imported_strava_ids("Rider") == {10, 13}

    rides = {r["filename"]: r for r in db.get_user_rides("Rider")}
    r = rides["strava_10.json"]
    assert r["date_time"] == "2026-09-01T07:00:00"
    assert r["distance_km"] == pytest.approx(14.4, rel=0.02)   # 1800 s at 8 m/s
    # ~30 min at 200 W with FTP 250: IF 0.8 -> TSS = 0.5 h * 0.64 * 100 = 32
    # (duration is last-first sample = 1799 s; TSS is stored rounded)
    assert r["tss"] == round(1799 * 200 * 0.8 / (250 * 3600) * 100) == 32
    assert r["norm_power"] == 200
    assert db.get_user_best_power("Rider")[300] == pytest.approx(200.0)


def test_import_history_is_resumable(connected, fake_strava):
    si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    fake_strava["streams"].clear()
    res = si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.imported == 0 and res.already_imported == 3
    assert fake_strava["streams"] == []


def test_import_history_pauses_before_rate_limit(connected, fake_strava):
    usage = sv.RateUsage()
    usage.update({"X-ReadRateLimit-Limit": "100,1000", "X-ReadRateLimit-Usage": "95,200"})
    res = si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250, usage=usage)
    assert res.imported == 0 and res.remaining == 3
    assert "rate limit" in res.stopped_reason
    assert fake_strava["streams"] == []


def test_import_history_stops_on_429(connected, monkeypatch, fake_strava):
    def limited(*a, **k):
        raise sv.StravaRateLimitError("Strava rate limit reached; try again in 15 minutes.")
    monkeypatch.setattr(sv, "get_streams", limited)
    res = si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.imported == 0 and "15 minutes" in res.stopped_reason


def test_import_requires_connection(temp_db):
    with pytest.raises(sv.StravaAuthError):
        si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)


def test_same_activity_twice_is_rejected(connected):
    payload = sv.build_ride_payload(activity(99), make_streams(n=60))
    df = sv.streams_to_dataframe(payload)
    meta = db.build_ride_metadata(df, 500, 5, 25, 200, 5, 0.8, filename="strava_99.json")
    assert db.save_ride("Rider", "strava_99.json", payload, meta, {})[0]
    ok, msg = db.save_ride("Rider", "strava_99.json", payload + b" ", meta, {})
    assert not ok and "already exists" in msg


def test_sync_starts_a_few_days_before_the_latest_imported_ride(connected, fake_strava, monkeypatch):
    si.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)         # newest imported: 14 on 09 Sep
    seen = {}

    def spy(token, after=None, before=None, page=1, per_page=200, usage=None):
        seen["after"] = pd.Timestamp(after, unit="s")
        return [activity(10, "2026-09-01T07:00:00Z"), activity(20, "2026-09-12T07:00:00Z")] if page == 1 else []

    monkeypatch.setattr(sv, "list_activities", spy)
    fake_strava["streams"].clear()
    res, since = si.sync_new_rides(CFG, "Rider", ftp=250)

    assert since == pd.Timestamp("2026-09-06 07:00:30")                # 09 Sep minus the 3-day margin
    assert seen["after"] == pd.Timestamp("2026-09-06 07:00:30")
    assert (res.imported, res.already_imported) == (1, 1)              # only the new ride is downloaded
    assert fake_strava["streams"] == [20]
    assert db.latest_strava_ride_date("Rider") == pd.Timestamp("2026-09-12 07:00:00")


def test_first_sync_looks_back_30_days(connected, fake_strava, monkeypatch):
    seen = {}
    monkeypatch.setattr(sv, "list_activities", lambda token, after=None, **k: seen.setdefault("after", after) and [])
    now = pd.Timestamp("2026-10-28 12:00:00")
    res, since = si.sync_new_rides(CFG, "Rider", ftp=250, now=now)
    assert since == now - pd.Timedelta(days=30) and res.found == 0
    assert db.latest_strava_ride_date("Rider") is None


def test_rides_between_asks_for_the_whole_end_day_and_lists_newest_first(connected, monkeypatch):
    seen = {}

    def spy(token, after=None, before=None, page=1, per_page=200, usage=None):
        seen.update(after=pd.Timestamp(after, unit="s"), before=pd.Timestamp(before, unit="s"))
        return [activity(1, "2026-09-02T07:00:00Z"), activity(2, "2026-09-10T07:00:00Z")]

    monkeypatch.setattr(sv, "list_activities", spy)
    got = si.list_rides_between(CFG, "Rider", date(2026, 9, 1), date(2026, 9, 10))
    assert [a["id"] for a in got] == [2, 1]
    assert seen["after"] == pd.Timestamp("2026-09-01") and seen["before"] == pd.Timestamp("2026-09-11")
