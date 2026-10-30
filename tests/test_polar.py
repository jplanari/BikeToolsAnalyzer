import pandas as pd
import pytest

from src.data import db
from src.data import polar as pl
from src.data import polar_import as pi
from src.data import strava as sv
from tests.test_strava import (make_streams, real_distances, temp_db, FakeResponse,  # noqa: F401 (fixtures)
                               _upload_ride)

CFG = pl.PolarConfig("cid", "secret", "http://localhost:8501/")


class Resp(FakeResponse):
    """requests.Response also has .content (the raw bytes)."""
    content = b"FITDATA"


def exercise(eid, local_start, sport="CYCLING", detailed="ROAD_BIKING", offset=120, duration="PT30M"):
    return {"id": eid, "start_time": local_start, "start_time_utc_offset": offset, "sport": sport,
            "detailed_sport_info": detailed, "duration": duration, "distance": 14400.0}


@pytest.fixture
def connected(temp_db):
    db.save_polar_auth("Rider", {"polar_user_id": "77", "access_token": "TOK"})


@pytest.fixture
def fake_polar(monkeypatch):
    """Two rides, a run and one that matches an upload; the FIT parser is replaced by Strava-style streams."""
    calls = {"download": []}
    items = [exercise("aaa111", "2026-09-01T09:00:00"), exercise("bbb222", "2026-09-03T09:00:00",
                                                                 sport="RUNNING", detailed="RUNNING"),
             exercise("ccc333", "2026-09-07T09:00:00", detailed="INDOOR_CYCLING"),
             exercise("ddd444", "2026-09-09T09:00:30")]
    starts = {e["id"]: pl.start_utc(e) for e in items}

    def fake_download(token, eid):
        calls["download"].append(eid)
        return b"FIT:" + eid.encode()

    def fake_prepare(filename, payload):
        assert filename.endswith(".fit") and payload.startswith(b"FIT:")
        start = f"{starts[filename[6:-4]]:%Y-%m-%dT%H:%M:%SZ}"
        df = sv.streams_to_dataframe(sv.build_ride_payload(
            {"id": 1, "start_date": start, "start_date_local": start}, make_streams(n=1800)))
        return df, 14_400.0, 18.0

    monkeypatch.setattr(pl, "list_exercises", lambda token: items)
    monkeypatch.setattr(pl, "download_fit", fake_download)
    monkeypatch.setattr("src.data.pipeline.prepare_ride", fake_prepare)
    return calls


# --- client ----------------------------------------------------------------------

def test_authorize_url():
    url = pl.authorize_url(CFG, "polar.xyz")
    assert url.startswith("https://flow.polar.com/oauth2/authorization?")
    assert "response_type=code" in url and "client_id=cid" in url and "state=polar.xyz" in url


def test_exchange_code_uses_basic_auth_and_returns_the_user_id(monkeypatch):
    seen = {}

    def post(url, auth=None, data=None, **k):
        seen.update(url=url, auth=auth, data=data)
        return FakeResponse(payload={"access_token": "A", "token_type": "bearer", "expires_in": 31535999,
                                     "x_user_id": 77})
    monkeypatch.setattr(pl.requests, "post", post)
    auth = pl.exchange_code(CFG, "code1")
    assert seen["url"] == pl.TOKEN_URL and seen["auth"] == ("cid", "secret")
    assert seen["data"]["grant_type"] == "authorization_code" and seen["data"]["code"] == "code1"
    assert auth == {"access_token": "A", "polar_user_id": "77"}


def test_token_errors(monkeypatch):
    for code, exc in ((401, pl.PolarAuthError), (429, pl.PolarRateLimitError), (500, pl.PolarError)):
        monkeypatch.setattr(pl.requests, "post", lambda *a, _c=code, **k: FakeResponse(_c, {}))
        with pytest.raises(exc):
            pl.exchange_code(CFG, "c")


def test_register_user_accepts_already_registered(monkeypatch):
    monkeypatch.setattr(pl.requests, "request", lambda *a, **k: FakeResponse(409, {}))
    assert pl.register_user("A", "biketools-1") is None
    monkeypatch.setattr(pl.requests, "request", lambda *a, **k: Resp(200, {"polar-user-id": 77}))
    assert pl.register_user("A", "biketools-1")["polar-user-id"] == 77
    monkeypatch.setattr(pl.requests, "request", lambda *a, **k: FakeResponse(400, {"error": "bad"}))
    with pytest.raises(pl.PolarError):
        pl.register_user("A", "biketools-1")


def test_api_errors(monkeypatch):
    for code, exc in ((401, pl.PolarAuthError), (403, pl.PolarAuthError), (404, pl.PolarNotFound),
                      (429, pl.PolarRateLimitError), (500, pl.PolarError)):
        monkeypatch.setattr(pl.requests, "request", lambda *a, _c=code, **k: FakeResponse(_c, {}))
        with pytest.raises(exc):
            pl.list_exercises("A")


def test_list_and_download(monkeypatch):
    calls = []

    def request(method, url, headers=None, **k):
        calls.append((method, url, headers["Authorization"]))
        return Resp(200, [exercise("x1", "2026-09-01T09:00:00")])
    monkeypatch.setattr(pl.requests, "request", request)
    assert [e["id"] for e in pl.list_exercises("TOK")] == ["x1"]
    assert pl.download_fit("TOK", "x1") == b"FITDATA"
    assert calls[0] == ("GET", "https://www.polaraccesslink.com/v3/exercises", "Bearer TOK")
    assert calls[1][1] == "https://www.polaraccesslink.com/v3/exercises/x1/fit"


def test_summaries():
    e = exercise("a", "2026-09-01T09:00:00", offset=120, duration="PT2H44M")
    assert pl.start_utc(e) == pd.Timestamp("2026-09-01 07:00:00")
    assert pl.duration_minutes(e) == 164 and pl.duration_minutes({"duration": "PT45M30S"}) == 46
    assert pl.duration_minutes({}) == 0
    assert pl.is_cycling(e) and pl.is_cycling(exercise("b", "2026-09-01T09:00:00", sport="OTHER", detailed="INDOOR_CYCLING"))
    assert not pl.is_cycling(exercise("c", "2026-09-01T09:00:00", sport="RUNNING", detailed="RUNNING"))
    assert pl.ride_filename("aB3x") == "polar_aB3x.fit" and db.polar_id_from_filename("polar_aB3x.fit") == "aB3x"
    assert db.polar_id_from_filename("wahoo_4.fit") is None


# --- storage and import ---------------------------------------------------------------

def test_auth_storage_round_trip(connected):
    assert db.get_polar_auth("Rider")["access_token"] == "TOK"
    db.delete_polar_auth("Rider")
    assert db.get_polar_auth("Rider") is None


def test_connect_registers_the_rider_and_stores_the_token(temp_db, monkeypatch):
    registered = {}
    monkeypatch.setattr(pl, "exchange_code", lambda cfg, code: {"access_token": "T", "polar_user_id": "9"})
    monkeypatch.setattr(pl, "register_user", lambda tok, member: registered.update(tok=tok, member=member))
    pi.connect(CFG, "Rider", "code")
    assert registered["tok"] == "T" and registered["member"].startswith("biketools-")
    assert db.get_polar_auth("Rider") == {"user_id": db.get_user_data("Rider")["id"], "polar_user_id": "9",
                                          "access_token": "T"}


def test_recent_rides_are_cycling_only_newest_first(connected, fake_polar):
    assert [e["id"] for e in pi.list_recent_rides("Rider")] == ["ddd444", "ccc333", "aaa111"]


def test_sync_imports_new_rides_once(connected, fake_polar):
    _upload_ride("2026-09-09 07:00:00")                          # same ride as ddd444, already uploaded
    res = pi.sync_new_rides("Rider", ftp=250)
    assert (res.found, res.imported, res.duplicates_of_uploads) == (3, 2, 1) and not res.failed
    assert fake_polar["download"] == ["aaa111", "ccc333"]
    assert db.get_imported_polar_ids("Rider") == {"aaa111", "ccc333"}
    rides = {r["filename"]: r for r in db.get_user_rides("Rider")}
    assert rides["polar_aaa111.fit"]["norm_power"] == 200
    assert db.get_ride_file(rides["polar_aaa111.fit"]["id"])[1] == b"FIT:aaa111"      # the FIT file is kept

    fake_polar["download"].clear()
    again = pi.sync_new_rides("Rider", ftp=250)
    assert (again.imported, again.already_imported) == (0, 2) and fake_polar["download"] == []


def test_sync_stops_at_the_rate_limit(connected, fake_polar, monkeypatch):
    def limited(token, eid):
        raise pl.PolarRateLimitError("Polar rate limit reached; try again in a few minutes.")
    monkeypatch.setattr(pl, "download_fit", limited)
    res = pi.sync_new_rides("Rider", ftp=250)
    assert res.imported == 0 and "rate limit" in res.stopped_reason


def test_not_connected(temp_db):
    with pytest.raises(pl.PolarAuthError):
        pi.list_recent_rides("Rider")
