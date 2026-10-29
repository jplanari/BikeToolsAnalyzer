from datetime import date

import pandas as pd
import pytest

from src.data import db
from src.data import strava as sv
from src.data import wahoo as wh
from src.data import wahoo_import as wi
from tests.test_strava import (make_streams, real_distances, temp_db, FakeResponse,  # noqa: F401 (fixtures)
                               _upload_ride)

CFG = wh.WahooConfig("cid", "secret", "http://localhost:8501/")


def workout(wid, starts, wtype=0, name=None):
    return {"id": wid, "name": name or f"Ride {wid}", "starts": starts, "minutes": 30,
            "workout_type_id": wtype, "workout_summary": None}


@pytest.fixture
def connected(temp_db):
    db.save_wahoo_auth("Rider", {"wahoo_user_id": 5, "wahoo_name": "Pep",
                                 "access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000})


@pytest.fixture
def fake_wahoo(monkeypatch):
    """Three cycling workouts, one run, one matching an upload; FIT parsing is replaced by Strava-style streams."""
    calls = {"detail": [], "download": []}
    items = [workout(10, "2026-09-01T07:00:00.000Z"), workout(11, "2026-09-03T07:00:00.000Z", wtype=1),
             workout(13, "2026-09-07T07:00:00.000Z"), workout(14, "2026-09-09T07:00:30.000Z")]

    def fake_list(token, page=1, per_page=100, usage=None):
        return {"workouts": items if page == 1 else [], "total": len(items), "page": page,
                "per_page": per_page, "order": "descending", "sort": "starts"}

    def fake_detail(token, wid, usage=None):
        calls["detail"].append(wid)
        if usage is not None:
            usage.update({"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": str(200 - len(calls["detail"]))})
        return {"id": wid, "workout_summary": {"file": {"url": f"https://cdn/{wid}.fit"}}}

    def fake_download(url):
        calls["download"].append(url)
        return b"FIT:" + url.encode()

    def fake_prepare(filename, payload):          # a FIT parser stand-in
        assert filename.endswith(".fit") and payload.startswith(b"FIT:")
        start = {w["id"]: w["starts"] for w in items}[int(filename[6:-4])]
        df = sv.streams_to_dataframe(sv.build_ride_payload(
            {"id": 1, "start_date": start, "start_date_local": start}, make_streams(n=1800)))
        return df, 14_400.0, 18.0

    monkeypatch.setattr(wh, "list_workouts", fake_list)
    monkeypatch.setattr(wh, "get_workout", fake_detail)
    monkeypatch.setattr(wh, "download_fit", fake_download)
    monkeypatch.setattr("src.data.pipeline.prepare_ride", fake_prepare)
    return calls


# --- client ----------------------------------------------------------------------

def test_authorize_url_asks_for_workouts_and_offline_access():
    url = wh.authorize_url(CFG, "wahoo.xyz")
    assert url.startswith("https://api.wahooligan.com/oauth/authorize?")
    assert "workouts_read" in url and "offline_data" in url and "state=wahoo.xyz" in url


def test_exchange_code_reads_name_and_expiry(monkeypatch):
    sent = {}

    def post(url, data=None, **k):
        sent.update(data)
        return FakeResponse(payload={"access_token": "A", "refresh_token": "R", "expires_in": 7200,
                                     "created_at": 1000})
    monkeypatch.setattr(wh.requests, "post", post)
    monkeypatch.setattr(wh, "api_get", lambda path, tok, **k: {"id": 5, "first": "Pep", "last": "P"})
    auth = wh.exchange_code(CFG, "code1")
    assert sent["grant_type"] == "authorization_code" and sent["redirect_uri"] == CFG.redirect_uri
    assert auth["expires_at"] == 8200 and auth["wahoo_name"] == "Pep P" and auth["wahoo_user_id"] == 5


def test_refresh_token_is_single_use_and_saved_at_once(connected, monkeypatch):
    db.save_wahoo_auth("Rider", {"access_token": "A", "refresh_token": "R", "expires_at": 0})
    monkeypatch.setattr(wh.requests, "post", lambda *a, **k: FakeResponse(
        payload={"access_token": "A2", "refresh_token": "R2", "expires_in": 7200}))
    auth = wi.get_valid_auth(CFG, "Rider")
    assert (auth["access_token"], auth["refresh_token"]) == ("A2", "R2")
    stored = db.get_wahoo_auth("Rider")
    assert stored["refresh_token"] == "R2" and stored["wahoo_name"] == "Pep"


def test_api_errors(monkeypatch):
    for code, exc in ((401, wh.WahooAuthError), (403, wh.WahooAuthError), (429, wh.WahooRateLimitError),
                      (500, wh.WahooError)):
        monkeypatch.setattr(wh.requests, "request", lambda *a, _c=code, **k: FakeResponse(_c, {}))
        with pytest.raises(exc):
            wh.api_get("/workouts", "A")


def test_rate_usage_reads_headers():
    u = wh.RateUsage()
    u.update({"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": "17", "X-RateLimit-Reset": "123"})
    assert (u.limit, u.remaining, u.seen) == (200, 17, True)


def test_filename_round_trip():
    assert wh.ride_filename(42) == "wahoo_42.fit" and db.wahoo_id_from_filename("wahoo_42.fit") == 42
    assert db.wahoo_id_from_filename("strava_42.json") is None and db.wahoo_id_from_filename("ride.fit") is None


# --- import ----------------------------------------------------------------------

def test_import_history(connected, fake_wahoo):
    _upload_ride("2026-09-09 07:00:00")
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)

    assert (res.found, res.imported, res.duplicates_of_uploads) == (3, 2, 1)   # the run is not listed as cycling
    assert res.failed == [] and res.stopped_reason == ""
    assert fake_wahoo["detail"] == [10, 13]
    assert db.get_imported_wahoo_ids("Rider") == {10, 13}
    assert db.latest_wahoo_ride_date("Rider") == pd.Timestamp("2026-09-07 07:00:00")

    rides = {r["filename"]: r for r in db.get_user_rides("Rider")}
    assert rides["wahoo_10.fit"]["norm_power"] == 200
    name, content = db.get_ride_file(rides["wahoo_10.fit"]["id"])
    assert name == "wahoo_10.fit" and content == b"FIT:https://cdn/10.fit"      # original file kept


def test_import_is_idempotent(connected, fake_wahoo):
    wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    fake_wahoo["detail"].clear()
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert (res.imported, res.already_imported) == (0, 3) and fake_wahoo["detail"] == []


def test_import_stops_at_the_rate_limit_and_resumes(connected, fake_wahoo, monkeypatch):
    real = wh.get_workout

    def tight(token, wid, usage=None):
        out = real(token, wid, usage)
        usage.update({"X-RateLimit-Limit": "25", "X-RateLimit-Remaining": "2"})
        return out
    monkeypatch.setattr(wh, "get_workout", tight)
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.imported == 1 and "rate limit" in res.stopped_reason and res.remaining == 2
    monkeypatch.setattr(wh, "get_workout", real)
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.imported == 2 and db.get_imported_wahoo_ids("Rider") == {10, 13, 14}


def test_scheduled_workout_without_a_file_is_counted_not_failed(connected, fake_wahoo, monkeypatch):
    monkeypatch.setattr(wh, "get_workout", lambda t, wid, usage=None: {"id": wid, "workout_summary": None})
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert (res.not_completed, res.imported, res.failed) == (3, 0, [])


def test_unparsable_file_is_reported_and_the_rest_continue(connected, fake_wahoo, monkeypatch):
    good = __import__("src.data.pipeline", fromlist=["prepare_ride"]).prepare_ride

    def flaky(filename, payload):
        if filename == "wahoo_10.fit":
            raise ValueError("no timestamped samples")
        return good(filename, payload)
    monkeypatch.setattr("src.data.pipeline.prepare_ride", flaky)
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.imported == 2 and len(res.failed) == 1 and "wahoo_10" not in db.get_imported_wahoo_ids("Rider")


def test_list_reads_pages_until_one_is_older_than_since(monkeypatch):
    pages = {1: [workout(i, f"2026-10-{30 - i:02d}T07:00:00Z") for i in range(1, 4)],
             2: [workout(i, f"2026-08-{30 - i:02d}T07:00:00Z") for i in range(4, 7)],
             3: pytest.fail}
    monkeypatch.setattr(wh, "list_workouts", lambda tok, page=1, per_page=100, usage=None: {
        "workouts": pages[page], "per_page": 3, "order": "descending"})
    got = wi.list_cycling_since({"access_token": "A"}, date(2026, 10, 1))
    assert [w["id"] for w in got] == [3, 2, 1]                  # oldest first, page 3 never requested


def test_sync_starts_a_few_days_before_the_latest_wahoo_ride(connected, fake_wahoo):
    wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    fake_wahoo["detail"].clear()
    res, since = wi.sync_new_rides(CFG, "Rider", ftp=250)
    assert since == pd.Timestamp("2026-09-06 07:00:30") and res.imported == 0


def test_first_sync_looks_back_30_days(connected, fake_wahoo):
    now = pd.Timestamp("2026-10-28 12:00:00")
    res, since = wi.sync_new_rides(CFG, "Rider", ftp=250, now=now)
    assert since == now - pd.Timedelta(days=30) and res.found == 0


def test_future_and_summary_less_planned_entries_are_not_even_fetched(connected, fake_wahoo, monkeypatch):
    soon = (pd.Timestamp.now("UTC") + pd.Timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = dict(workout(10, "2026-09-01T07:00:00Z"), workout_summary={"file": {"url": "https://cdn/10.fit"}})
    skipped = dict(workout(13, "2026-09-07T07:00:00Z"),
                   workout_summary={"duration_active_accum": "0", "file": None})      # list already says: not ridden
    items = [workout(14, soon), done, skipped]
    monkeypatch.setattr(wh, "list_workouts", lambda tok, page=1, per_page=100, usage=None: {
        "workouts": items, "per_page": 100, "order": "descending"})
    res = wi.import_history(CFG, "Rider", date(2026, 8, 1), ftp=250)
    assert res.found == 1 and res.imported == 1 and fake_wahoo["detail"] == [10]


def test_recent_list_hides_planned_workouts_that_were_never_ridden(connected, monkeypatch):
    soon = (pd.Timestamp.now("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = [workout(30, soon),                                              # future
             dict(workout(31, "2026-10-20T07:00:00Z"), workout_token="biketools-1-5", plan_id=9),   # missed
             dict(workout(32, "2026-10-19T07:00:00Z"), workout_token="biketools-1-4", plan_id=8),   # ridden
             workout(33, "2026-10-18T07:00:00Z")]                            # plain recorded ride
    monkeypatch.setattr(wh, "list_workouts", lambda tok, page=1, per_page=100, usage=None: {"workouts": items})
    asked = []

    def detail(tok, wid, usage=None):
        asked.append(wid)
        return {"id": wid, "workout_summary": {"file": {"url": "u"}} if wid == 32 else None}
    monkeypatch.setattr(wh, "get_workout", detail)
    got = wi.list_recent_rides(CFG, "Rider")
    assert [w["id"] for w in got] == [32, 33]
    assert asked == [31, 32]                      # only plan-linked entries cost a request
