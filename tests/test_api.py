"""The phone API: tokens, payloads and the WSGI app (called directly, no sockets)."""
import io
import json

import pandas as pd
import pytest

from src.api import app as api
from src.api import payloads
from src.data import db
from tests.test_nutrition import seeded  # noqa: F401 (fixture: Rider with four Strava rides)
from tests.test_strava import real_distances  # noqa: F401 (fixture the seeded one needs)


class Client:
    def __init__(self, token=None, **env):
        self.token, self.env = token, env

    def get(self, path, token="default", method="GET", headers=None, remote="10.0.0.1", body=b""):
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)
        env = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": query, "REMOTE_ADDR": remote,
               "wsgi.input": io.BytesIO(body), **self.env}
        tok = self.token if token == "default" else token
        if tok:
            env["HTTP_AUTHORIZATION"] = f"Bearer {tok}"
        for k, v in (headers or {}).items():
            env[k.upper().replace("-", "_") if k.lower() == "content-length" else "HTTP_" + k.upper().replace("-", "_")] = v
        out = {}

        def start(status, hdrs):
            out["status"], out["headers"] = status, dict(hdrs)
        body = b"".join(api.application(env, start))
        out["body"] = body
        out["code"] = int(out["status"].split()[0])
        if out["headers"].get("Content-Type", "").startswith("application/json") and body:
            out["json"] = json.loads(body)
        return out


@pytest.fixture(autouse=True)
def fresh_throttle(monkeypatch):
    monkeypatch.setattr(api, "_throttle", api.Throttle())
    payloads._prepared.cache_clear()


@pytest.fixture
def client(seeded):
    token, _ = db.create_api_token("Rider", "test phone")
    return Client(token)


# --- tokens -----------------------------------------------------------------------------------------------------

def test_token_is_shown_once_and_only_its_hash_is_stored(seeded):
    token, tid = db.create_api_token("Rider", "Pixel")
    assert token.startswith("bt_") and len(token) > 40
    with db.connect() as c:
        stored = [r[0] for r in c.execute("SELECT token_hash FROM api_tokens")]
    assert stored == [db.hash_token(token)] and token not in stored
    assert db.lookup_api_token(token) == "Rider"
    assert db.lookup_api_token(token + "x") is None and db.lookup_api_token("") is None and db.lookup_api_token(None) is None
    listed = db.list_api_tokens("Rider")
    assert [t['id'] for t in listed] == [tid] and listed[0]['label'] == "Pixel" and listed[0]['last_used_at']
    assert "token" not in listed[0]


def test_revoked_tokens_stop_working_and_only_the_owner_can_revoke(seeded):
    db.save_user("Other", 200, 150, 60.0, 170.0)
    token, tid = db.create_api_token("Rider")
    assert not db.revoke_api_token("Other", tid)                      # someone else's token
    assert db.lookup_api_token(token) == "Rider"
    assert db.revoke_api_token("Rider", tid) and not db.revoke_api_token("Rider", tid)
    assert db.lookup_api_token(token) is None and db.list_api_tokens("Rider") == []
    assert db.create_api_token("Nobody") == (None, None)


def test_deleting_a_rider_removes_their_tokens(seeded):
    token, _ = db.create_api_token("Rider")
    db.delete_user("Rider")
    assert db.lookup_api_token(token) is None


# --- the app ----------------------------------------------------------------------------------------------------

def test_ping_needs_no_token_but_everything_else_does(seeded, monkeypatch):
    monkeypatch.setattr(api, "_throttle", api.Throttle(limit=100))
    c = Client()
    assert c.get("/api/v1/ping")["json"] == {"ok": True, "app": "BikeTools", "api": 1}
    for path in ("/api/v1/me", "/api/v1/rides", "/api/v1/rides/1", "/api/v1/rides/1/series", "/api/v1/training/load",
                 "/api/v1/training/power"):
        r = c.get(path)
        assert r["code"] == 401 and r["headers"]["Cache-Control"] == "no-store", path
        assert c.get(path, token="bt_wrong")["code"] == 401


def test_repeated_bad_tokens_are_throttled(seeded):
    c = Client()
    codes = [c.get("/api/v1/me", token="bt_guess%d" % i)["code"] for i in range(api.FAIL_LIMIT + 2)]
    assert codes[:api.FAIL_LIMIT] == [401] * api.FAIL_LIMIT and codes[-1] == 429
    good, _ = db.create_api_token("Rider")
    assert c.get("/api/v1/me", token=good)["code"] == 429             # that address is locked out for the window
    assert c.get("/api/v1/me", token=good, remote="10.0.0.2")["code"] == 200      # others are not


def test_read_only_and_unknown_routes(client):
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        r = client.get("/api/v1/rides", method=method)
        assert r["code"] == 405 and "GET" in r["headers"]["Allow"]
    assert client.get("/api/v1/nothing")["code"] == 404
    assert client.get("/api/v1/rides/999999")["code"] == 404
    assert client.get("/api/v1/rides/abc")["code"] == 404
    assert client.get("/api/v1/rides?limit=x")["code"] == 400


def test_a_rider_cannot_read_another_riders_ride(client):
    db.save_user("Other", 200, 150, 60.0, 170.0)
    other, _ = db.create_api_token("Other")
    rid = db.get_user_rides("Rider")[0]['id']
    assert client.get(f"/api/v1/rides/{rid}")["code"] == 200
    for path in (f"/api/v1/rides/{rid}", f"/api/v1/rides/{rid}/series"):
        assert client.get(path, token=other)["code"] == 404
    assert client.get("/api/v1/rides", token=other)["json"]["total"] == 0


def test_cors_allows_the_app_origins_only(client):
    ok = client.get("/api/v1/ping", headers={"Origin": "https://localhost"})
    assert ok["headers"]["Access-Control-Allow-Origin"] == "https://localhost"
    bad = client.get("/api/v1/ping", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in bad["headers"]
    pre = client.get("/api/v1/me", method="OPTIONS", token=None, headers={"Origin": "capacitor://localhost"})
    assert pre["code"] == 204 and "Authorization" in pre["headers"]["Access-Control-Allow-Headers"]


def test_extra_origins_come_from_the_environment(client, monkeypatch):
    monkeypatch.setenv("BIKETOOLS_API_ORIGINS", "https://phone.example.org/")
    r = client.get("/api/v1/ping", headers={"Origin": "https://phone.example.org"})
    assert r["headers"]["Access-Control-Allow-Origin"] == "https://phone.example.org"


def test_server_errors_do_not_leak_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path /home/x")
    monkeypatch.setattr(payloads, "rides", boom)
    r = client.get("/api/v1/rides")
    assert r["code"] == 500 and "secret" not in r["body"].decode()


# --- static files -----------------------------------------------------------------------------------------------

def test_static_files_are_served_and_cannot_escape_the_folder(tmp_path, monkeypatch):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<h1>hi</h1>")
    (tmp_path / "web" / "app.js").write_text("1")
    (tmp_path / "secret.txt").write_text("nope")
    monkeypatch.setattr(api, "WEB_ROOT", tmp_path / "web")
    c = Client()
    assert c.get("/", token=None)["body"] == b"<h1>hi</h1>"
    r = c.get("/app.js", token=None)
    assert r["code"] == 200 and "javascript" in r["headers"]["Content-Type"]
    for evil in ("/../secret.txt", "/..%2fsecret.txt", "/%2e%2e/secret.txt", "/web/../../secret.txt", "//etc/passwd"):
        assert c.get(evil, token=None)["code"] == 404, evil
    assert c.get("/missing.png", token=None)["code"] == 404


# --- payloads ---------------------------------------------------------------------------------------------------

def test_me_and_rides(client):
    me = client.get("/api/v1/me")["json"]
    assert me["name"] == "Rider" and me["ftp"] == 250 and me["weight_kg"] == 70.0 and "token" not in json.dumps(me)
    r = client.get("/api/v1/rides?limit=3")["json"]
    assert r["total"] == 4 and len(r["rides"]) == 3
    whens = [x["when"] for x in r["rides"]]
    assert whens == sorted(whens, reverse=True)
    first = r["rides"][0]
    assert first["source"] == "strava" and first["distance_km"] > 0 and first["np"] > 0 and first["tss"] > 0
    assert len(client.get("/api/v1/rides?limit=3&offset=3")["json"]["rides"]) == 1
    assert len(client.get("/api/v1/rides?limit=100000")["json"]["rides"]) == 4        # capped, not an error


def test_ride_detail_matches_the_stored_metrics(client):
    rid = db.get_user_rides("Rider")[0]['id']
    r = client.get(f"/api/v1/rides/{rid}")["json"]
    with db.connect() as c:
        np_, tss, kj, carb = c.execute("SELECT norm_power, tss, kj, carb_g FROM rides WHERE id=?", (rid,)).fetchone()
    assert (r["np"], r["tss"], r["kj"]) == (np_, tss, round(kj))
    assert r["energy"]["carb_g"] == round(carb) and r["energy"]["kcal"] > 0 and r["energy"]["store_used_pct"] > 0
    assert r["energy"]["load"] in ("Light", "Moderate", "Heavy", "Very heavy")
    assert "intake" not in r["energy"]                                     # nothing logged
    assert sum(z["seconds"] for z in r["zones"]["power"]) > 0 and r["zones"]["power"][0]["zone"].startswith("Z")
    assert [p["seconds"] for p in r["power_curve"]] == sorted(p["seconds"] for p in r["power_curve"])
    assert all(p["seconds"] in payloads.CURVE_DURATIONS for p in r["power_curve"])


def test_logged_fuel_is_read_only_in_the_payload(client):
    rid = db.get_user_rides("Rider")[0]['id']
    db.set_ride_fuel(rid, 120, 1000)
    e = client.get(f"/api/v1/rides/{rid}")["json"]["energy"]
    assert e["intake"]["eaten_g"] == 120 and e["intake"]["fluid_ml"] == 1000
    assert e["intake"]["status"] in ("Below the guide", "Within the guide", "Above the guide", "No fuel needed")
    assert client.get(f"/api/v1/rides/{rid}", method="POST")["code"] == 405


def test_series_is_downsampled_with_a_track(client):
    rid = db.get_user_rides("Rider")[0]['id']
    s = client.get(f"/api/v1/rides/{rid}/series")["json"]
    assert s["points"] == payloads.SERIES_POINTS and len(s["power"]) == payloads.SERIES_POINTS
    assert len(s["hr"]) == len(s["elevation"]) == len(s["speed_kph"]) == s["points"]
    assert 150 < s["power"][100] < 260 and s["speed_kph"][100] == pytest.approx(28.8, abs=1.5)
    assert 2 <= len(s["track"]) <= payloads.TRACK_POINTS and all(len(p) == 2 for p in s["track"])
    assert s["seconds"] > 2000


def test_series_survives_a_missing_or_corrupt_file(client):
    rid = db.get_user_rides("Rider")[0]['id']
    with db.connect() as c:
        c.execute("DELETE FROM ride_files WHERE ride_id = ?", (rid,))
        c.execute("UPDATE rides SET file_content = NULL WHERE id = ?", (rid,))
    r = client.get(f"/api/v1/rides/{rid}/series")
    assert r["code"] == 200 and "error" in r["json"]


def test_training_load_matches_the_desktop_calculation(client):
    from src.analysis import training_load as tl
    j = client.get("/api/v1/training/load?days=60")["json"]
    today = pd.Timestamp(j["today"])
    pmc = tl.compute_pmc(db.get_user_tss_history("Rider"), today=today)
    assert j["now"]["ctl"] == pytest.approx(pmc.loc[:today].iloc[-1]["CTL"], abs=0.06)
    assert j["now"]["tsb"] == pytest.approx(pmc.loc[:today].iloc[-1]["TSB"], abs=0.06)
    assert len(j["days"]) == len(pmc.loc[:today]) <= 60 and j["days"][-1]["d"] == j["today"]
    assert len(client.get("/api/v1/training/load?days=1")["json"]["days"]) == len(j["days"])      # at least a week
    assert j["status"]["form"] and j["weeks"] and sum(w["rides"] for w in j["weeks"]) == 4


def test_training_power_has_curves_records_and_profile(client):
    j = client.get("/api/v1/training/power")["json"]
    labels = [w["label"] for w in j["windows"]]
    assert labels[0] == "Last 42 days" and labels[-1] == "All time"
    allt = j["windows"][-1]["points"]
    assert allt and all(p["watts"] > 0 and p["w_per_kg"] for p in allt)
    assert {p["label"] for p in j["profile"]} == {"5 s", "1 min", "5 min", "20 min"}
    assert j["records"] == allt and j["ftp"] == 250


def test_empty_rider_gets_empty_but_valid_answers(seeded):
    db.save_user("Fresh", 200, 150, 60.0, 170.0)
    c = Client(db.create_api_token("Fresh")[0])
    assert c.get("/api/v1/rides")["json"]["rides"] == []
    assert c.get("/api/v1/training/load")["json"] == {"empty": True}
    assert c.get("/api/v1/training/power")["json"] == {"empty": True}


def test_cli_creates_lists_and_revokes_tokens(seeded, capsys):
    from src.api.__main__ import main
    assert main(["token", "create", "Rider", "--label", "Pixel"]) == 0
    out = capsys.readouterr().out
    token = next(w for w in out.split() if w.startswith("bt_"))
    assert db.lookup_api_token(token) == "Rider"
    main(["token", "list", "Rider"])
    assert "Pixel" in capsys.readouterr().out
    tid = db.list_api_tokens("Rider")[0]["id"]
    assert main(["token", "revoke", "Rider", str(tid)]) == 0 and db.lookup_api_token(token) is None
    assert main(["token", "create", "Nobody"]) == 1
