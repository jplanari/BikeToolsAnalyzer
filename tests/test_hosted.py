"""A shared server: invitations, sign-up, Strava/Wahoo connections and imports for each rider, account deletion."""
import json
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from src.api import app as api
from src.api import jobs, providers
from src.data import db
from src.data.imports import strava_import as si
from src.data.imports import wahoo_import as wi
from src.data.providers import strava as sv
from src.data.providers import wahoo as wh
from src.security import Throttle
from tests.test_api import Client


@pytest.fixture(autouse=True)
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "h.db"))
    db.init_db()
    monkeypatch.setattr(db.passwords, "COST", {'n': 2 ** 10, 'r': 8, 'p': 1})
    for name, value in (("_throttle", Throttle()), ("_login_throttle", Throttle(limit=20, window=600)),
                        ("_register_throttle", Throttle(limit=8, window=3600))):
        monkeypatch.setattr(api, name, value)
    for var in ("BIKETOOLS_PUBLIC_URL", "BIKETOOLS_STRAVA_CLIENT_ID", "BIKETOOLS_STRAVA_CLIENT_SECRET",
                "BIKETOOLS_WAHOO_CLIENT_ID", "BIKETOOLS_WAHOO_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    jobs._jobs.clear()
    providers._redirects.clear()


def post(client, path, body, token="default", remote="10.0.0.1"):
    data = json.dumps(body).encode()
    return client.get(path, method="POST", token=token, headers={"Content-Length": str(len(data))}, body=data, remote=remote)


def signed_up(name="Nuria", password="a long password", code=None):
    code = code or db.create_invite()[0]
    r = post(Client(), "/api/v1/register", {"name": name, "password": password, "invite": code})
    assert r["code"] == 200, r
    return Client(r["json"]["token"]), r["json"]


# --- invitations ------------------------------------------------------------------------------------------------

def test_invitation_codes_have_uses_and_expire():
    code, iid = db.create_invite(uses=2, days=1)
    assert len(code) == 11 and code[5] == "-" and not set(code.replace("-", "")) & set("01OI")
    assert db.use_invite(code.lower().replace("-", " ")) and db.use_invite(code)          # typed any way
    assert not db.use_invite(code) and db.list_invites() == []
    assert not db.use_invite("") and not db.use_invite("AAAAA-AAAAA") and not db.use_invite(None)
    old, oid = db.create_invite(days=1)
    with db.connect() as c:
        c.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - 5, oid))
    assert not db.use_invite(old) and db.list_invites() == []
    stored = db.create_invite()
    with db.connect() as c:
        hashes = [r[0] for r in c.execute("SELECT code_hash FROM invites")]
    assert stored[0] not in " ".join(hashes) and stored[0].replace("-", "") not in " ".join(hashes)
    assert db.revoke_invite(stored[1]) and not db.revoke_invite(stored[1])


def test_a_refunded_invitation_can_be_used_again():
    code, _ = db.create_invite()
    assert db.use_invite(code)
    db.refund_invite(code)
    assert db.use_invite(code)


# --- sign-up ----------------------------------------------------------------------------------------------------

def test_sign_up_creates_a_rider_who_can_use_the_api_and_sign_in_again():
    client, body = signed_up("Nuria Vidal")
    assert body["me"]["name"] == "Nuria Vidal" and client.get("/api/v1/me")["json"]["name"] == "Nuria Vidal"
    user = db.get_user_data("Nuria Vidal")
    assert user["ftp_auto"] == 1 and user["lthr_auto"] == 1 and user["has_password"] is True
    again = post(Client(), "/api/v1/login", {"name": "Nuria Vidal", "password": "a long password"})
    assert again["code"] == 200


def test_an_invitation_works_once_and_a_wrong_one_is_refused():
    code, _ = db.create_invite()
    signed_up("First", code=code)
    r = post(Client(), "/api/v1/register", {"name": "Second", "password": "another long one", "invite": code})
    assert r["code"] == 403 and db.get_user_data("Second") is None
    r = post(Client(), "/api/v1/register", {"name": "Third", "password": "another long one", "invite": "ZZZZZ-ZZZZZ"})
    assert r["code"] == 403


def test_sign_up_checks_the_name_and_the_password_without_spending_the_code():
    code, _ = db.create_invite()
    signed_up("Marta")
    for body, expected in (({"name": "marta", "password": "long enough pw"}, 409),          # names are unique ignoring case
                           ({"name": "x", "password": "long enough pw"}, 400),
                           ({"name": "a/b", "password": "long enough pw"}, 400),
                           ({"name": "Ok Name", "password": "short"}, 400),
                           ({"name": "<b>x</b>", "password": "long enough pw"}, 400)):
        assert post(Client(), "/api/v1/register", {**body, "invite": code})["code"] == expected, body
    assert db.list_invites()[0]["uses_left"] == 1                                            # still unspent
    assert post(Client(), "/api/v1/register", {"name": "Good", "password": "long enough pw", "invite": code})["code"] == 200


def test_guessing_invitations_is_slowed_down():
    c = Client()
    codes = [post(c, "/api/v1/register", {"name": f"N{i}", "password": "long enough pw", "invite": "ZZZZZ-ZZZZZ"})["code"] for i in range(10)]
    assert codes == [403] * 8 + [429, 429]
    valid, _ = db.create_invite()
    assert post(c, "/api/v1/register", {"name": "Late", "password": "long enough pw", "invite": valid})["code"] == 429
    assert post(c, "/api/v1/register", {"name": "Late", "password": "long enough pw", "invite": valid}, remote="10.9.9.9")["code"] == 200


def test_cli_manages_invitations(capsys):
    from src.api.__main__ import main
    assert main(["invite", "create", "--uses", "3", "--days", "2", "--label", "club"]) == 0
    code = next(w for w in capsys.readouterr().out.split() if len(w) == 11 and w[5] == "-")
    main(["invite", "list"])
    assert "club" in capsys.readouterr().out and db.list_invites()[0]["uses_left"] == 3
    assert db.use_invite(code)
    assert main(["invite", "revoke", str(db.list_invites()[0]["id"])]) == 0 and db.list_invites() == []


# --- accounts ---------------------------------------------------------------------------------------------------

def test_a_rider_can_delete_their_account_with_the_password():
    client, _ = signed_up("Leaving")
    other, _ = signed_up("Staying")
    assert post(client, "/api/v1/account/delete", {"password": "wrong password"})["code"] == 403
    assert db.get_user_data("Leaving")
    assert post(client, "/api/v1/account/delete", {"password": "a long password"})["code"] == 200
    assert db.get_user_data("Leaving") is None and client.get("/api/v1/me")["code"] == 401
    assert other.get("/api/v1/me")["json"]["name"] == "Staying"


# --- Strava / Wahoo ----------------------------------------------------------------------------------------------

def configure(monkeypatch, public="https://bike.example.net"):
    monkeypatch.setenv("BIKETOOLS_PUBLIC_URL", public)
    for p in ("STRAVA", "WAHOO"):
        monkeypatch.setenv(f"BIKETOOLS_{p}_CLIENT_ID", "cid-" + p.lower())
        monkeypatch.setenv(f"BIKETOOLS_{p}_CLIENT_SECRET", "secret-" + p.lower())


def test_connections_report_what_is_set_up_here(monkeypatch):
    client, _ = signed_up()
    j = client.get("/api/v1/connections")["json"]
    assert j["strava"] == {"configured": False, "connected": False, "name": None} and j["job"] is None
    assert post(client, "/api/v1/connections/strava/start", {})["code"] == 409               # not set up on this server
    configure(monkeypatch)
    assert client.get("/api/v1/connections")["json"]["wahoo"]["configured"] is True
    assert post(client, "/api/v1/connections/polar/start", {})["code"] == 404


def test_start_gives_the_providers_address_pointing_back_at_the_server(monkeypatch):
    configure(monkeypatch)
    client, _ = signed_up("Nuria")
    for provider, host in (("strava", "www.strava.com"), ("wahoo", "api.wahooligan.com")):
        url = post(client, f"/api/v1/connections/{provider}/start", {})["json"]["url"]
        q = parse_qs(urlparse(url).query)
        assert urlparse(url).netloc == host and q["client_id"] == ["cid-" + provider]
        assert q["redirect_uri"] == ["https://bike.example.net/api/v1/oauth/callback"]
        assert ("wahoo." in q["state"][0]) == (provider == "wahoo")
    assert "secret" not in url


def test_callback_connects_the_rider_whose_state_it_carries(monkeypatch):
    configure(monkeypatch)
    client, _ = signed_up("Nuria")
    seen = {}

    def fake_exchange(cfg, code):
        seen["cfg"], seen["code"] = cfg, code
        return {"athlete_id": 7, "athlete_name": "Nuria V", "access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000}
    monkeypatch.setattr(sv, "exchange_code", fake_exchange)
    state = parse_qs(urlparse(post(client, "/api/v1/connections/strava/start", {})["json"]["url"]).query)["state"][0]
    page = Client().get(f"/api/v1/oauth/callback?state={state}&code=abc&scope=read,activity:read_all", token=None)
    assert page["code"] == 200 and "text/html" in page["headers"]["Content-Type"] and "Connected" in page["body"].decode()
    assert seen["code"] == "abc" and seen["cfg"].redirect_uri == "https://bike.example.net/api/v1/oauth/callback"
    assert client.get("/api/v1/connections")["json"]["strava"] == {"configured": True, "connected": True, "name": "Nuria V"}
    again = Client().get(f"/api/v1/oauth/callback?state={state}&code=abc&scope=read,activity:read_all", token=None)
    assert "Not connected" in again["body"].decode()                                          # a state works once


def test_callback_with_a_forged_or_refused_request_connects_nothing(monkeypatch):
    configure(monkeypatch)
    client, _ = signed_up("Nuria")
    monkeypatch.setattr(sv, "exchange_code", lambda *a: pytest.fail("must not be called"))
    page = Client().get("/api/v1/oauth/callback?state=guess&code=abc&scope=activity:read_all", token=None)
    assert "Not connected" in page["body"].decode()
    state = parse_qs(urlparse(post(client, "/api/v1/connections/strava/start", {})["json"]["url"]).query)["state"][0]
    refused = Client().get(f"/api/v1/oauth/callback?state={state}&error=access_denied", token=None)
    assert "Not connected" in refused["body"].decode()
    assert client.get("/api/v1/connections")["json"]["strava"]["connected"] is False
    xss = Client().get("/api/v1/oauth/callback?state=%3Cscript%3E", token=None)["body"].decode()
    assert "<script>" not in xss


def test_wahoo_callback_and_disconnect(monkeypatch):
    configure(monkeypatch)
    client, _ = signed_up("Nuria")
    monkeypatch.setattr(wh, "exchange_code", lambda cfg, code: {"access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000,
                                                                "wahoo_user_id": 3, "wahoo_name": "Nuria W"})
    state = parse_qs(urlparse(post(client, "/api/v1/connections/wahoo/start", {})["json"]["url"]).query)["state"][0]
    Client().get(f"/api/v1/oauth/callback?state={state}&code=xyz", token=None)
    assert client.get("/api/v1/connections")["json"]["wahoo"]["name"] == "Nuria W"
    assert post(client, "/api/v1/connections/wahoo/disconnect", {})["code"] == 200
    assert client.get("/api/v1/connections")["json"]["wahoo"]["connected"] is False


def test_riders_do_not_see_each_others_connections(monkeypatch):
    configure(monkeypatch)
    a, _ = signed_up("Ana")
    b, _ = signed_up("Beni")
    db.save_wahoo_auth("Ana", {"access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000, "wahoo_user_id": 1, "wahoo_name": "Ana W"})
    assert a.get("/api/v1/connections")["json"]["wahoo"]["connected"] is True
    assert b.get("/api/v1/connections")["json"]["wahoo"]["connected"] is False


# --- imports ----------------------------------------------------------------------------------------------------

def connect_strava(name="Nuria"):
    db.save_strava_auth(name, {"athlete_id": 1, "athlete_name": "N", "access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000})


def test_import_needs_a_connection_and_runs_one_job_at_a_time(monkeypatch):
    configure(monkeypatch)
    client, _ = signed_up("Nuria")
    assert post(client, "/api/v1/connections/strava/import", {"days": 30})["code"] == 409
    connect_strava()
    monkeypatch.setattr(jobs.threading, "Thread", lambda *a, **k: SimpleNamespace(start=lambda: None))     # do not run it yet
    assert post(client, "/api/v1/connections/strava/import", {"days": 30})["code"] == 200
    assert client.get("/api/v1/connections")["json"]["job"]["state"] == "queued"
    assert post(client, "/api/v1/connections/strava/import", {"days": 30})["code"] == 409                  # already running
    assert post(client, "/api/v1/connections/strava/import", {"days": "many"})["code"] == 400


def test_a_job_reports_progress_and_the_result(monkeypatch):
    configure(monkeypatch)
    signed_up("Nuria")
    connect_strava()
    calls = {}

    def fake_history(cfg, user, since, ftp, progress=None, usage=None, lthr=None):
        calls["since"], calls["ftp"] = since, ftp
        progress(1, 4, "Morning ride")
        assert jobs.status("Nuria")["state"] == "running" and jobs.status("Nuria")["name"] == "Morning ride"
        return si.ImportResult(found=4, imported=3, already_imported=1)
    monkeypatch.setattr(si, "import_history", fake_history)
    assert jobs.start("Nuria", "strava", 90, run=False)[0]
    jobs._run("Nuria", "strava", 90)
    s = jobs.status("Nuria")
    assert (s["state"], s["imported"], s["already"], s["total"], s["error"]) == ("done", 3, 1, 4, "") and s["finished"]
    assert calls["ftp"] == 200 and 85 <= (time.time() - calls["since"].timestamp()) / 86400 <= 95
    monkeypatch.setattr(si, "sync_new_rides", lambda *a, **k: (si.ImportResult(found=0), None))
    assert jobs.start("Nuria", "strava", None, run=False)[0]                                 # days None: only what is new
    jobs._run("Nuria", "strava", None)
    assert jobs.status("Nuria")["state"] == "done"


def test_a_failing_import_ends_the_job_with_an_error(monkeypatch):
    configure(monkeypatch)
    signed_up("Nuria")
    connect_strava()
    monkeypatch.setattr(si, "import_history", lambda *a, **k: (_ for _ in ()).throw(sv.StravaAuthError("revoked")))
    jobs.start("Nuria", "strava", 30, run=False)
    jobs._run("Nuria", "strava", 30)
    assert jobs.status("Nuria")["state"] == "error" and jobs.status("Nuria")["error"] == "reconnect"
    monkeypatch.setattr(si, "import_history", lambda *a, **k: 1 / 0)
    jobs.start("Nuria", "strava", 30, run=False)
    jobs._run("Nuria", "strava", 30)
    assert jobs.status("Nuria")["state"] == "error" and jobs.start("Nuria", "strava", 30, run=False)[0]    # can try again


def test_new_endpoints_need_a_token_and_post_only_where_it_makes_sense(monkeypatch):
    c = Client()
    for method, path in (("GET", "/api/v1/connections"), ("POST", "/api/v1/connections/strava/start"),
                         ("POST", "/api/v1/connections/strava/import"), ("POST", "/api/v1/account/delete")):
        assert c.get(path, method=method, token=None, headers={"Content-Length": "2"}, body=b"{}")["code"] == 401, path
    assert c.get("/api/v1/register")["code"] == 405 and c.get("/api/v1/connections", method="DELETE", token=None)["code"] == 405
