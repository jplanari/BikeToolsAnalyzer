"""Rider passwords: hashing, rules, the phone login and its limits."""
import json

import pytest

from src.api import app as api
from src.data import db
from src.security import Throttle
from tests.test_api import Client


@pytest.fixture(autouse=True)
def cheap_hashes(monkeypatch):
    monkeypatch.setattr(db.passwords, "COST", {'n': 2 ** 10, 'r': 8, 'p': 1})
    monkeypatch.setattr(api, "_throttle", Throttle())
    monkeypatch.setattr(api, "_login_throttle", Throttle(limit=20, window=600))


@pytest.fixture
def riders(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "a.db"))
    db.init_db()
    db.save_user("Ana", 250, 170, 60.0, 170.0)
    db.save_user("Beni", 200, 150, 80.0, 180.0)
    return tmp_path


def post(client, path, body, raw=None, remote="10.0.0.1"):
    data = raw if raw is not None else json.dumps(body).encode()
    return client.get(path, method="POST", headers={"Content-Length": str(len(data))}, body=data, remote=remote)


# --- the database layer ---------------------------------------------------------------------------------------------

def test_hash_is_salted_and_checks_only_the_right_password():
    a, b = db.hash_password("correct horse"), db.hash_password("correct horse")
    assert a != b and a.startswith("scrypt$") and "correct" not in a
    assert db.check_hash("correct horse", a) and not db.check_hash("correct horsf", a) and not db.check_hash("", a)
    for broken in ("", None, "scrypt$x", "md5$1$2$3$4$5", "scrypt$1024$8$1$@@@$@@@"):
        assert db.check_hash("correct horse", broken) is False


def test_set_change_and_remove(riders):
    assert not db.has_password("Ana") and db.get_user_data("Ana")['has_password'] is False
    assert db.set_user_password("Ana", "short")[0] is False                      # too short
    assert db.set_user_password("Ghost", "long enough")[0] is False
    assert db.set_user_password("Ana", "first password")[0] and db.has_password("Ana")
    assert "password_hash" not in db.get_user_data("Ana") and db.get_user_data("Ana")['has_password'] is True
    assert not db.set_user_password("Ana", "second password", current="wrong")[0]        # needs the current one
    assert db.verify_user_password("Ana", "first password")
    assert db.set_user_password("Ana", "second password", current="first password")[0]
    assert not db.verify_user_password("Ana", "first password") and db.verify_user_password("Ana", "second password")
    assert not db.clear_user_password("Ana", current="nope")[0] and db.has_password("Ana")
    assert db.clear_user_password("Ana", current="second password")[0] and not db.has_password("Ana")
    assert db.set_user_password("Ana", "forced one", force=True)[0]               # command line: no current needed


def test_unknown_riders_and_open_riders_never_verify(riders):
    assert not db.verify_user_password("Ghost", "anything") and not db.verify_user_password("Beni", "")
    assert not db.verify_user_password("Beni", None)                              # no password set: nothing opens it by login


def test_changing_the_password_signs_out_the_phones(riders):
    token, _ = db.create_api_token("Ana")
    other, _ = db.create_api_token("Beni")
    db.set_user_password("Ana", "brand new pw")
    assert db.lookup_api_token(token) is None and db.lookup_api_token(other) == "Beni"
    token2, _ = db.create_api_token("Ana")
    db.clear_user_password("Ana", current="brand new pw")
    assert db.lookup_api_token(token2) is None


def test_throttle_reports_when_to_retry():
    th = Throttle(limit=2, window=60)
    assert th.retry_after("k", now=100) == 0
    th.fail("k", now=100); th.fail("k", now=110)
    assert 45 < th.retry_after("k", now=111) <= 50 and th.blocked("k", now=111)
    th.reset("k")
    assert th.retry_after("k", now=111) == 0


# --- the phone login ------------------------------------------------------------------------------------------------

def test_phone_login_returns_a_working_token(riders):
    db.set_user_password("Ana", "ana's long password")
    c = Client()
    r = post(c, "/api/v1/login", {"name": "Ana", "password": "ana's long password", "label": "Pixel"})
    assert r["code"] == 200 and r["headers"]["Cache-Control"] == "no-store"
    token = r["json"]["token"]
    assert r["json"]["me"]["name"] == "Ana" and "password" not in json.dumps(r["json"])
    assert Client(token).get("/api/v1/me")["json"]["name"] == "Ana"
    assert [t["label"] for t in db.list_api_tokens("Ana")] == ["Pixel"]


def test_login_failures_look_the_same_whatever_the_reason(riders):
    db.set_user_password("Ana", "ana's long password")
    c = Client()
    bodies = [post(c, "/api/v1/login", b)["json"] for b in (
        {"name": "Ana", "password": "wrong"}, {"name": "Ghost", "password": "wrong"},
        {"name": "Beni", "password": "anything"}, {"name": "", "password": ""})]
    assert all(b == {"error": "wrong name or password"} for b in bodies)
    assert db.list_api_tokens("Ana") == [] and db.list_api_tokens("Beni") == []


def test_login_is_limited_per_address(riders):
    db.set_user_password("Ana", "ana's long password")
    c = Client()
    codes = [post(c, "/api/v1/login", {"name": "Ana", "password": f"guess {i}"})["code"] for i in range(12)]
    assert codes == [401] * 10 + [429, 429]
    assert post(c, "/api/v1/login", {"name": "Ana", "password": "ana's long password"})["code"] == 429       # even the right one
    assert post(c, "/api/v1/login", {"name": "Ana", "password": "ana's long password"}, remote="10.0.0.9")["code"] == 200


def test_login_is_limited_per_rider_across_addresses(riders):
    db.set_user_password("Ana", "ana's long password")
    c = Client()
    for i in range(20):                                                           # a different address every time
        assert post(c, "/api/v1/login", {"name": "Ana", "password": f"guess {i}"}, remote=f"10.1.0.{i}")["code"] == 401
    r = post(c, "/api/v1/login", {"name": "ana", "password": "ana's long password"}, remote="10.1.0.99")
    assert r["code"] == 429 and int(r["headers"]["Retry-After"]) > 0
    assert post(c, "/api/v1/login", {"name": "Beni", "password": "x"}, remote="10.1.0.99")["code"] == 401      # others unaffected


def test_login_rejects_bad_bodies_and_other_posts_stay_refused(riders):
    c = Client()
    assert post(c, "/api/v1/login", None, raw=b"not json")["code"] == 400
    assert post(c, "/api/v1/login", None, raw=b"[1,2]")["code"] == 400
    assert post(c, "/api/v1/login", None, raw=b"x" * 5000)["code"] == 400
    assert c.get("/api/v1/login", method="POST", headers={"Content-Length": "0"}, body=b"")["code"] == 400
    assert c.get("/api/v1/rides", method="POST")["code"] == 405
    assert c.get("/api/v1/login")["code"] == 405                                  # GET: nothing to read there


def test_cli_sets_and_removes_a_password(riders, monkeypatch):
    from src.api.__main__ import main
    answers = iter(["cli password 1", "cli password 1"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert main(["password", "set", "Ana"]) == 0 and db.verify_user_password("Ana", "cli password 1")
    answers = iter(["one password", "another one"])
    assert main(["password", "set", "Ana"]) == 1 and db.verify_user_password("Ana", "cli password 1")
    assert main(["password", "remove", "Ana"]) == 0 and not db.has_password("Ana")
