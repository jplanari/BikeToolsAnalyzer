from src.data import credentials as cr
from src.data import db
from tests.test_db import temp_db  # noqa: F401 (fixture)


def test_nothing_set_up():
    assert cr.resolve("strava", None, None) is None
    assert cr.resolve("strava", None, {"client_id": "1"}) is None
    assert cr.resolve("strava", None, {"client_id": " ", "client_secret": "x"}) is None


def test_secrets_only():
    got = cr.resolve("wahoo", None, {"client_id": "a", "client_secret": "b"})
    assert got["source"] == "secrets"
    assert got["redirect_uri"] == cr.PROVIDERS["wahoo"]["redirect"]


def test_app_wins_over_secrets():
    got = cr.resolve("strava", {"client_id": "app", "client_secret": "s", "redirect_uri": None, "timezone": None},
                     {"client_id": "file", "client_secret": "s"})
    assert got["client_id"] == "app" and got["source"] == "app"
    assert got["redirect_uri"] == cr.PROVIDERS["strava"]["redirect"]


def test_incomplete_app_row_falls_back_to_secrets():
    got = cr.resolve("polar", {"client_id": "", "client_secret": "s"}, {"client_id": "f", "client_secret": "g"})
    assert got["source"] == "secrets"


def test_empty_redirect_is_kept_for_polar():
    got = cr.resolve("polar", {"client_id": "a", "client_secret": "b", "redirect_uri": ""}, None)
    assert got["redirect_uri"] == ""


def test_db_roundtrip(temp_db):
    assert db.get_app_credentials("polar") is None
    db.save_app_credentials("polar", " id ", " secret ", "")
    row = db.get_app_credentials("polar")
    assert row["client_id"] == "id" and row["client_secret"] == "secret"
    db.save_app_credentials("polar", "id2", "s2", "http://x/")
    assert db.get_app_credentials("polar")["client_id"] == "id2"
    db.delete_app_credentials("polar")
    assert db.get_app_credentials("polar") is None
