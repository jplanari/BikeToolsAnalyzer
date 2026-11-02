"""Settings of a shared server: read from the environment, never from the rider."""
from src import serverconf
from src.data import db
from tests.test_db import temp_db  # noqa: F401 (fixture)


def test_server_mode_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("BIKETOOLS_SERVER_MODE", raising=False)
    assert serverconf.server_mode() is False
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("BIKETOOLS_SERVER_MODE", v)
        assert serverconf.server_mode() is True
    monkeypatch.setenv("BIKETOOLS_SERVER_MODE", "0")
    assert serverconf.server_mode() is False


def test_client_address_only_trusted_behind_a_proxy(monkeypatch):
    headers = {"X-Forwarded-For": "203.0.113.7, 10.0.0.2"}
    monkeypatch.delenv("BIKETOOLS_TRUST_PROXY", raising=False)
    assert serverconf.client_address(headers) is None                  # anyone could write that header
    monkeypatch.setenv("BIKETOOLS_TRUST_PROXY", "1")
    assert serverconf.client_address(headers) == "203.0.113.7"
    assert serverconf.client_address({}) is None


def test_smtp_from_the_environment_wins_over_the_database(temp_db, monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("BIKETOOLS_SMTP_"):
            monkeypatch.delenv(k)
    assert serverconf.smtp_from_env() is None
    db.save_smtp_settings("smtp.saved.example", 587, "starttls", "u", "p", "saved@example.com")
    assert db.get_smtp_settings()["host"] == "smtp.saved.example"
    monkeypatch.setenv("BIKETOOLS_SMTP_HOST", "smtp.server.example")
    assert db.get_smtp_settings()["host"] == "smtp.saved.example"      # a sender is needed too
    monkeypatch.setenv("BIKETOOLS_SMTP_SENDER", "bike@server.example")
    monkeypatch.setenv("BIKETOOLS_SMTP_PORT", "465")
    monkeypatch.setenv("BIKETOOLS_SMTP_SECURITY", "ssl")
    got = db.get_smtp_settings()
    assert (got["host"], got["port"], got["security"], got["sender"]) == ("smtp.server.example", 465, "ssl", "bike@server.example")


def test_unknown_client_address_is_still_throttled(temp_db, monkeypatch):
    from src.data.db import accounts as acc
    from src.security import Throttle
    monkeypatch.setattr(acc, "COST", {'n': 2 ** 10, 'r': 8, 'p': 1})
    monkeypatch.setattr(acc, "login_throttle", Throttle(3, 600))
    monkeypatch.setattr(acc, "deliver", lambda *a: None)
    for _ in range(3):
        assert acc.login("x@example.com", "whatever pass")[0] == "wrong"
    assert acc.login("x@example.com", "whatever pass")[0] == "throttled"
