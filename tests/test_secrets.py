import sqlite3

import pytest

pytest.importorskip("cryptography")

from src.data import db, secrets_store as sec


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.delenv(sec.KEY_ENV, raising=False)
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "t.db"))
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    return tmp_path / "t.db"


def _raw(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def test_roundtrip_and_plain_values_pass_through(temp_db):
    token = sec.encrypt("abc123")
    assert token.startswith(sec.PREFIX) and "abc123" not in token
    assert sec.decrypt(token) == "abc123"
    assert sec.decrypt("old plain token") == "old plain token"
    assert sec.encrypt(None) is None and sec.encrypt("") == ""
    assert sec.encrypt(token) == token                         # never encrypted twice


def test_key_file_is_private_and_reused(temp_db):
    import os, stat
    sec.encrypt("x")
    key = temp_db.parent / "biketools.key"
    assert key.exists() and stat.S_IMODE(os.stat(key).st_mode) == 0o600
    first = key.read_bytes()
    sec._cache.clear()
    sec.encrypt("y")
    assert key.read_bytes() == first


def test_tokens_are_encrypted_in_the_file_and_plain_through_the_api(temp_db):
    auth = {'athlete_id': 1, 'athlete_name': 'A', 'access_token': 'AT', 'refresh_token': 'RT', 'expires_at': 99}
    db.save_strava_auth("Ana", auth)
    assert _raw(temp_db, "SELECT access_token FROM strava_auth").startswith(sec.PREFIX)
    got = db.get_strava_auth("Ana")
    assert (got['access_token'], got['refresh_token']) == ('AT', 'RT')
    db.save_app_credentials("strava", "cid", "SECRET")
    assert "SECRET" not in _raw(temp_db, "SELECT client_secret FROM app_credentials")
    assert db.get_app_credentials("strava")['client_secret'] == "SECRET"
    db.save_polar_auth("Ana", {'polar_user_id': '7', 'access_token': 'PT'})
    assert db.get_polar_auth("Ana")['access_token'] == 'PT'


def test_smtp_password_is_encrypted(temp_db):
    if not hasattr(db, "save_smtp_settings"):
        pytest.skip("weekly e-mail not in this build")
    db.save_smtp_settings("h", 587, "starttls", "u", "pw", "me@x.com")
    assert "pw" != _raw(temp_db, "SELECT password FROM smtp_settings")
    assert db.get_smtp_settings()['password'] == "pw"


def test_existing_plain_secrets_are_encrypted_on_start(temp_db):
    conn = sqlite3.connect(temp_db)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    conn.execute("INSERT INTO strava_auth (user_id, access_token, refresh_token, expires_at) VALUES (?, 'AT', 'RT', 1)", (uid,))
    conn.execute("INSERT INTO app_credentials (provider, client_id, client_secret) VALUES ('wahoo', 'c', 'S')")
    conn.commit()
    conn.close()
    assert db.encrypt_plaintext_secrets() == 3
    assert db.encrypt_plaintext_secrets() == 0                  # idempotent
    assert db.get_strava_auth("Ana")['access_token'] == "AT"
    assert db.get_app_credentials("wahoo")['client_secret'] == "S"


def test_lost_key_reads_as_not_connected_instead_of_crashing(temp_db):
    db.save_app_credentials("strava", "cid", "SECRET")
    (temp_db.parent / "biketools.key").unlink()
    sec._cache.clear()
    assert db.get_app_credentials("strava")['client_secret'] == ""


def test_key_from_the_environment_wins(temp_db, monkeypatch):
    monkeypatch.setenv(sec.KEY_ENV, sec.new_key())
    token = sec.encrypt("v")
    assert sec.decrypt(token) == "v"
    assert not (temp_db.parent / "biketools.key").exists()
