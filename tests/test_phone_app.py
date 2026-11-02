import pytest

from src.viz import phone_app as pa


@pytest.fixture
def seeded_for_runner(tmp_path, monkeypatch):
    from src.data import db
    monkeypatch.setattr(db, 'DB_FILE', str(tmp_path / 'p.db'))
    db.init_db()


def test_pairing_links_carry_server_and_token_safely():
    link = pa.pairing_link("http://192.168.1.20:8765", "bt_a-b_c")
    assert link == "biketools://pair?server=http%3A%2F%2F192.168.1.20%3A8765&token=bt_a-b_c"
    web = pa.web_pairing_link("http://192.168.1.20:8765/", "bt_x")
    assert web.startswith("http://192.168.1.20:8765/#/pair?") and "token=bt_x" in web


def test_lan_address_is_an_http_url_with_the_port():
    a = pa.lan_address(1234)
    assert a.startswith("http://") and a.endswith(":1234")


def test_server_can_be_started_and_stopped_inside_a_program(seeded_for_runner):
    import json
    import urllib.request
    from src.api import runner
    ok, msg = runner.start("127.0.0.1", 0)
    try:
        assert ok and runner.running()
        assert runner.start("127.0.0.1", 0) == (True, "already running")          # twice is harmless
        port = runner.running()[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/ping", timeout=5) as r:
            assert json.load(r)["app"] == "BikeTools"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert b"BikeTools" in r.read()                                         # the web app is served too
    finally:
        runner.stop()
    assert runner.running() is None


def test_a_busy_port_is_reported_not_raised():
    import socket
    from src.api import runner
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        ok, msg = runner.start("127.0.0.1", s.getsockname()[1])
    assert not ok and msg and runner.running() is None
