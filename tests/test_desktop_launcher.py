import importlib.util
import os
import socket
import subprocess
import sys
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("launcher", os.path.join(ROOT, "desktop", "launcher.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_data_dir_override_is_created(tmp_path, monkeypatch):
    target = tmp_path / "mydata"
    monkeypatch.setenv("BIKETOOLS_DATA", str(target))
    assert launcher.data_dir() == str(target) and target.is_dir()


def test_default_data_dir_is_per_user_not_the_program_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("BIKETOOLS_DATA", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    d = launcher.data_dir()
    assert launcher.APP_NAME in d and not d.startswith(ROOT)


def test_free_port_is_bindable():
    p = launcher.free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", p))


def test_server_command_runs_this_launcher_in_server_mode():
    cmd = launcher.server_command(8765)
    assert cmd[-2:] == ["--server", "8765"] and cmd[0] == sys.executable


def test_wait_until_up_gives_up_when_the_server_died():
    class Dead:
        def poll(self):
            return 1
    assert launcher.wait_until_up(1, Dead(), timeout=2) is False


def _real_streamlit():
    return subprocess.run([sys.executable, "-c", "import streamlit.web.bootstrap"], capture_output=True).returncode == 0


@pytest.mark.skipif(os.environ.get("CI") is not None or not _real_streamlit(),
                    reason="starts a real Streamlit server; run locally with Streamlit installed")
def test_server_mode_serves_the_app(tmp_path):
    port = launcher.free_port()
    env = dict(os.environ, BIKETOOLS_DATA=str(tmp_path))
    p = subprocess.Popen(launcher.server_command(port), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert launcher.wait_until_up(port, p, timeout=90)
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/").status == 200
    finally:
        p.terminate()
        p.wait(10)
