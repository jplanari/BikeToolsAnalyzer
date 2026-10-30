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


def test_window_failure_falls_back_instead_of_crashing(monkeypatch):
    import types
    fake = types.ModuleType("webview")
    fake.create_window = lambda *a, **k: None

    def start():
        raise RuntimeError("You must have either QT or GTK")
    fake.start = start
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert launcher.open_window("http://127.0.0.1:1") is False


def test_window_closed_normally_is_not_a_failure(monkeypatch):
    import types
    fake = types.ModuleType("webview")
    fake.create_window = lambda *a, **k: None
    fake.start = lambda: None
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert launcher.open_window("http://127.0.0.1:1") is True


def test_linux_entry_runs_the_command_and_has_no_terminal():
    e = launcher.linux_desktop_entry(["/opt/My App/BikeToolsAnalyzer"])
    assert e.startswith("[Desktop Entry]") and 'Exec="/opt/My App/BikeToolsAnalyzer"' in e and "Terminal=false" in e


def test_macos_bundle_has_an_executable_launcher_and_a_plist():
    files = launcher.macos_app_files(["/Apps/it's/BikeToolsAnalyzer"])
    script, executable = files["Contents/MacOS/run"]
    assert executable and script.startswith("#!/bin/bash") and "exec '/Apps/it'\\''s/BikeToolsAnalyzer'" in script
    assert "CFBundleExecutable" in files["Contents/Info.plist"][0]


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="Linux desktop entry")
def test_install_shortcut_writes_a_desktop_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("HOME", str(tmp_path))
    made = launcher.install_shortcut()
    assert made and all(os.path.exists(p) and os.access(p, os.X_OK) for p in made)
