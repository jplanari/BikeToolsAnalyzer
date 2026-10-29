"""
Desktop launcher: starts the Streamlit server on a free local port and shows it in a native window
(pywebview), or in the default browser when pywebview is not installed. Works from source
(python desktop/launcher.py) and frozen with PyInstaller (see biketools.spec).

Your data lives in a per-user folder (BIKETOOLS_DATA overrides it), never next to the program, so the app can
be updated or moved without touching the database, the ride files or the backups.
"""
import multiprocessing
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

APP_NAME = "BikeToolsAnalyzer"
FROZEN = getattr(sys, "frozen", False)
ROOT = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_TIMEOUT_S = 60


def data_dir():
    """Per-user data folder, created on demand."""
    custom = os.environ.get("BIKETOOLS_DATA")
    if custom:
        path = custom
    elif sys.platform == "win32":
        path = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), APP_NAME)
    elif sys.platform == "darwin":
        path = os.path.join(os.path.expanduser("~/Library/Application Support"), APP_NAME)
    else:
        path = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def redirect_output_to_log():
    """A windowed (console-less) Windows/macOS build has no stdout/stderr (they are None) and Streamlit, click and
    logging crash on them. Everything goes to launcher.log in the data folder instead."""
    log = open(os.path.join(data_dir(), "launcher.log"), "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    return log


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_command(port):
    """Command that runs this launcher as the server process (same executable when frozen)."""
    base = [sys.executable] if FROZEN else [sys.executable, os.path.abspath(__file__)]
    return base + ["--server", str(port)]


def run_server(port):
    """Runs Streamlit in this process (blocking). The database and backups go to the data folder."""
    if FROZEN or sys.stdout is None:
        redirect_output_to_log()
    folder = data_dir()
    os.environ.setdefault("BIKETOOLS_DB", os.path.join(folder, "biketools.db"))
    os.chdir(folder)                       # backups/ and .streamlit/secrets.toml (Strava keys) live here
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from streamlit import config
    from streamlit.web import bootstrap
    options = {
        "server.port": port, "server.address": "127.0.0.1", "server.headless": True,
        "server.fileWatcherType": "none", "browser.gatherUsageStats": False,
        "global.developmentMode": False, "server.maxUploadSize": 500, "logger.level": "warning",
    }
    for key, value in options.items():          # bootstrap.run(flag_options=...) alone leaves the defaults in place
        config.set_option(key, value, "desktop launcher")
    bootstrap.run(os.path.join(ROOT, "app.py"), False, [], flag_options=options)


def wait_until_up(port, process, timeout=START_TIMEOUT_S):
    url = f"http://127.0.0.1:{port}/_stcore/health"
    end = time.time() + timeout
    while time.time() < end:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    multiprocessing.freeze_support()
    if "--server" in sys.argv:
        return run_server(int(sys.argv[sys.argv.index("--server") + 1]))
    if FROZEN or sys.stdout is None:
        redirect_output_to_log()
    port = free_port()
    log = open(os.path.join(data_dir(), "launcher.log"), "a")
    server = subprocess.Popen(server_command(port), stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    try:
        if not wait_until_up(port, server):
            print(f"BikeTools Analyzer could not start its server; see {os.path.join(data_dir(), 'launcher.log')}",
                  file=sys.stderr)
            return 1
        url = f"http://127.0.0.1:{port}"
        try:
            import webview
        except ImportError:
            webbrowser.open(url)
            print(f"BikeTools Analyzer is running at {url} (Ctrl+C to quit)")
            server.wait()
        else:
            webview.create_window("BikeTools Analyzer", url, width=1400, height=900)
            webview.start()                 # returns when the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
