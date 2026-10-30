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


def target_command():
    """What a shortcut has to run: the frozen executable, or this script with the current Python."""
    return [sys.executable] if FROZEN else [sys.executable, os.path.abspath(__file__)]


def linux_desktop_entry(command):
    quoted = " ".join('"%s"' % c.replace('"', '\\"') for c in command)
    return ("[Desktop Entry]\nType=Application\nName=BikeTools Analyzer\nComment=Cycling ride analysis\n"
            f"Exec={quoted}\nTerminal=false\nCategories=Sports;Utility;\nStartupWMClass=BikeToolsAnalyzer\n")


def macos_app_files(command):
    """Contents of a minimal .app bundle that starts the program: {relative path: (text, executable)}."""
    script = "#!/bin/bash\nexec " + " ".join("'%s'" % c.replace("'", "'\\''") for c in command) + "\n"
    plist = ('<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>'
             "<key>CFBundleName</key><string>BikeTools Analyzer</string>"
             "<key>CFBundleExecutable</key><string>run</string>"
             "<key>CFBundleIdentifier</key><string>com.biketools.analyzer</string>"
             "<key>CFBundlePackageType</key><string>APPL</string></dict></plist>\n")
    return {"Contents/MacOS/run": (script, True), "Contents/Info.plist": (plist, False)}


def install_shortcut():
    """Creates a desktop / application-menu shortcut. Returns the paths created."""
    command, made = target_command(), []
    if sys.platform == "win32":
        exe = command[0]
        args = subprocess.list2cmdline(command[1:])
        ps = ("$s=New-Object -ComObject WScript.Shell;"
              "foreach($d in @([Environment]::GetFolderPath('Desktop'),[Environment]::GetFolderPath('Programs'))){"
              "$l=$s.CreateShortcut((Join-Path $d 'BikeTools Analyzer.lnk'));"
              f"$l.TargetPath='{exe}';$l.Arguments='{args}';$l.WorkingDirectory='{os.path.dirname(exe)}';$l.Save()}}")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
        made = ["Desktop and Start menu: BikeTools Analyzer.lnk"]
    elif sys.platform == "darwin":
        app = os.path.expanduser("~/Applications/BikeTools Analyzer.app")
        for rel, (text, executable) in macos_app_files(command).items():
            path = os.path.join(app, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(text)
            if executable:
                os.chmod(path, 0o755)
        made = [app]
    else:
        entry = linux_desktop_entry(command)
        folders = [os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "applications")]
        desktop = os.path.expanduser("~/Desktop")
        if os.path.isdir(desktop):
            folders.append(desktop)
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "biketools-analyzer.desktop")
            with open(path, "w") as fh:
                fh.write(entry)
            os.chmod(path, 0o755)               # some desktops ask once to "Allow launching" on the icon
            made.append(path)
    return made


def tell(message):
    """A windowed Windows build has no console: show a message box there, print elsewhere."""
    if sys.platform == "win32" and (FROZEN or sys.stdout is None):
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "BikeTools Analyzer", 0x40)
    else:
        print(message)


def show_window(url):
    """Runs the native window in this process (blocking until it is closed)."""
    import webview
    webview.create_window("BikeTools Analyzer", url, width=1400, height=900)
    webview.start()


def open_window(url, log=None):
    """Shows the app in a native window. False when that is not possible (pywebview missing, no GTK/Qt/WebView2,
    or Qt aborting the whole process for a missing system library): the caller then uses the browser. The window
    runs in its own process precisely so that such a hard crash cannot take the launcher down with it."""
    try:
        code = subprocess.call(target_command() + ["--window", url], stdout=log, stderr=log,
                               stdin=subprocess.DEVNULL)
    except OSError:
        return False
    return code == 0


def main():
    multiprocessing.freeze_support()
    if "--install-shortcut" in sys.argv:
        try:
            tell("Shortcut created:\n" + "\n".join(install_shortcut()))
            return 0
        except Exception as e:
            tell(f"Could not create the shortcut: {e}")
            return 1
    if "--window" in sys.argv:
        if FROZEN or sys.stdout is None:
            redirect_output_to_log()
        try:
            show_window(sys.argv[sys.argv.index("--window") + 1])
            return 0
        except Exception as e:
            print(f"No native window ({type(e).__name__}: {e})", file=sys.stderr)
            return 3
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
        if not open_window(url, log):
            webbrowser.open(url)
            print(f"BikeTools Analyzer is running at {url} (Ctrl+C or close this program to quit)")
            server.wait()
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
