"""
Desktop launcher: starts the Streamlit server on a free local port and shows it in a native window
(pywebview), or in the default browser when pywebview is not installed. Works from source
(python desktop/launcher.py) and frozen with PyInstaller (see biketools.spec).

Your data lives in a per-user folder (BIKETOOLS_DATA overrides it), never next to the program, so the app can
be updated or moved without touching the database, the ride files or the backups.
"""
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
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


DEFAULT_PORT = 8501        # the redirect URI registered with Strava/Wahoo is http://localhost:8501/


def choose_port():
    """The port OAuth callbacks (Strava/Wahoo redirect_uri) expect: BIKETOOLS_PORT or 8501, so existing secrets.toml and
    app registrations keep working. Only when it is taken (e.g. the web version is running) a free one is used."""
    wanted = int(os.environ.get("BIKETOOLS_PORT") or DEFAULT_PORT)
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", wanted))
            return wanted
        except OSError:
            print(f"Port {wanted} is busy; using another one: Strava/Wahoo sign-in will not work this time.",
                  file=sys.stderr)
            return free_port()


def install_theme(folder):
    """Streamlit reads .streamlit/config.toml from the working directory, which is the data folder here: copy the
    app's theme there on every start (secrets.toml next to it is not touched)."""
    src = os.path.join(ROOT, ".streamlit", "config.toml")
    if os.path.exists(src):
        os.makedirs(os.path.join(folder, ".streamlit"), exist_ok=True)
        shutil.copyfile(src, os.path.join(folder, ".streamlit", "config.toml"))


def server_command(port):
    """Command that runs this launcher as the server process (same executable when frozen)."""
    base = [sys.executable] if FROZEN else [sys.executable, os.path.abspath(__file__)]
    return base + ["--server", str(port)]


STATUS_FILE = "status.json"
WATCH_INTERVAL_S = 2
IDLE_POLLS_BEFORE_QUIT = 30            # about a minute without any open page, after one had been open


def active_sessions():
    """How many pages are connected to the server right now (None when Streamlit cannot tell)."""
    try:
        from streamlit.runtime import Runtime
        return int(Runtime.instance()._session_mgr.num_active_sessions())
    except Exception:
        return None


def watch_sessions(folder, count=active_sessions, interval=WATCH_INTERVAL_S, idle_polls=IDLE_POLLS_BEFORE_QUIT,
                   quit_now=lambda: os._exit(0)):
    """Runs in the server: writes how many pages are open to status.json (so the launcher can tell that a window
    really showed the app, whatever process it belongs to) and ends the server once the last page has been closed
    for about a minute, so closing the window (or the browser tab) really quits the program.
    BIKETOOLS_KEEP_RUNNING=1 keeps the server running regardless."""
    seen, idle = False, 0
    while True:
        time.sleep(interval)
        n = count()
        try:
            with open(os.path.join(folder, STATUS_FILE), "w") as fh:
                json.dump({"sessions": n, "time": time.time()}, fh)
        except OSError:
            pass
        if n:
            seen, idle = True, 0
        elif seen and n == 0:
            idle += 1
            if idle >= idle_polls and not os.environ.get("BIKETOOLS_KEEP_RUNNING"):
                quit_now()
                return


def read_sessions(folder=None, max_age_s=10):
    """Pages connected according to the server's status.json, or None when it is missing or stale."""
    try:
        with open(os.path.join(folder or data_dir(), STATUS_FILE)) as fh:
            status = json.load(fh)
        if time.time() - float(status["time"]) <= max_age_s:
            return status["sessions"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def run_server(port):
    """Runs Streamlit in this process (blocking). The database and backups go to the data folder."""
    if FROZEN or sys.stdout is None:
        redirect_output_to_log()
    folder = data_dir()
    os.environ.setdefault("BIKETOOLS_DB", os.path.join(folder, "biketools.db"))
    os.environ["BIKETOOLS_DESKTOP"] = "1"      # the app saves files with a Save as dialog instead of a browser download
    os.chdir(folder)                       # backups/ and .streamlit/secrets.toml (Strava keys) live here
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    install_theme(folder)
    from streamlit import config
    from streamlit.web import bootstrap
    options = {
        "server.port": port, "server.address": "127.0.0.1", "server.headless": True,
        "server.fileWatcherType": "none", "browser.gatherUsageStats": False,
        "global.developmentMode": False, "server.maxUploadSize": 500, "logger.level": "warning",
    }
    for key, value in options.items():          # bootstrap.run(flag_options=...) alone leaves the defaults in place
        config.set_option(key, value, "desktop launcher")
    try:
        os.remove(os.path.join(folder, STATUS_FILE))
    except OSError:
        pass
    threading.Thread(target=watch_sessions, args=(folder,), daemon=True).start()
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


def unblock_downloaded_files(folders):
    """Windows marks files that came from a download (a zip from GitHub) as blocked, and .NET then refuses to load
    the DLLs the native window needs ("Failed to resolve Python.Runtime.Loader.Initialize"). Removing the mark is
    what 'Unblock' in a file's properties does. Returns how many files were unblocked."""
    if sys.platform != "win32":
        return 0
    count = 0
    for folder in folders:
        for here, _, names in os.walk(folder):
            for name in names:
                if name.lower().endswith((".dll", ".pyd", ".exe")):
                    try:
                        os.remove(os.path.join(here, name) + ":Zone.Identifier")
                        count += 1
                    except OSError:
                        pass                       # no mark on this file (the usual case), or not allowed
    return count


def show_window(url):
    """Runs the native window in this process (blocking until it is closed).
    On Windows the Edge WebView2 engine is required: without it pywebview would quietly fall back to the old
    Internet Explorer engine, which cannot show Streamlit, so asking for it makes that case an error instead
    and the launcher moves on to the next kind of window. The window keeps its cookies and local storage
    in the data folder between runs."""
    unblock_downloaded_files([os.path.join(ROOT, n) for n in ("pythonnet", "clr_loader", "webview")])
    import webview
    webview.create_window("BikeTools Analyzer", url, width=1400, height=900, min_size=(900, 600))
    kwargs = {"private_mode": False, "storage_path": os.path.join(data_dir(), "webview")}
    if sys.platform == "win32":
        kwargs["gui"] = "edgechromium"
    webview.start(**kwargs)


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


# --- a dedicated window whatever the machine has -------------------------------------------------------------------
# 1. pywebview (WebView2 / WebKit / Qt): a real native window.
# 2. The app window of a Chromium-based browser (Edge ships with Windows 10/11; Chrome, Chromium, Brave and Edge
#    elsewhere): no tabs, no address bar, its own profile in the data folder, so it is a window of its own and does
#    not depend on the default browser or on the browser's open windows.
# 3. The default browser, as a last resort.
# BIKETOOLS_WINDOW = auto (default) | webview | app | browser forces one of them.


def _windows_browsers():
    roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    parts = [("Microsoft", "Edge", "Application", "msedge.exe"), ("Google", "Chrome", "Application", "chrome.exe"),
             ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
             ("Chromium", "Application", "chrome.exe")]
    return [os.path.join(r, *p) for p in parts for r in roots if r]


def _mac_browsers():
    names = ["Microsoft Edge", "Google Chrome", "Brave Browser", "Chromium"]
    return [f"/Applications/{n}.app/Contents/MacOS/{n}" for n in names] + [
        os.path.expanduser(f"~/Applications/{n}.app/Contents/MacOS/{n}") for n in names]


def find_app_browser():
    """A Chromium-based browser that can show a page as an app window, or None."""
    if sys.platform == "win32":
        candidates = _windows_browsers()
    elif sys.platform == "darwin":
        candidates = _mac_browsers()
    else:
        candidates = [shutil.which(n) for n in ("microsoft-edge", "microsoft-edge-stable", "google-chrome",
                                                "google-chrome-stable", "chromium", "chromium-browser",
                                                "brave-browser")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def app_window_command(browser, url, profile_dir):
    """Command that opens `url` as an app window of its own (separate profile, so it never joins a running browser)."""
    return [browser, f"--app={url}", f"--user-data-dir={profile_dir}", "--window-size=1400,900", "--no-first-run",
            "--no-default-browser-check", "--disable-features=Translate", "--disable-session-crashed-bubble"]


APP_WINDOW_WAIT_S = 30       # how long the app window may take to show the page
APP_WINDOW_GONE_S = 45       # no page connected for this long: the window was closed


def open_app_window(url, log=None, sessions=None):
    """Shows the app in a Chromium app window (blocking until it is closed). False when there is no such browser or
    no page ever connected. It does not rely on the browser process staying alive: Edge and Chrome may hand the
    window to another process and exit, so a page connecting to the server is what proves the window is there, and
    the page going away is what says it was closed."""
    sessions = sessions or read_sessions
    browser = find_app_browser()
    if not browser:
        print("No Chromium-based browser found for an app window (Edge, Chrome, Chromium, Brave).", file=sys.stderr)
        return False
    profile = os.path.join(data_dir(), "app-window")
    os.makedirs(profile, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.Popen(app_window_command(browser, url, profile), stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL)
    except OSError as e:
        print(f"Could not start {browser}: {e}", file=sys.stderr)
        return False
    connected = False
    while time.time() - started < APP_WINDOW_WAIT_S:
        if sessions():
            connected = True
            break
        if proc.poll() is not None and time.time() - started > 3:
            break
        time.sleep(0.5)
    if not connected:
        print(f"{os.path.basename(browser)} showed no page within {time.time() - started:.0f} s "
              f"(exit code {proc.poll()}).", file=sys.stderr)
        if proc.poll() is None:
            proc.terminate()
        return False
    last_seen = time.time()
    while time.time() - last_seen < APP_WINDOW_GONE_S:    # until no page has been connected for a while
        if sessions():
            last_seen = time.time()
        time.sleep(1)
    if proc.poll() is None:                              # Edge may stay in the background after its window closed
        proc.terminate()
    return True


def window_modes():
    """The kinds of window to try, in order, from BIKETOOLS_WINDOW."""
    chosen = (os.environ.get("BIKETOOLS_WINDOW") or "auto").strip().lower()
    return {"webview": ["webview"], "app": ["app"], "browser": []}.get(chosen, ["webview", "app"])


def open_dedicated_window(url, log=None):
    """Tries each kind of window in turn; returns the one that was shown ('webview' or 'app'), else None."""
    for mode in window_modes():
        shown = open_window(url, log) if mode == "webview" else open_app_window(url, log)
        if shown:
            return mode
        print(f"No {mode} window; trying the next option.", file=sys.stderr)
    return None


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
    port = choose_port()
    log = open(os.path.join(data_dir(), "launcher.log"), "a")
    server = subprocess.Popen(server_command(port), stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    try:
        if not wait_until_up(port, server):
            print(f"BikeTools Analyzer could not start its server; see {os.path.join(data_dir(), 'launcher.log')}",
                  file=sys.stderr)
            return 1
        url = f"http://127.0.0.1:{port}"
        if not open_dedicated_window(url, log):
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
