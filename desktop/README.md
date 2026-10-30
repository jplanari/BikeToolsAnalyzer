# Desktop app

BikeTools Analyzer as a program with its own window: no terminal, no Python on the target machine.

Run from source (native window through pywebview; Qt on Linux; falls back to your browser if no window can open):

    pip install -r desktop/requirements-desktop.txt
    python desktop/launcher.py

Build the standalone app for the OS you are on (PyInstaller cannot cross-compile):

    pyinstaller desktop/biketools.spec --noconfirm   # result in dist/BikeToolsAnalyzer/

The *Desktop build* GitHub workflow does this for Windows, macOS and Linux (Actions → Desktop build → Run
workflow, or push a `v*` tag) and attaches the three zips.

**Linux native window:** Qt needs one system library on Ubuntu/Debian 22.04+: `sudo apt install libxcb-cursor0`. Without it
(or without any window backend) the app opens in your browser instead.

**Shortcut on the desktop / app menu.** After building (or unzipping the downloaded build), run once:

    dist/BikeToolsAnalyzer/BikeToolsAnalyzer --install-shortcut        (Windows: BikeToolsAnalyzer.exe)

It creates *BikeTools Analyzer* on the Desktop and in the Start menu (Windows), in `~/Applications` (macOS), or in the
application menu and `~/Desktop` (Linux; some desktops ask once to "Allow launching"). The shortcut points at the
folder where the app is now, so run the command again if you move it. From source: `python desktop/launcher.py --install-shortcut`.

**Your data** is kept per user, not inside the app: `%APPDATA%\BikeToolsAnalyzer` (Windows),
`~/Library/Application Support/BikeToolsAnalyzer` (macOS), `~/.local/share/BikeToolsAnalyzer` (Linux); set
`BIKETOOLS_DATA` to use another folder. It holds `biketools.db` and `backups/`. For Strava/Wahoo keys put
`.streamlit/secrets.toml` in that same folder. Updating = replace the app folder; the data stays.

Builds are unsigned: Windows SmartScreen and macOS Gatekeeper warn on first start (right-click → Open).
