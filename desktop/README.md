# Desktop app

BikeTools Analyzer as a program with its own window: no terminal, no Python on the target machine.

Run from source (native window through pywebview; Qt on Linux; falls back to your browser if no window can open):

    pip install -r desktop/requirements-desktop.txt
    python desktop/launcher.py

Build the standalone app for the OS you are on (PyInstaller cannot cross-compile):

    pyinstaller desktop/biketools.spec --noconfirm   # result in dist/BikeToolsAnalyzer/

The *Desktop build* GitHub workflow does this for Windows, macOS and Linux (Actions → Desktop build → Run
workflow, or push a `v*` tag) and attaches the three zips.

**Your data** is kept per user, not inside the app: `%APPDATA%\BikeToolsAnalyzer` (Windows),
`~/Library/Application Support/BikeToolsAnalyzer` (macOS), `~/.local/share/BikeToolsAnalyzer` (Linux); set
`BIKETOOLS_DATA` to use another folder. It holds `biketools.db` and `backups/`. For Strava/Wahoo keys put
`.streamlit/secrets.toml` in that same folder. Updating = replace the app folder; the data stays.

Builds are unsigned: Windows SmartScreen and macOS Gatekeeper warn on first start (right-click → Open).
