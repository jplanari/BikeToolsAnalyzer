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

**Strava / Wahoo sign-in.** The desktop app listens on port **8501** (set `BIKETOOLS_PORT` to change it), so
`redirect_uri = "http://localhost:8501/"` works for **Strava** as it is (callback domain `localhost`).

**Wahoo only accepts an https callback**, which a local app cannot offer without a certificate warning. The repo therefore
includes a static forwarding page, `docs/oauth/index.html`, that receives the redirect on https and sends the browser on to
`http://localhost:8501/` with the same `?code=…&state=…`. One-time setup:

1. GitHub → repository *Settings → Pages* → source *Deploy from a branch*, branch `main`, folder `/docs`.
2. In the Wahoo developer portal add this callback URL: `https://jplanari.github.io/BikeToolsAnalyzer/oauth/`
3. In `secrets.toml`, section `[wahoo]`: `redirect_uri = "https://jplanari.github.io/BikeToolsAnalyzer/oauth/"`

(Replace the user and repository name if you fork it.) The page sees only the one-time code, which is useless without the
client secret that stays in your `secrets.toml`; the forwarding target is fixed to localhost:8501. The app must be
running when you finish the sign-in. Publishing `/docs` also publishes the physics report that lives there.

**Your data** is kept per user, not inside the app: `%APPDATA%\BikeToolsAnalyzer` (Windows),
`~/Library/Application Support/BikeToolsAnalyzer` (macOS), `~/.local/share/BikeToolsAnalyzer` (Linux); set
`BIKETOOLS_DATA` to use another folder. It holds `biketools.db` and `backups/`. For Strava/Wahoo keys put
`.streamlit/secrets.toml` in that same folder. Updating = replace the app folder; the data stays.

Builds are unsigned: Windows SmartScreen and macOS Gatekeeper warn on first start (right-click → Open).
