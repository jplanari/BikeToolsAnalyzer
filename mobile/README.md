# BikeTools on your phone

A phone-first version of the dashboard: **Rides** (list and ride summary with power, heart rate, speed, elevation, route, zones, best efforts and the energy/fuel estimate), **Training** (fitness, fatigue, form, ramp rate, volume) and **Power** (power curves, critical power, profile, records). English, Spanish and Catalan; light and dark. It keeps the last data it loaded, so it still opens without a connection.

It is **read-only**. Uploading rides, importing from Strava/Wahoo, planning, the fuel log and the other analyses (durability, intensity distribution, climbs, aerodynamics...) stay in the desktop app.

```
phone (browser or Android app)  ──HTTP, token──▶  phone server (your computer)  ──▶  your BikeTools database
```

## 1. Start the phone server

In the desktop app open **User Corner → 📱 Phone app** and press **Start the phone server** (Windows may ask to allow network access). It runs while BikeTools is open.

Without the desktop app (a NAS, a small server, a terminal):

    python -m src.api serve --host 0.0.0.0 --port 8765

The database is the same one the desktop app uses (`BIKETOOLS_DB`).

## 2. Pair the phone

**Riders with a password:** set it in User Corner → 🔒 Password (or `python -m src.api password set "<rider>"` on a server). Then just choose Connect, type the server address, the rider name and the password. The server gives that phone its own token, so you can still revoke phones one by one; changing the password signs every phone out. The app refuses password sign-in over plain http outside a private network.

Or, without a password, in the same place press **Create access token**, give the phone a name and copy the token (it is shown once; only a hash is stored). Then either

- open the web link it shows in the phone's browser (`http://<computer>:8765/#/pair?...`) — the app opens in the browser and can be added to the home screen; or
- install the Android app (below), choose **Connect** and type the server address and paste the token, or open the `biketools://pair?...` link.

Each token belongs to one rider and sees only that rider's data. **Revoke** a token in the same place and that phone stops working.

## Android app (APK)

The APK is built by GitHub Actions from this folder (a Capacitor wrapper around `web/`).

- **Get it:** open the repository's **Releases** page on the phone, download `BikeTools-<version>.apk` and open it (Android asks once to allow installs from the browser).
- **Publish a new one (maintainers):** push a tag `mobile-v1.0.0`; the *Android APK* workflow builds the APK and attaches it to a release. Pull requests touching `mobile/` build a debug APK as a workflow artifact.
- **Signing:** add the repository secrets `ANDROID_KEYSTORE_BASE64` (the keystore file, base64), `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS` and `ANDROID_KEY_PASSWORD`. Without them the APK is signed with a throwaway debug key: it installs, but a later build cannot update it (uninstall first). Keep the keystore: losing it means users must uninstall to update. Create one with `keytool -genkeypair -v -keystore biketools.keystore -alias biketools -keyalg RSA -keysize 2048 -validity 10000`, then `base64 -w0 biketools.keystore`.

## Reaching your computer

- **Same Wi-Fi:** use the address the app suggests (`http://192.168.x.y:8765`). Plain http is acceptable on a network you trust.
- **Away from home:** do not open the port on your router. Use a VPN that joins the phone and the computer, such as **Tailscale** (then use the computer's Tailscale address), or put an https reverse proxy / tunnel (Caddy, Cloudflare Tunnel) in front and use its `https://` address. The app warns when you type a plain-http address that is not on a private network.
- If the phone page is served from a different web origin, list it in `BIKETOOLS_API_ORIGINS` (comma separated). The Android app needs nothing.

## Several riders on one server

The same server can host a group: each rider creates their own account from the phone, connects Strava and/or Wahoo there, and sees only their own data.

1. **Keep a computer on** (a home server or a spare machine) running the phone server (`python -m src.api`, or the desktop app's switch).
2. **Make it reachable.** Use a VPN such as Tailscale, or a tunnel such as Cloudflare Tunnel that gives you an https address. Do not expose a plain http port to the internet: passwords are refused over unencrypted public connections.
3. **Tell the server its public address**: `BIKETOOLS_PUBLIC_URL=https://bike.example.net`. Strava and Wahoo send riders back to `<that address>/api/v1/oauth/callback`.
4. **Register one API application with each provider** and give the server its keys (riders never see them):
   `BIKETOOLS_STRAVA_CLIENT_ID`, `BIKETOOLS_STRAVA_CLIENT_SECRET`, `BIKETOOLS_WAHOO_CLIENT_ID`, `BIKETOOLS_WAHOO_CLIENT_SECRET`.
   Strava accepts one callback domain per application, so use the domain of step 3. Wahoo only accepts an https redirect, so it needs the tunnel; Strava works with any address you registered.
5. **Invite people.** `python -m src.api invite create [--uses N] [--days D] [--label TEXT]` prints a code like `K7M2Q-9XWPA`; the same screen exists in User Corner → Invitations. `invite list` and `invite revoke <id>` manage them. Codes are stored hashed, work a limited number of times, and can expire.
6. A new rider opens the app, taps **I have an invitation code**, chooses a rider name and password, then **Settings → Data sources → Connect Strava / Wahoo** and **Import rides**. Rides are fetched by the server in the background; progress shows on the same page.

Riders can delete their own account (Settings → Delete my account), which erases their data on the server. Sign-up attempts are rate-limited per address.

## Security notes

- Tokens are random (256 bit), stored hashed, scoped to one rider, revocable, and never written to the server log. Ten wrong tokens from one address lock that address out for a minute (behind a reverse proxy set `BIKETOOLS_API_TRUST_PROXY=1` so the real client address is used).
- Passwords are optional, stored as salted scrypt hashes (never returned by any API) and checked in constant time; wrong passwords cost the same whether or not the rider exists. Ten wrong attempts from one address, or twenty for one rider within ten minutes, are refused for a while.
- Ride data is read-only through the API. The only things a phone can change are its own account (sign-up, deletion), its Strava/Wahoo connections and the imports it starts.
- Invitation codes are random, stored hashed, limited in uses and time; guessing them is rate-limited per address. Provider keys stay on the server and the OAuth `state` is single-use and tied to the rider who started it.
- Over plain http, anyone on the path can read the token and your data: that is why the VPN/https advice above matters.

## Development

    python -m pytest tests/test_api.py tests/test_mobile_locales.py tests/test_mobile_android.py
    python -m pytest tests/test_mobile_e2e.py        # needs Playwright + Chromium; skipped otherwise

- `src/api/` — `app.py` (routes, auth, CORS, static files), `payloads.py` (the JSON), `runner.py` (start/stop), `__main__.py` (CLI: `serve`, `token create|list|revoke`).
- `web/` — the app: no build step, plain ES modules and hand-drawn SVG charts. Open it through the phone server, not from disk.
- Text: the English text is the translation key, as on the desktop. After changing text in `web/js` run `python -m mobile.build_locales` (it reads the desktop catalogs and `i18n/phone.json`, and fails on anything untranslated).
- `android/` — `package.json`, `capacitor.config.json`, `patch.py` (deep link, version, icon, signing). The Android project itself is generated in CI (`npx cap add android`), so no generated files are committed.

## What is not verified

The APK build and the Android-only behaviour (the `biketools://` link, installing, the native HTTP bridge) are exercised by the *Android APK* workflow and need a device to confirm; the web app and API are covered by the automated tests, including a real browser at phone size.
