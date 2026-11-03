# Running a shared server

BikeTools can run as one web app for many riders: each signs up with an e-mail address, confirms it, and sees only
their own rides. The desktop app is unchanged; everything here is switched on by environment variables.

## What you need

- A machine that stays on, with Python 3.11+ (or a container), and a domain name with HTTPS (a reverse proxy such as
  Caddy or nginx in front of Streamlit; sign-in cookies and OAuth need HTTPS).
- A PostgreSQL database (`docs/DATABASE.md`). SQLite works for a handful of riders but is not meant for this.
- A mail account the server can send from (for the confirmation and password-reset e-mails).
- Optionally, API apps at Strava / Wahoo / Polar (one set for the whole server) if riders should connect them.

## Settings

| Variable | Meaning |
|---|---|
| `BIKETOOLS_SERVER_MODE=1` | Riders must sign in. The rider list, creating/deleting riders, provider credentials and the mail form disappear from the app. |
| `BIKETOOLS_PUBLIC_URL` | The address riders use, e.g. `https://bike.example.net`. Used in the links of e-mails. |
| `BIKETOOLS_DATABASE_URL` | `postgresql://user:password@host:5432/dbname` |
| `BIKETOOLS_SECRET_KEY` | A Fernet key; encrypts stored OAuth tokens. Keep it out of database backups. |
| `BIKETOOLS_TRUST_PROXY=1` | Set when a reverse proxy sends `X-Forwarded-For`. Lets sign-in attempts be limited per address. Never set it when Streamlit is reachable without the proxy. |
| `BIKETOOLS_SMTP_HOST`, `_PORT`, `_SECURITY` (`starttls`/`ssl`/`none`), `_USER`, `_PASSWORD`, `_SENDER` | The mail account. Host and sender are required. |

Strava / Wahoo / Polar credentials go in `.streamlit/secrets.toml` (the existing `[strava]`, `[wahoo]`, `[polar]`
sections) with `redirect_uri` set to the public address. In server mode credentials entered through the app are ignored.

Start: `BIKETOOLS_SERVER_MODE=1 ... streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
and point the proxy at it (WebSocket support on).

## How accounts behave

- Sign-up asks for a rider name, e-mail and password (8+ characters, stored as scrypt hashes). The e-mail is confirmed
  through a single-use link valid 3 days; nobody signs in before that.
- Sign-in, sign-up and "forgot password" give the same answer whether or not an address exists, and are rate-limited.
- A session lasts 30 days from the last visit (90 at most) and lives in a cookie holding a random token; the database
  keeps only its hash. Changing the password signs out the other sessions; "Sign out everywhere" ends all.
- A rider can delete their account (password required) and everything of theirs goes with it.
- The Strava / Wahoo / Polar return only attaches to the rider who started it.

## Not covered yet

- No admin screen: look at the database for the list of accounts; there is no way to ban or message riders.
- Sessions of 30 days mean a deleted browser cookie is the only "forget me" on a shared computer: use Sign out there.
- The cookie is written by a small script in the page; if a browser blocks it the rider has to sign in again each time
  the tab is reloaded.
- Not tried against a real Streamlit deployment from the development container (no Streamlit there): the UI tests run in CI,
  but the first real deployment deserves a walk-through (sign up, confirm, sign in, reload, sign out, reset password).
- Backups, connection pooling and a SQLite → PostgreSQL import are still to do.
