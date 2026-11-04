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

## Deploying with Docker (recommended)

The `deploy/` folder has everything for one server with Docker: the app, PostgreSQL, and Caddy, which gets the HTTPS
certificate by itself. Only Caddy is open to the internet (ports 80 and 443).

1. Point a domain name at the server and open ports 80 and 443.
2. `cd deploy`, then `cp .env.example .env` and fill it in: the domain, a database password, a `BIKETOOLS_SECRET_KEY`
   (the file says how to make one) and the mail account. Keep a copy of the secret key somewhere safe, apart from backups.
3. `touch secrets.toml` (put the `[strava]`, `[wahoo]`, `[polar]` sections in it when you have them; Docker needs the
   file to exist).
4. `docker compose up -d --build`, then open `https://your-domain`.
5. Check it: `docker compose ps` (all three running, `app` healthy) and `docker compose logs -f app`.

Update to a new version: `git pull && docker compose up -d --build`. The database lives in a Docker volume and is migrated
when the app starts. Without Docker, `deploy/biketools.service` is a systemd unit with the same settings.

**Backups.** `deploy/backup.sh /var/backups/biketools` writes a dated `pg_dump` and keeps the last 14; run it from cron
(`15 3 * * *  cd /opt/biketools/deploy && ./backup.sh /var/backups/biketools`) and copy the folder off the machine. Restore
into an empty database to test it before you need it:

    docker compose exec -T db createdb -U biketools -T template0 restore_test
    docker compose exec -T db pg_restore -U biketools -d restore_test --no-owner < biketools-DATE.dump

**Not tested by me.** These files were written without a Docker daemon available, so the first real build is the test.
`docker compose config` accepts them. Expect to fix small things on the first run and tell me what it says.

## Limits on uploads and storage

Strangers upload files to a public server, so these are on by default in server mode (all in MB; `0` switches one off):

| Variable | Default | Meaning |
|---|---|---|
| `BIKETOOLS_MAX_ACTIVITY_MB` | 50 | The biggest single ride file, after unzipping |
| `BIKETOOLS_MAX_UNPACKED_MB` | 500 | Everything unpacked from one upload (a zip that expands to gigabytes is cut off) |
| `BIKETOOLS_RIDER_QUOTA_MB` | 500 | The most original ride files one rider can store; the User Corner shows how much is used |

The upload box itself stops at 200 MB (`--server.maxUploadSize`), and Caddy at 250 MB. When a rider reaches their quota the
upload says so once and stops; deleting rides makes room. For comparison, one heavy cyclist uses about 17 MB of ride files a
year, so 500 MB is decades. On the desktop the limits are the old, looser ones (200 MB and 2 GB) and there is no quota.

## Not covered yet

- No admin screen: look at the database for the list of accounts; there is no way to ban or message riders.
- Sessions of 30 days mean a deleted browser cookie is the only "forget me" on a shared computer: use Sign out there.
- The cookie is written by a small script in the page; if a browser blocks it the rider has to sign in again each time
  the tab is reloaded.
- Not tried against a real Streamlit deployment from the development container (no Streamlit there): the UI tests run in CI,
  but the first real deployment deserves a walk-through (sign up, confirm, sign in, reload, sign out, reset password).
- Backups, connection pooling and a SQLite → PostgreSQL import are still to do.
