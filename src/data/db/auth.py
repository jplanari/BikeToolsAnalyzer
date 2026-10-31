"""Strava / Wahoo / Polar connections, app credentials, OAuth states and imported-id lookups."""
from src.data import secrets_store as _sec
import pandas as pd
import re
import secrets
import sqlite3
import time
from .core import connect
from .users import get_user_data

# --- Strava authorization --------------------------------------------------
# Tokens are stored per rider profile in the local database (plain text, like
# the rest of the local data). Strava may rotate the refresh token on every
# refresh, so save_strava_auth() must be called whenever it changes.

def save_strava_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO strava_auth (user_id, athlete_id, athlete_name, access_token, refresh_token, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                athlete_id = COALESCE(excluded.athlete_id, strava_auth.athlete_id),
                athlete_name = COALESCE(excluded.athlete_name, strava_auth.athlete_name),
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at
        ''', (user['id'], auth.get('athlete_id'), auth.get('athlete_name'),
              _sec.encrypt(auth['access_token']), _sec.encrypt(auth['refresh_token']), int(auth['expires_at'])))
    return True

def _decrypt_fields(row, fields):
    """A table row as a dict with its secret columns decrypted (None when there is no row)."""
    if not row:
        return None
    out = dict(row)
    for f in fields:
        out[f] = _sec.decrypt(out.get(f))
    return out

_SECRET_COLUMNS = (("strava_auth", "user_id", ("access_token", "refresh_token")),
                   ("wahoo_auth", "user_id", ("access_token", "refresh_token")),
                   ("polar_auth", "user_id", ("access_token",)),
                   ("app_credentials", "provider", ("client_secret",)),
                   ("smtp_settings", "id", ("password",)))

def encrypt_plaintext_secrets():
    """Encrypts secrets stored before encryption existed. Idempotent; does nothing without a usable key (and
    creates no key while there is nothing to encrypt). Returns how many values were encrypted."""
    with connect() as conn:
        pending = []
        for table, key, cols in _SECRET_COLUMNS:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
                continue
            for row in conn.execute(f"SELECT {key}, {', '.join(cols)} FROM {table}").fetchall():
                pending += [(table, key, row[0], col, value) for col, value in zip(cols, row[1:])
                            if value and not _sec.is_encrypted(value)]
        if not pending or not _sec.available():
            return 0
        for table, key, ident, col, value in pending:
            conn.execute(f"UPDATE {table} SET {col} = ? WHERE {key} = ?", (_sec.encrypt(value), ident))
    return len(pending)

def get_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM strava_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return _decrypt_fields(row, ('access_token', 'refresh_token'))

def delete_strava_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM strava_auth WHERE user_id = ?", (user['id'],))

# --- Wahoo authorization (same storage rules; Wahoo refresh tokens are single-use) ---

def save_wahoo_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO wahoo_auth (user_id, wahoo_user_id, wahoo_name, access_token, refresh_token, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                wahoo_user_id = COALESCE(excluded.wahoo_user_id, wahoo_auth.wahoo_user_id),
                wahoo_name = COALESCE(excluded.wahoo_name, wahoo_auth.wahoo_name),
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at
        ''', (user['id'], auth.get('wahoo_user_id'), auth.get('wahoo_name'),
              _sec.encrypt(auth['access_token']), _sec.encrypt(auth['refresh_token']), int(auth['expires_at'])))
    return True

def get_wahoo_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM wahoo_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return _decrypt_fields(row, ('access_token', 'refresh_token'))

def delete_wahoo_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM wahoo_auth WHERE user_id = ?", (user['id'],))

# --- API app credentials entered in the app (instead of .streamlit/secrets.toml) ---

def save_app_credentials(provider, client_id, client_secret, redirect_uri=None, timezone=None):
    with connect() as conn:
        conn.execute('''
            INSERT INTO app_credentials (provider, client_id, client_secret, redirect_uri, timezone)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET client_id = excluded.client_id, client_secret = excluded.client_secret,
                redirect_uri = excluded.redirect_uri, timezone = excluded.timezone
        ''', (provider, client_id.strip(), _sec.encrypt(client_secret.strip()), redirect_uri, timezone))

def get_app_credentials(provider):
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM app_credentials WHERE provider = ?", (provider,)).fetchone()
    return _decrypt_fields(row, ('client_secret',))

def delete_app_credentials(provider):
    with connect() as conn:
        conn.execute("DELETE FROM app_credentials WHERE provider = ?", (provider,))

# --- Polar authorization (tokens do not expire; stored like the other providers') ---

def save_polar_auth(user_name, auth):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO polar_auth (user_id, polar_user_id, access_token) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET polar_user_id = excluded.polar_user_id, access_token = excluded.access_token
        ''', (user['id'], auth.get('polar_user_id'), _sec.encrypt(auth['access_token'])))
    return True

def get_polar_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM polar_auth WHERE user_id = ?", (user['id'],)).fetchone()
    return _decrypt_fields(row, ('access_token',))

def delete_polar_auth(user_name):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("DELETE FROM polar_auth WHERE user_id = ?", (user['id'],))

def polar_id_from_filename(filename):
    """'polar_aB3x9.fit' -> 'aB3x9' (rides imported through Polar AccessLink)."""
    m = re.fullmatch(r"polar_([A-Za-z0-9]+)\.fit", filename or "")
    return m.group(1) if m else None

def get_imported_polar_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT filename FROM rides WHERE user_id = ? AND filename LIKE 'polar\\_%.fit' ESCAPE '\\'",
                            (user['id'],)).fetchall()
    return {i for i in (polar_id_from_filename(r[0]) for r in rows) if i is not None}

def get_wahoo_sent(user_name):
    """{planned session id: {plan_id, workout_id, starts, sent_at}} for sessions sent to Wahoo."""
    user = get_user_data(user_name)
    if not user:
        return {}
    with connect() as conn:
        rows = conn.execute("""SELECT w.session_id, w.plan_id, w.workout_id, w.starts, w.sent_at
                               FROM wahoo_sent w JOIN planned_sessions p ON p.id = w.session_id
                               WHERE p.user_id = ?""", (user['id'],)).fetchall()
    return {r[0]: {'plan_id': r[1], 'workout_id': r[2], 'starts': r[3], 'sent_at': r[4]} for r in rows}

def save_wahoo_sent(session_id, plan_id, workout_id, starts, sent_at):
    with connect() as conn:
        conn.execute("""INSERT INTO wahoo_sent (session_id, plan_id, workout_id, starts, sent_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET plan_id = excluded.plan_id,
                            workout_id = excluded.workout_id, starts = excluded.starts, sent_at = excluded.sent_at""",
                     (int(session_id), int(plan_id), int(workout_id), starts, sent_at))

def delete_wahoo_sent(session_id):
    with connect() as conn:
        conn.execute("DELETE FROM wahoo_sent WHERE session_id = ?", (int(session_id),))

def wahoo_id_from_filename(filename):
    """'wahoo_123.fit' -> 123 (rides imported through the Wahoo Cloud API)."""
    m = re.fullmatch(r"wahoo_(\d+)\.fit", (filename or "").lower())
    return int(m.group(1)) if m else None

def get_imported_wahoo_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT filename FROM rides WHERE user_id = ? AND filename LIKE 'wahoo\\_%.fit' ESCAPE '\\'",
                            (user['id'],)).fetchall()
    return {i for i in (wahoo_id_from_filename(r[0]) for r in rows) if i is not None}

def latest_wahoo_ride_date(user_name):
    """Start (naive UTC) of the newest ride imported from Wahoo, or None."""
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        row = conn.execute("SELECT MAX(date_time) FROM rides WHERE user_id = ? AND date_time IS NOT NULL "
                           "AND filename LIKE 'wahoo\\_%.fit' ESCAPE '\\'", (user['id'],)).fetchone()
    if not row or not row[0]:
        return None
    ts = pd.to_datetime(row[0], errors='coerce', utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)


OAUTH_STATE_TTL_S = 15 * 60

def create_oauth_state(user_name):
    """
    One-time CSRF token for the OAuth round trip. Stored in the DB rather
    than st.session_state because Strava redirects back into a new session.
    """
    user = get_user_data(user_name)
    if not user:
        return None
    state = secrets.token_urlsafe(24)
    now = int(time.time())
    with connect() as conn:
        conn.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - OAUTH_STATE_TTL_S,))
        conn.execute("INSERT INTO oauth_states (state, user_id, created_at) VALUES (?, ?, ?)",
                     (state, user['id'], now))
    return state

def consume_oauth_state(state):
    """Returns the user name the state was issued for (once), else None."""
    now = int(time.time())
    with connect() as conn:
        row = conn.execute('''SELECT u.name, s.created_at FROM oauth_states s
                              JOIN users u ON u.id = s.user_id WHERE s.state = ?''', (state,)).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
    if not row or now - row[1] > OAUTH_STATE_TTL_S:
        return None
    return row[0]

def get_imported_strava_ids(user_name):
    user = get_user_data(user_name)
    if not user:
        return set()
    with connect() as conn:
        rows = conn.execute("SELECT strava_activity_id FROM rides WHERE user_id = ? "
                            "AND strava_activity_id IS NOT NULL", (user['id'],)).fetchall()
    return {r[0] for r in rows}

def latest_strava_ride_date(user_name):
    """Start (naive UTC) of the newest ride imported from Strava, or None."""
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        row = conn.execute("SELECT MAX(date_time) FROM rides WHERE user_id = ? AND strava_activity_id IS NOT NULL "
                           "AND date_time IS NOT NULL", (user['id'],)).fetchone()
    if not row or not row[0]:
        return None
    ts = pd.to_datetime(row[0], errors='coerce', utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)
