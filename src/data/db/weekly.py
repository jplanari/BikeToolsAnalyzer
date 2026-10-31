"""SMTP account and the weekly e-mail subscriptions."""
from src.data import secrets_store as _sec
import pandas as pd
import sqlite3
from .auth import _decrypt_fields
from .core import connect
from .users import get_user_data

# --- Weekly e-mail (one SMTP account for the app, one subscription per user) ---

def save_smtp_settings(host, port, security, username, password, sender):
    with connect() as conn:
        conn.execute("""
            INSERT INTO smtp_settings (id, host, port, security, username, password, sender)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET host = excluded.host, port = excluded.port, security = excluded.security,
                username = excluded.username, password = excluded.password, sender = excluded.sender
        """, (host.strip(), int(port), security, (username or '').strip() or None, _sec.encrypt(password or None), sender.strip()))

def get_smtp_settings():
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM smtp_settings WHERE id = 1").fetchone()
    return _decrypt_fields(row, ('password',))

def delete_smtp_settings():
    with connect() as conn:
        conn.execute("DELETE FROM smtp_settings")

def save_weekly_email(user_name, enabled, recipient):
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("""
            INSERT INTO weekly_email (user_id, enabled, recipient) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET enabled = excluded.enabled, recipient = excluded.recipient
        """, (user['id'], 1 if enabled else 0, recipient.strip()))

def get_weekly_email(user_name):
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM weekly_email WHERE user_id = ?", (user['id'],)).fetchone()
    return dict(row) if row else None

def users_with_weekly_email():
    """Names of the users whose weekly e-mail is switched on."""
    with connect() as conn:
        rows = conn.execute("""SELECT u.name FROM weekly_email w JOIN users u ON u.id = w.user_id
                               WHERE w.enabled = 1 ORDER BY u.name""").fetchall()
    return [r[0] for r in rows]

def claim_weekly_email(user_name, week):
    """
    Reserves the e-mail of the week starting `week` (ISO date) for this process. True for exactly one caller:
    the update only matches while last_week is older, so two browser tabs or two users of one server never send
    the same week twice. Read get_weekly_email()['last_week'] first if you may need to release the claim.
    """
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        cur = conn.execute("""UPDATE weekly_email SET last_week = ?, last_status = 'sending', last_attempt = ?
                              WHERE user_id = ? AND enabled = 1 AND (last_week IS NULL OR last_week < ?)""",
                           (week, pd.Timestamp.now().isoformat(timespec='seconds'), user['id'], week))
        claimed = cur.rowcount == 1
    return claimed

def finish_weekly_email(user_name, status):
    """Stores the outcome of a send that worked."""
    user = get_user_data(user_name)
    if user:
        with connect() as conn:
            conn.execute("UPDATE weekly_email SET last_status = ? WHERE user_id = ?", (status, user['id']))

def release_weekly_email(user_name, status, previous_week):
    """A send that failed: put last_week back (None = never sent) so the next app visit tries again."""
    user = get_user_data(user_name)
    if user:
        with connect() as conn:
            conn.execute("UPDATE weekly_email SET last_status = ?, last_week = ? WHERE user_id = ?",
                         (status, previous_week, user['id']))
