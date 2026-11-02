"""Invitation codes: the way new riders get an account on a shared server. Only a hash is stored; the code is shown
once when it is made. A code has a number of uses and an optional expiry."""
import hashlib
import secrets
import time

from .core import connect

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"        # no 0/O, 1/I: a code is read aloud or copied by eye
CODE_LENGTH = 10
MAX_USES = 1000


def normalize(code):
    return "".join(ch for ch in str(code or "").upper() if ch.isalnum())


def _hash(code):
    return hashlib.sha256(("invite:" + normalize(code)).encode()).hexdigest()


def pretty(raw):
    return f"{raw[:5]}-{raw[5:]}"


def create_invite(uses=1, days=None, label=""):
    """Makes a code. Returns (code, id); the code is not recoverable afterwards."""
    uses = max(1, min(int(uses), MAX_USES))
    raw = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
    now = int(time.time())
    expires = now + int(float(days) * 86400) if days else None
    with connect() as conn:
        cur = conn.execute("INSERT INTO invites (code_hash, uses_left, expires_at, label, created_at) VALUES (?, ?, ?, ?, ?)",
                           (_hash(raw), uses, expires, (label or "").strip()[:60], now))
        return pretty(raw), cur.lastrowid


def list_invites():
    """Codes that can still be used: [{id, label, uses_left, expires_at, created_at}], newest first."""
    now = int(time.time())
    with connect() as conn:
        rows = conn.execute("""SELECT id, label, uses_left, expires_at, created_at FROM invites
                               WHERE uses_left > 0 AND (expires_at IS NULL OR expires_at > ?) ORDER BY id DESC""",
                            (now,)).fetchall()
    return [dict(zip(('id', 'label', 'uses_left', 'expires_at', 'created_at'), r)) for r in rows]


def revoke_invite(invite_id):
    with connect() as conn:
        return conn.execute("DELETE FROM invites WHERE id = ?", (int(invite_id),)).rowcount > 0


def use_invite(code):
    """Spends one use of a valid code; True if it was valid. Atomic, so two sign-ups cannot share the last use."""
    if len(normalize(code)) != CODE_LENGTH:
        return False
    with connect() as conn:
        return conn.execute("""UPDATE invites SET uses_left = uses_left - 1
                               WHERE code_hash = ? AND uses_left > 0 AND (expires_at IS NULL OR expires_at > ?)""",
                            (_hash(code), int(time.time()))).rowcount == 1


def refund_invite(code):
    """Gives a use back (the sign-up it was spent on did not happen)."""
    with connect() as conn:
        conn.execute("UPDATE invites SET uses_left = uses_left + 1 WHERE code_hash = ?", (_hash(code),))
