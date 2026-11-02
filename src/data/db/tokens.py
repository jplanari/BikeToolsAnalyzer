"""Access tokens for the phone app. One token per paired phone, tied to one rider; only a SHA-256 hash is stored,
so a copy of the database does not hand out working tokens. The token itself is shown once, when it is made."""
import hashlib
import secrets
import time

from .core import connect
from .users import get_user_data

TOKEN_PREFIX = "bt_"
LAST_USED_RESOLUTION_S = 60          # last_used_at is refreshed at most this often, to avoid a write per request
MAX_LABEL = 60


def hash_token(token):
    return hashlib.sha256(str(token).encode()).hexdigest()


def create_api_token(user_name, label=""):
    """Makes a token for a rider. Returns (token, id); (None, None) if the rider does not exist. The plain token is
    not recoverable afterwards."""
    user = get_user_data(user_name)
    if not user:
        return None, None
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with connect() as conn:
        cur = conn.execute("INSERT INTO api_tokens (user_id, token_hash, label, created_at) VALUES (?, ?, ?, ?)",
                           (user['id'], hash_token(token), (label or "").strip()[:MAX_LABEL], int(time.time())))
        return token, cur.lastrowid


def list_api_tokens(user_name):
    """[{id, label, created_at, last_used_at}] of a rider's live (not revoked) tokens, newest first."""
    user = get_user_data(user_name)
    if not user:
        return []
    with connect() as conn:
        rows = conn.execute("""SELECT id, label, created_at, last_used_at FROM api_tokens
                               WHERE user_id = ? AND revoked = 0 ORDER BY id DESC""", (user['id'],)).fetchall()
    return [{'id': r[0], 'label': r[1] or "", 'created_at': r[2], 'last_used_at': r[3]} for r in rows]


def revoke_api_token(user_name, token_id):
    """Stops a token working. Only the owner's tokens can be revoked. Returns True if one was revoked."""
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        cur = conn.execute("UPDATE api_tokens SET revoked = 1 WHERE id = ? AND user_id = ? AND revoked = 0",
                           (int(token_id), user['id']))
        return cur.rowcount > 0


def lookup_api_token(token):
    """The rider's name for a live token, or None."""
    if not token or not str(token).startswith(TOKEN_PREFIX):
        return None
    digest = hash_token(token)
    with connect() as conn:
        row = conn.execute("""SELECT t.id, u.name, t.last_used_at FROM api_tokens t JOIN users u ON u.id = t.user_id
                              WHERE t.token_hash = ? AND t.revoked = 0""", (digest,)).fetchone()
        if not row:
            return None
        now = int(time.time())
        if row[2] is None or now - row[2] >= LAST_USED_RESOLUTION_S:
            conn.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (now, row[0]))
        return row[1]
