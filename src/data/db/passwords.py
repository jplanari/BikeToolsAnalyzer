"""Rider passwords. Optional: a rider without one behaves as before. The password is kept as a salted scrypt hash
(standard library, parameters stored in the hash so they can be raised later); the hash never leaves this module
(get_user_data only says whether there is one). Changing or removing a password revokes the rider's phone tokens."""
import base64
import hashlib
import hmac
import secrets
import time

from src.i18n import t
from .core import connect

MIN_LENGTH, MAX_LENGTH = 8, 200
COST = {'n': 2 ** 15, 'r': 8, 'p': 3}        # OWASP's scrypt guidance: about 32 MiB, a fraction of a second
_MAXMEM = 128 * 1024 * 1024
_b64 = lambda b: base64.b64encode(b).decode()


def _scrypt(password, salt, n, r, p):
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=32)


def hash_password(password):
    salt = secrets.token_bytes(16)
    c = COST
    return f"scrypt${c['n']}${c['r']}${c['p']}${_b64(salt)}${_b64(_scrypt(password, salt, c['n'], c['r'], c['p']))}"


def check_hash(password, stored):
    """True if `password` matches a hash_password() string. Never raises on a malformed hash."""
    try:
        scheme, n, r, p, salt, digest = (stored or "").split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        got = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
        return hmac.compare_digest(got, expected)
    except (ValueError, TypeError):
        return False


_DUMMY = None


def _stored(user_name):
    with connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE name = ?", (user_name,)).fetchone()
    return row[0] if row else None


def has_password(user_name):
    return bool(_stored(user_name))


def verify_user_password(user_name, password):
    """True only for an existing rider with a password that matches. An unknown rider or one without a password
    costs the same time as a wrong password, so the answer does not reveal which riders exist."""
    global _DUMMY
    stored = _stored(user_name)
    if not stored:
        _DUMMY = _DUMMY or hash_password("dummy")
        check_hash(str(password or ""), _DUMMY)
        return False
    return check_hash(str(password or ""), stored)


def _problem(password):
    if not isinstance(password, str) or len(password) < MIN_LENGTH:
        return t("Use at least {n} characters.", n=MIN_LENGTH)
    if len(password) > MAX_LENGTH:
        return t("That password is too long.")
    return None


def set_user_password(user_name, new_password, current=None, force=False):
    """Sets or changes a rider's password. When one is already set, `current` must match (force=True skips that, for
    the command line on a server). Returns (ok, message)."""
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE name = ?", (user_name,)).fetchone():
            return False, t("User data not found.")
    if has_password(user_name) and not force and not verify_user_password(user_name, current):
        return False, t("The current password is not right.")
    problem = _problem(new_password)
    if problem:
        return False, problem
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE name = ?", (hash_password(new_password), user_name))
        conn.execute("UPDATE api_tokens SET revoked = 1 WHERE user_id = (SELECT id FROM users WHERE name = ?)", (user_name,))
    return True, t("Password saved. Phones signed in before have to sign in again.")


def clear_user_password(user_name, current=None, force=False):
    """Removes a rider's password (then anyone can open the rider again). Returns (ok, message)."""
    if not has_password(user_name):
        return True, ""
    if not force and not verify_user_password(user_name, current):
        return False, t("The current password is not right.")
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = NULL WHERE name = ?", (user_name,))
        conn.execute("UPDATE api_tokens SET revoked = 1 WHERE user_id = (SELECT id FROM users WHERE name = ?)", (user_name,))
    return True, t("Password removed.")
