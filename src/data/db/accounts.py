"""
Accounts for a shared server: e-mail + password, e-mail verification, password reset and browser sessions.

The desktop app does not use any of this (a rider there has no e-mail or password). On a server a rider IS an account:
`users.name` is the public rider name and the key everything else hangs off, `users.email` is what they sign in with.

* Passwords: salted scrypt (standard library), parameters stored in the hash; the hash never leaves this module.
* Tokens (verify / reset / session): random, only their SHA-256 is stored, so a copy of the database cannot be used to
  sign in. Verify and reset tokens work once and expire; a new reset invalidates the older ones.
* Answers do not reveal which e-mails have an account (sign-up, reset and sign-in say the same thing either way, and an
  unknown e-mail costs the same time as a wrong password). Guessing is throttled per e-mail and per client address.
* Sending the e-mails is `deliver` (see src/data/accounts_mail.py); tests replace it.
"""
import base64
import hashlib
import hmac
import re
import secrets
import time

from src.i18n import t
from src.security import Throttle
from .core import connect
from .dialect import integrity_errors
from .users import get_user_data, save_user

MIN_PASSWORD, MAX_PASSWORD = 8, 200
COST = {'n': 2 ** 15, 'r': 8, 'p': 3}        # OWASP's scrypt guidance: about 32 MiB, a fraction of a second
_MAXMEM = 128 * 1024 * 1024
VERIFY_TTL_S, RESET_TTL_S = 3 * 86400, 3600
SESSION_TTL_S = 30 * 86400
EMAIL = re.compile(r"[^@\s]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,}")
NAME_BAD = re.compile(r"[\x00-\x1f\x7f/\\<>]")

login_throttle = Throttle(limit=10, window=600)       # per e-mail
client_throttle = Throttle(limit=30, window=600)      # per client address
register_throttle = Throttle(limit=8, window=3600)    # per client address
register_unknown = Throttle(limit=300, window=3600)   # all sign-ups together, when the address is not known
reset_throttle = Throttle(limit=5, window=3600)       # per e-mail and per client address


def deliver(kind, email, token, name):
    """Sends the e-mail for `kind` ('verify', 'reset', 'exists'). Replaced by the server (and by tests)."""
    from src.data import accounts_mail
    accounts_mail.send(kind, email, token, name)


# --- passwords ------------------------------------------------------------------------------------------------------

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
        got = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
        return hmac.compare_digest(got, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


_dummy = None


def _spend_time():
    """The work of checking a password, for an e-mail that has no account."""
    global _dummy
    _dummy = _dummy or hash_password("dummy")
    check_hash("x", _dummy)


def password_problem(password):
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        return t("Use at least {n} characters.", n=MIN_PASSWORD)
    if len(password) > MAX_PASSWORD:
        return t("That password is too long.")
    return None


# --- tokens ---------------------------------------------------------------------------------------------------------

def _digest(token):
    return hashlib.sha256(str(token).encode()).hexdigest()


def _new_token():
    return secrets.token_urlsafe(32)


def _issue(conn, user_id, purpose, ttl):
    """A one-time token for `purpose`; older unused ones of the same purpose stop working."""
    conn.execute("DELETE FROM account_tokens WHERE user_id = ? AND purpose = ?", (user_id, purpose))
    token = _new_token()
    conn.execute("INSERT INTO account_tokens (token_hash, user_id, purpose, expires_at) VALUES (?, ?, ?, ?)",
                 (_digest(token), user_id, purpose, int(time.time()) + ttl))
    return token


def _consume(conn, token, purpose):
    """The user id a valid, unused token belongs to (the token is used up), else None."""
    h = _digest(token)
    row = conn.execute("SELECT user_id, expires_at FROM account_tokens WHERE token_hash = ? AND purpose = ?",
                       (h, purpose)).fetchone()
    if not row:
        return None
    conn.execute("DELETE FROM account_tokens WHERE token_hash = ?", (h,))
    return row[0] if row[1] >= time.time() else None


# --- sign-up, verification --------------------------------------------------------------------------------------------

def clean_email(email):
    email = (email or "").strip()
    return email if len(email) <= 254 and EMAIL.fullmatch(email) else None


def clean_name(name):
    name = " ".join((name or "").split())
    return name if 2 <= len(name) <= 40 and not NAME_BAD.search(name) else None


def _by_email(conn, email):
    return conn.execute("SELECT id, name, password_hash, email_verified_at FROM users WHERE lower(email) = lower(?)",
                        (email,)).fetchone()


def register(name, email, password, client=None):
    """Creates an account and sends the verification e-mail. Returns (ok, message). The same message comes back
    whether or not the e-mail already had an account (that owner is told by e-mail instead)."""
    bucket, key = (register_throttle, client) if client else (register_unknown, "all")
    if bucket.blocked(key):
        return False, t("Too many attempts. Wait a while and try again.")
    bucket.fail(key)
    name, mail = clean_name(name), clean_email(email)
    if not name:
        return False, t("Choose a rider name of 2 to 40 characters without / \\ < >.")
    if not mail:
        return False, t("That e-mail address does not look right.")
    problem = password_problem(password)
    if problem:
        return False, problem
    same = t("Almost done: we sent a message to {email}. Open the link in it to confirm your address.", email=mail)
    with connect() as conn:
        existing = _by_email(conn, mail)
        if existing:
            hint = (existing[1], mail)
        elif conn.execute("SELECT 1 FROM users WHERE lower(name) = lower(?)", (name,)).fetchone():
            return False, t("That rider name is taken.")
        else:
            hint = None
    if hint:
        deliver("exists", hint[1], None, hint[0])
        return True, same
    ok, msg = save_user(name)
    if not ok:
        return False, t("That rider name is taken.")
    try:
        with connect() as conn:
            uid = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()[0]
            conn.execute("UPDATE users SET email = ?, password_hash = ?, created_at = ? WHERE id = ?",
                         (mail, hash_password(password), int(time.time()), uid))
            token = _issue(conn, uid, 'verify', VERIFY_TTL_S)
    except integrity_errors():                      # the e-mail was taken a moment ago
        delete_account_row(name)
        return True, same
    deliver("verify", mail, token, name)
    return True, same


def delete_account_row(name):
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE name = ?", (name,))


def verify_email(token):
    """Confirms the address behind a verification token. Returns (ok, message)."""
    with connect() as conn:
        uid = _consume(conn, token, 'verify')
        if uid is None:
            return False, t("This link is not valid any more. Sign in to get a new one.")
        conn.execute("UPDATE users SET email_verified_at = ? WHERE id = ?", (int(time.time()), uid))
    return True, t("Your e-mail address is confirmed. You can sign in.")


def resend_verification(email, client=None):
    """Sends a new verification e-mail if the address has an unverified account. Always the same answer."""
    mail = clean_email(email)
    if mail and not reset_throttle.blocked(mail) and not (client and reset_throttle.blocked(client)):
        reset_throttle.fail(mail)
        if client:
            reset_throttle.fail(client)
        with connect() as conn:
            row = _by_email(conn, mail)
            token = _issue(conn, row[0], 'verify', VERIFY_TTL_S) if row and not row[3] else None
        if token:
            deliver("verify", mail, token, row[1])
    return True, t("If that address has an account waiting to be confirmed, we sent a new link.")


# --- sign-in, sessions ------------------------------------------------------------------------------------------------

OK = "ok"
WRONG = "wrong"
UNVERIFIED = "unverified"
THROTTLED = "throttled"


def login(email, password, client=None, label="browser"):
    """Signs in. Returns (status, token_or_None, rider_name_or_None); status is 'ok', 'wrong', 'unverified' or
    'throttled'. Unknown e-mail, no password and wrong password are all 'wrong', and cost the same time."""
    mail = (email or "").strip().lower()
    if login_throttle.blocked(mail) or (client and client_throttle.blocked(client)):
        return THROTTLED, None, None
    with connect() as conn:
        row = _by_email(conn, mail) if clean_email(mail) else None
    good = bool(row and row[2]) and check_hash(str(password or ""), row[2])
    if not row or not row[2]:
        _spend_time()
    if not good:
        login_throttle.fail(mail)
        if client:
            client_throttle.fail(client)
        return WRONG, None, None
    if not row[3]:
        return UNVERIFIED, None, None
    login_throttle.reset(mail)
    token, now = _new_token(), int(time.time())
    with connect() as conn:
        conn.execute("""INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen, label)
                        VALUES (?, ?, ?, ?, ?, ?)""", (_digest(token), row[0], now, now + SESSION_TTL_S, now, label[:60]))
    return OK, token, row[1]


def session_user(token):
    """The rider name a session token belongs to, or None. A session in use is extended (up to 90 days in all)."""
    if not token:
        return None
    now = int(time.time())
    with connect() as conn:
        row = conn.execute("""SELECT u.name, s.expires_at, s.created_at FROM sessions s JOIN users u ON u.id = s.user_id
                              WHERE s.token_hash = ?""", (_digest(token),)).fetchone()
        if not row:
            return None
        if row[1] < now:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest(token),))
            return None
        if row[1] - now < SESSION_TTL_S // 2 and now - row[2] < 90 * 86400:
            conn.execute("UPDATE sessions SET expires_at = ?, last_seen = ? WHERE token_hash = ?",
                         (now + SESSION_TTL_S, now, _digest(token)))
    return row[0]


def logout(token):
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest(token),))


def logout_everywhere(user_name):
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE name = ?)", (user_name,))


# --- passwords: change, reset -----------------------------------------------------------------------------------------

def change_password(user_name, current, new, keep_token=None):
    """The signed-in rider changes their password; every other session is signed out. Returns (ok, message)."""
    with connect() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE name = ?", (user_name,)).fetchone()
    if not row or not check_hash(str(current or ""), row[1]):
        return False, t("The current password is not right.")
    problem = password_problem(new)
    if problem:
        return False, problem
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), row[0]))
        if keep_token:
            conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash <> ?", (row[0], _digest(keep_token)))
        else:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row[0],))
    return True, t("Password changed.")


def request_password_reset(email, client=None):
    """Sends a reset link if the address has an account. Always the same answer."""
    mail = clean_email(email)
    if mail and not reset_throttle.blocked(mail) and not (client and reset_throttle.blocked(client)):
        reset_throttle.fail(mail)
        if client:
            reset_throttle.fail(client)
        with connect() as conn:
            row = _by_email(conn, mail)
            token = _issue(conn, row[0], 'reset', RESET_TTL_S) if row else None
        if token:
            deliver("reset", mail, token, row[1])
    return True, t("If that address has an account, we sent a link to choose a new password.")


def reset_password(token, new):
    """Sets a new password with a reset token (works once). Signs the rider out everywhere. Returns (ok, message)."""
    problem = password_problem(new)
    if problem:
        return False, problem
    with connect() as conn:
        uid = _consume(conn, token, 'reset')
        if uid is None:
            return False, t("This link is not valid any more. Ask for a new one.")
        # a person who can read the mailbox has proven the address, so it counts as confirmed
        conn.execute("UPDATE users SET password_hash = ?, email_verified_at = COALESCE(email_verified_at, ?) WHERE id = ?",
                     (hash_password(new), int(time.time()), uid))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    return True, t("Password changed. You can sign in.")


def delete_account(user_name, password):
    """Deletes the rider and everything of theirs, after checking the password. Returns (ok, message)."""
    with connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE name = ?", (user_name,)).fetchone()
    if not row or not check_hash(str(password or ""), row[0]):
        return False, t("The password is not right.")
    from .users import delete_user
    delete_user(user_name)
    return True, t("Your account and data were deleted.")


def account_info(user_name):
    """{'email', 'verified'} for the rider, or None."""
    with connect() as conn:
        row = conn.execute("SELECT email, email_verified_at FROM users WHERE name = ?", (user_name,)).fetchone()
    return {"email": row[0], "verified": bool(row[1])} if row else None
