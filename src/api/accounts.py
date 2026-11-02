"""Creating and deleting riders through the phone API (the server side of "sign up with an invitation")."""
import re

from src.data import db
from src.i18n import t

NAME_MIN, NAME_MAX = 2, 40
_BAD_NAME = re.compile(r"[\x00-\x1f\x7f<>/\\]")
WEIGHT_RANGE, HEIGHT_RANGE = (25.0, 250.0), (100.0, 250.0)


def clean_name(name):
    name = " ".join(str(name or "").split())
    if not NAME_MIN <= len(name) <= NAME_MAX or _BAD_NAME.search(name):
        return None
    return name


def name_taken(name):
    with db.connect() as conn:
        return conn.execute("SELECT 1 FROM users WHERE lower(name) = lower(?)", (name,)).fetchone() is not None


def _number(value, low_high, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if low_high[0] <= v <= low_high[1] else default


def register(name, password, invite, weight=None, height=None):
    """Creates a rider with a password, spending one use of the invitation. Returns (status, rider name or message)
    where status is 'ok', 'invalid' (bad input), 'invite' (code not valid) or 'taken'."""
    name = clean_name(name)
    if not name:
        return "invalid", "the name must be 2 to 40 characters without / \\ < >"
    if len(str(password or "")) < db.passwords.MIN_LENGTH or len(str(password)) > db.passwords.MAX_LENGTH:
        return "invalid", f"the password needs at least {db.passwords.MIN_LENGTH} characters"
    if name_taken(name):
        return "taken", "that name is already taken"
    if not db.use_invite(invite):
        return "invite", "that invitation code is not valid"
    ok, _ = db.save_user(name, None, None, _number(weight, WEIGHT_RANGE, 70.0), _number(height, HEIGHT_RANGE, 175.0))
    if not ok:                                              # lost a race for the name: give the invitation back
        db.refund_invite(invite)
        return "taken", "that name is already taken"
    db.set_user_password(name, password, force=True)
    return "ok", name


def delete_account(name, password):
    """Deletes the rider and everything of theirs after checking the password. Returns True when done."""
    if not db.verify_user_password(name, password):
        return False
    db.delete_user(name)
    return True
