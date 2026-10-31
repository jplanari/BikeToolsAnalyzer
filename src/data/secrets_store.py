"""
Encryption at rest for the tokens and passwords kept in the database (OAuth tokens, client secrets, the SMTP
password), so a copy of the .db file or of a backup does not hand out working credentials.

The key is NOT in the database: it comes from BIKETOOLS_SECRET_KEY, or from a key file created next to the
database (mode 0600) the first time something is stored. Keep that file out of backups and sync folders that
the database file is shared through, or the protection is gone. Lose it and the stored connections cannot be
read any more: you reconnect Strava / Wahoo / Polar and type the secrets again.

A stored value is 'enc:v1:<Fernet token>'. A value without that prefix is plain text from before encryption
and is still read (and upgraded by encrypt_plaintext_secrets).
"""
import base64
import os
import warnings

PREFIX = "enc:v1:"
KEY_ENV = "BIKETOOLS_SECRET_KEY"
_cache = {}


def _key_path():
    from src.data import db
    return os.path.join(os.path.dirname(os.path.abspath(db.DB_FILE)), "biketools.key")


def _fernet():
    """Fernet for the current key, or None when encryption is unavailable (no `cryptography`, no writable folder)."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None
    env = os.environ.get(KEY_ENV)
    if env:
        key = env.encode()
    else:
        path = _key_path()
        if path in _cache:
            return _cache[path]
        try:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    key = fh.read().strip()
            else:
                key = Fernet.generate_key()
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(key)
        except OSError as e:
            warnings.warn(f"Cannot keep an encryption key ({e}); secrets are stored unencrypted.")
            return None
    try:
        f = Fernet(key)
    except (ValueError, TypeError):
        warnings.warn("The encryption key is not valid; secrets are stored unencrypted.")
        return None
    if not env:
        _cache[_key_path()] = f
    return f


def available():
    return _fernet() is not None


def is_encrypted(value):
    return isinstance(value, str) and value.startswith(PREFIX)


def encrypt(value):
    """Plain text -> stored form. None / '' stay as they are; without a usable key the text is kept plain."""
    if value is None or value == "" or is_encrypted(value):
        return value
    f = _fernet()
    if f is None:
        return value
    return PREFIX + f.encrypt(str(value).encode()).decode()


def decrypt(value):
    """Stored form -> plain text. Plain values pass through; a value that cannot be decrypted (wrong or lost key)
    reads as '' so callers see 'not connected' instead of crashing."""
    if not is_encrypted(value):
        return value
    f = _fernet()
    if f is None:
        return ""
    try:
        return f.decrypt(value[len(PREFIX):].encode()).decode()
    except Exception:                       # InvalidToken: another key
        return ""


def new_key():
    """A fresh key for BIKETOOLS_SECRET_KEY."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()
