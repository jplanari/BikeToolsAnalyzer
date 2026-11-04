"""Settings of a shared server (as opposed to the desktop app). Everything comes from environment variables, so the
person running the server needs no screens for it, and riders never see or change these.

    BIKETOOLS_SERVER_MODE=1        riders must sign in; no rider list, no server settings in the app
    BIKETOOLS_PUBLIC_URL           the address riders use, e.g. https://bike.example.net (links in e-mails, OAuth)
    BIKETOOLS_TRUST_PROXY=1        the reverse proxy in front sets X-Forwarded-For (needed to throttle per address)
    BIKETOOLS_SMTP_HOST / _PORT / _SECURITY (starttls|ssl|none) / _USER / _PASSWORD / _SENDER   the mail account
    BIKETOOLS_DATABASE_URL         PostgreSQL (see docs/DATABASE.md)
"""
import os

TRUE = ("1", "true", "yes", "on")


def _flag(name):
    return os.environ.get(name, "").strip().lower() in TRUE


def server_mode():
    return _flag("BIKETOOLS_SERVER_MODE")


def trust_proxy():
    return _flag("BIKETOOLS_TRUST_PROXY")


def _megabytes(name, on_server, on_desktop):
    """A size limit in bytes from an environment variable in MB (0 = no limit); the defaults differ for a shared server."""
    default = on_server if server_mode() else on_desktop
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return int(value * 1024 * 1024) if value > 0 else 0


def max_activity_bytes():
    """BIKETOOLS_MAX_ACTIVITY_MB: the biggest single ride file (after unzipping). 50 MB on a server, 200 MB on the desktop."""
    return _megabytes("BIKETOOLS_MAX_ACTIVITY_MB", 50, 200)


def max_unpacked_bytes():
    """BIKETOOLS_MAX_UNPACKED_MB: everything unpacked from one upload (zip-bomb guard). 500 MB on a server, 2 GB on the desktop."""
    return _megabytes("BIKETOOLS_MAX_UNPACKED_MB", 500, 2048)


def rider_quota_bytes():
    """BIKETOOLS_RIDER_QUOTA_MB: the most original ride files one rider may store. 500 MB on a server, none on the desktop."""
    return _megabytes("BIKETOOLS_RIDER_QUOTA_MB", 500, 0)


def smtp_from_env():
    """The mail account of the server as the same dict the database gives, or None when not configured here."""
    host, sender = os.environ.get("BIKETOOLS_SMTP_HOST", "").strip(), os.environ.get("BIKETOOLS_SMTP_SENDER", "").strip()
    if not host or not sender:
        return None
    return {"host": host, "port": int(os.environ.get("BIKETOOLS_SMTP_PORT", "587") or 587),
            "security": os.environ.get("BIKETOOLS_SMTP_SECURITY", "starttls").strip() or "starttls",
            "username": os.environ.get("BIKETOOLS_SMTP_USER", "").strip() or None,
            "password": os.environ.get("BIKETOOLS_SMTP_PASSWORD") or None, "sender": sender}


def client_address(headers):
    """The visitor's address from the proxy's X-Forwarded-For (first entry), or None when it cannot be trusted."""
    if not trust_proxy():
        return None
    forwarded = (headers or {}).get("X-Forwarded-For") or (headers or {}).get("x-forwarded-for") or ""
    first = forwarded.split(",")[0].strip()
    return first[:64] or None
