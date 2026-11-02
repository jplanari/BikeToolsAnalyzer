"""The phone API as a WSGI application (standard library only), plus the phone web app's static files.

    GET /api/v1/ping                 no token; lets the app test a server address
    GET /api/v1/me                   the rider this token belongs to
    GET /api/v1/rides?limit&offset   ride list, newest first, duplicates collapsed
    GET /api/v1/rides/<id>           one ride's stored metrics, zones, power curve, energy
    GET /api/v1/rides/<id>/series    downsampled power / heart rate / speed / elevation and the track
    GET /api/v1/training/load        fitness, fatigue, form, volume
    GET /api/v1/training/power       power curves, records, critical power, profile

Everything else is read from `mobile/web`. The API only reads; nothing can be changed through it. A request needs
`Authorization: Bearer <token>` (tokens are made in the desktop app's User corner, or with `python -m src.api token
create`); each token belongs to one rider and sees only that rider's data."""
import json
import logging
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

from src.data import db
from . import payloads

log = logging.getLogger("biketools.api")

API = "/api/v1"
WEB_ROOT = Path(os.environ.get("BIKETOOLS_WEB_ROOT") or Path(__file__).resolve().parents[2] / "mobile" / "web")
APP_ORIGINS = ("https://localhost", "http://localhost", "capacitor://localhost")    # the Android app's own origins
FAIL_LIMIT, FAIL_WINDOW_S = 10, 60          # failed token checks allowed per address per window


def allowed_origins():
    extra = [o.strip().rstrip("/") for o in os.environ.get("BIKETOOLS_API_ORIGINS", "").split(",") if o.strip()]
    return set(APP_ORIGINS) | set(extra)


class Throttle:
    """Slows down guessing: after FAIL_LIMIT bad tokens from one address within FAIL_WINDOW_S, it is refused."""
    def __init__(self, limit=FAIL_LIMIT, window=FAIL_WINDOW_S):
        self.limit, self.window = limit, window
        self._fails = {}
        self._lock = threading.Lock()

    def _recent(self, key, now):
        hits = [t for t in self._fails.get(key, []) if now - t < self.window]
        if hits:
            self._fails[key] = hits
        else:
            self._fails.pop(key, None)
        return hits

    def blocked(self, key, now=None):
        with self._lock:
            return len(self._recent(key, now or time.time())) >= self.limit

    def fail(self, key, now=None):
        now = now or time.time()
        with self._lock:
            self._recent(key, now)
            self._fails.setdefault(key, []).append(now)
            if len(self._fails) > 5000:         # bound the memory a scan from many addresses could use
                self._fails.clear()


_throttle = Throttle()

_ROUTES = [
    (re.compile(r"^/ping$"), "ping", False),
    (re.compile(r"^/me$"), "me", True),
    (re.compile(r"^/rides$"), "rides", True),
    (re.compile(r"^/rides/(\d+)$"), "ride", True),
    (re.compile(r"^/rides/(\d+)/series$"), "series", True),
    (re.compile(r"^/training/load$"), "load", True),
    (re.compile(r"^/training/power$"), "power", True),
]


class BadRequest(Exception):
    """A query parameter the client got wrong (shown to it); other errors are the server's and are not."""


def _int_param(query, name, default, low, high):
    raw = query.get(name, [None])[0]
    if raw in (None, ""):
        return default
    try:
        return max(low, min(int(raw), high))
    except ValueError:
        raise BadRequest(f"{name} must be a whole number")


def _query(environ):
    from urllib.parse import parse_qs
    return parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)


def _dispatch(name, user, args, query):
    if name == "ping":
        return {"ok": True, "app": "BikeTools", "api": 1}
    if name == "me":
        return payloads.me(user)
    if name == "rides":
        return payloads.rides(user, _int_param(query, "limit", 30, 1, payloads.MAX_PAGE),
                              _int_param(query, "offset", 0, 0, 10 ** 7))
    if name == "ride":
        return payloads.ride(user, int(args[0]))
    if name == "series":
        return payloads.series(user, int(args[0]))
    if name == "load":
        return payloads.training_load(user, _int_param(query, "days", payloads.LOAD_DAYS, 7, 1100))
    if name == "power":
        return payloads.power(user)
    return None


def _bearer(environ):
    header = environ.get("HTTP_AUTHORIZATION", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _client(environ):
    if os.environ.get("BIKETOOLS_API_TRUST_PROXY") == "1":          # behind a reverse proxy the address is in this header
        fwd = environ.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
        if fwd:
            return fwd
    return environ.get("REMOTE_ADDR", "?")


def _static(path):
    """(status, content type, body) for a file of the web app; no way out of WEB_ROOT."""
    rel = path.lstrip("/") or "index.html"
    root = WEB_ROOT.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        return "404 Not Found", "text/plain; charset=utf-8", b"not found"
    if target.is_dir():
        target = target / "index.html"
    if not target.is_file():
        return "404 Not Found", "text/plain; charset=utf-8", b"not found"
    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype in ("application/javascript", "application/json", "image/svg+xml"):
        ctype += "; charset=utf-8" if "charset" not in ctype else ""
    return "200 OK", ctype, target.read_bytes()


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    origin = environ.get("HTTP_ORIGIN", "").rstrip("/")
    headers = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer")]
    if origin and origin in allowed_origins():
        headers += [("Access-Control-Allow-Origin", origin), ("Vary", "Origin"),
                    ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
                    ("Access-Control-Allow-Methods", "GET, OPTIONS"), ("Access-Control-Max-Age", "600")]

    def respond(status, ctype, body, extra=()):
        data = body if isinstance(body, bytes) else body.encode()
        h = headers + [("Content-Type", ctype), ("Content-Length", str(len(data)))] + list(extra)
        start_response(status, h)
        return [b"" if method == "HEAD" else data]

    def fail(status, message, extra=()):
        return respond(status, "application/json; charset=utf-8", json.dumps({"error": message}), extra)

    if method == "OPTIONS":
        return respond("204 No Content", "text/plain", b"")
    if method not in ("GET", "HEAD"):
        return fail("405 Method Not Allowed", "read-only API", [("Allow", "GET, HEAD, OPTIONS")])

    if not (path == API or path.startswith(API + "/")):
        status, ctype, body = _static(path)
        return respond(status, ctype, body, [("Cache-Control", "no-cache")])

    sub = path[len(API):] or "/"
    for pattern, name, needs_auth in _ROUTES:
        m = pattern.match(sub)
        if m:
            break
    else:
        return fail("404 Not Found", "unknown endpoint", [("Cache-Control", "no-store")])

    store = ("Cache-Control", "no-store")
    user = None
    if needs_auth:
        who = _client(environ)
        if _throttle.blocked(who):
            return fail("429 Too Many Requests", "too many failed attempts; wait a minute", [store, ("Retry-After", "60")])
        user = db.lookup_api_token(_bearer(environ))
        if not user:
            _throttle.fail(who)
            return fail("401 Unauthorized", "missing or invalid token", [store, ("WWW-Authenticate", "Bearer")])
    try:
        result = _dispatch(name, user, m.groups(), _query(environ))
    except BadRequest as e:
        return fail("400 Bad Request", str(e), [store])
    except Exception:
        log.exception("API error on %s", path)
        return fail("500 Internal Server Error", "something went wrong on the server", [store])
    if result is None:
        return fail("404 Not Found", "not found", [store])
    return respond("200 OK", "application/json; charset=utf-8", json.dumps(result, separators=(",", ":")), [store])
