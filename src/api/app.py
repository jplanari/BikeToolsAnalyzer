"""The phone API as a WSGI application (standard library only), plus the phone web app's static files.

    POST /api/v1/login               {name, password, label}: a rider with a password gets a token for this phone
    GET /api/v1/ping                 no token; lets the app test a server address
    GET /api/v1/me                   the rider this token belongs to
    GET /api/v1/rides?limit&offset   ride list, newest first, duplicates collapsed
    GET /api/v1/rides/<id>           one ride's stored metrics, zones, power curve, energy
    GET /api/v1/rides/<id>/series    downsampled power / heart rate / speed / elevation and the track
    GET /api/v1/training/load        fitness, fatigue, form, volume
    GET /api/v1/training/power       power curves, records, critical power, profile

Everything else is read from `mobile/web`. Apart from signing in, the API only reads; nothing can be changed through it. A request needs
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
from src.security import Throttle
from . import payloads

log = logging.getLogger("biketools.api")

API = "/api/v1"
WEB_ROOT = Path(os.environ.get("BIKETOOLS_WEB_ROOT") or Path(__file__).resolve().parents[2] / "mobile" / "web")
APP_ORIGINS = ("https://localhost", "http://localhost", "capacitor://localhost")    # the Android app's own origins
FAIL_LIMIT, FAIL_WINDOW_S = 10, 60          # failed token checks allowed per address per window


def allowed_origins():
    extra = [o.strip().rstrip("/") for o in os.environ.get("BIKETOOLS_API_ORIGINS", "").split(",") if o.strip()]
    return set(APP_ORIGINS) | set(extra)

_throttle = Throttle()
_login_throttle = Throttle(limit=20, window=600)      # per rider name, so a password cannot be guessed from many addresses
MAX_BODY = 4096

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


def _login(environ, respond, fail):
    """A rider with a password signs in from the phone and gets a token (the same kind the desktop app hands out)."""
    store = ("Cache-Control", "no-store")
    who = _client(environ)
    try:
        size = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        size = 0
    if size <= 0 or size > MAX_BODY:
        return fail("400 Bad Request", "expected a small JSON body", [store])
    try:
        body = json.loads(environ["wsgi.input"].read(size) or b"{}")
        name, password = str(body.get("name") or ""), str(body.get("password") or "")
        label = str(body.get("label") or "Phone")[:db.tokens.MAX_LABEL]
    except (ValueError, AttributeError, TypeError):
        return fail("400 Bad Request", "expected a JSON object with name and password", [store])
    key = name.strip().lower()
    for throttle, k in ((_throttle, who), (_login_throttle, key)):
        if throttle.blocked(k):
            return fail("429 Too Many Requests", "too many failed attempts; wait a while", [store, ("Retry-After", str(throttle.retry_after(k)))])
    if not name or not db.verify_user_password(name, password):
        _throttle.fail(who)
        _login_throttle.fail(key)
        return fail("401 Unauthorized", "wrong name or password", [store])
    _login_throttle.reset(key)
    token, _ = db.create_api_token(name, label)
    return respond("200 OK", "application/json; charset=utf-8", json.dumps({"token": token, "me": payloads.me(name)}), [store])


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
    if method == "POST" and path == API + "/login":
        return _login(environ, respond, fail)
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
