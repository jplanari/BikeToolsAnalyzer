"""The phone API as a WSGI application (standard library only), plus the phone web app's static files.

    POST /api/v1/register            {name, password, invite, weight?, height?}: a new rider, with an invitation code
    POST /api/v1/login               {name, password, label}: a rider with a password gets a token for this phone
    GET /api/v1/ping                 no token; lets the app test a server address
    GET /api/v1/me                   the rider this token belongs to
    GET /api/v1/rides?limit&offset   ride list, newest first, duplicates collapsed
    GET /api/v1/rides/<id>           one ride's stored metrics, zones, power curve, energy
    GET /api/v1/rides/<id>/series    downsampled power / heart rate / speed / elevation and the track
    GET /api/v1/training/load        fitness, fatigue, form, volume
    GET /api/v1/training/power       power curves, records, critical power, profile
    GET /api/v1/connections          Strava / Wahoo: set up on this server? connected? the last import
    POST /api/v1/connections/<p>/start | disconnect | import   connect a provider, remove it, import its rides
    GET /api/v1/oauth/callback       where Strava / Wahoo send the rider back (a small page, no token)
    POST /api/v1/account/delete      {password}: delete the rider and everything of theirs

Everything else is read from `mobile/web`. Riding data is only read: rides cannot be changed or deleted through the API. Accounts, provider connections and imports can. A request needs
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
from . import accounts, jobs, payloads, providers

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
_register_throttle = Throttle(limit=8, window=3600)   # failed sign-ups per address (a wrong invitation counts)
MAX_BODY = 4096
STORE = ("Cache-Control", "no-store")
JSON = "application/json; charset=utf-8"

# (method, path pattern, handler name, needs a token)
_ROUTES = [
    ("GET", re.compile(r"^/ping$"), "ping", False),
    ("GET", re.compile(r"^/me$"), "me", True),
    ("GET", re.compile(r"^/rides$"), "rides", True),
    ("GET", re.compile(r"^/rides/(\d+)$"), "ride", True),
    ("GET", re.compile(r"^/rides/(\d+)/series$"), "series", True),
    ("GET", re.compile(r"^/training/load$"), "load", True),
    ("GET", re.compile(r"^/training/power$"), "power", True),
    ("GET", re.compile(r"^/connections$"), "connections", True),
    ("GET", re.compile(r"^/oauth/callback$"), "oauth_callback", False),
    ("POST", re.compile(r"^/login$"), "login", False),
    ("POST", re.compile(r"^/register$"), "register", False),
    ("POST", re.compile(r"^/connections/(strava|wahoo)/(start|disconnect|import)$"), "connection", True),
    ("POST", re.compile(r"^/account/delete$"), "delete_account", True),
]


class BadRequest(Exception):
    """Something the client got wrong (shown to it); other errors are the server's and are not."""


class Refused(Exception):
    """A request answered with a status other than 200: (status line, message, extra headers)."""
    def __init__(self, status, message, extra=()):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, list(extra)


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


def _body(environ):
    """The JSON object a POST carries (small: MAX_BODY bytes)."""
    try:
        size = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        size = 0
    if size <= 0 or size > MAX_BODY:
        raise BadRequest("expected a small JSON body")
    try:
        body = json.loads(environ["wsgi.input"].read(size) or b"{}")
    except ValueError:
        raise BadRequest("expected a JSON object")
    if not isinstance(body, dict):
        raise BadRequest("expected a JSON object")
    return body


def _bearer(environ):
    header = environ.get("HTTP_AUTHORIZATION", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _trust_proxy():
    return os.environ.get("BIKETOOLS_API_TRUST_PROXY") == "1"


def _client(environ):
    if _trust_proxy():                       # behind a reverse proxy the address is in this header
        fwd = environ.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
        if fwd:
            return fwd
    return environ.get("REMOTE_ADDR", "?")


def _request_base(environ):
    """http(s)://host of this request, for the address providers send the rider back to."""
    scheme = (environ.get("HTTP_X_FORWARDED_PROTO") if _trust_proxy() else None) or environ.get("wsgi.url_scheme", "http")
    host = environ.get("HTTP_HOST") or environ.get("SERVER_NAME", "")
    return f"{scheme}://{host}" if host and scheme in ("http", "https") else ""


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


# --- handlers ---------------------------------------------------------------------------------------------------

def _check(throttle, key):
    if throttle.blocked(key):
        raise Refused("429 Too Many Requests", "too many failed attempts; wait a while",
                      [STORE, ("Retry-After", str(throttle.retry_after(key)))])


def _login(environ):
    body = _body(environ)
    name, password = str(body.get("name") or ""), str(body.get("password") or "")
    label = str(body.get("label") or "Phone")[:db.tokens.MAX_LABEL]
    who, key = _client(environ), name.strip().lower()
    _check(_throttle, who)
    _check(_login_throttle, key)
    if not name or not db.verify_user_password(name, password):
        _throttle.fail(who)
        _login_throttle.fail(key)
        raise Refused("401 Unauthorized", "wrong name or password", [STORE])
    _login_throttle.reset(key)
    token, _ = db.create_api_token(name, label)
    return {"token": token, "me": payloads.me(name)}


def _register(environ):
    body = _body(environ)
    who = _client(environ)
    _check(_register_throttle, who)
    status, result = accounts.register(body.get("name"), body.get("password"), body.get("invite"),
                                       body.get("weight"), body.get("height"))
    if status != "ok":
        _register_throttle.fail(who)
        code = {"invalid": "400 Bad Request", "taken": "409 Conflict", "invite": "403 Forbidden"}[status]
        raise Refused(code, result, [STORE])
    label = str(body.get("label") or "Phone")[:db.tokens.MAX_LABEL]
    token, _ = db.create_api_token(result, label)
    return {"token": token, "me": payloads.me(result)}


def _connection(user, provider, action, environ):
    if action == "start":
        url = providers.start(user, provider, providers.public_url(_request_base(environ)))
        if not url:
            raise Refused("409 Conflict", f"{provider} is not set up on this server", [STORE])
        return {"url": url}
    if action == "disconnect":
        providers.disconnect(user, provider)
        return {"ok": True}
    body = _body(environ)
    try:
        days = None if body.get("days") in (None, "", "new") else max(1, int(body["days"]))
    except (TypeError, ValueError):
        raise BadRequest("days must be a whole number")
    ok, reason = jobs.start(user, provider, days)
    if not ok:
        raise Refused("409 Conflict", {"running": "an import is already running",
                                       "not_connected": f"{provider} is not connected"}.get(reason, reason), [STORE])
    return {"ok": True}


def _callback_page(environ, query):
    from src.i18n import use_language, available_languages
    first = (environ.get("HTTP_ACCEPT_LANGUAGE", "en").split(",")[0].split("-")[0] or "en").lower()
    lang = first if first in available_languages() else "en"
    g = lambda k: query.get(k, [None])[0]
    with use_language(lang):
        from src.i18n import t
        ok, message = providers.finish(g("state"), g("code"), g("error"), g("scope"), _request_base(environ))
        title = t("Connected") if ok else t("Not connected")
        hint = t("You can close this page and go back to the BikeTools app.")
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>BikeTools</title><body style="font:18px system-ui;max-width:30rem;margin:20vh auto;padding:0 1rem;text-align:center">'
            f'<h1>{"✅" if ok else "⚠️"} {esc(title)}</h1><p>{esc(message)}</p><p style="color:#666">{esc(hint)}</p></body></html>')
    return html


def _dispatch(name, user, args, query, environ):
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
    if name == "connections":
        return {**providers.status(user), "job": jobs.status(user)}
    if name == "login":
        return _login(environ)
    if name == "register":
        return _register(environ)
    if name == "connection":
        return _connection(user, args[0], args[1], environ)
    if name == "delete_account":
        who, key = _client(environ), user.lower()
        _check(_login_throttle, key)
        if not accounts.delete_account(user, str(_body(environ).get("password") or "")):
            _login_throttle.fail(key)
            raise Refused("403 Forbidden", "wrong password", [STORE])
        return {"ok": True}
    return None


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    origin = environ.get("HTTP_ORIGIN", "").rstrip("/")
    headers = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer")]
    if origin and origin in allowed_origins():
        headers += [("Access-Control-Allow-Origin", origin), ("Vary", "Origin"),
                    ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
                    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), ("Access-Control-Max-Age", "600")]

    def respond(status, ctype, body, extra=()):
        data = body if isinstance(body, bytes) else body.encode()
        h = headers + [("Content-Type", ctype), ("Content-Length", str(len(data)))] + list(extra)
        start_response(status, h)
        return [b"" if method == "HEAD" else data]

    def fail(status, message, extra=()):
        return respond(status, JSON, json.dumps({"error": message}), extra)

    if method == "OPTIONS":
        return respond("204 No Content", "text/plain", b"")
    if not (path == API or path.startswith(API + "/")):
        if method not in ("GET", "HEAD"):
            return fail("405 Method Not Allowed", "not allowed", [("Allow", "GET, HEAD, OPTIONS")])
        status, ctype, body = _static(path)
        return respond(status, ctype, body, [("Cache-Control", "no-cache")])

    sub = path[len(API):] or "/"
    matches = [(m, r) for r in _ROUTES if (m := r[1].match(sub))]
    wanted = "GET" if method == "HEAD" else method
    hit = next(((m, r) for m, r in matches if r[0] == wanted), None)
    if not hit:
        if matches:
            allow = sorted({r[0] for _, r in matches} | ({"HEAD"} if any(r[0] == "GET" for _, r in matches) else set()) | {"OPTIONS"})
            return fail("405 Method Not Allowed", "method not allowed here", [STORE, ("Allow", ", ".join(allow))])
        return fail("404 Not Found", "unknown endpoint", [STORE])
    m, (_, _, name, needs_auth) = hit

    user = None
    try:
        if needs_auth:
            who = _client(environ)
            _check(_throttle, who)
            user = db.lookup_api_token(_bearer(environ))
            if not user:
                _throttle.fail(who)
                raise Refused("401 Unauthorized", "missing or invalid token", [STORE, ("WWW-Authenticate", "Bearer")])
        if name == "oauth_callback":
            return respond("200 OK", "text/html; charset=utf-8", _callback_page(environ, _query(environ)), [STORE])
        result = _dispatch(name, user, m.groups(), _query(environ), environ)
    except Refused as e:
        return fail(e.status, e.message, e.extra)
    except BadRequest as e:
        return fail("400 Bad Request", str(e), [STORE])
    except Exception:
        log.exception("API error on %s", path)
        return fail("500 Internal Server Error", "something went wrong on the server", [STORE])
    if result is None:
        return fail("404 Not Found", "not found", [STORE])
    return respond("200 OK", JSON, json.dumps(result, separators=(",", ":")), [STORE])
