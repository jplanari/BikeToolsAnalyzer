"""Strava and Wahoo for riders on a shared server: the sign-in round trip lands on the server itself
(/api/v1/oauth/callback) and the import runs there. The API app's credentials come from the environment
(BIKETOOLS_STRAVA_CLIENT_ID / _CLIENT_SECRET, BIKETOOLS_WAHOO_...), so the server can have its own app, or, when
those are not set, from the ones saved in the database. BIKETOOLS_PUBLIC_URL is the address the providers send the
rider back to (https is required by Wahoo); without it the address of the request is used."""
import os
import threading
import time

from src.data import db
from src.data.providers import credentials as cr
from src.data.providers import strava as sv
from src.data.providers import wahoo as wh
from src.i18n import t

NAMES = ("strava", "wahoo")
WAHOO_PREFIX = "wahoo."
CALLBACK_PATH = "/api/v1/oauth/callback"
_redirects = {}                       # state -> redirect address used at the start (the same one must be used to finish)
_lock = threading.Lock()


def public_url(request_base=""):
    return (os.environ.get("BIKETOOLS_PUBLIC_URL") or request_base or "").rstrip("/")


def credentials(provider):
    """{'client_id', 'client_secret', 'timezone'} or None."""
    prefix = f"BIKETOOLS_{provider.upper()}_"
    cid, secret = os.environ.get(prefix + "CLIENT_ID", "").strip(), os.environ.get(prefix + "CLIENT_SECRET", "").strip()
    if cid and secret:
        return {"client_id": cid, "client_secret": secret, "timezone": os.environ.get(prefix + "TIMEZONE", "")}
    return cr.resolve(provider, db.get_app_credentials(provider), None)


def configured(provider):
    return credentials(provider) is not None


def config(provider, redirect=""):
    c = credentials(provider)
    if not c:
        return None
    redirect = redirect or cr.PROVIDERS[provider]["redirect"]
    if provider == "strava":
        return sv.StravaConfig(c["client_id"], c["client_secret"], redirect)
    return wh.WahooConfig(c["client_id"], c["client_secret"], redirect, c.get("timezone") or "")


def status(user_name):
    """{'strava': {'configured', 'connected', 'name'}, 'wahoo': {...}} for the connections page."""
    out = {}
    for p in NAMES:
        auth = db.get_strava_auth(user_name) if p == "strava" else db.get_wahoo_auth(user_name)
        out[p] = {"configured": configured(p), "connected": bool(auth),
                  "name": ((auth.get("athlete_name") if p == "strava" else auth.get("wahoo_name")) if auth else None) or None}
    return out


def start(user_name, provider, public):
    """The address the rider opens (in the phone's browser) to allow access, or None when not set up here."""
    redirect = f"{public}{CALLBACK_PATH}"
    cfg = config(provider, redirect)
    state = db.create_oauth_state(user_name)
    if cfg is None or not state or not public:
        return None
    with _lock:
        _redirects[state] = redirect
        if len(_redirects) > 500:
            _redirects.pop(next(iter(_redirects)))
    if provider == "wahoo":
        return wh.authorize_url(cfg, WAHOO_PREFIX + state)
    return sv.authorize_url(cfg, state)


def finish(state, code, error, scope, request_public=""):
    """Ends the round trip at the callback. Returns (ok, message) for the page shown to the rider."""
    provider = "wahoo" if (state or "").startswith(WAHOO_PREFIX) else "strava"
    raw = (state or "")[len(WAHOO_PREFIX):] if provider == "wahoo" else (state or "")
    with _lock:
        redirect = _redirects.pop(raw, None)
    user = db.consume_oauth_state(raw)
    label = cr.PROVIDERS[provider]["label"]
    if not user:
        return False, t("The request expired or did not come from this server. Start again from the app.")
    if error or not code:
        return False, t("{label} access was not granted.", label=label)
    if provider == "strava" and not sv.granted_scope_ok(scope):
        return False, t("Connect again and keep the box that lets BikeTools read your activities ticked.")
    cfg = config(provider, redirect or f"{public_url(request_public)}{CALLBACK_PATH}")
    if cfg is None:
        return False, t("{label} is not set up on this server.", label=label)
    try:
        auth = sv.exchange_code(cfg, code) if provider == "strava" else wh.exchange_code(cfg, code)
    except (sv.StravaError, wh.WahooError) as e:
        return False, t("{label} did not accept the connection: {e}", label=label, e=str(e)[:160])
    (db.save_strava_auth if provider == "strava" else db.save_wahoo_auth)(user, auth)
    return True, t("{label} is connected.", label=label)


def disconnect(user_name, provider):
    if provider == "strava":
        auth, cfg = db.get_strava_auth(user_name), config("strava")
        if auth and cfg:
            sv.revoke(cfg, auth["access_token"])
        db.delete_strava_auth(user_name)
    else:
        db.delete_wahoo_auth(user_name)
