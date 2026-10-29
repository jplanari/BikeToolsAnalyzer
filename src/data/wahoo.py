"""
Wahoo Cloud API client (OAuth 2.0). Unlike Strava it hands out the ORIGINAL FIT file recorded
by the bike computer, so rides keep their full-resolution power, HR and cadence.

Kept free of Streamlit so it can be unit-tested; the sidebar reads the app credentials from
st.secrets and passes a WahooConfig in.

Docs: https://cloud-api.wahooligan.com/
"""
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import requests

AUTHORIZE_URL = "https://api.wahooligan.com/oauth/authorize"
TOKEN_URL = "https://api.wahooligan.com/oauth/token"
API_BASE = "https://api.wahooligan.com/v1"

# offline_data gives a long-lived refresh token; user_read gives the account name
SCOPE = "user_read workouts_read offline_data"

# workout_type_id is an unpublished enum; cycling outdoors is 0. Extend if your indoor rides
# come back as "skipped (not cycling)".
CYCLING_WORKOUT_TYPES = {0}

REFRESH_MARGIN_S = 300
TIMEOUT_S = 30
PER_PAGE = 100


class WahooError(Exception):
    pass


class WahooAuthError(WahooError):
    """Token missing, revoked or lacking scope: the user must reconnect."""


class WahooRateLimitError(WahooError):
    """HTTP 429 or our own budget guard."""


@dataclass
class WahooConfig:
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8501/"


@dataclass
class RateUsage:
    """Last seen X-RateLimit-* headers (the limit window is 5 minutes, then hourly and daily)."""
    limit: int = 25
    remaining: int = 25
    reset_at: float = 0.0
    seen: bool = field(default=False)

    def update(self, headers):
        try:
            if headers.get("X-RateLimit-Limit"):
                self.limit = int(headers["X-RateLimit-Limit"])
            if headers.get("X-RateLimit-Remaining"):
                self.remaining = int(headers["X-RateLimit-Remaining"])
                self.seen = True
            if headers.get("X-RateLimit-Reset"):
                self.reset_at = float(headers["X-RateLimit-Reset"])
        except ValueError:
            pass


# --- OAuth -----------------------------------------------------------------

def authorize_url(cfg, state):
    params = {"client_id": cfg.client_id, "redirect_uri": cfg.redirect_uri, "response_type": "code",
              "scope": SCOPE, "state": state}
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _token_request(data):
    try:
        r = requests.post(TOKEN_URL, data=data, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise WahooError(f"Could not reach Wahoo: {e}") from e
    if r.status_code in (400, 401):
        raise WahooAuthError(f"Wahoo rejected the request ({r.status_code}): {r.text[:200]}")
    if r.status_code == 429:
        raise WahooRateLimitError("Wahoo rate limit reached; try again in a few minutes.")
    if not r.ok:
        raise WahooError(f"Wahoo token endpoint error {r.status_code}")
    return r.json()


def _expires_at(tok, now=None):
    now = time.time() if now is None else now
    return int(tok.get("created_at") or now) + int(tok.get("expires_in") or 7200)


def exchange_code(cfg, code, now=None):
    """Swap the one-time authorization code for tokens and look up the account name."""
    tok = _token_request({"client_id": cfg.client_id, "client_secret": cfg.client_secret, "code": code,
                          "grant_type": "authorization_code", "redirect_uri": cfg.redirect_uri})
    auth = {"access_token": tok["access_token"], "refresh_token": tok["refresh_token"],
            "expires_at": _expires_at(tok, now), "wahoo_user_id": None, "wahoo_name": None}
    try:
        me = api_get("/user", auth["access_token"])
        auth["wahoo_user_id"] = me.get("id")
        auth["wahoo_name"] = " ".join(x for x in [me.get("first"), me.get("last")] if x) or me.get("email")
    except WahooError:
        pass                                   # the name is cosmetic
    return auth


def refresh_if_needed(cfg, auth, now=None):
    """
    Returns (auth, changed). Wahoo refresh tokens are single-use: the old one stops working as
    soon as a new one is issued, so callers must persist `auth` whenever changed is True.
    """
    now = time.time() if now is None else now
    if auth["expires_at"] - now > REFRESH_MARGIN_S:
        return auth, False
    tok = _token_request({"client_id": cfg.client_id, "client_secret": cfg.client_secret,
                          "refresh_token": auth["refresh_token"], "grant_type": "refresh_token"})
    new = dict(auth)
    new["access_token"] = tok["access_token"]
    new["refresh_token"] = tok.get("refresh_token", auth["refresh_token"])
    new["expires_at"] = _expires_at(tok, now)
    return new, True


def revoke(access_token):
    """Removes the app's permission for this account (best effort)."""
    try:
        requests.delete(f"{API_BASE}/permissions", headers={"Authorization": f"Bearer {access_token}"},
                        timeout=TIMEOUT_S)
    except requests.RequestException:
        pass


# --- API -------------------------------------------------------------------

def api_get(path, access_token, params=None, usage=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params,
                         headers={"Authorization": f"Bearer {access_token}"}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise WahooError(f"Could not reach Wahoo: {e}") from e
    if usage is not None:
        usage.update(r.headers)
    if r.status_code == 401:
        raise WahooAuthError("Wahoo authorization expired or was revoked. Please reconnect.")
    if r.status_code == 403:
        raise WahooAuthError("Missing permission (reconnect and allow workout access).")
    if r.status_code == 404:
        raise WahooError("Workout not found.")
    if r.status_code == 429:
        raise WahooRateLimitError("Wahoo rate limit reached; try again in a few minutes.")
    if not r.ok:
        raise WahooError(f"Wahoo API error {r.status_code}")
    return r.json()


def is_cycling(workout):
    return workout.get("workout_type_id") in CYCLING_WORKOUT_TYPES


def list_workouts(access_token, page=1, per_page=PER_PAGE, usage=None):
    """One page: {'workouts': [...], 'total', 'page', 'per_page', 'order', 'sort'}.
    Entries have workout_summary = null; use get_workout for the summary and file URL."""
    return api_get("/workouts", access_token, {"page": page, "per_page": per_page}, usage)


def get_workout(access_token, workout_id, usage=None):
    return api_get(f"/workouts/{workout_id}", access_token, usage=usage)


def fit_url(workout):
    return ((workout.get("workout_summary") or {}).get("file") or {}).get("url")


def download_fit(url):
    """The FIT file sits on a CDN: no auth, and it doesn't count against the API rate limit."""
    try:
        r = requests.get(url, timeout=TIMEOUT_S * 2, allow_redirects=True)
    except requests.RequestException as e:
        raise WahooError(f"Could not download the FIT file: {e}") from e
    if not r.ok or not r.content:
        raise WahooError(f"FIT download failed ({r.status_code})")
    return r.content


def ride_filename(workout_id):
    return f"wahoo_{int(workout_id)}.fit"
