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
# plans_write/workouts_write let the app send planned sessions to the head unit
SCOPE = "user_read workouts_read offline_data plans_read plans_write workouts_write"

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


class WahooNotFound(WahooError):
    """404: the plan or workout is gone (e.g. deleted in the Wahoo app)."""


class WahooRateLimitError(WahooError):
    """HTTP 429 or our own budget guard."""


@dataclass
class WahooConfig:
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8501/"
    timezone: str = ""           # IANA name for scheduling planned sessions; empty = this computer's


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

def api_request(method, path, access_token, params=None, data=None, usage=None):
    """One API call; returns the JSON body (None for 204)."""
    try:
        r = requests.request(method, f"{API_BASE}{path}", params=params, data=data,
                             headers={"Authorization": f"Bearer {access_token}"}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise WahooError(f"Could not reach Wahoo: {e}") from e
    if usage is not None:
        usage.update(r.headers)
    if r.status_code == 401:
        raise WahooAuthError("Wahoo authorization expired or was revoked. Please reconnect.")
    if r.status_code == 403:
        raise WahooAuthError("Wahoo did not allow this (reconnect Wahoo to grant the new permissions).")
    if r.status_code == 404:
        raise WahooNotFound("Not found on Wahoo.")
    if r.status_code == 429:
        raise WahooRateLimitError("Wahoo rate limit reached; try again in a few minutes.")
    if not r.ok:
        detail = ""
        try:
            detail = f": {r.text[:200]}"
        except Exception:
            pass
        raise WahooError(f"Wahoo API error {r.status_code}{detail}")
    return None if r.status_code == 204 or not r.content else r.json()


def api_get(path, access_token, params=None, usage=None):
    return api_request("GET", path, access_token, params=params, usage=usage)


def is_cycling(workout):
    return workout.get("workout_type_id") in CYCLING_WORKOUT_TYPES


def is_planned_only(workout):
    """
    True when the list entry is clearly a scheduled workout that was not ridden: it starts in the
    future, or its summary (when the list carries one) has no recorded file or active time.
    Entries with no summary at all are undecided (see may_be_planned).
    """
    from datetime import datetime, timezone
    try:
        start = datetime.fromisoformat(str(workout.get("starts")).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if start > datetime.now(timezone.utc):
            return True
    except (TypeError, ValueError):
        pass
    summary = workout.get("workout_summary")
    if isinstance(summary, dict):
        try:
            active = float(summary.get("duration_active_accum") or 0)
        except ValueError:
            active = 0
        return not fit_url(workout) and active <= 0
    return False


def may_be_planned(workout):
    """A recorded ride has no plan; anything linked to a plan (e.g. sent from this app) needs checking."""
    return bool(workout.get("plan_id")) or str(workout.get("workout_token") or "").startswith("biketools-")


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


# --- Planned sessions (plans + scheduled workouts) -------------------------------------------------

# workout_type_id is an unpublished enum: 0 is outdoor cycling (as in the rides list)
PLAN_WORKOUT_TYPE_ID = 0
SYNC_WINDOW_DAYS = 6             # the head unit only receives workouts from today to today + 6


def _plan_form(plan_json, name, filename, external_id=None, provider_updated_at=None):
    import base64
    uri = "data:application/json;base64," + base64.b64encode(plan_json.encode("utf-8")).decode("ascii")
    form = {"plan[file]": uri, "plan[filename]": filename, "plan[name]": name}
    if external_id:
        form["plan[external_id]"] = external_id
    if provider_updated_at:
        form["plan[provider_updated_at]"] = provider_updated_at
    return form


def create_plan(token, plan_json, name, filename, external_id, provider_updated_at, usage=None):
    return api_request("POST", "/plans", token, usage=usage,
                       data=_plan_form(plan_json, name, filename, external_id, provider_updated_at))


def update_plan(token, plan_id, plan_json, name, filename, provider_updated_at, usage=None):
    """external_id cannot be changed on update, so it is not sent."""
    return api_request("PUT", f"/plans/{int(plan_id)}", token, usage=usage,
                       data=_plan_form(plan_json, name, filename, None, provider_updated_at))


def find_plan(token, external_id, usage=None):
    """The plan with this external_id, or None."""
    data = api_request("GET", "/plans", token, params={"external_id": external_id}, usage=usage) or {}
    plans = data.get("plans") if isinstance(data, dict) else data
    return (plans or [None])[0]


def delete_plan(token, plan_id, usage=None):
    try:
        api_request("DELETE", f"/plans/{int(plan_id)}", token, usage=usage)
    except WahooNotFound:
        pass


def _workout_form(name, workout_token, starts_utc_iso, minutes, plan_id):
    return {"workout[name]": name, "workout[workout_token]": workout_token,
            "workout[workout_type_id]": PLAN_WORKOUT_TYPE_ID, "workout[starts]": starts_utc_iso,
            "workout[minutes]": int(minutes), "workout[plan_id]": int(plan_id)}


def schedule_workout(token, name, workout_token, starts_utc_iso, minutes, plan_id, usage=None):
    return api_request("POST", "/workouts", token, usage=usage,
                       data=_workout_form(name, workout_token, starts_utc_iso, minutes, plan_id))


def update_workout(token, workout_id, name, workout_token, starts_utc_iso, minutes, plan_id, usage=None):
    return api_request("PUT", f"/workouts/{int(workout_id)}", token, usage=usage,
                       data=_workout_form(name, workout_token, starts_utc_iso, minutes, plan_id))


def delete_workout(token, workout_id, usage=None):
    try:
        api_request("DELETE", f"/workouts/{int(workout_id)}", token, usage=usage)
    except WahooNotFound:
        pass
