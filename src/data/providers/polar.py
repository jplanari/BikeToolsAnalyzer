"""
Polar AccessLink API client (OAuth 2.0). Downloads the FIT file Polar Flow holds for each exercise.

Differences from Strava and Wahoo, from Polar's documentation:
  * access tokens do not expire (no refresh), but each rider must be registered with the API after authorizing;
  * only exercises uploaded to Flow in the last 30 days, and after the rider connected, can be listed, so history
    cannot be back-filled: rides must be synced at least once a month.

Kept free of Streamlit so it can be unit-tested; the sidebar passes a PolarConfig in.
Docs: https://www.polar.com/accesslink-api/
"""
import re
from dataclasses import dataclass
from urllib.parse import urlencode

import pandas as pd
import requests

AUTHORIZE_URL = "https://flow.polar.com/oauth2/authorization"
TOKEN_URL = "https://polarremote.com/v2/oauth2/token"
API_BASE = "https://www.polaraccesslink.com/v3"
TIMEOUT_S = 30
WINDOW_DAYS = 30                      # how far back Polar lets an app list exercises


class PolarError(Exception):
    pass


class PolarAuthError(PolarError):
    """Token missing or revoked: the rider must reconnect."""


class PolarNotFound(PolarError):
    """404: the exercise is gone."""


class PolarRateLimitError(PolarError):
    """HTTP 429."""


@dataclass
class PolarConfig:
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8501/"


# --- OAuth -----------------------------------------------------------------

def authorize_url(cfg, state):
    params = {"response_type": "code", "client_id": cfg.client_id, "redirect_uri": cfg.redirect_uri, "state": state}
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(cfg, code):
    """Swap the one-time code for a token. Returns {'access_token', 'polar_user_id'}; the rider still has to be registered."""
    try:
        r = requests.post(TOKEN_URL, auth=(cfg.client_id, cfg.client_secret),
                          data={"grant_type": "authorization_code", "code": code, "redirect_uri": cfg.redirect_uri},
                          headers={"Accept": "application/json"}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise PolarError(f"Could not reach Polar: {e}") from e
    if r.status_code in (400, 401):
        raise PolarAuthError(f"Polar rejected the request ({r.status_code}): {r.text[:200]}")
    if r.status_code == 429:
        raise PolarRateLimitError("Polar rate limit reached; try again in a few minutes.")
    if not r.ok:
        raise PolarError(f"Polar token endpoint error {r.status_code}")
    tok = r.json()
    return {"access_token": tok["access_token"], "polar_user_id": str(tok.get("x_user_id") or "")}


# --- API -------------------------------------------------------------------

def _request(method, path, access_token, **kw):
    headers = {"Authorization": f"Bearer {access_token}", **kw.pop("headers", {})}
    try:
        r = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=TIMEOUT_S, **kw)
    except requests.RequestException as e:
        raise PolarError(f"Could not reach Polar: {e}") from e
    if r.status_code in (401, 403):
        raise PolarAuthError("Polar authorization was revoked or expired. Please reconnect.")
    if r.status_code == 404:
        raise PolarNotFound("Not found on Polar.")
    if r.status_code == 429:
        raise PolarRateLimitError("Polar rate limit reached; try again in a few minutes.")
    return r


def register_user(access_token, member_id):
    """Registers the rider with this app (required once before their data can be read). Already registered is fine."""
    r = _request("POST", "/users", access_token, json={"member-id": str(member_id)},
                 headers={"Accept": "application/json"})
    if r.status_code == 409:                                  # already registered
        return None
    if not r.ok:
        raise PolarError(f"Polar could not register this account ({r.status_code}): {r.text[:200]}")
    return r.json() if r.content else None


def deregister_user(access_token, polar_user_id):
    """Removes the rider from this app and revokes the token (best effort)."""
    try:
        _request("DELETE", f"/users/{polar_user_id}", access_token)
    except PolarError:
        pass


def list_exercises(access_token):
    """Exercises uploaded to Flow in the last 30 days (summaries), as a list of dicts."""
    r = _request("GET", "/exercises", access_token, headers={"Accept": "application/json"})
    if not r.ok:
        raise PolarError(f"Polar API error {r.status_code}: {r.text[:200]}")
    data = r.json() if r.content else []
    return data if isinstance(data, list) else data.get("exercises", [])


def download_fit(access_token, exercise_id):
    r = _request("GET", f"/exercises/{exercise_id}/fit", access_token, headers={"Accept": "*/*"})
    if r.status_code == 204 or not r.content:
        raise PolarError("This exercise has no recorded data file.")
    if not r.ok:
        raise PolarError(f"Polar API error {r.status_code}: {r.text[:200]}")
    return r.content


# --- exercise summaries ------------------------------------------------------

def is_cycling(ex):
    text = f"{ex.get('sport') or ''} {ex.get('detailed_sport_info') or ''}".upper()
    return "CYCL" in text or "BIK" in text


def start_utc(ex):
    """Start of the exercise as naive UTC: Polar gives local time plus its UTC offset in minutes."""
    local = pd.to_datetime(ex.get("start_time"), errors="coerce")
    if pd.isna(local):
        return None
    if local.tzinfo is not None:
        return local.tz_convert("UTC").tz_localize(None)
    offset = ex.get("start_time_utc_offset")
    return local - pd.Timedelta(minutes=int(offset or 0))


def duration_minutes(ex):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", str(ex.get("duration") or ""))
    if not m:
        return 0
    d, h, mi, s = (float(x) if x else 0 for x in m.groups())
    return int(round(d * 1440 + h * 60 + mi + s / 60))


def ride_filename(exercise_id):
    return f"polar_{exercise_id}.fit"
