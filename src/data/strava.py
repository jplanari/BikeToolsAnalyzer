"""
Strava official API client (OAuth 2.0).

Kept free of Streamlit so it can be unit-tested: the dashboard reads the
app credentials from st.secrets and passes a StravaConfig in.

Docs: https://developers.strava.com/docs/authentication/
"""
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
REVOKE_URL = "https://www.strava.com/oauth/revoke"
API_BASE = "https://www.strava.com/api/v3"

# activity:read_all also covers "Only You" activities and privacy-zone data
SCOPE = "read,activity:read_all"

CYCLING_SPORT_TYPES = {
    "Ride", "VirtualRide", "GravelRide", "MountainBikeRide",
    "EBikeRide", "EMountainBikeRide", "Velomobile", "Handcycle",
}

STREAM_KEYS = ["time", "latlng", "altitude", "distance", "velocity_smooth",
               "heartrate", "cadence", "watts", "temp", "moving"]

# Refresh a bit before expiry so a request never goes out with a dead token
REFRESH_MARGIN_S = 300
TIMEOUT_S = 30


class StravaError(Exception):
    pass


class StravaAuthError(StravaError):
    """Token missing, revoked or lacking scope: the user must reconnect."""


class StravaRateLimitError(StravaError):
    """HTTP 429 or our own budget guard: retry after the 15-minute window."""


@dataclass
class StravaConfig:
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8501/"


@dataclass
class RateUsage:
    """Last seen read-rate usage, from the X-(Read)RateLimit-* headers."""
    limit_15min: int = 100
    limit_daily: int = 1000
    used_15min: int = 0
    used_daily: int = 0
    seen: bool = field(default=False)

    def update(self, headers):
        limit = headers.get("X-ReadRateLimit-Limit") or headers.get("X-RateLimit-Limit")
        usage = headers.get("X-ReadRateLimit-Usage") or headers.get("X-RateLimit-Usage")
        try:
            if limit:
                self.limit_15min, self.limit_daily = (int(x) for x in limit.split(","))
            if usage:
                self.used_15min, self.used_daily = (int(x) for x in usage.split(","))
                self.seen = True
        except ValueError:
            pass

    def remaining(self):
        return min(self.limit_15min - self.used_15min, self.limit_daily - self.used_daily)


# --- OAuth -----------------------------------------------------------------

def authorize_url(cfg, state):
    params = {
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": SCOPE,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _token_request(data):
    try:
        r = requests.post(TOKEN_URL, data=data, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise StravaError(f"Could not reach Strava: {e}") from e
    if r.status_code in (400, 401):
        raise StravaAuthError(f"Strava rejected the request ({r.status_code}): {r.text[:200]}")
    if r.status_code == 429:
        raise StravaRateLimitError("Strava rate limit reached; try again in 15 minutes.")
    if not r.ok:
        raise StravaError(f"Strava token endpoint error {r.status_code}")
    return r.json()


def exchange_code(cfg, code):
    """Swap the one-time authorization code for tokens."""
    tok = _token_request({
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "code": code,
        "grant_type": "authorization_code",
    })
    athlete = tok.get("athlete") or {}
    return {
        "athlete_id": athlete.get("id"),
        "athlete_name": " ".join(x for x in [athlete.get("firstname"), athlete.get("lastname")] if x),
        "access_token": tok["access_token"],
        "refresh_token": tok["refresh_token"],
        "expires_at": int(tok["expires_at"]),
    }


def refresh_if_needed(cfg, auth, now=None):
    """
    Returns (auth, changed). Strava may rotate the refresh token, and the old
    one stops working immediately, so callers must persist `auth` whenever
    changed is True.
    """
    now = time.time() if now is None else now
    if auth["expires_at"] - now > REFRESH_MARGIN_S:
        return auth, False
    tok = _token_request({
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "refresh_token": auth["refresh_token"],
        "grant_type": "refresh_token",
    })
    new = dict(auth)
    new["access_token"] = tok["access_token"]
    new["refresh_token"] = tok.get("refresh_token", auth["refresh_token"])
    new["expires_at"] = int(tok["expires_at"])
    return new, True


def revoke(cfg, token):
    """Revokes the app's access for this athlete (best effort)."""
    try:
        requests.post(REVOKE_URL, data={"token": token},
                      auth=(cfg.client_id, cfg.client_secret), timeout=TIMEOUT_S)
    except requests.RequestException:
        pass


def granted_scope_ok(scope_param):
    """The athlete may untick scopes on the consent page."""
    scopes = set((scope_param or "").replace(" ", ",").split(","))
    return bool(scopes & {"activity:read", "activity:read_all"})


# --- API -------------------------------------------------------------------

def api_get(path, access_token, params=None, usage=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params,
                         headers={"Authorization": f"Bearer {access_token}"},
                         timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise StravaError(f"Could not reach Strava: {e}") from e
    if usage is not None:
        usage.update(r.headers)
    if r.status_code == 401:
        raise StravaAuthError("Strava authorization expired or was revoked. Please reconnect.")
    if r.status_code == 403:
        raise StravaAuthError("Missing permission for this activity (reconnect and allow activity access).")
    if r.status_code == 404:
        raise StravaError("Activity not found.")
    if r.status_code == 429:
        raise StravaRateLimitError("Strava rate limit reached; try again in 15 minutes.")
    if not r.ok:
        raise StravaError(f"Strava API error {r.status_code}")
    return r.json()


def is_cycling(activity):
    return (activity.get("sport_type") or activity.get("type")) in CYCLING_SPORT_TYPES


def list_activities(access_token, after=None, before=None, page=1, per_page=200, usage=None):
    params = {"page": page, "per_page": per_page}
    if after is not None:
        params["after"] = int(after)
    if before is not None:
        params["before"] = int(before)
    return api_get("/athlete/activities", access_token, params, usage)


def get_streams(access_token, activity_id, usage=None):
    data = api_get(f"/activities/{int(activity_id)}/streams", access_token,
                   {"keys": ",".join(STREAM_KEYS), "key_by_type": "true"}, usage)
    # key_by_type=true gives {type: {...}}; be tolerant of the list form too
    if isinstance(data, list):
        data = {s["type"]: s for s in data if "type" in s}
    return data


# --- Stored payload <-> DataFrame -------------------------------------------

ACTIVITY_FIELDS = ["id", "name", "sport_type", "type", "start_date", "trainer",
                   "manual", "device_watts", "distance", "moving_time", "elapsed_time",
                   "total_elevation_gain"]


def ride_filename(activity_id):
    return f"strava_{int(activity_id)}.json"


def build_ride_payload(activity, streams):
    """
    Serialises an activity summary + its streams as the ride's stored file.
    The API does not provide the original GPX/FIT, so this JSON is what gets
    saved in rides.file_content (deterministic, so re-imports dedupe).
    """
    payload = {
        "source": "strava_api",
        "activity": {k: activity.get(k) for k in ACTIVITY_FIELDS},
        "streams": {k: v.get("data", []) for k, v in streams.items()},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def streams_to_dataframe(payload_bytes):
    """
    Converts a stored payload into the app's standard schema (same columns
    as parse_gpx/parse_fit): time (naive UTC), lat, lon, ele, hr, power,
    cadence, plus temp_c when recorded.
    """
    payload = json.loads(payload_bytes)
    act = payload.get("activity", {})
    s = payload.get("streams", {})

    t = s.get("time") or []
    if not t or not act.get("start_date"):
        return pd.DataFrame()

    n = len(t)
    start = pd.Timestamp(act["start_date"])
    start = start.tz_convert("UTC").tz_localize(None) if start.tzinfo else start

    def col(key):
        v = s.get(key)
        if v is None or len(v) != n:
            return np.full(n, np.nan)
        return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)

    df = pd.DataFrame({"time": start + pd.to_timedelta(np.asarray(t, dtype=float), unit="s")})

    latlng = s.get("latlng")
    if latlng and len(latlng) == n:
        arr = np.array([p if p else [np.nan, np.nan] for p in latlng], dtype=float)
        df["lat"], df["lon"] = arr[:, 0], arr[:, 1]
    else:
        df["lat"], df["lon"] = np.nan, np.nan

    df["ele"] = col("altitude")
    df["hr"] = col("heartrate")
    df["power"] = col("watts")
    df["cadence"] = col("cadence")
    if s.get("temp"):
        df["temp_c"] = col("temp")

    # Missing sensors stay as all-NaN columns, like parse_gpx/parse_fit,
    # because parts of the pipeline (e.g. aerodyn) expect them to exist.
    return df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
