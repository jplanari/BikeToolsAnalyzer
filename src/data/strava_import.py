"""
Strava import workflows (single ride and history since a date), built on the
API client in src/data/strava.py and the shared ride pipeline. No Streamlit.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.data import db
from src.data import strava as sv

# Ride starts closer than this to an existing ride are treated as the same
# ride (e.g. already uploaded as a GPX/FIT file), so TSS isn't counted twice.
DUPLICATE_START_TOLERANCE_S = 120
# Requests kept in reserve so the single-ride picker still works after a
# history import has used up the 15-minute budget.
REQUEST_RESERVE = 5


def get_valid_auth(cfg, user_name):
    """Stored auth for this rider, refreshed (and re-saved) if near expiry."""
    auth = db.get_strava_auth(user_name)
    if not auth:
        raise sv.StravaAuthError("This profile is not connected to Strava.")
    auth, changed = sv.refresh_if_needed(cfg, auth)
    if changed:
        db.save_strava_auth(user_name, auth)
    return auth


def list_recent_rides(cfg, user_name, limit=30, usage=None):
    auth = get_valid_auth(cfg, user_name)
    acts = sv.list_activities(auth['access_token'], per_page=min(200, limit * 2), usage=usage)
    return [a for a in acts if sv.is_cycling(a) and not a.get("manual")][:limit]


def fetch_ride_file(cfg, user_name, activity, usage=None):
    """Downloads one activity's streams; returns (filename, payload_bytes)."""
    auth = get_valid_auth(cfg, user_name)
    streams = sv.get_streams(auth['access_token'], activity['id'], usage=usage)
    if not streams.get("time"):
        raise sv.StravaError("This activity has no recorded data streams.")
    return sv.ride_filename(activity['id']), sv.build_ride_payload(activity, streams)


@dataclass
class ImportResult:
    found: int = 0
    imported: int = 0
    already_imported: int = 0
    duplicates_of_uploads: int = 0
    no_data: int = 0
    failed: list = field(default_factory=list)
    stopped_reason: str = ""

    @property
    def remaining(self):
        return (self.found - self.imported - self.already_imported
                - self.duplicates_of_uploads - self.no_data - len(self.failed))


def _start_utc(activity):
    ts = pd.to_datetime(activity.get("start_date"), errors="coerce", utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)


def _is_near(ts, others, tol_s):
    return ts is not None and any(abs((ts - o).total_seconds()) <= tol_s for o in others)


def list_cycling_since(auth, since, usage=None):
    """All non-manual cycling activities since `since` (date/datetime), oldest first."""
    if isinstance(since, datetime):
        after = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    else:
        after = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    acts, page = [], 1
    while True:
        batch = sv.list_activities(auth['access_token'], after=after.timestamp(),
                                   page=page, per_page=200, usage=usage)
        if not batch:
            break
        acts.extend(a for a in batch if sv.is_cycling(a) and not a.get("manual"))
        if len(batch) < 200:
            break
        page += 1
    return sorted(acts, key=lambda a: a.get("start_date") or "")


def import_history(cfg, user_name, since, ftp, progress=None, usage=None):
    """
    Imports every cycling activity since `since`. Idempotent: rides already
    imported are skipped, so if it stops at the rate limit, running it again
    later resumes where it left off.

    progress: optional callback(done, total, activity_name).
    """
    from src.data.pipeline import prepare_ride, summarize_ride  # parsers import streamlit

    usage = usage or sv.RateUsage()
    res = ImportResult()
    auth = get_valid_auth(cfg, user_name)

    try:
        activities = list_cycling_since(auth, since, usage)
    except sv.StravaRateLimitError as e:
        res.stopped_reason = str(e)
        return res
    res.found = len(activities)

    imported_ids = db.get_imported_strava_ids(user_name)
    upload_starts = db.get_ride_start_times(user_name)

    for i, act in enumerate(activities):
        name = act.get("name") or f"Activity {act['id']}"
        if progress:
            progress(i, res.found, name)

        if act['id'] in imported_ids:
            res.already_imported += 1
            continue
        if _is_near(_start_utc(act), upload_starts, DUPLICATE_START_TOLERANCE_S):
            res.duplicates_of_uploads += 1
            continue
        if usage.seen and usage.remaining() <= REQUEST_RESERVE:
            res.stopped_reason = ("Paused to stay within Strava's rate limit. "
                                  "Run the import again in 15 minutes to continue.")
            break

        try:
            # fetch_ride_file refreshes the token if needed (long imports can outlive one)
            filename, payload = fetch_ride_file(cfg, user_name, act, usage=usage)
            df, dist_m, ele_m = prepare_ride(filename, payload)
            summ = summarize_ride(df, ftp)
            meta = db.build_ride_metadata(df, dist_m, ele_m, summ['avg_speed_kph'],
                                          summ['norm_power'], summ['tss'], summ['if'],
                                          filename=filename)
            ok, msg = db.save_ride(user_name, filename, payload, meta, summ['power_curve'])
        except sv.StravaRateLimitError as e:
            res.stopped_reason = str(e)
            break
        except sv.StravaAuthError:
            raise
        except sv.StravaError as e:
            if "no recorded data" in str(e):
                res.no_data += 1
            else:
                res.failed.append(f"{name}: {e}")
            continue
        except Exception as e:  # unparsable streams etc.: report, keep going
            res.failed.append(f"{name}: {e}")
            continue

        if ok:
            res.imported += 1
            imported_ids.add(act['id'])
        elif "duplicate" in msg.lower() or "already exists" in msg.lower():
            res.already_imported += 1
        else:
            res.failed.append(f"{name}: {msg}")

    if progress:
        progress(res.found, res.found, "")
    return res
