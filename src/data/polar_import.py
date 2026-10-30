"""
Polar import workflows (single ride, sync), built on src/data/polar.py and the shared ride pipeline. The FIT file
Polar Flow holds is stored, so rides are analysed exactly like an uploaded file. No Streamlit.

Polar only lists exercises from the last 30 days, so there is no history import: sync regularly.
"""
import pandas as pd

from src.data import db
from src.data import polar as pl
from src.data.strava_import import ImportResult, DUPLICATE_START_TOLERANCE_S, _is_near


def get_auth(user_name):
    auth = db.get_polar_auth(user_name)
    if not auth:
        raise pl.PolarAuthError("This profile is not connected to Polar.")
    return auth


def connect(cfg, user_name, code):
    """Finishes the OAuth round trip: token, registration with AccessLink, storage."""
    auth = pl.exchange_code(cfg, code)
    user = db.get_user_data(user_name)
    pl.register_user(auth["access_token"], f"biketools-{user['id']}")
    db.save_polar_auth(user_name, auth)
    return auth


def list_recent_rides(user_name):
    """Cycling exercises Polar still lists (last 30 days), newest first."""
    auth = get_auth(user_name)
    rides = [e for e in pl.list_exercises(auth["access_token"]) if pl.is_cycling(e)]
    return sorted(rides, key=lambda e: e.get("start_time") or "", reverse=True)


def fetch_ride_file(user_name, exercise):
    """Downloads one exercise's FIT file; returns (filename, bytes)."""
    auth = get_auth(user_name)
    return pl.ride_filename(exercise["id"]), pl.download_fit(auth["access_token"], exercise["id"])


def sync_new_rides(user_name, ftp, progress=None, lthr=None):
    """
    Imports every cycling exercise Polar lists that is not in the history yet. Idempotent. Returns ImportResult.
    A ride that starts within 2 minutes of one already saved (another source) is skipped.
    """
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage    # parsers import streamlit

    res = ImportResult()
    auth = get_auth(user_name)
    try:
        exercises = sorted((e for e in pl.list_exercises(auth["access_token"]) if pl.is_cycling(e)),
                           key=lambda e: e.get("start_time") or "")
    except pl.PolarRateLimitError as e:
        res.stopped_reason = str(e)
        return res
    res.found = len(exercises)

    imported_ids = db.get_imported_polar_ids(user_name)
    known_starts = list(db.get_ride_start_times(user_name))

    for i, ex in enumerate(exercises):
        name = f"{pl.start_utc(ex):%Y-%m-%d}" if pl.start_utc(ex) is not None else f"Exercise {ex['id']}"
        if progress:
            progress(i, res.found, name)
        if str(ex["id"]) in imported_ids:
            res.already_imported += 1
            continue
        start = pl.start_utc(ex)
        if _is_near(start, known_starts, DUPLICATE_START_TOLERANCE_S):
            res.duplicates_of_uploads += 1
            continue
        try:
            filename, payload = fetch_ride_file(user_name, ex)
            df, dist_m, ele_m = prepare_ride(filename, payload)
            summ = summarize_ride(df, ftp, lthr)
            meta = db.build_ride_metadata(df, dist_m, ele_m, summ['avg_speed_kph'], summ['norm_power'], summ['tss'],
                                          summ['if'], filename=filename, metrics=metrics_for_storage(summ))
            ok, msg = db.save_ride(user_name, filename, payload, meta, summ['power_curve'])
        except pl.PolarRateLimitError as e:
            res.stopped_reason = str(e)
            break
        except pl.PolarAuthError:
            raise
        except pl.PolarError as e:
            if "no recorded data" in str(e):
                res.no_data += 1
            else:
                res.failed.append(f"{name}: {e}")
            continue
        except Exception as e:                      # unparsable FIT etc.: report, keep going
            res.failed.append(f"{name}: {e}")
            continue

        if ok:
            res.imported += 1
            imported_ids.add(str(ex["id"]))
            if start is not None:
                known_starts.append(start)
        elif "duplicate" in msg.lower() or "already exists" in msg.lower():
            res.already_imported += 1
        else:
            res.failed.append(f"{name}: {msg}")

    if progress:
        progress(res.found, res.found, "")
    return res
