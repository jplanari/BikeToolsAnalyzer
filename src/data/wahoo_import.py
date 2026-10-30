"""
Wahoo import workflows (single ride, history since a date, sync), built on src/data/wahoo.py
and the shared ride pipeline. The ORIGINAL FIT file is stored, so rides are analysed exactly
like an uploaded file. No Streamlit.
"""
import pandas as pd

from src.data import db
from src.data import wahoo as wh
from src.data.strava_import import ImportResult, DUPLICATE_START_TOLERANCE_S, _is_near

REQUEST_RESERVE = 3          # API requests kept back so the single-ride picker still works afterwards
SYNC_MARGIN_DAYS = 3
SYNC_FALLBACK_DAYS = 30


def get_valid_auth(cfg, user_name):
    """Stored auth for this rider, refreshed (and re-saved at once: refresh tokens are single-use)."""
    auth = db.get_wahoo_auth(user_name)
    if not auth:
        raise wh.WahooAuthError("This profile is not connected to Wahoo.")
    auth, changed = wh.refresh_if_needed(cfg, auth)
    if changed:
        db.save_wahoo_auth(user_name, auth)
    return auth


def _start_utc(workout):
    ts = pd.to_datetime(workout.get("starts"), errors="coerce", utc=True)
    return None if pd.isna(ts) else ts.tz_convert(None)


def list_cycling_since(auth, since, usage=None, until=None):
    """All cycling workouts that started on/after `since` (date/datetime), oldest first; with `until`, before it.
    The list endpoint has no date filter, so pages are read newest-first until one is entirely older."""
    after = pd.Timestamp(since)
    if after.tzinfo is not None:
        after = after.tz_convert('UTC').tz_localize(None)
    before = None if until is None else pd.Timestamp(until)
    if before is not None and before.tzinfo is not None:
        before = before.tz_convert('UTC').tz_localize(None)
    found, page = [], 1
    while True:
        data = wh.list_workouts(auth['access_token'], page=page, usage=usage)
        batch = data.get('workouts') or []
        if not batch:
            break
        starts = [_start_utc(w) for w in batch]
        found.extend(w for w, s in zip(batch, starts)
                     if s is not None and s >= after and (before is None or s < before) and wh.is_cycling(w) and not wh.is_planned_only(w))
        newest_first = str(data.get('order', 'descending')).lower().startswith('desc')
        if len(batch) < data.get('per_page', wh.PER_PAGE):
            break
        if newest_first and all(s is not None and s < after for s in starts):
            break
        page += 1
    return sorted(found, key=lambda w: w.get('starts') or '')


def list_rides_between(cfg, user_name, start, end, usage=None):
    """Cycling workouts from the start date to the end date, both included, newest first."""
    auth = get_valid_auth(cfg, user_name)
    day_after = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    return list(reversed(list_cycling_since(auth, start, usage, until=day_after)))


def list_recent_rides(cfg, user_name, limit=30, usage=None):
    auth = get_valid_auth(cfg, user_name)
    data = wh.list_workouts(auth['access_token'], page=1, per_page=min(wh.PER_PAGE, limit * 2), usage=usage)
    rides = [w for w in data.get('workouts') or [] if wh.is_cycling(w) and not wh.is_planned_only(w)]
    rides = sorted(rides, key=lambda w: w.get('starts') or '', reverse=True)
    out = []
    for w in rides:
        if len(out) >= limit:
            break
        if wh.may_be_planned(w):                    # linked to a plan: ridden only if it has a recorded file
            try:
                if not wh.fit_url(wh.get_workout(auth['access_token'], w['id'], usage=usage)):
                    continue
            except wh.WahooNotFound:
                continue
        out.append(w)
    return out


def fetch_ride_file(cfg, user_name, workout, usage=None):
    """Downloads one workout's FIT file; returns (filename, bytes)."""
    auth = get_valid_auth(cfg, user_name)
    detail = wh.get_workout(auth['access_token'], workout['id'], usage=usage)
    url = wh.fit_url(detail)
    if not url:
        raise wh.WahooError("This workout has no recorded data file.")
    return wh.ride_filename(workout['id']), wh.download_fit(url)


def import_history(cfg, user_name, since, ftp, progress=None, usage=None, lthr=None):
    """
    Imports every cycling workout since `since`. Idempotent: workouts already imported are
    skipped, so if it stops at the rate limit, running it again later resumes where it left off.
    progress: optional callback(done, total, name).
    """
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage  # parsers import streamlit

    usage = usage or wh.RateUsage()
    res = ImportResult()
    auth = get_valid_auth(cfg, user_name)
    try:
        workouts = list_cycling_since(auth, since, usage)
    except wh.WahooRateLimitError as e:
        res.stopped_reason = str(e)
        return res
    res.found = len(workouts)

    imported_ids = db.get_imported_wahoo_ids(user_name)
    known_starts = db.get_ride_start_times(user_name)     # uploads and Strava imports too

    for i, w in enumerate(workouts):
        name = w.get("name") or f"Workout {w['id']}"
        if progress:
            progress(i, res.found, name)
        if w['id'] in imported_ids:
            res.already_imported += 1
            continue
        if _is_near(_start_utc(w), known_starts, DUPLICATE_START_TOLERANCE_S):
            res.duplicates_of_uploads += 1
            continue
        if usage.seen and usage.remaining <= REQUEST_RESERVE:
            res.stopped_reason = "Paused to stay within Wahoo's rate limit. Run the import again in a few minutes."
            break
        try:
            filename, payload = fetch_ride_file(cfg, user_name, w, usage=usage)
            df, dist_m, ele_m = prepare_ride(filename, payload)
            summ = summarize_ride(df, ftp, lthr)
            meta = db.build_ride_metadata(df, dist_m, ele_m, summ['avg_speed_kph'], summ['norm_power'],
                                          summ['tss'], summ['if'], filename=filename,
                                          metrics=metrics_for_storage(summ))
            ok, msg = db.save_ride(user_name, filename, payload, meta, summ['power_curve'])
        except wh.WahooRateLimitError as e:
            res.stopped_reason = str(e)
            break
        except wh.WahooAuthError:
            raise
        except wh.WahooError as e:
            if "no recorded data" in str(e):
                res.not_completed += 1       # a scheduled workout has no file until it is ridden
            else:
                res.failed.append(f"{name}: {e}")
            continue
        except Exception as e:      # unparsable FIT etc.: report, keep going
            res.failed.append(f"{name}: {e}")
            continue

        if ok:
            res.imported += 1
            imported_ids.add(w['id'])
            start = _start_utc(w)
            if start is not None:
                known_starts.append(start)
        elif "duplicate" in msg.lower() or "already exists" in msg.lower():
            res.already_imported += 1
        else:
            res.failed.append(f"{name}: {msg}")

    if progress:
        progress(res.found, res.found, "")
    return res


def sync_new_rides(cfg, user_name, ftp, progress=None, usage=None, lthr=None, now=None):
    """Imports what is new on Wahoo since the last Wahoo ride. Returns (ImportResult, since)."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now('UTC').tz_localize(None)
    last = db.latest_wahoo_ride_date(user_name)
    since = (last - pd.Timedelta(days=SYNC_MARGIN_DAYS)) if last is not None \
        else now - pd.Timedelta(days=SYNC_FALLBACK_DAYS)
    return import_history(cfg, user_name, since.to_pydatetime(), ftp, progress=progress, usage=usage,
                          lthr=lthr), since
