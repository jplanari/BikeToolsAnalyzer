"""Saves uploaded ride files (one or many) to a rider's history, skipping rides already there."""
import pandas as pd

from src.data import db
from src.data.strava_import import ImportResult, DUPLICATE_START_TOLERANCE_S, _is_near


def _sport(activity):
    """Sport recorded in the file, lower case, or None when the file does not say."""
    if activity.kind == "fit":
        from src.data.fit import fit_sport
        return fit_sport(activity.data)
    if activity.kind == "tcx":
        from src.data.tcx import tcx_sport
        s = tcx_sport(activity.data)
        return {"biking": "cycling"}.get((s or "").lower(), (s or "").lower() or None)
    return None


def import_activities(user_name, activities, ftp, lthr=None, progress=None):
    """
    Imports each Activity (src.data.upload) as a ride. Idempotent: the same file, or a ride that starts within two
    minutes of one already saved (the same ride from Strava, Wahoo or another device), is skipped.
    Files that are not cycling (a run recorded on the same watch), or hold no timestamped samples (wellness and
    settings FIT files in a bulk export), are counted in `no_data` and not saved.
    """
    from src.data.fit import CYCLING_SPORTS
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage    # parsers import streamlit

    res = ImportResult(found=len(activities))
    known_starts = list(db.get_ride_start_times(user_name))
    for i, act in enumerate(activities):
        if progress:
            progress(i, res.found, act.name)
        try:
            sport = _sport(act)
            if sport not in CYCLING_SPORTS and sport not in ("cycling", "biking"):
                res.no_data += 1
                continue
            df, dist_m, ele_m = prepare_ride(act.name, act.data)
            summ = summarize_ride(df, ftp, lthr)
            meta = db.build_ride_metadata(df, dist_m, ele_m, summ['avg_speed_kph'], summ['norm_power'], summ['tss'],
                                          summ['if'], filename=act.name, metrics=metrics_for_storage(summ))
            start = pd.to_datetime(meta.get('date'), errors='coerce', utc=True)
            start = None if pd.isna(start) else start.tz_convert(None)
            if _is_near(start, known_starts, DUPLICATE_START_TOLERANCE_S):
                res.already_imported += 1
                continue
            ok, msg = db.save_ride(user_name, act.name, act.data, meta, summ['power_curve'])
        except ValueError:                       # no timestamped samples: not a ride (wellness, settings, empty)
            res.no_data += 1
            continue
        except Exception as e:                   # unreadable file: report, keep going
            res.failed.append(f"{act.name}: {e}")
            continue
        if ok:
            res.imported += 1
            if start is not None:
                known_starts.append(start)
        elif "duplicate" in msg.lower() or "already exists" in msg.lower():
            res.already_imported += 1
        else:
            res.failed.append(f"{act.name}: {msg}")
    if progress:
        progress(res.found, res.found, "")
    return res
