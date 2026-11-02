"""Imports from Strava / Wahoo for a rider, run in the background so the phone does not wait for them. One job per
rider at a time, a couple at once on the whole server (each one downloads and analyses rides)."""
import threading
import time
from datetime import timedelta

import pandas as pd

from src.data import db
from src.data.imports import strava_import as si
from src.data.imports import wahoo_import as wi
from src.data.providers import strava as sv
from src.data.providers import wahoo as wh
from . import providers

MAX_DAYS = 3650
CONCURRENT = 2
_slots = threading.BoundedSemaphore(CONCURRENT)
_lock = threading.Lock()
_jobs = {}                                    # rider name -> dict (the last job of each rider)


def status(user_name):
    with _lock:
        job = _jobs.get(user_name)
        return dict(job) if job else None


def _update(user_name, **fields):
    with _lock:
        _jobs[user_name].update(fields)


def start(user_name, provider, days=None, run=True):
    """Starts an import: the rides of the last `days`, or (days None) whatever is new since the last one. Returns
    (ok, reason) with reason 'running', 'not_connected' or ''."""
    if provider not in providers.NAMES:
        return False, "unknown"
    connected = providers.status(user_name)[provider]
    if not (connected["configured"] and connected["connected"]):
        return False, "not_connected"
    with _lock:
        current = _jobs.get(user_name)
        if current and current["state"] in ("queued", "running"):
            return False, "running"
        _jobs[user_name] = {"provider": provider, "state": "queued", "done": 0, "total": 0, "name": "", "imported": 0,
                            "already": 0, "failed": 0, "stopped": "", "error": "", "started": int(time.time()),
                            "finished": None}
    if run:
        threading.Thread(target=_run, args=(user_name, provider, days), name=f"import-{provider}", daemon=True).start()
    return True, ""


def _run(user_name, provider, days):
    with _slots:
        _update(user_name, state="running")
        try:
            user = db.get_user_data(user_name)
            ftp, lthr = user['ftp'], user['lthr']
            cfg = providers.config(provider)
            usage = sv.RateUsage() if provider == "strava" else wh.RateUsage()
            mod = si if provider == "strava" else wi

            def progress(done, total, name):
                _update(user_name, done=done, total=total, name=str(name)[:60])

            if days is None:
                res = mod.sync_new_rides(cfg, user_name, ftp, progress=progress, usage=usage, lthr=lthr)[0]
            else:
                since = (pd.Timestamp.now('UTC').tz_localize(None) - timedelta(days=min(int(days), MAX_DAYS))).to_pydatetime()
                res = mod.import_history(cfg, user_name, since, ftp, progress=progress, usage=usage, lthr=lthr)
            _update(user_name, state="done", imported=res.imported, already=res.already_imported + res.duplicates_of_uploads,
                    failed=len(res.failed), stopped=res.stopped_reason, done=res.found, total=res.found)
            _follow_the_data(user_name)
        except (sv.StravaAuthError, wh.WahooAuthError):
            _update(user_name, state="error", error="reconnect")
        except Exception as e:                 # whatever goes wrong must end the job, not leave it "running"
            _update(user_name, state="error", error=str(e)[:200])
        finally:
            _update(user_name, finished=int(time.time()))


def _follow_the_data(user_name):
    """FTP / LTHR the rider left empty follow the rides that have just arrived."""
    try:
        from src.data.processing.thresholds import backfill_hr20, refresh_auto_thresholds
        user = db.get_user_data(user_name)
        if user and user.get('lthr_auto'):
            backfill_hr20(user_name)
        refresh_auto_thresholds(user_name)
    except Exception:
        pass
