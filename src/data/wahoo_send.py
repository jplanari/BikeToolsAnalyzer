"""
Sends interval-built planned sessions to Wahoo (plan + scheduled workout) so they appear on the
ELEMNT, and keeps them in step when resent. No Streamlit.

Resending updates the same plan and workout (their ids are remembered in wahoo_sent). Head units
only re-download a plan when the workout's updated_at changes, which Wahoo does not bump for an
unchanged PUT, so each update moves the start by one second (:00 <-> :01).
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.analysis import wahoo_plan as wp
from src.analysis import workout as wo
from src.data import db
from src.data import wahoo as wh
from src.data.wahoo_import import get_valid_auth


class SendError(Exception):
    pass


def _tzinfo(name):
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception as e:
            raise SendError(f"Unknown time zone '{name}' in secrets.toml [wahoo] timezone.") from e
    return datetime.now().astimezone().tzinfo           # this computer's zone


def starts_utc(day, local_time, tz_name="", second=0):
    """ISO UTC start for a local date and time, e.g. '2026-10-31T06:00:00Z'."""
    d = pd.Timestamp(day).date()
    local = datetime(d.year, d.month, d.day, local_time.hour, local_time.minute, second, tzinfo=_tzinfo(tz_name))
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def in_sync_window(day, today=None):
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    return today <= pd.Timestamp(day).normalize() <= today + pd.Timedelta(days=wh.SYNC_WINDOW_DAYS)


def _ids(user_name, session_id):
    user = db.get_user_data(user_name)
    return f"biketools-{user['id']}-{int(session_id)}"


def send_session(cfg, user_name, session, ftp, local_time, outdoor=True, usage=None, now=None):
    """
    session: a planned-session row (id, date, note, blocks). Returns {'plan_id', 'workout_id', 'starts',
    'updated': bool}. Raises SendError for sessions that cannot be sent, wh.WahooError otherwise.
    """
    try:
        blocks = wo.clean_blocks(json.loads(session['blocks']))
    except (TypeError, ValueError):
        blocks = []
    if not blocks:
        raise SendError("Only sessions built from intervals can be sent (quick TSS entries have no steps).")
    name = (session.get('note') or wo.describe(blocks))[:80]
    plan = wp.plan_json(blocks, ftp, name, outdoor=outdoor)
    minutes = max(1, round(wo.session_stats(blocks, ftp)['duration_s'] / 60))
    ext_id = _ids(user_name, session['id'])
    filename = f"{ext_id}.json"
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    auth = get_valid_auth(cfg, user_name)
    token = auth['access_token']
    prev = db.get_wahoo_sent(user_name).get(int(session['id']))

    plan_id = workout_id = None
    if prev:
        try:
            wh.update_plan(token, prev['plan_id'], plan, name, filename, stamp, usage=usage)
            plan_id = prev['plan_id']
        except wh.WahooNotFound:
            prev = None                       # deleted on Wahoo's side: start over
    if not prev:
        found = wh.find_plan(token, ext_id, usage=usage)
        if found:
            plan_id = found['id']
            wh.update_plan(token, plan_id, plan, name, filename, stamp, usage=usage)
        else:
            plan_id = wh.create_plan(token, plan, name, filename, ext_id, stamp, usage=usage)['id']

    # alternate :00 / :01 on every resend so the head unit notices the change
    second = 1 if prev and prev['starts'].endswith(":00Z") else 0
    starts = starts_utc(session['date'], local_time, cfg.timezone, second)
    if prev:
        try:
            wh.update_workout(token, prev['workout_id'], name, ext_id, starts, minutes, plan_id, usage=usage)
            workout_id = prev['workout_id']
        except wh.WahooNotFound:
            prev = None
    if not prev:
        workout_id = wh.schedule_workout(token, name, ext_id, starts, minutes, plan_id, usage=usage)['id']

    db.save_wahoo_sent(session['id'], plan_id, workout_id, starts, stamp)
    return {'plan_id': plan_id, 'workout_id': workout_id, 'starts': starts, 'updated': prev is not None}


def remove_session(cfg, user_name, session_id, usage=None):
    """Deletes the workout and plan on Wahoo (if the session was sent) and forgets it here."""
    prev = db.get_wahoo_sent(user_name).get(int(session_id))
    if not prev:
        return False
    token = get_valid_auth(cfg, user_name)['access_token']
    wh.delete_workout(token, prev['workout_id'], usage=usage)
    wh.delete_plan(token, prev['plan_id'], usage=usage)
    db.delete_wahoo_sent(session_id)
    return True
