"""
FTP and LTHR left empty follow the rider's data: FTP = 95 % of the best 20-minute power and LTHR = 95 % of
the best 20-minute heart rate (the usual field-test rules), taken over the last 90 days, or over all rides
when nothing recent exists. Until a ride has a 20-minute effort the defaults (200 W / 170 bpm) stay.
Values the rider set by hand are never touched.
"""
import pandas as pd

from src.analysis import power_profile as pp
from src.data import db

WINDOW_DAYS = 90
MIN_LTHR, MAX_LTHR = 90, 215          # outside this a "20-minute HR" is an artefact, not a threshold


def estimate_ftp(records, today=None, window_days=WINDOW_DAYS):
    """95 % of the best 20-min power in the window (all rides when the window has none); None without one."""
    if records is None or records.empty:
        return None
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    for start in (today - pd.Timedelta(days=window_days - 1), None):
        best = pp.best_curve(records, start, today)
        row = best[best['duration'] == 1200]
        if not row.empty and row['watts'].iloc[0] > 0:
            return int(round(0.95 * float(row['watts'].iloc[0])))
    return None


def estimate_lthr(hr20_rows, today=None, window_days=WINDOW_DAYS):
    """95 % of the best 20-min mean HR; hr20_rows: [(date, bpm)]. None without a plausible value."""
    if not hr20_rows:
        return None
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    dates = pd.to_datetime([d for d, _ in hr20_rows], errors='coerce')
    bpm = pd.Series([b for _, b in hr20_rows], dtype=float)
    ok = bpm.between(MIN_LTHR / 0.95, MAX_LTHR / 0.95).to_numpy()
    for since in (today - pd.Timedelta(days=window_days - 1), None):
        keep = ok & ((dates >= since) if since is not None else True)
        if keep.any():
            return int(round(0.95 * bpm[keep].max()))
    return None


def refresh_auto_thresholds(user_name, today=None):
    """Stores new estimates for the thresholds that follow the data. Returns {'ftp': .., 'lthr': ..} changed."""
    user = db.get_user_data(user_name)
    if not user:
        return {}
    changed = {}
    if user.get('ftp_auto'):
        est = estimate_ftp(db.get_power_records(user_name), today)
        if est and est != user['ftp']:
            changed['ftp'] = est
    if user.get('lthr_auto'):
        est = estimate_lthr(db.get_hr20(user_name), today)
        if est and est != user['lthr']:
            changed['lthr'] = est
    if changed:
        db.set_auto_thresholds(user_name, **changed)
    return changed


def backfill_hr20(user_name, limit=25):
    """
    Computes the best-20-min HR of rides saved before it was stored (newest first, `limit` per call), so an
    old history can give an LTHR estimate. Rides that cannot be read are marked 0 and not retried.
    Returns how many rides were processed.
    """
    from src.analysis.hr import estimate_hr_threshold
    from src.data.pipeline import prepare_ride          # imports streamlit
    rows = db.get_rides_without_hr20(user_name, limit)
    for ride_id, filename in rows:
        peak = 0.0
        try:
            _, content = db.get_ride_file(ride_id)
            if content is not None:
                df, _, _ = prepare_ride(filename or "", content)
                if 'hr' in df.columns:
                    peak = float(estimate_hr_threshold(df['hr'])[1] or 0.0)
        except Exception:
            peak = 0.0
        db.set_ride_hr20(ride_id, peak)
    return len(rows)
