"""
Recomputes stored rides from their original files with the current
pipeline, so every ride in the history has the same, up-to-date metrics.
"""
import pandas as pd

from src.data import db


def ftp_used_for_ride(row, current_ftp):
    """
    FTP to score a stored ride with. Rides keep the FTP they were analysed
    with, so changing your FTP later doesn't rewrite your past TSS:
    1. ftp_at_ride, when stored (rides saved since Phase 1);
    2. otherwise NP / IF, which recovers the FTP used when it was saved;
    3. otherwise the profile's current FTP.
    """
    if row.get('ftp_at_ride'):
        return int(row['ftp_at_ride'])
    np_val, if_val = row.get('norm_power') or 0, row.get('if_factor') or 0
    if np_val > 0 and if_val > 0:
        return int(round(np_val / if_val))
    return current_ftp


def recompute_history(user_name, current_ftp, current_lthr, all_rides=False, progress=None):
    """
    Rebuilds outdated rides (or all of them). Returns (n_updated, [failures]).
    progress: optional callback(done, total).
    """
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage  # imports streamlit

    rows = db.get_rides_for_recompute(user_name, all_rides=all_rides)
    updated, failed = 0, []
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows))
        name = row['filename'] or f"ride {row['id']}"
        try:
            df, dist_m, ele_m = prepare_ride(row['filename'] or "", row['file_content'])
            ftp = ftp_used_for_ride(row, current_ftp)
            lthr = row.get('lthr_at_ride') or current_lthr
            summ = summarize_ride(df, ftp, lthr)
            meta = db.build_ride_metadata(df, dist_m, ele_m, summ['avg_speed_kph'],
                                          summ['norm_power'], summ['tss'], summ['if'],
                                          filename=row['filename'],
                                          metrics=metrics_for_storage(summ))
            if row.get('date_time'):
                meta['date'] = row['date_time']  # keep the stored start time
            db.update_ride_metrics(row['id'], meta, summ['power_curve'])
            updated += 1
        except Exception as e:  # unreadable file etc.: report and keep going
            failed.append(f"{name}: {e}")
    if progress:
        progress(len(rows), len(rows))
    return updated, failed
