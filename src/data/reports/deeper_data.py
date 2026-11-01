"""
Everything the "Deeper analytics" page shows, for one period and the period before it, gathered without
Streamlit or plotting. The numbers come from the same functions as the page (src.analysis.correlations),
so the weekly PDF and the screen agree.
"""
import pandas as pd

from src.analysis import correlations as co
from src.data import db


def collect_deeper_data(user_name, start, end):
    """
    start, end: inclusive dates of the period; the comparison period is the same number of days just before.
    Raises ValueError for an unknown user. Sections without data hold empty frames / None.
    """
    user = db.get_user_data(user_name)
    if not user:
        raise ValueError(f"unknown user {user_name!r}")
    a0, a1 = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    span = (a1 - a0) + pd.Timedelta(days=1)
    b0, b1 = a0 - span, a0 - pd.Timedelta(days=1)
    ftp = user['ftp']
    crank_mm = float(user.get('crank_mm') or 172.5)

    hr_a, hr_b = db.get_hist(user_name, 'p_hr', a0, a1), db.get_hist(user_name, 'p_hr', b0, b1)
    cad_a, cad_b = db.get_hist(user_name, 'p_cad', a0, a1), db.get_hist(user_name, 'p_cad', b0, b1)
    stats = db.get_ride_stats(user_name, b0, a1)
    ref_p = int(round(0.65 * ftp / 5) * 5) if ftp else 150

    # --- power vs heart rate ---------------------------------------------------------------
    hr = None
    if not hr_a.empty:
        fit_a, fit_b = co.linear_fit(hr_a), co.linear_fit(hr_b)
        hr_all = pd.concat([x for x in (hr_b, hr_a) if not x.empty])
        hr = {'ref_power': ref_p, 'fit': fit_a, 'fit_prev': fit_b,
              'at_ref': co.y_at_power(hr_a, ref_p), 'at_ref_prev': co.y_at_power(hr_b, ref_p),
              'cells': co.aggregate(hr_a), 'curve': co.mean_y_by_power(hr_a),
              'curve_prev': co.mean_y_by_power(hr_b) if not hr_b.empty else None,
              'weekly': co.weekly(hr_all, lambda g: co.y_at_power(g, ref_p, min_seconds=180))}

    # --- power vs cadence ------------------------------------------------------------------
    cad = None
    if not cad_a.empty:
        ref_cad = co.weighted_median_y(cad_a)
        ref_prev = co.weighted_median_y(cad_b) if not cad_b.empty else None
        cad_all = pd.concat([x for x in (cad_b, cad_a) if not x.empty])
        cad = {'median': ref_cad, 'median_prev': ref_prev, 'crank_mm': crank_mm,
               'quadrants': co.quadrants(cad_a, ftp, ref_cad),
               'quadrants_prev': co.quadrants(cad_b, ftp, ref_cad) if not cad_b.empty else None,
               'fit': co.linear_fit(cad_a, y_bin=5), 'cells': co.aggregate(cad_a),
               'curve': co.mean_y_by_power(cad_a, y_bin=5), 'by_zone': co.cadence_by_zone(cad_a, ftp),
               'weekly': co.weekly(cad_all, lambda g: co.weighted_median_y(g) if g['seconds'].sum() >= 600 else None)}

    # --- recovery and efficiency -----------------------------------------------------------
    in_a = (stats['date'] >= a0) & (stats['date'] < a1 + pd.Timedelta(days=1)) if not stats.empty else None
    hrr = stats.dropna(subset=['hrr60_bpm']) if not stats.empty else stats
    endu = co.endurance_rides(stats)

    def med(frame, col, this_period=True):
        if frame is None or frame.empty:
            return None
        mask = in_a.reindex(frame.index, fill_value=False)
        sel = frame[col][mask if this_period else ~mask].dropna()
        return float(sel.median()) if len(sel) else None

    recovery = {'hrr': med(hrr, 'hrr60_bpm'), 'hrr_prev': med(hrr, 'hrr60_bpm', False),
                'ef': med(endu, 'ef'), 'ef_prev': med(endu, 'ef', False),
                'decoupling': med(endu, 'decoupling_pct'), 'decoupling_prev': med(endu, 'decoupling_pct', False),
                'hrr_series': hrr.set_index('date')['hrr60_bpm'] if hrr is not None and not hrr.empty else None,
                'ef_series': endu.set_index('date')['ef'] if endu is not None and not endu.empty else None}
    return {'user': user, 'ftp': ftp, 'start': a0, 'end': a1, 'prev_start': b0, 'prev_end': b1,
            'hr': hr, 'cadence': cad, 'recovery': recovery}
