"""
Everything the PDF training report shows, gathered in one place (no Streamlit, no plotting).
The numbers come from the same functions as the Training page, so the report and the screen agree.
"""
import pandas as pd

from src.analysis import durability as du
from src.analysis import intensity as it
from src.analysis import power_profile as pp
from src.analysis import training_load as tl
from src.analysis.power import estimate_critical_power
from src.data import db


def collect_report_data(user_name, period_days=91, today=None):
    """
    period_days: length of the report period ending today (None = whole history).
    Raises ValueError for an unknown user. Sections without data come back empty (never None
    where a table is expected), so the layout code only has to check `.empty` / `if value`.
    """
    user = db.get_user_data(user_name)
    if not user:
        raise ValueError(f"unknown user {user_name!r}")
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    start = None if period_days is None else today - pd.Timedelta(days=period_days - 1)

    # --- load ---------------------------------------------------------------------------
    history = db.get_user_tss_history(user_name)
    plan_rows = db.get_planned_sessions(user_name, after=today)
    planned = plan_rows[['date', 'tss']].to_dict('records')
    pmc_all = tl.compute_pmc(history, today=today, planned=planned)
    ramp, ratio, mono = tl.ramp_rate(pmc_all), tl.acwr(pmc_all), tl.monotony_strain(pmc_all)

    def last_actual(series):
        s = series.loc[:today].dropna() if series is not None and len(series) else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else None

    now = {}
    if not pmc_all.loc[:today].empty:                 # empty when every ride is after `today` (a past week)
        row = pmc_all.loc[:today].iloc[-1]
        now = {'ctl': float(row['CTL']), 'atl': float(row['ATL']), 'tsb': float(row['TSB']),
               'ramp': last_actual(ramp), 'acwr': last_actual(ratio),
               'monotony': last_actual(mono['monotony']) if not mono.empty else None}
    pmc = pmc_all if start is None or pmc_all.empty else pmc_all.loc[start:]
    forecast = None
    if not pmc_all.empty and pmc_all['planned'].any():
        last = pmc_all.iloc[-1]
        forecast = {'date': pmc_all.index[-1], 'tsb': float(last['TSB']), 'ctl': float(last['CTL'])}

    # --- volume --------------------------------------------------------------------------
    stats = db.get_ride_stats(user_name, start, today)
    freq = 'M' if period_days is None or period_days > 190 else 'W'
    volume = tl.volume(stats, freq)
    totals = {
        'rides': int(len(stats)),
        'hours': float(stats['moving_s'].sum() / 3600) if len(stats) else 0.0,
        'km': float(stats['distance_km'].sum()) if len(stats) else 0.0,
        'elevation_m': float(stats['elevation_m'].sum()) if len(stats) else 0.0,
        'kj': float(stats['kj'].sum()) if len(stats) else 0.0,
        'tss': float(stats['tss'].sum()) if len(stats) else 0.0,
    }

    # --- power -----------------------------------------------------------------------------
    recs = db.get_power_records(user_name)
    curves = pp.window_curves(recs, today)
    recent = pp.as_dict(curves['Last 90 days'])
    all_time = pp.as_dict(curves['All time'])
    cp2 = estimate_critical_power(recent) if recent else None
    cp3 = pp.fit_cp3(recent) if recent else None
    cp_hist = pp.cp_history(recs, today=today)
    if start is not None and not cp_hist.empty:
        cp_hist = cp_hist.loc[start:]
    profile = pp.power_profile(recent, user['weight'], user['ftp']).merge(
        pp.power_profile(all_time, user['weight'], user['ftp'])[['duration', 'w_per_kg']].rename(
            columns={'w_per_kg': 'all_time'}), on='duration')
    records = curves['All time'].copy()
    records['recent'] = records['duration'].map(curves['Last 90 days'].set_index('duration')['watts']) \
        if not curves['Last 90 days'].empty else float('nan')

    # --- durability and intensity ------------------------------------------------------------
    durability = du.durability_table(db.get_power_records(user_name, start, today),
                                     db.get_kj_records(user_name, start, today), start, today)
    zone_kind = 'power'
    zones = db.get_zone_rows(user_name, 'power', start, today)
    if zones.empty:                                   # rides without a power meter
        zone_kind = 'hr'
        zones = db.get_zone_rows(user_name, 'hr', start, today)
    distribution = it.distribution(zones, freq) if not zones.empty else pd.DataFrame()
    shares, pi = None, None
    if not zones.empty:
        tot = it.three_zone(zones)
        total = sum(tot.values())
        if total:
            shares = {k: v / total for k, v in tot.items()}
            pi = it.polarization_index(tot['low'], tot['mid'], tot['high'])

    return {
        'user': user, 'today': today, 'start': start, 'period_days': period_days, 'freq': freq,
        'now': now, 'pmc': pmc, 'ramp': ramp.loc[pmc.index[0]:today] if not pmc.empty else ramp,
        'acwr': ratio.loc[pmc.index[0]:today] if not pmc.empty else ratio, 'forecast': forecast,
        'volume': volume, 'totals': totals, 'stats': stats,
        'curves': curves, 'cp2': cp2, 'cp3': cp3, 'cp_history': cp_hist, 'profile': profile,
        'records': records, 'durability': durability,
        'zone_kind': zone_kind, 'distribution': distribution, 'shares': shares, 'polarization': pi,
        'planned': plan_rows, 'suggestion': pp.suggest_threshold(recs, user['ftp'], today),
        'rides_needing_update': db.count_rides_needing_update(user_name),
    }
