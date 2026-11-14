"""
How the rider is evolving: the last week or month compared with the period before it, turned into plain findings.

Everything here is computed from stored data with fixed rules; nothing is guessed or generated. Each finding has a
level, the numbers behind it, and the key of the reference (src/references.py) its rule of thumb comes from, so a page,
the weekly e-mail or (later) a language model can present it without inventing anything.

Levels: 'warning' (act on it), 'caution' (keep an eye on it), 'good', 'neutral' (information). Rules of thumb, not
medical advice.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis import durability as du
from src.analysis import intensity as it
from src.analysis import power_profile as pp
from src.analysis import training_load as tl
from src.analysis.correlations import endurance_rides
from src.analysis.labels import dur_label, tid_label
from src.analysis.power import estimate_critical_power
from src.i18n import t

LEVELS = ('warning', 'caution', 'good', 'neutral')
BEST_DURATIONS = (5, 60, 300, 1200, 3600)
CP_WINDOW_DAYS = 90
MIN_CHANGE_PCT = 3.0              # smaller changes in CP or EF are within day-to-day noise
RAMP_AGGRESSIVE = 8.0             # CTL points per week (rule of thumb, see labels.ramp_status)
RAMP_BUILDING = 3.0
MONOTONY_HIGH = 2.0               # Foster 1998
DECOUPLING_GOOD = 5.0             # Friel
HRR_DROP_BPM = 3.0                # a fall in heart-rate recovery of this much is worth watching
MIN_RIDES_FOR_TREND = 2


@dataclass
class Finding:
    key: str
    area: str
    level: str
    text: str
    value: float = None
    previous: float = None
    ref: str = None
    extra: dict = field(default_factory=dict)


def _pct(new, old):
    return (new - old) / old * 100.0 if old else None


def _window(df, start, end, col='date'):
    if df is None or df.empty:
        return df
    return df[(df[col] >= start) & (df[col] < end + pd.Timedelta(days=1))]


# --- load --------------------------------------------------------------------------------------------------------

def _volume(stats_now, stats_prev, days):
    hours = float(stats_now['moving_s'].fillna(0).sum() / 3600) if stats_now is not None and not stats_now.empty else 0.0
    prev = float(stats_prev['moving_s'].fillna(0).sum() / 3600) if stats_prev is not None and not stats_prev.empty else 0.0
    tss = float(stats_now['tss'].fillna(0).sum()) if stats_now is not None and not stats_now.empty else 0.0
    tss_prev = float(stats_prev['tss'].fillna(0).sum()) if stats_prev is not None and not stats_prev.empty else 0.0
    n = 0 if stats_now is None else len(stats_now)
    change = _pct(tss, tss_prev)
    level = 'caution' if change is not None and change > 50 and tss > 0 else 'neutral'
    if n == 0:
        text = t("No rides in the last {days} days.", days=days)
    elif change is None:
        text = t("{n} rides, {hours:.1f} h and {tss:.0f} TSS in the last {days} days (nothing in the period before).",
                 n=n, hours=hours, tss=tss, days=days)
    else:
        text = t("{n} rides, {hours:.1f} h and {tss:.0f} TSS in the last {days} days: {change:+.0f} % load compared "
                 "with the {days} days before.", n=n, hours=hours, tss=tss, days=days, change=change)
        if level == 'caution':
            text += " " + t("The app flags rises above 50 %: sharp spikes in load are linked to overreaching and injury.")
    return Finding('volume', 'load', level, text, tss, tss_prev, 'gabbett-2016', {'hours': hours, 'hours_prev': prev})


def _fitness(pmc, today, days):
    if pmc is None or pmc.empty:
        return []
    past = pmc.loc[:today]
    if past.empty:
        return []
    now = past.iloc[-1]
    before = past.loc[:today - pd.Timedelta(days=days)]
    out = []
    if not before.empty:
        ctl0 = float(before['CTL'].iloc[-1])
        per_week = (float(now['CTL']) - ctl0) / days * 7
        if per_week > RAMP_AGGRESSIVE:
            level, say = 'warning', t("faster than the ~8 points a week usually considered sustainable")
        elif per_week >= RAMP_BUILDING:
            level, say = 'good', t("a steady build")
        elif per_week > -RAMP_BUILDING:
            level, say = 'neutral', t("about stable")
        else:
            level, say = 'caution', t("fitness is going down")
        out.append(Finding('fitness', 'load', level,
                           t("Fitness (CTL) {ctl:.0f}, {delta:+.1f} points a week over the period: {say}.",
                             ctl=float(now['CTL']), delta=per_week, say=say),
                           float(now['CTL']), ctl0, 'banister-1975', {'per_week': per_week}))
    tsb = float(now['TSB'])
    if tsb < -30:
        level, say = 'warning', t("very fatigued; plan easy days before anything important")
    elif tsb < -10:
        level, say = 'neutral', t("the productive fatigue of a training block")
    elif tsb <= 25:
        level, say = 'good', t("fresh enough to train or race well")
    else:
        level, say = 'caution', t("very fresh: fine before a race, but a long stretch like this loses fitness")
    out.append(Finding('form', 'load', level, t("Form (TSB) {tsb:+.0f}: {say}.", tsb=tsb, say=say), tsb, None,
                       'allen-coggan-2010'))
    mono = tl.monotony_strain(pmc.loc[:today])['monotony']
    m = mono.iloc[-1] if len(mono) else np.nan
    if pd.notna(m) and m > MONOTONY_HIGH:
        out.append(Finding('monotony', 'load', 'caution',
                           t("Training monotony {m:.1f} this week (above 2): days are very alike. Mixing hard and easy "
                             "days lowers the strain for the same load.", m=float(m)), float(m), None, 'foster-1998'))
    return out


# --- power -------------------------------------------------------------------------------------------------------

def _cp_trend(records, today, days):
    if records is None or records.empty:
        return None
    def cp_at(end):
        curve = pp.as_dict(pp.best_curve(records, end - pd.Timedelta(days=CP_WINDOW_DAYS - 1), end))
        res = estimate_critical_power(curve) if curve else None
        return res['cp'] if res else None
    now, before = cp_at(today), cp_at(today - pd.Timedelta(days=days))
    if not now:
        return None
    if not before:
        return Finding('cp', 'power', 'neutral', t("Critical power {cp:.0f} W (from your last 90 days).", cp=now),
                       now, None, 'monod-scherrer-1965')
    change = _pct(now, before)
    level = 'good' if change >= MIN_CHANGE_PCT else 'caution' if change <= -MIN_CHANGE_PCT else 'neutral'
    text = t("Critical power {cp:.0f} W, {change:+.1f} % against {days} days ago.", cp=now, change=change, days=days)
    if level == 'caution':
        text += " " + t("It can also just mean fewer maximal efforts of 3–20 min lately.")
    return Finding('cp', 'power', level, text, now, before, 'monod-scherrer-1965')


def _bests(records, start, today):
    if records is None or records.empty:
        return []
    now = pp.best_curve(records, start, today).set_index('duration')['watts']
    before = pp.best_curve(records, None, start - pd.Timedelta(days=1))
    before = before.set_index('duration')['watts'] if not before.empty else pd.Series(dtype=float)
    new = [(d, float(now[d]), float(before[d])) for d in BEST_DURATIONS
           if d in now.index and d in before.index and now[d] > before[d]]
    if not new:
        return []
    listed = ", ".join(t("{dur} {w:.0f} W (was {old:.0f})", dur=dur_label(d), w=w, old=o) for d, w, o in new)
    return [Finding('bests', 'power', 'good', t("New personal bests: {listed}.", listed=listed),
                    len(new), None, 'allen-coggan-2010', {'bests': {d: (w, o) for d, w, o in new}})]


# --- aerobic -----------------------------------------------------------------------------------------------------

def _aerobic(stats_now, stats_prev):
    out = []
    endu, endu_prev = endurance_rides(stats_now), endurance_rides(stats_prev)
    n, n_prev = (0 if endu is None else len(endu)), (0 if endu_prev is None else len(endu_prev))
    if n >= 1:
        ef = float(endu['ef'].median())
        if n >= MIN_RIDES_FOR_TREND and n_prev >= MIN_RIDES_FOR_TREND:
            ef0 = float(endu_prev['ef'].median())
            change = _pct(ef, ef0)
            level = 'good' if change >= MIN_CHANGE_PCT else 'caution' if change <= -MIN_CHANGE_PCT else 'neutral'
            out.append(Finding('ef', 'aerobic', level,
                               t("Efficiency factor {ef:.2f} on {n} steady rides of an hour or more, {change:+.1f} % "
                                 "against the period before: more power for the same heart rate means a better "
                                 "aerobic base.", ef=ef, n=n, change=change), ef, ef0, 'friel-2018'))
        dec = endu['decoupling_pct'].dropna()
        if len(dec):
            d = float(dec.median())
            level = 'good' if d < DECOUPLING_GOOD else 'caution'
            text = (t("Aerobic decoupling {d:.1f} % on long steady rides: under 5 %, your endurance holds up.", d=d)
                    if level == 'good' else
                    t("Aerobic decoupling {d:.1f} % on long steady rides: above 5 %, heart rate drifts up late in "
                      "long rides (endurance, heat or fuelling).", d=d))
            prev = endu_prev['decoupling_pct'].dropna() if endu_prev is not None and not endu_prev.empty else []
            out.append(Finding('decoupling', 'aerobic', level, text, d,
                               float(np.median(prev)) if len(prev) else None, 'friel-2018'))
    hrr = stats_now['hrr60_bpm'].dropna() if stats_now is not None and not stats_now.empty else pd.Series(dtype=float)
    hrr0 = stats_prev['hrr60_bpm'].dropna() if stats_prev is not None and not stats_prev.empty else pd.Series(dtype=float)
    hrr, hrr0 = hrr[hrr > 0], hrr0[hrr0 > 0]
    if len(hrr) >= 3 and len(hrr0) >= 3:
        a, b = float(hrr.median()), float(hrr0.median())
        if a <= b - HRR_DROP_BPM:
            out.append(Finding('hrr', 'recovery', 'caution',
                               t("Heart rate falls {a:.0f} bpm in the minute after hard efforts, against {b:.0f} bpm "
                                 "before: slower recovery can be a sign of accumulated fatigue.", a=a, b=b),
                               a, b, 'lamberts-2011'))
        elif a >= b + HRR_DROP_BPM:
            out.append(Finding('hrr', 'recovery', 'good',
                               t("Heart rate falls {a:.0f} bpm in the minute after hard efforts, against {b:.0f} bpm "
                                 "before: faster recovery.", a=a, b=b), a, b, 'lamberts-2011'))
    return out


# --- intensity and durability --------------------------------------------------------------------------------------

def _intensity(zone_rows):
    if zone_rows is None or zone_rows.empty:
        return None
    tot = it.three_zone(zone_rows)
    total = sum(tot.values())
    if not total:
        return None
    pi = it.polarization_index(tot['low'], tot['mid'], tot['high'])
    kind = tid_label(tot['low'], tot['mid'], tot['high'], pi)
    return Finding('intensity', 'intensity', 'neutral',
                   t("Time by power: {low:.0%} easy, {mid:.0%} moderate, {high:.0%} hard ({kind}).",
                     low=tot['low'] / total, mid=tot['mid'] / total, high=tot['high'] / total, kind=kind),
                   pi, None, 'treff-2019', {'low': tot['low'] / total, 'mid': tot['mid'] / total,
                                            'high': tot['high'] / total})


def _durability(records, work_records, start, today):
    table = du.work_table(records, work_records, 'total_kjkg', start, today)
    if table.empty:
        return None
    row = table[table['duration'] == 300]
    if row.empty:
        row = table
    row = row.sort_values('threshold').iloc[-1]
    return Finding('durability', 'durability', 'neutral',
                   t("After {x:g} kJ/kg of work you still produced {kept:.0f} % of this period's best {dur} power.",
                     x=row['threshold'], kept=row['retention_pct'], dur=dur_label(int(row['duration']))),
                   float(row['retention_pct']), None, 'sanchez-jimenez-2025')


# --- all together ------------------------------------------------------------------------------------------------

def evolution(today, days, pmc=None, stats=None, records=None, zone_rows=None, work_records=None):
    """
    Findings for the last `days` days ending `today`, compared with the `days` before. Inputs are what the database
    already returns: compute_pmc(), get_ride_stats(), get_power_records(), get_zone_rows('power') and
    get_work_records(), for any period covering both windows (extra rows are ignored). Sorted by level.
    """
    today = pd.Timestamp(today).normalize()
    start = today - pd.Timedelta(days=days - 1)
    prev_start, prev_end = start - pd.Timedelta(days=days), start - pd.Timedelta(days=1)
    stats_now = _window(stats, start, today)
    stats_prev = _window(stats, prev_start, prev_end)
    out = [_volume(stats_now, stats_prev, days)] if stats is not None else []
    out += _fitness(pmc, today, days)
    cp = _cp_trend(records, today, days)
    if cp:
        out.append(cp)
    out += _bests(records, start, today)
    out += _aerobic(stats_now, stats_prev)
    zones = _intensity(_window(zone_rows, start, today))
    if zones:
        out.append(zones)
    dur = _durability(records, work_records, start, today)
    if dur:
        out.append(dur)
    return sorted(out, key=lambda f: LEVELS.index(f.level))


def headline(findings):
    """One sentence: how many findings of each kind."""
    n = {lv: sum(1 for f in findings if f.level == lv) for lv in LEVELS}
    if not findings:
        return t("Not enough rides in this period to say how you are evolving.")
    if n['warning']:
        return t("{w} point(s) to act on, {c} to watch and {g} going well.", w=n['warning'], c=n['caution'], g=n['good'])
    if n['caution']:
        return t("Nothing alarming: {c} point(s) to watch and {g} going well.", c=n['caution'], g=n['good'])
    return t("Going well: {g} positive point(s) and nothing to worry about.", g=n['good'])
