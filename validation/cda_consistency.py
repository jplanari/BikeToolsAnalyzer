"""
CdA self-consistency on a rider's own rides.

Same rider, bike and position should give a stable CdA. This script fits the CdA of every saved ride the way the app
does and looks for what should NOT matter but does:

  * spread: how much CdA varies between rides (overall and per bike);
  * correlation of CdA with wind, speed, gradient and temperature: a clear trend means a modelling bias;
  * head-wind vs tail-wind halves of the same ride: if the wind model were right they agree; a ratio that is
    consistently away from 1 points at the wind (strength, direction or the 0.7 height factor);
  * with the weather service wind vs still air: how much the wind changes the answer.

It cannot say the CdA is *right* (no reference), only whether it behaves like a CdA. Reads the app's database;
writes nothing to it unless --fetch is given (then the weather of rides without any is fetched and stored,
as the app does when a ride is opened).

    python -m validation.cda_consistency --user Pep [--db path/to/bikethools.db] [--since 2026-01-01] [--fetch]
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

from src.physics import aerodyn as ae

MIN_SAMPLES = 120               # usable seconds needed for a fit (same as estimate_cda)
HEAD_THRESHOLD_MS = 0.5         # the head / tail wind halves need at least this much wind along the road
MIN_RIDES_FOR_STATS = 8
CORR_FLAG = 0.4                 # |Spearman rho| from here, and p below ALPHA divided by the number of tests, is flagged
ALPHA = 0.05


def _fit(k, p):
    """Least-squares CdA over the samples, the same estimator as aerodyn.estimate_cda."""
    return float((p * k).sum() / (k * k).sum()) if len(k) >= MIN_SAMPLES else np.nan


def ride_metrics(df, rider_mass, bike_mass, crr, loss=ae.DRIVETRAIN_LOSS):
    """
    CdA of one ride in several ways, plus the conditions it was ridden in. None when the ride cannot be fitted.
    `df` is the analysed ride: speed, power, grade_smooth, and bearing and wind columns when the ride has weather.
    """
    kw = dict(rider_mass=rider_mass, bike_mass=bike_mass, crr=crr, drivetrain_loss=loss)
    fit = ae.estimate_cda(df, **kw)
    if fit is None:
        return None
    has_wind = {'wind_speed', 'wind_deg', 'bearing'} <= set(df.columns) and df['wind_speed'].notna().any()
    still = df.drop(columns=[c for c in ('wind_speed', 'wind_deg') if c in df.columns])
    fit_still = ae.estimate_cda(still, **kw)

    b = ae._balance(df, rider_mass, bike_mass, crr, None, loss)
    ok = ae._usable(df, b) & b['unit_drag'].notna() & b['p_aero'].notna() & (b['unit_drag'] > 0)
    k, p = b['unit_drag'][ok], b['p_aero'][ok]
    row = {'cda': fit['cda'], 'cda_still_air': fit_still['cda'] if fit_still else np.nan, 'seconds': fit['seconds'],
           'has_wind': bool(has_wind), 'cda_head': np.nan, 'cda_tail': np.nan,
           'speed_ms': float(b['v'][ok].mean()), 'grade_pct': float(b['slope'][ok].mean() * 100.0),
           'temp_c': float(df['temp_c'].mean()) if 'temp_c' in df.columns and df['temp_c'].notna().any() else np.nan,
           'head_ms': np.nan, 'cross_ms': np.nan, 'wind_ms': np.nan}
    if has_wind:
        head, cross = ae._wind_components(df)
        row['head_ms'], row['cross_ms'] = float(head[ok].mean()), float(cross[ok].abs().mean())
        row['wind_ms'] = float(df['wind_speed'][ok].mean())
        row['cda_head'] = _fit(k[head[ok] > HEAD_THRESHOLD_MS], p[head[ok] > HEAD_THRESHOLD_MS])
        row['cda_tail'] = _fit(k[head[ok] < -HEAD_THRESHOLD_MS], p[head[ok] < -HEAD_THRESHOLD_MS])
    return row


def analyse(rides):
    """
    rides: DataFrame with one row per ride (ride_metrics output plus 'bike' and 'date').
    Returns a dict of findings: spread, per-bike spread, correlations, head/tail ratio, wind effect.
    """
    from scipy import stats
    out = {'n': len(rides)}
    c = rides['cda']
    out['mean'], out['sd'] = float(c.mean()), float(c.std(ddof=1)) if len(c) > 1 else np.nan
    out['cv_pct'] = out['sd'] / out['mean'] * 100.0 if out['mean'] else np.nan
    out['per_bike'] = {b: (len(g), float(g['cda'].mean()), float(g['cda'].std(ddof=1)) if len(g) > 1 else np.nan)
                       for b, g in rides.groupby('bike')}
    tests, corr = [], {}
    for col, label in (('head_ms', "head wind (m/s)"), ('cross_ms', "cross wind (m/s)"), ('speed_ms', "speed (m/s)"),
                       ('grade_pct', "gradient (%)"), ('temp_c', "temperature (C)")):
        sub = rides[['cda', col]].dropna()
        if len(sub) >= MIN_RIDES_FOR_STATS and sub[col].nunique() > 1:
            rho, pval = stats.spearmanr(sub['cda'], sub[col])
            tests.append((label, len(sub), float(rho), float(pval)))
    for label, n, rho, pval in tests:                  # Bonferroni: several conditions are tested at once
        corr[label] = (n, rho, pval, bool(abs(rho) >= CORR_FLAG and pval < ALPHA / len(tests)))
    out['corr'] = corr
    ht = rides[['cda_head', 'cda_tail']].dropna()
    if len(ht) >= MIN_RIDES_FOR_STATS:
        ratio = ht['cda_head'] / ht['cda_tail']
        diff = ratio - 1.0
        pval = 1.0 if (diff == 0).all() else float(stats.wilcoxon(diff).pvalue)
        out['head_tail'] = (len(ht), float(ratio.median()), float(ratio.quantile(0.25)), float(ratio.quantile(0.75)), pval)
    w = rides[rides['has_wind']][['cda', 'cda_still_air']].dropna()
    if len(w) >= 3:
        out['wind_effect'] = (len(w), float(((w['cda'] / w['cda_still_air']) - 1.0).median() * 100.0))
    return out


def report(user, rides, res, skipped):
    L = [f"# CdA self-consistency: {user}", "",
         f"{res['n']} rides fitted. Skipped: {skipped or 'none'}.", "",
         "A CdA that behaves like a CdA is stable between rides on the same bike and shows no trend with conditions it "
         "should not depend on. This is a consistency check, not an accuracy test.", "",
         "## Spread", "", f"All rides: mean {res['mean']:.3f} m2, SD {res['sd']:.3f} m2 ({res['cv_pct']:.1f} %).", "",
         "| Bike | Rides | Mean CdA | SD |", "|---|---|---|---|"]
    for b, (n, m, s) in res['per_bike'].items():
        L.append(f"| {b} | {n} | {m:.3f} | {'n/a' if np.isnan(s) else f'{s:.3f}'} |")
    L += ["", "Note: different positions (hoods, drops, tri bars) legitimately differ; compare like with like.", "",
          "## Trends with conditions (Spearman correlation of CdA across rides)", "",
          "A trend is flagged when |rho| >= 0.4 and p is below 0.05 divided by the number of conditions tested.", ""]
    if res['corr']:
        L += ["| Condition | Rides | rho | p | Flag |", "|---|---|---|---|---|"]
        for lab, (n, rho, p, flag) in res['corr'].items():
            L.append(f"| {lab} | {n} | {rho:+.2f} | {p:.3f} | {'possible bias' if flag else ''} |")
    else:
        L.append(f"Needs at least {MIN_RIDES_FOR_STATS} rides with the condition known.")
    L += ["", "## Head-wind half vs tail-wind half of the same ride", ""]
    if 'head_tail' in res:
        n, med, q1, q3, p = res['head_tail']
        L += [f"{n} rides: CdA(head wind) / CdA(tail wind) has median {med:.2f} (quartiles {q1:.2f} to {q3:.2f}; "
              f"Wilcoxon p = {p:.3f}).", "",
              "A median near 1 means the wind model is consistent with the power. Well above 1: the model "
              "under-states the head wind (or over-states the tail wind); well below 1, the reverse."]
    else:
        L.append(f"Needs at least {MIN_RIDES_FOR_STATS} rides with a head-wind and a tail-wind stretch, with weather.")
    L += ["", "## What the weather wind changes", ""]
    if 'wind_effect' in res:
        n, med = res['wind_effect']
        L.append(f"{n} rides: using the weather wind instead of still air changes the CdA by a median {med:+.1f} %.")
    else:
        L.append("Needs rides with weather (see --fetch).")
    return "\n".join(L) + "\n"


def load_rides(user_name, since=None, fetch=False):
    """(DataFrame of ride_metrics rows, skipped-reason text) for a user's saved rides. Heavy imports are local."""
    from src.analysis.bikes import bike_physics, label as bike_label
    from src.data import db, weather
    from src.data.gpx import calculate_bearing
    from src.data.pipeline import prepare_ride

    user = db.get_user_data(user_name)
    if not user:
        raise SystemExit(f"No user named {user_name!r} in {db.DB_FILE}")
    bikes = {b['id']: b for b in db.get_bikes(user_name)}
    default = next((b for b in bikes.values() if b.get('is_default')), None)
    rows, skips = [], {}

    def skip(why):
        skips[why] = skips.get(why, 0) + 1

    for r in db.get_user_rides(user_name):
        if since and (r['date_time'] or '') < since:
            continue
        filename, content = db.get_ride_file(r['id'])
        if content is None:
            skip("original file not stored")
            continue
        try:
            df, _, _ = prepare_ride(filename or "", content)
        except Exception:
            skip("file could not be read")
            continue
        if not {'power', 'speed'} <= set(df.columns) or df['power'].notna().sum() < MIN_SAMPLES:
            skip("no power or speed")
            continue
        df['bearing'] = calculate_bearing(df)
        stored = db.get_ride_weather(r['id'])
        if stored is None and fetch:
            df, stored, _ = weather.load_or_fetch(df)
            if stored is not None:
                db.save_ride_weather(r['id'], stored)
        elif stored is not None:
            df = weather.apply_weather(df, stored)
        else:
            skip("no stored weather (still air assumed, use --fetch)")
        found = db.get_ride_bike(user_name, content)
        bike_id = found[1] if found else None
        bike = None if bike_id == 0 else (bikes.get(bike_id) if bike_id else default)
        bike_mass, crr = bike_physics(bike)
        m = ride_metrics(df, user['weight'], bike_mass, crr)
        if m is None:
            skip("too few usable samples (indoor, stops, or an implausible fit)")
            continue
        m.update(ride_id=r['id'], date=r['date_time'], name=filename, bike=bike_label(bike))
        rows.append(m)
    text = "; ".join(f"{n} {why}" for why, n in skips.items())
    return pd.DataFrame(rows), text


def write_outputs(user, rides, res, skipped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"cda_consistency_{user}")
    with open(stem + ".md", "w") as f:
        f.write(report(user, rides, res, skipped))
    rides.to_csv(stem + ".csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = [('head_ms', "head wind along the road (m/s)"), ('speed_ms', "mean speed (m/s)"),
                  ('grade_pct', "mean gradient (%)"), ('temp_c', "temperature (C)")]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
        for ax, (col, lab) in zip(axes.flat, panels):
            sub = rides[['cda', col]].dropna()
            ax.scatter(sub[col], sub['cda'], s=18, color="#1F7A8C", alpha=0.85)
            if len(sub) >= 3 and sub[col].nunique() > 1:
                a, b = np.polyfit(sub[col], sub['cda'], 1)
                xs = np.array([sub[col].min(), sub[col].max()])
                ax.plot(xs, a * xs + b, color="#5B6575", lw=1)
            ax.set_xlabel(lab, fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("CdA (m2)", fontsize=8)
        fig.suptitle(f"CdA of each ride against conditions it should not depend on: {user}", fontsize=10)
        fig.tight_layout()
        fig.savefig(stem + ".png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"chart skipped: {e}", file=sys.stderr)
    return stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True)
    ap.add_argument("--db", help="database file (default: the app's, BIKETOOLS_DB or bikethools.db)")
    ap.add_argument("--since", help="only rides from this date (YYYY-MM-DD)")
    ap.add_argument("--fetch", action="store_true", help="fetch and store the weather of rides that have none")
    ap.add_argument("--out", default="validation/output")
    a = ap.parse_args(argv)
    if a.db:
        os.environ["BIKETOOLS_DB"] = a.db
    from src.data import db
    if a.db:
        db.DB_FILE = a.db
    rides, skipped = load_rides(a.user, a.since, a.fetch)
    if rides.empty:
        raise SystemExit(f"No ride could be fitted. Skipped: {skipped or 'nothing found'}")
    res = analyse(rides)
    stem = write_outputs(a.user, rides, res, skipped, a.out)
    print(report(a.user, rides, res, skipped))
    print(f"written to {stem}.(md|csv|png)")


if __name__ == "__main__":
    main()
