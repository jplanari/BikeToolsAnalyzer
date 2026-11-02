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

    python -m validation.cda_consistency --user Pep [--db path/to/bikethools.db] [--since 2026-01-01] [--fetch] [--tags rides.csv]
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.physics import aerodyn as ae
from validation import ride_types

MIN_SAMPLES = 120               # usable seconds needed for a fit (same as estimate_cda)
HEAD_THRESHOLD_10M_MS = 0.7     # head / tail halves need at least this much 10 m wind along the road (0.5 m/s at 0.7)
MIN_RIDES_FOR_STATS = 8
CORR_FLAG = 0.4                 # |Spearman rho| from here, and p below ALPHA divided by the number of tests, is flagged
ALPHA = 0.05


def _fit(k, p):
    """Least-squares CdA over the samples, the same estimator as aerodyn.estimate_cda."""
    return float((p * k).sum() / (k * k).sum()) if len(k) >= MIN_SAMPLES else np.nan


def _joint(k, p):
    """
    (CdA, constant power term in W) from the ordinary least-squares line p = CdA * k + delta over the samples.
    delta > 0 is resistance the balance does not account for (rolling resistance or drivetrain loss too low, a power
    meter reading high); it matters most where drag is small. NaN when there is too little to tell the two apart or
    the CdA is implausible.
    """
    if len(k) < MIN_SAMPLES or float(k.std()) == 0.0:
        return np.nan, np.nan
    kc = k - k.mean()
    slope = float((kc * (p - p.mean())).sum() / (kc * kc).sum())
    if not ae.CDA_VALID[0] <= slope <= ae.CDA_VALID[1]:
        return np.nan, np.nan
    return slope, float(p.mean() - slope * k.mean())


def ride_metrics(df, rider_mass, bike_mass, crr, loss=ae.DRIVETRAIN_LOSS, with_still_air=True):
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
    fit_still = ae.estimate_cda(still, **kw) if with_still_air else None

    b = ae._balance(df, rider_mass, bike_mass, crr, None, loss)
    ok = ae._usable(df, b) & b['unit_drag'].notna() & b['p_aero'].notna() & (b['unit_drag'] > 0)
    k, p = b['unit_drag'][ok], b['p_aero'][ok]
    row = {'cda': fit['cda'], 'cda_still_air': fit_still['cda'] if fit_still else np.nan, 'seconds': fit['seconds'],
           'has_wind': bool(has_wind), 'cda_head': np.nan, 'cda_tail': np.nan,
           'speed_ms': float(b['v'][ok].mean()), 'grade_pct': float(b['slope'][ok].mean() * 100.0),
           'temp_c': float(df['temp_c'].mean()) if 'temp_c' in df.columns and df['temp_c'].notna().any() else np.nan,
           'head_ms': np.nan, 'cross_ms': np.nan, 'wind_ms': np.nan,
           'cda_head_adj': np.nan, 'cda_tail_adj': np.nan}
    # the same ride with a constant extra power term allowed: CdA and delta together
    row['cda_joint'], row['delta_w'] = _joint(k, p)
    mean_v = float(b['v'][ok].mean())
    row['delta_crr'] = row['delta_w'] / ((rider_mass + bike_mass) * ae.G * mean_v) if np.isfinite(row['delta_w']) else np.nan
    p_adj = p - row['delta_w'] if np.isfinite(row['delta_w']) else None
    if has_wind:
        head, cross = ae._wind_components(df)
        row['head_ms'], row['cross_ms'] = float(head[ok].mean()), float(cross[ok].abs().mean())
        row['wind_ms'] = float(df['wind_speed'][ok].mean())
        # the halves are chosen on the 10 m wind, not the scaled one, so they are the same samples whatever
        # WIND_HEIGHT_FACTOR is (the sweep in wind_factor_sweep.py changes it)
        head10 = head[ok] / ae.WIND_HEIGHT_FACTOR if ae.WIND_HEIGHT_FACTOR else head[ok] * 0.0
        h_mask, t_mask = head10 > HEAD_THRESHOLD_10M_MS, head10 < -HEAD_THRESHOLD_10M_MS
        row['cda_head'], row['cda_tail'] = _fit(k[h_mask], p[h_mask]), _fit(k[t_mask], p[t_mask])
        if p_adj is not None:                 # head / tail halves once the constant term is taken out
            row['cda_head_adj'], row['cda_tail_adj'] = _fit(k[h_mask], p_adj[h_mask]), _fit(k[t_mask], p_adj[t_mask])
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
    jt = rides[['cda', 'cda_joint', 'delta_w', 'delta_crr', 'speed_ms']].dropna() if 'delta_w' in rides else None
    if jt is not None and len(jt) >= MIN_RIDES_FOR_STATS:
        try:
            p_delta = 1.0 if (jt['delta_w'] == 0).all() else float(stats.wilcoxon(jt['delta_w']).pvalue)
        except ValueError:
            p_delta = np.nan
        j = dict(n=len(jt), delta_med=float(jt['delta_w'].median()), delta_q1=float(jt['delta_w'].quantile(0.25)),
                 delta_q3=float(jt['delta_w'].quantile(0.75)), delta_p=p_delta,
                 frac_positive=float((jt['delta_w'] > 0).mean()), crr_med=float(jt['delta_crr'].median()),
                 cda_mean=float(jt['cda_joint'].mean()),
                 cda_cv=float(jt['cda_joint'].std(ddof=1) / jt['cda_joint'].mean() * 100.0),
                 rho_speed=float(stats.spearmanr(jt['cda_joint'], jt['speed_ms'])[0]),
                 rho_speed_plain=float(stats.spearmanr(jt['cda'], jt['speed_ms'])[0]))
        ha = rides[['cda_head_adj', 'cda_tail_adj']].dropna()
        j['ratio_adj'] = (len(ha), float((ha['cda_head_adj'] / ha['cda_tail_adj']).median())) if len(ha) >= MIN_RIDES_FOR_STATS else None
        out['joint'] = j
    if 'ride_type' in rides:
        by_type = {}
        for kind in ride_types.KNOWN:
            g = rides[rides['ride_type'] == kind]
            if len(g) < MIN_RIDES_FOR_STATS:
                continue
            ht_g = g[['cda_head', 'cda_tail']].dropna()
            ratio_g = ht_g['cda_head'] / ht_g['cda_tail']
            by_type[kind] = dict(n=len(g), cda_mean=float(g['cda'].mean()),
                                 cda_cv=float(g['cda'].std(ddof=1) / g['cda'].mean() * 100.0),
                                 n_ratio=len(ratio_g), ratio=float(ratio_g.median()) if len(ratio_g) else np.nan,
                                 ratio_q1=float(ratio_g.quantile(0.25)) if len(ratio_g) else np.nan,
                                 ratio_q3=float(ratio_g.quantile(0.75)) if len(ratio_g) else np.nan,
                                 _ratios=ratio_g.to_numpy())
        if len(by_type) == 2 and all(v['n_ratio'] >= MIN_RIDES_FOR_STATS for v in by_type.values()):
            a, b = (by_type[k]['_ratios'] for k in ride_types.KNOWN)
            by_type['p_ratio'] = float(stats.mannwhitneyu(a, b).pvalue)
        for v in by_type.values():
            if isinstance(v, dict):
                v.pop('_ratios', None)
        out['unlabelled'] = int((~rides['ride_type'].isin(ride_types.KNOWN)).sum())
        if by_type:
            out['by_type'] = by_type
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
    L += ["", "## With a constant extra power term allowed", ""]
    j = res.get('joint')
    if j:
        L += [f"Each ride is also fitted as power = CdA x drag + delta. A positive delta is resistance the balance does not "
              f"account for (rolling resistance or drivetrain loss too low, a power meter reading high). {j['n']} rides.", "",
              f"- delta: median {j['delta_med']:+.1f} W (quartiles {j['delta_q1']:+.1f} to {j['delta_q3']:+.1f}); "
              f"positive in {j['frac_positive'] * 100:.0f} % of rides; Wilcoxon p = {j['delta_p']:.3f}.",
              f"- equivalent change of the rolling resistance coefficient: {j['crr_med']:+.4f} "
              "(delta divided by mass x g x speed).",
              f"- CdA with delta: mean {j['cda_mean']:.3f} m2, spread {j['cda_cv']:.1f} % "
              f"(without: mean {res['mean']:.3f}, spread {res['cv_pct']:.1f} %).",
              f"- CdA against speed: rho {j['rho_speed']:+.2f} with delta, {j['rho_speed_plain']:+.2f} without."]
        if j['ratio_adj']:
            L.append(f"- head-wind / tail-wind CdA ratio after taking delta out: median {j['ratio_adj'][1]:.2f} "
                     f"({j['ratio_adj'][0]} rides; compare the ratio above).")
        L += ["", "Reading it: a clearly positive delta that weakens the CdA-with-speed trend points at unmodelled resistance "
              "(check the bike's Crr and tyres, and the power meter's zero offset) and not at the wind. The two "
              "parameters are partly interchangeable, so single-ride values of delta are noisy: look at the median over "
              "many rides."]
    else:
        L.append(f"Needs at least {MIN_RIDES_FOR_STATS} rides.")
    L += ["", "## Solo and group rides", ""]
    bt = res.get('by_type')
    if bt:
        L += ["| Rides | n | Mean CdA | CdA spread (%) | Head/tail CdA ratio (median, quartiles) | Rides in the ratio |",
              "|---|---|---|---|---|---|"]
        for kind in ride_types.KNOWN:
            v = bt.get(kind)
            if v:
                L.append(f"| {kind} | {v['n']} | {v['cda_mean']:.3f} | {v['cda_cv']:.1f} | "
                         + (f"{v['ratio']:.2f} ({v['ratio_q1']:.2f} to {v['ratio_q3']:.2f})" if v['n_ratio'] else "n/a")
                         + f" | {v['n_ratio']} |")
        if 'p_ratio' in bt:
            L += ["", f"Difference between the two head/tail ratios: Mann-Whitney p = {bt['p_ratio']:.3f}."]
        L += ["", f"Rides without a label (left out above): {res.get('unlabelled', 0)}.", "",
              "Drafting lowers the apparent CdA in head winds. If solo rides sit clearly closer to 1 than group rides, "
              "drafting explains part of the ratio; if both are equally far from 1, it does not."]
    else:
        L.append("Needs at least "
                 f"{MIN_RIDES_FOR_STATS} labelled rides of a type: use `--tags FILE` (make a template with "
                 "`--tags-template FILE`); rides imported from Strava from now on carry the group size.")
    L += ["", "## What the weather wind changes", ""]
    if 'wind_effect' in res:
        n, med = res['wind_effect']
        L.append(f"{n} rides: using the weather wind instead of still air changes the CdA by a median {med:+.1f} %.")
    else:
        L.append("Needs rides with weather (see --fetch).")
    return "\n".join(L) + "\n"


def prepare_user_rides(user_name, since=None, fetch=False, progress=False):
    """
    (list of prepared rides, {reason: count} of skipped ones) for a user's saved rides. A prepared ride is a dict with
    the analysed DataFrame (bearing and stored weather attached) and the physics the app would use for it.
    Heavy imports are local.
    """
    from src.analysis.bikes import bike_physics, label as bike_label
    from src.data import db
    from src.data.providers import weather
    from src.data.parsers.gpx import calculate_bearing
    from src.data.processing.pipeline import prepare_ride

    user = db.get_user_data(user_name)
    if not user:
        raise SystemExit(f"No user named {user_name!r} in {db.DB_FILE}")
    bikes = {b['id']: b for b in db.get_bikes(user_name)}
    default = next((b for b in bikes.values() if b.get('is_default')), None)
    prepared, skips = [], {}

    def skip(why):
        skips[why] = skips.get(why, 0) + 1

    rides = [r for r in db.get_user_rides(user_name) if not since or (r['date_time'] or '') >= since]
    for n, r in enumerate(rides, 1):
        if progress:
            print(f"ride {n}/{len(rides)}", file=sys.stderr, flush=True)
        filename, content = db.get_ride_file(user_name, r['id'])
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
        stored = db.get_ride_weather(user_name, r['id'])
        if stored is None and fetch:
            df, stored, _ = weather.load_or_fetch(df)
            if stored is not None:
                db.save_ride_weather(user_name, r['id'], stored)
        elif stored is not None:
            df = weather.apply_weather(df, stored)
        else:
            skip("no stored weather (still air assumed, use --fetch)")
        found = db.get_ride_bike(user_name, content)
        bike_id = found[1] if found else None
        bike = None if bike_id == 0 else (bikes.get(bike_id) if bike_id else default)
        bike_mass, crr = bike_physics(bike)
        prepared.append(dict(df=df, rider_mass=user['weight'], bike_mass=bike_mass, crr=crr, ride_id=r['id'],
                             date=r['date_time'], name=filename, bike=bike_label(bike),
                             ride_type=ride_types.type_from_content(content)))
    return prepared, skips


def load_rides(user_name, since=None, fetch=False, progress=False, tags=None):
    """(DataFrame of ride_metrics rows, skipped-reason text) for a user's saved rides. tags: see ride_types.read_tags."""
    prepared, skips = prepare_user_rides(user_name, since, fetch, progress)
    ride_types.apply_tags(prepared, tags)
    rows = []
    for p in prepared:
        m = ride_metrics(p['df'], p['rider_mass'], p['bike_mass'], p['crr'])
        if m is None:
            why = "too few usable samples (indoor, stops, or an implausible fit)"
            skips[why] = skips.get(why, 0) + 1
            continue
        m.update(ride_id=p['ride_id'], date=p['date'], name=p['name'], bike=p['bike'],
                 ride_type=p.get('ride_type') or ride_types.UNKNOWN)
        rows.append(m)
    return pd.DataFrame(rows), "; ".join(f"{n} {why}" for why, n in skips.items())


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


def read_tags_or_exit(path):
    if not path:
        return None
    try:
        return ride_types.read_tags(path)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot use the tags file {path}: {e}")


def write_tags_template(user, path, since=None):
    prepared, _ = prepare_user_rides(user, since, fetch=False, progress=True)
    ride_types.apply_tags(prepared)
    n = ride_types.write_template(prepared, path)
    c = ride_types.summary(prepared)
    print(f"{n} rides written to {path} ({c['solo']} solo and {c['group']} group already known from Strava, "
          f"{c['unknown']} to fill in: type solo or group in the last column).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True)
    ap.add_argument("--db", help="database file (default: the app's, BIKETOOLS_DB or bikethools.db)")
    ap.add_argument("--since", help="only rides from this date (YYYY-MM-DD)")
    ap.add_argument("--fetch", action="store_true", help="fetch and store the weather of rides that have none")
    ap.add_argument("--tags", help="CSV labelling rides solo or group (columns ride_id or name, and type)")
    ap.add_argument("--tags-template", help="write a CSV to label the rides in, then stop")
    ap.add_argument("--out", default="validation/output")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)          # Streamlit warns that there is no app runtime; not relevant here
    if a.db:
        os.environ["BIKETOOLS_DB"] = a.db
    from src.data import db
    if a.db:
        db.DB_FILE = a.db
    if a.tags_template:
        return write_tags_template(a.user, a.tags_template, a.since)
    tags = read_tags_or_exit(a.tags)
    rides, skipped = load_rides(a.user, a.since, a.fetch, progress=True, tags=tags)
    if rides.empty:
        raise SystemExit(f"No ride could be fitted. Skipped: {skipped or 'nothing found'}")
    res = analyse(rides)
    stem = write_outputs(a.user, rides, res, skipped, a.out)
    print(report(a.user, rides, res, skipped))
    print(f"written to {stem}.(md|csv|png)")


if __name__ == "__main__":
    main()
