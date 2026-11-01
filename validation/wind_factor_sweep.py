"""
Wind height factor sweep.

The app scales the weather service's 10 m wind to rider height by a fixed WIND_HEIGHT_FACTOR (0.7). If that is too
high, the head-wind part of a ride is blamed for too little CdA and the tail-wind part for too much: the CdA fitted on
the head-wind samples comes out LOWER than on the tail-wind samples of the same ride. The true CdA cannot depend on the
wind direction, so the factor at which that ratio is 1 is the one the data support.

This script refits every saved ride for a range of factors, finds where the median head/tail ratio crosses 1 (with a
bootstrap interval over rides), and checks the answer on rides it was not fitted on (alternate rides, and
earlier vs later rides).

What it cannot separate: drafting (a group ride shelters the rider from the head wind), terrain shelter that varies
with the direction of travel, and a wind error that grows with wind speed. A single factor is a simplification. Treat the
result as a hypothesis to confirm with the reference tests, not as a calibration to apply blindly.

    python -m validation.wind_factor_sweep --user Pep [--since 2026-01-01] [--factors 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] [--tags rides.csv]
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.physics import aerodyn as ae
from validation import cda_consistency as cc
from validation import ride_types

DEFAULT_FACTORS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BOOTSTRAPS = 500
MIN_RIDES = 8
MATERIAL = 0.05             # a difference in the factor smaller than this is not worth acting on


def sweep(prepared, factors=DEFAULT_FACTORS, progress=False):
    """Long DataFrame, one row per ride and factor: cda, cda_head, cda_tail, ratio (head / tail), speed, head wind."""
    rows = []
    saved = ae.WIND_HEIGHT_FACTOR
    try:
        for n, p in enumerate(prepared, 1):
            if progress:
                print(f"sweeping ride {n}/{len(prepared)}", file=sys.stderr, flush=True)
            for f in factors:
                ae.WIND_HEIGHT_FACTOR = float(f)
                m = cc.ride_metrics(p['df'], p['rider_mass'], p['bike_mass'], p['crr'], with_still_air=False)
                if m is None:
                    continue
                rows.append(dict(ride_id=p['ride_id'], date=p['date'], factor=float(f), cda=m['cda'],
                                 ride_type=p.get('ride_type') or ride_types.UNKNOWN,
                                 cda_head=m['cda_head'], cda_tail=m['cda_tail'], speed_ms=m['speed_ms'],
                                 head10_ms=m['head_ms'] / float(f) if f else np.nan, wind10_ms=m['wind_ms'],
                                 ratio=m['cda_head'] / m['cda_tail'] if m['cda_tail'] else np.nan,
                                 ratio_adj=m['cda_head_adj'] / m['cda_tail_adj'] if m['cda_tail_adj'] else np.nan,
                                 delta_w=m['delta_w']))
    finally:
        ae.WIND_HEIGHT_FACTOR = saved
    return pd.DataFrame(rows)


def crossing(factors, medians):
    """Factor where the median head/tail ratio crosses 1 (linear in log ratio between neighbours), or None."""
    f, m = np.asarray(factors, float), np.log(np.asarray(medians, float))
    for i in range(len(f) - 1):
        if np.isfinite(m[i]) and np.isfinite(m[i + 1]) and m[i] * m[i + 1] <= 0 and m[i] != m[i + 1]:
            return float(f[i] + (0 - m[i]) * (f[i + 1] - f[i]) / (m[i + 1] - m[i]))
    return None


def crossing_label(factors, medians, c=None):
    """Text for where a ratio curve reaches 1: the factor, or 'below / above the range'."""
    c = crossing(factors, medians) if c is None else c
    if c is not None:
        return f"{c:.2f}"
    m = np.asarray(medians, float)
    m = m[np.isfinite(m)]
    if len(m) and np.all(m < 1):
        return f"below {min(factors):.2f}"
    return f"above {max(factors):.2f}" if len(m) else "n/a"


def _medians(wide):
    return wide.median(axis=0, skipna=True).to_numpy()


def at_factor(factors, medians, target):
    """Median ratio read off (log-linear) at a given factor."""
    return float(np.exp(np.interp(target, factors, np.log(medians))))


def _bootstrap_crossing(values, factors, seed=1):
    """(5th, 95th percentile, number of resamples that crossed) of the crossing over resampled rides; None if none did."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(values), len(values))
        c = crossing(factors, np.nanmedian(values[idx], axis=0))
        if c is not None:
            found.append(c)
    return (float(np.percentile(found, 5)), float(np.percentile(found, 95)), len(found)) if found else None


def analyse(long_df, factors=DEFAULT_FACTORS, seed=1):
    """Findings: per-factor table, crossing with a bootstrap interval, and the held-out checks."""
    from scipy import stats
    factors = list(factors)
    wide = long_df.pivot(index='ride_id', columns='factor', values='ratio').reindex(columns=factors)
    wide = wide.dropna(how='all')
    out = {'n_rides': len(wide), 'factors': factors}
    table = []
    for f in factors:
        sub = long_df[long_df['factor'] == f]
        r = sub['ratio'].dropna()
        row = dict(factor=f, rides=len(r), median_ratio=float(r.median()) if len(r) else np.nan,
                   q1=float(r.quantile(0.25)) if len(r) else np.nan, q3=float(r.quantile(0.75)) if len(r) else np.nan,
                   mean_cda=float(sub['cda'].mean()), cv_pct=float(sub['cda'].std(ddof=1) / sub['cda'].mean() * 100))
        for key, col in (('rho_speed', 'speed_ms'), ('rho_head', 'head10_ms')):
            d = sub[['cda', col]].dropna()
            row[key] = float(stats.spearmanr(d['cda'], d[col])[0]) if len(d) >= MIN_RIDES and d[col].nunique() > 1 else np.nan
        ra = sub['ratio_adj'].dropna() if 'ratio_adj' in sub else sub['ratio'].iloc[0:0]
        row['median_ratio_adj'] = float(ra.median()) if len(ra) else np.nan
        row['delta_med'] = float(sub['delta_w'].median()) if 'delta_w' in sub else np.nan
        table.append(row)
    out['table'] = pd.DataFrame(table)
    meds = _medians(wide)
    out['crossing'] = crossing(factors, meds)
    out['crossing_adj'] = crossing(factors, out['table']['median_ratio_adj'].to_numpy())
    if len(wide) >= MIN_RIDES:
        out['crossing_ci'] = _bootstrap_crossing(wide.to_numpy(), factors, seed)
        # held-out: find the crossing on one set of rides, read the median ratio on the other at that factor
        dates = long_df.drop_duplicates('ride_id').set_index('ride_id')['date']
        order = [r for r in dates.sort_values().index if r in wide.index]
        half = len(order) // 2
        sets = {'alternate rides, set A': order[0::2], 'alternate rides, set B': order[1::2],
                'earlier rides': order[:half], 'later rides': order[half:]}
        out['sets'] = {name: (len(ids), crossing(factors, _medians(wide.loc[ids])), _medians(wide.loc[ids]))
                       for name, ids in sets.items()}
        held = {}
        for name, (a, b) in {'alternate rides': ('alternate rides, set A', 'alternate rides, set B'),
                             'earlier vs later rides': ('earlier rides', 'later rides')}.items():
            ca = out['sets'][a][1]
            mb = out['sets'][b][2]
            held[name] = None if ca is None else (ca, at_factor(factors, mb, ca), at_factor(factors, mb, 0.7))
        out['held_out'] = held
        # does the ratio depend on how windy the ride was? terciles (or halves) of the ride's mean 10 m wind
        wind = long_df.drop_duplicates('ride_id').set_index('ride_id')['wind10_ms'].reindex(wide.index).dropna()
        k = 3 if len(wind) >= 3 * MIN_RIDES else (2 if len(wind) >= 2 * MIN_RIDES else 0)
        strata = []
        if k:
            edges = np.quantile(wind, np.linspace(0, 1, k + 1))
            names = ("calm", "windy") if k == 2 else ("calmest third", "middle third", "windiest third")
            for n_, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                ids = wind[(wind >= lo) & ((wind <= hi) if n_ == k - 1 else (wind < hi))].index
                sub = long_df[long_df['ride_id'].isin(ids)]
                at07 = sub[sub['factor'] == min(factors, key=lambda f: abs(f - 0.7))]['cda']
                strata.append(dict(name=names[n_], rides=len(ids), wind_lo=float(lo), wind_hi=float(hi),
                                   medians=_medians(wide.loc[ids]), crossing=crossing(factors, _medians(wide.loc[ids])),
                                   cda_mean=float(at07.mean()), cda_cv=float(at07.std(ddof=1) / at07.mean() * 100)
                                   if len(at07) > 1 else np.nan))
        out['strata'] = strata
        # solo against group rides: drafting shelters a group rider, so only solo rides test the wind model cleanly
        types = {}
        if 'ride_type' in long_df:
            kinds = long_df.drop_duplicates('ride_id').set_index('ride_id')['ride_type'].reindex(wide.index)
            f07 = min(factors, key=lambda f: abs(f - 0.7))
            for kind in ride_types.KNOWN:
                ids = kinds[kinds == kind].index
                if len(ids) < MIN_RIDES:
                    continue
                at07 = long_df[(long_df['factor'] == f07) & long_df['ride_id'].isin(ids)]['cda']
                meds_k = _medians(wide.loc[ids])
                types[kind] = dict(rides=len(ids), medians=meds_k, crossing=crossing(factors, meds_k),
                                   ci=_bootstrap_crossing(wide.loc[ids].to_numpy(), factors, seed),
                                   cda_mean=float(at07.mean()),
                                   cda_cv=float(at07.std(ddof=1) / at07.mean() * 100) if len(at07) > 1 else np.nan)
            types['_unlabelled'] = int((~kinds.isin(ride_types.KNOWN)).sum())
        out['types'] = types
        # is the crossing really located? not if it is at the edge of the range or most resamples never crossed
        step = factors[1] - factors[0] if len(factors) > 1 else 0.1
        ci = out['crossing_ci']
        out['edge'] = bool(
            (out['crossing'] is None and np.all(meds[np.isfinite(meds)] < 1)) or
            (out['crossing'] is not None and (out['crossing'] - factors[0] <= step / 2 + 1e-9 or
                                              (ci is not None and ci[2] < 0.9 * BOOTSTRAPS))))
    return out


def report(user, res, skipped):
    t = res['table']
    L = [f"# Wind height factor sweep: {user}", "",
         f"{res['n_rides']} rides with a head-wind and a tail-wind stretch. Skipped before the sweep: {skipped or 'none'}.", "",
         "Ratio = CdA fitted on the head-wind samples / CdA fitted on the tail-wind samples of the same ride "
         "(samples chosen on the 10 m wind, so they are the same whatever the factor). It should be 1.", "",
         "| Factor | Median ratio | Quartiles | Ratio after removing a constant power term | Median constant term (W) | "
         "Mean CdA | CdA spread (%) | rho CdA~speed | rho CdA~head wind |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.factor:.2f} | {r.median_ratio:.2f} | {r.q1:.2f} to {r.q3:.2f} | {r.median_ratio_adj:.2f} | "
                 f"{r.delta_med:+.1f} | {r.mean_cda:.3f} | {r.cv_pct:.1f} | {r.rho_speed:+.2f} | {r.rho_head:+.2f} |")
    L += ["", "## Where the ratio reaches 1", ""]
    c = res['crossing']
    ci = res.get('crossing_ci')
    if res.get('edge'):
        lo = min(res['factors'])
        L.append(f"The ratio reaches 1 **at or below the lowest factor tested ({lo:.2f})**: the crossing is not located "
                 f"(only {ci[2] if ci else 0} of {BOOTSTRAPS} bootstrap resamples crossed inside the range, so no "
                 "interval is given). Re-run with lower factors, for example "
                 "`--factors 0.05,0.1,0.15,0.2,0.3,0.4`. Factor 0 cannot be used: it leaves no wind to define the "
                 "head-wind and tail-wind samples.")
        L.append("")
        L.append("Either way 0.70 is far from where the ratio reaches 1." if (c is None or c < 0.5) else "")
    elif c is None:
        L.append(f"The median ratio does not cross 1 between {min(res['factors'])} and {max(res['factors'])}. "
                 "Widen --factors, or the head/tail difference is not (only) a wind-strength effect.")
    else:
        L.append(f"Median ratio = 1 at a factor of **{c:.2f}**" +
                 (f" (bootstrap 90 % interval {ci[0]:.2f} to {ci[1]:.2f} over rides, {ci[2]} of {BOOTSTRAPS} resamples "
                  "crossed)." if ci else "."))
        L.append("")
        differs = ci and not (ci[0] - MATERIAL <= 0.70 <= ci[1] + MATERIAL)
        L.append("The app uses 0.70. " + (f"The interval (widened by {MATERIAL:.2f}, the smallest change worth making) "
                                          "excludes 0.70: the data do not support the current value."
                                          if differs else
                                          "The interval includes 0.70 (within a change of "
                                          f"{MATERIAL:.2f}): no evidence to change it."))
    ca = res.get('crossing_adj')
    L += ["", "With a constant extra power term taken out of each ride first (fitted as power = CdA x drag + delta), the "
          "head/tail ratio reaches 1 at " + (f"**{ca:.2f}**" if ca is not None else
                                            f"a factor {crossing_label(res['factors'], t['median_ratio_adj'].to_numpy(), None)}")
          + ". If this is far from the crossing above, part of the head/tail difference was unmodelled resistance and "
          "not the wind; if it is the same, the wind (or how you rode into it) is the cause."]
    L += ["", "## Is the effect the same on other rides?", ""]
    sets = res.get('sets') or {}
    if sets:
        L += ["Median head/tail ratio of each set of rides at each factor, and where it reaches 1:", "",
              "| Rides | n | " + " | ".join(f"{f:.2f}" for f in res['factors']) + " | Reaches 1 at |",
              "|---|---|" + "---|" * (len(res['factors']) + 1)]
        for name, (n, cr, meds_) in sets.items():
            L.append(f"| {name} | {n} | " + " | ".join(f"{m:.2f}" for m in meds_) + f" | "
                     f"{crossing_label(res['factors'], meds_, cr)} |")
        L += ["", "Factor found on one set, read on the other:", "",
              "| Split | Factor found on the first set | Median ratio on the other set at that factor | ...and at 0.70 |",
              "|---|---|---|---|"]
        for name, v in (res.get('held_out') or {}).items():
            L.append(f"| {name} | " + ("not inside the range | n/a | n/a" if v is None else
                                       f"{v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f}") + " |")
        L += ["", "If the factor found on one set brings the ratio on the other close to 1 (and 0.70 does not), the "
              "effect is not specific to a few rides. Sets that reach 1 at very different factors (or not at all) mean "
              "no single factor fits."]
    else:
        L.append(f"Needs at least {MIN_RIDES} rides.")
    L += ["", "## Does it depend on how windy the ride was?", ""]
    strata = res.get('strata') or []
    if strata:
        L += ["Rides grouped by their mean 10 m wind. CdA mean and spread are at the factor closest to 0.70.", "",
              "| Group | Rides | 10 m wind (m/s) | " + " | ".join(f"{f:.2f}" for f in res['factors']) +
              " | Reaches 1 at | CdA mean | CdA spread (%) |",
              "|---|---|---|" + "---|" * (len(res['factors']) + 3)]
        for g in strata:
            L.append(f"| {g['name']} | {g['rides']} | {g['wind_lo']:.1f} to {g['wind_hi']:.1f} | "
                     + " | ".join(f"{m:.2f}" for m in g['medians'])
                     + f" | {crossing_label(res['factors'], g['medians'], g['crossing'])} | {g['cda_mean']:.3f} | "
                     f"{g['cda_cv']:.1f} |")
        L += ["", "In the calm group the head-wind and tail-wind samples are few and the ratio is noisy, but if the CdA "
              "there is steadier (a lower spread) than in the windy group, the wind model is adding error and "
              "calm rides are the ones to trust."]
    else:
        L.append(f"Needs at least {2 * MIN_RIDES} rides.")
    L += ["", "## Solo and group rides", ""]
    types = {k: v for k, v in (res.get('types') or {}).items() if k in ride_types.KNOWN}
    if types:
        L += ["Group rides shelter the rider from the head wind, which lowers the CdA fitted into the wind. If drafting "
              "is what keeps the ratio below 1, solo rides reach 1 at a higher factor than group rides.", "",
              "| Rides | n | " + " | ".join(f"{f:.2f}" for f in res['factors'])
              + " | Reaches 1 at | 90 % interval | CdA mean | CdA spread (%) |",
              "|---|---|" + "---|" * (len(res['factors']) + 4)]
        for kind, v in types.items():
            ci_ = v['ci']
            L.append(f"| {kind} | {v['rides']} | " + " | ".join(f"{m:.2f}" for m in v['medians'])
                     + f" | {crossing_label(res['factors'], v['medians'], v['crossing'])} | "
                     + (f"{ci_[0]:.2f} to {ci_[1]:.2f}" if ci_ else "n/a")
                     + f" | {v['cda_mean']:.3f} | {v['cda_cv']:.1f} |")
        L += ["", f"Rides without a label (left out above): {res['types'].get('_unlabelled', 0)}.", ""]
        if len(types) == 2 and all(v['crossing'] is not None for v in types.values()):
            gap = types['solo']['crossing'] - types['group']['crossing']
            cs, cg = types['solo']['ci'], types['group']['ci']
            separate = bool(cs and cg and cs[0] > cg[1])
            L.append(f"Solo reaches 1 at {types['solo']['crossing']:.2f} and group at {types['group']['crossing']:.2f} "
                     f"(difference {gap:+.2f}). "
                     + ("The intervals do not overlap: solo rides need a higher factor, so drafting explains part of the gap."
                        if separate and gap > MATERIAL else
                        "The intervals overlap, so the difference is within the uncertainty of the data."
                        if gap > MATERIAL else
                        "No clear difference: drafting does not explain the gap."))
        if 'solo' in types and types['solo']['crossing'] is not None:
            at = types['solo']['crossing']
            L.append(f"Solo rides alone suggest a factor of about {at:.2f}"
                     + ("; 0.70 is not supported even for them." if (types['solo']['ci'] and types['solo']['ci'][1] + MATERIAL < 0.7)
                        else "; this includes 0.70 within the interval, so the current value may be fine."))
    else:
        L.append(f"Needs at least {MIN_RIDES} rides labelled solo or group: use `--tags FILE` (make a template with "
                 "`--tags-template FILE`); rides imported from Strava from now on carry the group size.")
    L += ["", "## Reading it", "",
          "- A factor well below 0.70 with a ratio that reaches 1 means the wind felt was weaker than the model says "
          "(shelter, trees, buildings, the 0.25-degree weather grid).",
          "- Look at the spread column too: a better factor should not make the CdA spread worse.",
          "- Not separable here: drafting, shelter that depends on the direction of travel, and errors that grow with "
          "wind speed. Compare solo and group rides before acting on the number.",
          "- Changing WIND_HEIGHT_FACTOR changes every stored CdA. Confirm with the reference tests first."]
    return "\n".join(L) + "\n"


def write_outputs(user, long_df, res, skipped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"wind_factor_sweep_{user}")
    with open(stem + ".md", "w") as f:
        f.write(report(user, res, skipped))
    long_df.to_csv(stem + ".csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = res['table']
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
        a.fill_between(t['factor'], t['q1'], t['q3'], color="#1F7A8C", alpha=0.18, linewidth=0)
        a.plot(t['factor'], t['median_ratio'], color="#1F7A8C", marker="o", ms=4)
        a.axhline(1.0, color="#5B6575", lw=1, ls="--")
        if res['crossing'] is not None:
            a.axvline(res['crossing'], color="#B7791F", lw=1)
            a.text(res['crossing'], a.get_ylim()[0], f" {res['crossing']:.2f}", color="#B7791F", fontsize=8, va="bottom")
        a.axvline(0.7, color="#999", lw=0.8, ls=":")
        a.set_xlabel("wind height factor", fontsize=8)
        a.set_ylabel("CdA head wind / CdA tail wind", fontsize=8)
        a.set_title("Should be 1 (dotted: current 0.70)", fontsize=9)
        b.plot(t['factor'], t['cv_pct'], color="#1F7A8C", marker="o", ms=4)
        b.set_xlabel("wind height factor", fontsize=8)
        b.set_ylabel("spread of CdA between rides, %", fontsize=8)
        b.set_title("Spread of CdA", fontsize=9)
        for ax in (a, b):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
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
    ap.add_argument("--factors", default=",".join(str(f) for f in DEFAULT_FACTORS))
    ap.add_argument("--tags", help="CSV labelling rides solo or group (columns ride_id or name, and type)")
    ap.add_argument("--out", default="validation/output")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    if a.db:
        os.environ["BIKETOOLS_DB"] = a.db
    from src.data import db
    if a.db:
        db.DB_FILE = a.db
    factors = [float(x) for x in a.factors.split(",")]
    tags = cc.read_tags_or_exit(a.tags)
    prepared, skips = cc.prepare_user_rides(a.user, a.since, a.fetch, progress=True)
    ride_types.apply_tags(prepared, tags)
    prepared = [p for p in prepared if {'wind_speed', 'wind_deg', 'bearing'} <= set(p['df'].columns)]
    skipped = "; ".join(f"{n} {why}" for why, n in skips.items())
    if not prepared:
        raise SystemExit(f"No ride with weather. Skipped: {skipped or 'nothing found'} (try --fetch)")
    long_df = sweep(prepared, factors, progress=True)
    res = analyse(long_df, factors)
    stem = write_outputs(a.user, long_df, res, skipped, a.out)
    print(report(a.user, res, skipped))
    print(f"written to {stem}.(md|csv|png)")


if __name__ == "__main__":
    main()
