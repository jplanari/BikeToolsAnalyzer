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

    python -m validation.wind_factor_sweep --user Pep [--since 2026-01-01] [--factors 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.physics import aerodyn as ae
from validation import cda_consistency as cc

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
                                 cda_head=m['cda_head'], cda_tail=m['cda_tail'], speed_ms=m['speed_ms'],
                                 head10_ms=m['head_ms'] / float(f) if f else np.nan,
                                 ratio=m['cda_head'] / m['cda_tail'] if m['cda_tail'] else np.nan))
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


def _medians(wide):
    return wide.median(axis=0, skipna=True).to_numpy()


def at_factor(factors, medians, target):
    """Median ratio read off (log-linear) at a given factor."""
    return float(np.exp(np.interp(target, factors, np.log(medians))))


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
        table.append(row)
    out['table'] = pd.DataFrame(table)
    meds = _medians(wide)
    out['crossing'] = crossing(factors, meds)
    if len(wide) >= MIN_RIDES:
        rng = np.random.default_rng(seed)
        values = wide.to_numpy()
        found = []
        for _ in range(BOOTSTRAPS):
            idx = rng.integers(0, len(values), len(values))
            c = crossing(factors, np.nanmedian(values[idx], axis=0))
            if c is not None:
                found.append(c)
        out['crossing_ci'] = (float(np.percentile(found, 5)), float(np.percentile(found, 95)), len(found)) if found else None
        # held-out: find the crossing on one set of rides, read the median ratio on the other at that factor
        dates = long_df.drop_duplicates('ride_id').set_index('ride_id')['date']
        order = [r for r in dates.sort_values().index if r in wide.index]
        splits = {'alternate rides': (order[0::2], order[1::2]), 'earlier vs later rides': (order[:len(order) // 2],
                                                                                           order[len(order) // 2:])}
        held = {}
        for name, (a, b) in splits.items():
            ca = crossing(factors, _medians(wide.loc[a]))
            held[name] = None if ca is None else (ca, at_factor(factors, _medians(wide.loc[b]), ca),
                                                   at_factor(factors, _medians(wide.loc[b]), 0.7))
        out['held_out'] = held
    return out


def report(user, res, skipped):
    t = res['table']
    L = [f"# Wind height factor sweep: {user}", "",
         f"{res['n_rides']} rides with a head-wind and a tail-wind stretch. Skipped before the sweep: {skipped or 'none'}.", "",
         "Ratio = CdA fitted on the head-wind samples / CdA fitted on the tail-wind samples of the same ride "
         "(samples chosen on the 10 m wind, so they are the same whatever the factor). It should be 1.", "",
         "| Factor | Median ratio | Quartiles | Mean CdA | CdA spread (%) | rho CdA~speed | rho CdA~head wind |",
         "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.factor:.2f} | {r.median_ratio:.2f} | {r.q1:.2f} to {r.q3:.2f} | {r.mean_cda:.3f} | {r.cv_pct:.1f} | "
                 f"{r.rho_speed:+.2f} | {r.rho_head:+.2f} |")
    L += ["", "## Where the ratio reaches 1", ""]
    c = res['crossing']
    if c is None:
        L.append(f"The median ratio does not cross 1 between {min(res['factors'])} and {max(res['factors'])}. "
                 "Widen --factors, or the head/tail difference is not (only) a wind-strength effect.")
    else:
        ci = res.get('crossing_ci')
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
    L += ["", "## Checked on rides not used for the fit", ""]
    held = res.get('held_out') or {}
    if held:
        L += ["| Split | Factor found on the first set | Median ratio on the other set at that factor | "
              "...and at 0.70 |", "|---|---|---|---|"]
        for name, v in held.items():
            L.append(f"| {name} | " + (" | ".join(["no crossing", "", ""]) if v is None else
                                        f"{v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} |") + " |")
        L.append("")
        L.append("If the factor found on one set brings the ratio on the other close to 1 (and 0.70 does not), the "
                 "effect is not specific to a few rides.")
    else:
        L.append(f"Needs at least {MIN_RIDES} rides.")
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
    ap.add_argument("--out", default="validation/output")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    if a.db:
        os.environ["BIKETOOLS_DB"] = a.db
    from src.data import db
    if a.db:
        db.DB_FILE = a.db
    factors = [float(x) for x in a.factors.split(",")]
    prepared, skips = cc.prepare_user_rides(a.user, a.since, a.fetch, progress=True)
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
