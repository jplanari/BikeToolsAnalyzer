"""
Wind direction and timing sweep.

The head-wind / tail-wind CdA ratio is well below 1 (see wind_factor_sweep.py). A weak wind effect can come from a
systematic error in the wind the model is given rather than from its strength. Two such errors are easy to test:

- a direction offset: the weather wind rotated by -80 ... +80 degrees (a wrong convention, a bearing computed
  against the wrong reference, a grid cell whose wind is turned by the terrain);
- a timing offset: the weather wind read 3 hours earlier ... 3 hours later (hourly values stamped at the start or the
  end of the hour, a time-zone slip).

Each ride is refitted for each setting, at the wind height factor(s) given. If the median head/tail ratio gets close to
1 at some offset and the CdA spread does not grow, that error is likely real and is worth finding in the code. If the
ratio barely moves, the wind is not turned or shifted, and shelter or the rider's position is the better explanation.

The head-wind and tail-wind samples are re-chosen with the shifted wind each time, as the model would do.

    python -m validation.wind_direction_sweep --user Pep [--since 2026-01-01] [--factors 0.7,0.4] [--tags rides.csv]
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.data import weather
from src.physics import aerodyn as ae
from validation import cda_consistency as cc
from validation import ride_types

ROTATIONS = (-80, -60, -40, -20, 0, 20, 40, 60, 80)         # degrees added to the direction the wind blows FROM
LAGS_H = (-3, -2, -1, 0, 1, 2, 3)                           # hours: the wind read at (time + lag)
DEFAULT_FACTORS = (0.7, 0.4)
MIN_RIDES = 8
MATERIAL_GAIN = 0.05          # the ratio must move this much closer to 1 to count as a better setting
CLOSE = 0.05                  # |ratio - 1| below this counts as "reaches 1"


def shifted(df, rotation_deg=0.0, lag_h=0.0):
    """Copy of the ride with the weather wind rotated and/or read `lag_h` hours later (clamped at the ends)."""
    out = df.copy()
    if not rotation_deg and not lag_h:
        return out
    deg = out['wind_deg'].to_numpy(dtype=float)
    spd = out['wind_speed'].to_numpy(dtype=float)
    if lag_h:
        if 'time' in out.columns:
            clock = pd.to_datetime(out['time'])
            t = (clock - clock.iloc[0]).dt.total_seconds().to_numpy()
        else:                                       # no clock (simulated rides): one sample per second
            t = np.arange(len(out), dtype=float)
        u, v = weather.wind_to_uv(spd, deg)
        good = np.isfinite(u) & np.isfinite(v)
        if good.sum() >= 2:
            u = np.interp(t + lag_h * 3600.0, t[good], u[good])
            v = np.interp(t + lag_h * 3600.0, t[good], v[good])
            spd, deg = weather.uv_to_wind(u, v)
    out['wind_speed'] = spd
    out['wind_deg'] = (deg + rotation_deg) % 360.0
    return out


def sweep(prepared, factors=DEFAULT_FACTORS, rotations=ROTATIONS, lags=LAGS_H, progress=False):
    """Long DataFrame, one row per ride, factor, kind ('rotation' or 'lag') and offset."""
    rows = []
    saved = ae.WIND_HEIGHT_FACTOR
    settings = [('rotation', r, r, 0) for r in rotations] + [('lag', l, 0, l) for l in lags if l != 0]
    try:
        for n, p in enumerate(prepared, 1):
            if progress:
                print(f"sweeping ride {n}/{len(prepared)}", file=sys.stderr, flush=True)
            for f in factors:
                ae.WIND_HEIGHT_FACTOR = float(f)
                for kind, offset, rot, lag in settings:
                    m = cc.ride_metrics(shifted(p['df'], rot, lag), p['rider_mass'], p['bike_mass'], p['crr'],
                                        with_still_air=False)
                    if m is None:
                        continue
                    rows.append(dict(ride_id=p['ride_id'], date=p['date'], factor=float(f), kind=kind, offset=offset,
                                     ride_type=p.get('ride_type') or ride_types.UNKNOWN, cda=m['cda'],
                                     ratio=m['cda_head'] / m['cda_tail'] if m['cda_tail'] else np.nan))
    finally:
        ae.WIND_HEIGHT_FACTOR = saved
    return pd.DataFrame(rows)


def analyse(long_df, factors=DEFAULT_FACTORS):
    """Per factor and kind: a table by offset, the best offset, and whether it is a material improvement."""
    out = {'n_rides': int(long_df['ride_id'].nunique()), 'factors': list(factors), 'results': {}}
    for f in factors:
        for kind in ('rotation', 'lag'):
            sub = long_df[(long_df['factor'] == f) & ((long_df['kind'] == kind) | (long_df['offset'] == 0))]
            if kind == 'lag':                       # the zero-offset rows are stored under 'rotation'
                sub = pd.concat([sub, long_df[(long_df['factor'] == f) & (long_df['kind'] == 'rotation')
                                              & (long_df['offset'] == 0)].assign(kind='lag')])
            sub = sub[sub['kind'] == kind]
            table = []
            for off, g in sub.groupby('offset'):
                r = g['ratio'].dropna()
                table.append(dict(offset=float(off), rides=len(r),
                                  median=float(r.median()) if len(r) else np.nan,
                                  q1=float(r.quantile(0.25)) if len(r) else np.nan,
                                  q3=float(r.quantile(0.75)) if len(r) else np.nan,
                                  cda_mean=float(g['cda'].mean()),
                                  cda_cv=float(g['cda'].std(ddof=1) / g['cda'].mean() * 100.0) if len(g) > 1 else np.nan))
            t = pd.DataFrame(table).sort_values('offset').reset_index(drop=True)
            res = dict(table=t, best=None, zero=None, improves=False, reaches=False)
            ok = t[t['median'].notna() & (t['rides'] >= MIN_RIDES)]
            if len(ok) and (ok['offset'] == 0).any():
                dist = (np.log(ok['median'])).abs()
                best = ok.loc[dist.idxmin()]
                zero = ok[ok['offset'] == 0].iloc[0]
                res['best'], res['zero'] = best.to_dict(), zero.to_dict()
                gain = abs(zero['median'] - 1.0) - abs(best['median'] - 1.0)
                res['improves'] = bool(best['offset'] != 0 and gain > MATERIAL_GAIN)
                res['reaches'] = bool(abs(best['median'] - 1.0) <= CLOSE)
            out['results'][(f, kind)] = res
    return out


def _verdict(res, kind):
    unit = "degrees" if kind == 'rotation' else "hours"
    b, z = res['best'], res['zero']
    if b is None:
        return "Too few rides to say."
    if res['improves'] and res['reaches']:
        return (f"Best at {b['offset']:+.0f} {unit}: the ratio reaches {b['median']:.2f} (from {z['median']:.2f} with no "
                f"offset), CdA spread {b['cda_cv']:.1f} % (from {z['cda_cv']:.1f} %). A real offset in the wind is "
                "likely: look for it in the code.")
    if res['improves']:
        return (f"Best at {b['offset']:+.0f} {unit}: {b['median']:.2f} against {z['median']:.2f} with no offset. Better, "
                "but still not near 1: part of the problem, not all of it.")
    return (f"No offset helps: the ratio is {z['median']:.2f} with none and {b['median']:.2f} at the best "
            f"({b['offset']:+.0f} {unit}). The wind is not turned or shifted by this much.")


def report(user, res, skipped):
    L = [f"# Wind direction and timing sweep: {user}", "",
         f"{res['n_rides']} rides. Skipped before the sweep: {skipped or 'none'}.", "",
         "Median CdA(head wind) / CdA(tail wind) with the weather wind rotated or read earlier/later. It should be 1. "
         "The head-wind and tail-wind samples are re-chosen with the shifted wind each time.", ""]
    for f in res['factors']:
        L += [f"## Wind height factor {f:.2f}", ""]
        for kind, title, unit in (('rotation', "Direction offset (degrees added to where the wind blows from)", "degrees"),
                                  ('lag', "Time offset (hours: the wind read at time + offset)", "hours")):
            r = res['results'][(f, kind)]
            L += [f"### {title}", "", f"| Offset ({unit}) | Rides | Median ratio | Quartiles | Mean CdA | CdA spread (%) |",
                  "|---|---|---|---|---|---|"]
            for x in r['table'].itertuples():
                L.append(f"| {x.offset:+.0f} | {x.rides} | {x.median:.2f} | {x.q1:.2f} to {x.q3:.2f} | "
                         f"{x.cda_mean:.3f} | {x.cda_cv:.1f} |")
            L += ["", _verdict(r, kind), ""]
    L += ["## Reading it", "",
          "- A direction offset only matters if the model really turns the wind; a best offset near 0 rules that out.",
          "- The two offsets are tested one at a time. An improvement that is real should show at the same offset for "
          "both wind factors and not make the CdA spread worse.",
          "- No improvement anywhere points at shelter that the weather grid cannot see, or at the rider's position "
          "changing with the wind. Only a reference test (a controlled out-and-back) can tell those apart."]
    return "\n".join(L) + "\n"


def write_outputs(user, long_df, res, skipped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"wind_direction_sweep_{user}")
    with open(stem + ".md", "w") as f:
        f.write(report(user, res, skipped))
    long_df.to_csv(stem + ".csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, kind, lab in zip(axes, ('rotation', 'lag'), ("direction offset (degrees)", "time offset (hours)")):
            for f, c in zip(res['factors'], ("#1F7A8C", "#B7791F", "#5B6575")):
                t = res['results'][(f, kind)]['table']
                ax.plot(t['offset'], t['median'], marker="o", ms=4, color=c, label=f"factor {f:.2f}")
            ax.axhline(1.0, color="#5B6575", lw=1, ls="--")
            ax.set_xlabel(lab, fontsize=8)
            ax.set_ylabel("CdA head wind / CdA tail wind", fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False)
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
    ap.add_argument("--only", choices=("solo", "group"), help="use only rides labelled so (needs --tags)")
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
    if a.only:
        prepared = [p for p in prepared if p['ride_type'] == a.only]
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
