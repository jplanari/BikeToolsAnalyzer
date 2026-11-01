"""
CdA error budget by simulation.

Rides with a known CdA (0.300 m2) are simulated, the inputs the estimator relies on are made wrong in controlled ways,
and the error of the fitted CdA is measured. Two outputs:

  1. one-at-a-time: how much CdA moves for a typical size of each error (sign and size);
  2. Monte Carlo: the spread of the CdA error when each error alone, and all together, are drawn at random.

It says which inputs matter and what accuracy is possible at all. It does NOT say the estimator matches reality:
that needs reference tests (validation/README.md). The error sizes in sim.DEFAULT_SD are assumptions.

    python -m validation.cda_error_budget --runs 200 --seed 1 --out validation/output
"""
import argparse
import csv
import os
import sys

import numpy as np

from validation import sim


def one_at_a_time(rng):
    """{scenario: {label: error or None}} for each step in sim.OAT_STEPS (and the error-free baseline)."""
    table = {}
    for name, sc in sim.SCENARIOS.items():
        truth = sim.simulate(sc)
        row = {'no error (baseline)': sim.cda_error(truth, sim.Errors(), rng)}
        for field, step, label in sim.OAT_STEPS:
            vals = []
            for sign in ((1,) if isinstance(step, bool) or field in sim.NOISE_FIELDS else (1, -1)):
                err = sim.Errors(**{field: step if isinstance(step, bool) else sign * step})
                vals.append(sim.cda_error(truth, err, np.random.default_rng(0)))
            row[label] = vals[0] if len(vals) == 1 else (vals[0], vals[1])
        table[name] = row
    return table


def monte_carlo(runs, rng, sd=None):
    """{scenario: {factor or 'ALL': (mean, sd, p5, p95, refused)}} of the relative CdA error."""
    sd = sd or sim.DEFAULT_SD
    out = {}
    for name, sc in sim.SCENARIOS.items():
        truth = sim.simulate(sc)
        res = {}
        for factor in list(sd) + ['ALL']:
            errs, refused = [], 0
            for _ in range(runs):
                e = sim.cda_error(truth, sim.draw(sd, rng, None if factor == 'ALL' else factor), rng)
                refused += e is None
                if e is not None:
                    errs.append(e)
            a = np.array(errs) if errs else np.array([np.nan])
            res[factor] = (float(a.mean()), float(a.std()), float(np.percentile(a, 5)), float(np.percentile(a, 95)),
                           refused)
        out[name] = res
    return out


def _pct(v):
    return "refused" if v is None else f"{v * 100:+.1f}"


def report(oat, mc, runs):
    L = ["# CdA error budget (simulation)", "",
         f"True CdA {sim.TRUE['cda']:.3f} m2, rider {sim.TRUE['rider_mass']:.0f} kg, bike {sim.TRUE['bike_mass']:.0f} kg, "
         f"Crr {sim.TRUE['crr']}, rho {sim.TRUE['rho']}. Errors are fitted CdA / true CdA - 1, in percent.", "",
         "The error sizes are assumptions (`validation/sim.py`); replace them with device specifications.", "",
         "## One error at a time (typical size; + / - are the two signs of the step)", ""]
    scen = list(oat)
    L += ["| Error | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
    for label in oat[scen[0]]:
        cells = []
        for s in scen:
            v = oat[s][label]
            cells.append(f"{_pct(v[0])} / {_pct(v[1])}" if isinstance(v, tuple) else _pct(v))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", f"## Random errors ({runs} runs per cell): spread of the CdA error (SD, percent)", "",
          "| Error source (SD) | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
    for f in list(sim.DEFAULT_SD) + ['ALL']:
        name = "**all together**" if f == 'ALL' else f"{f} ({sim.DEFAULT_SD[f]:g})"
        L.append(f"| {name} | " + " | ".join(f"{mc[s][f][1] * 100:.1f}" for s in scen) + " |")
    L += ["", "## All errors together: bias and 90 % interval of the CdA error (percent)", "",
          "| Scenario | Mean | 5 % | 95 % | Fits refused |", "|---|---|---|---|---|"]
    for s in scen:
        m, _, p5, p95, ref = mc[s]['ALL']
        L.append(f"| {s} | {m * 100:+.1f} | {p5 * 100:+.1f} | {p95 * 100:+.1f} | {ref} |")
    return "\n".join(L) + "\n"


def write_outputs(oat, mc, runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cda_error_budget.md"), "w") as f:
        f.write(report(oat, mc, runs))
    with open(os.path.join(out_dir, "cda_error_budget_montecarlo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "error_source", "sd", "mean_err", "sd_err", "p5_err", "p95_err", "refused"])
        for s, res in mc.items():
            for k, (m, sdv, p5, p95, ref) in res.items():
                w.writerow([s, k, sim.DEFAULT_SD.get(k, ''), m, sdv, p5, p95, ref])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        factors = list(sim.DEFAULT_SD)
        mean_sd = [np.mean([mc[s][f][1] for s in mc]) * 100 for f in factors]
        order = np.argsort(mean_sd)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.barh([factors[i] for i in order], [mean_sd[i] for i in order], color="#1F7A8C", height=0.6)
        for y, i in enumerate(order):
            ax.text(mean_sd[i] + 0.1, y, f"{mean_sd[i]:.1f}", va="center", fontsize=8, color="#333")
        ax.set_xlabel("SD of the CdA error alone, % (mean over the scenarios)")
        ax.set_title("Which input errors matter most for CdA (simulation, assumed error sizes)", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cda_error_budget.png"), dpi=150)
        plt.close(fig)
    except Exception as e:                                     # the tables are the result; the chart is a convenience
        print(f"chart skipped: {e}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=200, help="random draws per cell (default 200)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="validation/output", help="folder for the report, CSV and chart")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    oat = one_at_a_time(rng)
    mc = monte_carlo(a.runs, rng)
    write_outputs(oat, mc, a.runs, a.out)
    print(report(oat, mc, a.runs))
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
