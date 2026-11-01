"""
Removes physically impossible sensor readings before any metric is computed.

One bad power sample (a 6000 W spike from a pairing glitch or a bumped crank) would otherwise
become the 1-second record, inflate NP and the zone times, and sit in the power curve for ever;
a heart-rate strap that drops out reports 0 bpm or locks on to 250.

Only impossible values are touched, so real efforts are never trimmed:
- power above MAX_POWER_W, or negative: a short glitch (up to MAX_FIX_S samples) is replaced
  by interpolating its neighbours; a longer stretch becomes 0 W (the sensor was not reporting).
  Genuine gaps (missing values) are left as they are
- heart rate outside MIN_HR..MAX_HR: dropped (unknown), not guessed
- cadence above MAX_CADENCE: dropped
"""
import numpy as np
import pandas as pd

MAX_POWER_W = 2500          # beyond the best track sprinters' peak 1 s power
MIN_HR, MAX_HR = 30, 230
MAX_CADENCE = 220
MAX_FIX_S = 3               # longest run of bad power samples that is interpolated


def clean_channels(df):
    """Returns (cleaned copy, {'power': n, 'hr': n, 'cadence': n}) with the samples changed."""
    out = df.copy()
    fixed = {'power': 0, 'hr': 0, 'cadence': 0}

    if 'power' in out.columns:
        p = pd.to_numeric(out['power'], errors='coerce')
        bad = (p > MAX_POWER_W) | (p < 0)
        fixed['power'] = int(bad.sum())
        if fixed['power']:
            run = bad.groupby((bad != bad.shift()).cumsum()).transform('size')
            short = bad & (run <= MAX_FIX_S)
            guess = p.mask(bad).interpolate(limit_area='inside')
            fixed_p = p.copy()
            fixed_p[short] = guess[short]
            fixed_p[bad & ~short] = 0.0                  # sensor was not reporting: no guessing
            fixed_p[bad & fixed_p.isna()] = 0.0          # a bad sample with no neighbour to lean on
            out['power'] = fixed_p

    if 'hr' in out.columns:
        hr = pd.to_numeric(out['hr'], errors='coerce')
        bad = hr.notna() & ((hr < MIN_HR) | (hr > MAX_HR))
        fixed['hr'] = int(bad.sum())
        if fixed['hr']:
            out['hr'] = hr.mask(bad)

    if 'cadence' in out.columns:
        cad = pd.to_numeric(out['cadence'], errors='coerce')
        bad = cad > MAX_CADENCE
        fixed['cadence'] = int(bad.sum())
        if fixed['cadence']:
            out['cadence'] = cad.mask(bad)

    return out, fixed
