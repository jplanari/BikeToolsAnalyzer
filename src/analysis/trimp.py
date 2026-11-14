"""
Heart-rate training load (TRIMP), for rides with or without a power meter.

* Edwards (1993): minutes in five heart-rate zones of 10 % of maximum heart rate (50-60 % ... 90-100 %), weighted
  1 to 5 and summed. Needs only the maximum heart rate.
* Banister (1991): minutes x fractional heart-rate reserve x k e^(b x fractional reserve), second by second, with
  k = 0.64, b = 1.92 derived for men and k = 0.86, b = 1.67 for women. Needs maximum and resting heart rate. The
  profile has no sex field, so the men's constants are used; for a woman that reads 5-20 % low at endurance to
  threshold intensities (the ratio of the two formulas is 1.34 e^(-0.25 x)).

Only moving seconds with a heart rate count. Both return None when the profile lacks what they need.
"""
import numpy as np
import pandas as pd

EDWARDS_ZONES = (0.5, 0.6, 0.7, 0.8, 0.9)        # lower bounds, as fractions of maximum heart rate; weights 1..5
BANISTER_MEN = (0.64, 1.92)
BANISTER_WOMEN = (0.86, 1.67)


def _hr(hr_series, moving=None):
    hr = pd.to_numeric(pd.Series(hr_series), errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(hr) & (hr > 0)
    if moving is not None:
        ok &= np.asarray(pd.Series(moving).fillna(False), dtype=bool)
    return hr[ok]


def edwards(hr_series, max_hr, moving=None):
    """Edwards TRIMP (arbitrary units). 1 Hz samples."""
    if not max_hr or hr_series is None:
        return None
    hr = _hr(hr_series, moving)
    if len(hr) == 0:
        return None
    frac = hr / float(max_hr)
    weight = np.zeros(len(frac))
    for i, lo in enumerate(EDWARDS_ZONES):
        weight[frac >= lo] = i + 1
    return float(weight.sum() / 60.0)


def banister(hr_series, rest_hr, max_hr, moving=None, constants=BANISTER_MEN):
    """Banister TRIMP (arbitrary units). 1 Hz samples."""
    if not max_hr or not rest_hr or max_hr <= rest_hr or hr_series is None:
        return None
    hr = _hr(hr_series, moving)
    if len(hr) == 0:
        return None
    x = np.clip((hr - rest_hr) / float(max_hr - rest_hr), 0.0, 1.0)
    k, b = constants
    return float(np.sum(x * k * np.exp(b * x)) / 60.0)
