"""
Steady-state climbing model: how long a climb takes for a given power, system weight, CdA and
rolling resistance (and the reverse: the power needed for a target time).

On each short step of the climb the rider's wheel power balances gravity, rolling resistance and
air drag at constant speed,

    P * eff = v * ( m g (sin t + Crr cos t) + 1/2 rho CdA v^2 ),     tan t = gradient

and the time is the sum of step length / speed. No wind, no acceleration, constant power: it is
a what-if tool, accurate to a few percent on steady climbs, and the gaps between scenarios are
more reliable than any single estimate. Descents inside a climb are capped at MAX_SPEED_MS.
"""
import numpy as np

G = 9.80665
STEP_M = 25.0
MAX_SPEED_MS = 17.0            # about 60 km/h: nobody pedals a "climb" faster than this
MIN_POWER_W = 1.0


def climb_profile(dist_m, ele_m, step_m=STEP_M, smooth_steps=3):
    """
    Step lengths (m) and gradients (fraction) of a climb from its distance and elevation samples,
    resampled to `step_m` and lightly smoothed. Returns (lengths, grades) arrays; empty if unusable.
    """
    d = np.asarray(dist_m, dtype=float)
    e = np.asarray(ele_m, dtype=float)
    ok = ~(np.isnan(d) | np.isnan(e))
    d, e = d[ok], e[ok]
    if len(d) < 2:
        return np.array([]), np.array([])
    order = np.argsort(d, kind="stable")
    d, e = d[order], e[order]
    total = d[-1] - d[0]
    if total < step_m:
        return np.array([]), np.array([])
    n = max(1, int(round(total / step_m)))
    grid = np.linspace(d[0], d[-1], n + 1)
    elev = np.interp(grid, d, e)
    lengths = np.diff(grid)
    grades = np.diff(elev) / lengths
    if smooth_steps > 1 and len(grades) >= smooth_steps:
        k = np.ones(smooth_steps) / smooth_steps
        padded = np.pad(grades, smooth_steps // 2, mode="edge")
        grades = np.convolve(padded, k, mode="valid")[:len(grades)]
    return lengths, np.clip(grades, -0.20, 0.30)


def speed_for_power(power_w, grades, mass_kg, cda, crr, rho=1.225, efficiency=0.97):
    """Steady speed (m/s) on each gradient (array) for a constant power at the pedals."""
    grades = np.asarray(grades, dtype=float)
    p_wheel = max(float(power_w), MIN_POWER_W) * efficiency
    cos_t = 1.0 / np.sqrt(1.0 + grades ** 2)
    sin_t = grades * cos_t
    a = mass_kg * G * (sin_t + crr * cos_t)          # speed-independent force (N)
    b = 0.5 * rho * cda                              # drag coefficient (kg/m)
    lo = np.zeros_like(grades)
    hi = np.full_like(grades, MAX_SPEED_MS)
    for _ in range(50):                              # f(v) = a v + b v^3 - P has one positive root
        mid = (lo + hi) / 2
        f = a * mid + b * mid ** 3 - p_wheel
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return np.maximum((lo + hi) / 2, 0.05)


def climb_time(lengths, grades, power_w, mass_kg, cda, crr, rho=1.225, efficiency=0.97):
    """Result dict: time_s, avg_speed_kmh, vam_m_h, length_m, gain_m. None for an empty profile."""
    lengths = np.asarray(lengths, dtype=float)
    if len(lengths) == 0:
        return None
    v = speed_for_power(power_w, grades, mass_kg, cda, crr, rho, efficiency)
    t = float(np.sum(lengths / v))
    length = float(lengths.sum())
    gain = float(np.sum(lengths * np.asarray(grades)))
    return {"time_s": t, "avg_speed_kmh": length / t * 3.6, "length_m": length, "gain_m": gain,
            "vam_m_h": gain / t * 3600 if t > 0 else 0.0}


def power_for_time(lengths, grades, target_s, mass_kg, cda, crr, rho=1.225, efficiency=0.97,
                   lo=20.0, hi=1500.0):
    """Constant power (W) that finishes the climb in `target_s`; None if outside lo..hi."""
    lengths = np.asarray(lengths, dtype=float)
    if len(lengths) == 0 or target_s <= 0:
        return None
    fast = climb_time(lengths, grades, hi, mass_kg, cda, crr, rho, efficiency)["time_s"]
    slow = climb_time(lengths, grades, lo, mass_kg, cda, crr, rho, efficiency)["time_s"]
    if not (fast <= target_s <= slow):
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if climb_time(lengths, grades, mid, mass_kg, cda, crr, rho, efficiency)["time_s"] > target_s:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
