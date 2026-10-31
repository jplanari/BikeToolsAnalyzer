"""
Cycling power balance and CdA estimation from a recorded ride.

At the wheel, P_pedals * (1 - drivetrain_loss) = P_aero + P_roll + P_grav + P_accel, with

    P_roll  = m g Crr v                    P_grav  = m g grade v
    P_accel = m v dv/dt                    P_aero  = 1/2 rho CdA v_air |v_air| v

v is ground speed (the power is force x ground speed) and v_air is the air speed along the road:
ground speed plus the head-wind component (negative of a tail wind). CdA is what makes the balance
close, so every error in the other terms ends up in it: speed and grade are smoothed before use
(accelerations from raw 1 Hz speed are mostly noise) and the ride's CdA is a least-squares fit over
all usable samples instead of an average of per-second ratios, which is biased once noise is clipped.
"""
import numpy as np
import pandas as pd

G = 9.80665                      # m/s²
BIKE_MASS_KG = 8.0
CRR = 0.005
DRIVETRAIN_LOSS = 0.03
MIN_CDA_SPEED_MS = 5.0           # slower than this, drag is too small to tell from noise
MAX_DESCENT_GRADE_PCT = 5.0      # descents steeper than this (grade below -5 %) are not used for CdA: braking is likely and not in the balance
SPEED_SMOOTH_S = 5               # centred window for speed before differentiating
# Open-Meteo gives wind at 10 m; at a rider's height (~1.5 m) it is weaker (boundary-layer profile,
# shelter from hedges and buildings). 0.7 is a common rule of thumb for road cycling.
WIND_HEIGHT_FACTOR = 0.7
STALL_SPEED_MS = 0.5             # below this the recorded speed is "zero" (GPS lost, e.g. in a tunnel)
MIN_STALL_S = 10                 # a frozen stretch shorter than this is not bridged
MAX_STALL_S = 900                # ... and neither is one longer than this (a pause, not a tunnel)
MIN_JUMP_M = 50.0                # the position must jump at least this far when the fix comes back
CDA_VALID = (0.10, 0.80)         # outside this a fitted CdA is a sign of bad data, not a rider

_RD, _RV = 287.058, 461.495      # gas constants of dry air and water vapour, J/(kg K)


def get_air_density(elevation_m, temp_c=None, rel_humidity=0.0):
    """
    Air density (kg/m³) at an altitude, from the standard-atmosphere pressure, the temperature and
    the humidity. Without a temperature the standard-atmosphere one for that altitude is used.
    Works on scalars and arrays. (Sea level, 15 °C, dry: 1.225.)
    """
    h = np.asarray(elevation_m, dtype=float)
    t0, lapse, p0 = 288.15, 0.0065, 101325.0
    pressure = p0 * (1.0 - lapse * h / t0) ** (G / (_RD * lapse))
    t_k = (t0 - lapse * h) if temp_c is None else np.asarray(temp_c, dtype=float) + 273.15
    rh = np.clip(np.asarray(rel_humidity, dtype=float), 0.0, 1.0)
    p_sat = 610.94 * np.exp(17.625 * (t_k - 273.15) / (t_k - 273.15 + 243.04))      # Magnus, Pa
    p_vap = rh * p_sat
    rho = (pressure - p_vap) / (_RD * t_k) + p_vap / (_RV * t_k)
    return float(rho) if rho.ndim == 0 else rho


def _density(df, ele):
    """Per-sample air density from the ride's altitude, temperature and humidity (where known)."""
    ele = pd.Series(ele, index=df.index).ffill().bfill().fillna(0.0)
    temp = df['temp_c'] if 'temp_c' in df.columns and df['temp_c'].notna().any() else None
    hum = df['relative_humidity'].fillna(0.0) if 'relative_humidity' in df.columns else 0.0
    rho = get_air_density(ele.to_numpy(), None if temp is None else temp.ffill().bfill().to_numpy(),
                          hum.to_numpy() if hasattr(hum, 'to_numpy') else hum)
    return pd.Series(rho, index=df.index)


def _wind_components(df):
    """(head, cross) wind at rider height, m/s: head > 0 when the wind blows from ahead, cross is its size
    across the road. None when the ride has no wind data."""
    if not {'wind_speed', 'wind_deg', 'bearing'} <= set(df.columns):
        return None
    wind = df['wind_speed'].fillna(0) * WIND_HEIGHT_FACTOR
    # wind_deg is where the wind blows FROM: a wind from straight ahead has cos = 1, a head wind
    rad = np.radians(df['wind_deg'].fillna(0) - df['bearing'].fillna(0))
    return wind * np.cos(rad), wind * np.sin(rad)


def _air_speed(df, v):
    """Speed of the air relative to the rider ALONG the road (m/s, >= 0): what drags on the rider."""
    w = _wind_components(df)
    return v.copy() if w is None else (v + w[0]).abs()


def apparent_air_speed(df, v=None):
    """
    Magnitude of the air velocity relative to the rider (m/s), including the cross wind. This is what
    cools the skin (and, with no speed at all, is simply the wind), whereas drag only feels the part
    along the road. v defaults to the ride's smoothed ground speed.
    """
    if v is None:
        v = df['speed'].fillna(0).rolling(SPEED_SMOOTH_S, center=True, min_periods=1).mean() \
            if 'speed' in df.columns else pd.Series(0.0, index=df.index)
    w = _wind_components(df)
    return v.copy() if w is None else np.sqrt((v + w[0]) ** 2 + w[1] ** 2)


def find_gps_stalls(df):
    """
    Stretches where the rider was pedalling but the recorded speed and position were frozen, then jumped when
    the position came back (a tunnel, an underpass, a GPS dropout): speed is zero there, so the drag cannot be
    read from it. Returns [(i0, i1, v_avg, grade_pct)] with positional indexes: i0 is the last sample with a
    position, i1 the first sample after the stretch (where the position jumps), v_avg the average speed over
    i0..i1, (position at i1 - position at i0) / elapsed time, and grade_pct the average gradient of the stretch.
    Only stretches whose average speed is high enough to tell drag from noise are returned.
    """
    need = {'speed', 'power', 'dist_m', 'time'}
    if df.empty or not need <= set(df.columns):
        return []
    n = len(df)
    speed = df['speed'].fillna(0).to_numpy(dtype=float)
    power = df['power'].fillna(0).to_numpy(dtype=float)
    cad = df['cadence'].fillna(0).to_numpy(dtype=float) if 'cadence' in df.columns else power
    dist = pd.to_numeric(df['dist_m'], errors='coerce').to_numpy(dtype=float)
    t = (df['time'] - df['time'].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    ele = pd.to_numeric(df['ele'], errors='coerce').to_numpy(dtype=float) if 'ele' in df.columns else None

    moved = np.r_[np.inf, np.abs(np.diff(dist))]                # distance since the previous sample
    frozen = (speed < STALL_SPEED_MS) & (moved < 1.0)
    pedalling = (power > 0) & (cad > 0)

    out, k = [], 1
    while k < n:
        if not frozen[k]:
            k += 1
            continue
        a = b = k
        while b + 1 < n and (frozen[b + 1] or (b + 3 < n and frozen[b + 2:b + 4].any() and not (moved[b + 1] >= 1.0))):
            b += 1
        k = b + 1
        i0, i1 = a - 1, b + 1
        if i1 >= n or not (MIN_STALL_S <= t[b] - t[a] + 1 <= MAX_STALL_S):
            continue
        if pedalling[a:b + 1].mean() < 0.8:
            continue
        jump, dt = dist[i1] - dist[i0], t[i1] - t[i0]
        if not np.isfinite(jump) or jump < MIN_JUMP_M or dt <= 0 or jump / dt < MIN_CDA_SPEED_MS:
            continue
        grade = 0.0
        if ele is not None and np.isfinite(ele[i0]) and np.isfinite(ele[i1]):
            grade = float(np.clip((ele[i1] - ele[i0]) / jump * 100.0, -20.0, 20.0))
        out.append((i0, i1, float(jump / dt), grade))
    return out


def _balance(df, rider_mass, bike_mass, crr, rho, drivetrain_loss):
    """Terms of the power balance (W) and the quantities they use. None if the ride lacks the channels."""
    if df.empty or 'speed' not in df.columns or 'power' not in df.columns:
        return None
    m = rider_mass + bike_mass
    speed = df['speed'].copy()
    slope = (df['grade_smooth'] / 100.0 if 'grade_smooth' in df.columns
             else pd.Series(0.0, index=df.index)).copy()
    bridged = pd.Series(False, index=df.index)
    for i0, i1, v_avg, grade_pct in find_gps_stalls(df):       # tunnel: the average speed of the stretch stands in
        speed.iloc[i0 + 1:i1 + 1] = v_avg
        slope.iloc[i0 + 1:i1 + 1] = grade_pct / 100.0
        bridged.iloc[i0 + 1:i1 + 1] = True
    v = speed.rolling(SPEED_SMOOTH_S, center=True, min_periods=1).mean()
    cos_t = 1.0 / np.sqrt(1.0 + slope ** 2)
    accel = pd.Series(np.gradient(v.to_numpy()) if len(v) > 1 else np.zeros(len(v)), index=df.index)
    accel[bridged] = 0.0                                        # constant average speed over the stretch
    rho_s = _density(df, df['ele'] if 'ele' in df.columns else 0.0) if rho is None else \
        pd.Series(float(rho), index=df.index)
    v_air = _air_speed(df, v)
    p_wheel = df['power'] * (1 - drivetrain_loss)
    p_roll = m * G * crr * cos_t * v
    p_grav = m * G * slope * cos_t * v
    p_accel = m * v * accel
    p_aero = p_wheel - p_accel - p_roll - p_grav          # what is left for drag
    unit_drag = 0.5 * rho_s * v_air ** 2 * v              # aero power per unit CdA
    return {'v': v, 'slope': slope, 'bridged': bridged, 'p_aero': p_aero, 'p_roll': p_roll, 'p_grav': p_grav, 'p_accel': p_accel,
            'unit_drag': unit_drag}


def _usable(df, b):
    """Samples that can tell drag from noise: fast enough, and not on a steep descent where unmeasured braking
    would be booked as drag (a descent is also where the fit weights, which grow like v^6, are largest)."""
    return (b['v'] > MIN_CDA_SPEED_MS) & ~(b['slope'] * 100.0 < -MAX_DESCENT_GRADE_PCT)


def calculate_CdA(df, rider_mass, bike_mass=BIKE_MASS_KG, crr=CRR, rho=None, drivetrain_loss=DRIVETRAIN_LOSS):
    """
    (cda_series, p_aero_series): per-sample CdA (m², NaN where it can't be told: slower than 5 m/s,
    negative or implausible) for plotting, and the aerodynamic power (W) that closes the balance.
    rho=None uses the ride's altitude, temperature and humidity.
    """
    b = _balance(df, rider_mass, bike_mass, crr, rho, drivetrain_loss)
    if b is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    cda = (b['p_aero'] / b['unit_drag']).where(_usable(df, b))
    cda = cda.replace([np.inf, -np.inf], np.nan)
    cda = cda.mask((cda < 0) | (cda > 1.0))
    return cda, b['p_aero']


def estimate_cda(df, rider_mass, bike_mass=BIKE_MASS_KG, crr=CRR, rho=None, drivetrain_loss=DRIVETRAIN_LOSS):
    """
    The ride's CdA as the least-squares fit of P_aero = CdA x (1/2 rho v_air² v) over all samples
    faster than 5 m/s and not on a steep descent (grade below -5 %): sum(P k) / sum(k²), so fast samples (a lot of drag signal) count
    for more than slow ones, and noise cancels instead of being clipped. Returns {'cda', 'seconds'} or None when
    there are too few samples or the result is implausible.

    Where the speed was lost while pedalling (a tunnel: speed zero, then the position jumps) the stretch is
    included at its average speed, (position at the end - position at the start) / time; 'tunnel_seconds' says
    how many of the seconds used come from such stretches.
    """
    b = _balance(df, rider_mass, bike_mass, crr, rho, drivetrain_loss)
    if b is None:
        return None
    k = b['unit_drag']
    ok = _usable(df, b) & k.notna() & b['p_aero'].notna() & (k > 0)
    if ok.sum() < 120:
        return None
    k, p = k[ok], b['p_aero'][ok]
    cda = float((p * k).sum() / (k * k).sum())
    if not CDA_VALID[0] <= cda <= CDA_VALID[1]:
        return None
    return {'cda': cda, 'seconds': int(ok.sum()), 'tunnel_seconds': int((ok & b['bridged']).sum())}


def get_avg_cda(cda_series):
    """Mean of the per-sample series (kept for compatibility; estimate_cda is the better number)."""
    if cda_series is None or len(cda_series) == 0:
        return np.nan
    return cda_series.mean()


def get_avg_power(p_aero_series):
    if p_aero_series.empty:
        return np.nan
    return p_aero_series.mean()


def calculate_power_components(df, rider_mass, bike_mass=BIKE_MASS_KG, crr=CRR, rho=None,
                               drivetrain_loss=DRIVETRAIN_LOSS):
    """(p_grav, p_roll, p_accel) in W, on the same basis as calculate_CdA."""
    b = _balance(df, rider_mass, bike_mass, crr, rho, drivetrain_loss)
    if b is None:
        return pd.Series(dtype=float)
    return b['p_grav'], b['p_roll'], b['p_accel']
