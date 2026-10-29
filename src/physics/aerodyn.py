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


def _balance(df, rider_mass, bike_mass, crr, rho, drivetrain_loss):
    """Terms of the power balance (W) and the quantities they use. None if the ride lacks the channels."""
    if df.empty or 'speed' not in df.columns or 'power' not in df.columns:
        return None
    m = rider_mass + bike_mass
    v = df['speed'].rolling(SPEED_SMOOTH_S, center=True, min_periods=1).mean()
    slope = df['grade_smooth'] / 100.0 if 'grade_smooth' in df.columns else pd.Series(0.0, index=df.index)
    cos_t = 1.0 / np.sqrt(1.0 + slope ** 2)
    accel = pd.Series(np.gradient(v.to_numpy()) if len(v) > 1 else np.zeros(len(v)), index=df.index)
    rho_s = _density(df, df['ele'] if 'ele' in df.columns else 0.0) if rho is None else \
        pd.Series(float(rho), index=df.index)
    v_air = _air_speed(df, v)
    p_wheel = df['power'] * (1 - drivetrain_loss)
    p_roll = m * G * crr * cos_t * v
    p_grav = m * G * slope * cos_t * v
    p_accel = m * v * accel
    p_aero = p_wheel - p_accel - p_roll - p_grav          # what is left for drag
    unit_drag = 0.5 * rho_s * v_air ** 2 * v              # aero power per unit CdA
    return {'v': v, 'p_aero': p_aero, 'p_roll': p_roll, 'p_grav': p_grav, 'p_accel': p_accel,
            'unit_drag': unit_drag}


def _usable(df, b):
    """Samples that can tell drag from noise: fast enough, and not on a steep descent where unmeasured braking
    would be booked as drag (a descent is also where the fit weights, which grow like v^6, are largest)."""
    ok = b['v'] > MIN_CDA_SPEED_MS
    if 'grade_smooth' in df.columns:
        ok &= ~(df['grade_smooth'] < -MAX_DESCENT_GRADE_PCT)
    return ok


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
    return {'cda': cda, 'seconds': int(ok.sum())}


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
