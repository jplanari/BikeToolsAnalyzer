"""
Rides simulated with KNOWN physics, and the measurement errors a real ride carries, for checking how far the CdA
estimate drifts from the truth. The truth ride integrates speed from power, mass, grade, Crr, CdA and wind; the
observed ride is what the app would be given: the same ride seen through a power meter, a GPS, a barometer and a
weather model that are each a little wrong.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.physics import aerodyn as ae

G = ae.G
TRUE = dict(cda=0.30, rider_mass=70.0, bike_mass=8.0, crr=0.005, rho=1.225, loss=0.03)


@dataclass
class Errors:
    """Errors of the inputs. All zero = a perfect measurement."""
    power_gain: float = 0.0        # relative: +0.02 = the meter reads 2 % high
    power_offset_w: float = 0.0    # zero offset of the meter, W
    crr_rel: float = 0.0           # relative error of the Crr the fit assumes
    mass_kg: float = 0.0           # error of the rider + kit mass the fit assumes
    loss_abs: float = 0.0          # error of the assumed drivetrain loss (0.01 = one point of efficiency)
    rho_rel: float = 0.0           # relative error of the air density the fit assumes
    grade_bias_pct: float = 0.0    # altitude/gradient bias, percentage points
    grade_noise_pct: float = 0.0   # white noise on the gradient, percentage points
    speed_noise_ms: float = 0.0    # white noise on the speed, m/s
    wind_scale: float = 0.0        # relative error of the model wind speed (+0.3 = 30 % too strong)
    wind_add_ms: float = 0.0       # additive error of the model wind speed, m/s
    wind_dir_deg: float = 0.0      # error of the model wind direction, degrees
    wind_missing: bool = False     # no wind data at all: the fit assumes still air


# Typical sizes of each error (one standard deviation). ASSUMPTIONS: replace with the specification of the devices and
# weather source actually used (a power meter's stated accuracy, a barometer's drift, the weather model's wind error).
DEFAULT_SD = dict(power_gain=0.015, power_offset_w=3.0, crr_rel=0.15, mass_kg=1.0, loss_abs=0.01, rho_rel=0.015,
                  grade_bias_pct=0.2, grade_noise_pct=0.3, speed_noise_ms=0.1, wind_scale=0.25, wind_add_ms=1.0,
                  wind_dir_deg=30.0)

NOISE_FIELDS = ('grade_noise_pct', 'speed_noise_ms')      # sizes of a noise, never negative

# A step of each error for the one-at-a-time table: (field, +step, label)
OAT_STEPS = [
    ('power_gain', 0.02, "power meter reads 2 % high"),
    ('power_offset_w', 5.0, "power meter zero offset +5 W"),
    ('crr_rel', 0.20, "assumed Crr 20 % too high"),
    ('mass_kg', 2.0, "assumed mass 2 kg too high"),
    ('loss_abs', 0.02, "assumed drivetrain loss 2 points too high"),
    ('rho_rel', 0.03, "assumed air density 3 % too high"),
    ('grade_bias_pct', 0.3, "gradient biased +0.3 points"),
    ('grade_noise_pct', 0.5, "gradient noise 0.5 points"),
    ('speed_noise_ms', 0.3, "speed noise 0.3 m/s"),
    ('wind_scale', 0.30, "model wind 30 % too strong"),
    ('wind_add_ms', 1.5, "model wind +1.5 m/s"),
    ('wind_dir_deg', 40.0, "model wind direction off by 40 degrees"),
    ('wind_missing', True, "no wind data (still air assumed)"),
]


def simulate(scenario, cda=TRUE['cda'], n=2400):
    """Truth of a ride: arrays speed (m/s), power (W), grade (fraction), bearing (deg), wind10 (m/s), wind_from (deg)."""
    p = {k: TRUE[k] for k in ('rider_mass', 'bike_mass', 'crr', 'rho', 'loss')}
    mass = p['rider_mass'] + p['bike_mass']
    t = np.arange(n)
    if scenario['terrain'] == 'hilly':
        grade = 0.04 * np.sin(2 * np.pi * t / 700) + 0.01 * np.sin(2 * np.pi * t / 130)
        power = 250 + 60 * np.sin(2 * np.pi * t / 300)
    else:                                                       # flat with small rollers
        grade = 0.004 * np.sin(2 * np.pi * t / 200)
        power = 230 + 40 * np.sin(2 * np.pi * t / 400)
    bearing = {'constant': np.zeros(n), 'out_and_back': np.where(t < n // 2, 0.0, 180.0),
               'loop': (360.0 * t / n) % 360.0}[scenario['route']]
    wind10, wind_from = scenario['wind10'], scenario['wind_from']
    head = wind10 * ae.WIND_HEIGHT_FACTOR * np.cos(np.radians(wind_from - bearing))
    v = np.zeros(n)
    v[0] = 8.0
    for i in range(1, n):
        th, vi = np.arctan(grade[i]), v[i - 1]
        va = vi + head[i]
        force = mass * G * (np.sin(th) + p['crr'] * np.cos(th)) + 0.5 * p['rho'] * cda * va * abs(va)
        v[i] = max(vi + (power[i] * (1 - p['loss']) / max(vi, 1) - force) / mass, 0.5)
    return dict(speed=v, power=power, grade=grade, bearing=bearing, wind10=wind10, wind_from=wind_from)


SCENARIOS = {
    'hilly, still air': dict(terrain='hilly', route='constant', wind10=0.0, wind_from=0.0),
    'hilly out-and-back, 4 m/s wind': dict(terrain='hilly', route='out_and_back', wind10=4.0, wind_from=0.0),
    'flat out-and-back, 5 m/s cross wind': dict(terrain='flat', route='out_and_back', wind10=5.0, wind_from=90.0),
    'flat loop, 4 m/s wind': dict(terrain='flat', route='loop', wind10=4.0, wind_from=45.0),
}


def observe(truth, err, rng):
    """(ride DataFrame as the app would see it, keyword arguments for estimate_cda) under the errors `err`."""
    n = len(truth['speed'])
    speed = truth['speed'] + (rng.normal(0, err.speed_noise_ms, n) if err.speed_noise_ms else 0.0)
    grade = (truth['grade'] * 100.0 + err.grade_bias_pct
             + (rng.normal(0, err.grade_noise_pct, n) if err.grade_noise_pct else 0.0))
    df = pd.DataFrame({'speed': speed, 'power': truth['power'] * (1 + err.power_gain) + err.power_offset_w,
                       'grade_smooth': grade})
    has_wind = not err.wind_missing and (truth['wind10'] > 0 or err.wind_add_ms or err.wind_scale)
    if has_wind:
        df['wind_speed'] = max(truth['wind10'] * (1 + err.wind_scale) + err.wind_add_ms, 0.0)
        df['wind_deg'] = (truth['wind_from'] + err.wind_dir_deg) % 360.0
        df['bearing'] = truth['bearing']
    kwargs = dict(rider_mass=TRUE['rider_mass'] + err.mass_kg, bike_mass=TRUE['bike_mass'],
                  crr=TRUE['crr'] * (1 + err.crr_rel), rho=TRUE['rho'] * (1 + err.rho_rel),
                  drivetrain_loss=TRUE['loss'] + err.loss_abs)
    return df, kwargs


def cda_error(truth, err, rng, true_cda=TRUE['cda']):
    """Relative error of the fitted CdA (fit / truth - 1), or None when the fit is refused."""
    df, kw = observe(truth, err, rng)
    fit = ae.estimate_cda(df, **kw)
    return None if fit is None else fit['cda'] / true_cda - 1.0


def draw(sd, rng, only=None):
    """Random errors with the given standard deviations (all of them, or only the named one)."""
    out = {}
    for k, s in sd.items():
        if only and k != only:
            continue
        v = rng.normal(0, s)
        out[k] = abs(v) if k in NOISE_FIELDS else v
    return replace(Errors(), **out)
