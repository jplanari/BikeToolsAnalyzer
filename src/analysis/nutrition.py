"""
Energy and fuelling estimates from power. No food is logged: everything follows from the work done on the pedals.

Energy: a rider turns about 24 % of the energy they burn into work at the pedals. One kJ of work is therefore
1 / (0.24 x 4.184) = 0.996 kcal burned, i.e. kcal is almost exactly kJ (the usual rule of thumb). Only pedalling
counts: the body's resting energy while coasting or stopped is not added.

Fuel mix: the share of that energy coming from carbohydrate rather than fat rises with intensity. CARB_SHARE below
is a crossover curve (share of energy from carbohydrate against power as a fraction of FTP) from typical lab
values for trained riders: roughly a third at easy endurance pace, about 70 % at tempo and nearly all of it at
threshold and above. It is a population average: individuals differ (a very fat-adapted rider burns less
carbohydrate, a poorly trained one more), so the numbers are estimates, not measurements.

Fuelling: carbohydrate guidance in grams per hour by ride length (sports-nutrition consensus: nothing needed under
about 45 min, up to 30 g/h to 1 h, 30-60 g/h to 2 h, then 60-90 g/h on long rides), shifted down for very easy
rides and up for hard ones. A glycogen store of about 6 g per kg of body weight is assumed, and the app says how
much of it the ride would use if nothing were eaten. This is guidance, not medical or dietary advice.
"""
import numpy as np
import pandas as pd

from src.i18n import mark

GROSS_EFFICIENCY = 0.24                      # mechanical work / energy burned
KCAL_PER_KJ = 1.0 / (GROSS_EFFICIENCY * 4.184)
CARB_KCAL_PER_G = 4.0
FAT_KCAL_PER_G = 9.0
GLYCOGEN_G_PER_KG = 6.0                      # muscle + liver store of a trained rider
SMOOTH_S = 30                                # the body follows the average of the last half minute, not each second

# (power as a fraction of FTP, share of the energy that comes from carbohydrate)
CARB_SHARE = [(0.0, 0.25), (0.40, 0.30), (0.55, 0.40), (0.70, 0.55), (0.85, 0.72), (1.00, 0.88), (1.20, 1.00)]

# (ride hours below which the band applies, (low, high) g/h)
FUEL_BANDS = [(0.75, (0, 0)), (1.0, (0, 30)), (2.0, (30, 60)), (3.0, (45, 60)), (float('inf'), (60, 90))]
EASY_IF, HARD_IF = 0.60, 0.85                # intensity factor limits that move the band

# How big the carbohydrate bill of a ride is next to a full store (a rule of thumb)
LOADS = {
    'light': mark('Light'),
    'moderate': mark('Moderate'),
    'heavy': mark('Heavy'),
    'extreme': mark('Very heavy'),
}
LOAD_LIMITS = [(0.25, 'light'), (0.5, 'moderate'), (0.75, 'heavy'), (float('inf'), 'extreme')]

RIDE_KEYS = ('carb_g',)


def carb_share(fraction_of_ftp):
    """Share (0-1) of the energy that comes from carbohydrate at a power given as a fraction of FTP."""
    xs, ys = zip(*CARB_SHARE)
    return np.interp(fraction_of_ftp, xs, ys)


def kcal_from_kj(kj):
    return None if kj is None or pd.isna(kj) else float(kj) * KCAL_PER_KJ


def ride_energy(power, ftp):
    """Energy burned in a ride sampled at 1 Hz: {'kj', 'kcal', 'carb_g', 'fat_g', 'carb_share'}, or None without
    power or FTP. Carbohydrate is worked out second by second from the 30-second average power."""
    if power is None or not ftp or ftp <= 0:
        return None
    p = pd.to_numeric(pd.Series(power), errors='coerce').fillna(0).clip(lower=0)
    if p.sum() <= 0:
        return None
    smooth = p.rolling(SMOOTH_S, min_periods=1, center=True).mean()
    kcal_s = p / 1000.0 * KCAL_PER_KJ
    carb_kcal = float((kcal_s * carb_share(smooth / ftp)).sum())
    kcal = float(kcal_s.sum())
    return {'kj': float(p.sum() / 1000.0), 'kcal': kcal, 'carb_g': carb_kcal / CARB_KCAL_PER_G,
            'fat_g': (kcal - carb_kcal) / FAT_KCAL_PER_G, 'carb_share': carb_kcal / kcal}


def ride_metrics(power, ftp):
    """The per-ride number kept in the database (RIDE_KEYS); None when it cannot be estimated."""
    e = ride_energy(power, ftp)
    return {'carb_g': e['carb_g'] if e else None}


def fat_grams(kj, carb_g):
    """Grams of fat burned for a ride's work and carbohydrate, or None."""
    kcal = kcal_from_kj(kj)
    if kcal is None or carb_g is None or pd.isna(carb_g):
        return None
    return max(0.0, (kcal - carb_g * CARB_KCAL_PER_G) / FAT_KCAL_PER_G)


# --- what to eat --------------------------------------------------------------------------------------------------

def target_rate(moving_s, intensity_factor=None):
    """(low, high) grams of carbohydrate per hour worth taking in on a ride of this length and intensity."""
    hours = (moving_s or 0) / 3600.0
    low, high = next(band for limit, band in FUEL_BANDS if hours < limit)
    if high and intensity_factor is not None and not pd.isna(intensity_factor):
        if intensity_factor < EASY_IF:
            low, high = low / 2.0, high / 2.0
        elif intensity_factor >= HARD_IF:
            low = (low + high) / 2.0
    return float(low), float(high)


def target_total(moving_s, intensity_factor=None):
    """(low, high) grams of carbohydrate over the whole ride."""
    low, high = target_rate(moving_s, intensity_factor)
    hours = (moving_s or 0) / 3600.0
    return low * hours, high * hours


def glycogen_store(weight_kg):
    """Estimated glycogen in the body at the start of a fed ride (g)."""
    return float(weight_kg) * GLYCOGEN_G_PER_KG if weight_kg else None


def store_used(carb_g, weight_kg, eaten_g=0.0):
    """Fraction of a full glycogen store a ride would use, after the carbohydrate eaten during it; None if unknown."""
    store = glycogen_store(weight_kg)
    if carb_g is None or pd.isna(carb_g) or not store:
        return None
    return max(0.0, float(carb_g) - (eaten_g or 0.0)) / store


def load(fraction):
    """Key into LOADS for a fraction of the store, or None."""
    if fraction is None or pd.isna(fraction):
        return None
    return next(key for limit, key in LOAD_LIMITS if fraction < limit)


# --- a period, from stored ride rows --------------------------------------------------------------------------------

def by_period(stats, freq='W'):
    """Energy per week ('W') or month ('M') from get_ride_stats rows:
    DataFrame(kcal, carb_g, fat_g, carb_pct, hours, rides) indexed by period start; periods without a ride are
    kept as zeros. Rides with no power (no kJ) are left out; carbohydrate counts only the rides that have it."""
    cols = ['kcal', 'carb_g', 'fat_g', 'carb_pct', 'hours', 'rides']
    if stats is None or stats.empty or 'kj' not in stats.columns:
        return pd.DataFrame(columns=cols)
    s = stats.dropna(subset=['kj', 'date']).copy()
    s = s[s['kj'] > 0]
    if s.empty:
        return pd.DataFrame(columns=cols)
    s['kcal'] = s['kj'] * KCAL_PER_KJ
    has = s['carb_g'].notna() if 'carb_g' in s.columns else pd.Series(False, index=s.index)
    s['carb_g_ok'] = s['carb_g'].where(has, 0.0) if 'carb_g' in s.columns else 0.0
    s['kcal_c'] = s['kcal'].where(has, 0.0)
    s['hours'] = (s['moving_s'].fillna(s['duration_s']) if 'moving_s' in s.columns else 0) / 3600.0
    s['rides'] = 1
    rule = 'W-MON' if freq == 'W' else 'MS'
    dates = pd.to_datetime(s['date']).dt.tz_localize(None) if getattr(s['date'].dt, 'tz', None) is not None \
        else pd.to_datetime(s['date'])
    s = s.set_index(dates.dt.normalize())
    g = s.resample(rule, label='left', closed='left').agg(
        {'kcal': 'sum', 'carb_g_ok': 'sum', 'kcal_c': 'sum', 'hours': 'sum', 'rides': 'sum'})
    out = pd.DataFrame(index=g.index)
    out['kcal'] = g['kcal']
    out['carb_g'] = g['carb_g_ok']
    out['fat_g'] = ((g['kcal_c'] - g['carb_g_ok'] * CARB_KCAL_PER_G) / FAT_KCAL_PER_G).clip(lower=0)
    out['carb_pct'] = np.where(g['kcal_c'] > 0, g['carb_g_ok'] * CARB_KCAL_PER_G / g['kcal_c'] * 100, np.nan)
    out['hours'] = g['hours']
    out['rides'] = g['rides'].astype(int)
    return out[cols]


def totals(stats):
    """Whole-period numbers: {'kcal', 'carb_g', 'carb_pct', 'rides', 'hours'} (None for what is unknown)."""
    p = by_period(stats, 'M') if stats is not None else pd.DataFrame()
    if p.empty:
        return {'kcal': None, 'carb_g': None, 'carb_pct': None, 'rides': 0, 'hours': 0.0}
    kcal_c = (p['carb_g'] * CARB_KCAL_PER_G + p['fat_g'] * FAT_KCAL_PER_G).sum()
    carb = float(p['carb_g'].sum())
    return {'kcal': float(p['kcal'].sum()), 'carb_g': carb if carb > 0 else None,
            'carb_pct': float(p['carb_g'].sum() * CARB_KCAL_PER_G / kcal_c * 100) if kcal_c > 0 else None,
            'rides': int(p['rides'].sum()), 'hours': float(p['hours'].sum())}
