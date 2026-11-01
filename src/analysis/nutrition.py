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


# =====================================================================================================================
# What was eaten (logged by the rider)
# =====================================================================================================================
# Per ride the rider can log grams of carbohydrate eaten and millilitres of fluid drunk. Next to the estimate of what
# the ride burned (above) this gives the intake rate, how much of the carbohydrate burned was replaced, and a look at
# whether rides with and without enough fuel end differently. The last one is an observation across rides, not an
# experiment: heat, intensity, sleep and fitness all differ from ride to ride.

INTAKES = {
    'below': mark('Below the guide'),
    'on': mark('Within the guide'),
    'above': mark('Above the guide'),
    'none': mark('No fuel needed'),
}
MIN_FADE_RIDES = 4                           # rides in each group before the comparison says anything
FADE_MIN_MINUTES = 90                        # long enough for fuel to matter and for decoupling to be meaningful
INTAKE_KEYS = ('carb_eaten_g', 'fluid_ml')


def intake_rate(amount, moving_s):
    """Amount per hour of moving time (g/h of carbohydrate, ml/h of fluid), or None."""
    if amount is None or pd.isna(amount) or not moving_s or moving_s <= 0:
        return None
    return float(amount) / (moving_s / 3600.0)


def intake_status(rate, moving_s, intensity_factor=None):
    """Key into INTAKES: where a carbohydrate intake rate (g/h) sits against the guide for the ride, or None."""
    if rate is None or pd.isna(rate):
        return None
    low, high = target_rate(moving_s, intensity_factor)
    if high == 0:
        return 'none'
    return 'below' if rate < low else 'above' if rate > high else 'on'


def coverage(eaten_g, burned_g):
    """Share of the carbohydrate burned that was eaten (can pass 1), or None."""
    if eaten_g is None or burned_g is None or pd.isna(eaten_g) or pd.isna(burned_g) or burned_g <= 0:
        return None
    return float(eaten_g) / float(burned_g)


def ride_intake(summary, eaten_g, fluid_ml=None, weight_kg=None):
    """What a ride's log says: {'carb_rate', 'fluid_rate', 'coverage', 'status', 'net_g', 'store_used'}; every value
    None for what cannot be worked out. `summary` has moving_time_s, carb_g and optionally 'if'."""
    moving = summary.get('moving_time_s')
    rate = intake_rate(eaten_g, moving)
    burned = summary.get('carb_g')
    return {'carb_rate': rate, 'fluid_rate': intake_rate(fluid_ml, moving),
            'coverage': coverage(eaten_g, burned),
            'status': intake_status(rate, moving, summary.get('if')),
            'net_g': None if burned is None or eaten_g is None else float(burned) - float(eaten_g),
            'store_used': store_used(burned, weight_kg, eaten_g or 0.0)}


def logged_rides(stats):
    """The rides with a carbohydrate log, with what it implies: stats rows plus rate_g_h, guide_low, guide_high,
    status and coverage. Empty when nothing was logged."""
    if stats is None or stats.empty or 'carb_eaten_g' not in stats.columns:
        return pd.DataFrame()
    s = stats[stats['carb_eaten_g'].notna() & (stats['moving_s'].fillna(0) > 0)].copy()
    if s.empty:
        return s
    ifs = s['if_factor'] if 'if_factor' in s.columns else pd.Series(np.nan, index=s.index)
    s['rate_g_h'] = [intake_rate(e, m) for e, m in zip(s['carb_eaten_g'], s['moving_s'])]
    guides = [target_rate(m, i) for m, i in zip(s['moving_s'], ifs)]
    s['guide_low'], s['guide_high'] = [g[0] for g in guides], [g[1] for g in guides]
    s['status'] = [intake_status(r, m, i) for r, m, i in zip(s['rate_g_h'], s['moving_s'], ifs)]
    burned = s['carb_g'] if 'carb_g' in s.columns else pd.Series(np.nan, index=s.index)
    s['coverage'] = [coverage(e, b) for e, b in zip(s['carb_eaten_g'], burned)]
    return s


def intake_totals(stats):
    """Numbers for a period: rides that needed fuel (guide above zero), how many of them were logged and how they
    went. {'eligible', 'logged', 'rate_g_h', 'coverage', 'below', 'on', 'above'}."""
    out = {'eligible': 0, 'logged': 0, 'rate_g_h': None, 'coverage': None, 'below': 0, 'on': 0, 'above': 0}
    if stats is None or stats.empty or 'moving_s' not in stats.columns:
        return out
    ifs = stats['if_factor'] if 'if_factor' in stats.columns else pd.Series(np.nan, index=stats.index)
    out['eligible'] = int(sum(1 for m, i in zip(stats['moving_s'].fillna(0), ifs) if target_rate(m, i)[1] > 0))
    s = logged_rides(stats)
    s = s[s['guide_high'] > 0] if not s.empty else s
    if s.empty:
        return out
    out['logged'] = len(s)
    out['rate_g_h'] = float(s['carb_eaten_g'].sum() / (s['moving_s'].sum() / 3600.0))
    cov = s.dropna(subset=['carb_g'])
    out['coverage'] = coverage(cov['carb_eaten_g'].sum(), cov['carb_g'].sum()) if len(cov) else None
    for k in ('below', 'on', 'above'):
        out[k] = int((s['status'] == k).sum())
    return out


def intake_by_period(stats, freq='W'):
    """Carbohydrate burned and eaten per week ('W') or month ('M') over the rides that have a log:
    DataFrame(burned_g, eaten_g, rides) indexed by period start; empty when nothing was logged."""
    s = logged_rides(stats)
    cols = ['burned_g', 'eaten_g', 'rides']
    if s.empty:
        return pd.DataFrame(columns=cols)
    s = s.dropna(subset=['carb_g'])
    if s.empty:
        return pd.DataFrame(columns=cols)
    dates = pd.to_datetime(s['date'])
    dates = dates.dt.tz_localize(None) if getattr(dates.dt, 'tz', None) is not None else dates
    g = s.set_index(dates.dt.normalize()).assign(rides=1).resample(
        'W-MON' if freq == 'W' else 'MS', label='left', closed='left').agg(
        {'carb_g': 'sum', 'carb_eaten_g': 'sum', 'rides': 'sum'})
    g = g[g['rides'] > 0]
    return pd.DataFrame({'burned_g': g['carb_g'], 'eaten_g': g['carb_eaten_g'], 'rides': g['rides'].astype(int)})


def fade_comparison(stats, min_minutes=FADE_MIN_MINUTES):
    """Do long, steady rides with enough fuel end differently from those without? Compares the aerobic decoupling
    (power-to-HR drift between the halves; lower is steadier) of rides taken in at or above the low end of the
    guide with those below it. {'under', 'fuelled'} each {'n', 'decoupling'}, 'difference' (fuelled minus under, in
    points), 'rho' (Spearman, intake rate against decoupling), 'enough' (both groups have MIN_FADE_RIDES rides) and
    'rides' (the rows used). None when no ride qualifies."""
    s = logged_rides(stats)
    if s.empty or 'decoupling_pct' not in s.columns:
        return None
    m = ((s['moving_s'] >= min_minutes * 60) & s['decoupling_pct'].notna() & (s['guide_high'] > 0)
         & (s['vi'].fillna(99) <= 1.10))
    r = s[m]
    if r.empty:
        return None
    fuelled = r['rate_g_h'] >= r['guide_low']
    group = lambda x: {'n': int(len(x)), 'decoupling': float(x['decoupling_pct'].mean()) if len(x) else None}
    under, ok = group(r[~fuelled]), group(r[fuelled])
    enough = under['n'] >= MIN_FADE_RIDES and ok['n'] >= MIN_FADE_RIDES
    rho = float(r['rate_g_h'].corr(r['decoupling_pct'], method='spearman')) if len(r) >= 2 * MIN_FADE_RIDES else None
    if rho is not None and np.isnan(rho):
        rho = None
    diff = ok['decoupling'] - under['decoupling'] if enough else None
    return {'under': under, 'fuelled': ok, 'difference': diff, 'rho': rho, 'enough': enough, 'rides': r}
