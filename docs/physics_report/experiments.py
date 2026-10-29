import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import tests.conftest  # noqa: stubs for streamlit etc. when the app dependencies are not installed
import numpy as np, pandas as pd
from src.physics import aerodyn as ae, thermal as th

OUT = HERE
R = {}
rng = np.random.default_rng(42)

# ---------------- CdA synthetic ride ----------------
N = 3600
t = np.arange(N, dtype=float)
TRUE_CDA, MASS, BIKE, CRR, LOSS = 0.30, 72.0, 8.0, 0.005, 0.03
v_true = 9.5 + 2.5*np.sin(2*np.pi*t/240) + 1.2*np.sin(2*np.pi*t/71 + 1.0)
grade_true = 2.5*np.sin(2*np.pi*t/700) + 1.0*np.sin(2*np.pi*t/190)      # percent
def base_df(v, grade, wind=4.0, wdeg=30.0):
    return pd.DataFrame({'time': pd.date_range('2026-07-01 09:00', periods=N, freq='1s'),
        'speed': v, 'grade_smooth': grade, 'ele': 200.0, 'temp_c': 18.0, 'relative_humidity': 0.5,
        'wind_speed': wind, 'wind_deg': wdeg, 'bearing': 90.0, 'power': 0.0})
def forward_power(df, cda=TRUE_CDA, mass=MASS, crr=CRR):
    m = mass + BIKE
    v = df['speed'].to_numpy(); s = df['grade_smooth'].to_numpy()/100; cos = 1/np.sqrt(1+s**2)
    a = np.gradient(v)
    rho = ae.get_air_density(df['ele'].to_numpy(), df['temp_c'].to_numpy(), df['relative_humidity'].to_numpy())
    head = df['wind_speed'].to_numpy()*ae.WIND_HEIGHT_FACTOR*np.cos(np.radians(df['wind_deg']-df['bearing']))
    va = v + head
    p = m*ae.G*(crr*cos + s*cos)*v + m*v*a + 0.5*rho*cda*np.abs(va)*va*v
    return p/(1-LOSS)
df0 = base_df(v_true, grade_true)
df0['power'] = forward_power(df0)
R['synthetic_mean_power'] = float(df0.power.mean())
R['synthetic_mean_speed'] = float(df0.speed.mean())

def ar1(n, sigma, tau):
    a = np.exp(-1/tau); e = rng.normal(0, sigma*np.sqrt(1-a*a), n); x = np.zeros(n)
    for i in range(1, n): x[i] = a*x[i-1] + e[i]
    return x

est_ls, est_mean = [], []
for _ in range(300):
    d = df0.copy()
    d['power'] = df0.power + rng.normal(0, 8, N)                 # power-meter noise, 8 W (1 sigma)
    d['speed'] = df0.speed + rng.normal(0, 0.15, N)              # speed-sensor / GPS noise
    d['grade_smooth'] = df0.grade_smooth + ar1(N, 0.25, 30)      # smooth grade error, 0.25 %
    e = ae.estimate_cda(d, MASS)
    est_ls.append(np.nan if e is None else e['cda'])
    cs, _ = ae.calculate_CdA(d, MASS)
    est_mean.append(ae.get_avg_cda(cs))
est_ls, est_mean = np.array(est_ls), np.array(est_mean)
R['mc'] = {'true': TRUE_CDA, 'ls_mean': float(np.nanmean(est_ls)), 'ls_sd': float(np.nanstd(est_ls)),
           'mean_mean': float(np.nanmean(est_mean)), 'mean_sd': float(np.nanstd(est_mean)),
           'ls_failed': int(np.isnan(est_ls).sum())}
np.save(f'{OUT}/mc_ls.npy', est_ls); np.save(f'{OUT}/mc_mean.npy', est_mean)

# noise-free recovery
e0 = ae.estimate_cda(df0, MASS)
R['noisefree_cda'] = e0['cda']

# systematic error sources (noise free, perturb the estimator inputs / data)
def est(d=None, **kw):
    d = df0 if d is None else d
    kw.setdefault('rider_mass', MASS)
    return ae.estimate_cda(d, **kw)['cda']
rows = []
def add(label, val): rows.append((label, val - TRUE_CDA))
add('Rider+kit mass +2 kg', est(rider_mass=MASS+2))
add('Rolling resistance Crr +0.001', est(crr=CRR+0.001))
add('Drivetrain loss +2 %-points', est(drivetrain_loss=LOSS+0.02))
d = df0.copy(); d['power'] *= 1.02; add('Power-meter bias +2 %', est(d))
d = df0.copy(); d['grade_smooth'] += 0.1; add('Grade bias +0.1 %-points', est(d))
d = df0.copy(); d['wind_speed'] += 1.0; add('10 m wind speed +1 m/s', est(d))
d = df0.copy(); d['wind_deg'] += 30; add('Wind direction +30°', est(d))
d = df0.copy(); d['speed'] *= 1.01; add('Speed sensor +1 % (wheel circumference)', est(d))
rho_t = float(ae.get_air_density(200, 18, 0.5)); add('Air density +3 %', est(rho=rho_t*1.03))
d = df0.copy(); d['temp_c'] += 5; add('Temperature +5 K (density only)', est(d))
R['sys'] = rows
# no-wind-data case
d = df0.drop(columns=['wind_speed','wind_deg','bearing']); R['no_wind_input'] = est(d) - TRUE_CDA


# ---------------- Braking on steep descents ----------------
# Dynamics integrated physically (1 s Euler, no wind): power 250 +- 60 W, +-7 % grades. On descents steeper than 5 %
# the rider stops pedalling and brakes with a constant 40 N. The brake dissipation is not in the power
# balance, so without the filter it is booked as drag.
gb = 7.0*np.sin(2*np.pi*t/700)
pw = 250 + 60*np.sin(2*np.pi*t/300)
mt = MASS + BIKE; rho_b = float(ae.get_air_density(200, 18, 0.5))
vb = np.zeros(N); vb[0] = 8.0; powb = pw.copy()
for i in range(1, N):
    ang = np.arctan(gb[i]/100); vi = vb[i-1]; steep = gb[i] < -ae.MAX_DESCENT_GRADE_PCT
    p = 0.0 if steep else pw[i]; powb[i] = p
    force = mt*ae.G*(np.sin(ang) + CRR*np.cos(ang)) + 0.5*rho_b*TRUE_CDA*vi*vi + (40.0 if steep else 0.0)
    vn = max(vi + (p*(1-LOSS)/max(vi, 1) - force)/mt, 0.5)
    vb[i] = vn
dfb = base_df(vb, gb, wind=0.0).drop(columns=['wind_speed','wind_deg','bearing']); dfb['power'] = powb
steep_b = dfb['grade_smooth'] < -ae.MAX_DESCENT_GRADE_PCT
R['brake_steep_seconds'] = int(steep_b.sum()); R['brake_mean_speed_steep'] = float(dfb.speed[steep_b].mean())
R['brake_filtered'] = ae.estimate_cda(dfb, MASS)['cda'] - TRUE_CDA
_keep = ae.MAX_DESCENT_GRADE_PCT; ae.MAX_DESCENT_GRADE_PCT = 1e9           # filter switched off
R['brake_unfiltered'] = ae.estimate_cda(dfb, MASS)['cda'] - TRUE_CDA
ae.MAX_DESCENT_GRADE_PCT = _keep
_b = ae._balance(dfb, MASS, ae.BIKE_MASS_KG, CRR, None, LOSS)
_k = _b['unit_drag']; _ok = steep_b & (dfb.speed > 5)
R['brake_weight_share_steep'] = float((_k[_ok]**2).sum() / (_k[dfb.speed > 5]**2).sum())
R['brake_mean_residual_w'] = float((_b['p_aero'][_ok] - TRUE_CDA*_k[_ok]).mean())

# steady flat ride: Crr/CdA collinearity
dfl = base_df(np.full(N, 10.0), np.zeros(N), wind=0.0)
dfl['power'] = forward_power(dfl)
R['flat_crr_bias_per_0p001'] = ae.estimate_cda(dfl, MASS, crr=CRR+0.001)['cda'] - TRUE_CDA
R['hill_crr_bias_per_0p001'] = est(crr=CRR+0.001) - TRUE_CDA

# analytical std of the LS estimator: sigma / sqrt(sum k^2)
b = ae._balance(df0, MASS, BIKE, CRR, None, LOSS)
k = b['unit_drag']; ok = b['v'] > 5
R['k_sum_sq'] = float((k[ok]**2).sum()); R['analytic_sd_8W'] = float(8/np.sqrt((k[ok]**2).sum()))
R['k_mean'] = float(k[ok].mean())

# ---------------- Thermal ----------------
def ride(power, temp, rh=0.5, speed=8.0, n=3600, clothes=('Summer Jersey',), mass=70, height=175, wind=None):
    d = pd.DataFrame({'time': pd.date_range('2026-07-01 09:00', periods=n, freq='1s'), 'speed': speed,
        'power': power, 'temp_c': temp, 'relative_humidity': rh})
    return th.calculate_thermal_profile(d, mass, height, list(clothes))
R['bsa_70_175'] = th.du_bois_BSA(175, 70)
# rest equilibrium
rest = {}
for T in (5, 10, 20, 30):
    o = ride(0, T, speed=0, clothes=('Summer Jersey',) if T >= 20 else ('Summer Jersey','Long Sleeve Jersey'))
    rest[T] = (float(o.core_temp.iloc[0]), float(o.core_temp.iloc[-1]), float(o.skin_temp.iloc[-1]))
R['rest'] = rest
# time courses
curves = {}
for T in (10, 20, 30, 35):
    o = ride(200, T, rh=0.5, speed=8.0); curves[T] = o.core_temp.to_numpy()[::10].tolist()
R['curves'] = curves
scen = {}
for lab, kw in {'200W,10C':dict(power=200,temp=10,clothes=('Summer Jersey','Long Sleeve Jersey','Gilet (Wind Vest)')),
                '200W,20C':dict(power=200,temp=20),'200W,30C':dict(power=200,temp=30),
                '300W,25C':dict(power=300,temp=25),'150W,20C':dict(power=150,temp=20)}.items():
    o = ride(**kw); scen[lab] = [float(o.core_temp.iloc[-1]), float(o.skin_temp.iloc[-1]),
                                  float(o.core_temp.max()), float(o.heat_generated_W.mean()), float(o.heat_lost_W.mean())]
R['scen'] = scen
# humidity sweep
hum = {}
for T in (24, 28, 32):
    hum[T] = [float(ride(220, T, rh=rh).core_temp.iloc[-1]) for rh in (0.2,0.35,0.5,0.65,0.8,0.95)]
R['hum'] = hum
# sensitivity (+-20 %) on two scenarios
def sens(scn):
    base = ride(**scn).core_temp.iloc[-1]; out = {}
    for name in ['EFFICIENCY','M_REST_W_PER_M2','K_MIN_W_PER_M2K','H_CONV_COEFF','EVAP_COEFF_W_PER_M2',
                 'SWEAT_GAIN_W_PER_K','K_VASO_MAX_W_PER_M2K']:
        v0 = getattr(th, name); r = []
        for f in (0.8, 1.2):
            setattr(th, name, v0*f); r.append(float(ride(**scn).core_temp.iloc[-1] - base))
        setattr(th, name, v0); out[name] = r
    # clothing +-20 % via catalog
    c0 = dict(th.CLOTHING_CATALOG); r = []
    for f in (0.8, 1.2):
        for kk in th.CLOTHING_CATALOG: th.CLOTHING_CATALOG[kk] = c0[kk]*f
        r.append(float(ride(**scn).core_temp.iloc[-1] - base))
    th.CLOTHING_CATALOG.update(c0); out['CLOTHING'] = r
    return float(base), out
R['sens_warm'] = sens(dict(power=220, temp=28, rh=0.5))
R['sens_cool'] = sens(dict(power=180, temp=12, rh=0.7, clothes=('Summer Jersey','Long Sleeve Jersey')))
# convective coefficient comparison
R['h'] = {str(v): [th.H_CONV_COEFF*v**th.H_CONV_EXP, 8.3*v**0.6] for v in (1,2,4,5,10,15)}
# Du Bois examples
R['bsa'] = {f'{h}/{w}': th.du_bois_BSA(h, w) for h, w in ((165,55),(175,70),(185,85))}
# wind height factor
R['wind_factor'] = {a: (1.5/10)**a for a in (1/7, 0.2, 0.25)}
# air density table
R['rho'] = {f'{h},{T},{rh}': float(ae.get_air_density(h, T, rh)) for h, T, rh in
            ((0,15,0),(0,15,0.5),(0,30,0.8),(1000,20,0.5),(2000,10,0.5))}
# time constants at rest/exercise
A = R['bsa_70_175']; C_skin = 70*th.SKIN_MASS_FRACTION*th.C_SKIN
R['tau_skin_max_s'] = C_skin/(th.K_MIN_W_PER_M2K*A)
Kmax = (th.K_MIN_W_PER_M2K + th.K_VASO_MAX_W_PER_M2K)*A
R['tau_skin_min_s'] = C_skin/(Kmax + A/(0.155*0.15+1/(th.H_CONV_COEFF*10**th.H_CONV_EXP)))
json.dump(R, open(f'{OUT}/results.json','w'), indent=1, default=float)
print(json.dumps({k:R[k] for k in ('mc','noisefree_cda','sys','no_wind_input','flat_crr_bias_per_0p001','hill_crr_bias_per_0p001','analytic_sd_8W','rest','scen','sens_warm','sens_cool','h','tau_skin_max_s','tau_skin_min_s','bsa','wind_factor','rho')}, indent=1, default=float))
