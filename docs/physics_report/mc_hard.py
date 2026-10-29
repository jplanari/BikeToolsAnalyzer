import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import tests.conftest
import numpy as np, pandas as pd
from src.physics import aerodyn as ae
OUT = HERE
rng=np.random.default_rng(7)
N=3600; t=np.arange(N,dtype=float); MASS,BIKE,CRR,LOSS,CDA=72.,8.,0.005,0.03,0.30
v=6.5+1.8*np.sin(2*np.pi*t/200)+0.8*np.sin(2*np.pi*t/53)      # slow, hilly ride: many samples 4-8 m/s
g=3.0*np.sin(2*np.pi*t/500)
df=pd.DataFrame({'time':pd.date_range('2026-07-01',periods=N,freq='1s'),'speed':v,'grade_smooth':g,'ele':300.,'temp_c':15.,'relative_humidity':.5,'power':0.})
m=MASS+BIKE; s=g/100; cos=1/np.sqrt(1+s**2); a=np.gradient(v)
rho=float(ae.get_air_density(300,15,.5))
df['power']=(m*ae.G*(CRR*cos+s*cos)*v+m*v*a+0.5*rho*CDA*v**3)/(1-LOSS)
res={'ls':[],'mean':[]}
for _ in range(300):
    d=df.copy(); d['power']=df.power+rng.normal(0,25,N); d['speed']=df.speed+rng.normal(0,0.2,N)
    e=ae.estimate_cda(d,MASS); res['ls'].append(np.nan if e is None else e['cda'])
    cs,_=ae.calculate_CdA(d,MASS); res['mean'].append(ae.get_avg_cda(cs))
ls,mn=np.array(res['ls']),np.array(res['mean'])
out={'ls_mean':float(np.nanmean(ls)),'ls_sd':float(np.nanstd(ls)),'ls_fail':int(np.isnan(ls).sum()),'mean_mean':float(np.nanmean(mn)),'mean_sd':float(np.nanstd(mn)),'mean_power':float(df.power.mean()),'mean_speed':float(v.mean())}
np.save(f'{OUT}/mc2_ls.npy',ls); np.save(f'{OUT}/mc2_mean.npy',mn)
json.dump(out,open(f'{OUT}/mc_hard.json','w')); print(out)
