import os, json, numpy as np, matplotlib
os.chdir(os.path.dirname(os.path.abspath(__file__)))
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=json.load(open('results.json')); H=json.load(open('mc_hard.json'))
BLUE,VERM,GREEN,ORANGE='#0072B2','#D55E00','#009E73','#B07A00'   # Okabe-Ito; amber darkened for contrast on white
INK,MUTED='#222222','#666666'
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
  'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.linewidth':0.6,
  'grid.color':(0,0,0,0.10),'grid.linewidth':0.5,'axes.grid':True,'axes.axisbelow':True,'savefig.bbox':'tight','figure.dpi':150})

# 1. Monte Carlo CdA
fig,axs=plt.subplots(1,2,figsize=(6.4,2.5),sharey=False)
for ax,(a,b,ttl,mm) in zip(axs,[(np.load('mc_ls.npy'),np.load('mc_mean.npy'),'(a) Fast rolling ride',R['mc']),
                              (np.load('mc2_ls.npy'),np.load('mc2_mean.npy'),'(b) Slow, noisy, hilly ride',None)]):
    bins=np.linspace(min(np.nanmin(a),np.nanmin(b))-0.002,max(np.nanmax(a),np.nanmax(b))+0.002,34)
    ax.hist(b,bins=bins,color=VERM,alpha=0.75,label='mean of per-sample ratios',edgecolor='white',linewidth=0.4)
    ax.hist(a,bins=bins,color=BLUE,alpha=0.75,label='least-squares fit',edgecolor='white',linewidth=0.4)
    ax.axvline(0.30,color=INK,lw=1.0,ls='--',label='true CdA = 0.300')
    ax.set_title(ttl,fontsize=8.5,loc='left'); ax.set_xlabel(r'estimated CdA (m$^2$)'); ax.grid(axis='x',visible=False)
axs[0].set_ylabel('Monte-Carlo runs')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=7.5,loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig('figs/fig_cda_mc.pdf'); plt.close()

# 2. systematic errors
rows=sorted(R['sys']+[['No wind data supplied (wind ignored)',R['no_wind_input']]],key=lambda r:r[1])
fig,ax=plt.subplots(figsize=(6.2,3.0))
y=np.arange(len(rows)); vals=[r[1] for r in rows]
ax.barh(y,vals,color=BLUE,height=0.62); ax.axvline(0,color=INK,lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
for yi,v in zip(y,vals): ax.text(v+(0.0015 if v>=0 else -0.0015),yi,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=7.5)
ax.set_xlabel(r'bias in estimated CdA (m$^2$), true value 0.300'); ax.set_xlim(-0.065,0.105); ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig('figs/fig_cda_sys.pdf'); plt.close()

# 3. convective coefficient
v=np.linspace(0.2,15,300)
fig,ax=plt.subplots(figsize=(4.6,2.7))
ax.axvspan(4,15,color=(0,0,0,0.05),lw=0); ax.text(4.2,3,'above the range of the\nstandard correlation',fontsize=7,color=MUTED,va='bottom')
ax.plot(v,8.5*v**0.84,color=BLUE,lw=1.6); ax.plot(v,8.3*v**0.6+4.7,color=VERM,lw=1.6,ls='--'); ax.plot(v,8.3*v**0.6,color=GREEN,lw=1.4,ls=':')
ax.lines[0].set_label('model (combined, as implemented)'); ax.lines[1].set_label('ASHRAE convective + 4.7 radiative'); ax.lines[2].set_label('ASHRAE convective only')
ax.legend(frameon=False,fontsize=7.5,loc='upper left')
ax.set_xlabel('air speed over the body (m/s)'); ax.set_ylabel(r'heat-transfer coefficient (W m$^{-2}$ K$^{-1}$)'); ax.set_ylim(0,95); ax.set_xlim(0,15)
fig.tight_layout(); fig.savefig('figs/fig_hconv.pdf'); plt.close()

# 4. core temp time courses
fig,ax=plt.subplots(figsize=(4.8,2.8)); tt=np.arange(0,3600,10)/60
cols={'10':BLUE,'20':GREEN,'30':ORANGE,'35':VERM}; ls={'10':'-','20':'--','30':'-.','35':':'}
for T,c in R['curves'].items():
    ax.plot(tt[:len(c)],c,color=cols[T],ls=ls[T],lw=1.6); ax.text(60.6,c[-1],f'{T} °C',color=cols[T],va='center',fontsize=7.5)
ax.set_xlim(0,66); ax.set_xlabel('time (min)'); ax.set_ylabel('core temperature (°C)')
fig.tight_layout(); fig.savefig('figs/fig_core_curves.pdf'); plt.close()

# 5. humidity
fig,ax=plt.subplots(figsize=(4.6,2.8)); rh=np.array([20,35,50,65,80,95])
for T,c,l in zip(('24','28','32'),(BLUE,GREEN,VERM),('-','--','-.')):
    ax.plot(rh,R['hum'][T],color=c,ls=l,lw=1.6,marker='o',ms=3); ax.text(97,R['hum'][T][-1],f'{T} °C',color=c,va='center',fontsize=7.5)
ax.set_xlim(15,106); ax.set_xlabel('relative humidity (%)'); ax.set_ylabel('core temperature after 60 min (°C)')
fig.tight_layout(); fig.savefig('figs/fig_humidity.pdf'); plt.close()

# 6. tornado
lab={'EFFICIENCY':'gross efficiency','EVAP_COEFF_W_PER_M2':'evaporative capacity','K_VASO_MAX_W_PER_M2K':'vasodilation conductance',
 'H_CONV_COEFF':'convective coefficient','CLOTHING':'clothing insulation','K_MIN_W_PER_M2K':'basal core–skin conductance',
 'SWEAT_GAIN_W_PER_K':'sweat supply gain','M_REST_W_PER_M2':'resting metabolic rate','SETPOINT_GAIN_K':'exercise set-point gain'}
fig,axs=plt.subplots(1,2,figsize=(6.4,2.9),sharey=True)
order=sorted(lab,key=lambda k:-max(abs(x) for x in R['sens_warm'][1][k]))
for ax,key,ttl in zip(axs,('sens_warm','sens_cool'),('(a) 220 W, 28 °C, 50 % RH','(b) 180 W, 12 °C, 70 % RH')):
    base,d=R[key]; y=np.arange(len(order))[::-1]
    for yi,k in zip(y,order):
        lo,hi=d[k]; ax.barh(yi,lo,color=BLUE,height=0.55); ax.barh(yi,hi,color=VERM,height=0.55)
    ax.axvline(0,color=INK,lw=0.8); ax.set_title(f'{ttl}\nbaseline {base:.2f} °C',fontsize=8,loc='left'); ax.grid(axis='y',visible=False)
    ax.set_xlabel(r'change in final core temperature (K)')
axs[0].set_yticks(np.arange(len(order))[::-1]); axs[0].set_yticklabels([lab[k] for k in order])
axs[0].text(-0.05,-0.95,'−20 %',color=BLUE,fontsize=7.5,ha='right'); axs[0].text(0.05,-0.95,'+20 %',color=VERM,fontsize=7.5,ha='left')
axs[0].set_ylim(-1.3,len(order)-0.4)
fig.tight_layout(); fig.savefig('figs/fig_sens.pdf'); plt.close()
print('ok')
