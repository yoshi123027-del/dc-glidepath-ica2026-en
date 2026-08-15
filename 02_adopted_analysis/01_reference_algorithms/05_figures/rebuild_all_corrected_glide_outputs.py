from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'figs'; RES=ROOT/'results'; SW=ROOT/'sensitivity_workers'; MW=ROOT/'mvs_workers'
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
X0=1/12; T=40.0; R=.015; MU=.055; BETA=MU-R; SIGMA=.18; C=1.0

# ---------- common utilities ----------
def deposit(grid,mass,x,w):
    if x<=grid[0]: mass[0]+=w; return
    if x>=grid[-1]: mass[-1]+=w; return
    j=int(np.searchsorted(grid,x)-1); lam=(x-grid[j])/(grid[j+1]-grid[j])
    mass[j]+=w*(1-lam); mass[j+1]+=w*lam

def initial_fraction_from_amount(policy0,grid,x0=X0):
    pi0=float(np.interp(x0,grid,policy0*grid))
    return float(np.clip(pi0/max(x0,1e-14),0,1))

def forward_exact_initial(policy,grid,dt,r=R,beta=BETA,sigma=SIGMA,c=C,n_gh=5,x0=X0):
    N=policy.shape[0]; nx=len(grid)
    h,w=hermgauss(n_gh); ghx=np.sqrt(2)*h; ghw=w/np.sqrt(np.pi)
    p=np.zeros(nx); deposit(grid,p,x0,1.0)
    pmf=np.zeros((N+1,nx)); pmf[0]=p
    glide=np.empty(N); upper=np.empty(N)
    for n in range(N):
        if n==0:
            a=initial_fraction_from_amount(policy[0],grid,x0)
            glide[0]=a; upper[0]=float(a>=.999)
            pn=np.zeros(nx); pi=a*x0; drift=(r*x0+c+beta*pi)*dt; sd=sigma*pi*math.sqrt(dt)
            for z,qw in zip(ghx,ghw): deposit(grid,pn,max(x0+drift+sd*z,0.0),float(qw))
        else:
            glide[n]=float(np.dot(p,policy[n])); upper[n]=float(np.dot(p,policy[n]>=.999))
            pn=np.zeros(nx)
            for i,x in enumerate(grid):
                if p[i]<=0: continue
                a=float(policy[n,i]); pi=a*x; drift=(r*x+c+beta*pi)*dt; sd=sigma*pi*math.sqrt(dt)
                for z,qw in zip(ghx,ghw): deposit(grid,pn,max(x+drift+sd*z,0.0),float(p[i]*qw))
        total=pn.sum();
        if total>0: pn/=total
        p=pn; pmf[n+1]=p
    return pmf,glide,upper

def quantile(values,p,q):
    pp=p/p.sum(); return float(values[min(np.searchsorted(np.cumsum(pp),q),len(values)-1)])
def lower_cvar(values,p,alpha=.05):
    pp=p/p.sum(); rem=alpha; s=0.0
    for v,w in zip(values,pp):
        take=min(rem,float(w)); s+=take*float(v); rem-=take
        if rem<=1e-15: break
    return s/alpha
def upper_cvar(values,p,alpha=.05):
    pp=p/p.sum(); rem=alpha; s=0.0
    for v,w in zip(values[::-1],pp[::-1]):
        take=min(rem,float(w)); s+=take*float(v); rem-=take
        if rem<=1e-15: break
    return s/alpha
def stats(values,p):
    pp=p/p.sum(); mean=float(pp@values); var=float(pp@((values-mean)**2)); sd=math.sqrt(max(var,0)); cm3=float(pp@((values-mean)**3))
    return dict(mean=mean,stdev=sd,skewness=cm3/(sd**3+1e-30),q05=quantile(values,pp,.05),q50=quantile(values,pp,.5),q95=quantile(values,pp,.95),cvar05=lower_cvar(values,pp,.05),ucvar95=upper_cvar(values,pp,.05),third_central_moment=cm3)

def plot_glides(curves,times,out,ylabel='Mass-weighted risky fraction',ncol=3,figsize=(9.2,5.6)):
    fig,ax=plt.subplots(figsize=figsize)
    for label,curve,kwargs in curves: ax.plot(times,curve,label=label,linewidth=1.8,**kwargs)
    ax.set_xlabel('Years since entry'); ax.set_ylabel(ylabel); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=ncol)
    fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)

# ---------- monthly baseline ----------
with np.load(RES/'monthly_D0_policy_arrays.npz') as z: old={k:z[k].copy() for k in z.files}
times=old['times']; dt=float(times[1]-times[0]); decision_times=times[:-1]
xpc=old['xg_pc']; xtc=old['xg_tc']
policies={'PCMV':(old['pcmv_policy'],xpc),'DOMV':(old['domv_policy'],xpc),'cTCMV':(old['ctcmv_policy'],xtc),'dTCMV':(old['dtcmv_policy'],xtc),'CP':(np.full((len(decision_times),len(xtc)),0.4694735),xtc)}
monthly={}
for name,(P,xg) in policies.items():
    pmf,g,u=forward_exact_initial(P,xg,dt)
    monthly[name]=dict(policy=P,xg=xg,pmf=pmf,glide=g,upper=u,stats=stats(xg,pmf[-1]))

plot_glides([(n,monthly[n]['glide'],{}) for n in monthly],decision_times,FIG/'fig_all_strategies_glidepaths_D0_N480.png')
plot_glides([(n,monthly[n]['glide'],{}) for n in ['PCMV','DOMV']],decision_times,FIG/'fig_pcmv_domv_glidepaths_N480.png',ncol=2,figsize=(8.6,5.2))
# upper binding
fig,ax=plt.subplots(figsize=(9.2,5.6))
for n in ['PCMV','DOMV','cTCMV','dTCMV']: ax.plot(decision_times,monthly[n]['upper'],label=n,linewidth=1.8)
ax.set_xlabel('Years since entry'); ax.set_ylabel('Probability mass at upper constraint'); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/'fig_all_strategies_upper_binding_D0_N480.png',dpi=180); plt.close(fig)
# CDF and density
fig,ax=plt.subplots(figsize=(9.2,5.6))
for n in monthly: ax.plot(monthly[n]['xg'],np.cumsum(monthly[n]['pmf'][-1]/monthly[n]['pmf'][-1].sum()),label=n)
ax.set_xlabel('Terminal DC wealth'); ax.set_ylabel('CDF'); ax.set_xlim(0,220); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig(FIG/'fig_all_strategies_terminal_cdf_D0_N480.png',dpi=180); plt.close(fig)
# common uniform grid density
ug=np.linspace(0,220,1000)
fig,ax=plt.subplots(figsize=(9.2,5.6))
for n in monthly:
    xg=monthly[n]['xg']; p=monthly[n]['pmf'][-1]/monthly[n]['pmf'][-1].sum(); mass=np.zeros_like(ug)
    for x,wgt in zip(xg,p): deposit(ug,mass,float(x),float(wgt))
    dens=gaussian_filter1d(mass,sigma=5,mode='nearest')/(ug[1]-ug[0]); ax.plot(ug,dens,label=n)
ax.set_xlabel('Terminal DC wealth'); ax.set_ylabel('Smoothed density'); ax.set_xlim(0,220); ax.grid(alpha=.25); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig(FIG/'fig_all_strategies_terminal_density_D0_N480.png',dpi=180); plt.close(fig)
# save summary and arrays
rows=[]
for n in monthly:
    st=monthly[n]['stats']; rows.append({'strategy':n,**{k:st[k] for k in ['mean','stdev','skewness','q05','q50','q95','cvar05']},'avg_glide':float(monthly[n]['glide'].mean()),'upper_bind':float(monthly[n]['upper'].mean())})
pd.DataFrame(rows).to_csv(RES/'monthly_baseline_D0_summary.csv',index=False)
np.savez_compressed(RES/'monthly_D0_policy_arrays.npz',times=times,decision_times=decision_times,xg_pc=xpc,xg_tc=xtc,
    pcmv_policy=monthly['PCMV']['policy'],domv_policy=monthly['DOMV']['policy'],ctcmv_policy=monthly['cTCMV']['policy'],dtcmv_policy=monthly['dTCMV']['policy'],
    pcmv_pmf=monthly['PCMV']['pmf'],domv_pmf=monthly['DOMV']['pmf'],ctcmv_pmf=monthly['cTCMV']['pmf'],dtcmv_pmf=monthly['dTCMV']['pmf'],cp_pmf=monthly['CP']['pmf'],
    pcmv_glide=monthly['PCMV']['glide'],domv_glide=monthly['DOMV']['glide'],ctcmv_glide=monthly['cTCMV']['glide'],dtcmv_glide=monthly['dTCMV']['glide'],cp_glide=monthly['CP']['glide'],
    pcmv_upper=monthly['PCMV']['upper'],domv_upper=monthly['DOMV']['upper'],ctcmv_upper=monthly['cTCMV']['upper'],dtcmv_upper=monthly['dTCMV']['upper'])

# ---------- sensitivity workers ----------
keys=['baseline','D_alt','r_low','r_high','mu_low','mu_high','sigma_low','sigma_high','contrib_constant','contrib_linear','contrib_quadratic']
labels={'baseline':r'$D_T=0$','D_alt':r'$D_T=FV_r(c)$','r_low':r'$r=0.005$','r_high':r'$r=0.025$','mu_low':r'$\mu=0.045$','mu_high':r'$\mu=0.065$','sigma_low':r'$\sigma=0.14$','sigma_high':r'$\sigma=0.22$','contrib_constant':'Constant','contrib_linear':'Linear increase','contrib_quadratic':'Quadratic increase'}
strats=['PCMV','DOMV','cTCMV','dTCMV','CP']; S={}; srows=[]
for key in keys:
    with np.load(SW/f'{key}.npz') as z: S[key]={str(z['strategies'][i]):z['glides'][i].copy() for i in range(len(z['strategies']))}; t80=z['decision_times'].copy()
    srows.append(pd.read_csv(SW/f'{key}.csv'))
sdf=pd.concat(srows,ignore_index=True); sdf.to_csv(RES/'sensitivity_summary_corrected.csv',index=False)
np.savez_compressed(RES/'sensitivity_glidepaths_corrected.npz',decision_times=t80,scenario_keys=np.array(keys),strategies=np.array(strats),glides=np.stack([[S[k][s] for s in strats] for k in keys]))
def panels(use,filename,title):
    fig,axes=plt.subplots(2,2,figsize=(10.5,8),sharex=True)
    for ax,s in zip(axes.ravel(),strats[:4]):
        for k in use: ax.plot(t80,S[k][s],label=labels[k],linewidth=1.6)
        ax.set_title(s); ax.set_xlabel('Years since entry'); ax.set_ylabel('Mass-weighted risky proportion'); ax.set_ylim(0,1.04); ax.grid(alpha=.25)
    h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.945),ncol=min(3,len(l)),frameon=False)
    fig.suptitle(title,y=.995); fig.tight_layout(rect=(0,0,1,.89)); fig.savefig(FIG/filename,dpi=180); plt.close(fig)
labels['baseline']=r'$D_T=0$'
panels(['baseline','D_alt'],'fig_D_sensitivity_glidepaths_N80.png',r'Sensitivity to deterministic terminal benefit $D_T$')
labels['baseline']=r'$r=0.015$'
panels(['r_low','baseline','r_high'],'fig_r_sensitivity_glidepaths_N80.png',r'Sensitivity to $r$')
labels['baseline']=r'$\mu=0.055$'
panels(['mu_low','baseline','mu_high'],'fig_mu_sensitivity_glidepaths_N80.png',r'Sensitivity to $\mu$')
labels['baseline']=r'$\sigma=0.18$'
panels(['sigma_low','baseline','sigma_high'],'fig_sigma_sensitivity_glidepaths_N80.png',r'Sensitivity to $\sigma$')
panels(['contrib_constant','contrib_linear','contrib_quadratic'],'fig_contrib_profile_sensitivity_glidepaths_N80.png','Contribution-profile sensitivity with fixed total contributions')
plot_glides([(s,S['baseline'][s],{}) for s in strats],t80,FIG/'fig_revised_baseline_D0_glidepaths_N80.png')
# time-grid
# custom due different x arrays
fig,ax=plt.subplots(figsize=(9.2,5.6))
ax.plot(t80,S['baseline']['cTCMV'],'--',label='cTCMV N=80'); ax.plot(decision_times,monthly['cTCMV']['glide'],label='cTCMV N=480')
ax.plot(t80,S['baseline']['dTCMV'],'--',label='dTCMV N=80'); ax.plot(decision_times,monthly['dTCMV']['glide'],label='dTCMV N=480')
ax.set_xlabel('Years since entry'); ax.set_ylabel('Mass-weighted risky proportion'); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/'fig_tcmv_time_grid_glide_comparison_D0.png',dpi=180); plt.close(fig)
try: (FIG/'_dummy.png').unlink()
except FileNotFoundError: pass

# ---------- MVS worker aggregation ----------
fixed_eta=[0.,.5,1.,2.]; cal_eta=[0.,1.,2.,4.,8.]
def load_mvs(kind,eta):
    with np.load(MW/f'{kind}_{eta:g}.npz') as z: return {k:z[k].copy() for k in z.files}
fixed=[load_mvs('fixed',e) for e in fixed_eta]; cal=[load_mvs('cal',e) for e in cal_eta]
fixed_df=pd.concat([pd.read_csv(MW/f'fixed_{e:g}.csv') for e in fixed_eta],ignore_index=True)
cal_df=pd.concat([pd.read_csv(MW/f'cal_{e:g}.csv') for e in cal_eta],ignore_index=True)
fixed_df.to_csv(RES/'dtcmv_mvs_fixed_gamma_summary.csv',index=False); cal_df.to_csv(RES/'dtcmv_mvs_equal_mean_summary.csv',index=False)
mt=fixed[0]['decision_times']; mx=fixed[0]['x_grid']
np.savez_compressed(RES/'dtcmv_mvs_arrays.npz',times=np.linspace(0,T,len(mt)+1),decision_times=mt,x_grid=mx,fixed_eta_grid=np.array(fixed_eta),calibrated_eta_grid=np.array(cal_eta),gamma_calibrated=cal_df['gamma0'].to_numpy(),glide_fixed=np.stack([r['glide'] for r in fixed]),glide_calibrated=np.stack([r['glide'] for r in cal]),pmf_calibrated=np.stack([r['pmf'] for r in cal]),policy_calibrated=np.stack([r['policy'] for r in cal]))
plot_glides([(fr'$\eta_0={e:.2f}$',r['glide'],{}) for e,r in zip(fixed_eta,fixed)],mt,FIG/'fig_dtcmv_mvs_fixed_gamma_glidepaths.png',ylabel='Mass-weighted risky proportion',ncol=2,figsize=(9,5.4))
plot_glides([(fr'$\eta_0={e:.2f}$, $\gamma_0={g:.2f}$',r['glide'],{}) for e,g,r in zip(cal_eta,cal_df['gamma0'],cal)],mt,FIG/'fig_dtcmv_mvs_equal_mean_glidepaths.png',ylabel='Mass-weighted risky proportion',ncol=2,figsize=(9,5.4))
fig,ax=plt.subplots(figsize=(9,5.4))
for e,r in zip(cal_eta,cal): ax.plot(r['x_grid'],np.cumsum(r['pmf'][-1]/r['pmf'][-1].sum()),label=fr'$\eta_0={e:.2f}$')
ax.set_xlabel('Terminal DC wealth'); ax.set_ylabel('CDF'); ax.set_xlim(0,200); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/'fig_dtcmv_mvs_equal_mean_cdf.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,5.4)); ax.plot(cal_df['cvar05'],cal_df['ucvar95'],marker='o')
for _,r in cal_df.iterrows(): ax.annotate(fr'$\eta_0={r.eta0:.2f}$',(r.cvar05,r.ucvar95),xytext=(5,4),textcoords='offset points')
ax.set_xlabel('Lower-tail CVaR (5%)'); ax.set_ylabel('Upper-tail conditional mean (95%)'); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG/'fig_dtcmv_mvs_tail_frontier.png',dpi=180); plt.close(fig)

print('MONTHLY')
print(pd.DataFrame(rows).to_string(index=False))
print('\nSENSITIVITY')
print(sdf[['scenario','strategy','mean','stdev','q05','q95','cvar05','mean_glide','initial_glide','last_decision_glide']].to_string(index=False))
print('\nMVS FIXED')
print(fixed_df[['eta0','gamma0','mean','stdev','skewness','q05','q95','cvar05','ucvar95','mean_abs_glide']].to_string(index=False))
print('\nMVS CAL')
print(cal_df[['eta0','gamma0','mean','stdev','skewness','q05','q95','cvar05','ucvar95','mean_abs_glide']].to_string(index=False))
