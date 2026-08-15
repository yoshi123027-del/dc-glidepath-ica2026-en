from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'results'; FIG=ROOT/'figs'
arr=np.load(RES/'monthly_D0_policy_arrays.npz')
times=arr['times']; dt=float(times[1]-times[0]); N=len(times)-1
r=.015; mu=.055; beta=mu-r; sigma=.18; c=1.0; x0=1/12
h,w=hermgauss(5); zs=np.sqrt(2)*h; ws=w/np.sqrt(np.pi)

def deposit(p,xg,x,weight):
    if x<=xg[0]: p[0]+=weight; return
    if x>=xg[-1]: p[-1]+=weight; return
    j=int(np.searchsorted(xg,x)-1); lam=(x-xg[j])/(xg[j+1]-xg[j])
    p[j]+=weight*(1-lam); p[j+1]+=weight*lam

def quantile(xg,p,q):
    pp=p/p.sum(); return float(xg[min(np.searchsorted(np.cumsum(pp),q),len(xg)-1)])

def stats(xg,p):
    pp=p/p.sum(); m=float(pp@xg); v=float(pp@((xg-m)**2)); return m,math.sqrt(max(v,0))

def forward(n0,start,policy,xg, exact_start=False):
    p=np.zeros(len(xg))
    if exact_start and n0<N:
        # exact deterministic starting state over the first transition
        pi=float(np.interp(start,xg,xg*policy[n0]))
        drift=(r*start+c+beta*pi)*dt; sd=sigma*pi*math.sqrt(dt)
        for z,ww in zip(zs,ws): deposit(p,xg,max(0,start+drift+sd*z),ww)
        nstart=n0+1
    else:
        deposit(p,xg,start,1.0); nstart=n0
    for n in range(nstart,N):
        pn=np.zeros_like(p)
        for i,x in enumerate(xg):
            if p[i]<=0: continue
            a=float(policy[n,i]); pi=a*x
            drift=(r*x+c+beta*pi)*dt; sd=sigma*pi*math.sqrt(dt)
            for z,ww in zip(zs,ws): deposit(pn,xg,max(0,x+drift+sd*z),p[i]*ww)
        p=pn/pn.sum()
    return p/p.sum()

def backward_moments(policy,xg):
    M=np.empty((N+1,len(xg))); Q=np.empty_like(M)
    M[N]=xg; Q[N]=xg*xg
    for n in range(N-1,-1,-1):
        for i,x in enumerate(xg):
            pi=float(policy[n,i])*x
            nxt=x+(r*x+c+beta*pi)*dt+sigma*pi*math.sqrt(dt)*zs
            nxt=np.clip(nxt,xg[0],xg[-1])
            M[n,i]=float(ws@np.interp(nxt,xg,M[n+1]))
            Q[n,i]=float(ws@np.interp(nxt,xg,Q[n+1]))
    return M,Q

configs=[
 ('PCMV',arr['pcmv_policy'],arr['pcmv_pmf'],arr['xg_pc']),
 ('DOMV',arr['domv_policy'],arr['domv_pmf'],arr['xg_pc']),
 ('cTCMV',arr['ctcmv_policy'],arr['ctcmv_pmf'],arr['xg_tc']),
 ('dTCMV',arr['dtcmv_policy'],arr['dtcmv_pmf'],arr['xg_tc']),
]
years=[0,10,20,30,35,39]
rows=[]; valrows=[]
for name,pol,ownpmf,xg in configs:
    M,Q=backward_moments(pol,xg)
    for year in years:
        n=min(int(round(year/dt)),N-1)
        if year==0:
            start=x0; p=forward(0,start,pol,xg,exact_start=True)
            pi0=float(np.interp(start,xg,xg*pol[0])); risky=pi0/start
            nxt=start+(r*start+c+beta*pi0)*dt+sigma*pi0*math.sqrt(dt)*zs
            bm=float(ws@np.interp(np.clip(nxt,xg[0],xg[-1]),xg,M[1]))
            bq=float(ws@np.interp(np.clip(nxt,xg[0],xg[-1]),xg,Q[1]))
        else:
            start=quantile(xg,ownpmf[n],.5); p=forward(n,start,pol,xg)
            risky=float(np.interp(start,xg,pol[n])); idx=int(np.argmin(abs(xg-start)))
            bm=float(M[n,idx]); bq=float(Q[n,idx])
        fm,fs=stats(xg,p); bs=math.sqrt(max(bq-bm*bm,0))
        rows.append(dict(strategy=name,year=year,median_x=start,conditional_mean=fm,conditional_std=fs,risky_fraction=risky))
        valrows.append(dict(strategy=name,year=year,forward_mean=fm,backward_mean=bm,mean_residual=fm-bm,forward_std=fs,backward_std=bs,std_residual=fs-bs))

df=pd.DataFrame(rows); vf=pd.DataFrame(valrows)
df.to_csv(RES/'rolling_conditional_D0_N480.csv',index=False)
vf.to_csv(RES/'rolling_validation_D0_N480.csv',index=False)

# One compact 3-panel rolling figure.
fig,axs=plt.subplots(3,1,figsize=(9.2,10.2),sharex=True)
for name in df.strategy.unique():
    d=df[df.strategy==name]
    axs[0].plot(d.year,d.conditional_mean,marker='o',label=name)
    axs[1].plot(d.year,d.conditional_std,marker='o',label=name)
    axs[2].plot(d.year,d.risky_fraction,marker='o',label=name)
axs[0].set_ylabel('Conditional mean')
axs[1].set_ylabel('Conditional standard deviation')
axs[2].set_ylabel('Current risky fraction'); axs[2].set_xlabel('Years since entry')
for ax in axs: ax.grid(alpha=.25)
axs[0].legend(ncol=4)
fig.tight_layout(); fig.savefig(FIG/'fig_rolling_conditional_all_strategies_D0_N480.png',dpi=180); plt.close(fig)

# validation residual plot
fig,ax=plt.subplots(figsize=(8.8,4.5))
for name in vf.strategy.unique():
    d=vf[vf.strategy==name]
    ax.semilogy(d.year,np.maximum(np.abs(d.mean_residual),1e-16),marker='o',label=f'{name}: mean')
    ax.semilogy(d.year,np.maximum(np.abs(d.std_residual),1e-16),marker='x',linestyle='--',label=f'{name}: std')
ax.set_xlabel('Years since entry'); ax.set_ylabel('Absolute residual'); ax.grid(alpha=.25); ax.legend(ncol=2,fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'fig_rolling_validation_residual_D0_N480.png',dpi=180); plt.close(fig)
print(df.to_string(index=False))
print('\nMax residuals')
print(vf.groupby('strategy')[['mean_residual','std_residual']].agg(lambda s: np.max(np.abs(s))))
