from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jp_gamma25_common_mean"
FIG = ROOT / "figs" / "jp_gamma25_common_mean"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
coef = json.loads((OUT / "refined_equal_mean_coefficients.json").read_text(encoding="utf-8"))
TARGET = float(coef["target_mean"])

roll_path = ROOT / "scripts" / "03_rolling" / "recompute_d0_rolling.py"
spec = importlib.util.spec_from_file_location("roll_apply", roll_path)
roll = importlib.util.module_from_spec(spec); sys.modules["roll_apply"] = roll; spec.loader.exec_module(roll)
pc = roll.pcmod; family = roll.pcres["family"]; xg_pc = family["x_grid"]


def stats(grid, pmf):
    st = roll.stats_from_pmf(grid, pmf[-1], 0.0)
    p = pmf[-1] / pmf[-1].sum(); rem=.05; total=0.0
    for v,w in zip(grid[::-1],p[::-1]):
        take=min(rem,w); total += take*v; rem -= take
        if rem <= 1e-15: break
    st["ucvar95"] = total/.05
    return st

# PCMV at refined gamma and refined fixed point target.
pcfg = replace(roll.pcfg, gamma_p=float(coef["gamma_p"]))
pex = pc.solve_exact_target(pcfg, float(coef["pcmv_target_z"]))
P_p = pex["policy"][0].astype(float)
fwd_p = pc.forward_distribution(pcfg, xg_pc, P_p, pex["gh_x"], pex["gh_w"])
pmf_p = fwd_p["pmf"]; glide_p, upper_p = roll.decision_summary_from_policy(P_p, pmf_p, xg_pc, roll.x0); st_p=stats(xg_pc,pmf_p)

# DOMV.
dcfg = replace(roll.pcfg, gamma_d=float(coef["gamma_d"]))
dobj = pc.build_domv_policy(dcfg, family); P_D=dobj["policy"]
fwd_D=pc.forward_distribution(dcfg,xg_pc,P_D,family["gh_x"],family["gh_w"])
pmf_D=fwd_D["pmf"]; glide_D,upper_D=roll.decision_summary_from_policy(P_D,pmf_D,xg_pc,roll.x0); st_D=stats(xg_pc,pmf_D)

# cTCMV / dTCMV.
def tc(dynamic, g):
    M,Q,P=roll.solve_tc(dynamic,roll.N,roll.xg,roll.H,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.D,float(coef["gamma_c"]),float(g),roll.n_controls,roll.gh_x,roll.gh_w)
    pmf,glide,upper=roll.forward_policy(roll.N,roll.xg,roll.x0,0,P,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w)
    return M,Q,P,pmf,glide,upper,stats(roll.xg,pmf)
M_c,Q_c,P_c,pmf_c,glide_c,upper_c,st_c=tc(0,float(coef["rho_d"]))
M_d,Q_d,P_d,pmf_d,glide_d,upper_d,st_d=tc(1,float(coef["rho_d"]))

# CP.
theta=float(coef["theta_cp"]); P_cp=np.full((roll.N,roll.n_x),theta)
pmf_cp,glide_cp,upper_cp=roll.forward_policy(roll.N,roll.xg,roll.x0,0,P_cp,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w); st_cp=stats(roll.xg,pmf_cp)

cal=pd.DataFrame([
    ["PCMV","gamma_p",coef["gamma_p"],TARGET,st_p["mean"]],
    ["DOMV","gamma_d",coef["gamma_d"],TARGET,st_D["mean"]],
    ["cTCMV","gamma_c",coef["gamma_c"],TARGET,st_c["mean"]],
    ["dTCMV","rho_d",coef["rho_d"],TARGET,st_d["mean"]],
    ["CP","theta_cp",coef["theta_cp"],TARGET,st_cp["mean"]],
],columns=["strategy","parameter","value","target_mean","achieved_mean"])
cal["absolute_mean_error"]=(cal.achieved_mean-TARGET).abs(); cal.to_csv(OUT/"mv_equal_mean_calibration.csv",index=False)

rows=[]
for name,st,g,u in [("PCMV",st_p,glide_p,upper_p),("DOMV",st_D,glide_D,upper_D),("cTCMV",st_c,glide_c,upper_c),("dTCMV",st_d,glide_d,upper_d),("CP",st_cp,glide_cp,upper_cp)]:
    rows.append({"strategy":name,**st,"avg_glide":float(np.mean(g)),"upper_bind":float(np.mean(u)),"final_glide":float(g[-1])})
pd.DataFrame(rows).to_csv(OUT/"mv_baseline_summary.csv",index=False)

# Main figures.
times=np.arange(roll.N)*roll.dt
fig,ax=plt.subplots(figsize=(9.2,5.6))
for name,g in [("PCMV",glide_p),("DOMV",glide_D),("cTCMV",glide_c),("dTCMV",glide_d),("CP",glide_cp)]: ax.plot(times,g,label=name,lw=1.8)
ax.set(xlabel="Years since entry",ylabel="Mass-weighted risky proportion",ylim=(0,1.04)); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(FIG/"fig_all_strategies_glidepaths.png",dpi=180); plt.close(fig)
series=[("PCMV",xg_pc,pmf_p[-1]),("DOMV",xg_pc,pmf_D[-1]),("cTCMV",roll.xg,pmf_c[-1]),("dTCMV",roll.xg,pmf_d[-1]),("CP",roll.xg,pmf_cp[-1])]
fig,ax=plt.subplots(figsize=(9.2,5.6))
for name,x,p in series: ax.plot(x,np.cumsum(p/p.sum()),label=name,lw=1.8)
ax.set(xlabel="Terminal DC wealth",ylabel="CDF",xlim=(0,180),ylim=(0,1)); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(FIG/"fig_terminal_cdf.png",dpi=180); plt.close(fig)
ug=np.linspace(0,180,1200); dx=ug[1]-ug[0]
def uniform_mass(x,p):
    m=np.zeros_like(ug)
    for v,w in zip(x,p):
        if w<=0: continue
        q=(v-ug[0])/dx
        if q<=0:m[0]+=w
        elif q>=len(ug)-1:m[-1]+=w
        else:
            j=int(math.floor(q)); l=q-j; m[j]+=w*(1-l); m[j+1]+=w*l
    return m
fig,ax=plt.subplots(figsize=(9.2,5.6))
for name,x,p in series: ax.plot(ug,gaussian_filter1d(uniform_mass(x,p),5,mode="nearest")/dx,label=name,lw=1.8)
ax.set(xlabel="Terminal DC wealth",ylabel="Density",xlim=(0,160)); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(FIG/"fig_terminal_density.png",dpi=180); plt.close(fig)

# Rolling conditional profiles.
Mp,Qp=roll.policy_moments_pc(roll.N,xg_pc,P_p,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
MD,QD=roll.policy_moments_pc(roll.N,xg_pc,P_D,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
Mc,Qc=roll.policy_moments(roll.N,roll.xg,P_c,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
Md,Qd=roll.policy_moments(roll.N,roll.xg,P_d,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
strategies={"PCMV":(xg_pc,P_p,pmf_p,Mp,Qp),"DOMV":(xg_pc,P_D,pmf_D,MD,QD),"cTCMV":(roll.xg,P_c,pmf_c,Mc,Qc),"dTCMV":(roll.xg,P_d,pmf_d,Md,Qd)}
rrows=[]
for name,(gx,P,pmf,M,Q) in strategies.items():
    for year in [0,10,20,30,35,39]:
        n=min(int(round(year/roll.dt)),roll.N-1); xm=roll.median_state(gx,pmf[n]); risky=roll.interp_at(gx,P[n],xm)
        pcnd,_,_=roll.forward_policy(roll.N,gx,xm,n,P,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w); st=stats(gx,pcnd)
        mb=roll.interp_at(gx,M[n],xm); qb=roll.interp_at(gx,Q[n],xm)
        rrows.append({"strategy":name,"year":year,"median_x":xm,"mean":st["mean"],"stdev":st["stdev"],"skewness":st["skewness"],"q05":st["q05"],"q95":st["q95"],"cvar05":st["cvar05"],"ucvar95":st["ucvar95"],"risky_fraction":risky,"backward_mean":mb,"backward_stdev":math.sqrt(max(qb-mb*mb,0.0))})
rd=pd.DataFrame(rrows); rd.to_csv(OUT/"mv_rolling_conditional.csv",index=False)

# Clipped controls.
def dollars_to_fraction(g,d):
    cl=np.minimum(np.maximum(d,0.0),g[None,:]); out=np.zeros_like(cl); pos=g>0; out[:,pos]=cl[:,pos]/g[None,pos]; return np.clip(out,0,1)
A=roll.beta**2/roll.sigma**2; tau=roll.T-times; y0=roll.x0+roll.H[0]
z_uncon=y0*math.exp(roll.r*roll.T)+math.exp(A*roll.T)/float(coef["gamma_p"])
Pp_clip=dollars_to_fraction(xg_pc,roll.beta/roll.sigma**2*(z_uncon*np.exp(-roll.r*tau)[:,None]-(xg_pc[None,:]+roll.H[:-1,None])))
PD_clip=dollars_to_fraction(xg_pc,np.broadcast_to((roll.beta/(roll.sigma**2*float(coef["gamma_d"]))*np.exp((A-roll.r)*tau))[:,None],(roll.N,len(xg_pc))).copy())
Pc_clip=dollars_to_fraction(roll.xg,np.broadcast_to((roll.beta/(roll.sigma**2*float(coef["gamma_c"]))*np.exp(-roll.r*tau))[:,None],(roll.N,len(roll.xg))).copy())
def theta_dt(rho):
    def rhs(_t,s):
        i1,i2=s; th=A/(rho*roll.beta)*(math.exp(-i1)+rho*math.exp(-i2)-rho); return [-(roll.r+roll.beta*th-roll.sigma**2*th**2),-(roll.sigma**2*th**2)]
    sol=solve_ivp(rhs,(roll.T,0),[0,0],rtol=1e-10,atol=1e-12,dense_output=True,method="DOP853"); i1,i2=sol.sol(times); return A/(rho*roll.beta)*(np.exp(-i1)+rho*np.exp(-i2)-rho)
th=theta_dt(float(coef["rho_d"])); Pd_clip=dollars_to_fraction(roll.xg,th[:,None]*(roll.xg[None,:]+roll.H[:-1,None]))
clip=[]
for name,gx,P,pmf,gstrict,Pclip in [("PCMV",xg_pc,P_p,pmf_p,glide_p,Pp_clip),("DOMV",xg_pc,P_D,pmf_D,glide_D,PD_clip),("cTCMV",roll.xg,P_c,pmf_c,glide_c,Pc_clip),("dTCMV",roll.xg,P_d,pmf_d,glide_d,Pd_clip)]:
    pmfc,gc,uc=roll.forward_policy(roll.N,gx,roll.x0,0,Pclip,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w); sc=stats(gx,pmfc); ss=stats(gx,pmf)
    clip.append({"strategy":name,"glide_mae":float(np.mean(np.abs(gc-gstrict))),"glide_max_abs":float(np.max(np.abs(gc-gstrict))),"mean_clip_minus_strict":sc["mean"]-ss["mean"],"q05_clip_minus_strict":sc["q05"]-ss["q05"],"q95_clip_minus_strict":sc["q95"]-ss["q95"],"cvar05_clip_minus_strict":sc["cvar05"]-ss["cvar05"]})
pd.DataFrame(clip).to_csv(OUT/"mv_strict_vs_clip_summary.csv",index=False)

# dTCMV decomposition.
dec=[]
for year in [0,10,20,30,35,39]:
    n=min(int(round(year/roll.dt)),roll.N-1); xm=roll.median_state(roll.xg,pmf_d[n]); H=roll.H[n]; g=roll.interp_at(roll.xg,P_d[n],xm)
    dec.append({"year":year,"median_x":xm,"H_over_x":H/max(xm,1e-14),"Gamma":float(coef["rho_d"])/max(xm+H,1e-14),"theta_unconstrained":th[n],"strict_risky_fraction":g,"strict_total_wealth_fraction":g*xm/max(xm+H,1e-14)})
pd.DataFrame(dec).to_csv(OUT/"dtcmv_u_decomposition.csv",index=False)

# Annual summary, preserving MVS columns from the first pass when available.
idx=np.arange(0,roll.N,12); annual=pd.DataFrame({"year":times[idx],"PCMV":glide_p[idx],"DOMV":glide_D[idx],"cTCMV":glide_c[idx],"dTCMV":glide_d[idx],"CP":glide_cp[idx]})
old=OUT/"annual_glidepaths.csv"
if old.exists():
    od=pd.read_csv(old)
    for col in od.columns:
        if col.startswith("MVS_"): annual[col]=od[col].values[:len(annual)]
annual.to_csv(old,index=False)
np.savez_compressed(OUT/"mv_policy_arrays.npz",decision_times=times,xg_pc=xg_pc,xg_tc=roll.xg,pcmv_policy=P_p,domv_policy=P_D,ctcmv_policy=P_c,dtcmv_policy=P_d,pcmv_pmf=pmf_p,domv_pmf=pmf_D,ctcmv_pmf=pmf_c,dtcmv_pmf=pmf_d,cp_pmf=pmf_cp,pcmv_glide=glide_p,domv_glide=glide_D,ctcmv_glide=glide_c,dtcmv_glide=glide_d,cp_glide=glide_cp)

meta={"paper_wide_target_mean":TARGET,"target_definition":"dTCMV-MVS base-grid mean at gamma0=2.5, eta0=0","mv_calibration":{"gamma_p":coef["gamma_p"],"gamma_d":coef["gamma_d"],"gamma_c":coef["gamma_c"],"rho_d":coef["rho_d"],"theta_cp":coef["theta_cp"]},"mvs_gamma0":2.5,"note":"All production MV strategies are numerically recalibrated to the MVS eta0=0 target. The MVS baseline itself retains gamma0=2.5."}
(OUT/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print(cal.to_string(index=False),flush=True)
