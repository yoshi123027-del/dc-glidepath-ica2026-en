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
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Proposed paper-wide anchor: historical dTCMV-MVS, gamma0=2.5, eta0=0
# ------------------------------------------------------------
mvs_dir = ROOT / "scripts" / "01_solvers"
sys.path.insert(0, str(mvs_dir))
import dtcmv_mvs_solver_20260713 as mvs

mvs_base_cfg = replace(mvs.Config(), gamma0=2.5, eta0=0.0)
mvs_base = mvs.solve_case(mvs_base_cfg)
TARGET = float(mvs_base["stats"]["mean"])

# ------------------------------------------------------------
# Load the paper's monthly MV implementation and its target family.
# Importing this module reproduces the existing baseline once, then we reuse
# its compiled kernels and target family for the alternative calibration.
# ------------------------------------------------------------
roll_path = ROOT / "scripts" / "03_rolling" / "recompute_d0_rolling.py"
spec = importlib.util.spec_from_file_location("rollmod", roll_path)
roll = importlib.util.module_from_spec(spec)
sys.modules["rollmod"] = roll
spec.loader.exec_module(roll)
pc = roll.pcmod
family = roll.pcres["family"]
xg_pc = family["x_grid"]


def terminal_stats_with_ucvar(grid, pmf):
    st = roll.stats_from_pmf(grid, pmf[-1], 0.0)
    p = pmf[-1] / pmf[-1].sum()
    rem = 0.05
    total = 0.0
    for v, w in zip(grid[::-1], p[::-1]):
        take = min(rem, w)
        total += take * v
        rem -= take
        if rem <= 1e-15:
            break
    st["ucvar95"] = total / 0.05
    return st


def pcmv_at_target(target_mean):
    # For PCMV, z - E[W^z] = 1/gamma.  First invert E[W^z]=target_mean
    # on the already-solved target family, then obtain gamma algebraically.
    M0 = pc.interp_x_vector(xg_pc, family["M"][:, 0, :], roll.pcfg.x0)
    targets = family["targets"]
    order = np.argsort(M0)
    z = float(np.interp(target_mean, M0[order], targets[order]))
    gamma = 1.0 / (z - target_mean)
    cfg = replace(roll.pcfg, gamma_p=gamma)
    exact = pc.solve_exact_target(cfg, z)
    policy = exact["policy"][0].astype(float)
    fwd = pc.forward_distribution(cfg, xg_pc, policy, exact["gh_x"], exact["gh_w"])
    st = terminal_stats_with_ucvar(xg_pc, fwd["pmf"])
    return gamma, z, policy, fwd, st


def eval_domv(gamma):
    cfg = replace(roll.pcfg, gamma_d=float(gamma))
    d = pc.build_domv_policy(cfg, family)
    fwd = pc.forward_distribution(cfg, xg_pc, d["policy"], family["gh_x"], family["gh_w"])
    st = terminal_stats_with_ucvar(xg_pc, fwd["pmf"])
    return d["policy"], fwd, st


def eval_ctcmv(gamma):
    M, Q, P = roll.solve_tc(
        0, roll.N, roll.xg, roll.H, roll.dt, roll.r, roll.beta, roll.sigma,
        roll.c, roll.D, float(gamma), 2.5, roll.n_controls, roll.gh_x, roll.gh_w
    )
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    st = terminal_stats_with_ucvar(roll.xg, pmf)
    return M, Q, P, pmf, glide, upper, st


def eval_dtcmv_rho25():
    M, Q, P = roll.solve_tc(
        1, roll.N, roll.xg, roll.H, roll.dt, roll.r, roll.beta, roll.sigma,
        roll.c, roll.D, roll.gamma_c, 2.5, roll.n_controls, roll.gh_x, roll.gh_w
    )
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    st = terminal_stats_with_ucvar(roll.xg, pmf)
    return M, Q, P, pmf, glide, upper, st


def eval_cp(theta):
    P = np.full((roll.N, roll.n_x), float(theta))
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    st = terminal_stats_with_ucvar(roll.xg, pmf)
    return P, pmf, glide, upper, st


def bisection_mean(eval_fn, lo, hi, target, tol=0.015, max_iter=18, mean_index=-1):
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        res = eval_fn(mid)
        st = res[mean_index]
        err = float(st["mean"] - target)
        if best is None or abs(err) < abs(best[0]):
            best = (err, mid, res)
        if abs(err) <= tol:
            break
        # More aversion -> lower mean for the gamma calibrations.
        if err > 0:
            lo = mid
        else:
            hi = mid
    return best[1], best[2]

# PCMV exact inversion.
gamma_p, z_p, P_p, fwd_p, st_p = pcmv_at_target(TARGET)

# DOMV / cTCMV calibrations.
gamma_d, domv_res = bisection_mean(eval_domv, 0.02, 0.8, TARGET, tol=0.015, max_iter=16)
P_D, fwd_D, st_D = domv_res

gamma_c, c_res = bisection_mean(eval_ctcmv, 0.02, 0.5, TARGET, tol=0.015, max_iter=16)
M_c, Q_c, P_c, pmf_c, glide_c, upper_c, st_c = c_res

# dTCMV parameter is fixed by design at rho_d = 2.5; do not retune it.
M_d, Q_d, P_d, pmf_d, glide_d, upper_d, st_d = eval_dtcmv_rho25()

# CP: more theta -> higher mean, so use its own bisection direction.
def calibrate_cp():
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        res = eval_cp(mid)
        err = float(res[-1]["mean"] - TARGET)
        if best is None or abs(err) < abs(best[0]):
            best = (err, mid, res)
        if abs(err) <= 0.015:
            break
        if err > 0:
            hi = mid
        else:
            lo = mid
    return best[1], best[2]

theta_cp, cp_res = calibrate_cp()
P_cp, pmf_cp, glide_cp, upper_cp, st_cp = cp_res

# PC/DOMV decision summaries on their own forward distributions.
glide_p, upper_p = roll.decision_summary_from_policy(P_p, fwd_p["pmf"], xg_pc, roll.x0)
glide_D, upper_D = roll.decision_summary_from_policy(P_D, fwd_D["pmf"], xg_pc, roll.x0)
pmf_p = fwd_p["pmf"]
pmf_D = fwd_D["pmf"]

# ------------------------------------------------------------
# Baseline summary and calibration table
# ------------------------------------------------------------
cal_rows = [
    {"strategy":"PCMV","parameter":"gamma_p","value":gamma_p,"target_mean":TARGET,"achieved_mean":st_p["mean"]},
    {"strategy":"DOMV","parameter":"gamma_d","value":gamma_d,"target_mean":TARGET,"achieved_mean":st_D["mean"]},
    {"strategy":"cTCMV","parameter":"gamma_c","value":gamma_c,"target_mean":TARGET,"achieved_mean":st_c["mean"]},
    {"strategy":"dTCMV","parameter":"rho_d","value":2.5,"target_mean":TARGET,"achieved_mean":st_d["mean"]},
    {"strategy":"CP","parameter":"theta_cp","value":theta_cp,"target_mean":TARGET,"achieved_mean":st_cp["mean"]},
]
cal = pd.DataFrame(cal_rows)
cal["absolute_mean_error"] = (cal["achieved_mean"] - TARGET).abs()
cal.to_csv(OUT / "mv_equal_mean_calibration.csv", index=False)

summary_rows = []
for name, st, glide, upper in [
    ("PCMV",st_p,glide_p,upper_p),("DOMV",st_D,glide_D,upper_D),
    ("cTCMV",st_c,glide_c,upper_c),("dTCMV",st_d,glide_d,upper_d),
    ("CP",st_cp,glide_cp,upper_cp),
]:
    summary_rows.append({"strategy":name,**st,"avg_glide":float(np.mean(glide)),"upper_bind":float(np.mean(upper)),"final_glide":float(glide[-1])})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "mv_baseline_summary.csv", index=False)

# ------------------------------------------------------------
# Main MV figures
# ------------------------------------------------------------
times = np.arange(roll.N) * roll.dt
fig, ax = plt.subplots(figsize=(9.2,5.6))
for name,g in [("PCMV",glide_p),("DOMV",glide_D),("cTCMV",glide_c),("dTCMV",glide_d),("CP",glide_cp)]:
    ax.plot(times,g,label=name,linewidth=1.8)
ax.set_xlabel("経過年")
ax.set_ylabel("確率質量加重リスク資産比率")
ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig_all_strategies_glidepaths.png",dpi=180); plt.close(fig)

series=[("PCMV",xg_pc,pmf_p[-1]),("DOMV",xg_pc,pmf_D[-1]),("cTCMV",roll.xg,pmf_c[-1]),("dTCMV",roll.xg,pmf_d[-1]),("CP",roll.xg,pmf_cp[-1])]
fig,ax=plt.subplots(figsize=(9.2,5.6))
for name,vals,p in series:
    ax.plot(vals,np.cumsum(p/p.sum()),label=name,linewidth=1.8)
ax.set_xlabel("終端DC資産"); ax.set_ylabel("CDF"); ax.set_xlim(0,220); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig_terminal_cdf.png",dpi=180); plt.close(fig)

ugrid=np.linspace(0,220,1300)
def deposit_uniform(vals, probs, grid):
    mass=np.zeros_like(grid); dx=grid[1]-grid[0]
    for v,p in zip(vals,probs):
        if p<=0: continue
        u=(v-grid[0])/dx
        if u<=0: mass[0]+=p
        elif u>=len(grid)-1: mass[-1]+=p
        else:
            j=int(math.floor(u)); lam=u-j
            mass[j]+=p*(1-lam); mass[j+1]+=p*lam
    return mass
fig,ax=plt.subplots(figsize=(9.2,5.6))
for name,vals,p in series:
    density=gaussian_filter1d(deposit_uniform(vals,p,ugrid),sigma=5,mode="nearest")/(ugrid[1]-ugrid[0])
    ax.plot(ugrid,density,label=name,linewidth=1.8)
ax.set_xlabel("終端DC資産"); ax.set_ylabel("密度"); ax.set_xlim(0,180); ax.grid(alpha=.25); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig_terminal_density.png",dpi=180); plt.close(fig)

# ------------------------------------------------------------
# Rolling conditional profiles
# ------------------------------------------------------------
M_p_eval,Q_p_eval=roll.policy_moments_pc(roll.N,xg_pc,P_p,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
M_D_eval,Q_D_eval=roll.policy_moments_pc(roll.N,xg_pc,P_D,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
M_c_eval,Q_c_eval=roll.policy_moments(roll.N,roll.xg,P_c,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
M_d_eval,Q_d_eval=roll.policy_moments(roll.N,roll.xg,P_d,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,0.0,roll.gh_x,roll.gh_w)
strategies={
    "PCMV":(xg_pc,P_p,pmf_p,M_p_eval,Q_p_eval),"DOMV":(xg_pc,P_D,pmf_D,M_D_eval,Q_D_eval),
    "cTCMV":(roll.xg,P_c,pmf_c,M_c_eval,Q_c_eval),"dTCMV":(roll.xg,P_d,pmf_d,M_d_eval,Q_d_eval),
}
rolling_rows=[]
for name,(grid_x,policy,pmf,Me,Qe) in strategies.items():
    for year in [0,10,20,30,35,39]:
        n=min(int(round(year/roll.dt)),roll.N-1)
        xm=roll.median_state(grid_x,pmf[n])
        risk=roll.interp_at(grid_x,policy[n],xm)
        pcnd,_,_=roll.forward_policy(roll.N,grid_x,xm,n,policy,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w)
        st=terminal_stats_with_ucvar(grid_x,pcnd)
        mb=roll.interp_at(grid_x,Me[n],xm); qb=roll.interp_at(grid_x,Qe[n],xm)
        rolling_rows.append({"strategy":name,"year":year,"median_x":xm,"mean":st["mean"],"stdev":st["stdev"],"skewness":st["skewness"],"q05":st["q05"],"q95":st["q95"],"cvar05":st["cvar05"],"ucvar95":st["ucvar95"],"risky_fraction":risk,"backward_mean":mb,"backward_stdev":math.sqrt(max(qb-mb*mb,0.0))})
rolling_df=pd.DataFrame(rolling_rows); rolling_df.to_csv(OUT/"mv_rolling_conditional.csv",index=False)

fig,axes=plt.subplots(3,1,figsize=(9.2,11.0),sharex=True)
for name in strategies:
    d=rolling_df[rolling_df.strategy==name]
    axes[0].plot(d.year,d["mean"],marker="o",label=name)
    axes[1].plot(d.year,d.stdev,marker="o",label=name)
    axes[2].plot(d.year,d.risky_fraction,marker="o",label=name)
axes[0].set_ylabel("条件付き終端平均"); axes[1].set_ylabel("条件付き標準偏差"); axes[2].set_ylabel("現在リスク資産比率"); axes[2].set_xlabel("経過年")
for ax in axes: ax.grid(alpha=.25)
axes[0].legend(ncol=4); fig.tight_layout(); fig.savefig(FIG/"fig_rolling_conditional.png",dpi=180); plt.close(fig)

# ------------------------------------------------------------
# Clipped-unconstrained vs directly constrained baseline
# ------------------------------------------------------------
def dollars_to_fraction(grid,dollars):
    clipped=np.minimum(np.maximum(dollars,0.0),grid[None,:])
    out=np.zeros_like(clipped); pos=grid>0
    out[:,pos]=clipped[:,pos]/grid[None,pos]
    return np.clip(out,0,1)

A=roll.beta**2/roll.sigma**2
tau=roll.T-times
# Background wealth is identical on both grids.
y0=roll.x0+roll.H[0]
z_uncon=y0*math.exp(roll.r*roll.T)+math.exp(A*roll.T)/gamma_p
pc_des=z_uncon*np.exp(-roll.r*tau)[:,None]-(xg_pc[None,:]+roll.H[:-1,None])
P_p_clip=dollars_to_fraction(xg_pc,roll.beta/roll.sigma**2*pc_des)
P_D_clip=dollars_to_fraction(xg_pc,np.broadcast_to((roll.beta/(roll.sigma**2*gamma_d)*np.exp((A-roll.r)*tau))[:,None],(roll.N,len(xg_pc))).copy())
P_c_clip=dollars_to_fraction(roll.xg,np.broadcast_to((roll.beta/(roll.sigma**2*gamma_c)*np.exp(-roll.r*tau))[:,None],(roll.N,len(roll.xg))).copy())

def theta_dtcmv(rho):
    def rhs(_t,state):
        i1,i2=state
        theta=A/(rho*roll.beta)*(math.exp(-i1)+rho*math.exp(-i2)-rho)
        return np.array([-(roll.r+roll.beta*theta-roll.sigma**2*theta**2),-(roll.sigma**2*theta**2)])
    sol=solve_ivp(rhs,(roll.T,0.0),[0.0,0.0],method="DOP853",rtol=1e-10,atol=1e-12,dense_output=True)
    i1,i2=sol.sol(times)
    return A/(rho*roll.beta)*(np.exp(-i1)+rho*np.exp(-i2)-rho)
theta_rho=theta_dtcmv(2.5)
P_d_clip=dollars_to_fraction(roll.xg,theta_rho[:,None]*(roll.xg[None,:]+roll.H[:-1,None]))

clip_defs=[("PCMV",xg_pc,P_p,pmf_p,glide_p,P_p_clip),("DOMV",xg_pc,P_D,pmf_D,glide_D,P_D_clip),("cTCMV",roll.xg,P_c,pmf_c,glide_c,P_c_clip),("dTCMV",roll.xg,P_d,pmf_d,glide_d,P_d_clip)]
clip_rows=[]
fig,axes=plt.subplots(2,2,figsize=(11.2,8.2),sharex=True)
for ax,(name,grid_x,Pstrict,pmf_strict,gstrict,Pclip) in zip(axes.ravel(),clip_defs):
    pcpmf,cg,cup=roll.forward_policy(roll.N,grid_x,roll.x0,0,Pclip,roll.dt,roll.r,roll.beta,roll.sigma,roll.c,roll.gh_x,roll.gh_w)
    st_clip=terminal_stats_with_ucvar(grid_x,pcpmf)
    st_strict=terminal_stats_with_ucvar(grid_x,pmf_strict)
    clip_rows.append({"strategy":name,"glide_mae":float(np.mean(np.abs(cg-gstrict))),"glide_max_abs":float(np.max(np.abs(cg-gstrict))),"mean_clip_minus_strict":st_clip["mean"]-st_strict["mean"],"q05_clip_minus_strict":st_clip["q05"]-st_strict["q05"],"q95_clip_minus_strict":st_clip["q95"]-st_strict["q95"],"cvar05_clip_minus_strict":st_clip["cvar05"]-st_strict["cvar05"]})
    ax.plot(times,gstrict,label="厳密制約",linewidth=1.8); ax.plot(times,cg,"--",label="クリップ",linewidth=1.6)
    ax.set_title(name); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.set_xlabel("経過年"); ax.set_ylabel("リスク資産比率")
axes[0,0].legend(); fig.tight_layout(); fig.savefig(FIG/"fig_strict_vs_clip.png",dpi=180); plt.close(fig)
pd.DataFrame(clip_rows).to_csv(OUT/"mv_strict_vs_clip_summary.csv",index=False)

# dTCMV U-shape decomposition at median states.
decomp=[]
for year in [0,10,20,30,35,39]:
    n=min(int(round(year/roll.dt)),roll.N-1)
    xm=roll.median_state(roll.xg,pmf_d[n])
    g=roll.interp_at(roll.xg,P_d[n],xm)
    Hn=roll.H[n]
    decomp.append({"year":year,"median_x":xm,"H_over_x":Hn/max(xm,1e-14),"Gamma":2.5/max(xm+Hn,1e-14),"theta_unconstrained":theta_rho[n],"strict_risky_fraction":g,"strict_total_wealth_fraction":g*xm/max(xm+Hn,1e-14)})
pd.DataFrame(decomp).to_csv(OUT/"dtcmv_u_decomposition.csv",index=False)

# ------------------------------------------------------------
# MVS fixed-gamma and equal-mean analyses, now using the SAME paper-wide target.
# ------------------------------------------------------------
fixed_etas=[0.0,0.5,1.0,2.0]
fixed=[]
maps=mvs_base["maps"]
for eta in fixed_etas:
    rr=mvs.solve_case(replace(mvs_base_cfg,eta0=eta),maps=maps)
    fixed.append(rr)

cal_etas=[0.0,1.0,2.0,4.0,8.0]
cal_mvs=[]
for eta in cal_etas:
    if eta==0.0:
        rr=mvs_base
    else:
        rr=mvs.calibrate_gamma(mvs_base_cfg,eta,TARGET,maps,low=.05,high=20.0,tol=.03,max_iter=20)
    cal_mvs.append(rr)

fixed_rows=[]
for eta,rr in zip(fixed_etas,fixed):
    fixed_rows.append({"eta0":eta,"gamma0":rr["cfg"].gamma0,**rr["stats"],**rr["diagnostics"],"final_glide":float(rr["glide"][-1])})
pd.DataFrame(fixed_rows).to_csv(OUT/"mvs_fixed_gamma_summary.csv",index=False)
cal_rows2=[]
for eta,rr in zip(cal_etas,cal_mvs):
    cal_rows2.append({"eta0":eta,"gamma0":rr["cfg"].gamma0,**rr["stats"],**rr["diagnostics"],"final_glide":float(rr["glide"][-1])})
pd.DataFrame(cal_rows2).to_csv(OUT/"mvs_equal_mean_summary.csv",index=False)

mvs_roll=[]
for eta,rr in zip(cal_etas,cal_mvs):
    d=mvs.rolling_common_state(rr); d.insert(0,"eta0",eta); d.insert(1,"gamma0",rr["cfg"].gamma0); mvs_roll.append(d)
pd.concat(mvs_roll,ignore_index=True).to_csv(OUT/"mvs_rolling_summary.csv",index=False)

fig,ax=plt.subplots(figsize=(9.2,5.6))
for eta,rr in zip(fixed_etas,fixed): ax.plot(times,rr["glide"],label=fr"$\eta_0={eta:g}$")
ax.set_xlabel("経過年"); ax.set_ylabel("確率質量加重リスク資産比率"); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/"fig_mvs_fixed_gamma25.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(9.2,5.6))
for eta,rr in zip(cal_etas,cal_mvs): ax.plot(times,rr["glide"],label=fr"$\eta_0={eta:g},\ \gamma_0={rr['cfg'].gamma0:.3f}$")
ax.set_xlabel("経過年"); ax.set_ylabel("確率質量加重リスク資産比率"); ax.set_ylim(0,1.04); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/"fig_mvs_equal_mean.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(9.2,5.6))
for eta,rr in zip(cal_etas,cal_mvs):
    vals=rr["x_grid"]; ax.plot(vals,np.cumsum(rr["pmf"][-1]/rr["pmf"][-1].sum()),label=fr"$\eta_0={eta:g}$")
ax.set_xlabel("終端DC資産"); ax.set_ylabel("CDF"); ax.set_xlim(0,180); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(FIG/"fig_mvs_equal_mean_cdf.png",dpi=180); plt.close(fig)

# Annual glide data for easy downstream manuscript plotting.
annual_idx=np.arange(0,roll.N,12)
annual=pd.DataFrame({"year":times[annual_idx]})
for name,g in [("PCMV",glide_p),("DOMV",glide_D),("cTCMV",glide_c),("dTCMV",glide_d),("CP",glide_cp)]: annual[name]=g[annual_idx]
for eta,rr in zip(fixed_etas,fixed): annual[f"MVS_fixed_eta_{eta:g}"]=rr["glide"][annual_idx]
annual.to_csv(OUT/"annual_glidepaths.csv",index=False)

metadata={
    "paper_wide_target_mean":TARGET,
    "target_definition":"dTCMV-MVS base-grid mean at gamma0=2.5, eta0=0",
    "mvs_base_grid":{"n_steps":480,"n_x":151,"n_controls":25,"n_gh":5},
    "mv_grid":{"n_steps":480,"n_x":151,"n_controls":15,"n_gh":5,"parabolic_control_refinement":True},
    "mv_calibration":{"gamma_p":gamma_p,"pcmv_target_z":z_p,"gamma_d":gamma_d,"gamma_c":gamma_c,"rho_d":2.5,"theta_cp":theta_cp},
    "dtcmv_numerical_mean_gap_vs_mvs":float(st_d["mean"]-TARGET),
    "note":"rho_d=gamma0=2.5 is held fixed to preserve theoretical nesting; the small mean gap is numerical because the MV solver uses 15 controls plus parabolic refinement whereas the MVS solver uses 25 controls without refinement.",
}
(OUT/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

# Save arrays for follow-up analyses.
np.savez_compressed(OUT/"mv_policy_arrays.npz",
    decision_times=times,xg_pc=xg_pc,xg_tc=roll.xg,
    pcmv_policy=P_p,domv_policy=P_D,ctcmv_policy=P_c,dtcmv_policy=P_d,
    pcmv_pmf=pmf_p,domv_pmf=pmf_D,ctcmv_pmf=pmf_c,dtcmv_pmf=pmf_d,cp_pmf=pmf_cp,
    pcmv_glide=glide_p,domv_glide=glide_D,ctcmv_glide=glide_c,dtcmv_glide=glide_d,cp_glide=glide_cp)

print("TARGET",TARGET)
print(cal.to_string(index=False))
print(summary.to_string(index=False))
