from __future__ import annotations
import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

SENSITIVITY_DIR = Path(__file__).resolve().parents[1] / "04_sensitivity"
sys.path.insert(0, str(SENSITIVITY_DIR))
import low_balance_refined_sensitivity_20260714 as m

key=sys.argv[1]
out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
base=m.Config(D=0.0)
constant=m.contribution_steps(base,'constant')
d_fv=m.safe_asset_future_value(constant,base.r,base.dt)
scenarios={
'baseline':('Baseline D_T=0',base,'constant'),
'D_alt':(f'D_T=safe-asset FV={d_fv:.2f}',replace(base,D=d_fv),'constant'),
'r_low':('r=0.005',replace(base,r=0.005),'constant'),
'r_high':('r=0.025',replace(base,r=0.025),'constant'),
'mu_low':('mu=0.045',replace(base,mu=0.045),'constant'),
'mu_high':('mu=0.065',replace(base,mu=0.065),'constant'),
'sigma_low':('sigma=0.14',replace(base,sigma=0.14),'constant'),
'sigma_high':('sigma=0.22',replace(base,sigma=0.22),'constant'),
'contrib_constant':('Constant',base,'constant'),
'contrib_linear':('Linear increase',base,'linear'),
'contrib_quadratic':('Quadratic increase',base,'quadratic'),
}
label,cfg,profile=scenarios[key]
r=m.solve_scenario(cfg,profile)
strategies=['PCMV','DOMV','cTCMV','dTCMV','CP']
np.savez_compressed(out/f'{key}.npz',decision_times=np.arange(cfg.n_steps)*cfg.dt,strategies=np.array(strategies),glides=np.stack([r[s]['glide'] for s in strategies]))
rows=[]
for s in strategies:
 g=np.asarray(r[s]['glide']); rows.append({'scenario':key,'label':label,'profile':profile,'strategy':s,**r[s]['stats'],'mean_glide':float(g.mean()),'initial_glide':float(g[0]),'last_decision_glide':float(g[-1])})
pd.DataFrame(rows).to_csv(out/f'{key}.csv',index=False)
print(key,'done')
