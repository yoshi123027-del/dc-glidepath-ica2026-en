from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

SOLVER_DIR = Path(__file__).resolve().parents[1] / "01_solvers"
sys.path.insert(0, str(SOLVER_DIR))
import dtcmv_mvs_solver_20260713 as m

base=replace(m.Config(), n_x=251, n_controls=41, n_gh=7)
baseline=m.solve_case(base)
target=baseline['stats']['mean']; maps=baseline['maps']
rows=[]
results=[]
for eta in [0.0,1.0,2.0,4.0,8.0]:
    if eta==0:
        r=baseline
    else:
        r=m.calibrate_gamma(base,eta,target,maps,low=0.2,high=20.0,tol=0.04,max_iter=14)
    results.append(r)
    rows.append({'eta0':eta,'gamma0':r['cfg'].gamma0,**r['stats'],**r['diagnostics']})
    print('eta',eta,'gamma',r['cfg'].gamma0,r['stats'],flush=True)
pd.DataFrame(rows).to_csv(m.RES/'dtcmv_mvs_equal_mean_refined.csv',index=False)
