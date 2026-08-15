from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SOLVER_DIR = Path(__file__).resolve().parents[1] / "01_solvers"
sys.path.insert(0, str(SOLVER_DIR))
import dtcmv_mvs_solver_20260713 as m

kind=sys.argv[1]; eta=float(sys.argv[2]); gamma=float(sys.argv[3]); out=Path(sys.argv[4]); out.mkdir(parents=True,exist_ok=True)
cfg=m.Config(eta0=eta,gamma0=gamma)
r=m.solve_case(cfg)
np.savez_compressed(out/f'{kind}_{eta:g}.npz',decision_times=np.arange(cfg.n_steps)*cfg.dt,x_grid=r['x_grid'],policy=r['policy'],pmf=r['pmf'],glide=r['glide'],upper=r['upper'])
pd.DataFrame([{ 'kind':kind,'eta0':eta,'gamma0':gamma,**r['stats'],**r['diagnostics']}]).to_csv(out/f'{kind}_{eta:g}.csv',index=False)
print(kind,eta,gamma,'done',r['stats']['mean'],r['diagnostics']['mean_abs_glide'])
