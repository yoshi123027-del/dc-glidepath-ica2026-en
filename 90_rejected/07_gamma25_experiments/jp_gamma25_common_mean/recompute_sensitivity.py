from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_OUT = ROOT / "results" / "jp_gamma25_common_mean"
OUT = BASE_OUT / "sensitivity_N80"
FIG = ROOT / "figs" / "jp_gamma25_common_mean" / "sensitivity_N80"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
coef = json.loads((BASE_OUT / "refined_equal_mean_coefficients.json").read_text(encoding="utf-8"))

src = ROOT / "scripts" / "04_sensitivity" / "low_balance_refined_sensitivity_20260714.py"
spec = importlib.util.spec_from_file_location("sens", src)
sens = importlib.util.module_from_spec(spec); sys.modules["sens"] = sens; spec.loader.exec_module(sens)

# As in the original paper, the directional N=80 screen holds the monthly
# baseline calibration coefficients fixed while varying market/plan inputs.
base = sens.Config(
    D=0.0,
    gamma_p=float(coef["gamma_p"]),
    gamma_d=float(coef["gamma_d"]),
    gamma_c=float(coef["gamma_c"]),
    gamma_0=float(coef["rho_d"]),
    theta_cp=float(coef["theta_cp"]),
)
constant = sens.contribution_steps(base, "constant")
d_fv = sens.safe_asset_future_value(constant, base.r, base.dt)
scenarios: Dict[str, Tuple[str, sens.Config, str]] = {
    "baseline": ("Baseline D_T=0", base, "constant"),
    "D_alt": (f"D_T=safe-asset FV={d_fv:.2f}", replace(base, D=d_fv), "constant"),
    "r_low": ("r=0.005", replace(base, r=0.005), "constant"),
    "r_high": ("r=0.025", replace(base, r=0.025), "constant"),
    "mu_low": ("mu=0.045", replace(base, mu=0.045), "constant"),
    "mu_high": ("mu=0.065", replace(base, mu=0.065), "constant"),
    "sigma_low": ("sigma=0.14", replace(base, sigma=0.14), "constant"),
    "sigma_high": ("sigma=0.22", replace(base, sigma=0.22), "constant"),
    "contrib_constant": ("Constant", base, "constant"),
    "contrib_linear": ("Linear increase", base, "linear"),
    "contrib_quadratic": ("Quadratic increase", base, "quadratic"),
}
results={}; rows=[]
for key,(label,cfg,profile) in scenarios.items():
    print(f"Solving {key}: {label}",flush=True)
    result=sens.solve_scenario(cfg,profile); results[key]=result
    for strategy in ["PCMV","DOMV","cTCMV","dTCMV","CP"]:
        st=result[strategy]["stats"]; glide=result[strategy]["glide"]
        rows.append({"scenario":key,"label":label,"profile":profile,"strategy":strategy,**st,"mean_glide":float(np.mean(glide))})
pd.DataFrame(rows).to_csv(OUT/"sensitivity_summary.csv",index=False)
np.savez_compressed(OUT/"sensitivity_glidepaths.npz",decision_times=np.arange(base.n_steps)*base.dt,scenario_keys=np.array(list(scenarios.keys())),strategies=np.array(["PCMV","DOMV","cTCMV","dTCMV","CP"]),glides=np.stack([np.stack([results[k][s]["glide"] for s in ["PCMV","DOMV","cTCMV","dTCMV","CP"]]) for k in scenarios]))
sens.plot_panels(scenarios,results,["baseline","D_alt"],FIG/"fig_D_sensitivity_glidepaths_N80.png","Sensitivity to D_T")
sens.plot_panels(scenarios,results,["r_low","baseline","r_high"],FIG/"fig_r_sensitivity_glidepaths_N80.png","Sensitivity to r")
sens.plot_panels(scenarios,results,["mu_low","baseline","mu_high"],FIG/"fig_mu_sensitivity_glidepaths_N80.png","Sensitivity to mu")
sens.plot_panels(scenarios,results,["sigma_low","baseline","sigma_high"],FIG/"fig_sigma_sensitivity_glidepaths_N80.png","Sensitivity to sigma")
sens.plot_panels(scenarios,results,["contrib_constant","contrib_linear","contrib_quadratic"],FIG/"fig_contrib_profile_sensitivity_glidepaths_N80.png","Contribution-profile sensitivity with fixed total contributions")
print("done",flush=True)
