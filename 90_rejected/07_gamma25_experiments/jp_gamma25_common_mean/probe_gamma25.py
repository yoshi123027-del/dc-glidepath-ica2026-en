from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jp_gamma25_common_mean"
OUT.mkdir(parents=True, exist_ok=True)

# Historical MVS core: this is the proposed paper-wide normalization anchor.
mvs_dir = ROOT / "scripts" / "01_solvers"
sys.path.insert(0, str(mvs_dir))
import dtcmv_mvs_solver_20260713 as mvs

mvs_cfg = replace(mvs.Config(), gamma0=2.5, eta0=0.0)
mvs_res = mvs.solve_case(mvs_cfg)
target = float(mvs_res["stats"]["mean"])

# Load the production monthly MV implementation used in Section 6.
roll_path = ROOT / "scripts" / "03_rolling" / "recompute_d0_rolling.py"
spec = importlib.util.spec_from_file_location("rollmod", roll_path)
roll = importlib.util.module_from_spec(spec)
sys.modules["rollmod"] = roll
spec.loader.exec_module(roll)

# Re-solve production dTCMV at exactly rho_d = 2.5.
Md, Qd, Pd = roll.solve_tc(
    1, roll.N, roll.xg, roll.H, roll.dt, roll.r, roll.beta, roll.sigma,
    roll.c, roll.D, roll.gamma_c, 2.5, roll.n_controls, roll.gh_x, roll.gh_w
)
pmf_d, glide_d, upper_d = roll.forward_policy(
    roll.N, roll.xg, roll.x0, 0, Pd, roll.dt, roll.r, roll.beta,
    roll.sigma, roll.c, roll.gh_x, roll.gh_w
)
st_d = roll.stats_from_pmf(roll.xg, pmf_d[-1], 0.0)

mvs_glide = mvs_res["glide"]
n = min(len(mvs_glide), len(glide_d))
diff = glide_d[:n] - mvs_glide[:n]

out = {
    "proposed_target_mean_mvs_gamma2p5_eta0": target,
    "mvs_gamma2p5_eta0": {
        **{k: float(v) for k, v in mvs_res["stats"].items()},
        "final_glide": float(mvs_glide[-1]),
        "mean_glide": float(mvs_res["diagnostics"]["mean_abs_glide"]),
    },
    "production_mv_dtcmv_rho2p5": {
        **{k: float(v) for k, v in st_d.items()},
        "final_glide": float(glide_d[-1]),
        "mean_glide": float(glide_d.mean()),
    },
    "production_mv_minus_mvs": {
        "mean_gap": float(st_d["mean"] - target),
        "glide_mean_abs_gap": float(abs(diff).mean()),
        "glide_max_abs_gap": float(abs(diff).max()),
        "final_glide_gap": float(diff[-1]),
    },
}
(OUT / "gamma25_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
