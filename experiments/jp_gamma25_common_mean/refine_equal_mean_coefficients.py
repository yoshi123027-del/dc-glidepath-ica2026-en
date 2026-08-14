from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jp_gamma25_common_mean"
OUT.mkdir(parents=True, exist_ok=True)

# MVS anchor mean.
sys.path.insert(0, str(ROOT / "scripts" / "01_solvers"))
import dtcmv_mvs_solver_20260713 as mvs
base_mvs = mvs.solve_case(replace(mvs.Config(), gamma0=2.5, eta0=0.0))
TARGET = float(base_mvs["stats"]["mean"])

# Production monthly MV implementation.
roll_path = ROOT / "scripts" / "03_rolling" / "recompute_d0_rolling.py"
spec = importlib.util.spec_from_file_location("roll_refine", roll_path)
roll = importlib.util.module_from_spec(spec)
sys.modules["roll_refine"] = roll
spec.loader.exec_module(roll)
pc = roll.pcmod
family = roll.pcres["family"]
xg_pc = family["x_grid"]


def terminal_mean(grid, pmf):
    p = pmf[-1] / pmf[-1].sum()
    return float(p @ grid)


def eval_pcmv(gamma):
    cfg = replace(roll.pcfg, gamma_p=float(gamma))
    info = pc.find_pcmv_target(cfg, family)
    z, exact, residual = pc.refine_pcmv_target(cfg, family, info["z_root"], max_iter=4)
    policy = exact["policy"][0].astype(float)
    fwd = pc.forward_distribution(cfg, xg_pc, policy, exact["gh_x"], exact["gh_w"])
    return terminal_mean(xg_pc, fwd["pmf"]), z, residual


def eval_dtcmv(rho):
    M, Q, P = roll.solve_tc(
        1, roll.N, roll.xg, roll.H, roll.dt, roll.r, roll.beta, roll.sigma,
        roll.c, roll.D, roll.gamma_c, float(rho), roll.n_controls, roll.gh_x, roll.gh_w
    )
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    return terminal_mean(roll.xg, pmf)


def bisect_decreasing(eval_fn, lo, hi, tol=0.002, max_iter=18):
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = eval_fn(mid)
        mean = val[0] if isinstance(val, tuple) else val
        err = mean - TARGET
        if best is None or abs(err) < abs(best[0]):
            best = (err, mid, val)
        if abs(err) <= tol:
            break
        if err > 0:
            lo = mid
        else:
            hi = mid
    return best

p_best = bisect_decreasing(eval_pcmv, 0.10, 0.22)
d_best = bisect_decreasing(eval_dtcmv, 1.5, 4.0)

out = {
    "target_mean": TARGET,
    "gamma_p": float(p_best[1]),
    "pcmv_mean": float(p_best[2][0]),
    "pcmv_target_z": float(p_best[2][1]),
    "pcmv_fixed_point_residual": float(p_best[2][2]),
    "rho_d": float(d_best[1]),
    "dtcmv_mean": float(d_best[2]),
    "pcmv_abs_error": abs(float(p_best[2][0]) - TARGET),
    "dtcmv_abs_error": abs(float(d_best[2]) - TARGET),
}
(OUT / "refined_equal_mean_coefficients.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
