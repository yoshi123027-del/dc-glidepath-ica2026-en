from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jp_gamma25_common_mean"
OUT.mkdir(parents=True, exist_ok=True)

# Paper-wide anchor: dTCMV-MVS on its baseline grid with gamma0=2.5 and eta0=0.
sys.path.insert(0, str(ROOT / "scripts" / "01_solvers"))
import dtcmv_mvs_solver_20260713 as mvs
base_mvs = mvs.solve_case(replace(mvs.Config(), gamma0=2.5, eta0=0.0))
TARGET = float(base_mvs["stats"]["mean"])

# Production monthly MV implementation used in Section 6.  The module name is
# deliberately kept as rollmod because Numba's on-disk cache records it.
roll_path = ROOT / "scripts" / "03_rolling" / "recompute_d0_rolling.py"
spec = importlib.util.spec_from_file_location("rollmod", roll_path)
roll = importlib.util.module_from_spec(spec)
sys.modules["rollmod"] = roll
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


def eval_domv(gamma):
    cfg = replace(roll.pcfg, gamma_d=float(gamma))
    d = pc.build_domv_policy(cfg, family)
    fwd = pc.forward_distribution(cfg, xg_pc, d["policy"], family["gh_x"], family["gh_w"])
    return terminal_mean(xg_pc, fwd["pmf"])


def eval_ctcmv(gamma):
    M, Q, P = roll.solve_tc(
        0, roll.N, roll.xg, roll.H, roll.dt, roll.r, roll.beta, roll.sigma,
        roll.c, roll.D, float(gamma), 2.5, roll.n_controls, roll.gh_x, roll.gh_w
    )
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    return terminal_mean(roll.xg, pmf)


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


def eval_cp(theta):
    P = np.full((roll.N, roll.n_x), float(theta))
    pmf, glide, upper = roll.forward_policy(
        roll.N, roll.xg, roll.x0, 0, P, roll.dt, roll.r, roll.beta,
        roll.sigma, roll.c, roll.gh_x, roll.gh_w
    )
    return terminal_mean(roll.xg, pmf)


def bisect_decreasing(eval_fn, lo, hi, tol=0.0015, max_iter=20):
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


def bisect_increasing(eval_fn, lo, hi, tol=0.0015, max_iter=20):
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
            hi = mid
        else:
            lo = mid
    return best

p_best = bisect_decreasing(eval_pcmv, 0.10, 0.22)
D_best = bisect_decreasing(eval_domv, 0.12, 0.35)
c_best = bisect_decreasing(eval_ctcmv, 0.07, 0.22)
d_best = bisect_decreasing(eval_dtcmv, 1.5, 4.0)
cp_best = bisect_increasing(eval_cp, 0.15, 0.40)

out = {
    "target_mean": TARGET,
    "gamma_p": float(p_best[1]),
    "pcmv_mean": float(p_best[2][0]),
    "pcmv_target_z": float(p_best[2][1]),
    "pcmv_fixed_point_residual": float(p_best[2][2]),
    "gamma_d": float(D_best[1]),
    "domv_mean": float(D_best[2]),
    "gamma_c": float(c_best[1]),
    "ctcmv_mean": float(c_best[2]),
    "rho_d": float(d_best[1]),
    "dtcmv_mean": float(d_best[2]),
    "theta_cp": float(cp_best[1]),
    "cp_mean": float(cp_best[2]),
}
for strategy, key in [("pcmv","pcmv_mean"),("domv","domv_mean"),("ctcmv","ctcmv_mean"),("dtcmv","dtcmv_mean"),("cp","cp_mean")]:
    out[f"{strategy}_abs_error"] = abs(float(out[key]) - TARGET)

(OUT / "refined_equal_mean_coefficients.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
