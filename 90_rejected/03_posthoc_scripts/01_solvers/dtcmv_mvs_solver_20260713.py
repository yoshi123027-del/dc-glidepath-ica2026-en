from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figs"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    T: float = 40.0
    n_steps: int = 480
    x0: float = 1.0 / 12.0
    x_max: float = 300.0
    n_x: int = 151
    x_power: float = 1.6
    n_controls: int = 25
    n_gh: int = 5
    r: float = 0.015
    mu: float = 0.055
    sigma: float = 0.18
    c: float = 1.0
    D: float = 0.0
    gamma0: float = 2.5
    eta0: float = 0.0

    @property
    def beta(self) -> float:
        return self.mu - self.r

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


def make_grid(cfg: Config) -> np.ndarray:
    u = np.linspace(0.0, 1.0, cfg.n_x)
    return cfg.x_max * u ** cfg.x_power


def gh_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = hermgauss(n)
    return np.sqrt(2.0) * h.astype(np.float64), (w / np.sqrt(np.pi)).astype(np.float64)


def future_contribution_pv(cfg: Config) -> np.ndarray:
    H = np.zeros(cfg.n_steps + 1)
    H[-1] = cfg.D
    gross = 1.0 + cfg.r * cfg.dt
    for n in range(cfg.n_steps - 1, -1, -1):
        H[n] = (cfg.c * cfg.dt + H[n + 1]) / gross
    return H


@njit(cache=True)
def locate(xg: np.ndarray, x: float) -> Tuple[int, float]:
    nx = xg.size
    if x <= xg[0]:
        return 0, 0.0
    if x >= xg[nx - 1]:
        return nx - 2, (x - xg[nx - 2]) / (xg[nx - 1] - xg[nx - 2])
    lo = 0
    hi = nx - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xg[mid] <= x:
            lo = mid
        else:
            hi = mid
    lam = (x - xg[lo]) / (xg[lo + 1] - xg[lo])
    return lo, lam


@njit(cache=True)
def interp(v: np.ndarray, j: int, lam: float) -> float:
    return v[j] * (1.0 - lam) + v[j + 1] * lam


@njit(cache=True)
def precompute_transitions(
    xg: np.ndarray,
    n_controls: int,
    gh_x: np.ndarray,
    dt: float,
    r: float,
    beta: float,
    sigma: float,
    c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nx = xg.size
    ng = gh_x.size
    lo = np.empty((nx, n_controls, ng), dtype=np.int32)
    lam = np.empty((nx, n_controls, ng), dtype=np.float64)
    da = 1.0 / (n_controls - 1)
    rootdt = math.sqrt(dt)
    for i in range(nx):
        x = xg[i]
        for j in range(n_controls):
            a = j * da
            pi = a * x
            drift = (r * x + c + beta * pi) * dt
            sd = sigma * pi * rootdt
            for k in range(ng):
                xp = x + drift + sd * gh_x[k]
                if xp < 0.0:
                    xp = 0.0
                jj, ll = locate(xg, xp)
                lo[i, j, k] = jj
                lam[i, j, k] = ll
    return lo, lam


@njit(cache=True)
def expected_raw3_from_map(
    Fnext: np.ndarray,
    Gnext: np.ndarray,
    Knext: np.ndarray,
    lo_map: np.ndarray,
    lam_map: np.ndarray,
    i: int,
    j: int,
    gh_w: np.ndarray,
) -> Tuple[float, float, float]:
    f = 0.0
    g = 0.0
    k3 = 0.0
    for q in range(gh_w.size):
        jj = lo_map[i, j, q]
        ll = lam_map[i, j, q]
        w = gh_w[q]
        f += w * interp(Fnext, jj, ll)
        g += w * interp(Gnext, jj, ll)
        k3 += w * interp(Knext, jj, ll)
    return f, g, k3


@njit(cache=True)
def solve_mvs_numba(
    n_steps: int,
    xg: np.ndarray,
    H: np.ndarray,
    D: float,
    gamma0: float,
    eta0: float,
    n_controls: int,
    gh_w: np.ndarray,
    lo_map: np.ndarray,
    lam_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = xg.size
    F = np.empty((n_steps + 1, nx), dtype=np.float64)
    G = np.empty((n_steps + 1, nx), dtype=np.float64)
    K = np.empty((n_steps + 1, nx), dtype=np.float64)
    P = np.empty((n_steps, nx), dtype=np.float64)
    objective = np.empty((n_steps, nx), dtype=np.float64)
    concavity = np.empty((n_steps, nx), dtype=np.float64)
    da = 1.0 / (n_controls - 1)

    for i in range(nx):
        w = xg[i] + D
        F[n_steps, i] = w
        G[n_steps, i] = w * w
        K[n_steps, i] = w * w * w

    for n in range(n_steps - 1, -1, -1):
        Fnext = F[n + 1]
        Gnext = G[n + 1]
        Knext = K[n + 1]
        for i in range(nx):
            x = xg[i]
            wealth_base = max(x + H[n], 1e-12)
            gamma = gamma0 / wealth_base
            eta = eta0 / (wealth_base * wealth_base)
            vals = np.empty(n_controls, dtype=np.float64)
            fs = np.empty(n_controls, dtype=np.float64)
            gs = np.empty(n_controls, dtype=np.float64)
            ks = np.empty(n_controls, dtype=np.float64)
            best_j = 0
            best_val = -1e300
            for j in range(n_controls):
                f, g, k3 = expected_raw3_from_map(Fnext, Gnext, Knext, lo_map, lam_map, i, j, gh_w)
                var = g - f * f
                if var < 0.0 and var > -1e-8:
                    var = 0.0
                cm3 = k3 - 3.0 * f * g + 2.0 * f * f * f
                val = f - 0.5 * gamma * var + (eta / 3.0) * cm3
                vals[j] = val
                fs[j] = f
                gs[j] = g
                ks[j] = k3
                if val > best_val:
                    best_val = val
                    best_j = j

            # The MVS local objective need not be concave.  We therefore select
            # the global maximum over the full control grid and only record a
            # local second-difference diagnostic; no parabolic refinement is used.
            P[n, i] = best_j * da
            F[n, i] = fs[best_j]
            G[n, i] = gs[best_j]
            K[n, i] = ks[best_j]
            objective[n, i] = best_val
            if 0 < best_j < n_controls - 1:
                concavity[n, i] = vals[best_j - 1] - 2.0 * vals[best_j] + vals[best_j + 1]
            else:
                concavity[n, i] = np.nan
    return F, G, K, P, objective, concavity


@njit(cache=True)
def deposit(xg: np.ndarray, p: np.ndarray, x: float, weight: float) -> None:
    if x <= xg[0]:
        p[0] += weight
        return
    if x >= xg[-1]:
        p[-1] += weight
        return
    j, lam = locate(xg, x)
    p[j] += weight * (1.0 - lam)
    p[j + 1] += weight * lam


@njit(cache=True)
def forward_policy(
    n_steps: int,
    xg: np.ndarray,
    x0: float,
    P: np.ndarray,
    dt: float,
    r: float,
    beta: float,
    sigma: float,
    c: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = xg.size
    p = np.zeros(nx, dtype=np.float64)
    deposit(xg, p, x0, 1.0)
    pmf = np.zeros((n_steps + 1, nx), dtype=np.float64)
    pmf[0] = p
    # Feedback controls live on the N decision intervals, not at terminal T.
    glide = np.zeros(n_steps, dtype=np.float64)
    upper = np.zeros(n_steps, dtype=np.float64)
    rootdt = math.sqrt(dt)
    for n in range(n_steps):
        if n == 0:
            j0, l0 = locate(xg, x0)
            pi_grid = P[n] * xg
            pi0 = interp(pi_grid, j0, l0)
            a0 = min(max(pi0 / max(x0, 1e-14), 0.0), 1.0)
            glide[n] = a0
            upper[n] = 1.0 if a0 >= 0.999 else 0.0
        else:
            for i in range(nx):
                glide[n] += p[i] * P[n, i]
                if P[n, i] >= 0.999:
                    upper[n] += p[i]
        pn = np.zeros(nx, dtype=np.float64)
        if n == 0:
            x = x0
            a = glide[0]
            pi = a * x
            drift = (r * x + c + beta * pi) * dt
            sd = sigma * pi * rootdt
            for q in range(gh_x.size):
                xp = x + drift + sd * gh_x[q]
                if xp < 0.0:
                    xp = 0.0
                deposit(xg, pn, xp, gh_w[q])
        else:
            for i in range(nx):
                if p[i] <= 0.0:
                    continue
                x = xg[i]
                a = P[n, i]
                pi = a * x
                drift = (r * x + c + beta * pi) * dt
                sd = sigma * pi * rootdt
                for q in range(gh_x.size):
                    xp = x + drift + sd * gh_x[q]
                    if xp < 0.0:
                        xp = 0.0
                    deposit(xg, pn, xp, p[i] * gh_w[q])
        total = pn.sum()
        if total > 0.0:
            pn /= total
        p = pn
        pmf[n + 1] = p
    return pmf, glide, upper


def quantile(values: np.ndarray, probs: np.ndarray, q: float) -> float:
    cdf = np.cumsum(probs / probs.sum())
    return float(values[min(np.searchsorted(cdf, q), len(values) - 1)])


def lower_cvar(values: np.ndarray, probs: np.ndarray, alpha: float = 0.05) -> float:
    p = probs / probs.sum()
    remaining = alpha
    total = 0.0
    for v, w in zip(values, p):
        take = min(remaining, w)
        total += take * v
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / alpha


def upper_cvar(values: np.ndarray, probs: np.ndarray, alpha: float = 0.05) -> float:
    p = probs / probs.sum()
    remaining = alpha
    total = 0.0
    for v, w in zip(values[::-1], p[::-1]):
        take = min(remaining, w)
        total += take * v
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / alpha


def distribution_stats(xg: np.ndarray, pmf: np.ndarray, D: float = 0.0) -> Dict[str, float]:
    values = xg + D
    p = pmf / pmf.sum()
    mean = float(p @ values)
    var = float(p @ ((values - mean) ** 2))
    sd = math.sqrt(max(var, 0.0))
    cm3 = float(p @ ((values - mean) ** 3))
    skew = cm3 / (sd ** 3 + 1e-30)
    return {
        "mean": mean,
        "stdev": sd,
        "variance": var,
        "third_central_moment": cm3,
        "skewness": skew,
        "q05": quantile(values, p, 0.05),
        "q50": quantile(values, p, 0.50),
        "q95": quantile(values, p, 0.95),
        "cvar05": lower_cvar(values, p, 0.05),
        "ucvar95": upper_cvar(values, p, 0.05),
    }


def solve_case(cfg: Config, maps=None) -> Dict[str, object]:
    xg = make_grid(cfg)
    gh_x, gh_w = gh_nodes_weights(cfg.n_gh)
    H = future_contribution_pv(cfg)
    if maps is None:
        lo_map, lam_map = precompute_transitions(
            xg, cfg.n_controls, gh_x, cfg.dt, cfg.r, cfg.beta, cfg.sigma, cfg.c
        )
    else:
        lo_map, lam_map = maps
    F, G, K, P, obj, conc = solve_mvs_numba(
        cfg.n_steps, xg, H, cfg.D, cfg.gamma0, cfg.eta0,
        cfg.n_controls, gh_w, lo_map, lam_map
    )
    pmf, glide, upper = forward_policy(
        cfg.n_steps, xg, cfg.x0, P, cfg.dt, cfg.r, cfg.beta,
        cfg.sigma, cfg.c, gh_x, gh_w
    )
    stats = distribution_stats(xg, pmf[-1], cfg.D)
    # Initial backward-forward moment residuals.
    f0 = float(np.interp(cfg.x0, xg, F[0]))
    g0 = float(np.interp(cfg.x0, xg, G[0]))
    k0 = float(np.interp(cfg.x0, xg, K[0]))
    values = xg + cfg.D
    pT = pmf[-1]
    diagnostics = {
        "F0_backward": f0,
        "F0_forward": float(pT @ values),
        "G0_backward": g0,
        "G0_forward": float(pT @ (values ** 2)),
        "K0_backward": k0,
        "K0_forward": float(pT @ (values ** 3)),
        "mean_abs_glide": float(np.mean(glide)),
        "mean_upper_binding": float(np.mean(upper)),
        "terminal_boundary_mass": float(pmf[-1, -1]),
    }
    return {
        "cfg": cfg, "x_grid": xg, "H": H, "gh_x": gh_x, "gh_w": gh_w,
        "maps": (lo_map, lam_map), "F": F, "G": G, "K": K,
        "policy": P, "objective": obj, "concavity": conc,
        "pmf": pmf, "glide": glide, "upper": upper,
        "stats": stats, "diagnostics": diagnostics,
    }


def calibrate_gamma(
    base_cfg: Config,
    eta0: float,
    target_mean: float,
    maps,
    low: float = 0.05,
    high: float = 20.0,
    tol: float = 0.03,
    max_iter: int = 20,
) -> Dict[str, object]:
    cache: Dict[float, Dict[str, object]] = {}

    def eval_gamma(g: float) -> Dict[str, object]:
        key = round(float(g), 10)
        if key not in cache:
            cache[key] = solve_case(Config(**{**asdict(base_cfg), "gamma0": float(g), "eta0": float(eta0)}), maps=maps)
        return cache[key]

    lo_res = eval_gamma(low)
    hi_res = eval_gamma(high)
    # Mean typically decreases with gamma. Expand bracket if needed.
    while lo_res["stats"]["mean"] < target_mean and low > 1e-4:
        low *= 0.5
        lo_res = eval_gamma(low)
    while hi_res["stats"]["mean"] > target_mean and high < 200.0:
        high *= 2.0
        hi_res = eval_gamma(high)

    best = min(cache.values(), key=lambda z: abs(z["stats"]["mean"] - target_mean))
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_res = eval_gamma(mid)
        if abs(mid_res["stats"]["mean"] - target_mean) < abs(best["stats"]["mean"] - target_mean):
            best = mid_res
        if abs(mid_res["stats"]["mean"] - target_mean) <= tol:
            best = mid_res
            break
        if mid_res["stats"]["mean"] > target_mean:
            low = mid
        else:
            high = mid
    return best


def rolling_common_state(result: Dict[str, object], years=(20, 35)) -> pd.DataFrame:
    cfg: Config = result["cfg"]
    xg = result["x_grid"]
    pmf = result["pmf"]
    F, G, K, P = result["F"], result["G"], result["K"], result["policy"]
    rows = []
    for year in years:
        n = min(int(round(year / cfg.dt)), cfg.n_steps - 1)
        cdf = np.cumsum(pmf[n])
        for state_name, q in (("median", 0.5), ("high", 0.9)):
            x = float(xg[min(np.searchsorted(cdf, q), len(xg) - 1)])
            f = float(np.interp(x, xg, F[n]))
            g = float(np.interp(x, xg, G[n]))
            k3 = float(np.interp(x, xg, K[n]))
            var = max(g - f * f, 0.0)
            sd = math.sqrt(var)
            cm3 = k3 - 3.0 * f * g + 2.0 * f ** 3
            skew = cm3 / (sd ** 3 + 1e-30)
            glide = float(np.interp(x, xg, P[n]))
            rows.append({
                "year": year, "state": state_name, "x": x,
                "mean": f, "stdev": sd, "skewness": skew,
                "glide": glide,
            })
    return pd.DataFrame(rows)


def main() -> None:
    base_cfg = Config()
    # Compile and construct the common transition map once.
    baseline = solve_case(base_cfg)
    target_mean = baseline["stats"]["mean"]
    maps = baseline["maps"]

    fixed_eta_grid = [0.0, 0.5, 1.0, 2.0]
    calibrated_eta_grid = [0.0, 1.0, 2.0, 4.0, 8.0]
    fixed_results: List[Dict[str, object]] = []
    calibrated_results: List[Dict[str, object]] = []

    for eta in fixed_eta_grid:
        fixed = solve_case(Config(**{**asdict(base_cfg), "gamma0": 2.5, "eta0": eta}), maps=maps)
        fixed_results.append(fixed)
        print("fixed eta", eta, fixed["stats"], flush=True)

    for eta in calibrated_eta_grid:
        if eta == 0.0:
            cal = baseline
        else:
            cal = calibrate_gamma(base_cfg, eta, target_mean, maps, high=100.0, tol=0.05, max_iter=18)
        calibrated_results.append(cal)
        print("calibrated eta", eta, "gamma", cal["cfg"].gamma0, cal["stats"], flush=True)

    # Summary tables.
    rows_fixed = []
    rows_cal = []
    for eta, result in zip(fixed_eta_grid, fixed_results):
        rows_fixed.append({
            "eta0": eta, "gamma0": result["cfg"].gamma0,
            **result["stats"], **result["diagnostics"],
        })
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        rows_cal.append({
            "eta0": eta, "gamma0": result["cfg"].gamma0,
            **result["stats"], **result["diagnostics"],
        })
    pd.DataFrame(rows_fixed).to_csv(RES / "dtcmv_mvs_fixed_gamma_summary.csv", index=False)
    pd.DataFrame(rows_cal).to_csv(RES / "dtcmv_mvs_equal_mean_summary.csv", index=False)

    rolling = []
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        r = rolling_common_state(result)
        r.insert(0, "eta0", eta)
        r.insert(1, "gamma0", result["cfg"].gamma0)
        rolling.append(r)
    pd.concat(rolling, ignore_index=True).to_csv(RES / "dtcmv_mvs_rolling_summary.csv", index=False)

    # Save compact arrays.
    np.savez_compressed(
        RES / "dtcmv_mvs_arrays.npz",
        times=np.linspace(0.0, base_cfg.T, base_cfg.n_steps + 1),
        decision_times=np.arange(base_cfg.n_steps) * base_cfg.dt,
        x_grid=baseline["x_grid"],
        fixed_eta_grid=np.array(fixed_eta_grid),
        calibrated_eta_grid=np.array(calibrated_eta_grid),
        gamma_calibrated=np.array([r["cfg"].gamma0 for r in calibrated_results]),
        glide_fixed=np.stack([r["glide"] for r in fixed_results]),
        glide_calibrated=np.stack([r["glide"] for r in calibrated_results]),
        pmf_calibrated=np.stack([r["pmf"] for r in calibrated_results]),
        policy_calibrated=np.stack([r["policy"] for r in calibrated_results]),
    )

    times = np.arange(base_cfg.n_steps) * base_cfg.dt

    # Fixed-gamma glide paths.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for eta, result in zip(fixed_eta_grid, fixed_results):
        ax.plot(times, result["glide"], label=fr"$\eta_0={eta:.2f}$")
    ax.set_xlabel("Years since entry")
    ax.set_ylabel("Mass-weighted risky proportion")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_fixed_gamma_glidepaths.png", dpi=180)
    plt.close(fig)

    # Equal-mean glide paths.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        ax.plot(times, result["glide"], label=fr"$\eta_0={eta:.2f}$, $\gamma_0={result['cfg'].gamma0:.2f}$")
    ax.set_xlabel("Years since entry")
    ax.set_ylabel("Mass-weighted risky proportion")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_equal_mean_glidepaths.png", dpi=180)
    plt.close(fig)

    # Equal-mean terminal CDF.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        values = result["x_grid"] + result["cfg"].D
        ax.plot(values, np.cumsum(result["pmf"][-1]), label=fr"$\eta_0={eta:.2f}$")
    ax.set_xlabel("Terminal DC wealth")
    ax.set_ylabel("CDF")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_equal_mean_cdf.png", dpi=180)
    plt.close(fig)

    # Tail frontier.
    df_cal = pd.DataFrame(rows_cal)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(df_cal["cvar05"], df_cal["ucvar95"], marker="o")
    for _, row in df_cal.iterrows():
        ax.annotate(fr"$\eta_0={row['eta0']:.2f}$", (row["cvar05"], row["ucvar95"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Lower-tail CVaR (5%)")
    ax.set_ylabel("Upper-tail conditional mean (95%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_tail_frontier.png", dpi=180)
    plt.close(fig)

    # Rolling mean/stdev/skewness panels under equal-mean calibration.
    rolling_df = pd.concat(rolling, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    metrics = [("mean", "Conditional mean"), ("stdev", "Conditional stdev"), ("skewness", "Conditional skewness")]
    for ax, (metric, label) in zip(axes, metrics):
        for eta in calibrated_eta_grid:
            d = rolling_df[(rolling_df["eta0"] == eta) & (rolling_df["state"] == "median")]
            ax.plot(d["year"], d[metric], marker="o", label=fr"$\eta_0={eta:.2f}$")
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_rolling_metrics.png", dpi=180)
    plt.close(fig)

    metadata = {
        "config": asdict(base_cfg),
        "target_mean": target_mean,
        "fixed_eta_grid": fixed_eta_grid,
        "calibrated_eta_grid": calibrated_eta_grid,
        "gamma_calibrated": [r["cfg"].gamma0 for r in calibrated_results],
    }
    (RES / "dtcmv_mvs_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
