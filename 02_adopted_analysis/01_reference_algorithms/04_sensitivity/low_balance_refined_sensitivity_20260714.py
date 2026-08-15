"""Reproduce the D_T=0 baseline and sensitivity figures.

Strategies
----------
PCMV, DOMV, cTCMV, dTCMV, and CP60.

Sensitivity blocks
------------------
1. Deterministic terminal benefit D_T.
2. Safe rate r.
3. Risky expected return mu.
4. Risky volatility sigma.
5. Contribution profiles with the same total contributions:
   constant, linearly increasing, and quadratically increasing.

The code uses a bounded semi-Lagrangian / Markov--Gauss--Hermite
backward solver on 0 <= pi <= x and a forward distribution propagation.
The default sensitivity grid is N_t=80 because the purpose is cross-scenario
screening. The paper's main estimates can be recomputed at N_t=480 by changing
Config.n_steps and refining the other grids.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class Config:
    T: float = 40.0
    n_steps: int = 80
    x0: float = 1.0 / 12.0
    x_max: float = 300.0
    n_x: int = 101
    x_power: float = 2.5
    n_controls: int = 9
    n_gh: int = 3
    r: float = 0.015
    mu: float = 0.055
    sigma: float = 0.18
    D: float = 0.0
    gamma_p: float = 0.05
    gamma_d: float = 0.0912608
    gamma_c: float = 0.059638671875
    gamma_0: float = 1.193359375
    theta_cp: float = 0.4694735
    target_min: float = 40.0
    target_max: float = 340.0
    target_step: float = 20.0

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def beta(self) -> float:
        return self.mu - self.r


def x_grid(cfg: Config) -> np.ndarray:
    """Nonuniform wealth grid with explicit low-balance refinement.

    The state constraint 0 <= pi <= x and the ratio g=pi/x make the
    feedback especially sensitive near x=0.  A coarse first positive node
    can therefore create artificial zero-risk allocations.  We use a
    stronger power transformation and force the deterministic initial
    balance x0 onto the grid while preserving the requested number of nodes.
    """
    u = np.linspace(0.0, 1.0, cfg.n_x)
    grid = cfg.x_max * np.power(u, cfg.x_power)
    # Replace the closest interior node by x0.  This preserves array sizes,
    # removes interpolation across x=0 at the initial state, and keeps the
    # grid strictly increasing.
    j = int(np.argmin(np.abs(grid - cfg.x0)))
    j = min(max(j, 1), cfg.n_x - 2)
    grid[j] = cfg.x0
    grid = np.sort(grid)
    if np.any(np.diff(grid) <= 0.0):
        raise RuntimeError("wealth grid must be strictly increasing")
    return grid


def gh_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(n)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def contribution_steps(cfg: Config, kind: str = "constant", total: float | None = None) -> np.ndarray:
    """Return the contribution amount added in each time step.

    The total undiscounted contribution is kept fixed.  The continuous-time
    baseline c=1 over 40 years corresponds to total=40.
    """
    total = cfg.T if total is None else total
    if kind == "constant":
        weights = np.ones(cfg.n_steps)
    elif kind == "linear":
        weights = np.arange(1.0, cfg.n_steps + 1.0)
    elif kind == "quadratic":
        a = np.arange(1.0, cfg.n_steps + 1.0)
        weights = a * a
    else:
        raise ValueError(f"Unknown contribution profile: {kind}")
    return total * weights / weights.sum()


def safe_asset_future_value(steps: np.ndarray, r: float, dt: float) -> float:
    n = len(steps)
    return float(sum(steps[k] * (1.0 + r * dt) ** (n - 1 - k) for k in range(n)))


def interp(grid: np.ndarray, values: np.ndarray, x: float) -> float:
    if x <= grid[0]:
        return float(values[0])
    if x >= grid[-1]:
        slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
        return float(values[-1] + (x - grid[-1]) * slope)
    return float(np.interp(x, grid, values))


def transition_expectation(
    next_values: np.ndarray,
    grid: np.ndarray,
    x: float,
    risky_fraction: float,
    cfg: Config,
    c_step: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> float:
    pi = risky_fraction * x
    drift = (cfg.r * x + cfg.beta * pi) * cfg.dt + c_step
    sd = cfg.sigma * pi * math.sqrt(cfg.dt)
    total = 0.0
    for z, w in zip(gh_x, gh_w):
        xp = max(x + drift + sd * z, 0.0)
        total += float(w) * interp(grid, next_values, xp)
    return total


def refine_minimum(f: np.ndarray, j: int, da: float) -> float:
    if j == 0 or j == len(f) - 1:
        return j * da
    denom = f[j - 1] - 2.0 * f[j] + f[j + 1]
    if denom <= 1e-14:
        return j * da
    offset = 0.5 * (f[j - 1] - f[j + 1]) / denom
    return float(np.clip((j + np.clip(offset, -1.0, 1.0)) * da, 0.0, 1.0))


def refine_maximum(f: np.ndarray, j: int, da: float) -> float:
    if j == 0 or j == len(f) - 1:
        return j * da
    denom = f[j - 1] - 2.0 * f[j] + f[j + 1]
    if denom >= -1e-14:
        return j * da
    offset = 0.5 * (f[j - 1] - f[j + 1]) / denom
    return float(np.clip((j + np.clip(offset, -1.0, 1.0)) * da, 0.0, 1.0))


def parabolic_argmax(x: np.ndarray, y: np.ndarray) -> float:
    j = int(np.argmax(y))
    if j == 0 or j == len(x) - 1:
        return float(x[j])
    denom = y[j - 1] - 2.0 * y[j] + y[j + 1]
    if denom >= -1e-14:
        return float(x[j])
    offset = 0.5 * (y[j - 1] - y[j + 1]) / denom
    return float(x[j] + np.clip(offset, -1.0, 1.0) * (x[1] - x[0]))


def interpolate_target_array(targets: np.ndarray, array: np.ndarray, z: float, n: int, i: int) -> float:
    if z <= targets[0]:
        return float(array[0, n, i])
    if z >= targets[-1]:
        return float(array[-1, n, i])
    j = int(np.searchsorted(targets, z) - 1)
    lam = (z - targets[j]) / (targets[j + 1] - targets[j])
    return float((1.0 - lam) * array[j, n, i] + lam * array[j + 1, n, i])


def solve_target_family(cfg: Config, c_steps: np.ndarray) -> Dict[str, np.ndarray]:
    """Solve fixed-target quadratic-loss problems for PCMV and DOMV."""
    grid = x_grid(cfg)
    gh_x, gh_w = gh_nodes_weights(cfg.n_gh)
    targets = np.arange(cfg.target_min, cfg.target_max + 0.1 * cfg.target_step, cfg.target_step)
    n_z = len(targets)
    V = np.empty((n_z, cfg.n_steps + 1, cfg.n_x))
    M = np.empty_like(V)
    P = np.empty((n_z, cfg.n_steps, cfg.n_x))
    controls = np.linspace(0.0, 1.0, cfg.n_controls)
    da = controls[1] - controls[0]

    for z_idx, z in enumerate(targets):
        wealth = grid + cfg.D
        V[z_idx, -1] = (wealth - z) ** 2
        M[z_idx, -1] = wealth
        for n in range(cfg.n_steps - 1, -1, -1):
            for i, x in enumerate(grid):
                vals = np.array([
                    transition_expectation(V[z_idx, n + 1], grid, x, a, cfg, c_steps[n], gh_x, gh_w)
                    for a in controls
                ])
                j = int(np.argmin(vals))
                a_star = refine_minimum(vals, j, da)
                P[z_idx, n, i] = a_star
                V[z_idx, n, i] = transition_expectation(
                    V[z_idx, n + 1], grid, x, a_star, cfg, c_steps[n], gh_x, gh_w
                )
                M[z_idx, n, i] = transition_expectation(
                    M[z_idx, n + 1], grid, x, a_star, cfg, c_steps[n], gh_x, gh_w
                )
    return {"targets": targets, "x_grid": grid, "V": V, "M": M, "policy": P, "gh_x": gh_x, "gh_w": gh_w}


def select_pcmv_policy(cfg: Config, family: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
    targets = family["targets"]
    grid = family["x_grid"]
    V0 = np.array([interp(grid, family["V"][k, 0], cfg.x0) for k in range(len(targets))])
    score = targets - 1.0 / (2.0 * cfg.gamma_p) - 0.5 * cfg.gamma_p * V0
    z_star = parabolic_argmax(targets, score)
    policy = np.empty((cfg.n_steps, cfg.n_x))
    for n in range(cfg.n_steps):
        for i in range(cfg.n_x):
            policy[n, i] = interpolate_target_array(targets, family["policy"], z_star, n, i)
    return policy, z_star


def select_domv_policy(cfg: Config, family: Dict[str, np.ndarray]) -> np.ndarray:
    targets = family["targets"]
    policy = np.empty((cfg.n_steps, cfg.n_x))
    for n in range(cfg.n_steps):
        for i in range(cfg.n_x):
            score = targets - 1.0 / (2.0 * cfg.gamma_d) - 0.5 * cfg.gamma_d * family["V"][:, n, i]
            z_star = parabolic_argmax(targets, score)
            policy[n, i] = interpolate_target_array(targets, family["policy"], z_star, n, i)
    return policy


def background_wealth(cfg: Config, c_steps: np.ndarray) -> np.ndarray:
    H = np.zeros(cfg.n_steps + 1)
    H[-1] = cfg.D
    gross = 1.0 + cfg.r * cfg.dt
    for n in range(cfg.n_steps - 1, -1, -1):
        H[n] = (c_steps[n] + H[n + 1]) / gross
    return H


def solve_tcmv(cfg: Config, c_steps: np.ndarray, dynamic_risk_aversion: bool) -> np.ndarray:
    grid = x_grid(cfg)
    gh_x, gh_w = gh_nodes_weights(cfg.n_gh)
    H = background_wealth(cfg, c_steps)
    M = np.empty((cfg.n_steps + 1, cfg.n_x))
    Q = np.empty_like(M)
    P = np.empty((cfg.n_steps, cfg.n_x))
    wealth = grid + cfg.D
    M[-1] = wealth
    Q[-1] = wealth * wealth
    controls = np.linspace(0.0, 1.0, cfg.n_controls)
    da = controls[1] - controls[0]

    for n in range(cfg.n_steps - 1, -1, -1):
        for i, x in enumerate(grid):
            gamma = cfg.gamma_0 / max(x + H[n], 1e-12) if dynamic_risk_aversion else cfg.gamma_c
            mvals = np.empty(cfg.n_controls)
            qvals = np.empty(cfg.n_controls)
            score = np.empty(cfg.n_controls)
            for j, a in enumerate(controls):
                m = transition_expectation(M[n + 1], grid, x, a, cfg, c_steps[n], gh_x, gh_w)
                q = transition_expectation(Q[n + 1], grid, x, a, cfg, c_steps[n], gh_x, gh_w)
                var = max(q - m * m, 0.0)
                mvals[j] = m
                qvals[j] = q
                score[j] = m - 0.5 * gamma * var
            j = int(np.argmax(score))
            a_star = refine_maximum(score, j, da)
            P[n, i] = a_star
            M[n, i] = transition_expectation(M[n + 1], grid, x, a_star, cfg, c_steps[n], gh_x, gh_w)
            Q[n, i] = transition_expectation(Q[n + 1], grid, x, a_star, cfg, c_steps[n], gh_x, gh_w)
    return P


def deposit(grid: np.ndarray, mass: np.ndarray, x: float, weight: float) -> None:
    if x <= grid[0]:
        mass[0] += weight
        return
    if x >= grid[-1]:
        mass[-1] += weight
        return
    j = int(np.searchsorted(grid, x) - 1)
    lam = (x - grid[j]) / (grid[j + 1] - grid[j])
    mass[j] += weight * (1.0 - lam)
    mass[j + 1] += weight * lam


def forward_distribution(cfg: Config, policy: np.ndarray, c_steps: np.ndarray) -> Dict[str, np.ndarray]:
    grid = x_grid(cfg)
    gh_x, gh_w = gh_nodes_weights(cfg.n_gh)
    p = np.zeros(cfg.n_x)
    deposit(grid, p, cfg.x0, 1.0)
    pmf = np.zeros((cfg.n_steps + 1, cfg.n_x))
    pmf[0] = p
    # Controls are defined at t_0,...,t_{N-1}; terminal time T has no control.
    glide = np.zeros(cfg.n_steps)
    upper = np.zeros(cfg.n_steps)

    for n in range(cfg.n_steps):
        if n == 0:
            # x0 is deterministic.  Directly interpolate the feedback at x0;
            # the split of initial mass over grid nodes is only a propagation device.
            pi0 = float(np.interp(cfg.x0, grid, policy[n] * grid))
            a0 = float(np.clip(pi0 / cfg.x0, 0.0, 1.0))
            glide[n] = a0
            upper[n] = float(a0 >= 0.999)
        else:
            glide[n] = float(np.dot(p, policy[n]))
            upper[n] = float(np.dot(p, policy[n] >= 0.999))
        p_new = np.zeros(cfg.n_x)
        if n == 0:
            x = cfg.x0
            a = glide[0]
            pi = a * x
            drift = (cfg.r * x + cfg.beta * pi) * cfg.dt + c_steps[n]
            sd = cfg.sigma * pi * math.sqrt(cfg.dt)
            for z, w in zip(gh_x, gh_w):
                deposit(grid, p_new, max(x + drift + sd * z, 0.0), w)
        else:
            for i, x in enumerate(grid):
                if p[i] <= 0.0:
                    continue
                a = policy[n, i]
                pi = a * x
                drift = (cfg.r * x + cfg.beta * pi) * cfg.dt + c_steps[n]
                sd = cfg.sigma * pi * math.sqrt(cfg.dt)
                for z, w in zip(gh_x, gh_w):
                    deposit(grid, p_new, max(x + drift + sd * z, 0.0), p[i] * w)
        p_new /= p_new.sum()
        p = p_new
        pmf[n + 1] = p
    return {"pmf": pmf, "glide": glide, "upper": upper}


def terminal_stats(cfg: Config, pmf: np.ndarray) -> Dict[str, float]:
    values = x_grid(cfg) + cfg.D
    p = pmf[-1] / pmf[-1].sum()
    mean = float(np.dot(p, values))
    var = float(np.dot(p, (values - mean) ** 2))
    cdf = np.cumsum(p)

    def quantile(q: float) -> float:
        return float(values[min(np.searchsorted(cdf, q), len(values) - 1)])

    remaining = 0.05
    lower_sum = 0.0
    for value, prob in zip(values, p):
        take = min(remaining, float(prob))
        lower_sum += take * float(value)
        remaining -= take
        if remaining <= 1e-15:
            break
    return {
        "mean": mean,
        "stdev": math.sqrt(max(var, 0.0)),
        "q05": quantile(0.05),
        "q50": quantile(0.50),
        "q95": quantile(0.95),
        "cvar05": lower_sum / 0.05,
    }


def solve_scenario(cfg: Config, profile: str = "constant") -> Dict[str, Dict[str, np.ndarray | Dict[str, float]]]:
    c_steps = contribution_steps(cfg, profile)
    family = solve_target_family(cfg, c_steps)
    pcmv_policy, target = select_pcmv_policy(cfg, family)
    domv_policy = select_domv_policy(cfg, family)
    ctcmv_policy = solve_tcmv(cfg, c_steps, dynamic_risk_aversion=False)
    dtcmv_policy = solve_tcmv(cfg, c_steps, dynamic_risk_aversion=True)
    cp_policy = np.full((cfg.n_steps, cfg.n_x), cfg.theta_cp)

    policies = {
        "PCMV": pcmv_policy,
        "DOMV": domv_policy,
        "cTCMV": ctcmv_policy,
        "dTCMV": dtcmv_policy,
        "CP": cp_policy,
    }
    result: Dict[str, Dict[str, np.ndarray | Dict[str, float]]] = {
        "meta": {"target_pcmv": target, "contribution_profile": profile}  # type: ignore[dict-item]
    }
    for name, policy in policies.items():
        fwd = forward_distribution(cfg, policy, c_steps)
        result[name] = {
            "policy": policy,
            "pmf": fwd["pmf"],
            "glide": fwd["glide"],
            "upper": fwd["upper"],
            "stats": terminal_stats(cfg, fwd["pmf"]),
        }
    return result


def plot_panels(
    scenarios: Dict[str, Tuple[str, Config, str]],
    results: Dict[str, Dict[str, Dict[str, np.ndarray | Dict[str, float]]]],
    keys: Iterable[str],
    output: Path,
    title: str,
) -> None:
    strategies = ["PCMV", "DOMV", "cTCMV", "dTCMV"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)
    for ax, strategy in zip(axes.ravel(), strategies):
        for key in keys:
            label, cfg, _ = scenarios[key]
            times = np.arange(cfg.n_steps) * cfg.dt
            ax.plot(times, results[key][strategy]["glide"], label=label)  # type: ignore[arg-type]
        ax.set_title(strategy)
        ax.set_xlabel("Years since entry")
        ax.set_ylabel("Mass-weighted risky proportion")
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(output_dir: str = "d0_sensitivity_outputs") -> None:
    out = Path(output_dir)
    fig_dir = out / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = Config(D=0.0)
    constant = contribution_steps(base, "constant")
    d_fv = safe_asset_future_value(constant, base.r, base.dt)

    scenarios: Dict[str, Tuple[str, Config, str]] = {
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

    results: Dict[str, Dict[str, Dict[str, np.ndarray | Dict[str, float]]]] = {}
    rows = []
    for key, (label, cfg, profile) in scenarios.items():
        print(f"Solving {key}: {label}", flush=True)
        result = solve_scenario(cfg, profile)
        results[key] = result
        for strategy in ["PCMV", "DOMV", "cTCMV", "dTCMV", "CP"]:
            stats = result[strategy]["stats"]  # type: ignore[assignment]
            glide = result[strategy]["glide"]  # type: ignore[assignment]
            rows.append({
                "scenario": key,
                "label": label,
                "profile": profile,
                "strategy": strategy,
                **stats,  # type: ignore[arg-type]
                "mean_glide": float(np.mean(glide)),  # type: ignore[index]
            })
    pd.DataFrame(rows).to_csv(out / "sensitivity_summary.csv", index=False)

    # Save corrected decision-time glide paths for reproducibility and
    # cross-grid figures.  No artificial terminal entry is stored.
    np.savez_compressed(
        out / "sensitivity_glidepaths_corrected.npz",
        decision_times=np.arange(base.n_steps) * base.dt,
        scenario_keys=np.array(list(scenarios.keys())),
        strategies=np.array(["PCMV", "DOMV", "cTCMV", "dTCMV", "CP"]),
        glides=np.stack([
            np.stack([results[key][strategy]["glide"] for strategy in ["PCMV", "DOMV", "cTCMV", "dTCMV", "CP"]])
            for key in scenarios
        ]),
    )

    # Baseline five-strategy plot.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    times = np.arange(base.n_steps) * base.dt
    for strategy in ["PCMV", "DOMV", "cTCMV", "dTCMV", "CP"]:
        ax.plot(times, results["baseline"][strategy]["glide"], label=strategy)  # type: ignore[arg-type]
    ax.set_xlabel("Years since entry")
    ax.set_ylabel("Mass-weighted risky proportion")
    ax.set_ylim(0.0, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_revised_baseline_D0_glidepaths_N80.png", dpi=180)
    plt.close(fig)

    plot_panels(scenarios, results, ["baseline", "D_alt"], fig_dir / "fig_D_sensitivity_glidepaths_N80.png", "Sensitivity to D_T")
    plot_panels(scenarios, results, ["r_low", "baseline", "r_high"], fig_dir / "fig_r_sensitivity_glidepaths_N80.png", "Sensitivity to r")
    plot_panels(scenarios, results, ["mu_low", "baseline", "mu_high"], fig_dir / "fig_mu_sensitivity_glidepaths_N80.png", "Sensitivity to mu")
    plot_panels(scenarios, results, ["sigma_low", "baseline", "sigma_high"], fig_dir / "fig_sigma_sensitivity_glidepaths_N80.png", "Sensitivity to sigma")
    plot_panels(
        scenarios,
        results,
        ["contrib_constant", "contrib_linear", "contrib_quadratic"],
        fig_dir / "fig_contrib_profile_sensitivity_glidepaths_N80.png",
        "Contribution-profile sensitivity with fixed total contributions",
    )
    print(f"Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
