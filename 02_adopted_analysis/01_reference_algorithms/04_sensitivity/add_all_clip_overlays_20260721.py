"""Recompute all sensitivity figures with strict and clipped policies.

For each of PCMV, DOMV, cTCMV, and dTCMV, the script compares:

* solid line: directly constrained feedback solved on 0 <= pi <= x;
* dashed line: the corresponding unconstrained analytical policy projected
  pointwise onto 0 <= pi <= x.

Both policies are propagated under their own forward wealth distributions, so
all plotted glide paths are own-policy mass-weighted risky fractions.

The script imports ``low_balance_refined_sensitivity_20260714.py`` and leaves
its strict constrained solvers unchanged.  It adds the following projected
unconstrained controls, where Y_t = X_t + H_t, tau = T-t, beta = mu-r, and
A = beta^2/sigma^2:

PCMV:
    z_p = Y_0 exp(rT) + exp(AT)/gamma_p,
    pi^0_p(t,x) = beta/sigma^2 * [z_p exp(-r tau) - Y_t].

DOMV:
    pi^0_d(t) = beta/(sigma^2 gamma_d) * exp[(A-r) tau].

cTCMV:
    pi^0_c(t) = beta/(sigma^2 gamma_c) * exp(-r tau).

dTCMV:
    pi^0_rho(t,x) = theta_rho(t) Y_t,
where theta_rho solves the unconstrained Volterra equation represented by the
backward two-state ODE used in ``diagnostics/unconstrained_dtcmv_theta.py``.

Each dollar control is projected to [0,x], then converted to a risky fraction.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts" / "04_sensitivity" / "low_balance_refined_sensitivity_20260714.py"
STRATEGIES = ("PCMV", "DOMV", "cTCMV", "dTCMV")
CLIP_KEYS = {name: f"{name}_clip" for name in STRATEGIES}


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_sensitivity", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dollars_to_fraction(grid: np.ndarray, dollars: np.ndarray) -> np.ndarray:
    """Project dollar controls to [0,x] and convert them to risky fractions."""
    clipped = np.minimum(np.maximum(dollars, 0.0), grid[None, :])
    policy = np.zeros_like(clipped, dtype=float)
    positive = grid > 0.0
    policy[:, positive] = clipped[:, positive] / grid[None, positive]
    return np.clip(policy, 0.0, 1.0)


def pcmv_clip_policy(module, cfg, c_steps: np.ndarray) -> np.ndarray:
    grid = module.x_grid(cfg)
    H = module.background_wealth(cfg, c_steps)
    times = np.arange(cfg.n_steps) * cfg.dt
    tau = cfg.T - times
    A = cfg.beta**2 / cfg.sigma**2
    y0 = cfg.x0 + H[0]
    target = y0 * math.exp(cfg.r * cfg.T) + math.exp(A * cfg.T) / cfg.gamma_p
    desired = target * np.exp(-cfg.r * tau)[:, None] - (grid[None, :] + H[:-1, None])
    dollars = cfg.beta / cfg.sigma**2 * desired
    return dollars_to_fraction(grid, dollars)


def domv_clip_policy(module, cfg, c_steps: np.ndarray) -> np.ndarray:
    del c_steps
    grid = module.x_grid(cfg)
    times = np.arange(cfg.n_steps) * cfg.dt
    tau = cfg.T - times
    A = cfg.beta**2 / cfg.sigma**2
    dollars_by_time = (
        cfg.beta / (cfg.sigma**2 * cfg.gamma_d) * np.exp((A - cfg.r) * tau)
    )
    dollars = np.broadcast_to(dollars_by_time[:, None], (cfg.n_steps, cfg.n_x)).copy()
    return dollars_to_fraction(grid, dollars)


def ctcmv_clip_policy(module, cfg, c_steps: np.ndarray) -> np.ndarray:
    del c_steps
    grid = module.x_grid(cfg)
    times = np.arange(cfg.n_steps) * cfg.dt
    tau = cfg.T - times
    dollars_by_time = (
        cfg.beta / (cfg.sigma**2 * cfg.gamma_c) * np.exp(-cfg.r * tau)
    )
    dollars = np.broadcast_to(dollars_by_time[:, None], (cfg.n_steps, cfg.n_x)).copy()
    return dollars_to_fraction(grid, dollars)


def unconstrained_dtcmv_theta(cfg, decision_times: np.ndarray) -> np.ndarray:
    """Solve the unconstrained dTCMV coefficient for the supplied scenario."""
    if cfg.gamma_0 <= 0.0 or cfg.sigma <= 0.0 or cfg.beta <= 0.0:
        raise ValueError("dTCMV clip requires gamma_0>0, sigma>0, and mu-r>0")

    A = cfg.beta**2 / cfg.sigma**2
    rho = cfg.gamma_0

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        i1, i2 = state
        theta = A / (rho * cfg.beta) * (
            math.exp(-i1) + rho * math.exp(-i2) - rho
        )
        return np.array(
            [
                -(cfg.r + cfg.beta * theta - cfg.sigma**2 * theta**2),
                -(cfg.sigma**2 * theta**2),
            ],
            dtype=float,
        )

    sol = solve_ivp(
        rhs,
        t_span=(cfg.T, 0.0),
        y0=np.array([0.0, 0.0]),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    if not sol.success or sol.sol is None:
        raise RuntimeError(f"dTCMV theta solve failed: {sol.message}")

    i1, i2 = sol.sol(decision_times)
    theta = A / (rho * cfg.beta) * (
        np.exp(-i1) + rho * np.exp(-i2) - rho
    )
    return np.asarray(theta, dtype=float)


def dtcmv_clip_policy(module, cfg, c_steps: np.ndarray) -> np.ndarray:
    grid = module.x_grid(cfg)
    H = module.background_wealth(cfg, c_steps)
    times = np.arange(cfg.n_steps) * cfg.dt
    theta = unconstrained_dtcmv_theta(cfg, times)
    total_wealth = grid[None, :] + H[:-1, None]
    dollars = theta[:, None] * total_wealth
    return dollars_to_fraction(grid, dollars)


def build_clip_policies(module, cfg, c_steps: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "PCMV_clip": pcmv_clip_policy(module, cfg, c_steps),
        "DOMV_clip": domv_clip_policy(module, cfg, c_steps),
        "cTCMV_clip": ctcmv_clip_policy(module, cfg, c_steps),
        "dTCMV_clip": dtcmv_clip_policy(module, cfg, c_steps),
    }


def plot_panels_with_all_clips(
    scenarios: Dict[str, Tuple[str, object, str]],
    results: Dict[str, dict],
    keys: Iterable[str],
    output: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3), sharex=True)

    scenario_handles: list[Line2D] = []
    scenario_labels: list[str] = []
    for ax, strategy in zip(axes.ravel(), STRATEGIES):
        for key in keys:
            label, cfg, _ = scenarios[key]
            times = np.arange(cfg.n_steps) * cfg.dt
            line, = ax.plot(
                times,
                results[key][strategy]["glide"],
                label=label,
                linewidth=1.75,
            )
            ax.plot(
                times,
                results[key][CLIP_KEYS[strategy]]["glide"],
                linestyle="--",
                color=line.get_color(),
                linewidth=1.55,
            )
            if strategy == STRATEGIES[0]:
                scenario_handles.append(line)
                scenario_labels.append(label)

        ax.set_title(strategy)
        ax.set_xlabel("Years since entry")
        ax.set_ylabel("Mass-weighted risky proportion")
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.25)

    fig.legend(
        scenario_handles,
        scenario_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(scenario_labels)),
        frameon=False,
    )
    style_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="-", label="Strict constrained"),
        Line2D([0], [0], color="black", linewidth=1.6, linestyle="--", label="Clipped unconstrained"),
    ]
    fig.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, y=0.945)
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_gap_outputs(output_dir: Path, captured: list[tuple[object, str, dict]]) -> None:
    scenario_keys = [
        "baseline",
        "D_alt",
        "r_low",
        "r_high",
        "mu_low",
        "mu_high",
        "sigma_low",
        "sigma_high",
        "contrib_constant",
        "contrib_linear",
        "contrib_quadratic",
    ]
    if len(captured) != len(scenario_keys):
        raise RuntimeError(
            f"Expected {len(scenario_keys)} scenarios, captured {len(captured)}"
        )

    summary_rows: list[dict] = []
    path_rows: list[dict] = []
    strict_arrays: list[np.ndarray] = []
    clip_arrays: list[np.ndarray] = []

    for key, (cfg, profile, result) in zip(scenario_keys, captured):
        times = np.arange(cfg.n_steps) * cfg.dt
        strict_by_strategy = []
        clip_by_strategy = []
        for strategy in STRATEGIES:
            strict = np.asarray(result[strategy]["glide"], dtype=float)
            clip = np.asarray(result[CLIP_KEYS[strategy]]["glide"], dtype=float)
            gap = strict - clip
            j = int(np.argmax(np.abs(gap)))
            summary_rows.append(
                {
                    "scenario": key,
                    "profile": profile,
                    "strategy": strategy,
                    "r": cfg.r,
                    "mu": cfg.mu,
                    "sigma": cfg.sigma,
                    "D_T": cfg.D,
                    "mean_abs_gap": float(np.mean(np.abs(gap))),
                    "max_abs_gap": float(np.max(np.abs(gap))),
                    "year_of_max_gap": float(times[j]),
                    "signed_gap_at_max": float(gap[j]),
                    "initial_gap": float(gap[0]),
                    "year20_gap": float(gap[int(round(20.0 / cfg.dt))]),
                    "year35_gap": float(gap[int(round(35.0 / cfg.dt))]),
                }
            )
            for t, strict_value, clip_value, gap_value in zip(times, strict, clip, gap):
                path_rows.append(
                    {
                        "scenario": key,
                        "profile": profile,
                        "strategy": strategy,
                        "time": float(t),
                        "strict_glide": float(strict_value),
                        "clip_glide": float(clip_value),
                        "strict_minus_clip": float(gap_value),
                        "abs_gap": float(abs(gap_value)),
                    }
                )
            strict_by_strategy.append(strict)
            clip_by_strategy.append(clip)
        strict_arrays.append(np.stack(strict_by_strategy))
        clip_arrays.append(np.stack(clip_by_strategy))

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "all_strategies_strict_vs_clip_sensitivity_summary.csv",
        index=False,
    )
    pd.DataFrame(path_rows).to_csv(
        output_dir / "all_strategies_strict_vs_clip_sensitivity_paths.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "all_strategies_strict_vs_clip_sensitivity_glidepaths.npz",
        scenario_keys=np.array(scenario_keys),
        strategies=np.array(STRATEGIES),
        decision_times=np.arange(captured[0][0].n_steps) * captured[0][0].dt,
        strict_glides=np.stack(strict_arrays),
        clip_glides=np.stack(clip_arrays),
    )


def main(output_dir: str = "d0_sensitivity_outputs") -> None:
    module = load_base_module()
    original_solve_scenario = module.solve_scenario
    captured: list[tuple[object, str, dict]] = []

    def solve_scenario_with_all_clips(cfg, profile: str = "constant"):
        result = original_solve_scenario(cfg, profile)
        c_steps = module.contribution_steps(cfg, profile)
        for clip_key, policy in build_clip_policies(module, cfg, c_steps).items():
            fwd = module.forward_distribution(cfg, policy, c_steps)
            result[clip_key] = {
                "policy": policy,
                "pmf": fwd["pmf"],
                "glide": fwd["glide"],
                "upper": fwd["upper"],
                "stats": module.terminal_stats(cfg, fwd["pmf"]),
            }
        captured.append((cfg, profile, result))
        return result

    module.solve_scenario = solve_scenario_with_all_clips
    module.plot_panels = plot_panels_with_all_clips
    module.main(output_dir)

    out = Path(output_dir)
    write_gap_outputs(out, captured)
    print(f"Strict-vs-clip outputs written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="d0_sensitivity_outputs",
        help="Same output directory used by the existing sensitivity script.",
    )
    args = parser.parse_args()
    main(args.output_dir)
