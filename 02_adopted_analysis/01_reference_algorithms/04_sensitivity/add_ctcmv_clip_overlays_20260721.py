"""Add cTCMV analytical-clip overlays to the existing sensitivity figures.

This companion script imports ``low_balance_refined_sensitivity_20260714.py``,
keeps all existing PCMV/DOMV/cTCMV/dTCMV calculations unchanged, and overlays
the cTCMV analytical clip in the cTCMV panel:

    pi_c^0(t) = (mu-r)/(sigma^2 gamma_c) exp[-r(T-t)],
    pi_c^clip(t,x) = Proj_[0,x](pi_c^0(t)).

The strict and clipped policies are each propagated under their own forward
distribution. Hence the plotted glides are own-policy, mass-weighted risky
fractions rather than evaluations under a common exogenous state distribution.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts" / "04_sensitivity" / "low_balance_refined_sensitivity_20260714.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_sensitivity", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clipped_ctcmv_policy(module, cfg) -> np.ndarray:
    """Return the projected analytical cTCMV policy as a risky fraction."""
    grid = module.x_grid(cfg)
    times = np.arange(cfg.n_steps) * cfg.dt
    tau = cfg.T - times

    if cfg.gamma_c <= 0.0 or cfg.sigma <= 0.0:
        raise ValueError("gamma_c and sigma must be positive")

    unconstrained_dollar = (
        cfg.beta / (cfg.sigma**2 * cfg.gamma_c) * np.exp(-cfg.r * tau)
    )
    clipped_dollar = np.minimum(
        np.maximum(unconstrained_dollar[:, None], 0.0),
        grid[None, :],
    )

    policy = np.zeros_like(clipped_dollar)
    positive = grid > 0.0
    policy[:, positive] = clipped_dollar[:, positive] / grid[None, positive]
    return np.clip(policy, 0.0, 1.0)


def plot_panels_with_clip(
    scenarios: Dict[str, Tuple[str, object, str]],
    results: Dict[str, dict],
    keys: Iterable[str],
    output: Path,
    title: str,
) -> None:
    """Reproduce the existing four panels and add dashed cTCMV clip paths."""
    strategies = ["PCMV", "DOMV", "cTCMV", "dTCMV"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)

    for ax, strategy in zip(axes.ravel(), strategies):
        for key in keys:
            label, cfg, _ = scenarios[key]
            times = np.arange(cfg.n_steps) * cfg.dt
            line, = ax.plot(
                times,
                results[key][strategy]["glide"],
                label=label,
                linewidth=1.6,
            )
            if strategy == "cTCMV":
                ax.plot(
                    times,
                    results[key]["cTCMV_clip"]["glide"],
                    linestyle="--",
                    color=line.get_color(),
                    linewidth=1.45,
                )

        if strategy == "cTCMV":
            ax.set_title("cTCMV (solid: strict; dashed: clip)")
        else:
            ax.set_title(strategy)
        ax.set_xlabel("Years since entry")
        ax.set_ylabel("Mass-weighted risky proportion")
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(labels)),
        frameon=False,
    )
    fig.suptitle(title, y=0.945)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(output_dir: str = "d0_sensitivity_outputs") -> None:
    module = load_base_module()
    original_solve_scenario = module.solve_scenario
    captured: list[tuple[object, str, dict]] = []

    def solve_scenario_with_clip(cfg, profile: str = "constant"):
        result = original_solve_scenario(cfg, profile)
        c_steps = module.contribution_steps(cfg, profile)
        policy = clipped_ctcmv_policy(module, cfg)
        fwd = module.forward_distribution(cfg, policy, c_steps)
        result["cTCMV_clip"] = {
            "policy": policy,
            "pmf": fwd["pmf"],
            "glide": fwd["glide"],
            "upper": fwd["upper"],
            "stats": module.terminal_stats(cfg, fwd["pmf"]),
        }
        captured.append((cfg, profile, result))
        return result

    module.solve_scenario = solve_scenario_with_clip
    module.plot_panels = plot_panels_with_clip
    module.main(output_dir)

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

    out = Path(output_dir)
    summary_rows = []
    path_rows = []
    strict_glides = []
    clip_glides = []

    for key, (cfg, profile, result) in zip(scenario_keys, captured):
        strict = np.asarray(result["cTCMV"]["glide"], dtype=float)
        clip = np.asarray(result["cTCMV_clip"]["glide"], dtype=float)
        times = np.arange(cfg.n_steps) * cfg.dt
        gap = strict - clip
        j = int(np.argmax(np.abs(gap)))

        summary_rows.append(
            {
                "scenario": key,
                "profile": profile,
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
                    "time": float(t),
                    "strict_glide": float(strict_value),
                    "clip_glide": float(clip_value),
                    "strict_minus_clip": float(gap_value),
                    "abs_gap": float(abs(gap_value)),
                }
            )
        strict_glides.append(strict)
        clip_glides.append(clip)

    pd.DataFrame(summary_rows).to_csv(
        out / "ctcmv_strict_vs_clip_sensitivity_summary.csv",
        index=False,
    )
    pd.DataFrame(path_rows).to_csv(
        out / "ctcmv_strict_vs_clip_sensitivity_paths.csv",
        index=False,
    )
    np.savez_compressed(
        out / "ctcmv_strict_vs_clip_sensitivity_glidepaths.npz",
        scenario_keys=np.array(scenario_keys),
        decision_times=np.arange(captured[0][0].n_steps) * captured[0][0].dt,
        strict_glides=np.stack(strict_glides),
        clip_glides=np.stack(clip_glides),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="d0_sensitivity_outputs",
        help="Same output directory used by the existing sensitivity script.",
    )
    args = parser.parse_args()
    main(args.output_dir)
