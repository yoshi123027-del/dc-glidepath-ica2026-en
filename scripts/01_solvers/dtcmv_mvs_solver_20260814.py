from __future__ import annotations

"""Baseline-consistent dTCMV-MVS runner.

This module intentionally reuses the established 20260713 MVS numerical core while
changing the baseline variance-aversion scale to the dTCMV value used in the main
MV equal-mean comparison.  The previous 20260713 entry point is retained unchanged
for reproducibility of historical outputs.

The key nesting requirement is:
    eta0 = 0 and gamma0 = rho_d  =>  dTCMV-MVS reduces to dTCMV-MV.

For the paper baseline, rho_d = 1.193359375 (results/equal_mean_calibration.csv).
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOLVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLVER_DIR))
import dtcmv_mvs_solver_20260713 as core

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figs"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Main-paper dTCMV equal-mean calibration from results/equal_mean_calibration.csv.
BASELINE_DTCMV_GAMMA0 = 1.193359375


def baseline_config(**overrides) -> core.Config:
    data = asdict(core.Config())
    data["gamma0"] = BASELINE_DTCMV_GAMMA0
    data.update(overrides)
    return core.Config(**data)


def _saved_mv_glide() -> np.ndarray | None:
    """Return the main-paper dTCMV MV glide path when the baseline NPZ is present."""
    path = RES / "monthly_D0_policy_arrays.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "dtcmv_glide" not in data.files:
        return None
    return np.asarray(data["dtcmv_glide"], dtype=float)


def nesting_diagnostics(mvs_eta0_result: Dict[str, object]) -> Dict[str, float]:
    """Compare eta0=0 MVS against the saved main-paper dTCMV MV glide path.

    The two solvers use slightly different control-grid/refinement rules, so this is
    a numerical consistency diagnostic rather than a bit-for-bit identity test.
    """
    mv = _saved_mv_glide()
    if mv is None:
        return {}
    mvs = np.asarray(mvs_eta0_result["glide"], dtype=float)
    n = min(len(mv), len(mvs))
    diff = mvs[:n] - mv[:n]
    return {
        "mv_nesting_mean_abs_glide_gap": float(np.mean(np.abs(diff))),
        "mv_nesting_max_abs_glide_gap": float(np.max(np.abs(diff))),
        "mv_nesting_final_glide_gap": float(diff[-1]),
        "mv_saved_final_glide": float(mv[n - 1]),
        "mvs_eta0_final_glide": float(mvs[n - 1]),
    }


def main() -> None:
    base_cfg = baseline_config()
    baseline = core.solve_case(base_cfg)
    target_mean = baseline["stats"]["mean"]
    maps = baseline["maps"]

    fixed_eta_grid = [0.0, 0.5, 1.0, 2.0]
    calibrated_eta_grid = [0.0, 1.0, 2.0, 4.0, 8.0]
    fixed_results: List[Dict[str, object]] = []
    calibrated_results: List[Dict[str, object]] = []

    # Fixed-gamma sensitivity: gamma0 is held at the SAME value as the MV dTCMV
    # baseline.  This is the essential correction relative to the historical run
    # that hard-coded gamma0=2.5.
    for eta in fixed_eta_grid:
        fixed_cfg = baseline_config(eta0=eta)
        fixed = core.solve_case(fixed_cfg, maps=maps)
        fixed_results.append(fixed)
        print("fixed eta", eta, "gamma", fixed_cfg.gamma0, fixed["stats"], flush=True)

    # Equal-mean sensitivity starts from the same eta0=0 baseline and recalibrates
    # gamma0 only for positive skewness coefficients.
    for eta in calibrated_eta_grid:
        if eta == 0.0:
            cal = baseline
        else:
            cal = core.calibrate_gamma(
                base_cfg,
                eta,
                target_mean,
                maps,
                high=100.0,
                tol=0.05,
                max_iter=18,
            )
        calibrated_results.append(cal)
        print("calibrated eta", eta, "gamma", cal["cfg"].gamma0, cal["stats"], flush=True)

    nesting = nesting_diagnostics(fixed_results[0])
    if nesting:
        print("eta0=0 MV nesting diagnostics", nesting, flush=True)

    rows_fixed = []
    rows_cal = []
    for eta, result in zip(fixed_eta_grid, fixed_results):
        row = {
            "eta0": eta,
            "gamma0": result["cfg"].gamma0,
            **result["stats"],
            **result["diagnostics"],
        }
        if eta == 0.0:
            row.update(nesting)
        rows_fixed.append(row)

    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        rows_cal.append({
            "eta0": eta,
            "gamma0": result["cfg"].gamma0,
            **result["stats"],
            **result["diagnostics"],
        })

    pd.DataFrame(rows_fixed).to_csv(RES / "dtcmv_mvs_fixed_gamma_summary.csv", index=False)
    pd.DataFrame(rows_cal).to_csv(RES / "dtcmv_mvs_equal_mean_summary.csv", index=False)

    rolling = []
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        r = core.rolling_common_state(result)
        r.insert(0, "eta0", eta)
        r.insert(1, "gamma0", result["cfg"].gamma0)
        rolling.append(r)
    pd.concat(rolling, ignore_index=True).to_csv(RES / "dtcmv_mvs_rolling_summary.csv", index=False)

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

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for eta, result in zip(calibrated_eta_grid, calibrated_results):
        ax.plot(
            times,
            result["glide"],
            label=fr"$\eta_0={eta:.2f}$, $\gamma_0={result['cfg'].gamma0:.2f}$",
        )
    ax.set_xlabel("Years since entry")
    ax.set_ylabel("Mass-weighted risky proportion")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_equal_mean_glidepaths.png", dpi=180)
    plt.close(fig)

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

    df_cal = pd.DataFrame(rows_cal)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(df_cal["cvar05"], df_cal["ucvar95"], marker="o")
    for _, row in df_cal.iterrows():
        ax.annotate(
            fr"$\eta_0={row['eta0']:.2f}$",
            (row["cvar05"], row["ucvar95"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Lower-tail CVaR (5%)")
    ax.set_ylabel("Upper-tail conditional mean (95%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_tail_frontier.png", dpi=180)
    plt.close(fig)

    rolling_df = pd.concat(rolling, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    metrics = [
        ("mean", "Conditional mean"),
        ("stdev", "Conditional stdev"),
        ("skewness", "Conditional skewness"),
    ]
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
        "baseline_dtcmv_gamma0": BASELINE_DTCMV_GAMMA0,
        "baseline_dtcmv_gamma0_source": "results/equal_mean_calibration.csv",
        "eta0_zero_nesting_diagnostics": nesting,
        "historical_solver_retained": "scripts/01_solvers/dtcmv_mvs_solver_20260713.py",
    }
    (RES / "dtcmv_mvs_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
