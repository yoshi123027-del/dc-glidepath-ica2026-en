"""Export terminal distributions and statistics for Figure 6 fixed-gamma dTCMV-MVS policies.

Uses the exact same Config/solver/forward Gauss-Hermite engine as
scripts/01_solvers/dtcmv_mvs_solver_20260713.py. The four policies are
eta0 = 0, 0.5, 1, 2 with gamma0 fixed at 2.5.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = ROOT / "scripts" / "01_solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from dtcmv_mvs_solver_20260713 import (
    Config, solve_case, quantile, lower_cvar, upper_cvar,
)

FIG = ROOT / "figs"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

ETA_GRID = [0.0, 0.5, 1.0, 2.0]
GAMMA0 = 2.5


def extended_stats(result):
    cfg = result["cfg"]
    values = result["x_grid"] + cfg.D
    p = result["pmf"][-1].copy()
    p /= p.sum()
    mean = float(p @ values)
    var = float(p @ ((values - mean) ** 2))
    sd = float(np.sqrt(max(var, 0.0)))
    cm3 = float(p @ ((values - mean) ** 3))
    out = {
        "eta0": cfg.eta0,
        "gamma0": cfg.gamma0,
        "mean": mean,
        "stdev": sd,
        "variance": var,
        "third_central_moment": cm3,
        "skewness": cm3 / (sd ** 3 + 1e-30),
    }
    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        out[f"q{int(round(100*q)):02d}"] = quantile(values, p, q)
    out["cvar05"] = lower_cvar(values, p, 0.05)
    out["ucvar95"] = upper_cvar(values, p, 0.05)
    out.update(result["diagnostics"])
    return out


def make_summary_plot(results, summary):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

    for color, result in zip(colors, results):
        eta = result["cfg"].eta0
        values = result["x_grid"] + result["cfg"].D
        p = result["pmf"][-1]
        axes[0].plot(values, np.cumsum(p), color=color, label=fr"$\eta_0={eta:.2f}$")
        mask = p > 1e-14
        kde = gaussian_kde(values[mask], weights=p[mask], bw_method=0.18)
        xx = np.linspace(0.0, 260.0, 1000)
        axes[1].plot(xx, kde(xx), color=color, label=fr"$\eta_0={eta:.2f}$")

    axes[0].set(xlim=(0, 260), ylim=(0, 1), xlabel="Terminal DC wealth", ylabel="CDF",
                title="Terminal CDF by skewness coefficient")
    axes[1].set(xlim=(0, 260), xlabel="Terminal DC wealth", ylabel="Smoothed density",
                title="Smoothed terminal density")
    axes[0].legend(ncol=2)
    for ax in axes[:2]:
        ax.grid(alpha=0.25)

    y = np.arange(len(ETA_GRID))[::-1]
    for yy, color, (_, row) in zip(y, colors, summary.iterrows()):
        q05, med, mean, q95 = row.q05, row.q50, row["mean"], row.q95
        axes[2].hlines(yy, q05, q95, color=color)
        axes[2].vlines([q05, q95], yy-0.10, yy+0.10, color=color)
        axes[2].plot(med, yy, "o", color=color, ms=5)
        axes[2].plot(mean, yy, marker="D", mfc="white", mec=color, mew=1.2, ms=5, ls="None")
        axes[2].annotate(f"{q05:.2f}", (q05, yy), xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=8)
        axes[2].annotate(f"{q95:.2f}", (q95, yy), xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=8)
        axes[2].annotate(f"{med:.2f}", (med, yy), xytext=(-8 if abs(mean-med)<5 else 0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        axes[2].annotate(f"{mean:.2f}", (mean, yy), xytext=(8 if abs(mean-med)<5 else 0, 24), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    axes[2].set_yticks(y, [fr"$\eta_0={e:.2f}$" for e in ETA_GRID])
    axes[2].set(xlim=(0, 260), xlabel="Terminal DC wealth", title="q05, median, mean, and q95")
    axes[2].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG / "fig_dtcmv_mvs_fixed_gamma_terminal_distribution_en.png", dpi=180)
    plt.close(fig)


def main():
    base = Config(gamma0=GAMMA0, eta0=0.0)
    baseline = solve_case(base)
    maps = baseline["maps"]
    results = []
    for eta in ETA_GRID:
        cfg = Config(**{**asdict(base), "eta0": eta, "gamma0": GAMMA0})
        results.append(solve_case(cfg, maps=maps))

    summary = pd.DataFrame([extended_stats(r) for r in results])
    summary.to_csv(RES / "dtcmv_mvs_fixed_gamma_extended_summary.csv", index=False)

    dist = pd.DataFrame({"wealth": baseline["x_grid"]})
    for eta, result in zip(ETA_GRID, results):
        p = result["pmf"][-1]
        dist[f"pmf_eta_{eta:.2f}"] = p
        dist[f"cdf_eta_{eta:.2f}"] = np.cumsum(p)
    dist.to_csv(RES / "dtcmv_mvs_fixed_gamma_terminal_distribution.csv", index=False)

    np.savez_compressed(
        RES / "dtcmv_mvs_fixed_gamma_terminal_arrays.npz",
        x_grid=baseline["x_grid"],
        eta_grid=np.asarray(ETA_GRID),
        pmf=np.stack([r["pmf"][-1] for r in results]),
        glide=np.stack([r["glide"] for r in results]),
    )
    make_summary_plot(results, summary)


if __name__ == "__main__":
    main()
