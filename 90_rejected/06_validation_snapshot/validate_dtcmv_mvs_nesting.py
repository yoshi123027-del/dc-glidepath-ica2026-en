from __future__ import annotations

"""Regression diagnostic for the eta0=0 dTCMV-MVS -> dTCMV-MV nesting property."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = ROOT / "scripts" / "01_solvers"
sys.path.insert(0, str(SOLVER_DIR))
import dtcmv_mvs_solver_20260814 as m


def main() -> None:
    cfg = m.baseline_config(eta0=0.0)
    result = m.core.solve_case(cfg)
    diag = m.nesting_diagnostics(result)

    if not diag:
        raise FileNotFoundError(
            "results/monthly_D0_policy_arrays.npz is required for the MV nesting comparison"
        )

    print("dTCMV-MVS eta0=0 nesting diagnostic")
    print(f"gamma0 = {cfg.gamma0:.9f}")
    for key, value in diag.items():
        print(f"{key}: {value:.9f}")

    # eta0=0 has the same economic objective as dTCMV-MV.  The two numerical
    # implementations are not bitwise identical because the MVS solver uses a
    # global control-grid search without the MV solver's parabolic refinement.
    # These tolerances are therefore numerical, but tight enough to prevent a
    # recurrence of the historical gamma0=2.5 mismatch.
    if diag["mv_nesting_mean_abs_glide_gap"] > 0.02:
        raise AssertionError(
            "eta0=0 MVS mean glide gap is too large: "
            f"{diag['mv_nesting_mean_abs_glide_gap']:.6f}"
        )
    if diag["mv_nesting_max_abs_glide_gap"] > 0.05:
        raise AssertionError(
            "eta0=0 MVS maximum glide gap is too large: "
            f"{diag['mv_nesting_max_abs_glide_gap']:.6f}"
        )
    if abs(diag["mv_nesting_final_glide_gap"]) > 0.01:
        raise AssertionError(
            "eta0=0 MVS terminal glide gap is too large: "
            f"{diag['mv_nesting_final_glide_gap']:.6f}"
        )


if __name__ == "__main__":
    main()
