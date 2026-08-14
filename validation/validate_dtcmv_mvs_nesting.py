from __future__ import annotations

"""Regression diagnostic for the eta0=0 dTCMV-MVS -> dTCMV-MV nesting property."""

import sys
from pathlib import Path

import numpy as np

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

    # This guards against a recurrence of the former gamma0=2.5 mismatch,
    # which created an order-one terminal glide-path discrepancy.  It is not a
    # bitwise solver-equality test because the MV and MVS implementations use
    # different control-grid/refinement rules.
    if abs(diag["mv_nesting_final_glide_gap"]) > 0.15:
        raise AssertionError(
            "eta0=0 MVS no longer nests the main dTCMV MV baseline: "
            f"final glide gap={diag['mv_nesting_final_glide_gap']:.6f}"
        )


if __name__ == "__main__":
    main()
