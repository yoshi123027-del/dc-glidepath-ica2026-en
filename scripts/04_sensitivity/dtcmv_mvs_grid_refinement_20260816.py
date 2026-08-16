from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOLVER_PATH = ROOT / "scripts" / "01_solvers" / "dtcmv_mvs_solver_20260713.py"
RES = ROOT / "results"

spec = importlib.util.spec_from_file_location("dtcmv_mvs_solver", SOLVER_PATH)
solver = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(solver)


def run_configuration(x_max: float, n_x: int, n_controls: int):
    base = solver.Config(x_max=x_max, n_x=n_x, n_controls=n_controls, gamma0=2.5, eta0=0.0)
    baseline = solver.solve_case(base)
    maps = baseline["maps"]
    rows = []
    results = []
    for eta in (0.0, 0.5, 1.0, 2.0):
        if eta == 0.0:
            result = baseline
        else:
            cfg = solver.Config(**{**asdict(base), "eta0": eta})
            result = solver.solve_case(cfg, maps=maps)
        results.append(result)
        rows.append({
            "x_max": x_max,
            "n_x": n_x,
            "n_controls": n_controls,
            "n_steps": base.n_steps,
            "n_gh": base.n_gh,
            "eta0": eta,
            "gamma0": 2.5,
            **result["stats"],
            **result["diagnostics"],
        })
    return rows, results


def main() -> None:
    # The first row reproduces the coarse configuration that caused all q05 values
    # to land on one wealth-grid node. Subsequent rows progressively refine the
    # state/control grid and enlarge the upper wealth domain.
    configs = [
        (300.0, 151, 25),
        (300.0, 601, 49),
        (300.0, 1201, 49),
        (600.0, 2401, 97),
        (900.0, 3601, 97),
    ]
    all_rows = []
    final_results = None
    for x_max, n_x, n_controls in configs:
        rows, results = run_configuration(x_max, n_x, n_controls)
        all_rows.extend(rows)
        if (x_max, n_x, n_controls) == configs[-1]:
            final_results = results

    pd.DataFrame(all_rows).to_csv(RES / "dtcmv_mvs_grid_refinement_summary.csv", index=False)

    assert final_results is not None
    final_rows = []
    wide = {"wealth": final_results[0]["x_grid"]}
    for eta, result in zip((0.0, 0.5, 1.0, 2.0), final_results):
        final_rows.append({
            "eta0": eta,
            "gamma0": 2.5,
            **result["stats"],
            **result["diagnostics"],
        })
        pmf = result["pmf"][-1]
        tag = f"{eta:.2f}"
        wide[f"pmf_eta_{tag}"] = pmf
        wide[f"cdf_eta_{tag}"] = np.cumsum(pmf)

    pd.DataFrame(final_rows).to_csv(RES / "dtcmv_mvs_fixed_gamma_refined_summary.csv", index=False)
    pd.DataFrame(wide).to_csv(RES / "dtcmv_mvs_fixed_gamma_refined_terminal_distribution.csv", index=False)


if __name__ == "__main__":
    main()
