"""Equal-mean calibration metadata and reproducible target values.

The actual dynamic-programming / equilibrium solvers are in
pcmv_domv_solver_20260713.py and recompute_d0_rolling.py.  This file records
the calibrated coefficients used in the paper and verifies the achieved
means from the saved monthly baseline output.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TARGET_MEAN = 84.7770864
CALIBRATION = {
    "PCMV": ("gamma_p", 0.05),
    "DOMV": ("gamma_d", 0.0912608),
    "cTCMV": ("gamma_c", 0.059638671875),
    "dTCMV": ("gamma_0", 1.193359375),
    "CP": ("theta_cp", 0.4694735),
}

summary = pd.read_csv(RESULTS / "monthly_baseline_D0_summary.csv").set_index("strategy")
rows = []
for strategy, (parameter, value) in CALIBRATION.items():
    achieved = float(summary.loc[strategy, "mean"])
    rows.append({
        "strategy": strategy,
        "calibration_parameter": parameter,
        "calibrated_value": value,
        "target_mean": TARGET_MEAN,
        "achieved_mean": achieved,
        "absolute_mean_error": abs(achieved - TARGET_MEAN),
    })
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "equal_mean_calibration.csv", index=False)
print(out.to_string(index=False))
