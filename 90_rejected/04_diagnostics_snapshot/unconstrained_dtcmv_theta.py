from __future__ import annotations

"""Reproduce the unconstrained dTCMV coefficient theta_{rho_d}(t).

The integral equation is the unconstrained wealth-dependent TCMV equation used
in van Staden, Dang and Forsyth (2021), with the paper's baseline parameters.
We convert the Volterra integral equation into an autonomous two-state ODE and
integrate backward from T to 0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "diagnostics" / "dtcmv_u_shape_decomposition.csv"

T = 40.0
r = 0.015
mu = 0.055
beta = mu - r
sigma = 0.18
rho = 1.193359375
A = beta**2 / sigma**2

# Let
# I1(t)=int_t^T [r+beta*theta(s)-sigma^2*theta(s)^2] ds,
# I2(t)=int_t^T sigma^2*theta(s)^2 ds.
# The integral equation is
# theta(t)=A/[rho*beta] * [exp(-I1(t)) + rho exp(-I2(t)) - rho].
# Integrate I1, I2 backward from I1(T)=I2(T)=0. theta is algebraic.
def rhs(t: float, state: np.ndarray) -> np.ndarray:
    i1, i2 = state
    theta = A / (rho * beta) * (np.exp(-i1) + rho * np.exp(-i2) - rho)
    return np.array([
        -(r + beta * theta - sigma**2 * theta**2),
        -(sigma**2 * theta**2),
    ])


sol = solve_ivp(
    rhs,
    t_span=(T, 0.0),
    y0=np.array([0.0, 0.0]),
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
    dense_output=True,
)
if not sol.success:
    raise RuntimeError(sol.message)


def theta_at(t: float) -> float:
    i1, i2 = sol.sol(float(t))
    return float(A / (rho * beta) * (np.exp(-i1) + rho * np.exp(-i2) - rho))


frame = pd.read_csv(CSV)
# The first forward step starts from the exact deterministic state x0=1/12,
# not from the nearest/interpolated state-grid median.
if (frame["year"] == 0).any():
    idx0 = frame.index[frame["year"] == 0][0]
    x0 = 1.0 / 12.0
    h0 = float(frame.loc[idx0, "H"])
    frame.loc[idx0, "median_x"] = x0
    frame.loc[idx0, "H_over_x"] = h0 / x0
    frame.loc[idx0, "total_wealth_multiplier"] = 1.0 + h0 / x0
    frame.loc[idx0, "Gamma"] = rho / (x0 + h0)
    frame.loc[idx0, "effective_time_coefficient"] = (
        float(frame.loc[idx0, "strict_risky_fraction"]) / (1.0 + h0 / x0)
    )
frame["unconstrained_theta"] = [theta_at(y) for y in frame["year"]]
frame.to_csv(CSV, index=False)

print(frame[["year", "unconstrained_theta"]].to_string(index=False))
print(f"theta(T)={theta_at(T):.12f}")
