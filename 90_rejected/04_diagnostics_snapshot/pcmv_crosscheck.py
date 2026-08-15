from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics"
OUT.mkdir(exist_ok=True)

T = 40.0
N = 480
dt = T / N
x0 = 1.0 / 12.0
xmax = 300.0
nx = 151
na = 15
ngh = 5
xpwr = 1.6
r = 0.015
mu = 0.055
beta = mu - r
sigma = 0.18
c = 1.0
mstar = 104.772

x = xmax * np.linspace(0.0, 1.0, nx) ** xpwr
h, w = hermgauss(ngh)
gx = np.sqrt(2.0) * h
gw = w / np.sqrt(np.pi)


@njit(cache=True)
def locate(xx: float) -> tuple[int, float]:
    if xx <= x[0]:
        return 0, 0.0
    if xx >= x[-1]:
        j = x.size - 2
        return j, (xx - x[j]) / (x[-1] - x[j])
    lo = 0
    hi = x.size - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xx:
            lo = mid
        else:
            hi = mid
    return lo, (xx - x[lo]) / (x[lo + 1] - x[lo])


@njit(cache=True)
def expected_value(next_value: np.ndarray, xx: float, risky_fraction: float) -> float:
    investment = risky_fraction * xx
    drift = (r * xx + c + beta * investment) * dt
    diffusion = sigma * investment * math.sqrt(dt)
    value = 0.0
    for k in range(gx.size):
        xp = max(xx + drift + diffusion * gx[k], 0.0)
        j, lam = locate(xp)
        value += gw[k] * (next_value[j] * (1.0 - lam) + next_value[j + 1] * lam)
    return value


@njit(cache=True)
def solve() -> tuple[np.ndarray, np.ndarray]:
    value = np.empty((N + 1, nx))
    policy = np.empty((N, nx))
    value[N] = (x - mstar) ** 2
    control_step = 1.0 / (na - 1)

    for n in range(N - 1, -1, -1):
        for i in range(nx):
            xx = x[i]
            if xx <= 1e-15:
                best_fraction = 0.0
                best_value = expected_value(value[n + 1], xx, 0.0)
            else:
                candidates = np.empty(na)
                best_index = 0
                best_value = 1e300
                for j in range(na):
                    candidate = expected_value(value[n + 1], xx, j * control_step)
                    candidates[j] = candidate
                    if candidate < best_value:
                        best_value = candidate
                        best_index = j
                best_fraction = best_index * control_step
                if 0 < best_index < na - 1:
                    denominator = (
                        candidates[best_index - 1]
                        - 2.0 * candidates[best_index]
                        + candidates[best_index + 1]
                    )
                    if denominator > 1e-14:
                        offset = 0.5 * (
                            candidates[best_index - 1] - candidates[best_index + 1]
                        ) / denominator
                        offset = min(max(offset, -1.0), 1.0)
                        refined_fraction = (best_index + offset) * control_step
                        refined_value = expected_value(
                            value[n + 1], xx, refined_fraction
                        )
                        if refined_value < best_value:
                            best_fraction = refined_fraction
            value[n, i] = expected_value(value[n + 1], xx, best_fraction)
            policy[n, i] = best_fraction
    return value, policy


@njit(cache=True)
def deposit(mass: np.ndarray, xx: float, weight: float) -> None:
    if xx <= x[0]:
        mass[0] += weight
        return
    if xx >= x[-1]:
        mass[-1] += weight
        return
    j, lam = locate(xx)
    mass[j] += weight * (1.0 - lam)
    mass[j + 1] += weight * lam


@njit(cache=True)
def forward(policy: np.ndarray) -> np.ndarray:
    pmf = np.zeros((N + 1, nx))
    current = np.zeros(nx)
    deposit(current, x0, 1.0)
    pmf[0] = current

    for n in range(N):
        next_mass = np.zeros(nx)
        if n == 0:
            initial_investment = np.interp(x0, x, policy[0] * x)
            risky_fraction = min(max(initial_investment / x0, 0.0), 1.0)
            investment = risky_fraction * x0
            drift = (r * x0 + c + beta * investment) * dt
            diffusion = sigma * investment * math.sqrt(dt)
            for k in range(gx.size):
                deposit(next_mass, max(x0 + drift + diffusion * gx[k], 0.0), gw[k])
        else:
            for i in range(nx):
                if current[i] <= 0.0:
                    continue
                xx = x[i]
                risky_fraction = policy[n, i]
                investment = risky_fraction * xx
                drift = (r * xx + c + beta * investment) * dt
                diffusion = sigma * investment * math.sqrt(dt)
                for k in range(gx.size):
                    deposit(
                        next_mass,
                        max(xx + drift + diffusion * gx[k], 0.0),
                        current[i] * gw[k],
                    )
        next_mass /= next_mass.sum()
        current = next_mass
        pmf[n + 1] = current
    return pmf


value, policy = solve()
pmf = forward(policy)
first = np.empty_like(value)
second = np.empty_like(value)
for n in range(N + 1):
    spline = CubicSpline(x, value[n], bc_type="natural")
    first[n] = spline(x, 1)
    second[n] = spline(x, 2)

valid = second[:-1] > 1e-7
unconstrained = np.full_like(policy, np.nan)
unconstrained[valid] = (
    -beta * first[:-1][valid] / (sigma**2 * second[:-1][valid])
)
predicted = np.clip(unconstrained / np.maximum(x[None, :], 1e-12), 0.0, 1.0)
predicted[:, 0] = 0.0

weights = np.where(valid, pmf[:-1], 0.0)
denominator = weights.sum()
tolerance = 0.02


def classify(array: np.ndarray) -> np.ndarray:
    return np.where(array <= tolerance, 0, np.where(array >= 1.0 - tolerance, 2, 1))


region_agreement = classify(policy) == classify(np.nan_to_num(predicted, nan=-9.0))
difference = np.where(valid, np.abs(policy - predicted), 0.0)
away = (
    valid
    & (np.minimum(predicted, 1.0 - predicted) > 0.05)
    & (np.minimum(policy, 1.0 - policy) > 0.05)
)
away_weights = np.where(away, pmf[:-1], 0.0)
values = difference[valid].ravel()
quantile_weights = pmf[:-1][valid].ravel()
order = np.argsort(values)
values = values[order]
quantile_weights = quantile_weights[order]
q95 = values[np.searchsorted(np.cumsum(quantile_weights) / quantile_weights.sum(), 0.95)]

row = {
    "strategy": "PCMV",
    "prob_weighted_region_agreement": (weights * region_agreement).sum() / denominator,
    "prob_weighted_policy_mae": (weights * difference).sum() / denominator,
    "robust_interior_region_agreement": (
        (away_weights * region_agreement).sum() / away_weights.sum()
    ),
    "q95_difference": q95,
    "derivative_invalid_mass_share": 1.0 - denominator / pmf[:-1].sum(),
}
print(row)
pd.DataFrame([row]).to_csv(OUT / "pcmv_free_boundary_crosscheck.csv", index=False)
np.savez_compressed(OUT / "pcmv_diag.npz", x=x, value=value, policy=policy, pmf=pmf)
