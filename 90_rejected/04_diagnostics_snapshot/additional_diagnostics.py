from __future__ import annotations

"""Recompute constrained cTCMV/dTCMV policies used by additional diagnostics."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numpy.polynomial.hermite import hermgauss

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
x_power = 1.6
r = 0.015
mu = 0.055
beta = mu - r
sigma = 0.18
contribution = 1.0
terminal_benefit = 0.0
gamma_c = 0.059638671875
gamma0 = 1.193359375

u = np.linspace(0.0, 1.0, nx)
xgrid = xmax * u**x_power
h, w = hermgauss(ngh)
gh_x = np.sqrt(2.0) * h
gh_w = w / np.sqrt(np.pi)

human_capital = np.zeros(N + 1)
human_capital[-1] = terminal_benefit
for n in range(N - 1, -1, -1):
    human_capital[n] = (
        contribution * dt + human_capital[n + 1]
    ) / (1.0 + r * dt)


@njit(cache=True)
def locate(grid: np.ndarray, x: float) -> tuple[int, float]:
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[-1]:
        j = grid.size - 2
        return j, (x - grid[j]) / (grid[-1] - grid[j])
    lo = 0
    hi = grid.size - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo, (x - grid[lo]) / (grid[lo + 1] - grid[lo])


@njit(cache=True)
def interpolate(values: np.ndarray, j: int, weight: float) -> float:
    return values[j] * (1.0 - weight) + values[j + 1] * weight


@njit(cache=True)
def expected_moments(
    mean_next: np.ndarray,
    second_next: np.ndarray,
    wealth: float,
    risky_fraction: float,
) -> tuple[float, float]:
    investment = risky_fraction * wealth
    drift = (r * wealth + contribution + beta * investment) * dt
    diffusion = sigma * investment * math.sqrt(dt)
    mean = 0.0
    second = 0.0
    for k in range(gh_x.size):
        next_wealth = max(wealth + drift + diffusion * gh_x[k], 0.0)
        j, lam = locate(xgrid, next_wealth)
        mean += gh_w[k] * interpolate(mean_next, j, lam)
        second += gh_w[k] * interpolate(second_next, j, lam)
    return mean, second


@njit(cache=True)
def solve(strategy_code: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.empty((N + 1, nx))
    second = np.empty((N + 1, nx))
    policy = np.empty((N, nx))
    mean[N] = xgrid + terminal_benefit
    second[N] = (xgrid + terminal_benefit) ** 2
    control_step = 1.0 / (na - 1)

    for n in range(N - 1, -1, -1):
        for i in range(nx):
            wealth = xgrid[i]
            gamma = (
                gamma_c
                if strategy_code == 0
                else gamma0 / max(wealth + human_capital[n], 1e-14)
            )
            values = np.empty(na)
            best_index = 0
            best_value = -1e300
            for j in range(na):
                risky_fraction = j * control_step
                m, q = expected_moments(
                    mean[n + 1], second[n + 1], wealth, risky_fraction
                )
                variance = q - m * m
                if -1e-9 < variance < 0.0:
                    variance = 0.0
                objective = m - 0.5 * gamma * variance
                values[j] = objective
                if objective > best_value:
                    best_value = objective
                    best_index = j

            best_fraction = best_index * control_step
            if 0 < best_index < na - 1 and wealth > 1e-15:
                denominator = (
                    values[best_index - 1]
                    - 2.0 * values[best_index]
                    + values[best_index + 1]
                )
                if denominator < -1e-14:
                    offset = 0.5 * (
                        values[best_index - 1] - values[best_index + 1]
                    ) / denominator
                    offset = min(max(offset, -1.0), 1.0)
                    refined = (best_index + offset) * control_step
                    m, q = expected_moments(
                        mean[n + 1], second[n + 1], wealth, refined
                    )
                    variance = q - m * m
                    if -1e-9 < variance < 0.0:
                        variance = 0.0
                    if m - 0.5 * gamma * variance > best_value:
                        best_fraction = refined

            m, q = expected_moments(
                mean[n + 1], second[n + 1], wealth, best_fraction
            )
            mean[n, i] = m
            second[n, i] = q
            policy[n, i] = best_fraction
    return mean, second, policy


@njit(cache=True)
def deposit(mass: np.ndarray, wealth: float, weight: float) -> None:
    if wealth <= xgrid[0]:
        mass[0] += weight
        return
    if wealth >= xgrid[-1]:
        mass[-1] += weight
        return
    j, lam = locate(xgrid, wealth)
    mass[j] += weight * (1.0 - lam)
    mass[j + 1] += weight * lam


@njit(cache=True)
def forward(policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pmf = np.zeros((N + 1, nx))
    current = np.zeros(nx)
    deposit(current, x0, 1.0)
    pmf[0] = current
    glide = np.zeros(N)

    for n in range(N):
        next_mass = np.zeros(nx)
        if n == 0:
            investment = np.interp(x0, xgrid, policy[0] * xgrid)
            risky_fraction = min(max(investment / x0, 0.0), 1.0)
            glide[0] = risky_fraction
            drift = (r * x0 + contribution + beta * investment) * dt
            diffusion = sigma * investment * math.sqrt(dt)
            for k in range(gh_x.size):
                deposit(
                    next_mass,
                    max(x0 + drift + diffusion * gh_x[k], 0.0),
                    gh_w[k],
                )
        else:
            for i in range(nx):
                if current[i] <= 0.0:
                    continue
                wealth = xgrid[i]
                risky_fraction = policy[n, i]
                glide[n] += current[i] * risky_fraction
                investment = risky_fraction * wealth
                drift = (r * wealth + contribution + beta * investment) * dt
                diffusion = sigma * investment * math.sqrt(dt)
                for k in range(gh_x.size):
                    deposit(
                        next_mass,
                        max(wealth + drift + diffusion * gh_x[k], 0.0),
                        current[i] * gh_w[k],
                    )
        next_mass /= next_mass.sum()
        current = next_mass
        pmf[n + 1] = current
    return pmf, glide


def quantile(probability: np.ndarray, level: float) -> float:
    cumulative = np.cumsum(probability / probability.sum())
    return float(xgrid[min(np.searchsorted(cumulative, level), nx - 1)])


print("solving cTCMV")
mean_c, second_c, policy_c = solve(0)
print("solving dTCMV")
mean_d, second_d, policy_d = solve(1)
print("forward propagation")
pmf_c, glide_c = forward(policy_c)
pmf_d, glide_d = forward(policy_d)

np.savez_compressed(
    OUT / "tc_diagnostics_arrays.npz",
    xg=xgrid,
    H=human_capital,
    Mc=mean_c,
    Qc=second_c,
    Pc=policy_c,
    Md=mean_d,
    Qd=second_d,
    Pd=policy_d,
    pmfc=pmf_c,
    pmfd=pmf_d,
    gc=glide_c,
    gd=glide_d,
)

rows = []
for year in [0, 10, 20, 30, 35, 39]:
    n = min(int(round(year / dt)), N - 1)
    median_wealth = x0 if year == 0 else quantile(pmf_d[n], 0.5)
    risky_fraction = float(np.interp(median_wealth, xgrid, policy_d[n]))
    h_value = human_capital[n]
    multiplier = 1.0 + h_value / max(median_wealth, 1e-14)
    rows.append(
        {
            "year": year,
            "median_x": median_wealth,
            "H": h_value,
            "H_over_x": h_value / max(median_wealth, 1e-14),
            "total_wealth_multiplier": multiplier,
            "Gamma": gamma0 / (median_wealth + h_value),
            "strict_risky_fraction": risky_fraction,
            "effective_time_coefficient": risky_fraction / multiplier,
            "upper_bound": int(risky_fraction >= 0.999),
        }
    )

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "dtcmv_u_shape_decomposition.csv", index=False)
print(frame.to_string(index=False))
