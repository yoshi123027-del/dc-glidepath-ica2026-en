from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numpy.polynomial.hermite import hermgauss


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

T = 40.0
N = 480
DT = T / N
X0 = 1.0 / 12.0
R = 0.015
MU = 0.055
BETA = MU - R
SIGMA = 0.18
CONTRIBUTION = 1.0
D_TERMINAL = 0.0
GAMMA0 = 1.193359375
N_CONTROLS = 15
N_GH = 5
X_POWER = 1.6


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
def interp(values: np.ndarray, j: int, weight: float) -> float:
    return values[j] * (1.0 - weight) + values[j + 1] * weight


@njit(cache=True)
def deposit_tracked(
    grid: np.ndarray,
    mass: np.ndarray,
    x: float,
    weight: float,
) -> tuple[float, float]:
    """Deposit all mass while separately recording excursions beyond the grid."""
    if x <= grid[0]:
        mass[0] += weight
        return (weight if x < grid[0] else 0.0), 0.0
    if x >= grid[-1]:
        mass[-1] += weight
        return 0.0, (weight if x > grid[-1] else 0.0)
    j, lam = locate(grid, x)
    mass[j] += weight * (1.0 - lam)
    mass[j + 1] += weight * lam
    return 0.0, 0.0


@njit(cache=True)
def tracked_forward(
    policy: np.ndarray,
    grid: np.ndarray,
    x_start: float,
    n_start: int,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward propagation with diagnostics recorded before normalization."""
    steps = policy.shape[0] - n_start
    p = np.zeros(grid.size)
    deposit_tracked(grid, p, x_start, 1.0)
    pmf = np.zeros((steps + 1, grid.size))
    pmf[0] = p
    glide = np.zeros(steps)
    raw_mass = np.zeros(steps)
    raw_mass_error = np.zeros(steps)
    lower_excursion = np.zeros(steps)
    upper_excursion = np.zeros(steps)

    for step in range(steps):
        n = n_start + step
        p_next = np.zeros(grid.size)
        if step == 0:
            investment_grid = policy[n] * grid
            j0, lam0 = locate(grid, x_start)
            investment = interp(investment_grid, j0, lam0)
            risky_fraction = min(max(investment / max(x_start, 1e-14), 0.0), 1.0)
            glide[step] = risky_fraction
            drift = (R * x_start + CONTRIBUTION + BETA * investment) * DT
            diffusion = SIGMA * investment * math.sqrt(DT)
            for k in range(gh_x.size):
                x_next = x_start + drift + diffusion * gh_x[k]
                under, over = deposit_tracked(grid, p_next, x_next, gh_w[k])
                lower_excursion[step] += under
                upper_excursion[step] += over
        else:
            for i in range(grid.size):
                if p[i] <= 0.0:
                    continue
                x = grid[i]
                risky_fraction = policy[n, i]
                investment = risky_fraction * x
                glide[step] += p[i] * risky_fraction
                drift = (R * x + CONTRIBUTION + BETA * investment) * DT
                diffusion = SIGMA * investment * math.sqrt(DT)
                for k in range(gh_x.size):
                    x_next = x + drift + diffusion * gh_x[k]
                    weight = p[i] * gh_w[k]
                    under, over = deposit_tracked(grid, p_next, x_next, weight)
                    lower_excursion[step] += under
                    upper_excursion[step] += over

        total = p_next.sum()
        raw_mass[step] = total
        raw_mass_error[step] = abs(total - 1.0)
        if total <= 0.0:
            raise ValueError("forward probability mass vanished")
        p = p_next / total
        pmf[step + 1] = p

    return pmf, glide, raw_mass, raw_mass_error, lower_excursion, upper_excursion


@njit(cache=True)
def expected_mq(
    m_next: np.ndarray,
    q_next: np.ndarray,
    grid: np.ndarray,
    x: float,
    risky_fraction: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[float, float]:
    investment = risky_fraction * x
    drift = (R * x + CONTRIBUTION + BETA * investment) * DT
    diffusion = SIGMA * investment * math.sqrt(DT)
    mean = 0.0
    second = 0.0
    for k in range(gh_x.size):
        x_next = max(x + drift + diffusion * gh_x[k], 0.0)
        j, lam = locate(grid, x_next)
        mean += gh_w[k] * interp(m_next, j, lam)
        second += gh_w[k] * interp(q_next, j, lam)
    return mean, second


@njit(cache=True)
def expected_mq_clamped(
    m_next: np.ndarray,
    q_next: np.ndarray,
    grid: np.ndarray,
    x: float,
    risky_fraction: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the same boundary-deposit kernel used by tracked_forward."""
    investment = risky_fraction * x
    drift = (R * x + CONTRIBUTION + BETA * investment) * DT
    diffusion = SIGMA * investment * math.sqrt(DT)
    mean = 0.0
    second = 0.0
    for k in range(gh_x.size):
        x_next = max(x + drift + diffusion * gh_x[k], 0.0)
        if x_next <= grid[0]:
            j = 0
            lam = 0.0
        elif x_next >= grid[-1]:
            j = grid.size - 2
            lam = 1.0
        else:
            j, lam = locate(grid, x_next)
        mean += gh_w[k] * interp(m_next, j, lam)
        second += gh_w[k] * interp(q_next, j, lam)
    return mean, second


@njit(cache=True)
def backward_moments(
    policy: np.ndarray,
    grid: np.ndarray,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.empty((policy.shape[0] + 1, grid.size))
    second = np.empty_like(mean)
    mean[-1] = grid + D_TERMINAL
    second[-1] = (grid + D_TERMINAL) ** 2
    for n in range(policy.shape[0] - 1, -1, -1):
        for i in range(grid.size):
            m, q = expected_mq_clamped(
                mean[n + 1], second[n + 1], grid, grid[i], policy[n, i], gh_x, gh_w
            )
            mean[n, i] = m
            second[n, i] = q
    return mean, second


@njit(cache=True)
def initial_backward_moments(
    policy: np.ndarray,
    grid: np.ndarray,
    mean: np.ndarray,
    second: np.ndarray,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> tuple[float, float]:
    investment_grid = policy[0] * grid
    j0, lam0 = locate(grid, X0)
    investment = interp(investment_grid, j0, lam0)
    risky_fraction = min(max(investment / X0, 0.0), 1.0)
    return expected_mq_clamped(mean[1], second[1], grid, X0, risky_fraction, gh_x, gh_w)


@njit(cache=True)
def solve_dtcmv(
    grid: np.ndarray,
    human_capital: np.ndarray,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
    gamma0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.empty((N + 1, grid.size))
    second = np.empty_like(mean)
    policy = np.empty((N, grid.size))
    mean[-1] = grid + D_TERMINAL
    second[-1] = (grid + D_TERMINAL) ** 2
    control_step = 1.0 / (N_CONTROLS - 1)

    for n in range(N - 1, -1, -1):
        for i in range(grid.size):
            x = grid[i]
            gamma = gamma0 / max(x + human_capital[n], 1e-14)
            values = np.empty(N_CONTROLS)
            best_index = 0
            best_value = -1e300
            for j in range(N_CONTROLS):
                risky_fraction = j * control_step
                m, q = expected_mq(mean[n + 1], second[n + 1], grid, x, risky_fraction, gh_x, gh_w)
                variance = q - m * m
                if variance < 0.0 and variance > -1e-9:
                    variance = 0.0
                objective = m - 0.5 * gamma * variance
                values[j] = objective
                if objective > best_value:
                    best_value = objective
                    best_index = j

            best_fraction = best_index * control_step
            if 0 < best_index < N_CONTROLS - 1 and x > 1e-15:
                denominator = values[best_index - 1] - 2.0 * values[best_index] + values[best_index + 1]
                if denominator < -1e-14:
                    offset = 0.5 * (values[best_index - 1] - values[best_index + 1]) / denominator
                    offset = min(max(offset, -1.0), 1.0)
                    refined_fraction = (best_index + offset) * control_step
                    m_refined, q_refined = expected_mq(
                        mean[n + 1], second[n + 1], grid, x, refined_fraction, gh_x, gh_w
                    )
                    variance_refined = q_refined - m_refined * m_refined
                    if variance_refined < 0.0 and variance_refined > -1e-9:
                        variance_refined = 0.0
                    objective_refined = m_refined - 0.5 * gamma * variance_refined
                    if objective_refined > best_value:
                        best_fraction = refined_fraction

            m, q = expected_mq(mean[n + 1], second[n + 1], grid, x, best_fraction, gh_x, gh_w)
            policy[n, i] = best_fraction
            mean[n, i] = m
            second[n, i] = q

    return mean, second, policy


def lower_cvar(values: np.ndarray, probabilities: np.ndarray, alpha: float = 0.05) -> float:
    remaining = alpha
    total = 0.0
    for value, probability in zip(values, probabilities):
        take = min(remaining, probability)
        total += take * value
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / alpha


def upper_cvar(values: np.ndarray, probabilities: np.ndarray, alpha: float = 0.05) -> float:
    return lower_cvar(values[::-1], probabilities[::-1], alpha)


def terminal_statistics(grid: np.ndarray, pmf: np.ndarray) -> dict[str, float]:
    probabilities = pmf / pmf.sum()
    mean = float(probabilities @ grid)
    variance = float(probabilities @ ((grid - mean) ** 2))
    standard_deviation = math.sqrt(max(variance, 0.0))
    cumulative = np.cumsum(probabilities)

    def quantile(level: float) -> float:
        return float(grid[min(np.searchsorted(cumulative, level), grid.size - 1)])

    return {
        "mean": mean,
        "stdev": standard_deviation,
        "q05": quantile(0.05),
        "q50": quantile(0.50),
        "q95": quantile(0.95),
        "lcvar05": lower_cvar(grid, probabilities),
        "ucvar95": upper_cvar(grid, probabilities),
    }


def gauss_hermite() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(N_GH)
    return np.sqrt(2.0) * nodes.astype(np.float64), (weights / np.sqrt(np.pi)).astype(np.float64)


def summarize_saved_baseline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays = np.load(RESULTS / "monthly_D0_policy_arrays.npz")
    gh_x, gh_w = gauss_hermite()
    configurations = [
        ("PCMV", arrays["xg_pc"], arrays["pcmv_policy"], arrays["pcmv_pmf"]),
        ("DOMV", arrays["xg_pc"], arrays["domv_policy"], arrays["domv_pmf"]),
        ("cTCMV", arrays["xg_tc"], arrays["ctcmv_policy"], arrays["ctcmv_pmf"]),
        ("dTCMV", arrays["xg_tc"], arrays["dtcmv_policy"], arrays["dtcmv_pmf"]),
    ]
    moment_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    time_rows: list[dict[str, float | str | int]] = []

    for strategy, grid, policy, saved_pmf in configurations:
        pmf, _, raw_mass, raw_error, lower, upper = tracked_forward(policy, grid, X0, 0, gh_x, gh_w)
        mean, second = backward_moments(policy, grid, gh_x, gh_w)
        backward_mean, backward_second = initial_backward_moments(policy, grid, mean, second, gh_x, gh_w)
        probabilities = pmf[-1]
        forward_mean = float(probabilities @ grid)
        forward_second = float(probabilities @ (grid**2))
        forward_variance = max(forward_second - forward_mean**2, 0.0)
        backward_variance = max(backward_second - backward_mean**2, 0.0)
        moment_rows.append(
            {
                "strategy": strategy,
                "backward_mean": backward_mean,
                "forward_mean": forward_mean,
                "epsilon_M": abs(backward_mean - forward_mean),
                "backward_second_moment": backward_second,
                "forward_second_moment": forward_second,
                "epsilon_Q": abs(backward_second - forward_second),
                "backward_stdev": math.sqrt(backward_variance),
                "forward_stdev": math.sqrt(forward_variance),
                "stdev_abs_residual": abs(math.sqrt(backward_variance) - math.sqrt(forward_variance)),
            }
        )
        summary_rows.append(
            {
                "strategy": strategy,
                "x_max": float(grid[-1]),
                "max_raw_mass_error": float(raw_error.max()),
                "max_step_lower_excursion_mass": float(lower.max()),
                "cumulative_lower_excursion_flux": float(lower.sum()),
                "max_step_upper_excursion_mass": float(upper.max()),
                "cumulative_upper_excursion_flux": float(upper.sum()),
                "terminal_upper_boundary_mass": float(pmf[-1, -1]),
                "max_l1_difference_from_saved_pmf": float(np.max(np.sum(abs(pmf - saved_pmf), axis=1))),
            }
        )
        for n in range(policy.shape[0]):
            time_rows.append(
                {
                    "strategy": strategy,
                    "step": n,
                    "year": n * DT,
                    "raw_mass_before_normalization": raw_mass[n],
                    "raw_mass_error": raw_error[n],
                    "lower_excursion_mass": lower[n],
                    "upper_excursion_mass": upper[n],
                }
            )

    return pd.DataFrame(moment_rows), pd.DataFrame(summary_rows), pd.DataFrame(time_rows)


def run_xmax_sensitivity() -> pd.DataFrame:
    gh_x, gh_w = gauss_hermite()
    human_capital = np.zeros(N + 1)
    human_capital[-1] = D_TERMINAL
    for n in range(N - 1, -1, -1):
        human_capital[n] = (CONTRIBUTION * DT + human_capital[n + 1]) / (1.0 + R * DT)

    rows: list[dict[str, float | int]] = []
    baseline_grid = 300.0 * np.linspace(0.0, 1.0, 151) ** X_POWER
    for x_max in (300.0, 450.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0):
        # Preserve every baseline node on [0, 300] and append a nested tail.
        # This isolates upper-domain sensitivity from changes in the economically
        # important low- and middle-balance mesh.
        tail = np.arange(302.0, x_max + 0.1, 2.0)
        grid = np.concatenate((baseline_grid, tail))
        n_x = grid.size
        _, _, policy = solve_dtcmv(grid, human_capital, gh_x, gh_w, GAMMA0)
        pmf, glide, raw_mass, raw_error, lower, upper = tracked_forward(policy, grid, X0, 0, gh_x, gh_w)
        diagnostic_mean, diagnostic_second = backward_moments(policy, grid, gh_x, gh_w)
        backward_mean, backward_second = initial_backward_moments(
            policy, grid, diagnostic_mean, diagnostic_second, gh_x, gh_w
        )
        statistics = terminal_statistics(grid, pmf[-1])
        forward_second = float(pmf[-1] @ (grid**2))
        rows.append(
            {
                "x_max": x_max,
                "n_x": n_x,
                **statistics,
                "average_glide": float(glide.mean()),
                "epsilon_M": abs(backward_mean - statistics["mean"]),
                "epsilon_Q": abs(backward_second - forward_second),
                "max_raw_mass_error": float(raw_error.max()),
                "max_step_upper_excursion_mass": float(upper.max()),
                "cumulative_upper_excursion_flux": float(upper.sum()),
                "terminal_upper_boundary_mass": float(pmf[-1, -1]),
                "max_step_lower_excursion_mass": float(lower.max()),
            }
        )

    frame = pd.DataFrame(rows)
    reference = frame.iloc[-1]
    for column in ("mean", "stdev", "q05", "q50", "q95", "lcvar05", "ucvar95", "average_glide"):
        frame[f"abs_diff_vs_xmax2100_{column}"] = abs(frame[column] - float(reference[column]))
    return frame


def build_nested_grid(x_max: float) -> np.ndarray:
    baseline_grid = 300.0 * np.linspace(0.0, 1.0, 151) ** X_POWER
    tail = np.arange(302.0, x_max + 0.1, 2.0)
    return np.concatenate((baseline_grid, tail))


def human_capital_path() -> np.ndarray:
    human_capital = np.zeros(N + 1)
    human_capital[-1] = D_TERMINAL
    for n in range(N - 1, -1, -1):
        human_capital[n] = (CONTRIBUTION * DT + human_capital[n + 1]) / (1.0 + R * DT)
    return human_capital


def calibrate_xmax900_equal_mean(target_mean: float = 84.77708640828797) -> tuple[pd.DataFrame, pd.DataFrame]:
    gh_x, gh_w = gauss_hermite()
    grid = build_nested_grid(900.0)
    human_capital = human_capital_path()
    history: list[dict[str, float | int]] = []

    def evaluate(gamma0: float, iteration: int) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        _, _, policy = solve_dtcmv(grid, human_capital, gh_x, gh_w, gamma0)
        forward = tracked_forward(policy, grid, X0, 0, gh_x, gh_w)
        pmf, glide, _, raw_error, lower, upper = forward
        statistics = terminal_statistics(grid, pmf[-1])
        history.append(
            {
                "iteration": iteration,
                "gamma0": gamma0,
                "mean": statistics["mean"],
                "mean_residual": statistics["mean"] - target_mean,
                "stdev": statistics["stdev"],
                "average_glide": float(glide.mean()),
                "max_step_upper_excursion_mass": float(upper.max()),
            }
        )
        return statistics["mean"], policy, glide, forward

    lower_gamma = 0.6
    upper_gamma = 1.8
    lower_mean, _, _, _ = evaluate(lower_gamma, -2)
    upper_mean, _, _, _ = evaluate(upper_gamma, -1)
    if not (lower_mean >= target_mean >= upper_mean):
        raise RuntimeError(
            f"calibration bracket failed: mean({lower_gamma})={lower_mean}, "
            f"target={target_mean}, mean({upper_gamma})={upper_mean}"
        )

    best: tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...]] | None = None
    best_gamma = float("nan")
    for iteration in range(18):
        gamma0 = 0.5 * (lower_gamma + upper_gamma)
        candidate = evaluate(gamma0, iteration)
        mean = candidate[0]
        if best is None or abs(mean - target_mean) < abs(best[0] - target_mean):
            best = candidate
            best_gamma = gamma0
        if mean > target_mean:
            lower_gamma = gamma0
        else:
            upper_gamma = gamma0

    assert best is not None
    mean, policy, glide, forward = best
    pmf, _, _, raw_error, lower, upper = forward
    diagnostic_mean, diagnostic_second = backward_moments(policy, grid, gh_x, gh_w)
    backward_mean, backward_second = initial_backward_moments(
        policy, grid, diagnostic_mean, diagnostic_second, gh_x, gh_w
    )
    forward_second = float(pmf[-1] @ (grid**2))
    statistics = terminal_statistics(grid, pmf[-1])
    summary = pd.DataFrame(
        [
            {
                "x_max": 900.0,
                "n_x": grid.size,
                "target_mean": target_mean,
                "gamma0": best_gamma,
                **statistics,
                "average_glide": float(glide.mean()),
                "epsilon_M": abs(backward_mean - mean),
                "epsilon_Q": abs(backward_second - forward_second),
                "max_raw_mass_error": float(raw_error.max()),
                "max_step_upper_excursion_mass": float(upper.max()),
                "cumulative_upper_excursion_flux": float(upper.sum()),
                "terminal_upper_boundary_mass": float(pmf[-1, -1]),
                "max_step_lower_excursion_mass": float(lower.max()),
            }
        ]
    )
    return pd.DataFrame(history), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 5.11 moment, mass, and upper-grid diagnostics")
    parser.add_argument(
        "--skip-xmax-sensitivity",
        action="store_true",
        help="Only diagnose the saved monthly baseline without re-solving dTCMV",
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    moments, mass_summary, mass_by_time = summarize_saved_baseline()
    moments.to_csv(RESULTS / "moment_consistency_D0_N480.csv", index=False)
    mass_summary.to_csv(RESULTS / "mass_truncation_summary_D0_N480.csv", index=False)
    mass_by_time.to_csv(RESULTS / "mass_truncation_by_time_D0_N480.csv", index=False)
    print("Moment consistency")
    print(moments.to_string(index=False))
    print("\nMass and truncation summary")
    print(mass_summary.to_string(index=False))

    if not args.skip_xmax_sensitivity:
        sensitivity = run_xmax_sensitivity()
        sensitivity.to_csv(RESULTS / "xmax_sensitivity_dtcmv_D0_N480.csv", index=False)
        print("\ndTCMV x_max sensitivity")
        print(sensitivity.to_string(index=False))
        calibration, calibrated_summary = calibrate_xmax900_equal_mean()
        calibration.to_csv(RESULTS / "xmax900_equal_mean_calibration_dtcmv_D0_N480.csv", index=False)
        calibrated_summary.to_csv(RESULTS / "xmax900_equal_mean_dtcmv_D0_N480.csv", index=False)
        print("\ndTCMV x_max=900 equal-mean recalibration")
        print(calibrated_summary.to_string(index=False))


if __name__ == "__main__":
    main()
