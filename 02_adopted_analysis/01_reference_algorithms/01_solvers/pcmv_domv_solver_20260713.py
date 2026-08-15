from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads, get_num_threads
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class Config:
    T: float = 40.0
    n_steps: int = 80
    x0: float = 1.0 / 12.0
    x_max: float = 300.0
    n_x: int = 401
    x_power: float = 1.6
    n_controls: int = 25
    n_gh: int = 7
    r: float = 0.015
    mu: float = 0.055
    sigma: float = 0.180
    c: float = 1.0
    D: float = 0.0
    gamma_p: float = 0.050
    gamma_d: float = 0.0912608
    target_min: float = 40.0
    target_max: float = 340.0
    target_step: float = 5.0
    parabolic_control_refinement: bool = True

    @property
    def beta(self) -> float:
        return self.mu - self.r

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


def make_x_grid(cfg: Config) -> np.ndarray:
    u = np.linspace(0.0, 1.0, cfg.n_x)
    return cfg.x_max * np.power(u, cfg.x_power)


def gh_nodes_weights(n_gh: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = hermgauss(n_gh)
    return np.sqrt(2.0) * h.astype(np.float64), (w / np.sqrt(np.pi)).astype(np.float64)


@njit(cache=True)
def _interp_linear(x_grid: np.ndarray, values: np.ndarray, x: float) -> float:
    n = x_grid.size
    if x <= x_grid[0]:
        return values[0]
    if x >= x_grid[n - 1]:
        # Linear extrapolation. Reachable-mass diagnostics verify that this is immaterial.
        dx = x_grid[n - 1] - x_grid[n - 2]
        return values[n - 1] + (x - x_grid[n - 1]) * (values[n - 1] - values[n - 2]) / dx
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    lam = (x - x_grid[lo]) / (x_grid[lo + 1] - x_grid[lo])
    return values[lo] * (1.0 - lam) + values[lo + 1] * lam


@njit(cache=True)
def _expected_value(
    next_values: np.ndarray,
    x_grid: np.ndarray,
    x: float,
    fraction: float,
    dt: float,
    r: float,
    beta: float,
    sigma: float,
    c: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> float:
    pi = fraction * x
    drift = (r * x + c + beta * pi) * dt
    diffusion_scale = sigma * pi * math.sqrt(dt)
    out = 0.0
    for k in range(gh_x.size):
        xp = x + drift + diffusion_scale * gh_x[k]
        if xp < 0.0:
            xp = 0.0
        out += gh_w[k] * _interp_linear(x_grid, next_values, xp)
    return out


@njit(cache=True)
def _solve_one_target(
    target: float,
    n_steps: int,
    x_grid: np.ndarray,
    dt: float,
    r: float,
    beta: float,
    sigma: float,
    c: float,
    D: float,
    n_controls: int,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
    refine_control: bool,
    V_out: np.ndarray,
    M_out: np.ndarray,
    policy_out: np.ndarray,
) -> None:
    nx = x_grid.size
    # terminal conditions
    for i in range(nx):
        W = x_grid[i] + D
        V_out[n_steps, i] = (W - target) * (W - target)
        M_out[n_steps, i] = W
    da = 1.0 / (n_controls - 1)
    fvals = np.empty(n_controls, dtype=np.float64)

    for n in range(n_steps - 1, -1, -1):
        Vnext = V_out[n + 1]
        Mnext = M_out[n + 1]
        for i in range(nx):
            x = x_grid[i]
            if x <= 1e-15:
                best_a = 0.0
                best_v = _expected_value(Vnext, x_grid, x, 0.0, dt, r, beta, sigma, c, gh_x, gh_w)
            else:
                best_j = 0
                best_v = 1e300
                for j in range(n_controls):
                    a = j * da
                    val = _expected_value(Vnext, x_grid, x, a, dt, r, beta, sigma, c, gh_x, gh_w)
                    fvals[j] = val
                    if val < best_v:
                        best_v = val
                        best_j = j
                best_a = best_j * da
                # Local parabolic refinement of the bounded 1-D control search.
                if refine_control and best_j > 0 and best_j < n_controls - 1:
                    fl = fvals[best_j - 1]
                    fm = fvals[best_j]
                    fr = fvals[best_j + 1]
                    denom = fl - 2.0 * fm + fr
                    if denom > 1e-14:
                        offset = 0.5 * (fl - fr) / denom
                        if offset < -1.0:
                            offset = -1.0
                        if offset > 1.0:
                            offset = 1.0
                        a_ref = (best_j + offset) * da
                        if a_ref < 0.0:
                            a_ref = 0.0
                        if a_ref > 1.0:
                            a_ref = 1.0
                        v_ref = _expected_value(Vnext, x_grid, x, a_ref, dt, r, beta, sigma, c, gh_x, gh_w)
                        if v_ref < best_v:
                            best_v = v_ref
                            best_a = a_ref
            V_out[n, i] = best_v
            policy_out[n, i] = best_a
            M_out[n, i] = _expected_value(Mnext, x_grid, x, best_a, dt, r, beta, sigma, c, gh_x, gh_w)


@njit(parallel=True, cache=True)
def solve_target_family_numba(
    targets: np.ndarray,
    n_steps: int,
    x_grid: np.ndarray,
    dt: float,
    r: float,
    beta: float,
    sigma: float,
    c: float,
    D: float,
    n_controls: int,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
    refine_control: bool,
    V_all: np.ndarray,
    M_all: np.ndarray,
    P_all: np.ndarray,
) -> None:
    for j in prange(targets.size):
        _solve_one_target(
            targets[j], n_steps, x_grid, dt, r, beta, sigma, c, D,
            n_controls, gh_x, gh_w, refine_control,
            V_all[j], M_all[j], P_all[j]
        )


def solve_target_family(cfg: Config, targets: np.ndarray) -> Dict[str, np.ndarray]:
    x_grid = make_x_grid(cfg)
    gx, gw = gh_nodes_weights(cfg.n_gh)
    nt = len(targets)
    V = np.empty((nt, cfg.n_steps + 1, cfg.n_x), dtype=np.float64)
    M = np.empty_like(V)
    P = np.empty((nt, cfg.n_steps, cfg.n_x), dtype=np.float32)
    start = time.time()
    solve_target_family_numba(
        targets.astype(np.float64), cfg.n_steps, x_grid, cfg.dt,
        cfg.r, cfg.beta, cfg.sigma, cfg.c, cfg.D,
        cfg.n_controls, gx, gw, cfg.parabolic_control_refinement,
        V, M, P
    )
    return {
        "targets": targets,
        "x_grid": x_grid,
        "V": V,
        "M": M,
        "policy": P,
        "solve_seconds": np.array([time.time() - start]),
        "gh_x": gx,
        "gh_w": gw,
    }


def interp_x_vector(x_grid: np.ndarray, y_by_target: np.ndarray, x: float) -> np.ndarray:
    # y_by_target shape (n_target, n_x)
    if x <= x_grid[0]:
        return y_by_target[:, 0]
    if x >= x_grid[-1]:
        slope = (y_by_target[:, -1] - y_by_target[:, -2]) / (x_grid[-1] - x_grid[-2])
        return y_by_target[:, -1] + (x - x_grid[-1]) * slope
    j = np.searchsorted(x_grid, x) - 1
    lam = (x - x_grid[j]) / (x_grid[j + 1] - x_grid[j])
    return (1 - lam) * y_by_target[:, j] + lam * y_by_target[:, j + 1]


def parabolic_max_uniform(z: np.ndarray, s: np.ndarray) -> Tuple[float, int, bool]:
    j = int(np.argmax(s))
    if j == 0 or j == len(z) - 1:
        return float(z[j]), j, True
    h = z[1] - z[0]
    denom = s[j - 1] - 2.0 * s[j] + s[j + 1]
    if denom >= -1e-14:
        return float(z[j]), j, False
    offset = 0.5 * (s[j - 1] - s[j + 1]) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    return float(z[j] + h * offset), j, False


def interpolate_target_array(targets: np.ndarray, arr: np.ndarray, zstar: float, n: int, i: int) -> float:
    if zstar <= targets[0]:
        return float(arr[0, n, i])
    if zstar >= targets[-1]:
        return float(arr[-1, n, i])
    j = int(np.searchsorted(targets, zstar) - 1)
    lam = (zstar - targets[j]) / (targets[j + 1] - targets[j])
    return float((1.0 - lam) * arr[j, n, i] + lam * arr[j + 1, n, i])


def build_domv_policy(cfg: Config, family: Dict[str, np.ndarray], target_stride: int = 1) -> Dict[str, np.ndarray]:
    targets = family["targets"][::target_stride]
    V = family["V"][::target_stride]
    M = family["M"][::target_stride]
    P = family["policy"][::target_stride]
    nT = cfg.n_steps
    nx = cfg.n_x
    policy = np.zeros((nT, nx), dtype=np.float64)
    zstar = np.zeros((nT, nx), dtype=np.float64)
    fixed_resid = np.zeros((nT, nx), dtype=np.float64)
    direct_same = np.zeros((nT, nx), dtype=np.float64)
    boundary = np.zeros((nT, nx), dtype=np.float64)
    for n in range(nT):
        for i in range(nx):
            # Exact embedding score for J=E[W]-gamma/2 Var(W).
            S = targets - 1.0 / (2.0 * cfg.gamma_d) - 0.5 * cfg.gamma_d * V[:, n, i]
            z, j, hit = parabolic_max_uniform(targets, S)
            zstar[n, i] = z
            policy[n, i] = interpolate_target_array(targets, P, z, n, i)
            m = interpolate_target_array(targets, M, z, n, i)
            fixed_resid[n, i] = z - m - 1.0 / cfg.gamma_d
            boundary[n, i] = 1.0 if hit else 0.0
            # Direct MV score consistency diagnostic on the solved target grid.
            Q = V[:, n, i] + 2.0 * targets * M[:, n, i] - targets * targets
            J = M[:, n, i] - 0.5 * cfg.gamma_d * (Q - M[:, n, i] ** 2)
            direct_same[n, i] = 1.0 if int(np.argmax(J)) == int(np.argmax(S)) else 0.0
    return {
        "policy": policy,
        "zstar": zstar,
        "fixed_point_residual": fixed_resid,
        "direct_score_same": direct_same,
        "target_boundary_hit": boundary,
        "targets_used": targets,
    }


def find_pcmv_target(cfg: Config, family: Dict[str, np.ndarray]) -> Dict[str, float]:
    targets = family["targets"]
    xg = family["x_grid"]
    V0 = interp_x_vector(xg, family["V"][:, 0, :], cfg.x0)
    M0 = interp_x_vector(xg, family["M"][:, 0, :], cfg.x0)
    S = targets - 1.0 / (2.0 * cfg.gamma_p) - 0.5 * cfg.gamma_p * V0
    z_score, j_score, hit = parabolic_max_uniform(targets, S)

    R = targets - M0 - 1.0 / cfg.gamma_p
    roots = []
    for j in range(len(targets) - 1):
        if R[j] == 0.0:
            roots.append(targets[j])
        elif R[j] * R[j + 1] < 0.0:
            lam = -R[j] / (R[j + 1] - R[j])
            roots.append(targets[j] + lam * (targets[j + 1] - targets[j]))
    z_root = min(roots, key=lambda q: abs(q - z_score)) if roots else z_score
    return {
        "z_score": float(z_score),
        "z_root": float(z_root),
        "score_boundary_hit": bool(hit),
        "grid_argmax": float(targets[j_score]),
        "grid_residual_at_argmax": float(R[j_score]),
    }


def solve_exact_target(cfg: Config, target: float) -> Dict[str, np.ndarray]:
    return solve_target_family(cfg, np.array([target], dtype=np.float64))


def refine_pcmv_target(cfg: Config, family: Dict[str, np.ndarray], z0: float, max_iter: int = 3) -> Tuple[float, Dict[str, np.ndarray], float]:
    """Refine the scalarized PCMV target using z-E[W^z]-1/gamma=0.

    The full target family supplies a stable finite-difference slope.  Each
    iteration then solves the fixed-target control problem at the updated
    target.  This keeps the final PCMV feedback tied to an actually solved
    target rather than only to interpolation across the target grid.
    """
    targets = family["targets"]
    xg = family["x_grid"]
    M0_grid = interp_x_vector(xg, family["M"][:, 0, :], cfg.x0)
    R_grid = targets - M0_grid - 1.0 / cfg.gamma_p
    z = float(z0)
    exact = solve_exact_target(cfg, z)
    resid = float("nan")
    for _ in range(max_iter):
        m = float(interp_x_vector(xg, exact["M"][:, 0, :], cfg.x0)[0])
        resid = z - m - 1.0 / cfg.gamma_p
        if abs(resid) < 1e-7:
            break
        j = int(np.clip(np.searchsorted(targets, z) - 1, 0, len(targets) - 2))
        slope = (R_grid[j + 1] - R_grid[j]) / (targets[j + 1] - targets[j])
        if not np.isfinite(slope) or abs(slope) < 1e-6:
            slope = 1.0
        z_new = float(np.clip(z - resid / slope, targets[0], targets[-1]))
        if abs(z_new - z) < 1e-10:
            break
        z = z_new
        exact = solve_exact_target(cfg, z)
    m = float(interp_x_vector(xg, exact["M"][:, 0, :], cfg.x0)[0])
    resid = z - m - 1.0 / cfg.gamma_p
    return z, exact, resid


def policy_interp_at_x(x_grid: np.ndarray, p: np.ndarray, x: float) -> float:
    if x <= x_grid[0]: return float(p[0])
    if x >= x_grid[-1]: return float(p[-1])
    j = np.searchsorted(x_grid, x) - 1
    lam = (x - x_grid[j]) / (x_grid[j+1] - x_grid[j])
    return float((1-lam)*p[j] + lam*p[j+1])


def distribute_linear(x_grid: np.ndarray, mass: np.ndarray, x: float, w: float) -> Tuple[float, float]:
    # Returns underflow and overflow mass; mass is deposited at boundary if outside.
    if x <= x_grid[0]:
        mass[0] += w
        return (w if x < 0 else 0.0), 0.0
    if x >= x_grid[-1]:
        mass[-1] += w
        return 0.0, (w if x > x_grid[-1] else 0.0)
    j = np.searchsorted(x_grid, x) - 1
    lam = (x - x_grid[j]) / (x_grid[j+1] - x_grid[j])
    mass[j] += w * (1-lam)
    mass[j+1] += w * lam
    return 0.0, 0.0


def forward_distribution(cfg: Config, x_grid: np.ndarray, policy: np.ndarray, gh_x: np.ndarray, gh_w: np.ndarray) -> Dict[str, np.ndarray]:
    N = cfg.n_steps
    nx = len(x_grid)
    p = np.zeros(nx)
    distribute_linear(x_grid, p, cfg.x0, 1.0)
    pmf = np.zeros((N+1, nx), dtype=np.float64)
    pmf[0] = p
    # A feedback control exists only at the N decision times t_0,...,t_{N-1}.
    # There is no control at terminal time T.  Keep glide/binding arrays on
    # the decision grid, while pmf remains on the N+1 state-time grid.
    glide = np.zeros(N)
    upper_bind = np.zeros(N)
    lower_bind = np.zeros(N)
    overflow = np.zeros(N)
    underflow = np.zeros(N)
    mass_error = np.zeros(N+1)
    for n in range(N):
        frac = policy[n]
        if n == 0:
            # The initial state x0 is deterministic.  Evaluate the feedback
            # directly at x0 rather than averaging over the two grid nodes
            # used only to propagate probability mass.
            pi0 = float(np.interp(cfg.x0, x_grid, frac * x_grid))
            a0 = float(np.clip(pi0 / cfg.x0, 0.0, 1.0))
            glide[n] = a0
            upper_bind[n] = float(a0 >= 0.999)
            lower_bind[n] = float(a0 <= 0.001)
        else:
            glide[n] = float(np.dot(p, frac))
            upper_bind[n] = float(np.dot(p, frac >= 0.999))
            lower_bind[n] = float(np.dot(p, frac <= 0.001))
        pnew = np.zeros(nx)
        if n == 0:
            # Propagate the deterministic initial state directly.  Splitting x0
            # over grid nodes before the first decision would alter both the
            # selected investment amount and the first transition.
            x = cfg.x0
            a = glide[0]
            pi = a * x
            drift = (cfg.r*x + cfg.c + cfg.beta*pi)*cfg.dt
            sd = cfg.sigma*pi*math.sqrt(cfg.dt)
            for k in range(len(gh_x)):
                xp = x + drift + sd*gh_x[k]
                uf, of = distribute_linear(x_grid, pnew, max(xp, 0.0), gh_w[k])
                underflow[n] += uf
                overflow[n] += of
        else:
            for i in range(nx):
                if p[i] <= 0.0:
                    continue
                x = x_grid[i]
                a = frac[i]
                pi = a*x
                drift = (cfg.r*x + cfg.c + cfg.beta*pi)*cfg.dt
                sd = cfg.sigma*pi*math.sqrt(cfg.dt)
                for k in range(len(gh_x)):
                    xp = x + drift + sd*gh_x[k]
                    uf, of = distribute_linear(x_grid, pnew, max(xp, 0.0), p[i]*gh_w[k])
                    underflow[n] += uf
                    overflow[n] += of
        total = pnew.sum()
        mass_error[n+1] = abs(total - 1.0)
        if total > 0:
            pnew /= total
        p = pnew
        pmf[n+1] = p
    return {"pmf": pmf, "glide": glide, "upper_bind": upper_bind, "lower_bind": lower_bind,
            "overflow": overflow, "underflow": underflow, "mass_error": mass_error}


def discrete_quantile(values: np.ndarray, probs: np.ndarray, q: float) -> float:
    c = np.cumsum(probs)
    return float(values[min(np.searchsorted(c, q), len(values)-1)])


def lower_cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    order = np.argsort(values)
    v = values[order]
    p = probs[order]
    remaining = alpha
    total = 0.0
    for vi, pi in zip(v, p):
        take = min(remaining, pi)
        total += take*vi
        remaining -= take
        if remaining <= 1e-15:
            break
    return total/alpha


def terminal_stats(x_grid: np.ndarray, pmf: np.ndarray, D: float) -> Dict[str, float]:
    w = x_grid + D
    p = pmf / pmf.sum()
    mean = float(np.dot(p, w))
    var = float(np.dot(p, (w-mean)**2))
    sd = math.sqrt(max(var, 0.0))
    skew = float(np.dot(p, (w-mean)**3)/(sd**3 + 1e-30))
    return {
        "mean": mean, "stdev": sd, "variance": var, "skewness": skew,
        "q05": discrete_quantile(w,p,0.05), "q50": discrete_quantile(w,p,0.50),
        "q95": discrete_quantile(w,p,0.95), "cvar05": lower_cvar(w,p,0.05),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return float(values[min(np.searchsorted(cdf, q, side="left"), len(values) - 1)])


def weighted_diagnostics(domv: Dict[str,np.ndarray], fwd: Dict[str,np.ndarray]) -> Dict[str,float]:
    pmf = fwd["pmf"][:-1]
    abs_resid = np.abs(domv["fixed_point_residual"])
    boundary = domv["target_boundary_hit"] > 0.5
    denom = pmf.sum()
    interior_weight = np.where(boundary, 0.0, pmf)
    interior_denom = interior_weight.sum()
    return {
        "mean_abs_fixed_point_residual_all": float(np.sum(pmf * abs_resid) / denom),
        "mean_abs_fixed_point_residual_interior": float(np.sum(interior_weight * abs_resid) / interior_denom) if interior_denom > 0 else float("nan"),
        "weighted_q95_abs_fixed_point_residual": weighted_quantile(abs_resid, pmf, 0.95),
        "weighted_q99_abs_fixed_point_residual": weighted_quantile(abs_resid, pmf, 0.99),
        "weighted_q999_abs_fixed_point_residual": weighted_quantile(abs_resid, pmf, 0.999),
        "direct_embedding_argmax_agreement": float(np.sum(pmf * domv["direct_score_same"]) / denom),
        "target_boundary_hit_mass": float(np.sum(pmf * domv["target_boundary_hit"]) / denom),
    }


def run_case(cfg: Config, outdir: Path) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    targets = np.arange(cfg.target_min, cfg.target_max + 0.1*cfg.target_step, cfg.target_step)
    family = solve_target_family(cfg, targets)
    pcmv_cal = find_pcmv_target(cfg, family)
    exact = solve_exact_target(cfg, pcmv_cal["z_root"])
    z_refined = pcmv_cal["z_root"]
    xg = family["x_grid"]
    pcmv_policy = exact["policy"][0].astype(np.float64)
    # Fixed-point residual for the exact target at the initial state.
    M0_exact = float(interp_x_vector(xg, exact["M"][:,0,:], cfg.x0)[0])
    V0_exact = float(interp_x_vector(xg, exact["V"][:,0,:], cfg.x0)[0])
    pcmv_cal["z_exact"] = z_refined
    pcmv_cal["M0_exact"] = M0_exact
    pcmv_cal["fixed_point_residual_exact"] = pcmv_cal["z_exact"] - M0_exact - 1.0/cfg.gamma_p
    pcmv_cal["embedding_value_exact"] = pcmv_cal["z_exact"] - 1.0/(2*cfg.gamma_p) - 0.5*cfg.gamma_p*V0_exact

    domv = build_domv_policy(cfg, family, target_stride=1)
    domv_coarse = build_domv_policy(cfg, family, target_stride=2)  # 10-unit target grid sensitivity

    fwd_p = forward_distribution(cfg, xg, pcmv_policy, family["gh_x"], family["gh_w"])
    fwd_d = forward_distribution(cfg, xg, domv["policy"], family["gh_x"], family["gh_w"])
    fwd_d_coarse = forward_distribution(cfg, xg, domv_coarse["policy"], family["gh_x"], family["gh_w"])
    stats_p = terminal_stats(xg, fwd_p["pmf"][-1], cfg.D)
    stats_d = terminal_stats(xg, fwd_d["pmf"][-1], cfg.D)
    stats_dc = terminal_stats(xg, fwd_d_coarse["pmf"][-1], cfg.D)
    diag_d = weighted_diagnostics(domv, fwd_d)

    # target summaries on own DOMV distribution
    times = np.linspace(0,cfg.T,cfg.n_steps+1)
    decision_times = times[:-1]
    checkpoints = [0,10,20,30,35,39]
    target_rows=[]
    for yr in checkpoints:
        n=min(int(round(yr/cfg.T*cfg.n_steps)),cfg.n_steps-1)
        p=fwd_d["pmf"][n]
        target_rows.append({"n_steps":cfg.n_steps,"year":times[n],
                            "mass_weighted_target":float(np.dot(p,domv["zstar"][n])),
                            "mass_weighted_abs_fixed_point_residual":float(np.dot(p,np.abs(domv["fixed_point_residual"][n]))),
                            "target_boundary_mass":float(np.dot(p,domv["target_boundary_hit"][n])),
                            "mean_glide":float(fwd_d["glide"][n])})
    pd.DataFrame(target_rows).to_csv(outdir/f"domv_target_summary_N{cfg.n_steps}.csv",index=False)

    term_rows=[]
    for strat,st,fwd in [("PCMV",stats_p,fwd_p),("DOMV",stats_d,fwd_d),("DOMV_target_step10",stats_dc,fwd_d_coarse)]:
        term_rows.append({"n_steps":cfg.n_steps,"dt":cfg.dt,"n_x":cfg.n_x,"n_controls":cfg.n_controls,"n_gh":cfg.n_gh,"strategy":strat,**st,
                          "mean_glide":float(np.mean(fwd["glide"])),
                          "mean_upper_binding":float(np.mean(fwd["upper_bind"])),
                          "max_mass_error":float(np.max(fwd["mass_error"])),
                          "total_overflow_mass":float(np.sum(fwd["overflow"])),
                          "terminal_boundary_mass":float(fwd["pmf"][-1,-1])})
    pd.DataFrame(term_rows).to_csv(outdir/f"terminal_summary_N{cfg.n_steps}.csv",index=False)
    pd.DataFrame([{"n_steps":cfg.n_steps,**pcmv_cal}]).to_csv(outdir/f"pcmv_calibration_N{cfg.n_steps}.csv",index=False)
    pd.DataFrame([{"n_steps":cfg.n_steps,**diag_d}]).to_csv(outdir/f"domv_diagnostics_N{cfg.n_steps}.csv",index=False)

    # Save compact arrays needed for comparison and paper figures.
    np.savez_compressed(outdir/f"policies_distributions_N{cfg.n_steps}.npz",
        x_grid=xg,times=times,decision_times=decision_times,pcmv_policy=pcmv_policy,domv_policy=domv["policy"],
        domv_zstar=domv["zstar"],pcmv_pmf=fwd_p["pmf"],domv_pmf=fwd_d["pmf"],
        pcmv_glide=fwd_p["glide"],domv_glide=fwd_d["glide"],
        pcmv_upper_bind=fwd_p["upper_bind"],domv_upper_bind=fwd_d["upper_bind"])

    # Basic figures for this discretization.
    plt.figure(figsize=(7.2,4.5))
    plt.plot(decision_times, fwd_p["glide"], label="PCMV")
    plt.plot(decision_times, fwd_d["glide"], label="DOMV")
    plt.xlabel("Years")
    plt.ylabel("Mass-weighted risky fraction")
    plt.ylim(-0.02,1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/f"glidepaths_N{cfg.n_steps}.png",dpi=180)
    plt.close()

    plt.figure(figsize=(7.2,4.5))
    w=xg+cfg.D
    for name,fwd in [("PCMV",fwd_p),("DOMV",fwd_d)]:
        cdf=np.cumsum(fwd["pmf"][-1])
        plt.plot(w,cdf,label=name)
    plt.xlabel("Terminal total wealth")
    plt.ylabel("CDF")
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/f"terminal_cdf_N{cfg.n_steps}.png",dpi=180)
    plt.close()

    return {"cfg":cfg,"family":family,"pcmv_cal":pcmv_cal,"domv":domv,
            "fwd_p":fwd_p,"fwd_d":fwd_d,"stats_p":stats_p,"stats_d":stats_d,
            "diag_d":diag_d,"solve_seconds":float(family["solve_seconds"][0])+float(exact["solve_seconds"][0])}


def compare_cases(results: Dict[int,Dict[str,object]], outdir: Path) -> None:
    rows=[]
    for N,res in results.items():
        for strategy,key in [("PCMV","stats_p"),("DOMV","stats_d")]:
            rows.append({"n_steps":N,"dt":res["cfg"].dt,"strategy":strategy,**res[key],
                         "mean_glide":float(np.mean((res["fwd_p"] if strategy=="PCMV" else res["fwd_d"])["glide"]))})
    df=pd.DataFrame(rows)
    df.to_csv(outdir/"time_grid_convergence.csv",index=False)
    if 80 in results and 480 in results:
        base=df[df.n_steps==480].set_index("strategy")
        coarse=df[df.n_steps==80].set_index("strategy")
        diff=[]
        for s in ["PCMV","DOMV"]:
            row={"strategy":s}
            for col in ["mean","stdev","q05","q50","q95","cvar05","mean_glide"]:
                row[f"abs_diff_{col}"]=float(abs(coarse.loc[s,col]-base.loc[s,col]))
                row[f"rel_diff_{col}"]=float(abs(coarse.loc[s,col]-base.loc[s,col])/(abs(base.loc[s,col])+1e-12))
            diff.append(row)
        pd.DataFrame(diff).to_csv(outdir/"time_grid_80_vs_480_differences.csv",index=False)


def run_single_cli(n_steps: int, outdir: Path, refined: bool = False) -> None:
    if refined:
        cfg = Config(n_steps=n_steps, n_x=601, n_controls=41, n_gh=9)
    else:
        cfg = Config(n_steps=n_steps)
    print(f"Running N={n_steps}, Nx={cfg.n_x}, Na={cfg.n_controls}, GH={cfg.n_gh}", flush=True)
    res = run_case(cfg, outdir)
    metadata = {
        "config": asdict(cfg),
        "pcmv_calibration": res["pcmv_cal"],
        "domv_diagnostics": res["diag_d"],
        "solve_seconds": res["solve_seconds"],
    }
    (outdir / f"run_metadata_N{n_steps}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float)
    )
    print("Finished", n_steps, "PCMV", res["stats_p"], "DOMV", res["stats_d"], flush=True)


def consolidate_existing(outdir: Path) -> None:
    rows = []
    for n_steps in (80, 480):
        f = outdir / f"terminal_summary_N{n_steps}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["strategy"].isin(["PCMV", "DOMV"])].copy()
        rows.append(df)
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(outdir / "time_grid_convergence.csv", index=False)
    if set(all_df["n_steps"]) >= {80, 480}:
        base = all_df[all_df.n_steps == 480].set_index("strategy")
        coarse = all_df[all_df.n_steps == 80].set_index("strategy")
        diff = []
        for strategy in ("PCMV", "DOMV"):
            row = {"strategy": strategy}
            for col in ("mean", "stdev", "q05", "q50", "q95", "cvar05", "mean_glide"):
                signed = float(base.loc[strategy, col] - coarse.loc[strategy, col])
                row[f"monthly_minus_semiannual_{col}"] = signed
                row[f"relative_abs_diff_{col}"] = abs(signed) / (abs(float(base.loc[strategy, col])) + 1e-12)
            diff.append(row)
        pd.DataFrame(diff).to_csv(outdir / "time_grid_80_vs_480_differences.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible PCMV/DOMV monthly Markov-GH simulation")
    parser.add_argument("--n-steps", type=int, choices=[80, 480], help="Run one time grid")
    parser.add_argument("--all", action="store_true", help="Run 80 and 480 in separate subprocesses")
    parser.add_argument("--refined", action="store_true", help="Use Nx=601, 41 controls, GH=9")
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/data/pcmv_domv_refined_results"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        set_num_threads(min(12, get_num_threads()))
    except Exception:
        pass
    if args.all:
        for n_steps in (80, 480):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--n-steps", str(n_steps), "--outdir", str(args.outdir)]
            if args.refined:
                cmd.append("--refined")
            subprocess.run(cmd, check=True)
        consolidate_existing(args.outdir)
        return
    if args.n_steps is None:
        parser.error("Specify --n-steps 80, --n-steps 480, or --all")
    run_single_cli(args.n_steps, args.outdir, refined=args.refined)
    gc.collect()
    consolidate_existing(args.outdir)


if __name__ == "__main__":
    main()
