from __future__ import annotations

import importlib.util
import sys
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'figs'
RES = ROOT / 'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load PCMV/DOMV solver
# ------------------------------------------------------------
pc_path = ROOT / 'scripts' / '01_solvers' / 'pcmv_domv_solver_20260713.py'
spec = importlib.util.spec_from_file_location('pcmod', pc_path)
pcmod = importlib.util.module_from_spec(spec)
sys.modules['pcmod'] = pcmod
spec.loader.exec_module(pcmod)

# Monthly D_T=0 baseline.  The time grid is monthly; the state/control grids
# are moderate because this run is used to make Sections 6.1, 6.3, 6.4 and
# the rolling-evaluation exhibit internally consistent.
pcfg = replace(
    pcmod.Config(n_steps=480),
    D=0.0,
    n_x=151,
    n_controls=15,
    n_gh=5,
    target_step=5.0,
    x_max=300.0,
)

pc_out = ROOT / 'pc_d0'
pc_out.mkdir(exist_ok=True)
pcres = pcmod.run_case(pcfg, pc_out)

# ------------------------------------------------------------
# Fast sequential cTCMV/dTCMV solver for a fixed contribution flow
# ------------------------------------------------------------
T = 40.0
N = 480
dt = T / N
x0 = 1.0 / 12.0
x_max = 300.0
n_x = 151
n_controls = 15
n_gh = 5
x_power = 1.6
r = 0.015
mu = 0.055
beta = mu - r
sigma = 0.18
c = 1.0
D = 0.0
gamma_c = 0.059638671875
gamma0 = 1.193359375

u = np.linspace(0.0, 1.0, n_x)
xg = x_max * u**x_power
h, w = hermgauss(n_gh)
gh_x = np.sqrt(2.0) * h.astype(np.float64)
gh_w = (w / np.sqrt(np.pi)).astype(np.float64)

H = np.zeros(N + 1)
H[-1] = D
for n in range(N - 1, -1, -1):
    H[n] = (c * dt + H[n + 1]) / (1.0 + r * dt)

@njit(cache=True)
def locate(xg, x):
    nx = xg.size
    if x <= xg[0]:
        return 0, 0.0
    if x >= xg[nx - 1]:
        return nx - 2, (x - xg[nx - 2]) / (xg[nx - 1] - xg[nx - 2])
    lo = 0
    hi = nx - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xg[mid] <= x:
            lo = mid
        else:
            hi = mid
    lam = (x - xg[lo]) / (xg[lo + 1] - xg[lo])
    return lo, lam

@njit(cache=True)
def interp(v, j, lam):
    return v[j] * (1.0 - lam) + v[j + 1] * lam

@njit(cache=True)
def expected_mq(Mnext, Qnext, xg, x, a, dt, r, beta, sigma, c, gh_x, gh_w):
    pi = a * x
    drift = (r * x + c + beta * pi) * dt
    sd = sigma * pi * math.sqrt(dt)
    m = 0.0
    q = 0.0
    for k in range(gh_x.size):
        xp = x + drift + sd * gh_x[k]
        if xp < 0.0:
            xp = 0.0
        j, lam = locate(xg, xp)
        wk = gh_w[k]
        m += wk * interp(Mnext, j, lam)
        q += wk * interp(Qnext, j, lam)
    return m, q

@njit(cache=True)
def expected_mq_clamp(Mnext, Qnext, xg, x, a, dt, r, beta, sigma, c, gh_x, gh_w):
    pi = a * x
    drift = (r * x + c + beta * pi) * dt
    sd = sigma * pi * math.sqrt(dt)
    m = 0.0
    q = 0.0
    for k in range(gh_x.size):
        xp = x + drift + sd * gh_x[k]
        if xp <= xg[0]:
            j = 0
            lam = 0.0
        elif xp >= xg[-1]:
            j = xg.size - 2
            lam = 1.0
        else:
            j, lam = locate(xg, xp)
        wk = gh_w[k]
        m += wk * interp(Mnext, j, lam)
        q += wk * interp(Qnext, j, lam)
    return m, q

@njit(cache=True)
def solve_tc(strategy_code, N, xg, H, dt, r, beta, sigma, c, D,
             gamma_c, gamma0, n_controls, gh_x, gh_w):
    nx = xg.size
    M = np.empty((N + 1, nx), dtype=np.float64)
    Q = np.empty((N + 1, nx), dtype=np.float64)
    P = np.empty((N, nx), dtype=np.float64)
    da = 1.0 / (n_controls - 1)
    for i in range(nx):
        z = xg[i] + D
        M[N, i] = z
        Q[N, i] = z * z
    for n in range(N - 1, -1, -1):
        Mnext = M[n + 1]
        Qnext = Q[n + 1]
        for i in range(nx):
            x = xg[i]
            gamma = gamma_c if strategy_code == 0 else gamma0 / max(x + H[n], 1e-14)
            best_j = 0
            best_val = -1e300
            f = np.empty(n_controls, dtype=np.float64)
            for j in range(n_controls):
                a = j * da
                m, q = expected_mq(Mnext, Qnext, xg, x, a, dt, r, beta, sigma, c, gh_x, gh_w)
                var = q - m * m
                if var < 0.0 and var > -1e-9:
                    var = 0.0
                val = m - 0.5 * gamma * var
                f[j] = val
                if val > best_val:
                    best_val = val
                    best_j = j
            best_a = best_j * da
            if 0 < best_j < n_controls - 1 and x > 1e-15:
                denom = f[best_j - 1] - 2.0 * f[best_j] + f[best_j + 1]
                if denom < -1e-14:
                    off = 0.5 * (f[best_j - 1] - f[best_j + 1]) / denom
                    if off < -1.0:
                        off = -1.0
                    if off > 1.0:
                        off = 1.0
                    aref = (best_j + off) * da
                    mref, qref = expected_mq(Mnext, Qnext, xg, x, aref, dt, r, beta, sigma, c, gh_x, gh_w)
                    varref = qref - mref * mref
                    if varref < 0.0 and varref > -1e-9:
                        varref = 0.0
                    valref = mref - 0.5 * gamma * varref
                    if valref > best_val:
                        best_a = aref
            m, q = expected_mq(Mnext, Qnext, xg, x, best_a, dt, r, beta, sigma, c, gh_x, gh_w)
            P[n, i] = best_a
            M[n, i] = m
            Q[n, i] = q
    return M, Q, P

@njit(cache=True)
def deposit(xg, p, x, weight):
    if x <= xg[0]:
        p[0] += weight
        return
    if x >= xg[-1]:
        p[-1] += weight
        return
    j, lam = locate(xg, x)
    p[j] += weight * (1.0 - lam)
    p[j + 1] += weight * lam

@njit(cache=True)
def forward_policy(N, xg, xstart, nstart, P, dt, r, beta, sigma, c, gh_x, gh_w):
    nx = xg.size
    p = np.zeros(nx)
    deposit(xg, p, xstart, 1.0)
    steps = N - nstart
    pmf = np.zeros((steps + 1, nx))
    pmf[0] = p
    # Controls are defined only at the decision times; T has no control.
    glide = np.zeros(steps)
    upper = np.zeros(steps)
    for s in range(steps):
        n = nstart + s
        if s == 0:
            j0, lam0 = locate(xg, xstart)
            pi_grid = P[n] * xg
            pi0 = interp(pi_grid, j0, lam0)
            a0 = min(max(pi0 / max(xstart, 1e-14), 0.0), 1.0)
            glide[s] = a0
            upper[s] = 1.0 if a0 >= 0.999 else 0.0
        else:
            for i in range(nx):
                glide[s] += p[i] * P[n, i]
                if P[n, i] >= 0.999:
                    upper[s] += p[i]
        pn = np.zeros(nx)
        if s == 0:
            x = xstart
            a = glide[0]
            pi = a * x
            drift = (r * x + c + beta * pi) * dt
            sd = sigma * pi * math.sqrt(dt)
            for k in range(gh_x.size):
                xp = max(x + drift + sd * gh_x[k], 0.0)
                deposit(xg, pn, xp, gh_w[k])
        else:
            for i in range(nx):
                if p[i] <= 0.0:
                    continue
                x = xg[i]
                a = P[n, i]
                pi = a * x
                drift = (r * x + c + beta * pi) * dt
                sd = sigma * pi * math.sqrt(dt)
                for k in range(gh_x.size):
                    xp = x + drift + sd * gh_x[k]
                    if xp < 0.0:
                        xp = 0.0
                    deposit(xg, pn, xp, p[i] * gh_w[k])
        total = pn.sum()
        if total > 0.0:
            pn /= total
        p = pn
        pmf[s + 1] = p
    return pmf, glide, upper

M_c, Q_c, P_c = solve_tc(0, N, xg, H, dt, r, beta, sigma, c, D,
                         gamma_c, gamma0, n_controls, gh_x, gh_w)
M_d, Q_d, P_d = solve_tc(1, N, xg, H, dt, r, beta, sigma, c, D,
                         gamma_c, gamma0, n_controls, gh_x, gh_w)

pmf_c, glide_c, upper_c = forward_policy(N, xg, x0, 0, P_c, dt, r, beta, sigma, c, gh_x, gh_w)
pmf_d, glide_d, upper_d = forward_policy(N, xg, x0, 0, P_d, dt, r, beta, sigma, c, gh_x, gh_w)
theta_cp = 0.4694735
P_cp = np.full((N, n_x), theta_cp)
pmf_cp, glide_cp, upper_cp = forward_policy(N, xg, x0, 0, P_cp, dt, r, beta, sigma, c, gh_x, gh_w)

# ------------------------------------------------------------
# Common helper functions
# ------------------------------------------------------------
def quantile(values, probs, q):
    cdf = np.cumsum(probs / probs.sum())
    return float(values[min(np.searchsorted(cdf, q), len(values) - 1)])

def lower_cvar(values, probs, alpha=0.05):
    probs = probs / probs.sum()
    rem = alpha
    total = 0.0
    for v, p in zip(values, probs):
        take = min(rem, p)
        total += take * v
        rem -= take
        if rem <= 1e-15:
            break
    return total / alpha

def stats_from_pmf(xgrid, pmf, D=0.0):
    values = xgrid + D
    p = pmf / pmf.sum()
    mean = float(np.dot(p, values))
    var = float(np.dot(p, (values - mean) ** 2))
    sd = math.sqrt(max(var, 0.0))
    skew = float(np.dot(p, (values - mean) ** 3) / (sd ** 3 + 1e-30))
    return {
        'mean': mean,
        'stdev': sd,
        'skewness': skew,
        'q05': quantile(values, p, 0.05),
        'q50': quantile(values, p, 0.50),
        'q95': quantile(values, p, 0.95),
        'cvar05': lower_cvar(values, p, 0.05),
    }

def interp_at(xgrid, values, x):
    return float(np.interp(x, xgrid, values))

def median_state(xgrid, pmf):
    return quantile(xgrid, pmf, 0.50)

# PCMV/DOMV data from the solver
xg_pc = pcres['family']['x_grid']
times = np.linspace(0.0, T, N + 1)
P_p = pcres['family']['policy'][0]  # placeholder overwritten below
# run_case returns the exact PCMV policy through forward result only indirectly.
# It is saved in the compact npz produced by run_case.
pc_npz = np.load(pc_out / 'policies_distributions_N480.npz')
P_p = pc_npz['pcmv_policy']
P_D = pc_npz['domv_policy']
pmf_p = pc_npz['pcmv_pmf']
pmf_D = pc_npz['domv_pmf']
glide_p = pc_npz['pcmv_glide']
glide_D = pc_npz['domv_glide']
upper_p = pc_npz['pcmv_upper_bind']
upper_D = pc_npz['domv_upper_bind']

def decision_summary_from_policy(policy, pmf, grid, x_initial):
    n_steps = policy.shape[0]
    glide = np.empty(n_steps, dtype=float)
    upper = np.empty(n_steps, dtype=float)
    pi0 = float(np.interp(x_initial, grid, policy[0] * grid))
    a0 = float(np.clip(pi0 / max(x_initial, 1e-14), 0.0, 1.0))
    glide[0] = a0
    upper[0] = float(a0 >= 0.999)
    for n in range(1, n_steps):
        glide[n] = float(np.dot(pmf[n], policy[n]))
        upper[n] = float(np.dot(pmf[n], policy[n] >= 0.999))
    return glide, upper

glide_p, upper_p = decision_summary_from_policy(P_p, pmf_p, xg_pc, x0)
glide_D, upper_D = decision_summary_from_policy(P_D, pmf_D, xg_pc, x0)

stats_p = stats_from_pmf(xg_pc, pmf_p[-1], 0.0)
stats_D = stats_from_pmf(xg_pc, pmf_D[-1], 0.0)
stats_c = stats_from_pmf(xg, pmf_c[-1], 0.0)
stats_d = stats_from_pmf(xg, pmf_d[-1], 0.0)
stats_cp = stats_from_pmf(xg, pmf_cp[-1], 0.0)

baseline_rows = []
for name, st, glide, upper in [
    ('PCMV', stats_p, glide_p, upper_p),
    ('DOMV', stats_D, glide_D, upper_D),
    ('cTCMV', stats_c, glide_c, upper_c),
    ('dTCMV', stats_d, glide_d, upper_d),
    ('CP 60%', stats_cp, glide_cp, upper_cp),
]:
    baseline_rows.append({
        'strategy': name,
        **st,
        'avg_glide': float(np.mean(glide)),
        'upper_bind': float(np.mean(upper)),
    })

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(RES / 'monthly_baseline_D0_summary.csv', index=False)
np.savez_compressed(RES / 'monthly_D0_policy_arrays.npz',
    times=times, decision_times=times[:-1], xg_pc=xg_pc, xg_tc=xg,
    pcmv_policy=P_p, domv_policy=P_D, ctcmv_policy=P_c, dtcmv_policy=P_d,
    pcmv_pmf=pmf_p, domv_pmf=pmf_D, ctcmv_pmf=pmf_c, dtcmv_pmf=pmf_d, cp_pmf=pmf_cp,
    pcmv_glide=glide_p, domv_glide=glide_D, ctcmv_glide=glide_c, dtcmv_glide=glide_d, cp_glide=glide_cp,
    pcmv_upper=upper_p, domv_upper=upper_D, ctcmv_upper=upper_c, dtcmv_upper=upper_d)

# ------------------------------------------------------------
# Baseline figures under D_T=0
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.6))
for name, g in [('PCMV', glide_p), ('DOMV', glide_D), ('cTCMV', glide_c), ('dTCMV', glide_d), ('CP 60%', glide_cp)]:
    ax.plot(times[:-1], g, label=name, linewidth=1.8)
ax.set_xlabel('Years since entry')
ax.set_ylabel('Mass-weighted risky fraction')
ax.set_ylim(0, 1.04)
ax.grid(alpha=0.25)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(FIG / 'fig_all_strategies_glidepaths_D0_N480.png', dpi=180)
plt.close(fig)

# CDF and smoothed density
from scipy.ndimage import gaussian_filter1d

def deposit_uniform(vals, probs, grid):
    mass = np.zeros_like(grid)
    dx = grid[1] - grid[0]
    for v, p in zip(vals, probs):
        if p <= 0:
            continue
        u = (v - grid[0]) / dx
        if u <= 0:
            mass[0] += p
        elif u >= len(grid) - 1:
            mass[-1] += p
        else:
            j = int(math.floor(u))
            lam = u - j
            mass[j] += p * (1 - lam)
            mass[j + 1] += p * lam
    return mass

series = [
    ('PCMV', xg_pc, pmf_p[-1]),
    ('DOMV', xg_pc, pmf_D[-1]),
    ('cTCMV', xg, pmf_c[-1]),
    ('dTCMV', xg, pmf_d[-1]),
    ('CP 60%', xg, pmf_cp[-1]),
]
grid = np.linspace(0, 280, 1400)
fig, ax = plt.subplots(figsize=(9.2, 5.6))
for name, vals, probs in series:
    mass = deposit_uniform(vals, probs, grid)
    density = gaussian_filter1d(mass, sigma=5, mode='nearest') / (grid[1] - grid[0])
    ax.plot(grid, density, label=name, linewidth=1.8)
ax.set_xlabel('Terminal DC wealth')
ax.set_ylabel('Density')
ax.set_xlim(0, 230)
ax.grid(alpha=0.25)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(FIG / 'fig_all_strategies_terminal_density_D0_N480.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.2, 5.6))
for name, vals, probs in series:
    idx = np.argsort(vals)
    ax.plot(vals[idx], np.cumsum(probs[idx] / probs.sum()), label=name, linewidth=1.8)
ax.set_xlabel('Terminal DC wealth')
ax.set_ylabel('CDF')
ax.set_xlim(0, 260)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(FIG / 'fig_all_strategies_terminal_cdf_D0_N480.png', dpi=180)
plt.close(fig)

# ------------------------------------------------------------
# Generic policy-moment evaluation for rolling validation
# ------------------------------------------------------------
@njit(cache=True)
def policy_moments(N, xg, P, dt, r, beta, sigma, c, D, gh_x, gh_w):
    nx = xg.size
    M = np.empty((N + 1, nx))
    Q = np.empty((N + 1, nx))
    for i in range(nx):
        z = xg[i] + D
        M[N, i] = z
        Q[N, i] = z * z
    for n in range(N - 1, -1, -1):
        for i in range(nx):
            m, q = expected_mq_clamp(M[n + 1], Q[n + 1], xg, xg[i], P[n, i], dt, r, beta, sigma, c, gh_x, gh_w)
            M[n, i] = m
            Q[n, i] = q
    return M, Q

# Need a version on the PC grid.
@njit(cache=True)
def policy_moments_pc(N, xg, P, dt, r, beta, sigma, c, D, gh_x, gh_w):
    nx = xg.size
    M = np.empty((N + 1, nx))
    Q = np.empty((N + 1, nx))
    for i in range(nx):
        z = xg[i] + D
        M[N, i] = z
        Q[N, i] = z * z
    for n in range(N - 1, -1, -1):
        for i in range(nx):
            m, q = expected_mq_clamp(M[n + 1], Q[n + 1], xg, xg[i], P[n, i], dt, r, beta, sigma, c, gh_x, gh_w)
            M[n, i] = m
            Q[n, i] = q
    return M, Q

M_p_eval, Q_p_eval = policy_moments_pc(N, xg_pc, P_p, dt, r, beta, sigma, c, 0.0, gh_x, gh_w)
M_D_eval, Q_D_eval = policy_moments_pc(N, xg_pc, P_D, dt, r, beta, sigma, c, 0.0, gh_x, gh_w)
# c/d M,Q are already policy-consistent by construction, but recompute with the same evaluator for a symmetric diagnostic.
M_c_eval, Q_c_eval = policy_moments(N, xg, P_c, dt, r, beta, sigma, c, 0.0, gh_x, gh_w)
M_d_eval, Q_d_eval = policy_moments(N, xg, P_d, dt, r, beta, sigma, c, 0.0, gh_x, gh_w)

check_years = [0, 10, 20, 30, 35, 39]
rolling_rows = []

strategies = {
    'PCMV': (xg_pc, P_p, pmf_p, M_p_eval, Q_p_eval),
    'DOMV': (xg_pc, P_D, pmf_D, M_D_eval, Q_D_eval),
    'cTCMV': (xg, P_c, pmf_c, M_c_eval, Q_c_eval),
    'dTCMV': (xg, P_d, pmf_d, M_d_eval, Q_d_eval),
}

for name, (grid_x, policy, pmf, M_eval, Q_eval) in strategies.items():
    for year in check_years:
        n = min(int(round(year / dt)), N - 1)
        xm = median_state(grid_x, pmf[n])
        risk = interp_at(grid_x, policy[n], xm)
        # Conditional forward propagation from the median state.
        pmf_cond, _, _ = forward_policy(N, grid_x, xm, n, policy, dt, r, beta, sigma, c, gh_x, gh_w)
        st = stats_from_pmf(grid_x, pmf_cond[-1], 0.0)
        mb = interp_at(grid_x, M_eval[n], xm)
        qb = interp_at(grid_x, Q_eval[n], xm)
        sdb = math.sqrt(max(qb - mb * mb, 0.0))
        rolling_rows.append({
            'strategy': name,
            'year': year,
            'median_x': xm,
            'conditional_mean_forward': st['mean'],
            'conditional_std_forward': st['stdev'],
            'risky_fraction': risk,
            'conditional_mean_backward': mb,
            'conditional_std_backward': sdb,
            'mean_abs_residual': abs(st['mean'] - mb),
            'std_abs_residual': abs(st['stdev'] - sdb),
        })

rolling_df = pd.DataFrame(rolling_rows)
rolling_df.to_csv(RES / 'rolling_conditional_D0_N480.csv', index=False)
validation_df = rolling_df.groupby('strategy', as_index=False).agg(
    max_mean_residual=('mean_abs_residual', 'max'),
    max_std_residual=('std_abs_residual', 'max'),
    mean_mean_residual=('mean_abs_residual', 'mean'),
    mean_std_residual=('std_abs_residual', 'mean'),
)
validation_df.to_csv(RES / 'rolling_validation_D0_N480.csv', index=False)

# Rolling graph: mean, standard deviation, and risky fraction.
fig, axes = plt.subplots(3, 1, figsize=(9.2, 11.2), sharex=True)
for name in strategies:
    d = rolling_df[rolling_df.strategy == name]
    axes[0].plot(d.year, d.conditional_mean_forward, marker='o', label=name)
    axes[1].plot(d.year, d.conditional_std_forward, marker='o', label=name)
    axes[2].plot(d.year, d.risky_fraction, marker='o', label=name)
axes[0].set_ylabel('Conditional terminal mean')
axes[1].set_ylabel('Conditional terminal stdev')
axes[2].set_ylabel('Current risky fraction')
axes[2].set_xlabel('Years since entry')
for ax in axes:
    ax.grid(alpha=0.25)
axes[0].legend(ncol=4, loc='best')
fig.tight_layout()
fig.savefig(FIG / 'fig_rolling_conditional_all_strategies_D0_N480.png', dpi=180)
plt.close(fig)

# Residual graph
fig, ax = plt.subplots(figsize=(8.6, 5.2))
for name in strategies:
    d = rolling_df[rolling_df.strategy == name]
    ax.plot(d.year, d.mean_abs_residual, marker='o', label=f'{name}: mean')
ax.set_yscale('log')
ax.set_xlabel('Years since entry')
ax.set_ylabel('Absolute backward-forward mean residual')
ax.grid(alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(FIG / 'fig_rolling_validation_residual_D0_N480.png', dpi=180)
plt.close(fig)

print('\nBASELINE D0 MONTHLY')
print(baseline_df.to_string(index=False))
print('\nROLLING')
print(rolling_df.to_string(index=False))
print('\nVALIDATION')
print(validation_df.to_string(index=False))
