from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'figs'
RES = ROOT / 'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

arr = np.load(RES / 'monthly_D0_policy_arrays.npz', allow_pickle=True)
times = arr['times']
xg = arr['xg_tc']
P_c = arr['ctcmv_policy']
P_d = arr['dtcmv_policy']
pmf_d = arr['dtcmv_pmf']

T = float(times[-1])
N = len(times) - 1
dt = float(times[1] - times[0])
r = 0.015
mu = 0.055
beta = mu - r
sigma = 0.18
c = 1.0
gamma0 = 1.193359375

h, w = hermgauss(5)
gh_x = np.sqrt(2.0) * h.astype(float)
gh_w = (w / np.sqrt(np.pi)).astype(float)

H = np.zeros(N + 1)
for n in range(N - 1, -1, -1):
    H[n] = (c * dt + H[n + 1]) / (1.0 + r * dt)


def deposit(p: np.ndarray, x: float, weight: float) -> None:
    if x <= xg[0]:
        p[0] += weight
        return
    if x >= xg[-1]:
        p[-1] += weight
        return
    j = int(np.searchsorted(xg, x) - 1)
    lam = (x - xg[j]) / (xg[j + 1] - xg[j])
    p[j] += weight * (1.0 - lam)
    p[j + 1] += weight * lam


def discrete_quantile(p: np.ndarray, q: float) -> float:
    pp = p / p.sum()
    idx = min(int(np.searchsorted(np.cumsum(pp), q)), len(xg) - 1)
    return float(xg[idx])


def lower_cvar(p: np.ndarray, alpha: float = 0.05) -> float:
    pp = p / p.sum()
    rem = alpha
    total = 0.0
    for value, prob in zip(xg, pp):
        take = min(rem, prob)
        total += take * value
        rem -= take
        if rem <= 1e-15:
            break
    return total / alpha


def upper_cvar(p: np.ndarray, alpha: float = 0.05) -> float:
    pp = p / p.sum()
    rem = alpha
    total = 0.0
    for value, prob in zip(xg[::-1], pp[::-1]):
        take = min(rem, prob)
        total += take * value
        rem -= take
        if rem <= 1e-15:
            break
    return total / alpha


def stats(p: np.ndarray) -> Dict[str, float]:
    pp = p / p.sum()
    mean = float(np.dot(pp, xg))
    var = float(np.dot(pp, (xg - mean) ** 2))
    sd = math.sqrt(max(var, 0.0))
    skew = float(np.dot(pp, (xg - mean) ** 3) / (sd ** 3 + 1e-30))
    q05 = discrete_quantile(pp, 0.05)
    q50 = discrete_quantile(pp, 0.50)
    q95 = discrete_quantile(pp, 0.95)
    downside = q50 - q05
    upside = q95 - q50
    return {
        'mean': mean,
        'stdev': sd,
        'skewness': skew,
        'q05': q05,
        'q50': q50,
        'q95': q95,
        'cvar05': lower_cvar(pp, 0.05),
        'ucvar95': upper_cvar(pp, 0.05),
        'downside_spread': downside,
        'upside_spread': upside,
        'tail_asymmetry': upside / (downside + 1e-12),
    }


def forward(n0: int, x0: float, policy: np.ndarray) -> np.ndarray:
    p = np.zeros_like(xg)
    deposit(p, x0, 1.0)
    for n in range(n0, N):
        pn = np.zeros_like(p)
        for i, x in enumerate(xg):
            if p[i] <= 0.0:
                continue
            a = float(np.interp(x, xg, policy[n]))
            pi = a * x
            drift = (r * x + c + beta * pi) * dt
            sd = sigma * pi * math.sqrt(dt)
            for z, weight in zip(gh_x, gh_w):
                deposit(pn, max(0.0, x + drift + sd * z), p[i] * weight)
        total = pn.sum()
        if total > 0.0:
            pn /= total
        p = pn
    return p

review_years = [10, 20, 30, 35, 39]
state_quantiles = [('q10', 0.10), ('q50', 0.50), ('q90', 0.90)]
rows = []
conditional_pmfs: Dict[Tuple[int, str, str], np.ndarray] = {}

for year in review_years:
    n = min(int(round(year / dt)), N - 1)
    for state_name, q in state_quantiles:
        x = discrete_quantile(pmf_d[n], q)
        for policy_name, policy in [('dTCMV', P_d), ('cTCMV', P_c)]:
            p = forward(n, x, policy)
            conditional_pmfs[(year, state_name, policy_name)] = p
            st = stats(p)
            risky_fraction = float(np.interp(x, xg, policy[n]))
            risky_amount = risky_fraction * x
            rows.append({
                'year': year,
                'state': state_name,
                'state_quantile': q,
                'starting_x': x,
                'future_contribution_pv_H': H[n],
                'total_pension_wealth_x_plus_H': x + H[n],
                'dtcmv_risk_aversion_gamma': gamma0 / max(x + H[n], 1e-14),
                'policy': policy_name,
                'risky_fraction': risky_fraction,
                'risky_amount': risky_amount,
                **st,
            })

full = pd.DataFrame(rows)
full.to_csv(RES / 'rolling_quantile_detail_D0_N480.csv', index=False)

# Same-state incremental overlay relative to cTCMV.
wide = full.pivot(index=['year', 'state', 'state_quantile', 'starting_x',
                         'future_contribution_pv_H', 'total_pension_wealth_x_plus_H',
                         'dtcmv_risk_aversion_gamma'], columns='policy')
wide.columns = [f'{metric}_{policy}' for metric, policy in wide.columns]
wide = wide.reset_index()
for metric in ['risky_fraction', 'risky_amount', 'mean', 'stdev', 'skewness',
               'q05', 'q50', 'q95', 'cvar05', 'ucvar95', 'tail_asymmetry']:
    wide[f'delta_{metric}_d_minus_c'] = wide[f'{metric}_dTCMV'] - wide[f'{metric}_cTCMV']
wide.to_csv(RES / 'rolling_overlay_vs_ctcmv_D0_N480.csv', index=False)

# Practical diagnostic summary: low-state downside gain and high-state upside gain.
summary_rows = []
for year in review_years:
    low = wide[(wide.year == year) & (wide.state == 'q10')].iloc[0]
    med = wide[(wide.year == year) & (wide.state == 'q50')].iloc[0]
    high = wide[(wide.year == year) & (wide.state == 'q90')].iloc[0]
    summary_rows.append({
        'year': year,
        'low_state_x': low.starting_x,
        'low_state_delta_risky_amount': low.delta_risky_amount_d_minus_c,
        'low_state_cvar05_gain': low.delta_cvar05_d_minus_c,
        'low_state_q05_gain': low.delta_q05_d_minus_c,
        'median_state_delta_risky_amount': med.delta_risky_amount_d_minus_c,
        'high_state_x': high.starting_x,
        'high_state_delta_risky_amount': high.delta_risky_amount_d_minus_c,
        'high_state_q95_gain': high.delta_q95_d_minus_c,
        'high_state_skewness_gain': high.delta_skewness_d_minus_c,
        'high_state_cvar05_change': high.delta_cvar05_d_minus_c,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(RES / 'rolling_personalization_diagnostic_D0_N480.csv', index=False)

# Figure 1: dTCMV own-state rolling metrics.
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True)
donly = full[full.policy == 'dTCMV']
for state_name, _ in state_quantiles:
    d = donly[donly.state == state_name]
    axes[0, 0].plot(d.year, d.risky_fraction, marker='o', label=state_name)
    axes[0, 1].plot(d.year, d.risky_amount, marker='o', label=state_name)
    axes[1, 0].plot(d.year, d.cvar05, marker='o', label=state_name)
    axes[1, 1].plot(d.year, d.q95, marker='o', label=state_name)
axes[0, 0].set_ylabel('Current risky fraction')
axes[0, 1].set_ylabel('Current risky amount')
axes[1, 0].set_ylabel('Conditional CVaR 5%')
axes[1, 1].set_ylabel('Conditional 95% quantile')
for ax in axes.ravel():
    ax.grid(alpha=0.25)
    ax.set_xlabel('Years since entry')
axes[0, 0].legend(title='dTCMV own-path state')
fig.tight_layout()
fig.savefig(FIG / 'fig_dtcmv_rolling_state_quantiles_D0_N480.png', dpi=180)
plt.close(fig)

# Figure 2: incremental dTCMV overlay versus cTCMV at the same state.
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True)
for state_name, _ in state_quantiles:
    d = wide[wide.state == state_name]
    axes[0, 0].plot(d.year, d.delta_risky_amount_d_minus_c, marker='o', label=state_name)
    axes[0, 1].plot(d.year, d.delta_cvar05_d_minus_c, marker='o', label=state_name)
    axes[1, 0].plot(d.year, d.delta_q95_d_minus_c, marker='o', label=state_name)
    axes[1, 1].plot(d.year, d.delta_skewness_d_minus_c, marker='o', label=state_name)
for ax in axes.ravel():
    ax.axhline(0.0, linewidth=0.8, color='black')
    ax.grid(alpha=0.25)
    ax.set_xlabel('Years since entry')
axes[0, 0].set_ylabel(r'$\Delta$ risky amount')
axes[0, 1].set_ylabel(r'$\Delta$ CVaR 5%')
axes[1, 0].set_ylabel(r'$\Delta$ 95% quantile')
axes[1, 1].set_ylabel(r'$\Delta$ conditional skewness')
axes[0, 0].legend(title='Common starting state')
fig.tight_layout()
fig.savefig(FIG / 'fig_dtcmv_overlay_vs_ctcmv_D0_N480.png', dpi=180)
plt.close(fig)

# Figure 3: conditional CDFs at year 30 low/high dTCMV states.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
for ax, state_name, title in [(axes[0], 'q10', 'Year 30: low-balance state'),
                              (axes[1], 'q90', 'Year 30: high-balance state')]:
    for policy_name in ['cTCMV', 'dTCMV']:
        p = conditional_pmfs[(30, state_name, policy_name)]
        ax.plot(xg, np.cumsum(p / p.sum()), label=policy_name)
    ax.set_title(title)
    ax.set_xlabel('Conditional terminal DC wealth')
    ax.grid(alpha=0.25)
axes[0].set_ylabel('CDF')
axes[0].legend()
fig.tight_layout()
fig.savefig(FIG / 'fig_dtcmv_conditional_cdf_low_high_D0_N480.png', dpi=180)
plt.close(fig)

print('\nDetailed rolling quantile analysis')
print(full.to_string(index=False))
print('\nPersonalization diagnostic')
print(summary.to_string(index=False))
