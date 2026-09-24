# van Staden–Dang–Forsyth (2021): benchmark specification

This document fixes the external benchmark's inputs, numerical architecture, and reporting conventions.
Primary source: journal version, SIAM J. Financial Mathematics 12(2), 566–603,
DOI https://doi.org/10.1137/20M1338241 . Author-hosted journal PDF:
https://cs.uwaterloo.ca/~paforsyt/Distributions_2021.pdf .
Page references below use printed journal pages, not PDF indices.

| Item | Specification | Source |
|---|---|---|
| Initial wealth | 100, t0=0 | Table 5.1, p.599; eq.2.2 |
| Horizon | 10 years | Table 5.1, p.599 |
| r | 0.00623 | eq.5.1, p.597 |
| risky drift mu | 0.0816 (total drift, not excess) | eq.5.1 |
| sigma | 0.1863 | eq.5.1 |
| Wealth SDE | dW=(rW+(mu-r)u)dt+sigma*u*dB | eq.2.1, p.570 |
| Contributions / withdrawals | zero | text preceding eq.2.1 |
| Terminal background benefit | zero | objective is W(T), eqs.2.4–2.11 |
| Controls | risky dollars u; unconstrained, signed | eq.2.1; Assumption 3.1 p.574 |
| Insolvency / leverage | trading continues under insolvency; no leverage restriction | Assumption 3.1 |
| PCMV | minimize E[(W(T)-gamma/2)^2]; gamma from eq.4.2 | eqs.2.5,4.2, pp.572,579 |
| DOMV | re-solve E[W]-rho Var[W] at each state/time, connect first actions; fixed rho from eq.4.3 | eqs.2.7,4.3, pp.573,579 |
| cTCMV | one-step equilibrium for E[W]-rho Var[W]; rho from eq.4.4 | eqs.2.8–2.9,4.4 |
| dTCMV | one-step equilibrium; rho/(2W) wealth-dependent risk coefficient | eqs.2.10–2.11, p.574 |
| dTCMV calibration | choose rho such that expected W(T)=target, using equilibrium coefficient equation | eqs.3.5,4.5; p.575,579 |
| Target mean | 125 and 250, explicitly specified inputs | Assumption 4.1; Section 5 / Table 5.1 |
| Published outputs | parameter, mean, median, SD, skew, excess kurtosis; 1/5/10% quantiles and lower tail means; probabilities and conditional means below risk-free wealth / target | Table 5.1 p.599 |
| Additional theory-only outputs | q95 and full CDF | Lemmas 3.4–3.6, pp.576–578; not labelled published Table 5.1 values |

## Input/output firewall

Published table is loaded only by reporting after the numerical policy solve.
Targets 125/250 are inputs under Assumption 4.1, NOT estimates obtained by
inverting Table 5.1. PCMV/DOMV/cTCMV parameters use explicitly stated calibration
conditions (4.2–4.4). dTCMV uses a scalar root on the model's equilibrium ODE,
never published SD, median or rounded risk parameter. Unrounded parameters are
fixed across each strategy's refinement suite. No result matching grid selection.

## Important source discrepancy requiring explicit reporting

The printed exponent in (3.5) contains
`-(integral(r + b*theta - sigma^2*theta^2))`.
A direct moment derivation of the equilibrium in (2.10) instead gives
`-(integral(r + b*theta + sigma^2*theta^2))`:
with m=a*w, q=c*w^2, the first-order condition is
`theta=b/(rho*sigma^2) * (a/c + rho*(a*a/c-1))`.
Both signs will be solved independently and retained; the printed equation will
not be silently edited. Equilibrium-based results are distinguished from literal
(3.5) replication, and remaining disagreement with Table 5.1 is reported.

## Numerical architecture / predeclared experiments

Use discounted terminal-dollar state Y=exp(r*(T-t))*W. This exactly removes the
risk-free drift in the continuous SDE; the benchmark contains no DC contributions.
Re-use the repository's normalized GH generator. Backward conditional means and
second moments use linear interpolation; forward uses the transpose nonnegative
linear deposition. Unconstrained translation / scaling symmetries allow the full
backward lattice recursion to be computed from canonical states. Controls are
numerically maximized/minimized, never populated from theoretical controls.

PCMV solves a centered target-loss problem on a signed geometric lattice, and
DOMV numerically maximizes the original scalarization over the resulting family
of shifted target problems at every decision date. cTCMV uses a uniform lattice.
dTCMV uses a positive geometric lattice with exact frozen-proportion one-step
GBM transitions (no Euler negative-wealth clipping); this is a benchmark adapter,
not an unchanged invocation of the DC-specific Euler transition.

Baseline: Nt=320, GH7; logarithmic spacing .005 for PCMV/dTCMV, uniform spacing
.25 for DOMV/cTCMV. Time-only variants 160/640; space-only variants double/half
spacing; GH-only variants 3/15. PCMV control is risky dollars / centered surplus,
bound 8 with bounds 4/16 audit. dTCMV fraction bound 32 with 16/64 audit.
DOMV/cTCMV terminal-dollar bound 2000 with 1000/4000 audit.
Numerical optimizer uses global samples followed by local interval refinement;
optimizer-only variants are retained. No stochastic seeds are needed.
State domains and boundary audits: signed PCMV surplus |Y-K| <= 1e7,
positive dTCMV Y in [1e-8,1e7], DOMV/cTCMV Y in [-5000,5000]; domain-only
variants halve/double these bounds. Absorbed/out-of-domain mass and moment
losses are explicitly recorded. No finite-domain run is called an exact
unbounded problem.

All variants are reported, including failed/binding configurations. No arbitrary
pass/fail percentage replaces convergence evidence. MVS is out of scope.

## Additional consistency path, declared before its execution

The one-factor suite is retained in full. Because linear interpolation contributes
an error proportional to spacing squared per step, time-only refinement at fixed
spacing need not converge. An additional, explicitly coupled path uses Nt=640/1280
and both spacings divided by 2/4, with GH7 and all remaining settings unchanged.
Both targets and all four strategies are included. This path complements, and
does not replace, the one-factor attribution study. The finest declared path is
used consistently for the final comparison, without selecting the best-matching
result. No further result-dependent mesh selection is performed.

The DC engine's existing conditional-moment loop is extracted into the shared
`finite_model.interpolated_moments` kernel. Both the original constrained solve
and this external adapter invoke it. Regression against the pre-extraction DC
solver is required. Model mapping, admissible controls and symmetry storage are
benchmark-specific; validation does not imply every unchanged DC branch was run.
