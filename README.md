# ICA2026: Endogenous Glide Paths for Defined Contribution Pensions

## Research question

This repository contains the English-language materials for a study of defined contribution (DC) pension glide paths. Rather than imposing a glide-path shape exogenously, the study derives risky-asset allocation endogenously from the member's objective and current state.

Under common market, contribution, and DC investment constraints, the analysis compares terminal distributions at matched initial expected terminal wealth and conditional outcomes from common intermediate wealth states. The corresponding Japanese source repository is [`dc-glidepath-ica2026`](https://github.com/yoshi123027-del/dc-glidepath-ica2026).

## Strategies and practical interpretation

- **PCMV** represents commitment to the plan chosen at enrolment and serves as a design benchmark.
- **DOMV** re-optimises from the member's current state and is a candidate for periodic review.
- **cTCMV** is a time-consistent equilibrium policy with constant variance aversion and is a candidate for a standard default.
- **dTCMV** is a time-consistent equilibrium policy with variance aversion that depends on total pension wealth and is a candidate for personalised advice.
- **CP** is the constant-proportion benchmark.

The proposed operating structure has three layers—cTCMV for the standard default, DOMV for periodic review, and dTCMV for personalised advice—while PCMV remains the design benchmark. Mean–variance–skewness (MVS) results are retained as an exploratory extension rather than as an additional operating layer.

## Numerical method

All four optimisation concepts use a common Markov–Gauss–Hermite (MGH) backward–forward framework. The implementation distinguishes directly constrained feedback from the clipped approximation obtained by projecting the corresponding unconstrained policy onto the feasible interval. Saved policies are evaluated both by MGH forward propagation and by an independent one-million-path monthly Euler–Monte Carlo evaluator; no policy is re-optimised during the evaluator comparison.

The current baseline uses 480 monthly periods, a state ceiling of 600, 6,001 wealth nodes at spacing 0.1, 129 candidate controls, and seven-point Gauss–Hermite quadrature.

## Principal results

The table below reproduces the current independent Monte Carlo results. Expected terminal wealth is matched at approximately 84.78 so that distribution shape and the timing of risk-taking can be compared on a common basis.

| Policy | Mean | SD | q05 | Median | q95 | Lower 5% mean |
|---|---:|---:|---:|---:|---:|---:|
| PCMV | 84.7715 | 18.9424 | 38.1937 | 92.3501 | 100.2246 | 28.7985 |
| DOMV | 84.7831 | 24.4139 | 46.8690 | 83.4760 | 127.0602 | 39.2036 |
| cTCMV | 84.7913 | 23.1570 | 47.1606 | 84.4674 | 123.4706 | 38.5271 |
| dTCMV | 84.7863 | 32.3127 | 43.1728 | 79.2581 | 145.2516 | 37.3322 |
| CP | 84.7854 | 31.6874 | 45.2269 | 78.9299 | 144.1862 | 39.8995 |

The results illustrate that target-date, U-shaped, and intermediate glide paths can arise endogenously from different decision principles and specifications of variance aversion. They do not imply that one policy dominates on every outcome measure.

## Paper

The manuscript PDF is not distributed from this repository. This repository is intended as the English-language research, code, data, and validation companion.

## Results and reproducibility

- [`results/current/`](results/current/) contains the saved policies, calibrated parameters, and principal terminal-distribution and mean-glide-path outputs. The reported results can be inspected without rerunning the optimisation.
- [`recalibration/`](recalibration/) contains the finite model, optimisation and recalibration code, MGH and independent Monte Carlo evaluators, sensitivity analysis, evaluator comparison, and external benchmark implementation.
- [`CODEBOOK.md`](CODEBOOK.md) provides a concise path-by-path guide.

## Validation

- [MGH backward/forward consistency](results/validation/recalibration/) audits backward moments, forward distributions, probability mass, continuous controls, grids, boundaries, and quadrature.
- [MGH versus independent Euler–Monte Carlo](results/validation/evaluator_comparison/) applies both evaluators to the same five saved policies and compares terminal distributions and mean glide paths.
- [External benchmark: van Staden, Dang and Forsyth (2021)](results/validation/van_staden_2021/) applies the common MGH engine to an external mean–variance problem and compares policies and terminal distributions.

## Repository structure

| Folder | Purpose |
|---|---|
| [`recalibration/`](recalibration/) | Main numerical implementation |
| [`results/current/`](results/current/) | Principal saved policies and reported results |
| [`results/validation/`](results/validation/) | Validation and robustness checks |
| [`archive/`](archive/) | Legacy and exploratory materials |

## Archive

Earlier manuscripts, superseded grids and outputs, legacy code, and exploratory analyses are retained under [`archive/`](archive/) for auditability. They are not needed for reading or reproducing the current results. Some archived manuscripts remain in Japanese because they are preserved as historical source artifacts rather than as current English deliverables.

This repository is released under the [MIT License](LICENSE).
