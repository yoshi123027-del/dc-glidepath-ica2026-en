# DC Glide-Path Optimisation: ICA2026 Reproducibility Code

This repository provides reproducibility code, key intermediate results, and figure-generation data for an ICA2026 paper on constrained dynamic mean–variance optimisation for defined contribution (DC) pension plans.

## Models covered

- Pre-commitment mean–variance (PCMV)
- Dynamically optimal mean–variance (DOMV)
- Time-consistent mean–variance with constant variance aversion (cTCMV)
- Time-consistent mean–variance with total-pension-wealth-dependent variance aversion (dTCMV)
- Mean–variance–skewness (MVS) extensions
- Directly constrained controls versus clipped unconstrained approximations

All principal calculations impose the DC investment constraint

```text
0 <= risky investment <= current DC balance
```

which rules out short selling and borrowing against future contributions.

The MVS calculations should be interpreted as research extensions. In particular, the equality between cTCMVS and cTCMV is established for the constant-coefficient unconstrained Black–Scholes case and is inherited by the corresponding clipped controls. It is not established for directly constrained controls.

## Recommended environment

- Python 3.11 or 3.12
- NumPy, pandas, Matplotlib, SciPy, Numba, Pillow

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The bundled localisation script reproduces the Japanese-labelled paper figures. The numerical arrays, CSV files, and optimisation outputs are language-neutral and can be used to generate English-labelled figures separately.

## Validation of the four MV solution concepts

Appendix A.3 of the paper validates all four solution concepts because Table 5.1 of van Staden, Dang and Forsyth (2021) reports numerical examples for PCMV, DOMV, cTCMV, and dTCMV. The constrained monthly implementation developed for this paper is audited separately through independent forward propagation, probability-mass diagnostics, boundary diagnostics, and nested-grid checks.

```bash
python validation/external_validation_vanstaden2021_all_mv.py
python validation/run_all_validations.py
```

Validation outputs are stored in `results/validation/`. In the current reference version, all configured checks pass:

- PCMV: 58 external checks + 7 internal checks = 65/65
- DOMV: 58 external checks + 7 internal checks = 65/65
- cTCMV: 58 external checks + 7 internal checks = 65/65
- dTCMV: 52 external checks + 15 internal checks = 67/67

For PCMV, Table 5.1 is recomputed from the reflected-lognormal closed form. For DOMV and cTCMV, it is recomputed from the closed-form normal terminal distribution. For dTCMV, the published mean and standard deviation are used to identify the lognormal terminal distribution, from which the remaining distributional statistics are recomputed. This is a distribution-level external validation; it does not claim to reproduce the unpublished path of the time-varying coefficient. See [validation/README.md](validation/README.md) for details.

## Main execution order

A complete monthly recomputation is computationally intensive. For a quick inspection, first regenerate the paper figures from the bundled CSV and NPZ files:

```bash
python scripts/05_figures/localize_paper_figures_ja_20260717.py
```

The main recomputation sequence is:

```bash
python scripts/01_solvers/pcmv_domv_solver_20260713.py
python scripts/03_rolling/recompute_d0_rolling.py
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
python scripts/04_sensitivity/numerical_diagnostics_20260718.py
python validation/run_all_validations.py
python scripts/01_solvers/dtcmv_mvs_solver_20260713.py
python scripts/02_calibration/run_mvs_refined_calibration.py
```

The directory numbers indicate the broad workflow rather than a strict dependency order. See [scripts/README.md](scripts/README.md) and [CODEBOOK.md](CODEBOOK.md) for the role and execution status of each script.

## Directly constrained controls and clipped approximations

The 80-step-per-horizon sensitivity analysis compares, for PCMV, DOMV, cTCMV, and dTCMV:

- the feedback obtained by solving the constrained problem directly; and
- the clipped approximation obtained by projecting the corresponding unconstrained solution onto `0 <= pi <= x`.

In the figures:

- solid lines denote directly constrained feedback controls;
- dashed lines of the same colour denote clipped unconstrained approximations; and
- each policy is propagated forward under the wealth distribution that it generates itself.

![Directly constrained and clipped controls under expected-return sensitivity](supplementary/figures/fig_mu_sensitivity_glidepaths_N80.svg)

The PCMV approximation uses its fixed-target unconstrained solution; DOMV uses the unconstrained solution associated with re-optimisation at each state; cTCMV uses the constant-variance-aversion analytical solution; and dTCMV uses the time-varying coefficient obtained from the Volterra equation. See [`add_all_clip_overlays_20260721.py`](scripts/04_sensitivity/add_all_clip_overlays_20260721.py) for the implementation, the [supplementary figures page](supplementary/figures/README.md) for all five figures, and [`all_strategies_strict_vs_clip_sensitivity_summary.csv`](results/sensitivity/all_strategies_strict_vs_clip_sensitivity_summary.csv) for the numerical gap summary.

A small difference under the baseline parameters does not imply that the same approximation remains accurate after parameters change. In particular, PCMV and dTCMV show materially larger gaps between the directly constrained and clipped policies in some scenarios.

## Additional diagnostics

The dTCMV U-shaped glide-path decomposition reported in Table 10 and the supplementary cross-check between analytical projection regions and directly searched regions can be reproduced from `diagnostics/`:

```bash
python diagnostics/additional_diagnostics.py
python diagnostics/unconstrained_dtcmv_theta.py
python diagnostics/recompute_crosscheck.py
python diagnostics/pcmv_crosscheck.py
```

See [diagnostics/README.md](diagnostics/README.md) for the execution order and output CSV files.

## Directory structure

- `results/`: principal paper tables, calibration values, rolling evaluations, and policy arrays
- `results/validation/`: automated results for the four MV solution concepts
- `results/sensitivity/`: sensitivity gaps between directly constrained and clipped controls
- `validation/`: external and internal validation corresponding to Appendix A.3
- `diagnostics/`: the Table 10 U-shape decomposition and supplementary free-boundary cross-checks
- `figs/`: paper figures and source figures required for regeneration
- `supplementary/figures/`: supplementary figures omitted from the main paper, with English explanations
- `scripts/01_solvers/`: core PCMV, DOMV, and dTCMV–MVS solvers
- `scripts/02_calibration/`: equal-mean and MVS-coefficient calibration
- `scripts/03_rolling/`: rolling conditional evaluation and related figures
- `scripts/04_sensitivity/`: sensitivity analysis and numerical diagnostics
- `scripts/05_figures/`: paper-figure regeneration and localisation
- `scripts/90_workers/`: auxiliary workers for split or parallel execution; normally not run directly

## Supplementary figures

Sensitivity analysis, constraint diagnostics, rolling conditional evaluation, and detailed MVS figures are collected on the [supplementary figures page](supplementary/figures/README.md). Each figure is followed by an English explanation.

## Reproducibility notes

- `monthly_D0_policy_arrays.npz` contains policy and distribution arrays from the 40-year monthly baseline with 480 time steps.
- The sensitivity figures use an 80-step screening calculation. The solid and dashed glide paths are not evaluated under a common state distribution; each is based on the wealth distribution generated by its own policy.
- `numerical_diagnostics_20260718.py` recomputes pre-normalisation mass, lower- and upper-bound overflow, backward–forward moment consistency, and dTCMV upper-domain sensitivity.
- Because the dTCMV right-tail statistics on the baseline domain `x_max=300` are sensitive to the upper boundary, tail validity is assessed on nested grids with `x_max=900` or larger.
- Results with a positive MVS skewness coefficient are exploratory and may depend on non-concavity and discretisation.
- Figure generation from the bundled arrays is fast, but a complete recomputation from the solvers requires substantial CPU time and memory.
- The CSV and NPZ files corresponding to each reported result should be treated as the numerical source of record.

## Citation

When using this code, please cite the ICA2026 paper and the repository release after publication. Full bibliographic information will be added once the paper has been accepted and published.

## Licence

MIT License. See [LICENSE](LICENSE) for details.
