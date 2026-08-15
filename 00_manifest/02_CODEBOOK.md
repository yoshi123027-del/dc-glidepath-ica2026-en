# Codebook

## Core solvers

| File | Purpose |
| --- | --- |
| `scripts/01_solvers/pcmv_domv_solver_20260713.py` | Core solver for the directly constrained quadratic-loss problems underlying PCMV and DOMV |
| `scripts/03_rolling/recompute_d0_rolling.py` | Recomputes cTCMV, dTCMV, and constant-proportion strategies for the baseline case with $D_T=0$, together with terminal distributions and rolling conditional evaluations |
| `scripts/01_solvers/dtcmv_mvs_solver_20260713.py` | Backward recursion for the first, second, and third moments in the dTCMV–MVS extension |
| `scripts/02_calibration/run_mvs_refined_calibration.py` | Performs refined calibration of the MVS coefficients |

## Diagnostics and sensitivity analysis

| File | Purpose |
| --- | --- |
| `scripts/03_rolling/detailed_dtcmv_rolling_overlay_20260713.py` | dTCMV rolling conditional evaluation by wealth quantile |
| `scripts/04_sensitivity/low_balance_refined_sensitivity_20260714.py` | Local grid refinement and sensitivity analysis focused on low-wealth states |
| `scripts/04_sensitivity/rebuild_baseline_sensitivity.py` | Reaggregates the baseline and sensitivity results |
| `scripts/04_sensitivity/numerical_diagnostics_20260718.py` | Recomputes backward–forward moment consistency, pre-normalisation mass, lower- and upper-bound overflow, and dTCMV upper-domain sensitivity |
| `scripts/05_figures/rebuild_all_corrected_glide_outputs.py` | Rebuilds the corrected glide-path and distribution outputs |
| `scripts/03_rolling/rebuild_rolling_equal_mean.py` | Reconstructs rolling results under equal-mean calibration |
| `scripts/02_calibration/equal_mean_calibration_20260713.py` | Organises the calibration values used in equal-mean comparisons |
| `scripts/04_sensitivity/add_all_clip_overlays_20260721.py` | Compares directly constrained controls and clipped unconstrained approximations for all four MV solution concepts |

## Figure generation

| File | Purpose |
| --- | --- |
| `scripts/05_figures/localize_paper_figures_ja_20260717.py` | Regenerates the Japanese-labelled paper figures from the bundled NPZ and CSV files |
| `scripts/03_rolling/make_common_state_rolling_figure_20260713.py` | Generates the common-state rolling comparison figure |

## Auxiliary processes

`scripts/90_workers/mvs_worker.py` and `scripts/90_workers/sensitivity_worker.py` are worker processes used for split or parallel execution. They are intended to be invoked by their parent scripts rather than run independently.

## Principal data files

| File | Contents |
| --- | --- |
| `results/monthly_D0_policy_arrays.npz` | Wealth grid, policies, forward distributions, glide paths, and upper-bound binding rates for the monthly baseline |
| `results/monthly_baseline_D0_summary.csv` | Terminal-distribution and glide-path summary for all strategies |
| `results/equal_mean_calibration.csv` | Calibration values that produce the common target mean |
| `results/rolling_conditional_D0_N480.csv` | Rolling conditional statistics at representative dates and states |
| `results/rolling_validation_D0_N480.csv` | Backward–forward moment-consistency diagnostics for the rolling calculations |
| `results/moment_consistency_D0_N480.csv` | First- and second-moment consistency at time zero |
| `results/mass_truncation_summary_D0_N480.csv` | Strategy-level summary of pre-normalisation mass error and lower- and upper-bound overflow |
| `results/mass_truncation_by_time_D0_N480.csv` | Time-by-time record of pre-normalisation mass and boundary overflow |
| `results/xmax_sensitivity_dtcmv_D0_N480.csv` | Nested upper-domain sensitivity analysis for dTCMV |
| `results/xmax900_equal_mean_dtcmv_D0_N480.csv` | dTCMV summary after recalibration to the common mean on the enlarged domain |
| `results/rolling_quantile_detail_D0_N480.csv` | Rolling results by wealth quantile |
| `results/dtcmv_mvs_arrays.npz` | Policy, distribution, and glide-path arrays for dTCMV–MVS |

## Interpretation notes

- The monthly baseline uses 480 decision periods over 40 years.
- The sensitivity figures use a coarser 80-step screening grid and should not be interpreted as a complete convergence study.
- Directly constrained feedback controls and clipped approximations are different policies, even where their plotted glide paths appear close.
- Positive-skewness MVS results are exploratory because the objective can be non-concave and sensitive to the control grid, state grid, tail truncation, and interpolation.
