# Python Code Guide

The Python code is organised into numbered directories by purpose. The numbers indicate a broad reading and reproduction workflow rather than a strict dependency order. Run the commands from the repository root.

## Quick start

Regenerate the paper figures from the bundled CSV and NPZ files. This is much faster than rerunning the full monthly optimisation.

```bash
python scripts/05_figures/localize_paper_figures_ja_20260717.py
```

The script name retains the `_ja_` suffix because it reproduces the Japanese-labelled figures used in the Japanese manuscript. The underlying numerical outputs are language-neutral.

## 01_solvers: core solvers

| File | Purpose |
| --- | --- |
| `pcmv_domv_solver_20260713.py` | Directly constrained solvers for PCMV and DOMV |
| `dtcmv_mvs_solver_20260814.py` | **Current baseline-consistent dTCMV–MVS runner.** Uses the same dTCMV variance-aversion scale as the main MV equal-mean comparison, so the `eta0=0` case nests the MV baseline. |
| `dtcmv_mvs_solver_20260713.py` | Historical dTCMV–MVS numerical core retained for reproducibility of earlier outputs. Its old entry point used `gamma0=2.5` and should not be used for the current paper baseline. |

The corrected MVS baseline is `gamma0 = 1.193359375`, matching the dTCMV row of `results/equal_mean_calibration.csv`. The previous MVS-only baseline `gamma0=2.5` changed both the variance-aversion coefficient and the skewness coefficient at the same time and therefore did not provide a clean MV-to-MVS nesting comparison.

## 02_calibration: calibration

| File | Purpose |
| --- | --- |
| `equal_mean_calibration_20260713.py` | Organises calibration values for equal-mean comparisons |
| `run_mvs_refined_calibration.py` | Performs refined calibration of the MVS coefficients from the corrected dTCMV baseline |

## 03_rolling: rolling conditional evaluation

| File | Purpose |
| --- | --- |
| `recompute_d0_rolling.py` | Recomputes the baseline case, terminal distributions, and rolling conditional evaluations |
| `detailed_dtcmv_rolling_overlay_20260713.py` | Produces dTCMV rolling evaluations by wealth quantile |
| `make_common_state_rolling_figure_20260713.py` | Generates the common-state rolling comparison figure |
| `rebuild_rolling_equal_mean.py` | Reconstructs rolling results under equal-mean calibration |

## 04_sensitivity: sensitivity analysis

| File | Purpose |
| --- | --- |
| `low_balance_refined_sensitivity_20260714.py` | Sensitivity analysis of directly constrained solutions with a refined low-wealth grid |
| `add_all_clip_overlays_20260721.py` | Adds the directly constrained solution (solid line) and the corresponding clipped unconstrained approximation (same-colour dashed line) to all PCMV, DOMV, cTCMV, and dTCMV panels, and exports gap CSV/NPZ files |
| `add_ctcmv_clip_overlays_20260721.py` | Older auxiliary script restricted to cTCMV; use the preceding script for comparisons across all solution concepts |
| `rebuild_baseline_sensitivity.py` | Reaggregates the baseline and sensitivity results |
| `numerical_diagnostics_20260718.py` | Recomputes moment consistency, pre-normalisation mass, boundary overflow, and dTCMV upper-domain sensitivity |

Regenerate the sensitivity figures containing clipped approximations for all solution concepts with:

```bash
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
```

Within each panel, colours are held fixed for the same market and contribution scenario. Solid lines denote directly constrained feedback controls and dashed lines denote the clipped approximation constructed from the corresponding unconstrained solution. The PCMV, DOMV, cTCMV, and dTCMV approximations are derived from their own solution concepts rather than from one common formula. Each policy is propagated forward under the wealth distribution that it generates itself.

Principal outputs include:

- `figs/fig_D_sensitivity_glidepaths_N80.png` / `figs/fig_D_sensitivity_glidepaths_N80.svg`
- `figs/fig_r_sensitivity_glidepaths_N80.png` / `figs/fig_r_sensitivity_glidepaths_N80.svg`
- `figs/fig_mu_sensitivity_glidepaths_N80.png` / `figs/fig_mu_sensitivity_glidepaths_N80.svg`
- `figs/fig_sigma_sensitivity_glidepaths_N80.png` / `figs/fig_sigma_sensitivity_glidepaths_N80.svg`
- `figs/fig_contrib_profile_sensitivity_glidepaths_N80.png` / `figs/fig_contrib_profile_sensitivity_glidepaths_N80.svg`
- `all_strategies_strict_vs_clip_sensitivity_summary.csv`
- `all_strategies_strict_vs_clip_sensitivity_paths.csv`
- `all_strategies_strict_vs_clip_sensitivity_glidepaths.npz`

## 05_figures: output reconstruction and plotting

| File | Purpose |
| --- | --- |
| `rebuild_all_corrected_glide_outputs.py` | Reconstructs the corrected glide-path and distribution outputs in one run |
| `localize_paper_figures_ja_20260717.py` | Regenerates the Japanese-labelled paper figures |

## Validation

For the dTCMV–MVS nesting check, run:

```bash
python validation/validate_dtcmv_mvs_nesting.py
```

This compares the `eta0=0` MVS glide path under `gamma0=1.193359375` with the saved main-paper dTCMV MV glide path. Because the MV and MVS implementations use different control-grid/refinement rules, the check is a numerical consistency diagnostic rather than a bit-for-bit identity test.

## 90_workers: auxiliary workers

`mvs_worker.py` and `sensitivity_worker.py` support split or parallel execution. They are not normally run directly.

## Main recomputation sequence

```bash
python scripts/01_solvers/pcmv_domv_solver_20260713.py
python scripts/03_rolling/recompute_d0_rolling.py
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
python scripts/04_sensitivity/numerical_diagnostics_20260718.py
python scripts/01_solvers/dtcmv_mvs_solver_20260814.py
python scripts/02_calibration/run_mvs_refined_calibration.py
python validation/validate_dtcmv_mvs_nesting.py
```

After reorganisation, all scripts continue to read from and write to the repository-level `results/` and `figs/` directories.
