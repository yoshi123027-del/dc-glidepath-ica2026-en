# DC Glide Path ICA2026 — Clean Repository Layout

This repository is organized around the **currently adopted Japanese-paper analysis** as of 2026-08-15.

## Directory map

- `00_manifest/` — status, parameter ledger, reproducibility notes, and codebook.
- `01_final_paper/` — canonical paper naming and build notes.
- `02_adopted_analysis/` — the numerical algorithms actually adopted in the paper.
- `03_adopted_results/` — compact canonical result tables used to verify a rebuild.
- `04_supplementary/` — supplementary-analysis guidance.
- `90_rejected/` — superseded figures/results/scripts retained for traceability.

## Adopted analysis rule

The final Japanese-paper line returns to the **original numerical method and plotting logic** and changes calibration parameters only. The common-mean anchor is the dTCMV–MVS case with `gamma0 = 2.5` and `eta0 = 0`, whose terminal mean is `70.34483966999646`.

The original-method source tree is pinned under `02_adopted_analysis/01_reference_algorithms/` from commit `a3b83a590474a4d88642d7ce95cfddda9b60461a`.

The repository state immediately before this reorganization is recoverable from commit `aa52f01b138115f315702345374d27246dc027eb`.

## Final figure set

1. Figure 1 — terminal CDF + terminal density + q05–q95 interval.
2. Figure 2 — mass-weighted glide paths.
3. Figure 3 — cTCMV / dTCMV reachable-state policy heatmaps.
4. Figure 4 — strict vs clipped glide paths, displayed on a 0%–100% y-axis.
5. Figure 5 — common-state rolling comparison.
6. Figure 6 — MVS glide paths + terminal density + q05–q95 interval.

## Reproduction

The tested GitHub Actions workflows under `.github/workflows/` reproduce the parameter-only original-method analysis. Compare rebuilt summaries with `03_adopted_results/` before treating any new output as paper-ready.
