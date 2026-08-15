# Adopted result checks

These are compact **verification tables**, not an attempt to store every intermediate array in Git.

The full parameter-only computation is reproducible from `02_adopted_analysis/` and the numbered GitHub Actions workflows. Large arrays and complete figure bundles should be regenerated rather than duplicated across revisions.

Canonical checks:

- `01_baseline_summary.csv` — five-strategy monthly baseline under the final common-mean calibration.
- `02_mvs_fixed_gamma_summary.csv` — Figure 6 fixed-`gamma0=2.5` MVS cases.
- `03_strict_vs_clip_summary.csv` — Figure 4 policy comparison after recalibration.
- `04_numerical_checks.md` — moment consistency and state-grid truncation checks.

If a rebuild materially disagrees with these values, do not use its figures in the paper until the discrepancy is explained.
