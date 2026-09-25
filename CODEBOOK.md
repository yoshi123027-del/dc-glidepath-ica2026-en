# Code and Data Guide

The [top-level README](README.md) is the primary entry point for the project. Use this file when you need a concise map of the repository after reading the research overview.

## Current authoritative areas

| Path | Purpose |
|---|---|
| [`paper/`](paper/) | Manuscript context and navigation only; the manuscript PDF is not distributed here |
| [`recalibration/`](recalibration/) | Principal finite-model, optimisation, recalibration, evaluation, and validation code |
| [`results/current/`](results/current/) | Saved policies, calibration values, principal independent-MC outputs, and the current mean-glide-path figure |
| [`results/validation/recalibration/`](results/validation/recalibration/) | MGH consistency, grid, boundary, and quadrature checks |
| [`results/validation/evaluator_comparison/`](results/validation/evaluator_comparison/) | MGH versus independent Euler–MC comparison for identical saved policies |
| [`results/validation/van_staden_2021/`](results/validation/van_staden_2021/) | Inputs, outputs, and figures for the external benchmark |

## Historical areas

| Path | Purpose |
|---|---|
| [`archive/papers/`](archive/papers/) | Earlier manuscript versions and associated figures |
| [`archive/legacy_v5/`](archive/legacy_v5/) | Legacy calculations, code, and audit materials |
| [`archive/old_results/`](archive/old_results/) | Superseded numerical outputs not used for the current ICA2026 results |
| [`archive/exploratory/`](archive/exploratory/) | Exploratory materials, earlier manifests, retired automation assets, and historical reference material |

Files under `archive/` are retained for auditability. They are not authoritative current results, and some remain in Japanese by design.
