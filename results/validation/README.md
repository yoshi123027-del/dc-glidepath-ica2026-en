# Current validation and robustness checks

This directory contains the **current English validation outputs** for the ICA2026 research. Historical or superseded validation material belongs under [`../../archive/`](../../archive/) and should not be interpreted as current results.

| Folder | Contents |
|---|---|
| [`recalibration/`](recalibration/) | MGH backward/forward consistency, continuous-control audits, and grid, boundary, and Gauss–Hermite quadrature checks |
| [`evaluator_comparison/`](evaluator_comparison/) | MGH forward versus independent Euler–MC comparison using identical saved policies; the English report is `evaluator_report.md` |
| [`van_staden_2021/`](van_staden_2021/) | External benchmark based on van Staden, Dang and Forsyth (2021) |

The principal saved policies and reported results are in [`../current/`](../current/). Reproduction instructions are in [`../../recalibration/`](../../recalibration/), and the non-technical overview is in the [top-level README](../../README.md).
