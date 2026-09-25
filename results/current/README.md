# Current reported results

This directory contains the **authoritative saved numerical artefacts** underlying the principal results of the ICA2026 manuscript. These files are current; superseded grids and earlier outputs are kept under [`../../archive/`](../../archive/).

| Files | Contents |
|---|---|
| `PCMV`, `DOMV`, `cTCMV`, `dTCMV`, `CP` (`.json`, `.npz`) | Calibration parameters and saved policies |
| `config.json` | Common finite-model configuration |
| `independent_mc.csv` | Principal terminal-distribution statistics from the independent Euler–Monte Carlo evaluation |
| `independent_mc.npz` | Terminal samples and mean glide paths from the independent Euler–Monte Carlo evaluation |
| `paired_means.csv` | Supplementary paired mean differences based on common random numbers |
| `fig_glidepaths_recalibrated_v12.png` | Current English mean-glide-path figure corresponding to the recalibrated policies |

Expected terminal wealth is matched near 84.78. Wealth is expressed in the model's normalised units, with annual contribution equal to 1 and initial DC balance equal to 1/12.

For a readable summary of the strategy labels and the principal statistics, see the [top-level README](../../README.md). For validation and robustness checks, see [`../validation/`](../validation/). For reproduction instructions, see [`../../recalibration/`](../../recalibration/).
