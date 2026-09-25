# Main reported results

This directory contains the saved data that directly underlie the principal results in the current paper.

| Files | Contents |
|---|---|
| `PCMV`, `DOMV`, `cTCMV`, `dTCMV`, `CP` (`.json`, `.npz`) | Calibration parameters and saved policies |
| `config.json` | Common finite-model configuration |
| `independent_mc.csv` | Principal terminal-distribution statistics reported in the paper |
| `independent_mc.npz` | Terminal samples and mean glide paths from the independent Euler–Monte Carlo evaluation |
| `paired_means.csv` | Supplementary paired mean differences based on common random numbers |

See [`../validation/`](../validation/) for validation and robustness checks.
