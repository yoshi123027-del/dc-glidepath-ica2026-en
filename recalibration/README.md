# Numerical implementation

This directory contains the finite model, policy optimisation and recalibration routines, distribution evaluators, and validation code used in the paper. The principal saved results are in [`../results/current/`](../results/current/), and validation outputs are in [`../results/validation/`](../results/validation/).

## Code map

| File or folder | Role |
|---|---|
| `finite_model.py` | Common finite model, MGH transition kernel, and computational basis for PCMV, cTCMV, and dTCMV |
| `run.py` | Policy optimisation and recalibration to a common expected terminal wealth |
| `validate.py` | Independent Euler–Monte Carlo evaluation of saved policies |
| `checks.py`, `acceptance.py`, `dom_diagnostics.py` | Backward–forward consistency checks, acceptance decisions, and DOMV diagnostics |
| `sensitivity.py` | Robustness checks for the state grid, GH quadrature, and upper boundary |
| `evaluator_comparison.py`, `plot_evaluator_comparison.py` | MGH forward versus independent Euler–MC comparison for identical policies |
| `external_vanstaden_2021/` | External benchmark based on van Staden, Dang and Forsyth (2021) |

## Results and validation

- [Main policies and reported results](../results/current/)
- [Finite-model checks and robustness results](../results/validation/recalibration/)
- [MGH vs independent Euler–MC](../results/validation/evaluator_comparison/)
- [External benchmark](../results/validation/van_staden_2021/)

Install the dependencies with `python -m pip install -r requirements.txt` from the repository root. The calculations are intentionally separated by script; run only the module needed for a full reproduction. The stored outputs are sufficient for inspecting the reported results.
