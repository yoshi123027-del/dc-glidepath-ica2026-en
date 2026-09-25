# Numerical implementation and reproduction

This directory contains the finite model, policy optimisation and recalibration routines, distribution evaluators, and numerical validation code used for the ICA2026 research. The principal saved results are in [`../results/current/`](../results/current/), and validation outputs are in [`../results/validation/`](../results/validation/).

The four optimised strategies are PCMV, DOMV, cTCMV, and dTCMV. CP is a constant-proportion benchmark rather than a fifth optimisation concept.

## Quick start

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

If you only want to inspect the reported results, **do not rerun the optimisation**. The saved policies and independent Monte Carlo outputs in [`../results/current/`](../results/current/) are sufficient.

## Code map

| File or folder | Role |
|---|---|
| `finite_model.py` | Common finite model and Markov–Gauss–Hermite (MGH) transition machinery |
| `run.py` | Policy optimisation and recalibration to a common expected terminal wealth |
| `validate.py` | Independent off-grid monthly Euler–Monte Carlo evaluation of saved policies |
| `checks.py`, `acceptance.py`, `dom_diagnostics.py` | Backward/forward checks, acceptance diagnostics, and DOMV-specific diagnostics |
| `sensitivity.py` | Robustness checks for the state grid, Gauss–Hermite quadrature, and upper boundary |
| `evaluator_comparison.py`, `plot_evaluator_comparison.py` | MGH forward versus independent Euler–MC comparison for identical saved policies |
| `external_vanstaden_2021/` | External benchmark based on van Staden, Dang and Forsyth (2021) |

## A. Inspect saved results only

No numerical calculation is required. Start with:

- [Principal saved policies and results](../results/current/)
- [Validation and robustness results](../results/validation/)

## B. Reproduce the evaluator comparison

The following commands compare MGH forward propagation with the saved independent Euler–Monte Carlo evaluation while holding the policies fixed:

```bash
python recalibration/evaluator_comparison.py \
  --data-dir results/current \
  --out results/validation/evaluator_comparison

python recalibration/plot_evaluator_comparison.py \
  --data-dir results/current \
  --validation-dir results/validation/evaluator_comparison
```

By default, `evaluator_comparison.py` reuses the stored one-million-path Monte Carlo output. The optional `--rerun-mc` flag intentionally regenerates that Monte Carlo evaluation and is therefore much more expensive. See [`EVALUATOR_COMPARISON.md`](EVALUATOR_COMPARISON.md) for the detailed definitions and output inventory.

## C. Full optimisation and recalibration

A full production-grid rebuild is **computationally and storage intensive**, particularly the target-family calculation used by DOMV. It is not required to inspect or validate the published saved results.

The current production configuration is stored in [`../results/current/config.json`](../results/current/config.json): 480 monthly periods, 6,001 wealth nodes over a state range up to 600, 129 candidate controls, and seven-point Gauss–Hermite quadrature. To reproduce that configuration in a separate output directory, the intended sequence is:

```bash
python recalibration/run.py family --nx 6001 --nc 129 --ng 7 --xmax 600 --step 2 --eval-h 0.025 --eval-ng 31 --out results/reproduction_full
python recalibration/run.py dom    --nx 6001 --nc 129 --ng 7 --xmax 600 --step 2 --eval-h 0.025 --eval-ng 31 --out results/reproduction_full
python recalibration/run.py pc     --nx 6001 --nc 129 --ng 7 --xmax 600 --eval-h 0.025 --eval-ng 31 --out results/reproduction_full
python recalibration/run.py tc     --nx 6001 --nc 129 --ng 7 --xmax 600 --eval-h 0.025 --eval-ng 31 --out results/reproduction_full
python recalibration/run.py cp     --nx 6001 --nc 129 --ng 7 --xmax 600 --eval-h 0.025 --eval-ng 31 --out results/reproduction_full
```

`family` must precede `dom`, because DOMV reads the saved target-family arrays. The `pc`, `tc`, and `cp` tasks do not depend on that family calculation and may be run independently. Using a separate output directory avoids overwriting the authoritative files in `results/current/`.

To regenerate the independent one-million-path terminal evaluation after a full rebuild, the corresponding command is:

```bash
python recalibration/validate.py results/reproduction_full --paths 1000000 --seed 20260912
```

That last command is intentionally expensive and is not needed for ordinary repository use.

## Results and validation

- [Main policies and reported results](../results/current/)
- [Finite-model checks and robustness results](../results/validation/recalibration/)
- [MGH versus independent Euler–MC](../results/validation/evaluator_comparison/)
- [External benchmark](../results/validation/van_staden_2021/)

For a non-technical overview of the research question, strategy labels, and principal results, return to the [top-level README](../README.md).
