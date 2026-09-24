# MGH forward vs independent Euler–MC

This validation compares two evaluators using exactly the same five saved risky-dollar feedback policies: `PCMV`, `DOMV`, `cTCMV`, `dTCMV`, and `CP`. It does not optimise a policy, recalibrate a coefficient, alter a market assumption, or select a new random seed. The common expected-terminal-wealth target is 84.78.

## Inputs and outputs

- Inputs: [`results/current/`](../results/current/) contains the policy arrays, parameters, configuration, and saved independent 1,000,000-path Euler–MC output.
- Outputs: [`results/validation/evaluator_comparison/`](../results/validation/evaluator_comparison/) contains the terminal-distribution comparison, glide-path comparison, mass audit, validation metadata, and figures.

MGH forward propagation uses the finite wealth grid, GH7 shocks, and non-negative linear mass deposition. Independent Euler–MC evaluates risky-dollar interpolation at the exact initial state without an upper wealth cap. The comparison therefore tests differences in induced distributions under fixed policies; it does not establish continuous-time exactness or a uniform discretisation-error bound.

## Reproduce

Run from the repository root after installing `requirements.txt`.

```bash
python recalibration/evaluator_comparison.py \
  --data-dir results/current \
  --out results/validation/evaluator_comparison
python recalibration/plot_evaluator_comparison.py \
  --data-dir results/current \
  --validation-dir results/validation/evaluator_comparison
```

Adding `--rerun-mc` to the first command regenerates the independent 1,000,000-path Euler–MC evaluation before checking it against the saved output. Without that flag, the script reuses the stored MC paths and evaluates only the MGH forward propagation.

## Saved comparison files

| File | Contents |
|---|---|
| `evaluator_distribution_comparison.csv` | Mean, SD, quantiles, tail means, and skewness for both evaluators |
| `evaluator_comparison.csv` | Exact CDF distance and signed terminal-statistic differences |
| `evaluator_glidepath_comparison.csv` | Maximum gap, MAE, RMSE, and timing of the largest glide-path difference |
| `evaluator_glidepaths.csv` | Monthly MGH, MC, and signed glide-path differences |
| `evaluator_mass_audit.csv` | MGH mass, moment, and boundary diagnostics |
| `evaluator_validation.json` | Input hashes, settings, and validation checks |
| `fig_evaluator_*.png` / `.pdf` | Terminal-CDF and mean-glide-path overlays |
