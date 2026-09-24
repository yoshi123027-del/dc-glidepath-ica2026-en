# External MGH benchmark: van Staden, Dang and Forsyth (2021)

This module applies the repository's Markov–Gauss–Hermite (MGH) backward-policy and forward-distribution architecture to the unconstrained mean–variance benchmark of van Staden, Dang and Forsyth (2021). It is a numerical policy solve and forward distribution calculation, not a reconstruction of a published table from closed-form moments.

The benchmark uses its own state spaces, controls, objectives, and boundary treatment while sharing the interpolated conditional-moment kernel and normalized GH generator with the DC implementation. It therefore provides evidence about the shared numerical kernel, not a claim that every DC-specific branch is unchanged in the external problem.

## Scope

- PCMV, DOMV, cTCMV, and dTCMV are solved numerically on the declared finite lattices.
- The suite includes baseline, one-factor, and coupled space–time refinement runs for the stated target means.
- The output records mass conservation, boundary occupation, control-domain checks, policy comparisons, distribution comparisons, and convergence results.
- dTCMV is reported as a partial external validation: the source's printed equation and its objective-derived equilibrium expression have an unresolved sign discrepancy. Both calculations are retained; see [BENCHMARK_SPECIFICATION.md](BENCHMARK_SPECIFICATION.md).

## Reproduce

Run from the repository root after installing `requirements.txt`.

```bash
python recalibration/external_vanstaden_2021/run.py --suite all
python recalibration/external_vanstaden_2021/audit.py
python recalibration/external_vanstaden_2021/regression.py
python recalibration/external_vanstaden_2021/report.py
```

The complete stored outputs are in [`results/validation/van_staden_2021/`](../../results/validation/van_staden_2021/). The strategy-specific ZIP files contain complete run arrays and metadata; extract them into that output directory before regenerating reports without re-running the policy solve.

## Key outputs

| File | Contents |
|---|---|
| `published_vs_mgh.csv` | Published and MGH values with absolute and relative differences |
| `final_summary.csv` | Final distribution statistics, CDF gaps, and policy errors |
| `convergence.csv` | All mesh and numerical variants |
| `policy_comparison.csv` | Policy comparisons at selected dates and wealth levels |
| `cdf_comparison.csv.gz` | Final-grid masses and CDFs |
| `boundary_control_diagnostics.csv` | Mass, boundary, moment, and control-domain diagnostics |
| `external_cdf_comparison.*`, `external_convergence.*` | Publication figures |
