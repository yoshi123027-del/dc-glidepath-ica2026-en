# MGH versus Independent Monte Carlo Evaluation of Identical Saved Policies

This report summarises the differences between MGH forward evaluation and independent Euler–Monte Carlo evaluation while holding fixed the five policies used in the current paper.

## Results

The policies, calibration coefficients, target of 84.78, and market assumptions are unchanged. The MGH evaluation uses forward probability-mass propagation over 480 months with wealth-grid spacing 0.1, 6,001 state points, and seven-point Gauss–Hermite quadrature. The Monte Carlo results use the one million paths underlying the paper. A verification run that regenerated every path with the same seed produced a maximum terminal-wealth difference of 5.90e-9 and a maximum mean-allocation difference below 1.63e-14. The saved Monte Carlo values used in the paper were not replaced.

| Policy | Maximum CDF difference | Maximum glide-path difference (pp) | MAE (pp) | RMSE (pp) | Time of maximum difference (years) |
|---|---:|---:|---:|---:|---:|
| PCMV | 0.008514 | 0.0684 | 0.0184 | 0.0239 | 9.92 |
| DOMV | 0.002079 | 0.0948 | 0.0353 | 0.0455 | 39.50 |
| cTCMV | 0.001819 | 0.1688 | 0.0278 | 0.0470 | 13.50 |
| dTCMV | 0.001257 | 0.5889 | 0.0704 | 0.1224 | 34.75 |
| CP | 0.001434 | 0.0000 | 0.0000 | 0.0000 | — |

An allocation difference of 0.01 equals one percentage point (pp). The CP difference is numerical round-off and remains below 8.6e-12 pp.

The signs below are **MGH minus MC**. Wealth is expressed in the same units as in the paper.

| Policy | Mean difference | SD difference | q05 difference | Median difference | q95 difference | Lower 5% mean difference |
|---|---:|---:|---:|---:|---:|---:|
| PCMV | -0.0042 | +0.0169 | -0.0937 | -0.0501 | +0.0754 | -0.0727 |
| DOMV | +0.0030 | +0.0524 | -0.0690 | +0.0240 | +0.1398 | -0.1263 |
| cTCMV | -0.0371 | -0.0100 | -0.0606 | -0.0674 | -0.0706 | -0.1346 |
| dTCMV | -0.0517 | -0.0947 | -0.0728 | +0.0419 | -0.3516 | -0.0791 |
| CP | -0.0054 | +0.0568 | -0.0269 | -0.0299 | +0.1138 | -0.0857 |

- The terminal CDFs broadly overlap. PCMV has the largest maximum CDF difference, 0.008514 (approximately 0.8514 percentage points); the steps in its concentrated discrete CDF should also be taken into account.
- dTCMV has the largest mean-glide-path difference, 0.5889 pp at 34.75 years. This is the difference obtained when the same policy is weighted by two different state distributions; it is not a change in the optimised policy.
- dTCMV also has the largest SD and q95 differences, while its maximum CDF difference is the smallest of the five policies. No simple correspondence or causal relation between the glide-path and CDF discrepancies is claimed.
- CP provides a reference case: its mean glide paths coincide, but its terminal distributions still differ because of evaluator effects such as state discretisation and probability-mass allocation.
- CP has an SD of 31.7442 under MGH and 31.6874 under MC, a relative difference of approximately 0.18%.
- These findings do not change the principal Monte Carlo values, the main distributional patterns, or the design conclusions based on the decision principles. They also do not prove convergence to the continuous-time exact solution, global optimality, or a uniform error bound.

## Initialisation and numerical audits

Following the definition in Appendix B of the paper, the exact initial balance of 1/12 is used and probability mass is assigned to the grid only after the first transition. This is explicitly distinguished from the two-point initial mass allocation used in `checks.py`.

The maximum monthly probability-mass error is below 2.2e-14; negative mass and lower-boundary mass are both zero. The largest upper-boundary mass is 6.43e-8 for CP and 3.16e-9 for dTCMV. Forward and backward first and second moments were checked, including the additional one-step variance induced by mass allocation. The Monte Carlo evaluation produced no paths above the state ceiling and no negative-balance steps.

For the MGH lower 5% mean, probability mass at the boundary grid point is apportioned so that the tail contains exactly 5%. The maximum CDF difference is computed using both left and right limits over the full combined support rather than over the displayed plotting range. Separate small-example tests cover CDFs with atoms and fractional tail allocation.

## Reproduction

```bash
python -m pip install -r requirements.txt
python recalibration/evaluator_comparison.py \
  --data-dir results/current \
  --out results/validation/evaluator_comparison
python recalibration/plot_evaluator_comparison.py \
  --data-dir results/current \
  --validation-dir results/validation/evaluator_comparison
```

See [`recalibration/EVALUATOR_COMPARISON.md`](../../../recalibration/EVALUATOR_COMPARISON.md) for the detailed input mapping, initialisation, statistical definitions, and output inventory.
