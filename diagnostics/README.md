# Additional Diagnostics for the ICA2026 Paper

This directory reproduces numerical diagnostics added after the main monthly baseline was generated.

## Table 10: dTCMV U-shaped glide-path decomposition

Run:

```bash
python diagnostics/additional_diagnostics.py
python diagnostics/unconstrained_dtcmv_theta.py
```

The first command recomputes the constrained cTCMV and dTCMV policies, their forward distributions, and the state-dependent decomposition. The second command solves the unconstrained dTCMV Volterra equation through its equivalent backward ODE and appends `unconstrained_theta` to `dtcmv_u_shape_decomposition.csv`.

## Supplementary free-boundary cross-check

The final paper omits the longer free-boundary diagnostic table because of the page limit, but the numerical cross-check remains available in this repository. After running `additional_diagnostics.py`, execute:

```bash
python diagnostics/recompute_crosscheck.py
python diagnostics/pcmv_crosscheck.py
```

The outputs are:

- `free_boundary_crosscheck.csv`: probability-weighted agreement between the analytical projection regions and the directly searched regions for cTCMV and dTCMV.
- `pcmv_free_boundary_crosscheck.csv`: the corresponding probability-weighted agreement for PCMV.

The scripts create intermediate NPZ files locally. Those arrays are not required to inspect the reported CSV results and can be regenerated from the scripts.

## Numerical specification

The baseline parameters and grids match the paper:

- 40-year horizon;
- monthly time steps;
- 151 non-uniform wealth nodes;
- 15 control nodes; and
- five-point Gauss–Hermite quadrature.

The cross-check is a diagnostic of consistency between the theoretical region classification and the discrete optimiser. It is not a substitute for a general convergence proof.
