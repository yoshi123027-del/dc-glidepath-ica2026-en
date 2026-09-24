# Additional diagnostics for the ICA2026 paper

This directory reproduces the diagnostics added after the main monthly baseline was generated.

## Table 10: dTCMV U-shaped glide-path decomposition

Run:

```bash
python diagnostics/additional_diagnostics.py
python diagnostics/unconstrained_dtcmv_theta.py
```

The first command recomputes the constrained cTCMV/dTCMV policies, forward distributions and the state-dependent decomposition. The second command solves the unconstrained dTCMV Volterra equation through its equivalent backward ODE and appends `unconstrained_theta` to `dtcmv_u_shape_decomposition.csv`.

## Appendix A.4: free-boundary cross-check

After `additional_diagnostics.py`, run:

```bash
python diagnostics/recompute_crosscheck.py
python diagnostics/pcmv_crosscheck.py
```

The outputs are:

- `free_boundary_crosscheck.csv`: cTCMV and dTCMV probability-weighted region agreement.
- `pcmv_free_boundary_crosscheck.csv`: PCMV probability-weighted region agreement.

The scripts create intermediate NPZ files locally. Those intermediate arrays are not required to inspect the reported CSV results and may be regenerated from the scripts.

The baseline parameters and grids match the paper: 40 years, monthly time steps, 151 non-uniform wealth nodes, 15 control nodes and five-point Gauss-Hermite quadrature.
