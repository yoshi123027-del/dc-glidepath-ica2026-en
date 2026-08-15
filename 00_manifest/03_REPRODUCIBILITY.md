# Reproducibility guide

## Principle

The final Japanese-paper analysis deliberately keeps the original numerical algorithms fixed and changes only the calibration parameters. The reference algorithm tree is copied verbatim from commit:

`a3b83a590474a4d88642d7ce95cfddda9b60461a`

under `02_adopted_analysis/01_reference_algorithms/`.

## Parameter changes

See `02_adopted_analysis/02_final_parameters/final_parameters.json`.

The important values are:

- `gamma_p = 0.14286739349365235`
- `gamma_d = 0.20613769531249998`
- `gamma_c = 0.12565490722656253`
- `rho_d = 2.5083699226379395`
- `theta_cp = 0.273138427734375`
- common target mean `70.34483966999646`
- MVS Figure 6 uses `gamma0 = 2.5` and `eta0 = 0, 0.5, 1, 2`

## Successfully tested computation

The parameter-only original-method pipeline completed successfully in GitHub Actions run `31868267203`. Its uploaded artifact was `parameter-only-original-method-final` (artifact id `9242688960`, SHA256 `130c69883ff9b793d78b522a72ea6bd086e09c2d843832c7ebf4e8ed7f66101c`).

The tested workflows are retained under `.github/workflows/` in numbered order.

## Verification rule

A new run is accepted only after its baseline and MVS summaries agree with the compact reference tables in `03_adopted_results/` to numerical tolerance. The paper figures should then be generated from those rebuilt outputs.

## Historical / rejected work

`90_rejected/` contains snapshots of the pre-reorganization figures, results, scripts and experiments. These remain available for audit but are not the source for the current paper.
