# Numerical checks for the adopted recalibration

## Backward / forward moment consistency

Maximum initial mean residuals are approximately `2.9e-12`; second-moment residuals are approximately `2.3e-10`. The forward distributions therefore reproduce the backward moments to numerical precision.

## Baseline state-grid truncation (`x_max = 300`, `n_x = 151`)

For dTCMV:

- terminal mean: `70.342401`
- standard deviation: `22.857705`
- q05: `39.135790`
- q50: `67.412194`
- q95: `111.931229`
- average glide: `0.283891`
- terminal upper-boundary mass: `3.597578e-07`

## `x_max` sensitivity

With spacing preserved while expanding the grid:

- `x_max=300`: mean `70.342401`, stdev `22.857705`, average glide `0.283891`
- `x_max=450`: mean `70.342513`, stdev `22.858473`, average glide `0.283890`
- `x_max=600`: mean `70.342513`, stdev `22.858474`, average glide `0.283890`
- `x_max=900`: mean `70.342513`, stdev `22.858474`, average glide `0.283890`

The baseline result is therefore not materially driven by the upper state-grid boundary.

Source: successful parameter-only original-method Actions run `31868267203`.
