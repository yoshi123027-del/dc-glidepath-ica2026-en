from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics"
p = OUT / "tc_diagnostics_arrays.npz"
z = np.load(p)
x = z["xg"]
H = z["H"]
Mc = z["Mc"]
Qc = z["Qc"]
Pc = z["Pc"]
Md = z["Md"]
Qd = z["Qd"]
Pd = z["Pd"]
pmfc = z["pmfc"]
pmfd = z["pmfd"]
beta = 0.04
sigma = 0.18
gc = 0.059638671875
g0 = 1.193359375


def spline_derivs(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.empty_like(array)
    d2 = np.empty_like(array)
    for n in range(array.shape[0]):
        spline = CubicSpline(x, array[n], bc_type="natural")
        d1[n] = spline(x, 1)
        d2[n] = spline(x, 2)
    return d1, d2


def classify(fraction: np.ndarray, tol: float = 0.02) -> np.ndarray:
    out = np.ones_like(fraction, dtype=np.int8)
    out[fraction <= tol] = 0
    out[fraction >= 1.0 - tol] = 2
    return out


Fcx, Fcxx = spline_derivs(Mc)
Vc = Mc - 0.5 * gc * (Qc - Mc**2)
Vcx, Vcxx = spline_derivs(Vc)
Dc = Vcxx - gc * Fcx**2
uc = np.full_like(Pc, np.nan)
maskc = Dc[:-1] < -1e-7
uc[maskc] = -beta * Vcx[:-1][maskc] / (sigma**2 * Dc[:-1][maskc])
fc = np.clip(uc / np.maximum(x[None, :], 1e-12), 0.0, 1.0)
fc[:, 0] = 0.0

Fdx, Fdxx = spline_derivs(Md)
Gdx, Gdxx = spline_derivs(Qd)
Gamma = g0 / np.maximum(x[None, :] + H[:, None], 1e-12)
Ad = (1.0 + Gamma * Md) * Fdx - 0.5 * Gamma * Gdx
Bd = (1.0 + Gamma * Md) * Fdxx - 0.5 * Gamma * Gdxx
ud = np.full_like(Pd, np.nan)
maskd = Bd[:-1] < -1e-7
ud[maskd] = -beta * Ad[:-1][maskd] / (sigma**2 * Bd[:-1][maskd])
fd = np.clip(ud / np.maximum(x[None, :], 1e-12), 0.0, 1.0)
fd[:, 0] = 0.0

rows = []
for name, policy, pmf, predicted, valid_mask in [
    ("cTCMV", Pc, pmfc, fc, maskc),
    ("dTCMV", Pd, pmfd, fd, maskd),
]:
    valid = valid_mask & np.isfinite(predicted)
    weights = np.where(valid, pmf[:-1], 0.0)
    denominator = weights.sum()
    difference = np.where(valid, np.abs(policy - predicted), 0.0)
    region_agreement = classify(policy) == classify(np.nan_to_num(predicted, nan=-9.0))
    agreement = (weights * region_agreement).sum() / denominator
    mae = (weights * difference).sum() / denominator

    away = (
        valid
        & (np.minimum(predicted, 1.0 - predicted) > 0.05)
        & (np.minimum(policy, 1.0 - policy) > 0.05)
    )
    away_weights = np.where(away, pmf[:-1], 0.0)
    away_denominator = away_weights.sum()
    interior_agreement = (
        (away_weights * region_agreement).sum() / away_denominator
        if away_denominator > 0.0
        else np.nan
    )

    values = difference[valid].ravel()
    quantile_weights = pmf[:-1][valid].ravel()
    order = np.argsort(values)
    values = values[order]
    quantile_weights = quantile_weights[order]
    cdf = np.cumsum(quantile_weights) / quantile_weights.sum()
    q95 = values[np.searchsorted(cdf, 0.95)]

    rows.append(
        [
            name,
            agreement,
            mae,
            interior_agreement,
            q95,
            1.0 - denominator / pmf[:-1].sum(),
        ]
    )

frame = pd.DataFrame(
    rows,
    columns=[
        "strategy",
        "prob_weighted_region_agreement",
        "prob_weighted_policy_mae",
        "robust_interior_region_agreement",
        "q95_difference",
        "derivative_invalid_mass_share",
    ],
)
frame.to_csv(OUT / "free_boundary_crosscheck.csv", index=False)
print(frame.to_string(index=False))
