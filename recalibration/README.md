# Numerical implementation

このディレクトリには、論文で用いる有限モデル、方策の最適化・再較正、分布評価、および検証用コードを収録しています。保存済みの主要結果は [`../results/current/`](../results/current/)、検証結果は [`../results/validation/`](../results/validation/) にあります。

## Code map

| File or folder | Role |
|---|---|
| `finite_model.py` | 共通の有限モデル、MGH遷移核、PCMV・cTCMV・dTCMVの計算基盤 |
| `run.py` | 方策の最適化と共通期待終価への再較正 |
| `validate.py` | 保存済み方策の独立Euler–Monte Carlo評価 |
| `checks.py`, `acceptance.py`, `dom_diagnostics.py` | 後退・前進整合性、受入判定、DOMVの診断 |
| `sensitivity.py` | 格子、GH求積、上限境界に関する頑健性確認 |
| `evaluator_comparison.py`, `plot_evaluator_comparison.py` | 同一方策に対するMGH前進と独立Euler–MCの比較 |
| `external_vanstaden_2021/` | van Staden, Dang and Forsyth (2021) に基づく外部ベンチマーク |

## Results and validation

- [Main policies and reported results](../results/current/)
- [Finite-model checks and robustness results](../results/validation/recalibration/)
- [MGH vs independent Euler–MC](../results/validation/evaluator_comparison/)
- [External benchmark](../results/validation/van_staden_2021/)

Install the dependencies with `python -m pip install -r requirements.txt` from the repository root. The calculations are intentionally separated by script; run only the module needed for a full reproduction. The stored outputs are sufficient for inspecting the reported results.
