# コードブック

## 中核ソルバー

| ファイル | 役割 |
| --- | --- |
| `scripts/01_solvers/pcmv_domv_solver_20260713.py` | PCMV・DOMVの制約付き二次損失問題を解く中核ソルバー |
| `scripts/03_rolling/recompute_d0_rolling.py` | $D_T=0$基準ケースのcTCMV・dTCMV・CP、終端分布、ローリング評価を再計算 |
| `scripts/01_solvers/dtcmv_mvs_solver_20260713.py` | dTCMV--MVSの一次・二次・三次モーメント後退計算 |
| `scripts/02_calibration/run_mvs_refined_calibration.py` | MVS係数の精緻較正を実行 |

## 診断・感応度分析

| ファイル | 役割 |
| --- | --- |
| `scripts/03_rolling/detailed_dtcmv_rolling_overlay_20260713.py` | dTCMVの残高分位点別ローリング評価 |
| `scripts/04_sensitivity/low_balance_refined_sensitivity_20260714.py` | 低残高領域を中心とした局所精緻化・感応度分析 |
| `scripts/04_sensitivity/rebuild_baseline_sensitivity.py` | 基準ケースと感応度結果の再集計 |
| `scripts/04_sensitivity/numerical_diagnostics_20260718.py` | 5.11節のモーメント整合性、正規化前質量、上下端超過量、dTCMV上端格子感応度を再計算 |
| `scripts/05_figures/rebuild_all_corrected_glide_outputs.py` | 修正後のグライドパス・分布出力を一括再構築 |
| `scripts/03_rolling/rebuild_rolling_equal_mean.py` | 共通平均較正のローリング結果を再構築 |
| `scripts/02_calibration/equal_mean_calibration_20260713.py` | 共通平均比較用の較正値を整理 |

## 作図

| ファイル | 役割 |
| --- | --- |
| `scripts/05_figures/localize_paper_figures_ja_20260717.py` | 付属NPZ/CSVから論文図を日本語で再生成 |
| `scripts/03_rolling/make_common_state_rolling_figure_20260713.py` | 共通状態ローリング比較図を生成 |

## 補助プロセス

`scripts/90_workers/mvs_worker.py`と`scripts/90_workers/sensitivity_worker.py`は、並列または分割実行時のワーカーです。単独での実行よりも、呼出元スクリプトから利用することを想定しています。

## 主要データ

| ファイル | 内容 |
| --- | --- |
| `results/monthly_D0_policy_arrays.npz` | 月次基準ケースの格子、方策、前進分布、グライドパス、上限制約率 |
| `results/monthly_baseline_D0_summary.csv` | 全戦略の終端分布・グライドパス要約 |
| `results/equal_mean_calibration.csv` | 共通平均を実現する較正値 |
| `results/rolling_conditional_D0_N480.csv` | 代表時点のローリング条件付き統計量 |
| `results/rolling_validation_D0_N480.csv` | 後退・前進モーメント整合性診断 |
| `results/moment_consistency_D0_N480.csv` | 時点0の一次・二次モーメント整合性 |
| `results/mass_truncation_summary_D0_N480.csv` | 正規化前質量誤差と上下端超過量の戦略別要約 |
| `results/mass_truncation_by_time_D0_N480.csv` | 正規化前質量と上下端超過量の時点別記録 |
| `results/xmax_sensitivity_dtcmv_D0_N480.csv` | dTCMVの入れ子上端格子感応度 |
| `results/xmax900_equal_mean_dtcmv_D0_N480.csv` | 広領域で共通平均へ再較正したdTCMVの要約 |
| `results/rolling_quantile_detail_D0_N480.csv` | 残高分位点別ローリング結果 |
| `results/dtcmv_mvs_arrays.npz` | dTCMV--MVSの方策・分布・グライドパス配列 |

