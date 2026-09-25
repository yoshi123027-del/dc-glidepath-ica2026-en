# Pythonコード案内

Pythonコードは役割ごとに番号付きフォルダへ整理しています。番号は厳密な依存順ではなく、再現作業を読むときの大まかな順序です。コマンドはリポジトリのルートで実行してください。

## まず試す場合

同梱済みのCSV/NPZから論文図を再生成します。完全な月次最適化より短時間で確認できます。

```bash
python scripts/05_figures/localize_paper_figures_ja_20260717.py
```

## 01_solvers：中核ソルバー

| ファイル | 役割 |
| --- | --- |
| `pcmv_domv_solver_20260713.py` | PCMV・DOMVの制約付きソルバー |
| `dtcmv_mvs_solver_20260713.py` | dTCMV--MVSのモーメント後退計算 |

## 02_calibration：較正

| ファイル | 役割 |
| --- | --- |
| `equal_mean_calibration_20260713.py` | 共通平均比較用の較正値を整理 |
| `run_mvs_refined_calibration.py` | MVS係数の精緻較正 |

## 03_rolling：ローリング評価

| ファイル | 役割 |
| --- | --- |
| `recompute_d0_rolling.py` | 基準ケース、終端分布、ローリング評価を再計算 |
| `detailed_dtcmv_rolling_overlay_20260713.py` | 残高分位点別のローリング評価 |
| `make_common_state_rolling_figure_20260713.py` | 共通状態ローリング比較図を生成 |
| `rebuild_rolling_equal_mean.py` | 共通平均較正のローリング結果を再構築 |

## 04_sensitivity：感応度分析

| ファイル | 役割 |
| --- | --- |
| `low_balance_refined_sensitivity_20260714.py` | 低残高領域を精緻化した厳密制約解の感応度分析 |
| `add_all_clip_overlays_20260721.py` | PCMV・DOMV・cTCMV・dTCMVの全パネルに、厳密制約解（実線）と対応する無制約クリップ近似（同色点線）を重ね、差分CSV/NPZを出力 |
| `add_ctcmv_clip_overlays_20260721.py` | cTCMVだけを対象とした旧補助スクリプト。全解概念の比較には上記スクリプトを使用 |
| `rebuild_baseline_sensitivity.py` | 基準ケースと感応度結果を再集計 |
| `numerical_diagnostics_20260718.py` | モーメント整合性、正規化前質量、境界超過量、dTCMV上端格子感応度を再計算 |

全解概念のクリップ近似を含む感応度図は次で再生成します。

```bash
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
```

各パネルでは、同じ市場・拠出シナリオの色を維持し、実線を厳密制約フィードバック、点線を対応する無制約解析解のクリップ近似として表示します。PCMV、DOMV、cTCMV、dTCMVの各無制約解は同一式を流用せず、それぞれの解概念に対応する式から構成します。両方策はそれぞれ自ら生成する残高分布で前進伝播します。

主な出力は次のとおりです。

- `figs/fig_D_sensitivity_glidepaths_N80.png` / `figs/fig_D_sensitivity_glidepaths_N80.svg`
- `figs/fig_r_sensitivity_glidepaths_N80.png` / `figs/fig_r_sensitivity_glidepaths_N80.svg`
- `figs/fig_mu_sensitivity_glidepaths_N80.png` / `figs/fig_mu_sensitivity_glidepaths_N80.svg`
- `figs/fig_sigma_sensitivity_glidepaths_N80.png` / `figs/fig_sigma_sensitivity_glidepaths_N80.svg`
- `figs/fig_contrib_profile_sensitivity_glidepaths_N80.png` / `figs/fig_contrib_profile_sensitivity_glidepaths_N80.svg`
- `all_strategies_strict_vs_clip_sensitivity_summary.csv`
- `all_strategies_strict_vs_clip_sensitivity_paths.csv`
- `all_strategies_strict_vs_clip_sensitivity_glidepaths.npz`

## 05_figures：出力・作図

| ファイル | 役割 |
| --- | --- |
| `rebuild_all_corrected_glide_outputs.py` | 修正済みグライドパス・分布出力を一括再構築 |
| `localize_paper_figures_ja_20260717.py` | 論文図を日本語で再生成 |

## 90_workers：補助ワーカー

`mvs_worker.py`と`sensitivity_worker.py`は分割実行用です。通常は直接実行しません。

## 主要な再計算順序

```bash
python scripts/01_solvers/pcmv_domv_solver_20260713.py
python scripts/03_rolling/recompute_d0_rolling.py
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
python scripts/04_sensitivity/numerical_diagnostics_20260718.py
python scripts/01_solvers/dtcmv_mvs_solver_20260713.py
python scripts/02_calibration/run_mvs_refined_calibration.py
```

各スクリプトは移動後も、リポジトリ直下の `results/` と `figs/` を参照します。
