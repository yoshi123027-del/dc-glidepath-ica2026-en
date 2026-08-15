# 全解概念の厳密制約解とクリップ近似

このディレクトリには、年80分割の感応度分析について、PCMV、DOMV、cTCMV、dTCMVの厳密制約解と無制約クリップ近似を比較した結果を保存します。

## ファイル

- `all_strategies_strict_vs_clip_sensitivity_summary.csv`：シナリオ・解概念ごとの平均絶対差、最大絶対差、最大差が生じる時点（リポジトリ収録）
- `all_strategies_strict_vs_clip_sensitivity_paths.csv`：全決定時点の厳密解、クリップ解および差分（実行時生成）
- `all_strategies_strict_vs_clip_sensitivity_glidepaths.npz`：シナリオ、解概念、決定時点を配列として保存した圧縮データ（実行時生成）

## 比較方法

厳密制約解は、各時点・各残高で `0 <= pi <= x` を直接課して後退計算します。クリップ近似は、各解概念に対応する無制約解析解を同区間へ事後射影します。PCMV、DOMV、cTCMV、dTCMVで同じ無制約式を使用しているわけではありません。

両方策はそれぞれ独自に前進伝播し、それぞれが生成する残高分布で質量加重したリスク資産比率を比較しています。したがって、CSVの差分は共通の外生状態経路上で方策だけを比較した値ではありません。

## 再生成

```bash
python scripts/04_sensitivity/add_all_clip_overlays_20260721.py --output-dir d0_sensitivity_outputs
```

感応度図は `supplementary/figures/` に掲載しています。
