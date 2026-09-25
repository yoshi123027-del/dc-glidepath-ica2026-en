# ICA2026: DC年金の内生的グライドパス

## About this research

本リポジトリは、確定拠出年金（DC）のグライドパスを、あらかじめ外生的に定めるのではなく、加入者の目的関数と現在の状態から内生的に導く研究の公開資料です。
共通のDC制約・市場・拠出条件の下で、初期期待終価をそろえた終端分布と、途中残高から見た条件付き成果を比較します。
PCMVは加入時点の計画へのコミットメント、DOMVは現在の状態からの再最適化を表します。
cTCMVとdTCMVは時間整合的な均衡方策であり、それぞれ定数および総年金富依存の分散回避係数を用います。
実務上は、cTCMVを標準デフォルト、DOMVを定期見直し、dTCMVを個別化助言の候補として位置付け、PCMVを設計上の比較ベンチマークとして扱います。

## Paper

- [Current Japanese paper (PDF)](paper/current/ICA2026_Japanese_revised_v15.pdf)
- [TeX source](paper/current/ICA2026_Japanese_revised_v15.tex)
- [Paper source and figures](paper/README.md)

## Main results

[`results/current/`](results/current/) には、保存済み方策、較正パラメータ、独立Euler–Monte Carloによる主要な終端分布・平均グライドパスの出力を収録しています。論文の主要表・図を確認するだけであれば、数値計算を再実行する必要はありません。

## Code and reproduction

主要な数値実装は [`recalibration/`](recalibration/) にあります。有限モデル、最適化・再較正、MGH評価、独立Monte Carlo評価、感応度分析、評価器比較、外部ベンチマークの役割と入口は [recalibration README](recalibration/README.md) にまとめています。

## Validation

- [MGH backward / forward consistency](results/validation/recalibration/)：保存済み方策について、後退モーメント、前進分布、確率質量、連続制御の監査を照合します。
- [MGH vs independent Euler–Monte Carlo](results/validation/evaluator_comparison/)：同一の5方策をMGH前進と独立100万経路Euler–MCで評価し、終端分布と平均グライドパスを比較します。
- [Grid and boundary robustness](results/validation/recalibration/)：評価格子、GH求積、上限境界に対する頑健性を記録しています。
- [External benchmark: van Staden, Dang and Forsyth (2021)](results/validation/van_staden_2021/)：共通MGH核を外部の平均–分散問題に適用し、方策と終端分布を比較します。

## Repository structure

| Folder | Purpose |
|---|---|
| [`paper/current/`](paper/current/) | Current paper, TeX source, and figures |
| [`recalibration/`](recalibration/) | Main numerical implementation |
| [`results/current/`](results/current/) | Main reported policies and results |
| [`results/validation/`](results/validation/) | Validation and robustness checks |
| [`archive/`](archive/) | Legacy and exploratory materials |

## Archive

過去の論文版、旧格子の結果、旧コード、探索的な分析資料は [`archive/`](archive/) に保存しています。現在の論文を読む際は、通常この領域を参照する必要はありません。

This repository is released under the [MIT License](LICENSE).
