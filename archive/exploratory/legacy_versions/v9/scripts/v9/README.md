> v9作成時の記録です。現在の保存場所と最新版はリポジトリのREADMEを参照してください。

# v9追加分析・作図

リポジトリのルートから実行します。

```text
python -m pip install -r requirements.txt
python scripts/v9/v9_analysis.py
python scripts/v9/v9_targets.py
python scripts/v9/v9_legacy_figures.py
```

入力は`results/recalibration_v8/fine/`、集計出力は`results/v9/`、図は`paper/v9/figs/`です。方策の再最適化は行いません。最初のコマンドは3つの共通状態を各20万経路で評価します。

日本語フォントはWindowsのMeiryo、LinuxのNoto Sans CJK、`fonts/NotoSansCJKjp-Regular.otf`から選びます。別のフォントは環境変数`ICA_JAPANESE_FONT`にファイルパスを設定してください。

旧監査図だけは`archive/v8/paper/ICA2026_Japanese_revised_v8.tex`に記録された座標を読みます。新方策のデータとは分離しています。再生成はTeX内の数表を自動変更しません。

完全な再最適化は[recalibration/README.md](../../recalibration/README.md)を参照してください。
