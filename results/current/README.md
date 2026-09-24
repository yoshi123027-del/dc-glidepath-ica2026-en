# Main reported results

このディレクトリには、現在の論文の主要結果に直接対応する保存済みデータを収録しています。

| Files | Contents |
|---|---|
| `PCMV`, `DOMV`, `cTCMV`, `dTCMV`, `CP` (`.json`, `.npz`) | 較正パラメータと保存済み方策 |
| `config.json` | 有限モデルの共通設定 |
| `independent_mc.csv` | 論文の主要な終端分布統計量 |
| `independent_mc.npz` | 独立Euler–Monte Carloの終端標本と平均グライドパス |
| `paired_means.csv` | 共通乱数に基づく平均差の補足結果 |

検証・頑健性確認は [`../validation/`](../validation/) を参照してください。
