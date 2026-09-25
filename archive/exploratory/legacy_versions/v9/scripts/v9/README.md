> This is a record from the preparation of v9. See the repository README for the current layout and reference version.

# Additional v9 Analyses and Figures

The commands below are the historical v9 commands. Run them from the archived v9 root (`archive/exploratory/legacy_versions/v9`) only after adapting the input path described below.

```text
python -m pip install -r requirements.txt
python scripts/v9/v9_analysis.py
python scripts/v9/v9_targets.py
python scripts/v9/v9_legacy_figures.py
```

The original inputs were in `results/recalibration_v8/fine/`; summary outputs were written to `results/v9/`, and figures to `paper/v9/figs/`. The policies were not re-optimised. The first command evaluated three common states using 200,000 paths per state. The original input directory is not part of the current repository layout, so these scripts are retained for auditability rather than as the current reproduction route.

For the archived Japanese-labelled figures, the scripts select Meiryo on Windows, Noto Sans CJK on Linux, or `fonts/NotoSansCJKjp-Regular.otf`. Set the `ICA_JAPANESE_FONT` environment variable to use another font file.

The legacy audit figures alone read coordinates recorded in the archived v8 TeX source. Those coordinates are separate from the newer policy data. Regenerating the figures does not modify the numerical tables in the TeX source.

For the current optimisation and validation workflow, see the top-level [recalibration guide](../../../../../../recalibration/README.md).
