# Paper

[`current/`](current/) はICA2026論文の参照版です。PDF、TeXソース、およびTeXから参照する図を同じ場所に収録しています。

- [Current Japanese paper (PDF)](current/ICA2026_Japanese_revised_v15.pdf)
- [TeX source](current/ICA2026_Japanese_revised_v15.tex)
- [Figure files](current/figs/)

## Typesetting

XeLaTeX、Latin Modern、Harano Ajiフォント、`xeCJK`を使用します。数値計算の再実行は不要です。

```bash
cd paper/current
xelatex -interaction=nonstopmode -halt-on-error ICA2026_Japanese_revised_v15.tex
xelatex -interaction=nonstopmode -halt-on-error ICA2026_Japanese_revised_v15.tex
```
