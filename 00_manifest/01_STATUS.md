# Current adopted status — 2026-08-15

## Canonical Japanese paper

The current adopted manuscript is the **parameter-only / original-method** revision. The compiled paper fixed in the working session is named:

`ICA2026_JP_final_20260815.pdf`

and its TeX source is:

`ICA2026_JP_final_20260815.tex`

The paper contains the final Figure 1 / Figure 4 / Figure 6 presentation decisions recorded below.

## Final calibration

- Common terminal-mean anchor: `70.34483966999646`
- PCMV: `gamma_p = 0.14286739349365235`
- PCMV endogenous target: `m_p,base = 77.32192859847035`
- DOMV: `gamma_d = 0.20613769531249998`
- cTCMV: `gamma_c = 0.12565490722656253`
- dTCMV: `rho_d = 2.5083699226379395`
- CP: `theta_cp = 0.273138427734375`
- MVS Figure 6: `gamma0 = 2.5`, `eta0 in {0, 0.5, 1, 2}`

## Final figure presentation

- Figure 1: CDF on top; terminal-density and q05–q95 panels below.
- Figure 4: y-axis displayed from 0 to 100%; no clipping of the plotted curves.
- Figure 6: MVS glide paths on top; terminal-density and q05–q95 panels below.

## Numerical-method freeze

The adopted analysis uses the original-method algorithms from commit:

`a3b83a590474a4d88642d7ce95cfddda9b60461a`

The later corrected MVS-nesting implementation on pre-reorganization main is retained only for history because the final paper deliberately returned to the original `gamma0 = 2.5` MVS specification.

## Reorganization provenance

- Pre-reorganization main: `aa52f01b138115f315702345374d27246dc027eb`
- Successful final parameter-only Actions run: `31868267203`
- Final parameter-only artifact id: `9242688960`
- Artifact SHA256: `130c69883ff9b793d78b522a72ea6bd086e09c2d843832c7ebf4e8ed7f66101c`
