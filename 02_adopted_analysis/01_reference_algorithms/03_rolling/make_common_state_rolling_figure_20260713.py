from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'results' / 'rolling_quantile_detail_D0_N480.csv'
OUT = ROOT / 'figs' / 'fig_common_state_rolling_selected_D0_N480.png'
TABLE = ROOT / 'results' / 'selected_common_state_rolling_D0_N480.csv'

df = pd.read_csv(CSV)
selected = [
    (30, 'q50', 'Year 30, median state'),
    (30, 'q90', 'Year 30, high state'),
    (35, 'q10', 'Year 35, low state'),
]

rows = []
for year, state, label in selected:
    part = df[(df['year'] == year) & (df['state'] == state)].copy()
    for strategy in ['cTCMV', 'dTCMV']:
        r = part[part['policy'] == strategy].iloc[0]
        rows.append({
            'state_label': label,
            'year': year,
            'state': state,
            'starting_x': r['starting_x'],
            'strategy': strategy,
            'mean': r['mean'],
            'stdev': r['stdev'],
            'median': r['q50'],
            'q05': r['q05'],
            'cvar05': r['cvar05'],
            'q95': r['q95'],
            'skewness': r['skewness'],
        })
sel = pd.DataFrame(rows)
sel.to_csv(TABLE, index=False)

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharex=True)
colors = {'cTCMV': '#4c9f45', 'dTCMV': '#f28e2b'}
ypos = {'cTCMV': 1, 'dTCMV': 0}

for ax, (year, state, label) in zip(axes, selected):
    part = sel[(sel['year'] == year) & (sel['state'] == state)]
    xstate = float(part['starting_x'].iloc[0])
    for strategy in ['cTCMV', 'dTCMV']:
        r = part[part['strategy'] == strategy].iloc[0]
        y = ypos[strategy]
        ax.hlines(y, r['q05'], r['q95'], color=colors[strategy], linewidth=2.3)
        ax.vlines([r['q05'], r['q95']], y - 0.10, y + 0.10, color=colors[strategy], linewidth=1.3)
        ax.scatter(r['mean'], y, s=34, facecolor='white', edgecolor='black', zorder=3)
        ax.text(r['mean'], y + 0.18, f"{r['mean']:.1f}", ha='center', va='bottom', fontsize=8)
        ax.text(r['q05'], y - 0.18, f"{r['q05']:.1f}", ha='center', va='top', fontsize=7)
        ax.text(r['q95'], y - 0.18, f"{r['q95']:.1f}", ha='center', va='top', fontsize=7)
    ax.set_yticks([1, 0], ['cTCMV', 'dTCMV'])
    ax.set_title(f"{label}\n($x={xstate:.2f}$)", fontsize=9.5)
    ax.set_xlabel('Conditional terminal wealth')
    ax.grid(axis='x', alpha=0.25)
    ax.set_ylim(-0.55, 1.55)

xmin = min(sel['q05']) - 5
xmax = max(sel['q95']) + 8
for ax in axes:
    ax.set_xlim(xmin, xmax)

fig.suptitle('Common-state rolling comparison: q05-q95 interval and mean', fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT, dpi=220, bbox_inches='tight')
plt.close(fig)
print(OUT)
print(TABLE)
