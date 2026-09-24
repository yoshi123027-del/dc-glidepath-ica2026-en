from pathlib import Path
import sys,re,json
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'pydeps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from figure_style import apply_style
apply_style(B)
O=B;src=(B.parents[1]/'archive/v8/paper/ICA2026_Japanese_revised_v8.tex').read_text(encoding='utf-8')
data={}
for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',src,re.S):
    lab=re.search(r'\\label\{([^}]+)\}',block)
    if not lab or lab[1] not in ['fig:cdf','fig:glide']:continue
    series=[]
    for coords,label in re.findall(r'coordinates\s*\{(.*?)\};\\addlegendentry\{([^}]+)\}',block,re.S):
        xy=np.array([(float(a),float(b)) for a,b in re.findall(r'\(([-\d.]+),([-\d.]+)\)',coords)])
        series.append((label,xy));
    data[lab[1]]=[dict(label=l,coordinates=x.tolist()) for l,x in series]
    if lab[1]=='fig:cdf':
        fig,ax=plt.subplots(figsize=(9.2,5.6))
        for l,x in series:ax.plot(x[:,0],x[:,1],lw=1.8,label=l)
        ax.set(xlabel='終端DC富',ylabel='累積確率',xlim=(0,220),ylim=(0,1));ax.grid(alpha=.25);ax.legend(ncol=3)
        name='legacy_cdf'
    else:
        fig,axes=plt.subplots(2,2,figsize=(11.2,7.8),sharex=True,sharey=True)
        for j,ax in enumerate(axes.flat):
            for k in range(2):
                l,x=series[j*2+k];ax.plot(x[:,0],x[:,1],lw=1.8,ls='-' if k==0 else '--',label=l)
            ax.set(title=['PCMV','DOMV','cTCMV','dTCMV'][j],xlabel='加入後経過年数',ylabel='平均リスク資産比率',ylim=(0,1.04))
            ax.grid(alpha=.25);ax.legend(fontsize=9)
        name='legacy_clipped'
    fig.tight_layout();fig.savefig(O/'paper/v9/figs'/(name+'.pdf'));fig.savefig(O/'paper/v9/figs'/(name+'.png'),dpi=180);plt.close(fig)
(O/'results/v9/legacy_plot_coordinates.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
print('Legacy figures reformatted from verified v8 plot coordinates')
