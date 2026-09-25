"""Plot saved evaluator-comparison outputs without a new evaluation."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='results/current')
    ap.add_argument('--validation-dir', default='results/validation/evaluator_comparison')
    ap.add_argument('--out')
    a=ap.parse_args()
    d=Path(a.validation_dir)
    o=Path(a.out) if a.out else d
    o.mkdir(parents=True, exist_ok=True)
    z=np.load(d/'evaluator_comparison.npz')
    mc=np.load(Path(a.data_dir)/'independent_mc.npz')
    x=z['grid'];names=z['names'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    for kind in ['cdf','glidepath']:
        fig,axs=plt.subplots(2,3,figsize=(10,5.4),layout='constrained');axs=axs.ravel()
        for j,name in enumerate(names):
            ax=axs[j];ax.set_title(name,fontweight='bold',loc='left');c=colors[j]
            if kind=='cdf':
                p=z['mgh_terminal_mass'][j];F=np.cumsum(p/p.sum());v=np.sort(mc['terminal'][j]);ec=np.arange(1,len(v)+1)/len(v)
                ax.step(x,F,where='post',color=c,lw=1.6,label='MGH');ax.step(v,ec,where='post',color='black',lw=1.1,ls=(0,(2,2)),label='Euler-MC')
                ax.set_xlim(0,200);ax.set_ylim(0,1.02);ax.set_xlabel('Terminal wealth');ax.set_ylabel('CDF')
                if j==0:
                    sub=ax.inset_axes([.55,.16,.42,.39]);sub.step(x,F,where='post',color=c,lw=1.1);sub.step(v,ec,where='post',color='black',ls=':',lw=1.1);sub.set_xlim(98,102);sub.set_ylim(.80,1.005);sub.tick_params(labelsize=6);sub.set_title('Upper CDF detail',fontsize=6)
            else:
                t=np.arange(480)/12;ax.plot(t,z['mgh_glide'][j]*100,color=c,lw=1.6);ax.plot(t,z['mc_glide'][j]*100,color='black',lw=1.1,ls=(0,(2,2)));ax.set_xlim(0,40);ax.set_ylim(0,103);ax.set_xlabel('Years since inception');ax.set_ylabel('Mean risky allocation (%)')
        axs[-1].axis('off');axs[-1].legend(handles=[Line2D([0],[0],color='#0072B2',lw=1.6,label='MGH forward (solid)'),Line2D([0],[0],color='black',ls=(0,(2,2)),lw=1.1,label='Independent Euler-MC (dotted)')],loc='center',frameon=False)
        for ext in ['png','pdf']:
            fig.savefig(o/f'fig_evaluator_{kind}_comparison.{ext}',dpi=220)
        plt.close(fig)


if __name__=='__main__':
    main()
