from pathlib import Path
import sys,json,csv,math
B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B/'pydeps'))
import numpy as np
from numba import njit,prange,set_num_threads
set_num_threads(4)
O=B;D=O/'results/v9';F=O/'paper/v9/figs'
D.mkdir(parents=True,exist_ok=True);F.mkdir(exist_ok=True)
NAMES=['PCMV','DOMV','cTCMV','dTCMV','CP']
SRC=B.parents[1]/'results/recalibration_v8/fine'
P=np.stack([np.load(SRC/(n+'.npz'))['policy'] for n in NAMES])

@njit(cache=True,parallel=True)
def rolling(P,n0,x0,paths,seed):
    J,N,nx=P.shape;blocks=40;bs=paths//blocks
    out=np.empty((J,paths));bad=np.zeros((blocks,2),np.int64)
    for block in prange(blocks):
        np.random.seed(seed+1009*block)
        for k in range(bs):
            x=np.full(J,x0)
            for n in range(n0,N):
                z=np.random.normal()
                for j in range(J):
                    xx=x[j];f=xx/.1;i=int(f);w=f-i
                    if i>=nx-1:p=P[j,n,-1]*xx/600
                    else:p=P[j,n,i]*(1-w)+P[j,n,i+1]*w
                    p=min(max(p,0.),xx)
                    xx+=(.015*xx+1+.04*p)/12+.18*p/math.sqrt(12)*z
                    if xx<0:bad[block,0]+=1;xx=0.
                    if xx>600:bad[block,1]+=1
                    x[j]=xx
            out[:,block*bs+k]=x
    return out,bad

def savecsv(name,rows):
    with (D/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rows=[];pair=[]
states=[(20,20.),(30,45.),(35,30.)]
for s,(year,x0) in enumerate(states):
    X,bad=rolling(P,int(year*12),x0,200000,20260915+s*100000)
    for j,name in enumerate(NAMES):
        x=X[j];sd=x.std(ddof=1);q=np.quantile(x,[.05,.5,.95]);i=int(x0/.1)
        rows.append(dict(year=year,x=x0,strategy=name,paths=len(x),seed=20260915+s*100000,mean=x.mean(),se=sd/math.sqrt(len(x)),sd=sd,q05=q[0],median=q[1],q95=q[2],lcvar=np.sort(x)[:len(x)//20].mean(),shortfall60=(x<60).mean(),es60=np.maximum(60-x,0).mean(),fraction=P[j,year*12,i]/x0))
    diff=X[3]-X[2];se=diff.std(ddof=1)/math.sqrt(len(diff))
    pair.append(dict(year=year,x=x0,contrast='dTCMV-cTCMV',mean=diff.mean(),se=se,low=diff.mean()-1.96*se,high=diff.mean()+1.96*se,negative_steps=int(bad[:,0].sum()),above600_steps=int(bad[:,1].sum())))
    print('rolling',year,x0,rows[-2],flush=True)
savecsv('rolling_v9.csv',rows);savecsv('rolling_pairs_v9.csv',pair)

# Marginal approximate intervals for tails: independent equal blocks,
# estimates based on the full sample; SE of block estimates (not a claim
# of simultaneous rank significance). Preserve the original v8 policies.
z=np.load(B.parents[1]/'results/recalibration_v8/fine/independent_mc.npz');X=z['terminal'];g=z['glide']
tails=[]
for j,name in enumerate(NAMES):
    v=X[j];blocks=v.reshape(100,-1);q=np.quantile(blocks,[.05,.95],axis=1)
    lc=np.sort(blocks,axis=1)[:,:500].mean(axis=1)
    for metric,estimate,vals in [('q05',np.quantile(v,.05),q[0]),('q95',np.quantile(v,.95),q[1]),('lcvar',np.sort(v)[:50000].mean(),lc)]:
        se=vals.std(ddof=1)/10
        tails.append(dict(strategy=name,metric=metric,estimate=estimate,se=se,low=estimate-1.984*se,high=estimate+1.984*se,block_mean=vals.mean(),blocks=100,block_size=10000))
savecsv('tail_intervals_v9.csv',tails)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from figure_style import apply_style
apply_style(B)
colors=plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
def finish(fig,name):
    fig.tight_layout();fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.2,5.6))
for j,n in enumerate(NAMES):ax.plot(np.arange(480)/12,g[j],label=n,lw=1.8,color=colors[j])
ax.set(xlabel='加入後経過年数',ylabel='確率加重リスク資産比率',ylim=(0,1.04));ax.grid(alpha=.25);ax.legend(ncol=3);finish(fig,'glidepaths_v9')
fig,ax=plt.subplots(figsize=(9.2,5.6))
for j,n in enumerate(NAMES):
    pr=np.linspace(0,1,2001);ax.plot(np.quantile(X[j],pr),pr,label=n,lw=1.8,color=colors[j])
ax.set(xlabel='終端DC富',ylabel='累積確率',xlim=(0,220),ylim=(0,1));ax.grid(alpha=.25);ax.legend(ncol=3);finish(fig,'terminal_cdf_v9')
for j in [2,3]:
    fig,ax=plt.subplots(figsize=(9.2,5.6));grid=np.arange(1,1601)*.1
    mesh=ax.pcolormesh(np.arange(480)/12,grid,(P[j,:,1:1601]/grid).T,shading='auto',vmin=0,vmax=1,cmap='viridis',rasterized=True)
    ax.set(xlabel='加入後経過年数',ylabel='DC残高',title=NAMES[j]+'：状態別リスク資産比率（固定残高範囲）',ylim=(0,160))
    fig.colorbar(mesh,ax=ax,label='リスク資産比率');finish(fig,'heatmap_'+NAMES[j]+'_v9')
fig,axes=plt.subplots(1,3,figsize=(11.2,3.9))
for ax,(year,x0) in zip(axes,states):
    for n,y,c in [('cTCMV',1,'#4c9f45'),('dTCMV',0,'#f28e2b')]:
        r=next(r for r in rows if r['year']==year and r['strategy']==n)
        ax.hlines(y,r['q05'],r['q95'],color=c,lw=2.3);ax.vlines([r['q05'],r['q95']],y-.1,y+.1,color=c,lw=1.3)
        ax.scatter(r['mean'],y,s=34,facecolor='white',edgecolor='black',zorder=3)
        ax.text(r['mean'],y+.18,f"{r['mean']:.1f}",ha='center',fontsize=8)
        for key in ['q05','q95']:ax.text(r[key],y-.25,f"{r[key]:.1f}",ha='center',fontsize=7)
    ax.set_yticks([1,0],['cTCMV','dTCMV']);ax.set_title(f'{year}年・共通残高 {x0:.0f}',fontsize=10)
    ax.set(xlabel='条件付き終端富',ylim=(-.55,1.55));ax.grid(axis='x',alpha=.25)
fig.suptitle('共通状態ローリング比較：q05–q95区間と平均',fontsize=11.5);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(F/'rolling_v9.pdf');fig.savefig(F/'rolling_v9.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.2,5.6))
for j,n in enumerate(NAMES):
    rr=[r for r in tails if r['strategy']==n and r['metric']=='lcvar'][0]
    ax.errorbar(rr['estimate'],j,xerr=1.984*rr['se'],fmt='o',color=colors[j],capsize=4)
ax.set_yticks(range(5),NAMES);ax.set(xlabel='下位5%平均と近似95%区間',ylabel='戦略');ax.grid(axis='x',alpha=.25);finish(fig,'tail_intervals_v9')
(D/'analysis_metadata.json').write_text(json.dumps(dict(date='2026-09-15',source_commit='028a9ff9d4a41adb10495b2d05ff7191ac971121',states=states,rolling_paths=200000,rolling_blocks=40,policy='v8 fine unchanged',h=.1,dt=1/12,tail_method='100 independent blocks; marginal approximate 95% intervals; no multiplicity correction',figure_style_source='scripts/05_figures/localize_paper_figures_ja_20260717.py',heatmap='full fixed wealth range; no reachable-probability envelope'),indent=2),encoding='utf-8')
print('Analysis and figures complete',flush=True)
