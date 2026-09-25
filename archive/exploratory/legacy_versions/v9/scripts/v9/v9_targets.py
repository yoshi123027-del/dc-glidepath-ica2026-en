from pathlib import Path
import sys,csv
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'pydeps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from figure_style import apply_style
apply_style(B)
O=B;z=np.load(B.parents[1]/'results/recalibration_v8/fine/independent_mc.npz');X=z['terminal'];names=['PCMV','DOMV','cTCMV','dTCMV','CP']
fig,axes=plt.subplots(1,2,figsize=(11.2,4.6));rows=[]
for j,n in enumerate(names):
    x=X[j];sx=np.sort(x);cs=np.r_[0,np.cumsum(sx)];thresholds=np.linspace(20,130,441);idx=np.searchsorted(sx,thresholds,side='left');p=idx/len(x);e=(idx*thresholds-cs[idx])/len(x)
    axes[0].plot(thresholds,p,label=n,lw=1.8);axes[1].plot(thresholds,e,label=n,lw=1.8)
    for k in [40,60,80,100]:
        d=np.maximum(k-x,0);prob=(x<k).mean()
        rows.append(dict(target=k,strategy=n,prob=prob,prob_se=np.sqrt(prob*(1-prob)/len(x)),expected_shortfall=d.mean(),es_se=d.std(ddof=1)/np.sqrt(len(x))))
for ax in axes:ax.set_xlabel('目標DC富');ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=9)
axes[0].set_ylabel('目標不足確率');axes[1].set_ylabel('期待不足額');fig.tight_layout();fig.savefig(O/'paper/v9/figs/target_sensitivity_v9.pdf');fig.savefig(O/'paper/v9/figs/target_sensitivity_v9.png',dpi=200)
with (O/'results/v9/target_sensitivity_v9.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
text=r'''
\section{退職目標を変えたときの不足分析}
\label{app:targets}
\subsection{一つの閾値に依存しない成果説明}
本文では目標60を用いたが，加入者が必要とする所得水準は一つではない．同じ方策・同じ終価標本を固定し，目標だけを変えると，不足の頻度と深さがどう変わるかを評価できる．ここでは20から130の範囲を描き，40，60，80，100を詳細表に示す．各閾値へ期待終価を再較正した結果ではない．

この分析の目的は，一つの指標の順位を普遍的な優劣に置き換えないことである．目標を高くすると不足確率が高まること自体は自然だが，その増加の仕方は分布形状に依存する．PCMVのターゲット近傍の圧縮，定率運用の右裾，均衡方策の状態依存性は，不足曲線の異なる領域へ現れる．

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{target_sensitivity_v9.pdf}
\caption{新較正5戦略の目標不足確率と期待不足額．同じ100万経路の標本で目標のみを変更．}\label{fig:targets}
\end{figure}

\subsection{確率と深さの関係}
非負終価$W$について，期待不足額は分布関数の積分として
\begin{equation}
S(b)=\E[(b-W)_+]=\int_0^b F_W(u)\dd u
\end{equation}
と表せる．したがって，連続点では$S'(b)=F_W(b)$であり，不足確率は不足額曲線の傾きを表す．ただし，ある一点で不足確率が小さいことだけでは，そこまで積分した期待不足額も小さいとは限らない．この関係が二つの指標を併記する理由である．

図\ref{fig:targets}の不足確率曲線が交差する場合，ある目標では有利な方策が，別の目標では不利となる．期待不足額でも同じ順位になるとは限らない．全目標での優位を論じるには，点の比較より強い分布の順序が必要であり，本稿は単一目標の比較から確率優越を主張しない．

\begin{table}[htbp]\centering\small
\caption{目標別の不足確率と期待不足額（新較正方策）}\label{tab:targets}
\begin{tabular}{rlrrrr}\toprule
目標 & 方策 & 不足確率 & 確率SE & 期待不足額 & 不足額SE\\\midrule
'''
for i,r in enumerate(rows):
    pass
for k in [40,60,80,100]:
    for r in [r for r in rows if r['target']==k]:
        text+=f"{k} & {r['strategy']} & {r['prob']:.4f} & {r['prob_se']:.5f} & {r['expected_shortfall']:.4f} & {r['es_se']:.5f}"+r' \\'+'\n'
    if k<100:text+=r'\midrule'+'\n'
text+=r'''\bottomrule\end{tabular}\end{table}

\subsection{退職所得への換算と解釈の限界}
年額拠出30万円，現価係数20の説明例では，残高目標40，60，80，100は年額所得60，90，120，150万円に対応する．既存のDB給付を別に受け取る場合は，DCに必要な所得部分へ目標を割り当てる．ただし，本計算の終端給付は$D_T=0$であり，実在する加入者の全老後所得を推定した表ではない．

目標を達成する確率を高める手段には，投資方策だけでなく，拠出額，拠出期間，退職時期，支出目標の変更がある．これらを同時に変えると別の問題となるため，本図では投資方策以外を固定した．特に退職直前の低残高状態で，配分変更だけによって不足を解消できるとは説明しない．

目標別の表も同じ市場係数を仮定した条件付きの結果である．標本SEはモンテカルロの不確実性を示すが，将来リターンや拠出継続性の不確実性を含めた予測区間ではない．実務では入力仮定の感応度と合わせて読む必要がある．
'''
(B/'results/v9/target_tables.tex').write_text(text,encoding='utf-8')
print('Target sensitivity generated')
