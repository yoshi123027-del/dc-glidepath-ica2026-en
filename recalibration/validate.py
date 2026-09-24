"""Independent off-grid normal Euler Monte Carlo and numerical diagnostics."""
import argparse,json,csv,math,time
from pathlib import Path
import finite_model as fm
import numpy as np
from numba import njit,prange,set_num_threads

NAMES=['PCMV','DOMV','cTCMV','dTCMV','CP']

@njit(cache=True,parallel=True)
def mc(P,h,paths,seed,blocks=100):
    J,N,nx=P.shape;dt=40/N;bs=paths//blocks
    terminal=np.empty((J,paths));glide=np.zeros((blocks,J,N));hit=np.zeros((blocks,J));neg=np.zeros((blocks,J));state=np.zeros((blocks,J,3))
    for block in prange(blocks):
        np.random.seed(seed+block*1009)
        for k in range(bs):
            x=np.full(J,1/12);ever=np.zeros(J,np.bool_)
            for n in range(N):
                z=np.random.normal()
                for j in range(J):
                    xx=x[j]
                    # Independent implementation: risky DOLLARS interpolation,
                    # no state mass deposition, no wealth cap. Endpoint fraction
                    # continuation above grid is constrained to [0,1].
                    f=xx/h;i=int(f);l=f-i
                    if i>=nx-1:p=P[j,n,nx-1]*xx/((nx-1)*h)
                    else:p=(1-l)*P[j,n,i]+l*P[j,n,i+1]
                    p=min(max(p,0.),xx)
                    if xx>0:glide[block,j,n]+=p/xx/bs
                    xx+= (.015*xx+1+.04*p)*dt+.18*p*math.sqrt(dt)*z
                    if xx<0:neg[block,j]+=1;xx=0.
                    if xx>(nx-1)*h:ever[j]=True
                    x[j]=xx
            for j in range(J):terminal[j,block*bs+k]=x[j];hit[block,j]+=ever[j]
    return terminal,glide,hit,neg

def main():
    global NAMES
    ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--paths',type=int,default=1000000);ap.add_argument('--seed',type=int,default=20260912);ap.add_argument('--names',default=','.join(NAMES));a=ap.parse_args()
    NAMES=a.names.split(',')
    if a.paths<=0 or a.paths%100:
        raise ValueError('paths must be a positive multiple of 100')
    out=Path(a.directory);zs=[np.load(out/(n+'.npz')) for n in NAMES];P=np.stack([z['policy'] for z in zs]);h=float(zs[0]['xmax'])/(int(zs[0]['nx'])-1)
    t=time.time();X,g,hit,neg=mc(P,h,a.paths,a.seed)
    g=g.mean(axis=0);hit=hit.sum(axis=0);neg=neg.sum(axis=0)
    rows=[]
    for j,n in enumerate(NAMES):
        v=X[j];m=float(v.mean());sd=float(v.std(ddof=1));q=np.quantile(v,[.05,.5,.95]);lc=np.sort(v)[:a.paths//20].mean()
        rows.append(dict(strategy=n,paths=a.paths,seed=a.seed,mean=m,mean_se=sd/math.sqrt(a.paths),sd=sd,q05=q[0],median=q[1],q95=q[2],lcvar05=lc,shortfall60=float(np.mean(v<60)),expected_shortfall60=float(np.maximum(60-v,0).mean()),ever_above_cap=hit[j]/a.paths,negative_steps=int(neg[j])))
    paired=[]
    for j in range(len(NAMES)):
        for k in range(j+1,len(NAMES)):
            diff=X[j]-X[k];m=diff.mean();se=diff.std(ddof=1)/math.sqrt(a.paths)
            paired.append(dict(first=NAMES[j],second=NAMES[k],mean_difference=m,se=se,ci_low=m-1.96*se,ci_high=m+1.96*se))
    for filename,data in [('independent_mc.csv',rows),('paired_means.csv',paired)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    np.savez_compressed(out/'independent_mc.npz',terminal=X,glide=g,names=NAMES,seed=a.seed)
    print(json.dumps(rows,indent=2),flush=True);print('seconds',time.time()-t)

if __name__=='__main__':main()
