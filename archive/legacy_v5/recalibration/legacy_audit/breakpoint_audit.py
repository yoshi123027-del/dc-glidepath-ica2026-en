from pathlib import Path
import numpy as np,csv,json
BASE=Path(__file__).resolve().parent;R=BASE.parents[1];O=BASE/'audit_results'
z=np.load(R/'results/monthly_D0_policy_arrays.npz');x=z['xg_tc'];N=480;h=1/12;r=.015;be=.04;sig=.18
xx,ww=np.polynomial.hermite.hermgauss(5);xx*=np.sqrt(2);ww/=np.sqrt(np.pi)
H=np.zeros(N+1)
for n in range(N-1,-1,-1):H[n]=(h+H[n+1])/(1+r*h)
rows=[]
for name,key,gamma in [('cTCMV','ctcmv',.059638671875),('dTCMV','dtcmv',1.193359375)]:
    P=z[key+'_policy'];pmf=z[key+'_pmf'];M=x.copy();Q=x*x;mx=0;weighted=0;positive=0;worst=None
    for n in range(N-1,-1,-1):
        newM=np.empty_like(x);newQ=np.empty_like(x)
        for i,xi in enumerate(x):
            g=gamma if name=='cTCMV' else gamma/(xi+H[n])
            base=xi+(r*xi+1)*h;slopes=xi*(be*h+sig*np.sqrt(h)*xx)
            bp=(x[:,None]-base)/np.where(abs(slopes)>1e-20,slopes,np.nan)
            candidates=np.unique(np.r_[0.,1.,P[n,i],bp[(bp>=0)&(bp<=1)]])
            ys=base+candidates[:,None]*slopes
            m=np.interp(ys,x,M)@ww;q=np.interp(ys,x,Q)@ww
            obj=m-.5*g*(q-m*m)
            saved_idx=np.argmin(abs(candidates-P[n,i]));k=int(obj.argmax())
            defect=max(0.,obj[k]-obj[saved_idx]);weighted+=pmf[n,i]*defect/N
            positive+=pmf[n,i]*(defect>1e-8)/N
            if defect>mx:mx=defect;worst=(n,xi,P[n,i],candidates[k])
            newM[i]=m[saved_idx];newQ[i]=q[saved_idx]
        M,Q=newM,newQ
        if n%120==0:print(name,n,flush=True)
    rows.append(dict(strategy=name,max_defect=mx,weighted_defect=weighted,weighted_mass_above_1e_8=positive,worst_month=worst[0],worst_wealth=worst[1],saved_action=worst[2],best_action=worst[3],mean0=np.interp(1/12,x,M)))
with (O/'breakpoint_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
