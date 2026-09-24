"""Analytical anchors, forward/backward identities, and global action audit."""
import argparse,json,csv,math,time
from pathlib import Path
import finite_model as fm
import numpy as np
from numba import njit

@njit(cache=True)
def forward(P,h,z,w):
    N,nx=P.shape;dt=40/N;mass=np.zeros(nx);j=int((1/12)/h);l=(1/12)/h-j;mass[j]=1-l;mass[j+1]=l
    overflow=0.
    for n in range(N):
        nxt=np.zeros(nx)
        for i in range(nx):
            if mass[i]<1e-300:continue
            for k in range(len(z)):
                y=i*h*(1+.015*dt)+dt+P[n,i]*(.04*dt+.18*np.sqrt(dt)*z[k])
                if y>(nx-1)*h:overflow+=mass[i]*w[k]
                y=min(max(y,0.),(nx-1)*h);j=min(int(y/h),nx-2);l=y/h-j
                nxt[j]+=mass[i]*w[k]*(1-l);nxt[j+1]+=mass[i]*w[k]*l
        mass=nxt
    x=np.arange(nx)*h;m=np.sum(mass*x);q=np.sum(mass*x*x)
    return m,np.sqrt(max(0.,q-m*m)),np.sum(mass),overflow

def breakpoints(x,h,xmax,z):
    a=x*(1+.015/12)+1/12;d=x*(.04/12+.18/np.sqrt(12)*z);all_a=[0.,1.]
    for v in d:
        if abs(v)<1e-15:continue
        low=max(0,int(np.floor(min(a,a+v)/h))-1);high=min(int(xmax/h),int(np.ceil(max(a,a+v)/h))+1)
        aa=(np.arange(low,high+1)*h-a)/v;all_a.extend(aa[(aa>0)&(aa<1)])
    return np.unique(all_a)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();out=Path(a.directory)
    cfg=json.loads((out/'config.json').read_text());model=fm.Model(**cfg);results={};audit=[]
    if cfg['N']!=480:
        raise ValueError('The breakpoint audit in this script is monthly (N=480).')
    for name,kind in [('PCMV',0),('cTCMV',1),('dTCMV',2)]:
        zz=np.load(out/(name+'.npz'));param=float(zz['parameter']);P,M,Q,mean,sd=model.solve(kind,param)
        # Reproduction verifies that saved policy arrays are the actual optimisers.
        assert np.array_equal(P,zz['policy'])
        mm,ss,mass,over=forward(P,model.h,model.z,model.w)
        m0=fm.val(M[0],fm.x0,model.h);q0=fm.val(Q[0],fm.x0,model.h)
        if kind==0:q0=q0+2*param*m0-param*param
        sd0=np.sqrt(max(0,q0-m0*m0))
        results[name]=dict(backward_mean=m0,forward_mean=mm,backward_sd=sd0,forward_sd=ss,mass=mass,expected_cap_events=over)
        assert abs(mm-m0)<1e-7 and abs(ss-sd0)<1e-7 and abs(mass-1)<1e-10
        for n in [0,60,120,240,360,420,468,479]:
            H=(1-(1+.015/12)**(-(480-n)))/.015
            # Explicit sample; not a whole-state-space or probability-weighted bound.
            for rawx in [1/12,5,15,30,50,75,100,150,250,400,550]:
                i=min(round(rawx/model.h),cfg['nx']-1);x=i*model.h
                if x==0:continue
                aa=breakpoints(x,model.h,cfg['xmax'],model.z)
                y=x*(1+.015/12)+1/12+aa[:,None]*x*(.04/12+.18/np.sqrt(12)*model.z)
                grid=np.arange(cfg['nx'])*model.h
                f=np.interp(y,grid,M[n+1])@model.w;q=np.interp(y,grid,Q[n+1])@model.w
                gamma=param/(x+H) if kind==2 else param
                score=-q if kind==0 else f-.5*gamma*(q-f*f)
                saved=-Q[n,i] if kind==0 else M[n,i]-.5*gamma*(Q[n,i]-M[n,i]**2)
                gap=max(0.,float(score.max()-saved))
                audit.append(dict(strategy=name,month=n,wealth=x,global_continuous_action_gap=gap,best_fraction=float(aa[np.argmax(score)]),saved_fraction=float(P[n,i]/x)))
    # Constant-proportion analytical Euler benchmark versus multiple evaluation meshes.
    zz=np.load(out/'CP.npz');theta=float(zz['parameter']);m=1/12;q=m*m;dt=1/12;A=1+(.015+.04*theta)*dt
    for _ in range(480):q,m=(A*A+.18**2*theta*theta*dt)*q+2*A*dt*m+dt*dt,A*m+dt
    cp=dict(analytic_mean=m,analytic_sd=np.sqrt(q-m*m),meshes=[])
    for h in [.2,.1,.05,.025]:
        mean,sd=model.evaluate(zz['policy'],eh=h,ng=9);cp['meshes'].append(dict(h=h,mean=mean,sd=sd))
    results['CP']=cp
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    with (out/'action_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
