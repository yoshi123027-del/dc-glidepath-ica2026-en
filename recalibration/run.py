"""Generate target families and recalibrate all strategies to mean 84.78."""
import argparse,json,time
from pathlib import Path
import finite_model as fm
import numpy as np
from numba import njit

@njit(cache=True)
def select_target(V,P,targets,gamma):
    nt,N,nx=P.shape;best=np.full((N,nx),-1e300);out=np.zeros((N,nx));chosen=np.zeros((N,nx),np.int32)
    for j in range(nt):
        for n in range(N):
            for i in range(nx):
                score=targets[j]-.5*gamma*V[j,n,i]
                if score>best[n,i]:best[n,i]=score;out[n,i]=P[j,n,i];chosen[n,i]=j
    return out,chosen

def save(out,name,result,param,cfg,extra=None):
    p,m,q,mean,sd=result
    np.savez_compressed(out/(name+'.npz'),policy=p,parameter=param,**cfg)
    d=dict(strategy=name,parameter=param,mean=mean,sd=sd,**cfg)
    if extra:d.update(extra)
    (out/(name+'.json')).write_text(json.dumps(d,indent=2));return d

def calibrate(fn,lo,hi,increasing=True,tol=.002,maxit=18):
    history=[];best=None
    for it in range(maxit):
        mid=(lo+hi)/2;res=fn(mid);err=res[3]-84.78
        history.append(dict(parameter=mid,mean=res[3],sd=res[4]))
        if best is None or abs(err)<abs(best[1][3]-84.78):best=(mid,res)
        if abs(err)<tol:break
        if (err<0)==increasing:lo=mid
        else:hi=mid
    return best[0],best[1],history

def main():
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['tc','family','dom','pc','cp']);ap.add_argument('--nx',type=int,default=3001);ap.add_argument('--nc',type=int,default=65);ap.add_argument('--ng',type=int,default=5);ap.add_argument('--xmax',type=float,default=600);ap.add_argument('--step',type=float,default=2);ap.add_argument('--out',default='results/current');ap.add_argument('--eval-h',type=float,default=.025);ap.add_argument('--eval-ng',type=int,default=31);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model=fm.Model(nx=a.nx,nc=a.nc,ng=a.ng,xmax=a.xmax,eval_h=a.eval_h,eval_ng=a.eval_ng)
    (out/'config.json').write_text(json.dumps(model.cfg,indent=2))
    if a.task=='tc':
        for kind,name,lo,hi in [(1,'cTCMV',.02,.15),(2,'dTCMV',.5,4.)]:
            param,res,hist=calibrate(lambda p:model.solve(kind,p),lo,hi,False)
            save(out,name,res,param,model.cfg,dict(calibration=hist))
    if a.task=='pc':
        param,res,hist=calibrate(lambda p:model.solve(0,p),85.,135.)
        gamma=1/(param-res[3])
        save(out,'PCMV',res,param,model.cfg,dict(implied_gamma=gamma,calibration=hist))
    if a.task=='family':
        # Terminal wealth is capped at xmax in this finite model. For gamma>=.02,
        # the embedding optimum z=E[W]+1/gamma lies in [0,xmax+50].
        targets=np.arange(0,a.xmax+50+a.step*.5,a.step)
        np.save(out/'targets.npy',targets)
        shape=(len(targets),model.cfg['N'],a.nx)
        V=np.lib.format.open_memmap(out/'family_V.npy',mode='w+',dtype='float64',shape=shape)
        P=np.lib.format.open_memmap(out/'family_P.npy',mode='w+',dtype='float32',shape=shape)
        for j,z in enumerate(targets):
            t=time.time();p,m,v=fm.solve(0,z,model.ix,model.wt,model.w,model.h,model.cfg['N'])
            V[j]=v[:-1];P[j]=p
            if j%10==0:V.flush();P.flush();print('family',j+1,len(targets),'z',z,'seconds',time.time()-t,flush=True)
        V.flush();P.flush()
    if a.task=='dom':
        V=np.load(out/'family_V.npy',mmap_mode='r');P=np.load(out/'family_P.npy',mmap_mode='r');targets=np.load(out/'targets.npy')
        def run(g):
            p,ch=select_target(V,P,targets,g)
            # Target family policies are stored as float32. Restore feasibility
            # after rounding at the upper constraint before evaluation/export.
            p=np.minimum(np.maximum(p,0.),np.arange(model.cfg['nx'])*model.h)
            mean,sd=model.evaluate(p)
            print('DOMV',g,mean,sd,flush=True)
            return p,ch,None,mean,sd
        param,res,hist=calibrate(run,.02,.2,False)
        ch=res[1];save(out,'DOMV',res,param,model.cfg,dict(calibration=hist,target_step=a.step,target_max=float(targets[-1]),upper_boundary_fraction=float(np.mean(ch==len(targets)-1))))
    if a.task=='cp':
        def run(th):
            dt=40/480;A=1+(.015+.04*th)*dt;m=1/12;q=m*m
            for _ in range(480):q,m=(A*A+.18**2*th*th*dt)*q+2*A*dt*m+dt*dt,A*m+dt
            p=np.tile(th*np.linspace(0,a.xmax,a.nx),(480,1))
            return p,None,None,m,np.sqrt(q-m*m)
        param,res,hist=calibrate(run,0.,1.,tol=1e-8,maxit=40)
        save(out,'CP',res,param,model.cfg,dict(calibration=hist,analytic=True))

if __name__=='__main__':main()
