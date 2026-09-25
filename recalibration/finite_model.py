"""Monotone finite-state/action Markov model. Exhaustive action optimisation.

Linear mass deposition, a common upper cap, and fixed GH shocks define the
finite model. Optimality is ONLY within this model/action set. Independent
off-grid Monte Carlo and refinement measure approximation error separately.
"""
import sys, math, time, json, argparse
from pathlib import Path
dep=Path(__file__).resolve().parents[2]/'pydeps'
if dep.exists():sys.path.insert(0,str(dep))
import numpy as np
from numba import njit, prange, set_num_threads
set_num_threads(4)
T=40.;r=.015;b=.04;s=.18;x0=1/12

def gh(n):
    z,w=np.polynomial.hermite.hermgauss(n)
    return z*np.sqrt(2),w/np.sqrt(np.pi)

@njit(cache=True)
def interpolated_moments(M,Q,ix,wt,w):
    """Shared MGH conditional-moment kernel, independent of model constraints."""
    f=0.;q=0.
    for k in range(len(w)):
        j=ix[k];l=wt[k]
        f+=w[k]*(M[j]+l*(M[j+1]-M[j]))
        q+=w[k]*(Q[j]+l*(Q[j+1]-Q[j]))
    return f,q

@njit(cache=True)
def mapping(nx,xmax,nc,N,z):
    dt=T/N;h=xmax/(nx-1)
    ix=np.empty((nx,nc,len(z)),np.int32);wt=np.empty((nx,nc,len(z)))
    for i in range(nx):
        x=i*h
        for a in range(nc):
            for k in range(len(z)):
                y=np.minimum(np.maximum(x*(1+r*dt)+dt+x*a/(nc-1)*(b*dt+s*np.sqrt(dt)*z[k]),0.),xmax)
                j=min(int(y/h),nx-2);ix[i,a,k]=j;wt[i,a,k]=y/h-j
    return ix,wt

@njit(cache=True,parallel=True)
def solve(kind,param,ix,wt,w,h,N):
    nx,nc,ng=ix.shape;dt=T/N
    M=np.empty((N+1,nx));Q=np.empty_like(M);P=np.empty((N,nx),np.float64)
    for i in range(nx):M[N,i]=i*h;Q[N,i]=(i*h-param)**2 if kind==0 else (i*h)**2
    H=0.
    for n in range(N-1,-1,-1):
        H=(dt+H)/(1+r*dt)
        for i in prange(nx):
            gamma=param/(i*h+H) if kind==2 else param
            best=-1e300;ba=0.;bf=0.;bq=0.
            for a in range(nc):
                f,q=interpolated_moments(M[n+1],Q[n+1],ix[i,a],wt[i,a],w)
                score=-q if kind==0 else f-.5*gamma*(q-f*f)
                if score>best:best=score;ba=a;bf=f;bq=q
            P[n,i]=ba/(nc-1)*i*h;M[n,i]=bf;Q[n,i]=bq
    return P,M,Q

@njit(cache=True)
def val(v,x,h):
    y=min(max(x,0.),(len(v)-1)*h);i=min(int(y/h),len(v)-2);l=y/h-i
    return v[i]+l*(v[i+1]-v[i])

@njit(cache=True)
def evaluate(P,h,N,z,w,eval_h):
    xmax=(P.shape[1]-1)*h;nx=int(round(xmax/eval_h))+1
    m=np.arange(nx)*eval_h;q=m*m;dt=T/N
    for n in range(N-1,-1,-1):
        if n==0:
            p=val(P[0],x0,h);mean=0.;second=0.
            for k in range(len(z)):
                y=x0*(1+r*dt)+dt+p*(b*dt+s*math.sqrt(dt)*z[k])
                mean+=w[k]*val(m,y,eval_h);second+=w[k]*val(q,y,eval_h)
            return mean,math.sqrt(max(0.,second-mean*mean))
        mm=np.empty(nx);qq=np.empty(nx)
        for i in range(nx):
            x=i*eval_h;p=val(P[n],x,h);f=0.;v=0.
            for k in range(len(z)):
                y=x*(1+r*dt)+dt+p*(b*dt+s*math.sqrt(dt)*z[k])
                f+=w[k]*val(m,y,eval_h);v+=w[k]*val(q,y,eval_h)
            mm[i]=f;qq[i]=v
        m=mm;q=qq
    mean=val(m,x0,eval_h);v=val(q,x0,eval_h)-mean*mean
    return mean,math.sqrt(max(v,0.))

class Model:
    def __init__(self,nx=3001,xmax=600.,nc=65,N=480,ng=5,eval_h=.025,eval_ng=31):
        self.cfg=dict(nx=nx,xmax=xmax,nc=nc,N=N,ng=ng,eval_h=eval_h,eval_ng=eval_ng)
        self.h=xmax/(nx-1);self.z,self.w=gh(ng)
        self.ix,self.wt=mapping(nx,xmax,nc,N,self.z)
    def solve(self,kind,param):
        t=time.time();p,m,q=solve(kind,param,self.ix,self.wt,self.w,self.h,self.cfg['N'])
        mean,sd=self.evaluate(p)
        print('solve',kind,round(param,8),self.cfg,'evaluation',mean,sd,'seconds',round(time.time()-t,2),flush=True)
        return p,m,q,mean,sd
    def evaluate(self,p,eh=None,ng=None):
        if eh is None:eh=self.cfg['eval_h']
        if ng is None:ng=self.cfg['eval_ng']
        z,w=gh(ng)
        return evaluate(p,self.h,self.cfg['N'],z,w,eh)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nx',type=int,default=3001);ap.add_argument('--nc',type=int,default=65);ap.add_argument('--kind',type=int,default=2);ap.add_argument('--param',type=float,default=1.19)
    a=ap.parse_args();Model(nx=a.nx,nc=a.nc).solve(a.kind,a.param)
