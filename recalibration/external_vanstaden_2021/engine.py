"""Unconstrained MGH adapter: numerical backward optimization + adjoint deposition.
No analytical optimal policy or published output is an input to this module.
Symmetry reduction is exact on the interior infinite uniform/geometric lattices.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from finite_model import gh,interpolated_moments
import numpy as np
from scipy.optimize import minimize_scalar

R=.00623; B=.0816-R; S=.1863; T=10.; W0=100.

def second_lattice(y,h,kind):
    """Linear interpolation of terminal square on an unbounded lattice."""
    if kind=='uniform':
        lo=np.floor(y/h)*h;hi=lo+h
    else:
        ab=np.maximum(np.abs(y),1e-300)
        lo=np.exp(np.floor(np.log(ab)/h)*h);hi=lo*np.exp(h)
        y=ab
    lam=(y-lo)/(hi-lo)
    return (1-lam)*lo*lo+lam*hi*hi

def lattice_moments(y,h,kind,w,a=1.,q=1.):
    """Invoke the actual shared DC-engine conditional-moment kernel.

    Moment arrays are formed at the two neighboring nodes for each GH point;
    repeated nodes are harmless. Scaling/translation reduces storage, not the
    interpolation or quadrature algorithm.
    """
    if kind=='uniform':
        lo=np.floor(y/h)*h;hi=lo+h
    else:
        ab=np.maximum(np.abs(y),1e-300)
        mag=np.exp(np.floor(np.log(ab)/h)*h)
        lo=np.where(y>=0,mag,-mag*np.exp(h));hi=np.where(y>=0,mag*np.exp(h),-mag)
    lam=(y-lo)/(hi-lo);nodes=np.column_stack([lo,hi]).ravel()
    if kind=='uniform':M=nodes+a;Q=nodes*nodes+2*a*nodes+q
    else:M=a*nodes;Q=q*nodes*nodes
    return interpolated_moments(M,Q,np.arange(0,len(nodes),2),lam,w)

def optimize(fn,limit,samples=257,tol=1e-9):
    a=np.linspace(-limit,limit,samples);f=np.array([fn(v) for v in a]);best=int(np.argmin(f));x=a[best];val=f[best]
    # Refine the five best sampled neighborhoods; do not use theoretical controls.
    for j in np.argsort(f)[:5]:
        lo=a[max(j-1,0)];hi=a[min(j+1,len(a)-1)]
        if lo==hi:continue
        z=minimize_scalar(fn,bounds=(lo,hi),method='bounded',options={'xatol':tol,'maxiter':150})
        if z.fun<val:x=float(z.x);val=float(z.fun)
    return x,abs(x)>=limit*(1-1e-6)

def backward(strategy,param,cfg):
    N=cfg['Nt'];dt=T/N;h=cfg['h'];z,w=gh(cfg['gh']);k=B*dt+S*np.sqrt(dt)*z
    ctl=np.empty(N);a=np.ones(N+1);q=np.ones(N+1);hits=0
    if strategy in ['PCMV','DOMV']:
        # Centered terminal loss Y^2. Translation creates the complete target family.
        aa,hit=optimize(lambda u: lattice_moments(1+u*k,h,'log',w)[1],cfg['control'],cfg['samples'],cfg['tol'])
        for n in range(N-1,-1,-1):
            ctl[n]=aa;a[n],q[n]=lattice_moments(1+aa*k,h,'log',w,a[n+1],q[n+1]);hits+=hit
        if strategy=='DOMV':
            pc=ctl.copy();dom_hits=0;targets=np.zeros(N)
            for n in range(N):
                # maximize E[Y_T]-rho Var[Y_T] over translated target family.
                # Objective excluding current discounted wealth is translation invariant.
                c=q[n]-a[n]*a[n]
                def obj(y):return -((a[n]-1)*y-param*c*y*y)
                lim=cfg['control']/max(abs(pc[n]),1e-12)
                yy,bound=optimize(obj,lim,cfg['samples'],cfg['tol']);targets[n]=yy;ctl[n]=pc[n]*yy;dom_hits+=bound
            return ctl,dict(backward_a=a,backward_q=q,embedding_residual=targets,control_bound_steps=dom_hits)
        return ctl,dict(backward_a=a,backward_q=q,control_bound_steps=hits)
    if strategy=='cTCMV':
        a[-1]=0.;q[-1]=0.
        for n in range(N-1,-1,-1):
            an=a[n+1];qn=q[n+1]
            def moments(u):
                return lattice_moments(u*k,h,'uniform',w,an,qn)
            def obj(u):
                mm,qq=moments(u);return -(mm-param*(qq-mm*mm))
            u,hit=optimize(obj,cfg['control'],cfg['samples'],cfg['tol']);ctl[n]=u;a[n],q[n]=moments(u);hits+=hit
    else:
        for n in range(N-1,-1,-1):
            risk=.5*param*np.exp(R*(T-n*dt));an=a[n+1];qn=q[n+1]
            def moments(th):
                f=np.exp((B*th-.5*S*S*th*th)*dt+S*th*np.sqrt(dt)*z)
                return lattice_moments(f,h,'log',w,an,qn)
            def obj(th):
                mm,qq=moments(th);return -(mm-risk*(qq-mm*mm))
            u,hit=optimize(obj,cfg['control'],cfg['samples'],cfg['tol']);ctl[n]=u;a[n],q[n]=moments(u);hits+=hit
    return ctl,dict(backward_a=a,backward_q=q,control_bound_steps=hits)

def make_grid(strategy,cfg):
    h=cfg['forward_h'];L=cfg['domain']
    if strategy in ['DOMV','cTCMV']:return np.arange(-int(L/h),int(L/h)+1)*h
    pos=np.exp(np.arange(np.floor(np.log(1e-8)/h),np.ceil(np.log(L)/h)+1)*h)
    return np.r_[-pos[::-1],0.,pos] if strategy=='PCMV' else pos

def forward(strategy,param,ctl,cfg):
    x=make_grid(strategy,cfg);N=len(ctl);dt=T/N;z,w=gh(cfg['gh']);k=B*dt+S*np.sqrt(dt)*z
    initial=W0*np.exp(R*T)-(param/2 if strategy=='PCMV' else 0)
    p=None;audit=[];surv=None;hit_prob=0.;maxerr=0.;boundary_low=0.;boundary_high=0.
    for n,u in enumerate(ctl):
        state=np.array([initial]) if n==0 else x;mass=np.ones(1) if n==0 else p;alive=np.ones(1) if n==0 else surv
        if strategy=='dTCMV':raw=state[:,None]*np.exp((B*u-.5*S*S*u*u)*dt+S*u*np.sqrt(dt)*z)
        elif strategy=='PCMV':raw=state[:,None]*(1+u*k)
        else:raw=state[:,None]+u*k
        weights=mass[:,None]*w;outside=(raw<x[0])|(raw>x[-1]);lower=float(np.sum(weights*(raw<x[0])));upper=float(np.sum(weights*(raw>x[-1])))
        y=np.clip(raw,x[0],x[-1]);ix=np.minimum(np.searchsorted(x,y,side='right')-1,len(x)-2);ix=np.maximum(ix,0);l=(y-x[ix])/(x[ix+1]-x[ix])
        def deposit(aw):
            return np.bincount(ix.ravel(),weights=(aw*(1-l)).ravel(),minlength=len(x))+np.bincount((ix+1).ravel(),weights=(aw*l).ravel(),minlength=len(x))
        p=deposit(weights);live_weights=alive[:,None]*w;hit_prob+=float(np.sum(live_weights*outside));surv=deposit(live_weights*(~outside))
        maxerr=max(maxerr,abs(p.sum()-1));boundary_low=max(boundary_low,p[0]);boundary_high=max(boundary_high,p[-1])
        audit.append([n+1,p.sum(),p.min(),p[0],p[-1],lower,upper,hit_prob,float(np.sum(weights*(raw-y))),float(np.sum(weights*(raw*raw-y*y)))])
    wealth=x+(param/2 if strategy=='PCMV' else 0)
    return wealth,p,np.array(audit)
