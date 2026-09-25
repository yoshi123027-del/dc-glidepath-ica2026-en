"""Compare fixed saved policies; never optimises or recalibrates.

Read policy and Monte Carlo inputs from ``results/current`` and write the
evaluator comparison under ``results/validation/evaluator_comparison`` by
default.
"""
import argparse,csv,hashlib,json,platform
from pathlib import Path
import numpy as np
NAMES=['PCMV','DOMV','cTCMV','dTCMV','CP']
T=40.;r=.015;b=.04;sigma=.18;x0=1/12

def transitions(x,p,h,nx,z):
    raw=x[:,None]*(1+r/12)+1/12+p[:,None]*(b/12+sigma/np.sqrt(12)*z)
    y=np.clip(raw,0,(nx-1)*h);ix=np.minimum((y/h).astype(int),nx-2);l=y/h-ix
    return raw,y,ix,l

def forward(P,h,z,w):
    N,nx=P.shape;grid=np.arange(nx)*h
    masses=np.zeros((N+1,nx));glide=np.zeros(N);audit=np.zeros((N,6))
    for n in range(N):
        x=np.array([x0]) if n==0 else grid
        mass=np.ones(1) if n==0 else masses[n]
        control=np.array([np.interp(x0,grid,P[n])]) if n==0 else P[n]
        glide[n]=mass@np.divide(control,x,out=np.zeros_like(control),where=x>0)
        raw,y,ix,l=transitions(x,control,h,nx,z);a=mass[:,None]*w
        nxt=np.bincount(ix.ravel(),weights=(a*(1-l)).ravel(),minlength=nx)+np.bincount((ix+1).ravel(),weights=(a*l).ravel(),minlength=nx)
        masses[n+1]=nxt
        mu=np.sum(a*y);q=np.sum(a*y*y);extra=np.sum(a*l*(1-l)*h*h)
        audit[n]=[np.sum(a*(raw>grid[-1])),np.sum(a*(raw<0)),mu,q,extra,nxt@(grid*grid)-q-extra]
    return masses,glide,audit

def backward(P,h,z,w):
    N,nx=P.shape;grid=np.arange(nx)*h;m=grid.copy();q=grid**2
    for n in range(N-1,-1,-1):
        x=np.array([x0]) if n==0 else grid
        p=np.array([np.interp(x0,grid,P[n])]) if n==0 else P[n]
        _,_,ix,l=transitions(x,p,h,nx,z)
        m=((m[ix]*(1-l)+m[ix+1]*l)*w).sum(axis=1)
        q=((q[ix]*(1-l)+q[ix+1]*l)*w).sum(axis=1)
    return float(m[0]),float(np.sqrt(q[0]-m[0]**2))

def weighted_quantile(x,p,a):
    return float(x[min(np.searchsorted(np.cumsum(p),a,side='left'),len(x)-1)])
def tail(x,p,a=.05):
    take=np.minimum(p,np.maximum(0.,a-np.r_[0.,np.cumsum(p)[:-1]]))
    assert abs(take.sum()-a)<1e-12
    return float(take@x/a)
def stats(x,p=None):
    if p is None:
        q=np.quantile(x,[.01,.05,.5,.95]);m=x.mean();sd=x.std(ddof=1);s=np.sort(x)
        low=s[:len(s)//20].mean();up=s[-len(s)//20:].mean();v=x.var();sk=np.mean((x-m)**3)/v**1.5
    else:
        p=p/p.sum();m=p@x;v=p@((x-m)**2);sd=np.sqrt(v);sk=p@((x-m)**3)/v**1.5
        q=[weighted_quantile(x,p,a) for a in [.01,.05,.5,.95]];low=tail(x,p);up=tail(x[::-1],p[::-1])
    return dict(mean=float(m),sd=float(sd),q01=float(q[0]),q05=float(q[1]),median=float(q[2]),q95=float(q[3]),lcvar05=float(low),ucvar05=float(up),skewness=float(sk))
def ks(x,p,sample):
    # Exact supremum over both left and right limits at ALL support points.
    v=np.sort(sample);cdf=np.cumsum(p/p.sum());before=np.r_[0.,cdf[:-1]]
    d1=np.max(np.abs(cdf-np.searchsorted(v,x,side='right')/len(v)))
    d2=np.max(np.abs(before-np.searchsorted(v,x,side='left')/len(v)))
    u,left,count=np.unique(v,return_index=True,return_counts=True)
    idx=np.searchsorted(x,u,side='right');f=np.r_[0.,cdf][idx]
    idxl=np.searchsorted(x,u,side='left');fl=np.r_[0.,cdf][idxl]
    return float(max(d1,d2,np.max(np.abs(f-(left+count)/len(v))),np.max(np.abs(fl-left/len(v)))))
def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='results/current')
    ap.add_argument('--out',default='results/validation/evaluator_comparison')
    ap.add_argument('--rerun-mc',action='store_true')
    a=ap.parse_args()
    data=Path(a.data_dir)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((data/'config.json').read_text());assert (cfg['nx'],cfg['xmax'],cfg['N'],cfg['ng'])==(6001,600.,480,7)
    h=cfg['xmax']/(cfg['nx']-1);grid=np.arange(cfg['nx'])*h;z,w=np.polynomial.hermite.hermgauss(cfg['ng']);z=z*np.sqrt(2);w=w/np.sqrt(np.pi)
    inputs={p.name:sha(p) for p in [data/(n+ext) for n in NAMES for ext in ['.npz','.json']]+[data/'config.json',data/'independent_mc.npz',data/'independent_mc.csv']}
    saved=np.load(data/'independent_mc.npz');assert list(saved['names'])==NAMES
    X=saved['terminal'];G=saved['glide'];seed=int(saved['seed']);assert X.shape==(5,1000000) and seed==20260912
    policies=[np.load(data/(n+'.npz'))['policy'] for n in NAMES]
    validation=dict(seed=seed,paths=1000000,blocks=100,mc_source=str(data/'independent_mc.npz'),mc_regenerated=a.rerun_mc,initial_state=x0,initial_transition='exact initial state; risky dollars interpolation',mgh_h=h,mgh_ng=7,mc_upper_cap=False,policy_sha256=inputs,strategies={})
    if a.rerun_mc:
        from validate import mc
        xx,gg,hit,neg=mc(np.stack(policies),h,1000000,seed)
        print("MC max terminal reproduction error",float(abs(xx-X).max()),flush=True)
        assert np.allclose(xx,X,rtol=0,atol=1e-7)
        assert np.allclose(gg.mean(axis=0),G,rtol=0,atol=1e-11)
        validation.update(mc_terminal_bitwise_reproduced=bool(np.array_equal(xx,X)),mc_terminal_max_reproduction_error=float(abs(xx-X).max()),mc_glide_max_reproduction_error=float(abs(gg.mean(axis=0)-G).max()),mc_cap_path_counts=hit.sum(axis=0).tolist(),mc_negative_step_counts=neg.sum(axis=0).tolist())
    old=list(csv.DictReader((data/'independent_mc.csv').open()));dis=[];comp=[];gl=[];series=[];massall=[];auditrows=[]
    for j,(name,P) in enumerate(zip(NAMES,policies)):
        assert P.shape==(480,6001)
        masses,g,au=forward(P,h,z,w);massall.append(masses);p=masses[-1];s=stats(grid,p);t=stats(X[j]);diff=g-G[j]
        for key in ['mean','sd','q05','median','q95','lcvar05']:assert abs(t[key]-float(old[j][key]))<1e-10
        for ev,st in [('MGH',s),('MC',t)]:dis.append(dict(strategy=name,evaluator=ev,**st))
        comp.append(dict(strategy=name,D_KS=ks(grid,p,X[j]),**{'delta_'+k:s[k]-t[k] for k in s}))
        gl.append(dict(strategy=name,max_abs_pp=100*abs(diff).max(),mae_pp=100*abs(diff).mean(),rmse_pp=100*np.sqrt(np.mean(diff**2)),max_month=int(np.argmax(abs(diff))),max_year=float(np.argmax(abs(diff))/12)))
        mean,sd=backward(P,h,z,w)
        assert abs(mean-s['mean'])<1e-8 and abs(sd-s['sd'])<1e-8
        assert abs(masses[1:].sum(axis=1)-1).max()<1e-10 and masses.min()>=0
        assert abs(au[:,5]).max()<1e-7
        assert abs(masses[1:]@grid-au[:,2]).max()<1e-9
        if name=='CP':
            theta=float(np.load(data/'CP.npz')['parameter']);assert abs(g-theta).max()<1e-10 and abs(G[j]-theta).max()<1e-10
            validation['cp_theta']=theta;validation['cp_max_deviation_mgh']=float(abs(g-theta).max());validation['cp_max_deviation_mc']=float(abs(G[j]-theta).max())
        validation['strategies'][name]=dict(max_mass_error=float(abs(masses[1:].sum(axis=1)-1).max()),min_mass=float(masses.min()),max_lower_mass=float(masses[1:,0].max()),max_upper_mass=float(masses[1:,-1].max()),expected_upper_crossings=float(au[:,0].sum()),expected_lower_crossings=float(au[:,1].sum()),backward_mean=mean,backward_sd=sd,max_local_second_moment_error=float(abs(au[:,5]).max()))
        for n in range(480):
            series.append(dict(strategy=name,month=n,year=n/12,mgh=g[n],mc=G[j,n],delta=diff[n],delta_pp=100*diff[n]))
            mass=masses[n+1];auditrows.append(dict(strategy=name,month=n+1,total_mass=mass.sum(),min_mass=mass.min(),lower_mass=mass[0],upper_mass=mass[-1],upper_crossing_mass=au[n,0],lower_crossing_mass=au[n,1],mean=mass@grid,second_moment=mass@(grid**2),mean_identity_error=mass@grid-au[n,2],second_identity_error=au[n,5],deposition_variance=au[n,4]))
        print(name,comp[-1],gl[-1],flush=True)
    for f,rows in [('evaluator_distribution_comparison.csv',dis),('evaluator_comparison.csv',comp),('evaluator_glidepath_comparison.csv',gl),('evaluator_glidepaths.csv',series),('evaluator_mass_audit.csv',auditrows)]:writecsv(out/f,rows)
    import tempfile,shutil
    tmp=Path(tempfile.mktemp(suffix='.npz'))
    np.savez_compressed(tmp,names=NAMES,grid=grid,mgh_terminal_mass=np.array(massall)[:,-1,:],mgh_glide=np.array([np.array([v['mgh'] for v in series if v['strategy']==n]) for n in NAMES]),mc_glide=G,mc_source='independent_mc.npz',seed=seed,initial_wealth=x0)
    shutil.copyfile(tmp,out/'evaluator_comparison.npz');tmp.unlink()
    for filename,digest in inputs.items():assert sha(data/filename)==digest
    validation['environment']=dict(python=platform.python_version(),numpy=np.__version__)
    (out/'evaluator_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
if __name__=='__main__':main()
