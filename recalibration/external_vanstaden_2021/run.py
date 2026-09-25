import argparse,json,sys,csv,platform,hashlib,subprocess,time,datetime
from pathlib import Path
import numpy as np
import scipy
from engine import backward,forward,R,T,W0
from reference import parameter,distribution,policy,ode,dtparam,B,S
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/validation/van_staden_2021'
NAMES=['PCMV','DOMV','cTCMV','dTCMV']

def config(s):return dict(Nt=320,gh=7,h=.25 if s=='cTCMV' else .005,forward_h=.25 if s in ['DOMV','cTCMV'] else .005,domain=5000. if s in ['DOMV','cTCMV'] else 1e7,control=8. if s=='PCMV' else 32. if s=='dTCMV' else 2000.,samples=257,tol=1e-9)
def variants(s):
    c=config(s);out={'baseline':c}
    for k,v in [('time_coarse',dict(Nt=160)),('time_fine',dict(Nt=640)),('space_coarse',dict(h=c['h']*2,forward_h=c['forward_h']*2)),('space_fine',dict(h=c['h']/2,forward_h=c['forward_h']/2)),('gh_coarse',dict(gh=3)),('gh_fine',dict(gh=15)),('control_narrow',dict(control=c['control']/2)),('control_wide',dict(control=c['control']*2)),('optimizer_coarse',dict(samples=129,tol=1e-6)),('optimizer_fine',dict(samples=513,tol=1e-11)),('domain_narrow',dict(domain=c['domain']/2)),('domain_wide',dict(domain=c['domain']*2))]:out[k]={**c,**v}
    return out

def tail(x,p,a):
    take=np.minimum(p,np.maximum(a-np.r_[0,np.cumsum(p)[:-1]],0));return float(take@x/a)
def stats(x,p):
    p=p/p.sum();m=p@x;v=p@((x-m)**2);c=np.cumsum(p)
    out=dict(mean=m,sd=np.sqrt(v),skewness=p@((x-m)**3)/v**1.5,excess_kurtosis=p@((x-m)**4)/v**2-3)
    for name,a in [('q01',.01),('q05',.05),('q10',.1),('median',.5),('q95',.95)]:out[name]=x[min(np.searchsorted(c,a),len(x)-1)]
    for name,a in [('lcvar01',.01),('lcvar05',.05),('lcvar10',.1)]:out[name]=tail(x,p,a)
    return {k:float(v) for k,v in out.items()}
def savecsv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def run(s,target,variant,cfg,param):
    key=f'{s}_{int(target)}_{variant}';path=OUT/(key+'.npz')
    if path.exists() and (OUT/(key+'.json')).exists():return
    t=time.time();u,diag=backward(s,param,cfg);x,p,audit=forward(s,param,u,cfg);st=stats(x,p);th,cdf,ppf=distribution(s,target,param)
    f=cdf(x);cu=np.cumsum(p/p.sum());dc=float(max(np.max(abs(cu-f)),np.max(abs(np.r_[0,cu[:-1]]-f))))
    times=np.arange(cfg['Nt'])*T/cfg['Nt'];tests=[]
    for n in sorted(set([0,cfg['Nt']//2,cfg['Nt']-1])):
        for wealth in [50.,100.,250.]:
            tau=T-times[n]
            num=u[n]*(wealth-param/2*np.exp(-R*tau)) if s=='PCMV' else u[n]*wealth if s=='dTCMV' else u[n]*np.exp(-R*tau)
            exact=float(policy(s,param,times[n],wealth));tests.append(dict(strategy=s,target=target,variant=variant,time=times[n],wealth=wealth,mgh=float(num),theory=exact,abs_error=abs(num-exact),rel_error=abs(num-exact)/abs(exact) if exact else None))
    ma=diag['backward_a'][0];mq=diag['backward_q'][0];y0=W0*np.exp(R*T)-(param/2 if s=='PCMV' else 0)
    if s=='PCMV':bm=param/2+ma*y0;bv=(mq-ma*ma)*y0*y0
    elif s=='dTCMV':bm=ma*y0;bv=(mq-ma*ma)*y0*y0
    elif s=='cTCMV':bm=y0+ma;bv=mq-ma*ma
    else:bm=None;bv=None
    meta=dict(strategy=s,target=target,variant=variant,parameter=param,config=cfg,statistics=st,theory=th,D_CDF=dc,policy_max_abs=max(v['abs_error'] for v in tests),policy_max_rel=max(v['rel_error'] or 0 for v in tests),control_bound_steps=int(diag['control_bound_steps']),max_mass_error=float(abs(audit[:,1]-1).max()),min_mass=float(audit[:,2].min()),max_lower_boundary_mass=float(audit[:,3].max()),max_upper_boundary_mass=float(audit[:,4].max()),boundary_hit_probability=float(audit[-1,7]),terminal_near_lower_mass=float(p[:3].sum()),terminal_near_upper_mass=float(p[-3:].sum()),boundary_first_moment_loss=float(audit[:,8].sum()),boundary_second_moment_loss=float(audit[:,9].sum()),backward_mean=bm,backward_sd=float(np.sqrt(max(bv,0))) if bv is not None else None,seconds=time.time()-t)
    np.savez_compressed(path,wealth=x,mass=p,control=u,times=times,theory_cdf=f,audit=audit,**{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
    (OUT/(key+'.json')).write_text(json.dumps(meta,indent=2));savecsv(OUT/(key+'_policy.csv'),tests)
    print(key, 'mean',round(st['mean'],6),'sd',round(st['sd'],6),'CDF',round(dc,6),'bound',meta['boundary_hit_probability'],'sec',round(meta['seconds'],1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['baseline','refinement','coupled','all'],default='all');ap.add_argument('--strategy',choices=NAMES);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    paramsfile=OUT/'benchmark_parameters.json'
    if paramsfile.exists():params=json.loads(paramsfile.read_text())
    else:
        params={str(t):{s:parameter(s,t) for s in NAMES} for t in [125,250]};paramsfile.write_text(json.dumps(params,indent=2))
    for s in NAMES:
        if args.strategy and s!=args.strategy:continue
        if args.suite in ['baseline','all']:
            for t in [125,250]:run(s,t,'baseline',config(s),params[str(t)][s])
        if args.suite in ['refinement','all']:
            for name,cfg in variants(s).items():
                if name!='baseline':run(s,250,name,cfg,params['250'][s])
        if args.suite in ['coupled','all']:
            for level,factor in [('coupled_mid',2),('coupled_fine',4)]:
                c=config(s);c.update(Nt=c['Nt']*factor,h=c['h']/factor,forward_h=c['forward_h']/factor)
                for t in [125,250]:run(s,t,level,c,params[str(t)][s])
    if not (OUT/'run_metadata.json').exists():
        info=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),execution_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,source_url='https://cs.uwaterloo.ca/~paforsyt/Distributions_2021.pdf',source_doi='10.1137/20M1338241',published_outputs_used_as_inputs=False,discounted_state=True,random_seed=None,engine='GH quadrature, linear interpolation, numerical backward control optimization, transpose mass deposition; unconstrained symmetry reduction',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
        (OUT/'run_metadata.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':main()
