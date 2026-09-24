"""Reporting only: published outputs enter here, never the solver."""
import json,csv,hashlib,datetime,platform,subprocess,gzip,importlib.metadata
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import R,B,S,T,W0,gh,lattice_moments
from run import ROOT,OUT,NAMES,savecsv
from reference import distribution

MAP={'stdev':'sd','var_01':'q01','var_05':'q05','var_10':'q10','cvar_01':'lcvar01','cvar_05':'lcvar05','cvar_10':'lcvar10'}

def internal(m,d):
    s=m['strategy'];c=m['config'];u=d['control'];dt=T/c['Nt'];z,w=gh(c['gh']);k=B*dt+S*np.sqrt(dt)*z
    x0=W0*np.exp(R*T)-(m['parameter']/2 if s=='PCMV' else 0)
    if s in ['PCMV','dTCMV']:
        f=1+u[0]*k if s=='PCMV' else np.exp((B*u[0]-.5*S*S*u[0]**2)*dt+S*u[0]*np.sqrt(dt)*z)
        mm,qq=lattice_moments(x0*f,c['forward_h'],'log',w,d['backward_a'][1],d['backward_q'][1]);sd=np.sqrt(max(qq-mm*mm,0))
        if s=='PCMV':mm+=m['parameter']/2
    else:
        a=q=0.
        for v in u[:0:-1]:a,q=lattice_moments(v*k,c['forward_h'],'uniform',w,a,q)
        mm,qq=lattice_moments(x0+u[0]*k,c['forward_h'],'uniform',w,a,q);sd=np.sqrt(max(qq-mm*mm,0))
    return float(mm),float(sd)

def main():
    published=json.loads((Path(__file__).parent/'published_table_5_1.json').read_text())['values']
    meta=[json.loads(p.read_text()) for p in sorted(OUT.glob('*.json')) if '_baseline' in p.name or any(v in p.name for v in ['_coarse','_fine','_narrow','_wide','_mid'])]
    comparison=[];convergence=[];audits=[];policy=[];summ=[];cdfrows=[]
    for m in meta:
        key=f"{m['strategy']}_{m['target']}_{m['variant']}";d=np.load(OUT/(key+'.npz'));st=m['statistics'];s=m['strategy'];target=m['target'];v=m['variant'];c=m['config']
        bm,bs=internal(m,d)
        m['backward_mean']=bm;m['backward_sd']=bs
        (OUT/(key+'.json')).write_text(json.dumps(m,indent=2))
        row=dict(strategy=s,target=target,variant=v,**c,**{k:st[k] for k in ['mean','sd','q05','q95']},D_CDF=m['D_CDF'],policy_max_abs=m['policy_max_abs'],policy_max_rel=m['policy_max_rel'])
        convergence.append(row)
        audits.append(dict(strategy=s,target=target,variant=v,**{k:m[k] for k in ['max_mass_error','min_mass','max_lower_boundary_mass','max_upper_boundary_mass','boundary_hit_probability','terminal_near_lower_mass','terminal_near_upper_mass','boundary_first_moment_loss','boundary_second_moment_loss','control_bound_steps']},backward_mean=bm,forward_mean=st['mean'],mean_abs_gap=abs(bm-st['mean']),backward_sd=bs,forward_sd=st['sd'],sd_abs_gap=abs(bs-st['sd']),max_control_abs=float(max(abs(d['control'])))))
        with (OUT/(key+'_policy.csv')).open() as f:policy.extend(list(csv.DictReader(f)))
        if v!='coupled_fine':continue
        x=d['wealth'];p=d['mass']/d['mass'].sum();out=dict(st)
        for name,threshold in [('risk_free',W0*np.exp(R*T)),('target',target)]:
            use=x<threshold;pr=p[use].sum();out['prob_below_'+name]=float(pr);out['conditional_mean_below_'+name]=float(x[use]@p[use]/pr)
        out['parameter']=m['parameter']/200 if s=='dTCMV' else m['parameter']
        errs=[]
        for metric,pub in published[str(float(target))][s].items():
            val=out[MAP.get(metric,metric)];ab=abs(val-pub);rel=ab/abs(pub) if pub else None
            comparison.append(dict(strategy=s,target=target,metric=metric,published=pub,mgh=val,abs_error=ab,relative_error=rel,role='rounded_calibration_parameter' if metric=='parameter' else 'output',reference='Table 5.1 p.599',variant=v))
            if metric!='parameter' and rel is not None:errs.append((rel,metric))
        summ.append(dict(strategy=s,target=target,**st,D_CDF=m['D_CDF'],max_published_relative_error=max(errs)[0],max_error_metric=max(errs)[1],policy_max_abs=m['policy_max_abs'],policy_max_rel=m['policy_max_rel'],theory_sd=m['theory']['sd']))
        # Retain every node, no smoothing or subsampling for CDF statistics.
        cu=np.cumsum(p)
        for xx,pp,cc,tt in zip(x,p,cu,d['theory_cdf']):cdfrows.append(dict(strategy=s,target=target,wealth=xx,mass=pp,mgh_cdf=cc,theory_cdf=tt))
    assert len(meta)==72,len(meta)
    for name,rows in [('published_vs_mgh',comparison),('convergence',convergence),('boundary_control_diagnostics',audits),('policy_comparison',policy),('final_summary',summ)]:savecsv(OUT/(name+'.csv'),rows)
    with gzip.open(OUT/'cdf_comparison.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cdfrows[0]));writer.writeheader();writer.writerows(cdfrows)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axs=plt.subplots(4,2,figsize=(8.2,9.2),layout='constrained')
    for i,s in enumerate(NAMES):
        for j,target in enumerate([125,250]):
            m=next(a for a in meta if a['strategy']==s and a['target']==target and a['variant']=='coupled_fine');d=np.load(OUT/f'{s}_{target}_coupled_fine.npz');_,cdf,ppf=distribution(s,target,m['parameter']);ax=axs[i,j]
            lo,hi=ppf(.001),ppf(.995);xx=np.linspace(lo,hi,1500)
            ax.step(d['wealth'],np.cumsum(d['mass']),where='post',color='#245783',lw=1.3,label='MGH backward / forward')
            ax.plot(xx,cdf(xx),'--',color='#c56b25',lw=1.2,label='Equilibrium ODE reference' if s=='dTCMV' else 'Continuous-time theory')
            ax.set(xlim=(lo,hi),ylim=(0,1.015),title=f'{s} | target {target} | max CDF gap {m["D_CDF"]:.5f}',xlabel='Terminal wealth',ylabel='CDF')
            if i in [0,3] and j==0:ax.legend(fontsize=6.8,loc='lower right')
    for ext in ['pdf','png']:fig.savefig(OUT/f'external_cdf_comparison.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(8,5.5),layout='constrained')
    for ax,s in zip(axs.ravel(),NAMES):
        for target in [125,250]:
            vals=[next(a for a in meta if a['strategy']==s and a['target']==target and a['variant']==v) for v in ['baseline','coupled_mid','coupled_fine']]
            ax.plot([320,640,1280],[a['D_CDF'] for a in vals],'o-',label=f'Target {target}')
        ax.set(title=s,xlabel='Nt (spacing reduced proportionally)',ylabel='Maximum CDF gap',xticks=[320,640,1280]);ax.legend(fontsize=8)
    for ext in ['pdf','png']:fig.savefig(OUT/f'external_convergence.{ext}',dpi=220)
    plt.close(fig)
    info=json.loads((OUT/'run_metadata.json').read_text());info.update(report_execution_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'recalibration/finite_model.py']},run_count=len(meta),numba=importlib.metadata.version("numba"))
    (OUT/'run_metadata.json').write_text(json.dumps(info,indent=2))
    (OUT/'model_parameters.json').write_text(json.dumps(dict(initial_wealth=100,horizon=10,r=R,mu=R+B,sigma=S,contribution=0,terminal_benefit=0,constraints='unconstrained signed risky dollars; insolvency permitted; positive proportional branch for dTCMV',targets=[125,250],parameter_source='BENCHMARK_SPECIFICATION.md; eqs.4.2-4.5',dTCMV_source_discrepancy='eq.3.5 printed sign differs from moment equilibrium; both retained'),indent=2))
    print(json.dumps(summ,indent=2));print('max mean/sd consistency gaps',max(a['mean_abs_gap'] for a in audits),max(a['sd_abs_gap'] for a in audits))
if __name__=='__main__':main()
