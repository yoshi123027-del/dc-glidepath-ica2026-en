"""Independent interpolation/symmetry audit and printed-equation diagnostic."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from engine import second_lattice,gh,R,B,S,T,optimize
from reference import dtparam,ode
from run import OUT,savecsv

def main():
    rows=[]
    for path in sorted(OUT.glob('*_baseline.json')):
        m=json.loads(path.read_text());s=m['strategy'];c=m['config'];p=m['parameter'];d=np.load(path.with_suffix('.npz'));N=c['Nt'];dt=T/N;z,w=gh(c['gh']);k=B*dt+S*np.sqrt(dt)*z;h=c['h']
        if s=='DOMV':continue
        # Direct dense-lattice interpolation of future moment ARRAYS; compared
        # against the symmetry reduction, at nodes well away from boundaries.
        grid=np.arange(-20000,20001)*h if s=='cTCMV' else np.exp(np.arange(-6000,6001)*h)
        if s=='PCMV':grid=np.r_[-grid[::-1],0,grid]
        for n in [0,N//2,N-1]:
            a=d['backward_a'][n+1];q=d['backward_q'][n+1];u=d['control'][n]
            states=[-100.,0.,100.] if s=='cTCMV' else [-np.exp(-100*h),1.,np.exp(100*h)] if s=='PCMV' else [np.exp(-100*h),1.,np.exp(100*h)]
            for x in states:
                if s=='cTCMV':
                    ma=grid+a;qa=grid*grid+2*a*grid+q;y=x+u*k
                    red_m=x+w@(u*k)+a;red_q=x*x+2*x*(a+w@(u*k))+w@(second_lattice(u*k,h,'uniform')+2*a*u*k+q)
                else:
                    ma=a*grid;qa=q*grid*grid
                    f=np.exp((B*u-.5*S*S*u*u)*dt+S*u*np.sqrt(dt)*z) if s=='dTCMV' else 1+u*k;y=x*f
                    red_m=a*x*(w@f);red_q=q*x*x*(w@second_lattice(f,h,'log'))
                full_m=w@np.interp(y,grid,ma);full_q=w@np.interp(y,grid,qa)
                em=abs(full_m-red_m);eq=abs(full_q-red_q)
                assert em<1e-8*max(1,abs(full_m));assert eq<1e-8*max(1,abs(full_q))
                rows.append(dict(strategy=s,target=m['target'],month=n,state=x,mean_abs_error=em,second_abs_error=eq))
    savecsv(OUT/'symmetry_full_lattice_audit.csv',rows)
    literal=[]
    params=json.loads((OUT/'benchmark_parameters.json').read_text())
    for target in [125,250]:
        for sign,name in [(-1,'literal_printed_eq_3_5'),(1,'equilibrium_derived_from_eq_2_10')]:
            rho=dtparam(target,sign);sol=ode(rho,sign);I,J=sol.y[:,-1];mean=100*np.exp(R*T+B*I);lv=S*S*J;sd=mean*np.sqrt(np.expm1(lv))
            literal.append(dict(target=target,formulation=name,rho=rho,table_parameter=rho/200,mean=mean,sd=sd,median=mean*np.exp(-lv/2),skewness=(np.exp(lv)+2)*np.sqrt(np.expm1(lv)),excess_kurtosis=np.exp(4*lv)+2*np.exp(3*lv)+3*np.exp(2*lv)-6))
    savecsv(OUT/'dtcmv_printed_equation_diagnostic.csv',literal)
    print('Full-lattice interpolation audit passed;',len(rows),'points')
    for v in literal:print(v)
if __name__=='__main__':main()
