"""Independent continuous-time benchmark; never imported by numerical engine."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm,lognorm
R=.00623;B=.0816-R;S=.1863;T=10.;W0=100.;A=(B/S)**2

def ode(rho,sign=1,rtol=2e-11):
    def rhs(t,y):
        th=B/(rho*S*S)*(np.exp(np.clip(-R*t-B*y[0]-sign*S*S*y[1],-700,80))+rho*np.exp(-S*S*y[1])-rho)
        return [th,th*th]
    sol=solve_ivp(rhs,[0,T],[0,0],method='DOP853',rtol=rtol,atol=rtol*.01,dense_output=True)
    return sol

def dtparam(target,sign=1):
    def score(logrho):
        sol=ode(np.exp(logrho),sign)
        if not sol.success:return np.nan
        return R*T+B*sol.y[0,-1]-np.log(target/W0)
    xx=np.linspace(-6,6,121);last=None
    for x in xx:
        f=score(x)
        if np.isfinite(f):
            if last is not None and f*last[1]<0:return np.exp(brentq(score,last[0],x,xtol=1e-12))
            last=(x,f)
    raise ValueError('no successful bracket')

def parameter(strategy,target):
    rf=W0*np.exp(R*T)
    if strategy=='PCMV':return 2*rf+2*np.exp(A*T)/np.expm1(A*T)*(target-rf)
    if strategy=='DOMV':return np.expm1(A*T)/(2*(target-rf))
    if strategy=='cTCMV':return A*T/(2*(target-rf))
    return dtparam(target)

def distribution(strategy,target,param):
    rf=W0*np.exp(R*T)
    if strategy=='PCMV':
        upper=param/2;scale=upper-rf;lv=A*T;lm=-1.5*lv
        d=lognorm(s=np.sqrt(lv),scale=np.exp(lm))
        cdf=lambda x:d.sf(np.maximum((upper-np.asarray(x))/scale,0))
        ppf=lambda p:upper-scale*d.ppf(1-np.asarray(p))
        mean=upper-scale*d.mean();sd=scale*d.std();sk=-float(d.stats(moments='s'));ku=float(d.stats(moments='k'))
    elif strategy in ['DOMV','cTCMV']:
        sd=np.sqrt(.5*np.expm1(2*A*T))/(2*param) if strategy=='DOMV' else np.sqrt(A*T)/(2*param)
        mean=rf+(np.expm1(A*T) if strategy=='DOMV' else A*T)/(2*param);d=norm(loc=mean,scale=sd);cdf=d.cdf;ppf=d.ppf;sk=0.;ku=0.
    else:
        sol=ode(param);I,J=sol.y[:,-1];lv=S*S*J;lm=np.log(W0)+R*T+B*I-.5*lv
        d=lognorm(s=np.sqrt(lv),scale=np.exp(lm));cdf=d.cdf;ppf=d.ppf;mean=d.mean();sd=d.std();sk=float(d.stats(moments='s'));ku=float(d.stats(moments='k'))
    return dict(mean=mean,sd=sd,median=float(ppf(.5)),q05=float(ppf(.05)),q95=float(ppf(.95)),skewness=sk,excess_kurtosis=ku),cdf,ppf

def policy(strategy,param,t,x):
    tau=T-np.asarray(t)
    if strategy=='PCMV':return B/(S*S)*(param/2*np.exp(-R*tau)-x)
    if strategy=='DOMV':return B/(2*param*S*S)*np.exp((A-R)*tau)+np.zeros_like(x)
    if strategy=='cTCMV':return B/(2*param*S*S)*np.exp(-R*tau)+np.zeros_like(x)
    sol=ode(param);I,J=sol.sol(tau);th=B/(param*S*S)*(np.exp(-R*tau-B*I-S*S*J)+param*np.exp(-S*S*J)-param)
    return th*x
