from pathlib import Path
import numpy as np
import math, csv, json, hashlib, argparse

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
OUT=BASE/'audit_results';OUT.mkdir(exist_ok=True)
Z=np.load(REPO/'results/monthly_D0_policy_arrays.npz')
r,b,s,c,T,dt,x0=.015,.04,.18,1.,40.,1/12,1/12
rho=1.193359375; gc=.059638671875;gd=.0912608;target=104.772

def theta_ode(sign=1, sub=20):
    # tau=T-t; A=log first moment multiplier, B=log second multiplier.
    state=np.zeros(2); vals=[b/(rho*s*s)];ab=[state.copy()]
    h=dt/sub
    def f(q):
        if sign==1:
            th=b/(rho*s*s)*(np.exp(q[0]-q[1])+rho*np.exp(2*q[0]-q[1])-rho)
            return np.array([r+b*th,2*(r+b*th)+s*s*th*th])
        th=b/(rho*s*s)*(np.exp(-q[0])+rho*np.exp(-q[1])-rho)
        return np.array([r+b*th-s*s*th*th,s*s*th*th])
    for n in range(480):
        for j in range(sub):
            k1=f(state);k2=f(state+h*k1/2);k3=f(state+h*k2/2);k4=f(state+h*k3)
            state+=h*(k1+2*k2+2*k3+k4)/6
        if sign==1:
            v=b/(rho*s*s)*(np.exp(state[0]-state[1])+rho*np.exp(2*state[0]-state[1])-rho)
        else:v=b/(rho*s*s)*(np.exp(-state[0])+rho*np.exp(-state[1])-rho)
        vals.append(v);ab.append(state.copy())
    return np.array(vals)[::-1],np.array(ab)[::-1]

def cp_exact(th=.4694735):
    a=r+b*th;v=s*s*th*th;A=1+a*dt;m=x0;q=x0*x0
    for n in range(480):q,m=(A*A+v*dt)*q+2*A*c*dt*m+c*c*dt*dt,A*m+c*dt
    k=2*a+v;mc=(x0+c/a)*math.exp(a*T)-c/a
    qc=math.exp(k*T)*(x0*x0+2*c*((x0+c/a)*math.expm1((a-k)*T)/(a-k)-(c/a)*math.expm1(-k*T)/(-k)))
    return dict(euler_mean=m,euler_sd=math.sqrt(q-m*m),continuous_mean=mc,continuous_sd=math.sqrt(qc-mc*mc))

def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paths',type=int,default=200000);ap.add_argument('--start',type=int,default=0);ap.add_argument('--wealth',type=float,default=x0);ap.add_argument('--tag',default='baseline');ap.add_argument('--method',choices=['normal','gh','buyhold'],default='normal');args=ap.parse_args()
    theta,ab=theta_ode();old,_=theta_ode(-1);fine,_=theta_ode(sub=40)
    assert np.max(abs(theta-fine))<1e-8
    rows=[]
    for y in [0,10,20,30,35,39,40]:
        n=y*12;A,B=ab[n];aa,bb=np.exp(A),np.exp(B)
        # Hamiltonian FOC beta*((1+rho*a)*a-rho*b)-sigma^2*rho*b*theta=0
        residual=b*((1+rho*aa)*aa-rho*bb)-s*s*rho*bb*theta[n]
        rows.append(dict(year=y,theta_correct=theta[n],theta_reference=old[n],foc_residual=residual))
    csvwrite('theta_audit.csv',rows)
    (OUT/'cp_exact.json').write_text(json.dumps(cp_exact(),indent=2))
    names=['PCMV_saved','DOMV_saved','cTCMV_saved','dTCMV_saved','CP','PCMV_clip_fixed_target','DOMV_clip','cTCMV_clip','dTCMV_clip']
    policies=[Z[k] for k in ['pcmv_policy','domv_policy','ctcmv_policy','dtcmv_policy']]
    grids=[Z['xg_pc'],Z['xg_pc'],Z['xg_tc'],Z['xg_tc']]
    npath=args.paths
    rng=np.random.default_rng(20260911+args.start)
    X=np.full((len(names),npath),args.wealth)
    glide=np.zeros((480-args.start,len(names))); high=np.zeros(len(names));low=np.zeros(len(names));hit=np.zeros_like(X,dtype=bool)
    gx,gw=np.polynomial.hermite.hermgauss(5);gx*=np.sqrt(2);gw/=np.sqrt(np.pi)
    for n in range(args.start,480):
        shocks=rng.standard_normal(npath) if args.method!='gh' else rng.choice(gx,size=npath,p=gw)
        pi=np.empty_like(X)
        for j in range(4):
            grid=grids[j];p=policies[j][n]*grid
            pi[j]=np.interp(X[j],grid,p)
            # Above grid continue endpoint risky fraction; never cap wealth.
            idx=X[j]>grid[-1];pi[j,idx]=policies[j][n,-1]*X[j,idx]
        tau=T-n*dt;H=c*(-math.expm1(-r*tau))/r
        pi[4]=.4694735*X[4]
        pi[5]=b/(s*s)*(target*math.exp(-r*tau)-X[5]-H)
        pi[6]=b/(gd*s*s)*math.exp((b*b/(s*s)-r)*tau)
        pi[7]=b/(gc*s*s)*math.exp(-r*tau)
        pi[8]=theta[n]*(X[8]+H)
        pi=np.minimum(np.maximum(pi,0),X)
        glide[n-args.start]=np.mean(np.divide(pi,X,out=np.zeros_like(pi),where=X>0),axis=1)
        if args.method=='buyhold':
            X=(X-pi)*math.exp(r*dt)+pi*np.exp((r+b-.5*s*s)*dt+s*math.sqrt(dt)*shocks)+c*dt
        else:X+= (r*X+c+b*pi)*dt+s*pi*math.sqrt(dt)*shocks
        low+=np.sum(X<0,axis=1);X=np.maximum(X,0);hit|=X>300
        if n%120==119:print(args.tag,args.method,'month',n+1,flush=True)
    rows=[]
    for j,name in enumerate(names):
        v=X[j];mean=v.mean();sd=v.std(ddof=1);sv=np.sort(v);k=int(.05*npath)
        # Independent paths, pointwise normal approximations for mean and paired differences.
        rows.append(dict(strategy=name,paths=npath,mean=mean,mean_se=sd/math.sqrt(npath),sd=sd,q05=np.quantile(v,.05),q50=np.quantile(v,.5),q95=np.quantile(v,.95),lcvar05=sv[:k].mean(),ucvar05=sv[-k:].mean(),shortfall60=np.mean(v<60),expected_shortfall60=np.maximum(60-v,0).mean(),avg_glide=glide[:,j].mean(),ever_above300=hit[j].mean(),negative_steps=int(low[j])))
    csvwrite(args.tag+'_'+args.method+'.csv',rows)
    paired=[]
    for saved,clip in [(0,5),(1,6),(2,7),(3,8)]:
        dif=X[clip]-X[saved];se=dif.std(ddof=1)/math.sqrt(npath)
        paired.append(dict(strategy=names[saved],mean_diff=dif.mean(),paired_se=se,ci_low=dif.mean()-1.96*se,ci_high=dif.mean()+1.96*se,lcvar_diff=rows[clip]['lcvar05']-rows[saved]['lcvar05'],glide_mae=np.mean(abs(glide[:,clip]-glide[:,saved]))))
    csvwrite(args.tag+'_'+args.method+'_paired.csv',paired)
    np.savez_compressed(OUT/(args.tag+'_'+args.method+'_plot.npz'),names=np.array(names),glide=glide,quantile_probs=np.linspace(.001,.999,999),quantiles=np.quantile(X,np.linspace(.001,.999,999),axis=1),times=np.arange(args.start,480)/12)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
