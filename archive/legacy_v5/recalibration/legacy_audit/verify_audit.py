from pathlib import Path
import json, math, csv, hashlib
import numpy as np
from audit import theta_ode, cp_exact

B=Path(__file__).resolve().parent;O=B/'audit_results';R=B.parents[1]
z=np.load(R/'results/monthly_D0_policy_arrays.npz');grid=z['xg_tc'];h=1/12;r=.015;beta=.04;sig=.18;rho=1.193359375
th,_=theta_ode()
# A separate I1/I2 implementation with the original 2014 positive variance sign.
state=np.zeros(2);v=[beta/(rho*sig**2)];step=h/40
def rhs(y):
    t=beta/(rho*sig**2)*(np.exp(-y[0])+rho*np.exp(-y[1])-rho)
    return np.array([r+beta*t+sig**2*t*t,sig**2*t*t])
for n in range(480):
    for k in range(40):
        a=rhs(state);b=rhs(state+step*a/2);c=rhs(state+step*b/2);d=rhs(state+step*c)
        state+=step*(a+2*b+2*c+d)/6
    v.append(beta/(rho*sig**2)*(np.exp(-state[0])+rho*np.exp(-state[1])-rho))
theta_difference=float(np.max(np.abs(th-np.array(v)[::-1])))
assert theta_difference<1e-8
gx,gw=np.polynomial.hermite.hermgauss(5);gx*=np.sqrt(2);gw/=np.sqrt(np.pi)
def forward_cp():
    p=None
    for n in range(480):
        x=np.array([1/12]) if n==0 else grid
        mass=np.ones(1) if n==0 else p
        pi=.4694735*x
        xp=x[:,None]+(r*x+1+beta*pi)[:,None]*h+sig*pi[:,None]*np.sqrt(h)*gx
        xp=np.clip(xp,0,300);j=np.searchsorted(grid,xp,side='right')-1;j=np.clip(j,0,len(grid)-2)
        lam=(xp-grid[j])/(grid[j+1]-grid[j]);w=mass[:,None]*gw
        p=np.bincount(j.ravel(),(w*(1-lam)).ravel(),minlength=len(grid))+np.bincount((j+1).ravel(),(w*lam).ravel(),minlength=len(grid))
    return p
p=forward_cp();mean=p@grid;sd=np.sqrt(p@(grid*grid)-mean*mean)
rows=list(csv.DictReader((R/'results/monthly_baseline_D0_summary.csv').open(encoding='utf-8-sig')));ref=next(x for x in rows if x['strategy']=='CP')
assert abs(mean-float(ref['mean']))<1e-8 and abs(sd-float(ref['stdev']))<1e-8
for f in O.glob('*.csv'):
    if '_paired' in f.name or f.name in ['theta_audit.csv','breakpoint_audit.csv']:continue
    for row in csv.DictReader(f.open(encoding='utf-8-sig')):
        if 'lcvar05' not in row:continue
        assert float(row['lcvar05'])<=float(row['q05'])<=float(row['q50'])<=float(row['q95'])<=float(row['ucvar05'])
        assert 0<=float(row['shortfall60'])<=1
        assert float(row['expected_shortfall60'])>=0
base=list(csv.DictReader((O/'baseline_normal.csv').open(encoding='utf-8-sig')));cp=next(x for x in base if x['strategy']=='CP');ex=cp_exact()
cp_z=(float(cp['mean'])-ex['euler_mean'])/float(cp['mean_se']);assert abs(cp_z)<3
# Algebraic variance identity for diverse non-uniform cells.
rng=np.random.default_rng(6543);a=rng.random(100);width=1+rng.random(100);t=rng.random(100);zz=a+t*width
var=(1-t)*(a-zz)**2+t*(a+width-zz)**2
assert np.max(abs(var-(zz-a)*(a+width-zz)))<1e-14
report=dict(theta_independent_ode_max_difference=theta_difference,cp_grid_reproduced_mean=float(mean),cp_grid_reproduced_sd=float(sd),cp_normal_mean_zscore=cp_z,cp_sd_relative_grid_error=float(sd/ex['euler_sd']-1),tail_order_checks='pass',linear_deposit_variance_identity='pass',tests='pass')
(O/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
