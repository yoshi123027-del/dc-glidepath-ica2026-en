"""Final acceptance checks; fail visibly rather than round away mean differences."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();out=Path(a.directory)
    names=['PCMV','DOMV','cTCMV','dTCMV','CP'];mean_errors={}
    for name in names:
        p=json.loads((out/(name+'.json')).read_text());mean_errors[name]=p['mean']-84.78
        assert abs(mean_errors[name])<=.01,(name,mean_errors[name])
        z=np.load(out/(name+'.npz'));x=np.arange(int(z['nx']))*float(z['xmax'])/(int(z['nx'])-1)
        assert np.all(np.isfinite(z['policy']))
        assert np.min(z['policy'])>=-1e-10 and np.max(z['policy']-x)<=1e-10
    rows=list(csv.DictReader((out/'independent_mc.csv').open()))
    assert len(rows)==5
    for r in rows:
        assert int(r['paths'])>=1000000
        assert float(r['lcvar05'])<=float(r['q05'])<=float(r['median'])<=float(r['q95'])
        assert int(r['negative_steps'])==0
    # Bonferroni 95% simultaneous normal-approximation CIs for all 10 pairs.
    # Prespecified practical equivalence margin: +/-0.1 terminal wealth units.
    # This is a numerical/statistical tolerance, not exact equality of real numbers.
    pairs=list(csv.DictReader((out/'paired_means.csv').open()));intervals=[]
    for r in pairs:
        m=float(r['mean_difference']);se=float(r['se']);lo=m-2.807033768343811*se;hi=m+2.807033768343811*se
        intervals.append(dict(first=r['first'],second=r['second'],lower=lo,upper=hi))
        assert lo>-.1 and hi<.1,(r['first'],r['second'],lo,hi)
    checks=json.loads((out/'checks.json').read_text())
    for n in ['PCMV','cTCMV','dTCMV']:
        r=checks[n];assert abs(r['forward_mean']-r['backward_mean'])<1e-7
        assert abs(r['forward_sd']-r['backward_sd'])<1e-7
    report=dict(status='pass',calibration_mean_errors=mean_errors,calibration_tolerance=.01,simultaneous_mean_difference_margin=.1,simultaneous_confidence_approx=.95,intervals=intervals,scope='Numerical finite-model optimality/equilibrium and off-grid statistical equivalence; not exact continuous-time optimality.')
    (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
