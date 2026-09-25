"""Evaluate saved policies on finer independent meshes and re-solve boundary cases."""
import argparse,json,csv
from pathlib import Path
import finite_model as fm
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--boundary',action='store_true');ap.add_argument('--recalibrate-boundary',action='store_true');a=ap.parse_args();out=Path(a.directory)
    cfg=json.loads((out/'config.json').read_text());model=fm.Model(**cfg);rows=[]
    for name in ['PCMV','DOMV','cTCMV','dTCMV','CP']:
        f=out/(name+'.npz')
        if not f.exists():continue
        z=np.load(f)
        for h,ng in [(.05,9),(.025,9),(.025,15),(.025,31),(.025,61)]:
            m,s=model.evaluate(z['policy'],eh=h,ng=ng)
            rows.append(dict(strategy=name,variant='evaluation',nx=cfg['nx'],xmax=cfg['xmax'],nc=cfg['nc'],ng=ng,eval_h=h,mean=m,sd=s))
    if a.boundary:
        large=fm.Model(nx=9001,xmax=900.,nc=129,ng=7)
        for name,kind in [('PCMV',0),('cTCMV',1),('dTCMV',2)]:
            z=np.load(out/(name+'.npz'));p,m,q,mean,sd=large.solve(kind,float(z['parameter']))
            rows.append(dict(strategy=name,variant='reoptimise_cap900_fixed_parameter',nx=9001,xmax=900,nc=129,ng=31,eval_h=.025,mean=mean,sd=sd))
    with (out/'sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if a.recalibrate_boundary:
        from run import calibrate
        large=fm.Model(nx=9001,xmax=900,nc=129,ng=7)
        p,res,hist=calibrate(lambda v:large.solve(2,v),1.58,1.64,False,tol=.005,maxit=16)
        (out/'cap900_recalibration.json').write_text(json.dumps(dict(parameter=p,mean=res[3],sd=res[4],calibration=hist,xmax=900),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
