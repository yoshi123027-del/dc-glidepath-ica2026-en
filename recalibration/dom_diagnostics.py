"""DOMV target spacing sensitivity with the coefficient held fixed."""
import argparse,json
from pathlib import Path
import numpy as np
import finite_model as fm
from run import select_target

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();out=Path(a.directory)
    cfg=json.loads((out/'config.json').read_text());model=fm.Model(**cfg)
    saved=np.load(out/'DOMV.npz');gamma=float(saved['parameter'])
    V=np.load(out/'family_V.npy',mmap_mode='r');P=np.load(out/'family_P.npy',mmap_mode='r');targets=np.load(out/'targets.npy')
    rows=[]
    for stride in [1,2]:
        # The strided endpoint is 648 rather than 650 when stride=2. It still
        # covers the optimum at this fixed calibrated gamma; check explicitly.
        assert targets[::stride][-1]>=cfg['xmax']+1/gamma
        p,ch=select_target(V[::stride],P[::stride],targets[::stride],gamma)
        m,s=model.evaluate(p)
        rows.append(dict(target_step=float(2*stride),gamma=gamma,mean=m,sd=s,max_selected_target=float(targets[::stride][ch].max()),upper_boundary_fraction=float(np.mean(ch==len(targets[::stride])-1)),embedding_error_bound=gamma*(2*stride)**2/8))
    (out/'dom_target_sensitivity.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
