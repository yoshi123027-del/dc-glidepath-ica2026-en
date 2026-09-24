"""Shared-kernel regression against the pinned, pre-extraction DC implementation."""
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'recalibration'))
import finite_model as current
BASE='db548f7b3904a1961bba2e6892f0a83ae7e94011'
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'original_model.py'
    path.write_bytes(subprocess.check_output(['git','show',BASE+':recalibration/finite_model.py'],cwd=ROOT))
    spec=importlib.util.spec_from_file_location('original_model',path);old=importlib.util.module_from_spec(spec);sys.modules['original_model']=old;spec.loader.exec_module(old)
    z,w=current.gh(7);ix,wt=current.mapping(61,60.,17,12,z);rows=[]
    for kind,param in [(0,85.),(1,.03),(2,1.19)]:
        a=old.solve(kind,param,ix,wt,w,1.,12);b=current.solve(kind,param,ix,wt,w,1.,12)
        errors=[float(np.max(abs(x-y))) for x,y in zip(a,b)]
        assert all(v==0 for v in errors),errors
        rows.append(dict(kind=kind,parameter=param,max_policy_error=errors[0],max_mean_error=errors[1],max_second_error=errors[2]))
    out=ROOT/'results/validation/van_staden_2021/shared_kernel_regression.json'
    out.write_text(json.dumps(dict(base_commit=BASE,results=rows),indent=2));print(rows)
