from pathlib import Path
import json,hashlib
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk,rotation_sample,multi_warp
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
ROOT=Path('outputs/music_structure_r56_01')
CFG=read(Path(__file__).with_name('protocol.json'))
SCIENCE=('common.py','music.py','slots.py','planning.py','metrics.py','study.py','protocol.json')

def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R56 output required: '+str(p))
 p.mkdir(parents=True);return p

def binding():return {name:sha(Path(__file__).with_name(name)) for name in SCIENCE}

def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def kernel(features):
 x=np.asarray(features,float);d=np.mean((x[:,None]-x[None,:])**2,axis=tuple(range(2,x.ndim+1)));v=d[np.triu_indices(len(x),1)];scale=float(np.median(v[v>1e-12])) if np.any(v>1e-12) else 1.
 return np.exp(-d/max(scale,1e-12))
