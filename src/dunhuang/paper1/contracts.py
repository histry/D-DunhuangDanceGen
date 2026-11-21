"""Small shared identities; scientific producers remain in their frozen packages."""
import hashlib,json
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays

ROOT=Path('outputs/paper1_sdmr_integration_p0_01')
BASE='17bb6df3107498f472e87f2fb8bd3baf5bb200cd'
CFG=dict(songs=[0,1,2],arm='B0',source_units=16,max_warps=2,scan_paths=64,gap=48,
         rates=[.75,1.33],event_limit=3,proposal_radius=15,event_tolerance=3,
         max_local_geometry_requests=2496,training_updates=0,can_route=False)

def identity(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def binding():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.html')}

def fresh(p):
    p=Path(p);p.mkdir(parents=True,exist_ok=False);return p

def seal(p):
    write(p/'manifest.json',dict(files={str(q.relative_to(p)):sha(q) for q in sorted(p.rglob('*')) if q.is_file() and q.name!='manifest.json'},science=binding()))

def check_seal(p):
    m=read(p/'manifest.json')
    if any(sha(p/n)!=v for n,v in m['files'].items()):raise ValueError('Frozen artifact differs')
    return m

def source_reads(q,source,target):
    q=np.asarray(q,float);a,b=source;A,B=target
    if (q.shape!=(B-A,) or not np.isfinite(q).all() or not (np.diff(q)>0).all()
        or q[0]!=a or q[-1]!=b-1 or np.floor(q).min()<a or np.ceil(q).max()>=b):
        raise ValueError('Actual source/target frame support violated')
    return q
