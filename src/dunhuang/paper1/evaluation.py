"""Variable-length diagnostics, without changing the frozen 240-frame R66 API."""
import numpy as np
from dunhuang.research.complete_r36.core import rotation_sample,relative_fk
from dunhuang.research.rhythm_r66.metrics import _motion
from .contracts import source_reads,array_hash

def measure(R,source,J,parents,q,interval,**unused):
    R=np.asarray(R);q=source_reads(q,interval,[0,len(R)]);a,b=interval
    expected=rotation_sample(source,q)
    if not np.array_equal(R,expected):raise ValueError('Candidate differs from actual source sampler')
    uniform=rotation_sample(source,np.linspace(a,b-1,len(R)));P=relative_fk(R,J,parents);U=relative_fk(uniform,J,parents)
    def values(X):return dict(path_length_mean_m=float(np.linalg.norm(np.diff(X,axis=0),axis=-1).sum(0).mean()),range_rms_m=float(np.sqrt(np.mean(np.var(X,axis=0)))))
    return dict(frames=len(R),fps=30,actual=_motion(R,P),same_duration_uniform=_motion(uniform,U),
                geometric_range=dict(actual=values(P),uniform=values(U)),sampler_exact=True,rotation_sha256=array_hash(R),
                world_root_contact='UNKNOWN',native24='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE')
