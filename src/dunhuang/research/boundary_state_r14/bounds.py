"""Strict necessary exclusion for immutable Native four-frame FK jerk templates."""
import numpy as np

def immutable_jerk_bound(fk,limit,edit=(60,90),fps=30):
    fk=np.asarray(fk)
    if fk.dtype!=np.float32 or fk.ndim!=3 or fk.shape[1:]!=(24,3) or not np.isfinite(fk).all():
        raise ValueError("Exact finite Native FP32 FK required")
    a,b=edit
    if not 0<=a<b<=len(fk) or not np.isfinite(limit) or limit<=0:raise ValueError("Invalid edit/limit")
    # Match motion_geometry.physical: float32 diff, scale, then norm.
    jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*float(fps)**3,axis=-1)
    starts=np.arange(len(jerk));fixed=((starts+3)<a)|(starts>=b)
    indices=np.argwhere(fixed[:,None]&(jerk>limit))
    witnesses=[]
    if len(indices):
        i,j=max(indices,key=lambda pair:float(jerk[tuple(pair)]))
        witnesses=[dict(metric="joint_jerk_mps3_max",template=[int(i),int(i+4)],joint=int(j),value=float(jerk[i,j]),limit=float(limit))]
    return dict(edit=list(edit),fps=fps,immutable_templates=int(fixed.sum()),native_global_max=float(jerk.max()) if jerk.size else 0.,
        immutable_jerk_lower_bound=float(jerk[fixed].max()) if fixed.any() else 0.,witnesses=witnesses,
        necessary_status="EXCLUDED_IMMUTABLE_JERK" if witnesses else "UNKNOWN",
        sufficient_feasibility_claimed=False,other_metrics="UNKNOWN: global floor/support/population/percentiles may change",
        proof="Frame-local FK and a four-frame derivative wholly outside edit are unchanged; global maximum is at least that fixed value")
