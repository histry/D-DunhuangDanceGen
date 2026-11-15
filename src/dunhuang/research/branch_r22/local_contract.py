"""Output support is explicit; incomplete skeletons never enter full Native."""
import numpy as np

def require_native24(motion,valid_joints,fps):
    x=np.asarray(motion);mask=np.asarray(valid_joints,dtype=bool)
    if x.ndim!=2 or x.shape[1]!=151 or mask.shape!=(24,) or not mask.all():
        raise ValueError("INCOMPLETE_NATIVE24: no padding or fabricated joints permitted")
    if fps!=30 or len(x)<4 or not np.isfinite(x).all():raise ValueError("Invalid Native clock or values")
    return x

def common22(points,fps=30):
    p=np.asarray(points)
    if p.ndim!=3 or p.shape[1:]!=(22,3) or len(p)<4 or not np.isfinite(p).all():raise ValueError("Common22 shape")
    speed=np.linalg.norm(np.diff(p,axis=0)*fps,axis=-1)
    jerk=np.linalg.norm(np.diff(p,n=3,axis=0)*fps**3,axis=-1)
    return dict(activity_mps=float(speed.mean()),jerk_p95_mps3=float(np.percentile(jerk,95)),jerk_max_mps3=float(jerk.max()),
        full_native_status="NOT_EVALUABLE",representation="SMPLX22_COMMON_GEOMETRY_ONLY",operational_qualified=False)
