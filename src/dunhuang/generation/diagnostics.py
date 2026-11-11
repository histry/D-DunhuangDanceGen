"""Source-following diagnostics, never physical or semantic qualification."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,motion151
from .soft_reference import build_soft_reference

def reference_error(motion,plan,scale,bias):
    motion=motion151(motion)
    values,mask,_,audit=build_soft_reference(plan,len(motion),scale,bias)
    values=np.concatenate([values[0]]+[v[75:] for v in values[1:]])[:len(motion)]
    mask=np.concatenate([mask[0]]+[v[75:] for v in mask[1:]])[:len(motion)]
    observed=mask[:,79:127].reshape(-1,8,6).all(-1)
    raw=(values-bias)/scale
    source=raw[:,79:127].reshape(-1,8,6)[observed]
    generated=motion[:,79:127].reshape(-1,8,6)[observed]
    if not len(source):return dict(mean_radians=None,p95_radians=None,observed_joint_frames=0,scope='supported source pose following only')
    delta=matrices(source,row=True).swapaxes(-1,-2)@matrices(generated)
    angles=Rotation.from_matrix(delta).magnitude()
    return dict(mean_radians=float(angles.mean()),p95_radians=float(np.percentile(angles,95)),
        observed_joint_frames=len(angles),scope='supported source pose following only; not music or cultural semantic quality')

def upperbody_activity(motion):
    x=motion151(motion)
    r=matrices(x[:,79:127].reshape(-1,8,6))
    angles=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude()*30
    return dict(mean_radians_per_second=float(angles.mean()),p95_radians_per_second=float(np.percentile(angles,95)))
