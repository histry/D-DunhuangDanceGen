"""Independent metrics on saved motion states, never on viewer-only geometry."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.contracts import matrices

def metrics(R,reference,root,J,parents,boundaries=(),gap=None):
    p=relative_fk(R,J,parents);q=relative_fk(reference,J,parents)
    angle=Rotation.from_matrix((reference.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(len(R),22)
    velocity=np.linalg.norm(np.diff(p,axis=0)*30,axis=-1).mean(-1)
    jerk=np.linalg.norm(np.diff(p,n=3,axis=0)*27000,axis=-1).mean(-1)
    edges=np.zeros(len(jerk),bool)
    for b in boundaries:edges[max(0,b-3):min(len(edges),b)]=True
    protected=np.ones(len(R),bool)
    if gap:protected[gap[0]:gap[1]]=False
    world=p+root[:,None];foot_speed=np.linalg.norm(np.diff(world[:,[7,8,10,11]],axis=0)*30,axis=-1)
    return dict(source_angle_rad=float(angle.mean()),protected_source_angle_rad=float(angle[protected].mean()),
        local_fk_rmse_m=float(np.sqrt(np.mean((p-q)**2))),jerk_mean=float(jerk.mean()),jerk_p95=float(np.percentile(jerk,95)),
        jerk_peak=float(jerk.max()),jerk_area=float(jerk.sum()/30),boundary_jerk_peak=float(jerk[edges].max()) if edges.any() else None,
        internal_jerk_mean=float(jerk[~edges].mean()) if (~edges).any() else None,
        body_speed_m_s=float(velocity.mean()),peak_speed_m_s=float(velocity.max()),peak_speed_frame=int(velocity.argmax()+1),
        arm_extent_rms_m=float(np.sqrt(np.mean(np.ptp(p[:,[16,17,18,19,20,21]],axis=0)**2))),
        model_root_speed_m_s=float(np.linalg.norm(np.diff(root,axis=0)*30,axis=-1).mean()),
        foot_world_speed_proxy_m_s=float(foot_speed.mean()),foot_slide='NOT_EVALUABLE_NO_CONTACT_TRUTH',
        full_native='NOT_EVALUABLE_22_OF24',metric_frame='COMMON22_ROOT_RELATIVE_FK_30HZ_EXCEPT_EXPLICIT_MODEL_ROOT_AND_WORLD_FOOT_PROXY')

def decode(raw):
    return matrices(raw[:,7:].reshape(-1,22,6),row=True),np.asarray(raw[:,4:7])

def curves(R,reference,root,J,parents):
    p=relative_fk(R,J,parents);q=relative_fk(reference,J,parents)
    return dict(source_error=np.sqrt(np.mean((p-q)**2,axis=(1,2))).tolist(),jerk=[0.,0.,0.]+np.linalg.norm(np.diff(p,n=3,axis=0)*27000,axis=-1).mean(-1).tolist(),
                root_speed=[0.]+np.linalg.norm(np.diff(root,axis=0)*30,axis=-1).tolist())
