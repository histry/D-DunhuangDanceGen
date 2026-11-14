"""Physical SI/separate-seam metrics; known-pose and event semantics are distinct."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import MotionAdapter,matrices,Q
from dunhuang.research.condition_dependency.codec import origin,restore
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.state_bridge.metrics import FEET

def decode(normalized,scaler,old):
    raw=((normalized-scaler["bias"])/scaler["scale"]).astype(np.float32)
    return restore(MotionAdapter.to_native(raw),origin(old))

def qmetrics(pred,target,scaler):
    p=(pred-scaler["bias"])/scaler["scale"];g=(target-scaler["bias"])/scaler["scale"]
    known=np.r_[52:60,90:98]
    P=matrices(p[:,7:].reshape(-1,24,6),row=True);G=matrices(g[:,7:].reshape(-1,24,6),row=True)
    angles=Rotation.from_matrix((G.swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().reshape(150,24)
    return dict(known_root_rmse_m=float(np.sqrt(np.mean((p[known,4:7]-g[known,4:7])**2))),
        known_pose_rad=float(angles[known].mean()),unknown_pose_rad=float(angles[60:90].mean()),
        reconstruction=float(np.mean((pred[60:90,4:]-target[60:90,4:])**2)))

def joins(native,fk):
    R=matrices(native[:,7:13]);result=[]
    jerk=np.linalg.norm(np.diff(fk.astype(float),n=3,axis=0)*30**3,axis=-1);centers=np.arange(len(jerk))+1.5
    for t in (60,90):
        angle=float(Rotation.from_matrix(R[t-1].T@R[t]).magnitude())
        foot=np.linalg.norm(fk[t,FEET]-fk[t-1,FEET],axis=-1)
        m=(centers>=t-3)&(centers<=t+3);signal=jerk[m]
        result.append(dict(seam=t,root_step_m=float(np.linalg.norm(native[t,4:7]-native[t-1,4:7])),root_turn_rad=angle,
            foot_step_max_m=float(foot.max()),foot_step_mean_m=float(foot.mean()),peak_jerk_mps3=float(signal.max()),
            jerk_time_integral_mps2=float(np.sum(signal.max(-1))/30)))
    return result

def physical(sample,old,scaler):
    generated=decode(sample,scaler,old);edited=old.copy();edited[60:90]=generated[60:90]
    assert np.array_equal(edited[:60],old[:60]) and np.array_equal(edited[90:],old[90:])
    fk,audit=audit_native(edited)
    P=matrices(edited[60:90,7:].reshape(-1,24,6));G=matrices(old[60:90,7:].reshape(-1,24,6))
    pose=float(Rotation.from_matrix((G.swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().mean())
    return edited,fk,audit,dict(joins=joins(edited,fk),pose_error_rad=pose,
        root_rmse_m=float(np.sqrt(np.mean((edited[60:90,4:7]-old[60:90,4:7])**2))),
        activity_mps=float(np.linalg.norm(np.diff(fk[60:90],axis=0)*30,axis=-1).mean()),
        native_ok=audit["absolute"]["ok"],reasons=audit["absolute"]["reasons"])
