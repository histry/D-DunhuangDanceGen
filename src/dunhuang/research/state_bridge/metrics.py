"""SI backward differences and exact symmetric FK algebra, not causal attribution."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
FEET=[7,8,10,11]
LEGS=[1,2,4,5,7,8,10,11]
def rotvec(R):return Rotation.from_matrix(np.asarray(R).reshape(-1,3,3)).as_rotvec().reshape(np.asarray(R).shape[:-2]+(3,))
def d(x,order):return np.diff(np.asarray(x,dtype=np.float64),n=order,axis=0)*30**order

def states(x,fk,frames):
    R=matrices(x[:,7:].reshape(-1,24,6));w=rotvec(R[:-1].swapaxes(-1,-2)@R[1:])*30
    rd=[None]+[d(x[:,4:7],n) for n in (1,2,3)]
    fd=[None]+[d(fk,n) for n in (1,2,3)]
    out=[]
    for t in frames:
        out.append(dict(frame=t,root_world_m=x[t,4:7].tolist(),root_rotation=R[t,0].tolist(),
            root_d1_mps=rd[1][t-1].tolist(),root_d2_mps2=rd[2][t-2].tolist(),root_d3_mps3=rd[3][t-3].tolist(),
            local_rotation_step_rad=np.linalg.norm(w[t-1]/30,axis=-1).tolist(),angular_velocity_rps=w[t-1].tolist(),
            angular_acceleration_rps2=((w[t-1]-w[t-2])*30).tolist(),
            feet_world_m=fk[t,FEET].tolist(),feet_horizontal_speed_mps=np.linalg.norm(fd[1][t-1,FEET][:,[0,2]],axis=-1).tolist(),
            feet_recent9_world_m=fk[t-8:t+1,FEET].tolist(),
            fk_d1_mps=np.linalg.norm(fd[1][t-1],axis=-1).tolist(),fk_d2_mps2=np.linalg.norm(fd[2][t-2],axis=-1).tolist(),
            fk_d3_mps3=np.linalg.norm(fd[3][t-3],axis=-1).tolist()))
    return out

def seam(x,fk,t,atol):
    R=matrices(x[:,7:13]).astype(float);root=x[:,4:7].astype(float)
    p=fk[:,FEET].astype(float);local=np.einsum("tji,tkj->tki",R,p-root[:,None])
    parts=dict(root_translation=np.broadcast_to(root[t]-root[t-1],(4,3)).copy(),
        root_rotation=.5*np.einsum("ij,kj->ki",R[t]-R[t-1],local[t]+local[t-1]),
        local_pose=.5*np.einsum("ij,kj->ki",R[t]+R[t-1],local[t]-local[t-1]))
    delta=p[t]-p[t-1];error=float(abs(sum(parts.values())-delta).max());assert error<=atol
    power=float(np.sum(delta**2))
    return dict(frame=t,root_step_m=float(np.linalg.norm(root[t]-root[t-1])),
        root_rotation_step_rad=float(np.linalg.norm(rotvec(R[t-1].T@R[t]))),
        local_leg_step_rad=np.linalg.norm(rotvec(matrices(x[t-1,7:].reshape(24,6)).swapaxes(-1,-2)@matrices(x[t,7:].reshape(24,6))),axis=-1)[LEGS].tolist(),
        foot_step_m=np.linalg.norm(delta,axis=-1).tolist(),horizontal_foot_step_m=np.linalg.norm(delta[:,[0,2]],axis=-1).tolist(),
        components={k:dict(vector_m=v.tolist(),rms_m=float(np.sqrt(np.mean(v*v))),signed_projection_share=float(np.sum(v*delta)/power) if power>1e-18 else None) for k,v in parts.items()},
        identity_error_m=error,causal_attribution=False)

def pair_quality(pred,target,fk,target_fk):
    a,b=10,40 # ROI global[50,100), unknown global[60,90)
    P=matrices(pred[a:b,7:].reshape(-1,24,6));G=matrices(target[a:b,7:].reshape(-1,24,6))
    joints=np.linalg.norm(rotvec(G.swapaxes(-1,-2)@P),axis=-1)
    boundary=np.r_[10:13,37:40]
    return dict(root_rmse_m=float(np.sqrt(np.mean((pred[a:b,4:7]-target[a:b,4:7])**2))),
        pose_error_rad=float(joints.mean()),feet_rmse_m=float(np.sqrt(np.mean((fk[a:b,FEET]-target_fk[a:b,FEET])**2))),
        boundary_root_rmse_m=float(np.sqrt(np.mean((pred[boundary,4:7]-target[boundary,4:7])**2))),
        boundary_feet_rmse_m=float(np.sqrt(np.mean((fk[boundary][:,FEET]-target_fk[boundary][:,FEET])**2))))

def direction(response,target_delta):
    x=np.asarray(response,float);v=np.asarray(target_delta,float);den=float(np.sum(v*v));norm=float(np.linalg.norm(x));tn=float(np.linalg.norm(v))
    return dict(response_norm=float(norm),target_norm=tn,projection_gain=float(np.sum(x*v)/den) if den>1e-16 else None,
        cosine=float(np.sum(x*v)/(norm*tn)) if norm*tn>1e-16 else None)
