"""Shared SI geometry and bounded *input* permissions, not an output Root writer."""
import numpy as np
from dunhuang.contracts import matrices,sixd
from dunhuang.research.state_bridge.metrics import FEET,rotvec

def shuffled_target(recipient,donor):
    # Express the donor's relative Root motion in the recipient's observed left frame.
    A=matrices(recipient[59,7:13]);B=matrices(donor[59,7:13])
    turn=A@B.T;out=recipient.copy();out[60:90,13:]=donor[60:90,13:]
    out[60:90,4:7]=recipient[59,4:7]+(donor[60:90,4:7]-donor[59,4:7])@turn.T
    out[60:90,7:13]=sixd(turn@matrices(donor[60:90,7:13]))
    return out

def input_mask(shape,arm,legs):
    if arm not in ("J1","J2","J3","J4"):raise ValueError("Editable arm required")
    k=np.zeros(shape,bool);k[:,60:90,4:13]=True
    if arm!="J1":
        for j in legs:k[:,60:90,7+j*6:13+j*6]=True
    return k

def response(native,base,cfg):
    if native.shape!=base.shape or native.dtype!=np.float32:raise ValueError("Native FP32 contract required")
    a,b=cfg["edit"]
    if not np.array_equal(native[:a],base[:a]) or not np.array_equal(native[b:],base[b:]):
        raise ValueError("Protected output frames changed")
    R=matrices(native[a:b,7:].reshape(-1,24,6));G=matrices(base[a:b,7:].reshape(-1,24,6))
    angles=np.linalg.norm(rotvec(G.swapaxes(-1,-2)@R),axis=-1)
    legs=cfg["leg_joints"];other=[j for j in range(1,24) if j not in legs]
    values=dict(root_translation_m=float(np.linalg.norm(native[a:b,4:7]-base[a:b,4:7],axis=-1).max()),
        root_angle_rad=float(angles[:,0].max()),leg_angle_rad=float(angles[:,legs].max()),other_angle_rad=float(angles[:,other].max()))
    return dict(values,within_radius=all(values[k]<=v+1e-7 for k,v in cfg["physical_radius"].items()))

def metrics(native,fk,target,target_fk):
    R=matrices(native[60:90,7:].reshape(-1,24,6));G=matrices(target[60:90,7:].reshape(-1,24,6))
    angles=np.linalg.norm(rotvec(G.swapaxes(-1,-2)@R),axis=-1)
    upper=[j for j in range(1,24) if j not in (1,2,4,5,7,8,10,11)]
    out=dict(root_orientation_rad=float(angles[:,0].mean()),upperbody_pose_rad=float(angles[:,upper].mean()),
        feet_rmse_m=float(np.sqrt(np.mean((fk[60:90,FEET]-target_fk[60:90,FEET])**2))))
    out["boundary_errors"]=[]
    for seam in (60,90):
        lo,hi=(59,60) if seam==60 else (89,90)
        pd=np.diff(native[:,4:7].astype(float),axis=0);gd=np.diff(target[:,4:7].astype(float),axis=0)
        pa=np.diff(native[:,4:7].astype(float),n=2,axis=0);ga=np.diff(target[:,4:7].astype(float),n=2,axis=0)
        endpoint=60 if seam==60 else 89
        out["boundary_errors"].append(dict(seam=seam,
            root_position_m=float(np.linalg.norm(native[endpoint,4:7]-target[endpoint,4:7])),
            root_velocity_mps=float(np.linalg.norm(pd[hi-1]-gd[hi-1])*30),
            root_acceleration_mps2=float(np.linalg.norm(pa[hi-2]-ga[hi-2])*900)))
    return out

def protected(current,base,cfg):
    p=cfg["protection"];epsilon=1e-8
    seams=all(x[k]<=y[k]*(1+p["seam_relative"])+epsilon for x,y in zip(current["joins"],base["joins"])
        for k in ("root_step_m","root_turn_rad","foot_step_max_m","peak_jerk_mps3"))
    boundary=all(x[k]<=y[k]*(1+p["seam_relative"])+epsilon for x,y in zip(current["boundary_errors"],base["boundary_errors"])
        for k in ("root_position_m","root_velocity_mps","root_acceleration_mps2"))
    activity=abs(current["activity_mps"]/max(base["activity_mps"],1e-8)-1)<=p["activity_relative"]
    pose=current["root_orientation_rad"]<=base["root_orientation_rad"]+p["upperbody_pose_rad"] and current["pose_error_rad"]<=base["pose_error_rad"]+p["source_pose_rad"] and current["upperbody_pose_rad"]<=base["upperbody_pose_rad"]+p["upperbody_pose_rad"]
    return dict(both_seams_nonregression=seams,boundary_state_nonregression=boundary,activity_protected=activity,source_protected=pose,
        root_improved=current["root_rmse_m"]<base["root_rmse_m"],feet_improved=current["feet_rmse_m"]<base["feet_rmse_m"])

def qualification(records,cfg):
    def rows(arm):return {r["case"]:r for r in records if r["arm"]==arm}
    arm={f:rows(f) for f in cfg["arms"]};groups=cfg["gate"]["groups"]
    if any(sum(r["arm"]==f for r in records)!=groups for f in cfg["arms"]):raise ValueError("One aggregate per group required")
    if any(set(v)!=set(range(groups)) for v in arm.values()):raise ValueError("Complete fixed population required")
    qualified={a:[i for i,r in rr.items() if r.get("accepted",False) and all(r["protection"].values())] for a,rr in arm.items() if a!="J0"}
    gains={}
    for other in ("J1","J4"):
        ratios=[1-arm["J3"][i]["metrics"]["feet_rmse_m"]/max(arm[other][i]["metrics"]["feet_rmse_m"],1e-10) for i in range(groups)]
        gains[other]=dict(group_gains=ratios,wins=sum(v>0 for v in ratios),mean_group_gain=float(np.mean(ratios)))
    passed=len(qualified["J3"])>=cfg["gate"]["minimum_qualified_groups"] and len(qualified["J3"])>len(qualified["J1"])
    passed=passed and all(x["wins"]>=cfg["gate"]["minimum_qualified_groups"] and x["mean_group_gain"]>=cfg["gate"]["minimum_gain"] for x in gains.values())
    return dict(qualified_groups=qualified,foot_gains=gains,local_capacity_passed=bool(passed),native_gate_separate=True,
        population_training_admitted=False,operational_qualified=False)
