"""Whole Core geometry and stage clock; missing Root/contact never become labels."""
import numpy as np
from dunhuang.io import checked
from dunhuang.contracts import motion151
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions

def build(plan,cfg):
    T=plan["frames"];value=np.zeros((T,151),np.float32);known=np.zeros((T,24),bool)
    value[:,7:]=np.tile([1,0,0,0,1,0],24) # masked placeholders, never source observations
    weights=np.zeros((T,24),np.float32);stages=[];cursor=0
    for e in plan["events"]:
        a,b=e["music_interval"];lo,hi=e["source_interval"];joints=e["observed_joints"]
        if a!=cursor or joints!=list(range(22)):raise ValueError("Complete, fitted SMPL22 Core required")
        if e.get("root_translation_observed") is not False:raise ValueError("Unverified Root translation cannot enter reference")
        src=motion151(np.load(checked(e["reference"])));pos=source_positions(e,np.arange(a,b));sample=resample_native(src,pos)
        value[a:b,7:139]=sample[:,7:139];known[a:b,:22]=True
        weights[a:b,:22]=cfg["base_pose_strength"]
        warp=np.asarray(e["time_warp"]);times=np.arange(a,b)
        for p in e["full_phases"]:
            if not lo<=p["frame"]<hi:raise ValueError("Phase outside complete source Core")
            tick=float(np.interp(p["frame"],warp[:,0],warp[:,1]))
            js=[j for j in p["observed_joints"] if 0<=j<22]
            bump=cfg["phase_pose_bonus"]*np.exp(-.5*((times-tick)/cfg["phase_sigma_frames"])**2)
            weights[a:b,js]=np.maximum(weights[a:b,js],cfg["base_pose_strength"]+bump[:,None])
            stages.append(dict(event_uid=e["event_uid"],source_frame=p["frame"],music_frame=tick,type=p["type"],joints=js,semantic_truth="UNKNOWN"))
        distance=np.minimum(np.arange(b-a),np.arange(b-a)[::-1])/cfg["boundary_fade_frames"]
        u=np.minimum(distance,1);fade=.3+.7*u**3*(10+u*(-15+6*u))
        weights[a:b]*=fade[:,None];cursor=b
    if cursor!=T or not known[:,:22].all() or known[:,22:].any():raise ValueError("Incomplete source support")
    return value,known,weights,stages
