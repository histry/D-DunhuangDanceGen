"""Whole-record kinematic phase proposals. No music input or semantic/force labels."""
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices

UPPER=tuple(range(12,20))
LOWER=(1,2,4,5,7,8,10,11)

def motion_profiles(native,observed_joints,fk=None,root_observed=False):
    x=np.asarray(native)
    if x.ndim!=2 or x.shape[1]!=151 or len(x)<2:raise ValueError("Real continuous native151 source required")
    if len(set(observed_joints))!=len(observed_joints) or any(type(j) is not int or not 0<=j<24 for j in observed_joints):raise ValueError("Observed joint identity differs")
    mask=np.zeros(24,bool);mask[observed_joints]=True
    rotations=x[:,7:].reshape(-1,24,6)
    if not np.isfinite(rotations[:,mask]).all():raise ValueError("Nonfinite observed source rotation")
    identity=np.array([1,0,0,0,1,0],dtype=x.dtype)
    r=matrices(np.where(mask[None,:,None],rotations,identity)).astype(np.float64)
    omega=np.zeros((len(x),24,3),np.float64)
    omega[1:]=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,24,3)*30
    omega[0]=omega[1];speed=np.linalg.norm(omega,axis=-1)
    if not mask[list(UPPER)].all():raise ValueError("All actual controlled upper8 joints must be observed")
    result=dict(omega=omega,upper8=speed[:,UPPER].mean(1),old_upper10=speed[:,12:22].mean(1) if mask[12:22].all() else np.full(len(x),np.nan),
        lower=speed[:,LOWER].mean(1) if mask[list(LOWER)].all() else np.full(len(x),np.nan),
        root_yaw_radps=omega[:,0,1] if mask[0] else np.full(len(x),np.nan))
    vector=omega[:,UPPER].reshape(len(x),-1);norm=np.linalg.norm(vector,axis=-1)
    cosine=(vector[1:]*vector[:-1]).sum(-1)/np.maximum(norm[1:]*norm[:-1],1e-12)
    result["upper_direction_change"]=np.r_[0,1-np.clip(cosine,-1,1)]
    if root_observed and np.isfinite(x[:,4:7]).all():
        velocity=np.diff(x[:,4:7],axis=0)*30;result["root_horizontal_speed_mps"]=np.r_[np.linalg.norm(velocity[:1][:,(0,2)],axis=-1),np.linalg.norm(velocity[:,(0,2)],axis=-1)]
    if fk is not None:
        if not root_observed:raise ValueError("World FK requires observed metric Root")
        fk=np.asarray(fk)
        if fk.shape!=(len(x),24,3) or not np.isfinite(fk).all():raise ValueError("Trusted aligned FK support differs")
        v=np.diff(fk[:,UPPER],axis=0)*30
        result["fk_upper_speed_mps"]=np.r_[np.linalg.norm(v[:1],axis=-1).mean(-1),np.linalg.norm(v,axis=-1).mean(-1)]
    return result

def _scale(x):
    return max(float(np.median(np.abs(x-np.median(x)))*1.4826),1e-8)

def detect(profiles,config):
    speed=np.asarray(profiles["upper8"]);events=[]
    def peaks(signal,kind,floor,joints):
        scale=_scale(signal)
        ids,props=find_peaks(signal,distance=config["minimum_peak_distance"],prominence=max(scale,floor))
        for i,prominence in zip(ids,props["prominences"]):
            events.append(dict(frame=int(i),type=kind,observed_joints=list(joints),score=float(prominence/(prominence+scale+floor)),
                score_scope="uncalibrated_kinematic_candidate",evidence_span=[max(0,int(i)-1),min(len(speed),int(i)+2)],semantic_status="UNKNOWN"))
    peaks(speed,"upper_speed_peak",config["speed_prominence_floor_radps"],UPPER)
    growth=np.maximum(np.diff(speed,prepend=speed[0])*30,0)
    peaks(growth,"upper_motion_onset",config["acceleration_prominence_floor_radps2"],UPPER)
    direction=profiles["upper_direction_change"].copy();direction[speed<config["hold_max_speed_radps"]]=0
    peaks(direction,"upper_direction_turn",.2,UPPER)
    threshold=min(float(np.quantile(speed,config["hold_quantile"])),config["hold_max_speed_radps"])
    low=speed<=threshold;edges=np.diff(np.r_[False,low,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a>=config["hold_min_frames"]:
            events.append(dict(frame=(int(a)+int(b)-1)//2,type="upper_pose_hold",observed_joints=list(UPPER),score=.5,
                score_scope="uncalibrated_kinematic_candidate",evidence_span=[int(a),int(b)],semantic_status="UNKNOWN"))
    yaw=profiles["root_yaw_radps"]
    if np.isfinite(yaw).all():peaks(abs(yaw),"root_turn_proxy",config["speed_prominence_floor_radps"],(0,))
    return sorted(events,key=lambda p:(p["frame"],p["type"]))

def phases_for_core(events,a,b,controlled_peak):
    result=[]
    for e in events:
        event=dict(e)
        if event["type"]=="upper_pose_hold":
            lo,hi=event["evidence_span"];lo=max(lo,a);hi=min(hi,b)
            if hi-lo<6:continue
            event["frame"]=(lo+hi-1)//2
            event["representative_rule"]="middle of observed hold/Core intersection"
        if a<=event["frame"]<b:
            event["evidence_truncated_by_core"]=event["evidence_span"][0]<a or event["evidence_span"][1]>b
            result.append(event)
    if not any(e["frame"]==controlled_peak and e["type"]=="upper_speed_peak" for e in result):
        result.append(dict(frame=controlled_peak,type="controlled_core_maximum",observed_joints=list(UPPER),score=None,
            score_scope="maximum in fixed Core, not necessarily a full-record local extremum",evidence_span=[max(a,controlled_peak-1),min(b,controlled_peak+2)],semantic_status="UNKNOWN"))
    return sorted(result,key=lambda p:(p["frame"],p["type"]))

def choose_anchor(phases,controlled_peak,a,b,observed_joints,upper_speed,radius=12):
    if radius!=12:raise ValueError("Registered source context radius differs")
    allowed=set(observed_joints)
    eligible=[p for p in phases if p["type"]!="root_turn_proxy" and set(UPPER)<=set(p["observed_joints"])<=allowed and a+radius<=p["frame"]<b-radius]
    direct=next((p for p in eligible if p["frame"]==controlled_peak),None)
    if direct:return dict(direct,selection="actual controlled maximum, no relocation")
    if not eligible:return None
    selected=max(eligible,key=lambda p:(float(upper_speed[p["frame"]]),p["score"] or 0.,-p["frame"]))
    return dict(selected,selection="alternate actually detected upper-body phase; maximum remains unchanged")

def choose_keys(phases,anchor,a,b):
    if anchor is None:return []
    keys=[anchor];used={anchor["frame"]}
    priority=("upper_motion_onset","upper_pose_hold","upper_speed_peak","upper_direction_turn")
    for kind in priority:
        candidates=[p for p in phases if p["type"]==kind and p["frame"] not in used and a+12<=p["frame"]<b-12]
        if candidates:
            p=max(candidates,key=lambda p:((p["score"] or 0),-p["frame"]));keys.append(p);used.add(p["frame"])
        if len(keys)==3:break
    return sorted(keys,key=lambda p:p["frame"])
