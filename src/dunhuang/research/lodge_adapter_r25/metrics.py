"""Source proxies include wrists; quality attribution never changes motion."""
import numpy as np
from scipy.signal import find_peaks
from dunhuang.research.lodge_reference_r24.compiler import angular_errors,frequency_readout
from dunhuang.research.branch_r22.local_contract import common22

def regional_jerk(points,cue,core=(64,160)):
    j=np.linalg.norm(np.diff(np.asarray(points,float),n=3,axis=0)*30**3,axis=-1)
    a,b=core;near=np.convolve(np.asarray(cue,int),np.ones(5,int),"same")>0
    labels=[]
    for t in range(len(j)):
        span=np.arange(t,t+4)
        labels.append("cue_neighborhood" if near[span].any() else "core_interior_unprompted" if ((span>=a)&(span<b)).all() else "outside_core" if ((span<a)|(span>=b)).all() else "core_edge")
    return {name:dict(templates=sum(x==name for x in labels),p95=float(np.percentile(j[np.array(labels)==name],95)) if name in labels else None,
        maximum=float(j[np.array(labels)==name].max()) if name in labels else None) for name in ("cue_neighborhood","core_interior_unprompted","core_edge","outside_core")}

def source_metrics(motion,reference,points,source_points,mask):
    errors=angular_errors(motion,reference);upper8=angular_errors(motion,reference,list(range(12,20)))
    upper10=angular_errors(motion,reference,list(range(12,22)));wrist=angular_errors(motion,reference,[20,21])
    p=points[:,[20,21]]-points[:,[16,17]];q=source_points[:,[20,21]]-source_points[:,[16,17]]
    def angle(x):
        a=x[:,[16,17]]-x[:,[18,19]];b=x[:,[20,21]]-x[:,[18,19]]
        return np.arccos(np.clip(np.sum(a*b,-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12),-1,1))
    return dict(body22_rad=float(errors[mask].mean()),upper8_rad=float(upper8[mask].mean()),upper10_rad=float(upper10[mask].mean()),wrist_rotation_rad=float(wrist[mask].mean()),
        wrist_relative_shoulder_rmse_m=float(np.sqrt(np.mean((p-q)[mask]**2))),
        elbow_angle_mae_rad=float(np.abs(angle(points)-angle(source_points))[mask].mean()),
        double_arm_wrist_relation_rmse_m=float(np.sqrt(np.mean(((points[:,20]-points[:,21])-(source_points[:,20]-source_points[:,21]))[mask]**2))))

def motion_metrics(motion,points,cue,core=(64,160)):
    result=common22(points);result["frequency"]=frequency_readout(points);result["regions"]=regional_jerk(points,cue,core)
    feet=points[:,[7,8,10,11]];horizontal=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0)*30,axis=-1)
    near=feet[:-1,:,1]<=-1.15
    result["foot_geometry_proxy"]=dict(near_ground_horizontal_speed_p95=float(np.percentile(horizontal[near],95)) if near.any() else None,
        near_ground_elements=int(near.sum()),minimum_foot_height_m=float(feet[:,:,1].min()),official_ground_y_m=-1.2,
        contact_truth="UNKNOWN",measured_contact=False)
    result["root_speed_p95_mps"]=float(np.percentile(np.linalg.norm(np.diff(motion[:,4:7],axis=0)*30,axis=-1),95))
    return result

def timing_proxy(motion,reference,music,core=(64,160)):
    from .reference import features,landmarks
    a,b=core;rot=motion[a:b,7:];ref=reference[a:b,7:]
    speed=np.linalg.norm(features(rot)[:,12:22,6:],axis=-1).mean(-1);peaks=find_peaks(speed,distance=5)[0]
    target=landmarks(ref);nearest=[int(peaks[np.argmin(abs(peaks-t))]) for t in target] if len(peaks) else []
    beats=np.flatnonzero(music[:,-1]>.5)
    return dict(phase_landmarks_source=target,phase_matched_generated_proxy=nearest,
        phase_timing_proxy_mae_frames=float(np.mean(abs(np.array(nearest)-target))) if nearest else None,
        phase_order_proxy_inversions=sum((target[i]-target[j])*(nearest[i]-nearest[j])<0 for i in range(len(nearest)) for j in range(i+1,len(nearest))),
        phase_order_proxy_collapses=sum(target[i]!=target[j] and nearest[i]==nearest[j] for i in range(len(nearest)) for j in range(i+1,len(nearest))),
        beat_distance_proxy_frames=float(np.mean(np.min(abs((a+peaks)[:,None]-beats),axis=1))) if len(peaks) and len(beats) else None,
        scope="Kinematic proxy peaks, not semantic event truth; no perceptual music coordination claim")
