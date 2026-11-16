"""Full observed 22-joint source diagnostics on contiguous regions and cross-edge templates."""
import numpy as np
from dunhuang.research.state_r27.common import metrics
from dunhuang.research.contract_r28.input_contract import foot_proxy

def measure(raw,points,body,target):
    if raw.shape!=(256,139) or points.shape!=(256,22,3) or body.shape!=(256,132) or target.shape!=points.shape:raise ValueError("Common full256 clock required")
    p=np.asarray(points,np.float64);target=np.asarray(target,np.float64)
    relative=p-p[:,:1];truth=target-target[:,:1];error=relative-truth
    values=np.zeros((256,139),np.float32);values[:,7:]=body
    regions={}
    for name,(a,b) in dict(before=(0,64),core=(64,160),after=(160,256),full=(0,256)).items():
        mask=np.zeros(256,bool);mask[a:b]=True
        geometry=metrics(raw,points,values,target,mask=mask)
        paired={str(k):dict(rmse=float(np.sqrt(np.mean((np.diff(error[a:b],n=k,axis=0)*30**k)**2))),templates=b-a-k) for k in (1,2,3)}
        jerk={key:float(np.percentile(np.linalg.norm(np.diff(x[a:b],n=3,axis=0)*30**3,axis=-1),95))
            for key,x in dict(world=p,Root_translation=p[:,:1],Root_relative_body=relative).items()}
        regions[name]=dict(geometry=geometry,paired_body=paired,jerk=jerk,floor_proxy=foot_proxy(p[a:b]))
    cross={}
    for boundary in (64,160):
        cross[str(boundary)]={}
        for k in (1,2,3):
            start=np.arange(256-k);mask=(start<boundary)&(start+k>=boundary);d=np.diff(error,n=k,axis=0)*30**k
            actual=np.diff(relative,n=k,axis=0)*30**k
            cross[str(boundary)][str(k)]=dict(rmse=float(np.sqrt(np.mean(d[mask]**2))),actual_p95=float(np.percentile(np.linalg.norm(actual[mask],axis=-1),95)),templates=int(mask.sum()))
    return dict(regions=regions,cross_edges=cross,Root_truth="UNKNOWN",contact_truth="UNKNOWN")

def coverage_gate(records,cfg):
    def mean(arm,e,region,key):
        rows=[r for r in records if r["arm"]==arm and r["mode"]=="correct" and r["event"]==e]
        return float(np.mean([r["errors"][e]["regions"][region]["geometry"][key] for r in rows]))
    def outer(arm,e):return (64*mean(arm,e,"before","body22_rad")+96*mean(arm,e,"after","body22_rad"))/160
    def cross(arm,e):
        rows=[r for r in records if r["arm"]==arm and r["mode"]=="correct" and r["event"]==e]
        return float(np.mean([r["errors"][e]["cross_edges"][str(b)]["2"]["rmse"] for r in rows for b in (64,160)]))
    gates=[]
    for e in (0,1):
        ratios={k:mean("Coverage",e,"core",k)/mean("Frozen",e,"core",k) for k in ("body22_rad","world_wrist_rmse_m","body_activity_ratio","arm_range_ratio")}
        outside_ratio=outer("Coverage",e)/outer("Extra",e);cross_ratio=cross("Coverage",e)/cross("Extra",e)
        c=cfg["coverage_gate"];lo,hi=c["core_activity_extent_ratio"]
        passed=max(ratios["body22_rad"],ratios["world_wrist_rmse_m"])<=c["core_error_max_ratio"] and all(lo<=ratios[k]<=hi for k in ("body_activity_ratio","arm_range_ratio")) and outside_ratio<=c["outside_error_max_extra_ratio"] and cross_ratio<c["cross_D2_max_extra_ratio"]
        gates.append(dict(event=e,core_ratios=ratios,outside_angle_ratio_to_Extra=outside_ratio,cross_D2_ratio_to_Extra=cross_ratio,passed=passed))
    return dict(events=gates,context_training_admitted=all(x["passed"] for x in gates),operational_qualified=False,whole_song_generations=0)
