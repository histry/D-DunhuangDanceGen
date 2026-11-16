"""Same Core support, physical derivatives and unambiguous aliases of historical metrics."""
import numpy as np
from dunhuang.research.state_r27.common import metrics as original_metrics
from dunhuang.research.time_r29.dynamics import decomposition
from dunhuang.research.lodge_adapter_r25.model import PARTS

ALIASES={"world_wrist_rmse_m":"world_heading_shoulder_to_wrist_vector_rmse_m",
 "pelvis_wrist_rmse_m":"pelvis_heading_shoulder_to_wrist_vector_rmse_m",
 "double_wrist_rmse_m":"world_heading_inter_wrist_vector_rmse_m"}

def geometry(raw,points,truth,target):
    old=original_metrics(raw,points,truth,target)
    return dict(old,**{new:old[key] for key,new in ALIASES.items()})

def absolute(points):
    p=np.asarray(points,dtype=np.float64)[64:160]
    if p.shape!=(96,22,3) or not np.isfinite(p).all():raise ValueError("Finite full Core22 required")
    rel=p-p[:,:1];v=np.linalg.norm(np.diff(rel,axis=0)*30,axis=-1).mean()
    extent=np.sqrt(np.mean(np.ptp(rel[:,[16,17,18,19,20,21]],axis=0)**2))
    return dict(root_relative_activity_mps=float(v),arm_extent_m=float(extent))

def paired(points,target):
    p=np.asarray(points,dtype=np.float64)[64:160];t=np.asarray(target,dtype=np.float64)[64:160]
    if p.shape!=(96,22,3) or t.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Matched finite Core22 required")
    result={}
    for name,x,y in (("world",p,t),("Root_translation",p[:,:1],t[:,:1]),
                     ("Root_relative_body",p-p[:,:1],t-t[:,:1])):
        result[name]={}
        for k in (1,2,3):
            delta=(np.diff(x,n=k,axis=0)-np.diff(y,n=k,axis=0))*30**k
            norms=np.linalg.norm(delta,axis=-1)
            result[name][str(k)]=dict(rmse=float(np.sqrt(np.mean(delta**2))),
                p95=float(np.percentile(norms,95)),templates=96-k,units={1:"m/s",2:"m/s2",3:"m/s3"}[k])
    result["body_groups"]={}
    rel=(p-p[:,:1])-(t-t[:,:1])
    for i,group in enumerate(PARTS):
        result["body_groups"][str(i)]={str(k):float(np.sqrt(np.mean((np.diff(rel[:,group],n=k,axis=0)*30**k)**2))) for k in (1,2,3)}
    return result

def measure(raw,points,truth,target):
    return dict(metrics=geometry(raw,points,truth,target),source_absolute=absolute(target),
        generated_absolute=absolute(points),paired_dynamics=paired(points,target),dynamics=decomposition(points))
